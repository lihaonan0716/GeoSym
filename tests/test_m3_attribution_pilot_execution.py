"""Execution-path safety for the M3 attribution pilot.

The plan-freezing half of this runner was already covered; the paid execution
half was not wired at all until now, so these tests target the parts that spend
money or decide whether money was already spent:

* the HTTP attempt cap is enforced *before* each request, and counts
  SDK-internal transport retries rather than jobs;
* the resume checkpoint is keyed on experiment identity only, so a resumed run
  does not re-bill completed jobs;
* a mismatched plan refuses to resume rather than blending two runs;
* rows carry their own chain text, so a future scoring repair is a re-analysis
  instead of a re-run.

No network calls: the adapter layer is replaced with a counter.
"""

import json
from pathlib import Path

import pytest

from scripts import run_m3_attribution_pilot as pilot


class FakeAdapter:
    """Stands in for BaseLLMAdapter, counting calls instead of issuing them."""

    def __init__(self, fail_times: int = 0) -> None:
        self.calls = 0
        self.fail_times = fail_times

    def _call_api(self, image_b64: str, prompt: str) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("transient")
        return "{}"


@pytest.fixture
def adapter_base():
    import evaluator.adapters.base as module
    return module


class TestAttemptCapIsEnforcedAtTheHttpLayer:
    def test_cap_blocks_before_the_request_is_made(self, adapter_base, monkeypatch):
        """A refused call must not reach `_call_api` -- the cap bounds money
        committed, not money already spent."""
        counter = pilot.AttemptCounter(cap=2)
        fake = FakeAdapter()
        monkeypatch.setattr(adapter_base, "_MAX_API_RETRIES", 1)

        with pilot.count_llm_http_attempts(counter):
            bound = adapter_base.BaseLLMAdapter._call_with_retry
            bound(fake, "img", "prompt")
            bound(fake, "img", "prompt")
            with pytest.raises(pilot.LLMHttpAttemptCapExhausted):
                bound(fake, "img", "prompt")

        assert fake.calls == 2
        assert counter.http_attempts == 2
        assert counter.cap_exhausted is True

    def test_transport_retries_consume_the_cap(self, adapter_base, monkeypatch):
        """A job that retries twice costs two attempts, not one.

        Counting jobs instead of attempts is how a run silently exceeds its
        authorised spend while every individual call looks correct.
        """
        counter = pilot.AttemptCounter(cap=5)
        fake = FakeAdapter(fail_times=1)
        monkeypatch.setattr(adapter_base, "_MAX_API_RETRIES", 2)
        monkeypatch.setattr(adapter_base, "is_transient_error", lambda exc: True)

        with pilot.count_llm_http_attempts(counter):
            adapter_base.BaseLLMAdapter._call_with_retry(fake, "img", "prompt")

        assert counter.http_attempts == 2

    def test_the_original_method_is_restored(self, adapter_base):
        """A leaked patch would silently cap unrelated later runs."""
        original = adapter_base.BaseLLMAdapter._call_with_retry
        with pilot.count_llm_http_attempts(pilot.AttemptCounter(cap=1)):
            assert adapter_base.BaseLLMAdapter._call_with_retry is not original
        assert adapter_base.BaseLLMAdapter._call_with_retry is original

    def test_restored_even_when_the_body_raises(self, adapter_base):
        original = adapter_base.BaseLLMAdapter._call_with_retry
        with pytest.raises(RuntimeError):
            with pilot.count_llm_http_attempts(pilot.AttemptCounter(cap=1)):
                raise RuntimeError("boom")
        assert adapter_base.BaseLLMAdapter._call_with_retry is original


class TestCheckpointIdentity:
    def _plan(self):
        return pilot.build_plan(pilot.PILOT_8_ROSTER, "GLM-4.6V", "Qwen3-VL-Plus")

    def test_checkpoint_stores_plan_hash_not_a_timestamp(self, tmp_path: Path):
        plan = self._plan()
        counter = pilot.AttemptCounter(cap=plan["llm_http_attempt_cap"])
        out = tmp_path / "paid_result.json"
        payload = pilot.write_result(plan, out, [], counter, "complete")
        assert payload["plan_sha256"] == plan["plan_sha256"]

    def test_the_same_plan_rebuilt_keeps_the_same_hash(self):
        """A hash that moves between runs rejects its own resume and re-bills.

        Caught for real here: the hash originally covered the whole plan
        including `created_at_utc`, so every invocation produced a new identity
        and a resumed run would have paid twice for rows already on disk.
        """
        first, second = self._plan(), self._plan()
        assert first["plan_sha256"] == second["plan_sha256"]

    def test_timestamp_is_excluded_from_the_identity_fields(self):
        assert "created_at_utc" not in pilot.EXPERIMENT_IDENTITY_FIELDS

    def test_identity_covers_what_defines_the_experiment(self):
        """Dropping a field here would let two different experiments share a
        checkpoint, which is the opposite failure: blending unrelated runs."""
        for field in ("conditions", "selected_ids", "model_provenance",
                      "prompt_hashes", "llm_http_attempt_cap"):
            assert field in pilot.EXPERIMENT_IDENTITY_FIELDS

    def test_changing_the_roster_changes_the_identity(self):
        base = self._plan()
        shorter = pilot.build_plan(pilot.PILOT_8_ROSTER[:4], "GLM-4.6V", "Qwen3-VL-Plus")
        assert base["plan_sha256"] != shorter["plan_sha256"]

    def test_changing_the_model_changes_the_identity(self):
        base = self._plan()
        other = pilot.build_plan(pilot.PILOT_8_ROSTER, "GLM-4.6V", "Qwen3.5-Plus")
        assert base["plan_sha256"] != other["plan_sha256"]

    def test_resume_counts_prior_attempts_so_they_are_not_re_billed(self, tmp_path: Path):
        plan = self._plan()
        out = tmp_path / "paid_result.json"
        rows = [{"sample_id": "absolute-clean_render_0031",
                 "condition": "E5_raw", "status": "success",
                 "total_score": 90.0, "llm_http_attempts": 4}]
        counter = pilot.AttemptCounter(cap=plan["llm_http_attempt_cap"])
        counter.http_attempts = 4
        pilot.write_result(plan, out, rows, counter, "running_paid_run")

        checkpoint = json.loads(out.read_text(encoding="utf-8"))
        restored = [r for r in checkpoint["results"] if r["status"] == "success"]
        assert sum(r["llm_http_attempts"] for r in restored) == 4
        assert {(r["sample_id"], r["condition"]) for r in restored} == {
            ("absolute-clean_render_0031", "E5_raw")}

    def test_mismatched_plan_refuses_to_resume(self, tmp_path: Path):
        plan = self._plan()
        out = tmp_path / "paid_result.json"
        counter = pilot.AttemptCounter(cap=plan["llm_http_attempt_cap"])
        pilot.write_result(plan, out, [], counter, "complete")

        stale = json.loads(out.read_text(encoding="utf-8"))
        stale["plan_sha256"] = "0" * 64
        out.write_text(json.dumps(stale), encoding="utf-8")

        with pytest.raises(ValueError, match="checkpoint plan hash mismatch"):
            pilot.run_paid(plan, out)


class TestRosterActuallyResolves:
    """The plan's existence check and the loaders must agree on the path rule.

    Caught for real: `build_plan` built the family directory itself, so review
    passed, while the loaders joined the bare sample id onto the dataset root
    and found nothing. The run died on the first sample -- harmlessly, before
    any paid call, but a path rule verified only inside the planner is not
    verified at all.
    """

    def test_subpath_includes_the_family_directory(self):
        assert pilot.dataset_subpath("absolute-clean_render_0031") == (
            "absolute-clean_render/absolute-clean_render_0031")

    def test_piecewise_family_keeps_its_underscore(self):
        """`piecewise_linear` splits on '-', not '_'; splitting wrong silently
        points at a directory that does not exist."""
        assert pilot.dataset_subpath("piecewise_linear-clean_render_0031") == (
            "piecewise_linear-clean_render/piecewise_linear-clean_render_0031")

    @pytest.mark.parametrize("sample_id", pilot.PILOT_8_ROSTER)
    def test_every_rostered_sample_loads(self, sample_id):
        """Assert the loader succeeds, not merely that a path string exists."""
        dataset = Path(__file__).resolve().parents[1] / "dataset"
        if not dataset.is_dir():
            pytest.skip("dataset is distributed separately and is gitignored")
        gt, gt_data = pilot.load_ground_truth(sample_id)
        assert gt_data, sample_id
        assert pilot.resolve_image(sample_id).is_file()


class TestRowsCarryTheirOwnText:
    def test_row_persists_chain_text_not_just_a_hash(self):
        pipeline = {"final_parsed": {"expression": "x^2"},
                    "steps": [{"name": "L1", "response": "reading"}]}
        row = pilot.result_row("cubic-clean_render_0030", "E5_raw", pipeline,
                               {"total_score": 88.0, "dimension_scores": {}}, None,
                               judge_reused=False, attempts_before=0, attempts_after=3)
        assert "chain_text" in row
        assert row["final_parsed"] == {"expression": "x^2"}
        assert row["llm_http_attempts"] == 3

    def test_error_rows_record_the_reason_and_score_nothing(self):
        row = pilot.result_row("linear-clean_render_0018", "E6_flat_single", None,
                               None, None, judge_reused=False,
                               attempts_before=2, attempts_after=2,
                               error="geosym_extraction_failed: no grid")
        assert row["status"] == "error"
        assert row["total_score"] is None
        assert "no grid" in row["error"]


class TestEmptyOutputIsNotScoredAsAbility:
    """An empty body is a failed call, not a failed inference.

    Caught for real in the first pilot: the runner never set
    GEOSYM_GLM_MAX_TOKENS, the adapter fell back to its 4096 default, and three
    rows came back with finish_reason=length and a zero-length body. They were
    scored 0.0 and 15.0, and all three landed on one arm -- so the bias sat
    directly on the between-arm contrast (invariant 6). Removing those samples
    moved that arm from lowest mean to highest.
    """

    def test_empty_chain_with_blank_fields_is_empty_output(self):
        pipeline = {"final_parsed": {"expression": "", "domain": "", "range": "",
                                     "coordinates": ""}}
        assert pilot.is_empty_output(pipeline, "=== FlatSingleGeoSym ===\n") is True

    def test_a_real_answer_is_not_empty_output(self):
        pipeline = {"final_parsed": {"expression": "-2*abs(x+1)+4"}}
        assert pilot.is_empty_output(pipeline, "=== L1 ===\nlong reading text " * 10) is False

    def test_short_chain_but_populated_fields_is_not_empty(self):
        """Judged on content, not on chain length alone."""
        pipeline = {"final_parsed": {"expression": "x^2"}}
        assert pilot.is_empty_output(pipeline, "=== L1 ===") is False

    def test_empty_output_row_becomes_an_error_not_a_zero(self):
        pipeline = {"final_parsed": {"expression": "", "domain": ""}}
        row = pilot.result_row("rational-clean_render_0031", "E6_flat_single",
                               pipeline, {"total_score": 15.0, "dimension_scores": {}},
                               {"judge_inference_score": 0.0},
                               judge_reused=False, attempts_before=0, attempts_after=1)
        assert row["status"] == "error"
        assert row["total_score"] is None
        assert row["error"].startswith("empty_model_output")
        # The judge verdict on an empty body carries no information either.
        assert row["judge_result"] is None

    def test_summary_reports_empty_rows_per_arm(self):
        rows = [
            {"sample_id": "a", "condition": "E5_raw", "status": "success",
             "total_score": 70.0, "error": ""},
            {"sample_id": "b", "condition": "E6_flat_single", "status": "error",
             "total_score": None, "error": "empty_model_output_finish_reason_length"},
            {"sample_id": "c", "condition": "E7_form_first", "status": "success",
             "total_score": 80.0, "error": ""},
        ]
        summary = pilot.summarize(rows)
        assert summary["by_condition"]["E6_flat_single"]["empty_output_rows"] == 1
        assert summary["path_check"]["empty_output_rows_total"] == 1
        assert summary["path_check"]["empty_outputs_are_evenly_spread"] is False


class TestTokenBudgetIsFrozenNotInherited:
    def test_plan_pins_max_tokens_for_glm(self):
        plan = pilot.build_plan(pilot.PILOT_8_ROSTER, "GLM-4.6V", "Qwen3-VL-Plus")
        assert plan["model_provenance"]["max_tokens"] == pilot.GLM_MAX_TOKENS

    def test_budget_is_above_the_adapter_default_that_truncated(self):
        """4096 is the adapter default that produced the empty rows."""
        assert pilot.GLM_MAX_TOKENS > 4096

    def test_max_tokens_is_part_of_the_experiment_identity(self):
        """A different budget is a different experiment, so checkpoints must
        not be shared across budgets."""
        assert "model_provenance" in pilot.EXPERIMENT_IDENTITY_FIELDS

    def test_runner_sets_the_env_var_before_calling(self):
        import inspect
        assert "GEOSYM_GLM_MAX_TOKENS" in inspect.getsource(pilot.run_paid)


class TestConfirmationRoster:
    @pytest.fixture(autouse=True)
    def _requires_dataset(self):
        if not (Path(__file__).resolve().parents[1] / "dataset" / "ground_truth").is_dir():
            pytest.skip("dataset is distributed separately and is gitignored")

    def test_roster_size_matches_the_frozen_weights(self):
        roster = pilot.build_confirmation_roster()
        assert len(roster) == pilot.CONFIRMATION_SAMPLE_COUNT == 110

    def test_family_counts_match_the_frozen_weights(self):
        import collections
        counts = collections.Counter(
            s.split("-")[0] for s in pilot.build_confirmation_roster())
        assert dict(counts) == pilot.CONFIRMATION_FAMILY_WEIGHTS

    def test_transcendental_families_are_weighted_above_ceiling_families(self):
        """Samples spent on families already at the ceiling buy no information."""
        w = pilot.CONFIRMATION_FAMILY_WEIGHTS
        assert w["exponential"] > w["linear"]
        assert w["rational"] > w["quadratic"]

    def test_pilot_samples_are_excluded_so_they_are_not_re_billed(self):
        roster = pilot.build_confirmation_roster()
        assert not set(roster) & set(pilot.PILOT_8_ROSTER)

    def test_no_duplicates(self):
        roster = pilot.build_confirmation_roster()
        assert len(roster) == len(set(roster))

    def test_selection_is_deterministic_across_calls(self):
        assert pilot.build_confirmation_roster() == pilot.build_confirmation_roster()

    def test_selection_is_answer_independent(self):
        """Ordering is by sha256 of the id, so no score or coverage value can
        steer which samples are chosen (invariant 5)."""
        import inspect
        source = inspect.getsource(pilot.build_confirmation_roster)
        assert "sha256" in source
        for outcome in ("score", "coverage", "correct", "answer"):
            assert f'"{outcome}"' not in source


class TestSharding:
    @pytest.fixture(autouse=True)
    def _requires_dataset(self):
        if not (Path(__file__).resolve().parents[1] / "dataset" / "ground_truth").is_dir():
            pytest.skip("dataset is distributed separately and is gitignored")

    @pytest.mark.parametrize("count", [2, 3, 4, 6, 8])
    def test_shards_partition_the_roster_exactly(self, count):
        roster = pilot.build_confirmation_roster()
        shards = [pilot.shard_roster(roster, i, count) for i in range(count)]
        assert sum(len(s) for s in shards) == len(roster)
        assert set().union(*shards) == set(roster)
        for i in range(count):
            for j in range(i + 1, count):
                assert not set(shards[i]) & set(shards[j])

    @pytest.mark.parametrize("count", [2, 4, 6])
    def test_shard_caps_sum_to_the_full_cap_not_a_multiple(self, count):
        """Each worker gets a SLICE of the cap.

        Handing every worker the full cap would authorise `count` times the
        approved spend -- the concurrency version of the over-authorisation
        found during the M1/M2 retirement.
        """
        roster = pilot.build_confirmation_roster()
        full = pilot.call_caps(len(roster))["llm_http_attempt_cap"]
        shard_total = sum(
            pilot.call_caps(len(pilot.shard_roster(roster, i, count)))["llm_http_attempt_cap"]
            for i in range(count))
        assert shard_total == full

    def test_invalid_shard_is_rejected(self):
        roster = pilot.build_confirmation_roster()
        for index, count in ((0, 0), (-1, 4), (4, 4), (5, 4)):
            with pytest.raises(ValueError, match="invalid_shard"):
                pilot.shard_roster(roster, index, count)

    def test_sharding_is_by_sample_so_arms_stay_together(self):
        """All three arms of a sample must run in one worker: the per-image
        result files are not written atomically, so two writers clobber."""
        roster = pilot.build_confirmation_roster()
        shard = pilot.shard_roster(roster, 0, 4)
        assert all(isinstance(s, str) for s in shard)
        # No condition appears in the shard definition at all.
        assert not set(shard) & set(pilot.PILOT_CONDITIONS)


class TestSummaryDoesNotOverclaim:
    def test_eight_sample_summary_reports_path_health_not_a_winner(self):
        rows = [{"sample_id": "s", "condition": c, "status": "success",
                 "total_score": 50.0} for c in pilot.PILOT_CONDITIONS]
        summary = pilot.summarize(rows)
        assert summary["path_check"]["every_arm_produced_a_scored_row"] is True
        assert "must not be read as an effect" in summary["path_check"]["reading"]

    def test_missing_arm_is_flagged(self):
        rows = [{"sample_id": "s", "condition": "E5_raw",
                 "status": "success", "total_score": 50.0}]
        summary = pilot.summarize(rows)
        assert summary["path_check"]["every_arm_produced_a_scored_row"] is False
