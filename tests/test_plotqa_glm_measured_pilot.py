"""Safety tests for the GLM PlotQA pilot runner.

The runner spends money, so the properties worth testing are the ones whose
failure is expensive or silent: that the default performs no call, that the cap
is enforced before a request rather than after, that the roster is selected
without reference to any answer, and that a same-model baseline is present --
without it the pilot compares across backbones and measures nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Dev-local scripts (`scripts/*` is gitignored): skip rather than abort the
# suite's collection in a clean clone.
_pilot = pytest.importorskip("scripts.run_plotqa_glm_measured_pilot")
_public = pytest.importorskip("scripts.run_public_benchmark_pilot")

BASELINE_CONDITION = _pilot.BASELINE_CONDITION
CHART_TYPES = _pilot.CHART_TYPES
CONDITIONS = _pilot.CONDITIONS
EXPERIMENT_IDENTITY_FIELDS = _pilot.EXPERIMENT_IDENTITY_FIELDS
MEASURED_CONDITION = _pilot.MEASURED_CONDITION
NEUTRAL_BAND = _pilot.NEUTRAL_BAND
PILOT_SAMPLE_COUNT = _pilot.PILOT_SAMPLE_COUNT
RETRY_HEADROOM = _pilot.RETRY_HEADROOM
SAMPLES_PER_CHART_TYPE = _pilot.SAMPLES_PER_CHART_TYPE
experiment_identity_hash = _pilot.experiment_identity_hash
main = _pilot.main
review_plan = _pilot.review_plan
rows_of = _pilot.rows_of
select_roster = _pilot.select_roster
summarize = _pilot.summarize

AttemptCounter = _public.AttemptCounter


def _sample(sample_id: str, qa_type: str) -> dict:
    return {
        "sample_id": sample_id,
        "image_path": f"/tmp/{sample_id}.png",
        "question": "q",
        "answer_type": "numeric",
        "normalized_answer": 1.0,
        "metadata": {"qa_type": qa_type},
    }


def _population(per_type: int = 10) -> tuple[list[dict], dict[str, dict]]:
    samples = [
        _sample(f"{qa_type}_{i:03d}", qa_type)
        for qa_type in CHART_TYPES
        for i in range(per_type)
    ]
    records = {s["sample_id"]: {"sample_id": s["sample_id"]} for s in samples}
    return samples, records


class TestRosterIsAnswerIndependent:
    def test_roster_is_stratified_across_chart_types(self):
        samples, records = _population()
        roster = select_roster(samples, records)
        assert len(roster) == PILOT_SAMPLE_COUNT
        by_type: dict[str, int] = {}
        for sample in roster:
            by_type[sample["metadata"]["qa_type"]] = by_type.get(sample["metadata"]["qa_type"], 0) + 1
        assert by_type == {qa_type: SAMPLES_PER_CHART_TYPE for qa_type in CHART_TYPES}

    def test_selection_is_deterministic(self):
        samples, records = _population()
        first = [s["sample_id"] for s in select_roster(samples, records)]
        second = [s["sample_id"] for s in select_roster(list(reversed(samples)), records)]
        assert first == second

    def test_selection_ignores_answers_and_scores(self):
        """Changing every answer must not move the roster."""
        samples, records = _population()
        baseline = [s["sample_id"] for s in select_roster(samples, records)]
        for sample in samples:
            sample["normalized_answer"] = 99999.0
            sample["coverage"] = 1.0
        assert [s["sample_id"] for s in select_roster(samples, records)] == baseline

    def test_samples_without_a_record_are_excluded(self):
        """A sample the measured arm cannot inject for is not comparable."""
        samples, records = _population()
        dropped = samples[0]["sample_id"]
        records.pop(dropped)
        roster = select_roster(samples, records)
        assert dropped not in {s["sample_id"] for s in roster}

    def test_official_qa_type_is_used_not_the_extractor_guess(self):
        """The extractor collapses vbar and hbar into 'bar'; metadata does not."""
        samples, records = _population()
        for sample in samples:
            # An extractor-style label that would destroy the stratification.
            records[sample["sample_id"]]["chart_geometry"] = {"chart_type": "bar"}
        roster = select_roster(samples, records)
        types = {s["metadata"]["qa_type"] for s in roster}
        assert types == set(CHART_TYPES)


class TestCapIsFailClosed:
    def test_counter_raises_before_exceeding_cap(self):
        counter = AttemptCounter(2)
        counter.consume_llm("s", "c", "answer")
        counter.consume_llm("s", "c", "answer")
        with pytest.raises(RuntimeError, match="cap exceeded"):
            counter.consume_llm("s", "c", "answer")

    def test_attempt_cap_exceeds_job_cap_only_by_headroom(self):
        samples, records = _population()
        roster = select_roster(samples, records)
        job_cap = len(roster) * len(CONDITIONS)
        assert job_cap == PILOT_SAMPLE_COUNT * len(CONDITIONS)
        assert job_cap + RETRY_HEADROOM < job_cap * 2, "headroom must not permit a full re-run"


class TestPrepaidReviewBlocks:
    @staticmethod
    def _valid_plan() -> dict:
        samples, records = _population()
        roster = select_roster(samples, records)
        job_cap = len(roster) * len(CONDITIONS)
        return {
            "selected_samples": roster,
            "selected_ids": [s["sample_id"] for s in roster],
            "sample_condition_job_cap": job_cap,
            "llm_http_attempt_cap": job_cap + RETRY_HEADROOM,
            "ocr_http_attempt_cap": 0,
            "paid_calls_executed": 0,
            "conditions": list(CONDITIONS),
            "temperature": 0.0,
        }

    def test_valid_plan_passes(self):
        assert review_plan(self._valid_plan())["status"] == "no_blocking_findings"

    @pytest.mark.parametrize(
        "mutation,expected",
        [
            ({"ocr_http_attempt_cap": 5}, "ocr_cap_not_zero"),
            ({"paid_calls_executed": 3}, "plan_records_paid_calls"),
            ({"temperature": 0.7}, "temperature_not_zero"),
            ({"conditions": [MEASURED_CONDITION]}, "no_same_model_baseline"),
        ],
    )
    def test_unsafe_plans_are_blocked(self, mutation, expected):
        plan = {**self._valid_plan(), **mutation}
        review = review_plan(plan)
        assert review["status"] == "blocking_findings"
        assert expected in review["blocking_findings"]

    def test_inflated_attempt_cap_is_blocked(self):
        plan = self._valid_plan()
        plan["llm_http_attempt_cap"] = 10_000
        assert review_plan(plan)["status"] == "blocking_findings"

    def test_duplicate_roster_entries_are_blocked(self):
        plan = self._valid_plan()
        plan["selected_ids"] = [plan["selected_ids"][0]] * len(plan["selected_ids"])
        assert "duplicate_samples_in_roster" in review_plan(plan)["blocking_findings"]


class TestVerdictBandIsFrozen:
    @staticmethod
    def _results(baseline_correct: int, measured_correct: int, n: int = PILOT_SAMPLE_COUNT):
        rows = []
        for i in range(n):
            rows.append(
                {
                    "sample_id": f"s{i}",
                    "condition": BASELINE_CONDITION,
                    "correct": i < baseline_correct,
                    "parseable": True,
                }
            )
            rows.append(
                {
                    "sample_id": f"s{i}",
                    "condition": MEASURED_CONDITION,
                    "correct": i < measured_correct,
                    "parseable": True,
                }
            )
        return rows

    def test_large_deficit_reads_as_harmful(self):
        verdict = summarize(self._results(10, 10 - NEUTRAL_BAND - 1))["verdict"]
        assert verdict["decision"] == "evidence_harmful_do_not_expand"

    def test_within_band_reads_as_neutral(self):
        verdict = summarize(self._results(8, 8 + NEUTRAL_BAND))["verdict"]
        assert verdict["decision"] == "neutral_evidence_gain_does_not_convert"

    def test_above_band_requires_approval_not_automatic_expansion(self):
        verdict = summarize(self._results(5, 5 + NEUTRAL_BAND + 1))["verdict"]
        assert verdict["decision"] == "positive_request_approval_to_expand"
        assert "approval" in verdict["decision"]

    def test_paired_crosstab_is_reported(self):
        """A margin of totals hides which samples moved; the crosstab does not."""
        paired = summarize(self._results(6, 6))["paired"]
        assert sum(paired.values()) == PILOT_SAMPLE_COUNT


class TestDryRunSpendsNothing:
    def test_default_invocation_performs_no_paid_call(self, tmp_path, monkeypatch):
        def explode(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("dry run attempted a paid call")

        monkeypatch.setattr(
            "scripts.run_plotqa_glm_measured_pilot.call_with_hard_timeout", explode
        )
        code = main(["--output-dir", str(tmp_path)])
        assert code == 0
        plan = json.loads((tmp_path / "frozen_plan.json").read_text(encoding="utf-8"))
        assert plan["paid_calls_executed"] == 0
        assert plan["dry_run"]["paid_calls_executed"] == 0
        assert not (tmp_path / "paid_result.json").exists()

    def test_plan_records_provenance_required_by_invariant_12(self, tmp_path):
        main(["--output-dir", str(tmp_path)])
        plan = json.loads((tmp_path / "frozen_plan.json").read_text(encoding="utf-8"))
        provenance = plan["model_provenance"]
        assert provenance["api_model_id"]
        assert provenance["temperature"] == 0.0
        assert plan["created_at_utc"]
        assert plan["prompt_hashes"]

    def test_roster_hash_is_frozen_in_the_plan(self, tmp_path):
        main(["--output-dir", str(tmp_path)])
        plan = json.loads((tmp_path / "frozen_plan.json").read_text(encoding="utf-8"))
        assert plan["roster_sha256"]
        assert len(plan["selected_ids"]) == PILOT_SAMPLE_COUNT


class TestCheckpointIdentityIsResumable:
    """The resume guard must reject a different experiment, not its own run.

    The first version hashed the whole plan including `created_at_utc`, so the
    hash changed on every rebuild and resuming always raised "checkpoint plan
    hash mismatch". The guard was correct in intent and unusable in practice; a
    run that died mid-way could not be continued without re-paying.
    """

    @staticmethod
    def _plan(**overrides) -> dict:
        samples, records = _population()
        roster = select_roster(samples, records)
        base = {
            "schema_version": "plotqa_glm_measured_pilot_plan_v1",
            "stage": "plotqa_glm_measured_evidence_pilot",
            "conditions": list(CONDITIONS),
            "selected_ids": [s["sample_id"] for s in roster],
            "roster_sha256": "abc",
            "reasoning_model": "GLM-4.6V",
            "temperature": 0.0,
            "prompt_hashes": {"baseline_e0": "h1", "measured_raw": "h2"},
            "scorer_version": "v1",
            "sample_condition_job_cap": 24,
            "llm_http_attempt_cap": 28,
            "ocr_http_attempt_cap": 0,
            "source_artifacts": {"plan": {"sha256": "s1"}},
        }
        base.update(overrides)
        return base

    def test_timestamp_does_not_change_the_identity(self):
        early = self._plan(created_at_utc="2026-09-20T00:00:00+00:00")
        later = self._plan(created_at_utc="2026-09-20T23:59:59+00:00")
        assert experiment_identity_hash(early) == experiment_identity_hash(later)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("selected_ids", ["different"]),
            ("conditions", ["only_one"]),
            ("reasoning_model", "GPT-5.5"),
            ("temperature", 0.7),
            ("prompt_hashes", {"baseline_e0": "changed", "measured_raw": "h2"}),
            ("llm_http_attempt_cap", 9999),
            ("source_artifacts", {"plan": {"sha256": "tampered"}}),
        ],
    )
    def test_meaningful_changes_do_change_the_identity(self, field, value):
        """Anything that could alter a row's meaning must break resume."""
        assert experiment_identity_hash(self._plan()) != experiment_identity_hash(
            self._plan(**{field: value})
        )

    def test_identity_fields_cover_every_result_defining_input(self):
        for required in ("selected_ids", "conditions", "reasoning_model", "temperature"):
            assert required in EXPERIMENT_IDENTITY_FIELDS


class TestArtifactReaders:
    def test_rows_of_handles_both_frozen_shapes(self):
        assert rows_of([{"a": 1}]) == [{"a": 1}]
        assert rows_of({"records": [{"a": 1}]}) == [{"a": 1}]
        assert rows_of({"results": [{"a": 1}]}) == [{"a": 1}]
        assert rows_of({"nothing": 1}) == []
