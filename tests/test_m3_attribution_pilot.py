"""Safety properties of the M3 attribution pilot plan.

These lock the cost-gating and fairness rules that make the pilot approvable:
caps derived from the condition list, a judge distinct from the reasoning
model, and attribution arms kept out of the headline matrix.
"""

import importlib.util

import pytest

#: Three tests below import the paid runner inside their own bodies. It is
#: dev-local (`scripts/*` is gitignored), so mark just those rather than
#: skipping a module whose other tests check the plan's cost gating.
requires_runner = pytest.mark.skipif(
    importlib.util.find_spec("scripts.run_glm46v_geosym_bench_240x7") is None,
    reason="paid runner is dev-local; absent in a clean clone")

from app.reasoning_pipeline import (
    ATTRIBUTION_CONDITION_MAP,
    CONDITION_MAP,
    CONDITION_ORDER,
    DISPATCHABLE_CONDITION_MAP,
    _format_detected_coords,
)
from scripts.run_m3_attribution_pilot import (
    PILOT_8_ROSTER,
    PILOT_CONDITIONS,
    REFERENCE_CONDITION,
    build_plan,
    call_caps,
    review_plan,
)


def _plan(**kw):
    return build_plan(
        kw.get("roster", PILOT_8_ROSTER),
        kw.get("reasoning_model", "GLM-4.6V"),
        kw.get("judge_model", "Qwen3-VL-Plus"),
    )


class TestMatrixIsolation:
    def test_attribution_arms_are_not_in_the_headline_matrix(self):
        """E6/E7 must not appear in CONDITION_ORDER.

        reasoning_schema validation requires a record for every condition in
        that list, so adding them would fail validation on every existing run.
        """
        for arm in ("E6_flat_single", "E7_form_first"):
            assert arm not in CONDITION_MAP
            assert arm not in CONDITION_ORDER
            assert arm in ATTRIBUTION_CONDITION_MAP
            assert arm in DISPATCHABLE_CONDITION_MAP

    def test_matrix_still_has_exactly_the_six_live_conditions(self):
        assert CONDITION_ORDER == ["E0", "E1", "E2", "E3", "E4", "E5_raw"]

    def test_reference_arm_is_geosym_itself(self):
        assert REFERENCE_CONDITION == "E5_raw"
        assert REFERENCE_CONDITION in CONDITION_MAP


class TestCostGating:
    def test_caps_derive_from_the_condition_list(self):
        """A literal cap over-authorises the moment an arm changes."""
        caps = call_caps(8)
        assert caps["sample_condition_job_cap"] == 8 * len(PILOT_CONDITIONS)
        # 3 + 1 + 3 reasoning calls per sample.
        assert caps["reasoning_llm_call_cap"] == 8 * 7
        assert caps["judge_llm_call_cap"] == 8 * len(PILOT_CONDITIONS)
        assert caps["ocr_http_attempt_cap"] == 0

    def test_http_cap_covers_one_retry_and_no_more(self):
        caps = call_caps(8)
        calls = caps["reasoning_llm_call_cap"] + caps["judge_llm_call_cap"]
        assert caps["llm_http_attempt_cap"] == calls * 2

    def test_caps_scale_with_sample_count(self):
        assert call_caps(16)["reasoning_llm_call_cap"] == 2 * call_caps(8)["reasoning_llm_call_cap"]

    def test_plan_defaults_to_spending_nothing(self):
        plan = _plan()
        assert plan["paid_calls_executed"] == 0
        assert plan["dry_run"]["paid_calls_executed"] == 0

    def test_review_flags_a_tampered_cap(self):
        plan = _plan()
        plan["llm_http_attempt_cap"] = 99999
        review = review_plan(plan)
        assert review["status"] == "blocking_findings_present"
        assert any("llm_cap_not_" in f for f in review["blocking_findings"])


class TestScientificGuards:
    def test_clean_plan_has_no_blocking_findings(self):
        assert review_plan(_plan())["status"] == "no_blocking_findings"

    def test_judge_must_differ_from_reasoning_model(self):
        """Invariant 12: a model must not grade its own chain."""
        plan = _plan(judge_model="GLM-4.6V")
        assert plan["paid_run_approved"] is False
        assert plan["dry_run"]["stop_reason"] == "judge_model_must_differ_from_reasoning_model"
        assert "judge_model_equals_reasoning_model" in review_plan(plan)["blocking_findings"]

    def test_temperature_is_zero(self):
        assert _plan()["model_provenance"]["temperature"] == 0.0

    def test_duplicate_roster_entries_block_the_run(self):
        dup = PILOT_8_ROSTER[:1] + PILOT_8_ROSTER[:1]
        plan = _plan(roster=dup)
        assert plan["paid_run_approved"] is False
        assert plan["dry_run"]["stop_reason"] == "roster_contains_duplicates"

    def test_missing_ground_truth_blocks_the_run(self):
        plan = _plan(roster=("linear-clean_render_9999",))
        assert plan["paid_run_approved"] is False
        assert str(plan["dry_run"]["stop_reason"]).startswith("missing_ground_truth")

    def test_roster_covers_every_family_once(self):
        counts = _plan()["family_counts"]
        assert sorted(counts) == [
            "absolute", "cubic", "exponential", "linear",
            "logarithmic", "piecewise_linear", "quadratic", "rational",
        ]
        assert set(counts.values()) == {1}

    def test_prompt_wording_is_pinned_by_hash(self):
        hashes = _plan()["prompt_hashes"]
        assert hashes["reasoning_L3_form_first_prompt.txt"]
        assert hashes["ablation_flat_single_geosym_prompt.txt"]
        # The layered arm's own templates are pinned too, or a silent edit to
        # L1/L2 would confound the E7 contrast.
        assert hashes["ablation_coords_only_L1_prompt.txt"]
        assert hashes["reasoning_L2_prompt.txt"]

    def test_pilot_does_not_claim_to_decide_the_hypotheses(self):
        """Eight samples cannot separate arms; the plan must say so."""
        plan = _plan()
        assert "cannot separate arms" in plan["pilot_decides_nothing_about_hypotheses"]
        assert plan["falsification_signals"]["H1_layering"]
        assert plan["falsification_signals"]["H2_form_first"]

    def test_falsification_signals_are_frozen_before_spending(self):
        """Both signals must name their arms, their direction, and their test.

        Asserted as properties rather than as literal thresholds: the original
        fixed thresholds (-1.0 total points, +5.0 expression points) were
        replaced after a cross-batch probe measured per-cell noise at sd 31.9,
        which puts the standard error of an arm mean near 3.0 even at n=110. A
        1.0-point rule was inside the noise, so the decision is now a paired
        test requiring significance AND a positive direction.
        """
        signals = _plan()["falsification_signals"]
        layering = signals["H1_layering"]
        assert "E5_raw" in layering and "E6_flat_single" in layering
        assert "alpha=0.05" in layering
        assert "positive" in layering
        # The losing outcome must be spelled out, so it cannot be renamed later.
        assert "never as a layering gain" in layering

        form_first = signals["H2_form_first"]
        assert "E7_form_first" in form_first and "E5_raw" in form_first
        assert "transcendental" in form_first
        assert "unsupported" in form_first

    def test_empty_output_exclusion_is_part_of_the_frozen_rule(self):
        """An empty body scored as zero would bias whichever arm truncates more."""
        shared = _plan()["falsification_signals"]["shared_rules"]
        assert "empty" in shared
        assert "invariant 6" in shared

    def test_merged_experiments_are_declared_in_the_plan(self):
        """A and B run as one batch; the merge must be auditable, not implicit."""
        merged = _plan()["merged_experiments"]
        assert merged["merged"] == ["A_layering", "B_form_first"]
        assert merged["h2_transcendental_sample_count"] == 80
        assert merged["calls_saved_versus_sequential"] == 600


class TestEvidenceSymmetry:
    def test_all_arms_share_one_coordinate_renderer(self):
        """A drifting payload would confound the layering contrast."""
        assert _format_detected_coords([(0, 1), (2, 5)]) == "(0, 1), (2, 5)"

    def test_empty_coordinates_render_explicitly(self):
        assert _format_detected_coords([]) == "No coordinate points were detected."

    def test_plan_records_the_fairness_controls(self):
        controls = " ".join(_plan()["fairness_controls"])
        assert "byte-identical M0 coordinate evidence" in controls
        assert "invariant 6" in controls


class TestPersistence:
    def test_plan_requires_per_layer_raw_text(self):
        """The 240x7 artifact stored hashes; two defects became unfixable."""
        requirement = _plan()["persistence_requirement"]
        assert "per-layer raw text" in requirement

    @requires_runner
    def test_scoring_inputs_captures_raw_text_and_parsed_answer(self):
        from scripts.run_glm46v_geosym_bench_240x7 import scoring_inputs

        pipeline = {
            "steps": {
                "L1": {"raw": "l1 text", "parsed": {"a": 1}},
                "L3": {"raw": "l3 text", "parsed": {"expression": "x^2"}},
            },
            "final_parsed": {"expression": "x^2"},
        }
        captured = scoring_inputs(pipeline)
        assert captured["step_raw_text"]["L3"] == "l3 text"
        assert captured["final_parsed"]["expression"] == "x^2"

    @requires_runner
    def test_scoring_inputs_is_empty_for_an_absent_pipeline(self):
        from scripts.run_glm46v_geosym_bench_240x7 import scoring_inputs

        assert scoring_inputs({}) == {}

    @requires_runner
    def test_scoring_inputs_skips_steps_without_text(self):
        from scripts.run_glm46v_geosym_bench_240x7 import scoring_inputs

        captured = scoring_inputs({"steps": {"L1": {"raw": ""}, "L2": {"raw": "t"}}})
        assert "L1" not in captured["step_raw_text"]
        assert captured["step_raw_text"]["L2"] == "t"


class TestPaidOcrIsImpossibleOnThisPath:
    """The plan freezes `ocr_http_attempt_cap = 0`; the code must make it true.

    The first implementation wrapped only the LLM counter, so M0 extraction took
    the default route -- and that route is paid. `recognize_intersections` only
    tries the free clean-grid extractor first when
    `GEOSYM_CLEAN_GRID_FALLBACK_FIRST` is set, which it is not, so it fell
    through to `ChartExtractor` and billed one Mathpix call per sample. A probe
    measured exactly one `requests.post` per image, while the result file
    recorded `actual_ocr_http_attempts: 0` as a hardcoded constant.
    """

    def test_a_mathpix_call_raises_and_is_counted(self):
        import recognizer.mathpix.mathpix_image_ocr as mathpix_module

        from scripts.run_m3_attribution_pilot import OcrCallAttempted, forbid_paid_ocr

        with forbid_paid_ocr() as attempts:
            with pytest.raises(OcrCallAttempted):
                mathpix_module.requests.post("https://api.mathpix.com", json={})
            assert attempts[0] == 1

    def test_transport_is_restored_afterwards(self):
        import recognizer.mathpix.mathpix_image_ocr as mathpix_module

        from scripts.run_m3_attribution_pilot import forbid_paid_ocr

        original = mathpix_module.requests.post
        with forbid_paid_ocr():
            assert mathpix_module.requests.post is not original
        assert mathpix_module.requests.post is original

    def test_clean_grid_only_replaces_the_paid_extractor(self):
        import evaluator.adapters.recognizer as recognizer_module

        from scripts.run_m3_attribution_pilot import clean_grid_only_m0

        original = recognizer_module.RecognizerAdapter.recognize_intersections
        with clean_grid_only_m0():
            assert recognizer_module.RecognizerAdapter.recognize_intersections is not original
        assert recognizer_module.RecognizerAdapter.recognize_intersections is original

    def test_extraction_under_both_guards_bills_nothing(self):
        """The real check: run M0 on one image of every family, measure zero."""
        from app.reasoning_pipeline import extract_geosym_coords
        from scripts.run_m3_attribution_pilot import clean_grid_only_m0, forbid_paid_ocr

        images = ROOT_IMAGES = __import__("pathlib").Path("dataset/images")
        if not images.is_dir():
            pytest.skip("dataset/images not present")
        picks = [sorted(d.glob("*.png"))[0]
                 for d in sorted(x for x in images.iterdir() if x.is_dir())
                 if list(d.glob("*.png"))]
        if not picks:
            pytest.skip("no benchmark images available")

        with forbid_paid_ocr() as attempts, clean_grid_only_m0():
            for image in picks:
                coords, _facts, ok, _reason = extract_geosym_coords(
                    "GLM-4.6V", "dummy-key", image, {})
                assert ok is True
                assert coords
            assert attempts[0] == 0

    def test_result_records_a_measured_ocr_count_not_a_literal(self):
        """`actual_ocr_http_attempts` must come from the counter."""
        import inspect

        from scripts import run_m3_attribution_pilot as runner

        source = inspect.getsource(runner.write_result)
        assert '"actual_ocr_http_attempts": ocr_http_attempts,' in source
        assert '"actual_ocr_http_attempts": 0,' not in source

    def test_a_nonzero_measured_count_reaches_the_result_file(self, tmp_path):
        from scripts import run_m3_attribution_pilot as runner

        plan = _plan()
        payload = runner.write_result(
            plan, tmp_path / "r.json", [], runner.AttemptCounter(10),
            "complete", 7)
        assert payload["actual_ocr_http_attempts"] == 7
