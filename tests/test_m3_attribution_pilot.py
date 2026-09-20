"""Safety properties of the M3 attribution pilot plan.

These lock the cost-gating and fairness rules that make the pilot approvable:
caps derived from the condition list, a judge distinct from the reasoning
model, and attribution arms kept out of the headline matrix.
"""

import pytest

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
        plan = _plan()
        assert "E6_flat_single >= E5_raw - 1.0" in plan["falsification_signals"]["H1_layering"]
        assert "+5.0" in plan["falsification_signals"]["H2_form_first"]


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

    def test_scoring_inputs_is_empty_for_an_absent_pipeline(self):
        from scripts.run_glm46v_geosym_bench_240x7 import scoring_inputs

        assert scoring_inputs({}) == {}

    def test_scoring_inputs_skips_steps_without_text(self):
        from scripts.run_glm46v_geosym_bench_240x7 import scoring_inputs

        captured = scoring_inputs({"steps": {"L1": {"raw": ""}, "L2": {"raw": "t"}}})
        assert "L1" not in captured["step_raw_text"]
        assert captured["step_raw_text"]["L2"] == "t"
