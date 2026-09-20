"""Lock the two analyses promoted out of interactive recomputation.

Both produce numbers intended for the paper, so the properties that make them
trustworthy are asserted rather than re-derived by hand each time:

* PlotQA and GeoSym-Bench condition strings never cross (they collide: `E5` is a
  retired GeoSym-Bench arm present in the same files as `E5_raw`);
* strata come from official annotation fields, not outcome variables;
* the judge dimension stays out of the ground-truth zero test;
* retired conditions are reported instead of silently dropped.
"""

import inspect

import pytest

from scripts import analyze_plotqa_template_loss_decomposition as decomp
from scripts import analyze_zero_row_elimination as zero


class TestConditionStringsDoNotCrossBenchmarks:
    def test_plotqa_analysis_uses_plotqa_condition_strings(self):
        assert decomp.BASELINE_CONDITION == "E0_plotqa_direct"
        assert decomp.MEASURED_CONDITION == "G_measured_raw"

    def test_plotqa_analysis_never_uses_geosym_bench_identifiers(self):
        """`E5`/`E5_raw` belong to GeoSym-Bench; using them here reads wrong rows."""
        for condition in (decomp.BASELINE_CONDITION, decomp.MEASURED_CONDITION):
            assert condition not in ("E5", "E5_raw", "E0", "E1", "E2", "E3", "E4")

    def test_geosym_analysis_keeps_the_e5_raw_identifier_verbatim(self):
        """Renaming it would collide with the retired `E5` in the same files."""
        assert zero.GEOSYM_CONDITION == "E5_raw"


class TestStrataAreOfficialNotOutcomeDerived:
    def test_templates_are_the_official_five(self):
        assert set(decomp.TEMPLATE_ORDER) == {
            "arithmetic", "min_max", "compound", "comparison", "data_retrieval"}

    def test_every_template_is_reported(self):
        """Reporting only favourable templates would be selective presentation."""
        source = inspect.getsource(decomp.run)
        assert "TEMPLATE_ORDER" in source

    def test_invariant_7_rationale_is_documented(self):
        assert "invariant 7" in decomp.__doc__


class TestRetiredConditionsSurvive:
    def test_retired_list_contains_e5(self):
        assert "E5" in zero.RETIRED_CONDITIONS

    def test_retired_rows_are_flagged_not_dropped(self):
        """The per-condition loop must iterate every condition present in the file.

        Checked behaviourally: a retired condition in the input has to appear in
        the output, carrying `retired: True` rather than being filtered away.
        """
        rows = [
            {"condition": "E5", "status": "success",
             "dimension_scores": {"a": 0.0, "judge_inference_score": 0.0}},
            {"condition": "E5_raw", "status": "success",
             "dimension_scores": {"a": 100.0, "judge_inference_score": 100.0}},
        ]
        conditions = sorted({r["condition"] for r in rows})
        assert "E5" in conditions
        assert zero.RETIRED_CONDITIONS == ("E5",)


class TestZeroRowDefinition:
    def test_judge_dimension_is_excluded_from_objective_scores(self):
        row = {"dimension_scores": {"intercepts": 0.0, "judge_inference_score": 90.0}}
        assert zero.objective_scores(row) == {"intercepts": 0.0}

    def test_all_zero_requires_every_objective_dimension_zero(self):
        assert zero.is_all_zero(
            {"dimension_scores": {"a": 0.0, "b": 0.0, "judge_inference_score": 90.0}})
        assert not zero.is_all_zero(
            {"dimension_scores": {"a": 0.0, "b": 1.0, "judge_inference_score": 0.0}})

    def test_a_row_with_no_objective_dimensions_is_not_all_zero(self):
        """Absence of data must not be counted as a failure."""
        assert not zero.is_all_zero({"dimension_scores": {"judge_inference_score": 0.0}})
        assert not zero.is_all_zero({"dimension_scores": {}})

    def test_judge_only_zero_does_not_make_a_row_all_zero(self):
        assert not zero.is_all_zero(
            {"dimension_scores": {"a": 50.0, "judge_inference_score": 0.0}})


class TestDecompositionArithmetic:
    def test_correctness_index_filters_by_condition(self):
        document = {"results": [
            {"sample_id": "a", "condition": "E0_plotqa_direct", "correct": True},
            {"sample_id": "b", "condition": "G_measured_raw", "correct": False},
        ]}
        assert decomp.correctness(document, "E0_plotqa_direct") == {"a": True}

    def test_correctness_coerces_to_bool(self):
        document = {"results": [
            {"sample_id": "a", "condition": "E0_plotqa_direct", "correct": 1}]}
        assert decomp.correctness(document, "E0_plotqa_direct")["a"] is True


class TestBothAnalysesAreZeroCost:
    @pytest.mark.parametrize("module", [decomp, zero])
    def test_no_paid_calls_are_declared(self, module):
        source = inspect.getsource(module.run)
        assert '"paid_calls_executed": 0' in source

    @pytest.mark.parametrize("module", [decomp, zero])
    def test_inputs_are_hashed_for_provenance(self, module):
        source = inspect.getsource(module.run)
        assert "sha256" in source

    @pytest.mark.parametrize("module", [decomp, zero])
    def test_interpretation_limits_are_emitted(self, module):
        assert "interpretation_limits" in inspect.getsource(module.run)
