"""The attribution analysis must implement the pre-registered rule.

Written while the confirmation run was still executing, so none of these
thresholds or directions could be chosen to fit the outcome.

What they pin:

* a claim needs significance AND the stated direction -- a significant result in
  the opposite direction is reported as a refutation, not renamed;
* failed calls are excluded rather than scored as incorrect, because the failure
  rate is not guaranteed equal across arms (invariant 6);
* every official template appears in the stratified table, including strata that
  favour the baseline (invariant 8);
* the paired test is McNemar on binary outcomes, not a t-test on 0/1.
"""

import json
from pathlib import Path

import pytest

from scripts import analyze_plotqa_numeric_attribution as analysis


def row(sample_id, condition, correct, *, status="success", error="",
        template="min_max", share=None):
    return {"sample_id": sample_id, "condition": condition, "correct": correct,
            "status": status, "error": error, "qa_template": template,
            "both_label_share": share}


def paired(n, treated_correct, control_correct, *, treatment="G_measured_raw",
           baseline="E0_plotqa_direct", template="min_max"):
    """n samples where the treatment/control outcomes are as given."""
    rows = []
    for i in range(n):
        sid = f"s{i:03d}"
        rows.append(row(sid, treatment, treated_correct, template=template))
        rows.append(row(sid, baseline, control_correct, template=template))
    return rows


class TestDecisionRuleNeedsSignificanceAndDirection:
    def test_consistent_treatment_advantage_is_detected(self):
        # 20 samples treatment-right/baseline-wrong: maximally discordant.
        result = analysis.paired_binary_contrast(
            paired(20, True, False), "G_measured_raw", "E0_plotqa_direct")
        assert result["paired_difference"] == pytest.approx(1.0)
        assert result["significant_at_alpha"] is True
        assert result["treatment_ahead_significantly"] is True
        assert result["baseline_ahead_significantly"] is False

    def test_significant_baseline_advantage_is_not_reported_as_treatment_win(self):
        result = analysis.paired_binary_contrast(
            paired(20, False, True), "G_measured_raw", "E0_plotqa_direct")
        assert result["significant_at_alpha"] is True
        assert result["direction_positive"] is False
        assert result["treatment_ahead_significantly"] is False
        assert result["baseline_ahead_significantly"] is True

    def test_all_concordant_pairs_yield_no_significance(self):
        """No discordant pairs means no evidence either way, not a win."""
        result = analysis.paired_binary_contrast(
            paired(30, True, True), "G_measured_raw", "E0_plotqa_direct")
        assert result["discordant_pairs"]["only_treatment"] == 0
        assert result["mcnemar_exact_p"] == 1.0
        assert result["treatment_ahead_significantly"] is False

    def test_a_small_lead_on_few_discordances_is_not_significant(self):
        rows = []
        for i in range(40):
            rows.append(row(f"s{i}", "G_measured_raw", i < 21))
            rows.append(row(f"s{i}", "E0_plotqa_direct", i >= 21))
        result = analysis.paired_binary_contrast(
            rows, "G_measured_raw", "E0_plotqa_direct")
        assert result["paired_difference"] > 0
        # 21 vs 19 discordant: nowhere near alpha=0.05.
        assert result["significant_at_alpha"] is False
        assert result["treatment_ahead_significantly"] is False

    def test_confidence_interval_brackets_the_difference(self):
        rows = []
        for i in range(30):
            rows.append(row(f"s{i}", "G_measured_raw", i < 25))
            rows.append(row(f"s{i}", "E0_plotqa_direct", i < 15))
        result = analysis.paired_binary_contrast(
            rows, "G_measured_raw", "E0_plotqa_direct")
        lo, hi = result["ci95"]
        assert lo < result["paired_difference"] < hi

    def test_fewer_than_two_pairs_refuses_to_test(self):
        result = analysis.paired_binary_contrast(
            paired(1, True, False), "G_measured_raw", "E0_plotqa_direct")
        assert result["insufficient_data"] is True


class TestPairingAndExclusion:
    def test_only_samples_answered_in_both_arms_are_paired(self):
        rows = [
            row("a", "G_measured_raw", True),
            row("a", "E0_plotqa_direct", False),
            row("b", "G_measured_raw", True),  # no partner
        ]
        result = analysis.paired_binary_contrast(
            rows, "G_measured_raw", "E0_plotqa_direct")
        assert result["n_paired"] == 1

    def test_failed_rows_are_not_paired(self):
        """A timeout must not become a wrong answer for the arm that timed out."""
        rows = [
            row("a", "E0_plotqa_direct", True),
            row("a", "G_measured_raw", None, status="error",
                error="APITimeoutError: Request timed out."),
        ]
        result = analysis.paired_binary_contrast(
            rows, "G_measured_raw", "E0_plotqa_direct")
        assert result["n_paired"] == 0

    def test_failures_are_counted_per_arm_and_unevenness_flagged(self):
        rows = [
            row("a", "G_measured_raw", None, status="error", error="APITimeoutError: x"),
            row("b", "G_measured_raw", True),
            row("a", "E0_plotqa_direct", True),
            row("b", "E0_plotqa_direct", True),
        ]
        report = analysis.failure_report(rows)
        assert report["by_condition"]["G_measured_raw"]["failed"] == 1
        assert report["by_condition"]["E0_plotqa_direct"]["failed"] == 0
        assert report["total_failed"] == 1
        assert report["evenly_spread"] is False

    def test_empty_output_is_counted_separately_from_other_failures(self):
        rows = [row("a", "G_measured_raw", None, status="error",
                    error="empty_model_output")]
        report = analysis.failure_report(rows)
        assert report["by_condition"]["G_measured_raw"]["empty_output"] == 1


class TestEveryStratumIsReported:
    def test_a_stratum_favouring_the_baseline_still_appears(self):
        """Invariant 8: losses are reported at the same prominence as wins."""
        rows = paired(10, True, False, template="min_max")
        rows += paired(10, False, True, template="arithmetic")
        table = analysis.per_stratum(rows)
        assert set(table) == {"min_max", "arithmetic"}
        assert table["arithmetic"]["E0_plotqa_direct"]["accuracy"] == 1.0
        assert table["arithmetic"]["G_measured_raw"]["accuracy"] == 0.0

    def test_template_filter_restricts_the_contrast(self):
        rows = paired(6, True, False, template="min_max")
        rows += paired(6, False, True, template="arithmetic")
        restricted = analysis.paired_binary_contrast(
            rows, "G_measured_raw", "E0_plotqa_direct", templates=("min_max",))
        assert restricted["n_paired"] == 6
        assert restricted["paired_difference"] == pytest.approx(1.0)

    def test_a_stratum_with_no_scored_rows_reports_none_not_zero(self):
        rows = [row("a", "G_measured_raw", None, status="error",
                    error="x", template="min_max")]
        table = analysis.per_stratum(rows)
        assert table["min_max"]["G_measured_raw"]["accuracy"] is None


class TestLabelCompletenessAssociation:
    def test_higher_share_when_correct_is_an_association(self):
        rows = [row(f"c{i}", "G_measured_raw", True, share=0.9) for i in range(8)]
        rows += [row(f"w{i}", "G_measured_raw", False, share=0.2) for i in range(8)]
        result = analysis.label_completeness_association(rows)
        assert result["direction_positive"] is True
        assert result["supports_hypothesis"] is True

    def test_inverted_association_does_not_support_the_hypothesis(self):
        rows = [row(f"c{i}", "G_measured_raw", True, share=0.1) for i in range(8)]
        rows += [row(f"w{i}", "G_measured_raw", False, share=0.9) for i in range(8)]
        result = analysis.label_completeness_association(rows)
        assert result["significant_at_alpha"] is True
        assert result["direction_positive"] is False
        assert result["supports_hypothesis"] is False

    def test_missing_shares_are_skipped_not_treated_as_zero(self):
        rows = [row(f"c{i}", "G_measured_raw", True, share=None) for i in range(8)]
        rows += [row(f"w{i}", "G_measured_raw", False, share=0.5) for i in range(8)]
        result = analysis.label_completeness_association(rows)
        assert result["insufficient_data"] is True

    def test_only_the_measured_arm_contributes(self):
        rows = [row(f"c{i}", "E0_plotqa_direct", True, share=0.9) for i in range(8)]
        rows += [row(f"w{i}", "E0_plotqa_direct", False, share=0.1) for i in range(8)]
        assert analysis.label_completeness_association(rows)["insufficient_data"] is True


class TestVerdictsNameTheOutcomeHonestly:
    def test_baseline_ahead_is_named_penalty_confirmed(self):
        block = analysis.paired_binary_contrast(
            paired(20, False, True), "G_measured_raw", "E0_plotqa_direct")
        assert analysis._verdict_h_n1(block) == "numeric_penalty_confirmed_on_current_extractor"

    def test_measured_ahead_is_named_a_refutation_not_a_win(self):
        """H_N1 predicted the penalty; the measured arm winning REFUTES it."""
        block = analysis.paired_binary_contrast(
            paired(20, True, False), "G_measured_raw", "E0_plotqa_direct")
        assert analysis._verdict_h_n1(block).startswith("REFUTED")

    def test_null_result_is_not_dressed_up(self):
        block = analysis.paired_binary_contrast(
            paired(30, True, True), "G_measured_raw", "E0_plotqa_direct")
        assert analysis._verdict_h_n1(block) == (
            "no_significant_difference_penalty_not_reproduced_at_this_n")

    def test_h_n2_evidence_only_ahead_is_a_refutation(self):
        block = analysis.paired_binary_contrast(
            paired(20, False, True, baseline="Measured_flat_answer_baseline"),
            "G_measured_raw", "Measured_flat_answer_baseline")
        assert analysis._verdict_h_n2(block).startswith("REFUTED")


class TestEndToEnd:
    def _write(self, path: Path, rows):
        path.write_text(json.dumps({
            "stage": "confirmation", "stage_status": "complete",
            "plan_sha256": "a" * 64, "actual_llm_http_attempts": 240,
            "llm_http_attempt_cap": 480, "cap_exhausted": False,
            "actual_ocr_http_attempts": 0, "records_sha256": "b" * 64,
            "model_provenance": {"reasoning_model": "GPT-5.5", "temperature": 0.0},
            "results": rows}), encoding="utf-8")

    def test_analysis_reports_the_measured_ocr_count(self, tmp_path: Path):
        path = tmp_path / "paid_result.confirmation.json"
        self._write(path, paired(10, True, False))
        results = analysis.run(path)
        assert results["input"]["ocr_http_attempts_measured"] == 0
        assert results["paid_calls_executed"] == 0

    def test_analysis_states_it_is_not_a_victory_claim(self, tmp_path: Path):
        path = tmp_path / "paid_result.confirmation.json"
        self._write(path, paired(10, True, False))
        results = analysis.run(path)
        assert "not a headline claim" in results["attribution_not_victory"]

    def test_markdown_renders_without_error(self, tmp_path: Path):
        path = tmp_path / "paid_result.confirmation.json"
        rows = paired(10, True, False, template="min_max")
        rows += paired(10, False, True, template="arithmetic")
        rows += [row(f"f{i}", "Measured_flat_answer_baseline", i % 2 == 0)
                 for i in range(10)]
        self._write(path, rows)
        text = analysis.render(analysis.run(path))
        assert "PlotQA numeric attribution" in text
        assert "arithmetic" in text and "min_max" in text
        assert "OCR HTTP attempts (measured, not asserted)" in text
