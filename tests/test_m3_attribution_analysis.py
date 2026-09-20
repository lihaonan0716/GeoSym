"""The attribution analysis must implement the pre-registered rule, not a
rule chosen after the numbers are visible.

Written before the confirmation run. What these tests pin:

* the decision requires significance AND a positive direction, so a large but
  noisy difference cannot be read as support;
* empty-output rows are excluded rather than scored as zeros, because the
  truncation rate is not guaranteed equal across arms (invariant 6);
* shard files are merged only when they share one experiment identity;
* the transcendental family list for H_B is frozen in code, not selected from
  results (invariant 7).
"""

import json
from pathlib import Path

import pytest

from scripts import analyze_m3_attribution as analysis


def row(sample_id, condition, score, *, status="success", error=""):
    return {"sample_id": sample_id, "condition": condition,
            "total_score": score, "status": status, "error": error}


def pair(prefix, n, treated, control, *, treatment="E5_raw", baseline="E6_flat_single"):
    rows = []
    for i in range(n):
        rows.append(row(f"{prefix}-clean_render_{i:04d}", treatment, treated))
        rows.append(row(f"{prefix}-clean_render_{i:04d}", baseline, control))
    return rows


class TestDecisionRuleRequiresBothSignificanceAndDirection:
    def test_consistent_positive_effect_is_supported(self):
        result = analysis.paired_contrast(
            pair("cubic", 20, 80.0, 70.0), "E5_raw", "E6_flat_single")
        assert result["mean_delta"] == pytest.approx(10.0)
        assert result["supports_hypothesis"] is True

    def test_significant_but_negative_is_not_support(self):
        """A significant result in the wrong direction must never read as a win."""
        result = analysis.paired_contrast(
            pair("cubic", 20, 60.0, 75.0), "E5_raw", "E6_flat_single")
        assert result["significant_at_alpha"] is True
        assert result["direction_positive"] is False
        assert result["supports_hypothesis"] is False

    def test_noisy_positive_mean_is_not_support(self):
        """This is the case the fixed 1.0-point threshold got wrong: a positive
        mean that the noise cannot distinguish from zero."""
        rows = []
        for i, (a, b) in enumerate([(100.0, 0.0), (0.0, 100.0), (60.0, 55.0)]):
            rows.append(row(f"rational-clean_render_{i:04d}", "E5_raw", a))
            rows.append(row(f"rational-clean_render_{i:04d}", "E6_flat_single", b))
        result = analysis.paired_contrast(rows, "E5_raw", "E6_flat_single")
        assert result["mean_delta"] > 0
        assert result["supports_hypothesis"] is False

    def test_confidence_interval_and_effect_size_are_reported(self):
        # Deltas must vary: a constant difference has zero variance, which
        # degenerates the interval to a point. Real scores always vary.
        rows = []
        for i, (a, b) in enumerate([(82.0, 70.0), (78.0, 71.0), (85.0, 69.0),
                                    (80.0, 73.0), (79.0, 68.0)]):
            rows.append(row(f"cubic-clean_render_{i:04d}", "E5_raw", a))
            rows.append(row(f"cubic-clean_render_{i:04d}", "E6_flat_single", b))
        result = analysis.paired_contrast(rows, "E5_raw", "E6_flat_single")
        assert len(result["ci95"]) == 2
        assert result["ci95"][0] < result["mean_delta"] < result["ci95"][1]
        assert result["cohens_d_paired"] is not None
        assert result["supports_hypothesis"] is True

    def test_both_parametric_and_rank_based_tests_are_reported(self):
        result = analysis.paired_contrast(
            pair("cubic", 20, 80.0, 70.0), "E5_raw", "E6_flat_single")
        assert "paired_t" in result and "wilcoxon" in result

    def test_fewer_than_two_pairs_refuses_to_test(self):
        result = analysis.paired_contrast(
            pair("cubic", 1, 80.0, 70.0), "E5_raw", "E6_flat_single")
        assert result["insufficient_data"] is True


class TestPairingIsBySample:
    def test_only_samples_scored_in_both_arms_are_paired(self):
        rows = [
            row("cubic-clean_render_0001", "E5_raw", 80.0),
            row("cubic-clean_render_0001", "E6_flat_single", 70.0),
            row("cubic-clean_render_0002", "E5_raw", 90.0),  # no partner
        ]
        assert analysis.paired_contrast(rows, "E5_raw", "E6_flat_single")["n_paired"] == 1

    def test_error_rows_are_not_paired(self):
        rows = [
            row("cubic-clean_render_0001", "E5_raw", 80.0),
            row("cubic-clean_render_0001", "E6_flat_single", None,
                status="error", error="empty_model_output_finish_reason_length"),
        ]
        result = analysis.paired_contrast(rows, "E5_raw", "E6_flat_single")
        assert result["n_paired"] == 0


class TestEmptyOutputHandling:
    def test_empty_rows_are_counted_per_arm(self):
        rows = [
            row("a-clean_render_0001", "E5_raw", 80.0),
            row("a-clean_render_0002", "E6_flat_single", None,
                status="error", error="empty_model_output_finish_reason_length"),
            row("a-clean_render_0003", "E6_flat_single", 70.0),
        ]
        report = analysis.empty_output_report(rows)
        assert report["by_condition"]["E6_flat_single"]["empty"] == 1
        assert report["total_empty"] == 1
        assert report["evenly_spread"] is False

    def test_uneven_truncation_is_flagged_because_it_biases_the_contrast(self):
        rows = [row("a-clean_render_0001", "E5_raw", 80.0)]
        rows += [row(f"a-clean_render_{i:04d}", "E6_flat_single", None,
                     status="error", error="empty_model_output_finish_reason_length")
                 for i in range(3)]
        assert analysis.empty_output_report(rows)["evenly_spread"] is False


class TestFrozenSubsetForHypothesisB:
    def test_transcendental_families_are_frozen_in_code(self):
        assert set(analysis.TRANSCENDENTAL_FAMILIES) == {
            "exponential", "logarithmic", "rational", "piecewise_linear"}

    def test_family_filter_restricts_the_contrast(self):
        rows = pair("exponential", 5, 80.0, 60.0,
                    treatment="E7_form_first", baseline="E5_raw")
        rows += pair("linear", 5, 10.0, 90.0,
                     treatment="E7_form_first", baseline="E5_raw")
        restricted = analysis.paired_contrast(
            rows, "E7_form_first", "E5_raw",
            families=analysis.TRANSCENDENTAL_FAMILIES)
        assert restricted["n_paired"] == 5
        assert restricted["mean_delta"] == pytest.approx(20.0)


class TestShardMerging:
    def _write(self, path: Path, plan_hash: str, rows: list[dict]):
        path.write_text(json.dumps({
            "plan_sha256": plan_hash, "stage_status": "complete",
            "actual_llm_http_attempts": 10, "cap_exhausted": False,
            "results": rows}), encoding="utf-8")

    def test_shards_sharing_an_identity_are_merged(self, tmp_path: Path):
        self._write(tmp_path / "paid_result.shard00of02.json", "a" * 64,
                    [row("cubic-clean_render_0001", "E5_raw", 80.0)])
        self._write(tmp_path / "paid_result.shard01of02.json", "a" * 64,
                    [row("cubic-clean_render_0002", "E5_raw", 70.0)])
        rows, provenance = analysis.collect_rows(tmp_path)
        assert len(rows) == 2
        assert len(provenance) == 2

    def test_mismatched_identities_refuse_to_merge(self, tmp_path: Path):
        self._write(tmp_path / "paid_result.shard00of02.json", "a" * 64,
                    [row("cubic-clean_render_0001", "E5_raw", 80.0)])
        self._write(tmp_path / "paid_result.shard01of02.json", "b" * 64,
                    [row("cubic-clean_render_0002", "E5_raw", 70.0)])
        with pytest.raises(ValueError, match="disagree on plan_sha256"):
            analysis.collect_rows(tmp_path)

    def test_missing_results_raise_rather_than_report_an_empty_run(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            analysis.collect_rows(tmp_path)

    def test_provenance_hashes_every_input(self, tmp_path: Path):
        self._write(tmp_path / "paid_result.json", "a" * 64,
                    [row("cubic-clean_render_0001", "E5_raw", 80.0)])
        _, provenance = analysis.collect_rows(tmp_path)
        assert len(provenance[0]["sha256"]) == 64


class TestUnderpoweredRunsAreLabelled:
    def test_small_sample_emits_a_power_warning(self, tmp_path: Path):
        rows = pair("cubic", 3, 80.0, 70.0)
        (tmp_path / "paid_result.json").write_text(json.dumps({
            "plan_sha256": "a" * 64, "stage_status": "complete",
            "actual_llm_http_attempts": 10, "cap_exhausted": False,
            "results": rows}), encoding="utf-8")
        results = analysis.run(tmp_path)
        assert results["power_warning"] is not None
        assert "path check" in results["power_warning"]
