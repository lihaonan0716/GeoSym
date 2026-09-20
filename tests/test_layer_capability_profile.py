import json
from pathlib import Path

import pytest

from scripts.analyze_layer_capability_profile import (
    LAYER_SPEC,
    build_profile,
    paired_delta,
    verify_layer_spec_matches_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_layer_spec_matches_live_config():
    """The restated weights must track config/default_config.yaml.

    Invariant 13: the YAML is authoritative. A drift here would mis-attribute
    the layer effect while every printed number still looked plausible.
    """
    findings = verify_layer_spec_matches_config(ROOT / "config" / "default_config.yaml")
    assert findings == [], findings


def test_layer_spec_drift_is_detected(tmp_path: Path):
    """A changed weight must be reported, not silently absorbed."""
    config = tmp_path / "cfg.yaml"
    config.write_text(
        "reasoning:\n"
        "  layers:\n"
        "    G1_visual_readout:\n"
        "      weight: 0.30\n"
        "      dimensions:\n"
        "        intercepts: 0.90\n"   # drifted from 0.50
        "        extrema: 0.50\n"
        "    G2_quantitative_analysis:\n"
        "      weight: 0.30\n"
        "      dimensions:\n"
        "        predictions: 0.25\n"
        "        rate_of_change: 0.15\n"
        "        monotonicity: 0.35\n"
        "        symmetry: 0.25\n"
        "    G3_mathematical_inference:\n"
        "      weight: 0.40\n"
        "      dimensions:\n"
        "        expression: 0.30\n"
        "        domain: 0.20\n"
        "        range: 0.20\n"
        "        judge_inference_score: 0.30\n",
        encoding="utf-8",
    )
    findings = verify_layer_spec_matches_config(config)
    assert any("intercepts" in f for f in findings), findings


def test_paired_delta_is_paired_not_difference_of_means():
    """Pairing must survive: identical means with opposite per-sample moves."""
    treatment = [10.0, 20.0, 30.0]
    control = [20.0, 30.0, 10.0]
    result = paired_delta(treatment, control)
    assert result["n"] == 3
    # Means are equal, so an unpaired test would report exactly 0 with no spread.
    assert result["delta"] == 0.0
    assert result["sd_of_differences"] > 0


def _artifact(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "paid_result.json"
    path.write_text(json.dumps({"results": rows}), encoding="utf-8")
    return path


def _dims(value: float) -> dict[str, float]:
    return {
        "intercepts": value, "extrema": value,
        "predictions": value, "rate_of_change": value,
        "monotonicity": value, "symmetry": value,
        "expression": value, "domain": value, "range": value,
        "judge_inference_score": value,
    }


def test_profile_excludes_incomplete_samples(tmp_path: Path):
    """A sample missing one condition must drop out of every contrast."""
    rows = []
    for sample in ("s0", "s1"):
        for condition in ("E0", "E5_raw"):
            rows.append({
                "sample_id": sample, "condition": condition, "status": "success",
                "family": "linear", "total_score": 50.0, "dimension_scores": _dims(50.0),
            })
    # s2 has only the GeoSym arm, so it is not complete-case.
    rows.append({
        "sample_id": "s2", "condition": "E5_raw", "status": "success",
        "family": "linear", "total_score": 99.0, "dimension_scores": _dims(99.0),
    })

    profile = build_profile(_artifact(tmp_path, rows), ROOT / "config" / "default_config.yaml")
    assert profile["sample_counts"]["samples_seen"] == 3
    assert profile["sample_counts"]["complete_case_samples"] == 2
    # The 99.0 outlier lives only in the incomplete sample, so it must not move the mean.
    assert profile["per_condition"]["E5_raw"]["means"]["total"] == pytest.approx(50.0)


def test_profile_reports_retired_arm_as_historical(tmp_path: Path):
    """E5 must be carried and labelled, never silently dropped or promoted."""
    rows = []
    for sample in ("s0", "s1"):
        for condition, value in (("E0", 40.0), ("E5", 80.0), ("E5_raw", 81.0)):
            rows.append({
                "sample_id": sample, "condition": condition, "status": "success",
                "family": "cubic", "total_score": value, "dimension_scores": _dims(value),
            })

    profile = build_profile(_artifact(tmp_path, rows), ROOT / "config" / "default_config.yaml")
    assert profile["per_condition"]["E5"]["retired"] is True
    assert profile["per_condition"]["E5_raw"]["retired"] is False
    assert profile["geosym_condition"] == "E5_raw"
    contrast = profile["paired_contrasts"]["E5_raw_minus_E5"]
    assert contrast["baseline_retired"] is True


def test_profile_separates_judge_from_objective_l3(tmp_path: Path):
    """L3 must be reportable without the judge dimension.

    A judge-driven L3 result is a scoring artifact, not a capability finding,
    so the two must not be collapsible into one number.
    """
    dims = _dims(50.0)
    dims["judge_inference_score"] = 0.0
    rows = []
    for sample in ("s0", "s1"):
        for condition in ("E0", "E5_raw"):
            rows.append({
                "sample_id": sample, "condition": condition, "status": "success",
                "family": "rational", "total_score": 50.0, "dimension_scores": dict(dims),
            })

    profile = build_profile(_artifact(tmp_path, rows), ROOT / "config" / "default_config.yaml")
    means = profile["per_condition"]["E5_raw"]["means"]
    assert means["L3_inference_objective_only"] == pytest.approx(50.0)
    assert means["judge_only"] == pytest.approx(0.0)
    assert means["L3_inference"] < means["L3_inference_objective_only"]


def test_profile_skips_failed_rows(tmp_path: Path):
    """Non-success rows must be counted as skipped, not scored as zero."""
    rows = [
        {"sample_id": "s0", "condition": "E0", "status": "success",
         "family": "linear", "total_score": 60.0, "dimension_scores": _dims(60.0)},
        {"sample_id": "s0", "condition": "E5_raw", "status": "success",
         "family": "linear", "total_score": 70.0, "dimension_scores": _dims(70.0)},
        {"sample_id": "s1", "condition": "E0", "status": "pipeline_error",
         "family": "linear", "dimension_scores": {}},
    ]
    profile = build_profile(_artifact(tmp_path, rows), ROOT / "config" / "default_config.yaml")
    assert profile["sample_counts"]["skipped_rows"]["pipeline_error"] == 1
    assert profile["per_condition"]["E5_raw"]["means"]["total"] == pytest.approx(70.0)


def test_profile_declares_zero_paid_calls(tmp_path: Path):
    rows = [
        {"sample_id": "s0", "condition": "E0", "status": "success",
         "family": "linear", "total_score": 60.0, "dimension_scores": _dims(60.0)},
        {"sample_id": "s0", "condition": "E5_raw", "status": "success",
         "family": "linear", "total_score": 70.0, "dimension_scores": _dims(70.0)},
    ]
    profile = build_profile(_artifact(tmp_path, rows), ROOT / "config" / "default_config.yaml")
    assert profile["paid_calls_executed"] == 0
    assert profile["analysis_type"] == "zero_cost_offline_reanalysis"
    assert len(LAYER_SPEC) == 3
