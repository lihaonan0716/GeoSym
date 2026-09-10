import json
from pathlib import Path

from scripts.analyze_plotqa_remaining_bottlenecks import build_audit


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_remaining_bottleneck_audit_recommends_m0_repair(tmp_path: Path) -> None:
    confirmation_path = tmp_path / "confirmation.json"
    projection_path = tmp_path / "projection.json"
    repair_path = tmp_path / "repair3.json"
    flat_path = tmp_path / "flat.json"

    write_json(
        confirmation_path,
        {
            "stage_status": "stage_failed_exhausted",
            "results": [
                {
                    "sample_id": "s0",
                    "condition": "G_full_measured_geosym",
                    "correct": False,
                    "normalized_prediction": 0,
                    "reference": 3,
                    "raw_text": "minimum is 0",
                },
                {
                    "sample_id": "s1",
                    "condition": "G_full_measured_geosym",
                    "correct": True,
                },
            ],
        },
    )
    write_json(
        projection_path,
        {
            "stage_status": "projection_passed",
            "summary": {"G_full_measured_geosym": {"correct": 48}},
        },
    )
    write_json(
        repair_path,
        {
            "stage_status": "stage_failed_exhausted",
            "summary": {"correct": 5, "previous_stage4_correct": 5},
            "results": [
                {
                    "sample_id": "s0",
                    "condition": "G_full_measured_geosym",
                    "correct": False,
                    "normalized_prediction": 0,
                }
            ],
        },
    )
    write_json(
        flat_path,
        {
            "stage_status": "measured_flat_parseability_gap_classified",
            "failures": [{"sample_id": "s2", "failure_label": "extractor_gap_no_target_candidates"}],
        },
    )

    audit = build_audit(confirmation_path, projection_path, repair_path, flat_path, tmp_path / "out")

    assert audit["summary"]["fresh_g_full_wrong"] == 1
    assert audit["fresh_g_full_wrong_cases"][0]["failure_label"] == "m0_minimum_or_axis_scale_gap"
    assert audit["recommended_next_direction"] == "m0_axis_and_series_repair_before_more_50_sample_confirmation"
    assert "Do not run another 50-sample confirmation" in audit["stop_or_pivot_signal"]
    assert (tmp_path / "out" / "remaining_bottleneck_audit.json").exists()
