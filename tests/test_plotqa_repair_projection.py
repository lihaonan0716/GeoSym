import json

from scripts.analyze_plotqa_repair_projection import analyze


def test_stage5_projection_replaces_only_repair_full_rows(tmp_path) -> None:
    confirmation = {
        "stage_status": "stage_failed_exhausted",
        "cached_baseline_correct": {"E0_plotqa_direct": 1, "E1_plotqa_cot": 1},
        "results": [],
    }
    for idx in range(50):
        sample_id = f"s{idx}"
        confirmation["results"].extend(
            [
                {
                    "sample_id": sample_id,
                    "condition": "G_full_measured_geosym",
                    "completed": True,
                    "parseable": True,
                    "correct": idx != 0,
                    "normalized_prediction": idx,
                },
                {
                    "sample_id": sample_id,
                    "condition": "G_measured_raw",
                    "completed": True,
                    "parseable": True,
                    "correct": False,
                    "normalized_prediction": idx,
                },
                {
                    "sample_id": sample_id,
                    "condition": "Measured_flat_answer_baseline",
                    "completed": idx >= 4,
                    "parseable": idx >= 4,
                    "correct": False,
                    "normalized_prediction": idx if idx >= 4 else None,
                },
            ]
        )
    repair = {
        "stage_status": "stage_passed",
        "results": [
            {"sample_id": "s0", "condition": "G_full_measured_geosym", "completed": True, "parseable": True, "correct": True, "normalized_prediction": 3}
        ],
    }
    confirmation_path = tmp_path / "confirmation.json"
    repair_path = tmp_path / "repair.json"
    confirmation_path.write_text(json.dumps(confirmation), encoding="utf-8")
    repair_path.write_text(json.dumps(repair), encoding="utf-8")

    result = analyze(confirmation_path, repair_path, tmp_path / "out")

    projected = json.loads((tmp_path / "out" / "projection_result.json").read_text(encoding="utf-8"))
    assert result["paid_calls_executed"] == 0
    assert projected["replacements"] == [
            {
                "sample_id": "s0",
                "previous_prediction": 0,
                "previous_correct": False,
                "repair_prediction": 3,
                "repair_correct": True,
        }
    ]
    assert projected["summary"]["G_measured_raw"]["correct"] == 0
    assert projected["summary"]["G_full_measured_geosym"]["correct"] == 50
    assert projected["stage_status"] == "projection_passed"
    assert projected["go_gate"]["measured_flat_completed_parseable_at_least_48_of_50_projected"] is False
    assert projected["nonblocking_caveats"][0]["label"] == "measured_flat_control_parseability_below_stage4_operational_gate"
