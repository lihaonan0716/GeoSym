import json
from pathlib import Path

from scripts.rescore_plotqa_signed_difference_normalization import rescore


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_signed_difference_rescore_changes_only_matching_full_rows(tmp_path: Path) -> None:
    stage_plan = tmp_path / "stage_plan.json"
    confirmation = tmp_path / "confirmation.json"
    diagnostic_plan = tmp_path / "diagnostic_plan.json"
    diagnostic_result = tmp_path / "diagnostic_result.json"
    records = tmp_path / "records.json"

    write_json(
        stage_plan,
        {
            "stage": "stage5_50_sample_confirmation_after_repair",
            "conditions": ["G_full_measured_geosym", "G_measured_raw", "Measured_flat_answer_baseline"],
            "cached_conditions_reused": [],
            "selected_samples": [{"sample_id": "s0"}, {"sample_id": "s1"}],
            "selected_ids": ["s0", "s1"],
            "llm_http_attempt_cap": 150,
            "plan_sha256": "stage-plan",
            "source_artifacts": {"diagnostic_result": {"path": str(diagnostic_result)}},
            "interpretation_limits": [],
        },
    )
    write_json(diagnostic_result, {"results": []})
    write_json(
        confirmation,
        {
            "plan_path": str(stage_plan),
            "actual_llm_http_attempts": 2,
            "results": [
                {
                    "sample_id": "s0",
                    "condition": "G_full_measured_geosym",
                    "answer_type": "numeric",
                    "reference": -17_870_000.0,
                    "completed": True,
                    "parseable": True,
                    "normalized_prediction": 17_900_000.0,
                    "raw_text": (
                        '{"raw_text":"Rwanda in 2001 is about 19.1 million and Chile in 2003 is about 1.2 million, '
                        'so the difference is about 17.9 million.","normalized_prediction":"17900000"}'
                    ),
                    "correct": False,
                    "numeric_error": 35_770_000.0,
                    "tolerance": 893_500.0,
                },
                {
                    "sample_id": "s1",
                    "condition": "G_measured_raw",
                    "answer_type": "numeric",
                    "reference": 1.0,
                    "completed": True,
                    "parseable": True,
                    "normalized_prediction": 1.0,
                    "raw_text": "{}",
                    "correct": True,
                },
                {
                    "sample_id": "s1",
                    "condition": "Measured_flat_answer_baseline",
                    "answer_type": "numeric",
                    "reference": 1.0,
                    "completed": True,
                    "parseable": True,
                    "normalized_prediction": 1.0,
                    "raw_text": "{}",
                    "correct": True,
                },
            ],
        },
    )
    write_json(
        diagnostic_plan,
        {
            "selected_samples": [
                {
                    "sample_id": "s0",
                    "answer_type": "numeric",
                    "question": "What is the difference between the net bilateral aid flow in Chile in 2003 and the net bilateral aid flow in Rwanda in 2001?",
                    "normalized_answer": -17_870_000.0,
                },
                {"sample_id": "s1", "answer_type": "numeric", "question": "q", "normalized_answer": 1.0},
            ]
        },
    )
    write_json(records, [{"sample_id": "s0", "measured_points": [{"series_label": "Chile"}, {"series_label": "Rwanda"}]}])

    result = rescore(confirmation, diagnostic_plan, records, tmp_path / "out")

    assert result["rescore_paid_calls_executed"] == 0
    assert result["normalization_changes"] == [
        {
            "sample_id": "s0",
            "previous_prediction": 17_900_000.0,
            "new_prediction": -17_900_000.0,
            "previous_correct": False,
            "new_correct": True,
        }
    ]
    full = next(row for row in result["results"] if row["condition"] == "G_full_measured_geosym")
    raw = next(row for row in result["results"] if row["condition"] == "G_measured_raw")
    assert full["normalized_prediction"] == -17_900_000.0
    assert raw["normalized_prediction"] == 1.0
    assert (tmp_path / "out" / "rescore_result.json").exists()
