import json
from pathlib import Path

from scripts.prepare_public_benchmark_pilot import canonical_json_bytes, sha256_hex_bytes
from scripts.rescore_plotqa_measured_flat_deterministic import rescore


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_flat_deterministic_rescore_only_fills_unparseable_rows(tmp_path: Path) -> None:
    plan_path = tmp_path / "stage_plan.json"
    diagnostic_plan_path = tmp_path / "diagnostic_plan.json"
    confirmation_path = tmp_path / "confirmation.json"
    records_path = tmp_path / "records.json"
    output_dir = tmp_path / "out"
    sample = {
        "sample_id": "s0",
        "question": "In how many years is the value greater than 0.2?",
        "answer_type": "numeric",
        "normalized_answer": 2.0,
    }
    stage_plan = {
        "stage": "stage5_50_sample_confirmation_after_repair",
        "selected_ids": ["s0"],
        "selected_samples": [sample],
        "conditions": ["Measured_flat_answer_baseline", "G_full_measured_geosym"],
        "llm_http_attempt_cap": 10,
        "plan_sha256": "",
        "interpretation_limits": [],
        "source_artifacts": {"diagnostic_result": {"path": str(tmp_path / "diagnostic_result.json")}},
    }
    stage_plan["plan_sha256"] = sha256_hex_bytes(canonical_json_bytes({k: v for k, v in stage_plan.items() if k != "plan_sha256"}))
    write_json(plan_path, stage_plan)
    write_json(diagnostic_plan_path, {"selected_samples": [sample]})
    write_json(tmp_path / "diagnostic_result.json", {"results": [{"sample_id": "s0", "condition": "E0_plotqa_direct", "correct": False}]})
    write_json(
        records_path,
        [
            {
                "sample_id": "s0",
                "ruler": {"confidence": "medium"},
                "measured_points": [{"y_value": 0.1}, {"y_value": 0.3}, {"y_value": 0.4}],
            }
        ],
    )
    write_json(
        confirmation_path,
        {
            "plan_path": str(plan_path),
            "actual_llm_http_attempts": 1,
            "results": [
                {
                    "sample_id": "s0",
                    "condition": "Measured_flat_answer_baseline",
                    "answer_type": "numeric",
                    "reference": 2.0,
                    "completed": False,
                    "parseable": False,
                    "normalized_prediction": None,
                },
                {
                    "sample_id": "s0",
                    "condition": "G_full_measured_geosym",
                    "answer_type": "numeric",
                    "reference": 2.0,
                    "completed": True,
                    "parseable": True,
                    "normalized_prediction": 99.0,
                    "correct": False,
                },
            ],
        },
    )

    result = rescore(confirmation_path, diagnostic_plan_path, records_path, output_dir)

    rows = {(row["sample_id"], row["condition"]): row for row in result["results"]}
    assert rows[("s0", "Measured_flat_answer_baseline")]["normalized_prediction"] == 2.0
    assert rows[("s0", "Measured_flat_answer_baseline")]["deterministic_measured_flat_rescore"] is True
    assert rows[("s0", "G_full_measured_geosym")]["normalized_prediction"] == 99.0
    assert result["rescore_paid_calls_executed"] == 0
