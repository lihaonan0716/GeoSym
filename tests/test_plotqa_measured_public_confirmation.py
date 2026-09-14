import json
from pathlib import Path

from scripts.run_plotqa_measured_public_confirmation import build_plan, review_plan


def test_plotqa_measured_public_500_plan_freezes_caps(tmp_path: Path) -> None:
    samples = [
        {
            "sample_id": f"s{i}",
            "question": "What is the value?",
            "image_path": "image.png",
            "answer_type": "numeric",
            "normalized_answer": 1.0,
        }
        for i in range(500)
    ]
    source_plan = {
        "benchmark": "PlotQA",
        "selected_samples": samples,
        "frozen_before_paid_run": {
            "model_provenance": {"display_name": "GPT-5.5", "api_model_id": "gpt-5.5"},
            "reasoning_model": "gpt-5.5",
            "scorer_version": "test_scorer",
        },
    }
    records = [
        {"sample_id": sample["sample_id"], "schema_violations": [], "measured_points": [{"fact_id": "m0"}]}
        for sample in samples
    ]
    stage1 = {"stage_status": "stage_passed", "sample_count": 500}
    plan_path = tmp_path / "plotqa_plan.json"
    records_path = tmp_path / "records.json"
    stage1_path = tmp_path / "stage1.json"
    plan_path.write_text(json.dumps(source_plan), encoding="utf-8")
    records_path.write_text(json.dumps(records), encoding="utf-8")
    stage1_path.write_text(json.dumps(stage1), encoding="utf-8")

    plan = build_plan(plan_path, records_path, stage1_path, tmp_path)
    review = review_plan(plan)

    assert plan["paid_run_approved"] is True
    assert plan["sample_condition_job_cap"] == 1500
    assert plan["llm_http_attempt_cap"] == 1500
    assert plan["ocr_http_attempt_cap"] == 0
    assert plan["dry_run"]["paid_calls_executed"] == 0
    assert review["status"] == "no_blocking_findings"
