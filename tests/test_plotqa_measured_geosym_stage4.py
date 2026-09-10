import json

from scripts.run_plotqa_measured_geosym_stage4 import build_plan, review_plan


def test_stage4_plan_requires_passed_stage3_and_freezes_cap(tmp_path) -> None:
    diagnostic_plan = {
        "selected_samples": [{"sample_id": f"s{i}", "question": "q", "image_path": "image.png"} for i in range(50)],
        "frozen_before_paid_run": {"model_provenance": {"display_name": "model"}, "reasoning_model": "model"},
    }
    diagnostic_result = {
        "results": [
            {"sample_id": f"s{i}", "condition": condition, "correct": True}
            for i in range(50)
            for condition in ("E0_plotqa_direct", "E1_plotqa_cot")
        ]
    }
    measured_records = [{"sample_id": f"s{i}", "schema_violations": [], "measured_points": [{"fact_id": "m0"}]} for i in range(50)]
    stage3_result = {"stage_status": "stage_passed"}
    diagnostic_plan_path = tmp_path / "plan.json"
    diagnostic_result_path = tmp_path / "result.json"
    measured_records_path = tmp_path / "records.json"
    stage3_result_path = tmp_path / "stage3.json"
    diagnostic_plan_path.write_text(json.dumps(diagnostic_plan), encoding="utf-8")
    diagnostic_result_path.write_text(json.dumps(diagnostic_result), encoding="utf-8")
    measured_records_path.write_text(json.dumps(measured_records), encoding="utf-8")
    stage3_result_path.write_text(json.dumps(stage3_result), encoding="utf-8")

    plan = build_plan(diagnostic_plan_path, diagnostic_result_path, measured_records_path, stage3_result_path, tmp_path)
    review = review_plan(plan)

    assert plan["paid_run_approved"] is True
    assert plan["llm_http_attempt_cap"] == 150
    assert plan["ocr_http_attempt_cap"] == 0
    assert len(plan["selected_ids"]) == 50
    assert review["status"] == "no_blocking_findings"


def test_stage4_runner_can_freeze_stage5_confirmation_source(tmp_path) -> None:
    diagnostic_plan = {
        "selected_samples": [{"sample_id": f"s{i}", "question": "q", "image_path": "image.png"} for i in range(50)],
        "frozen_before_paid_run": {"model_provenance": {"display_name": "model"}, "reasoning_model": "model"},
    }
    diagnostic_result = {
        "results": [
            {"sample_id": f"s{i}", "condition": condition, "correct": True}
            for i in range(50)
            for condition in ("E0_plotqa_direct", "E1_plotqa_cot")
        ]
    }
    measured_records = [{"sample_id": f"s{i}", "schema_violations": [], "measured_points": [{"fact_id": "m0"}]} for i in range(50)]
    stage3_result = {"stage_status": "stage_passed"}
    stage5_repair = {"stage_status": "stage_passed", "summary": {"correct": 4}}
    diagnostic_plan_path = tmp_path / "plan.json"
    diagnostic_result_path = tmp_path / "result.json"
    measured_records_path = tmp_path / "records.json"
    stage3_result_path = tmp_path / "stage3.json"
    stage5_repair_path = tmp_path / "stage5.json"
    diagnostic_plan_path.write_text(json.dumps(diagnostic_plan), encoding="utf-8")
    diagnostic_result_path.write_text(json.dumps(diagnostic_result), encoding="utf-8")
    measured_records_path.write_text(json.dumps(measured_records), encoding="utf-8")
    stage3_result_path.write_text(json.dumps(stage3_result), encoding="utf-8")
    stage5_repair_path.write_text(json.dumps(stage5_repair), encoding="utf-8")

    plan = build_plan(diagnostic_plan_path, diagnostic_result_path, measured_records_path, stage3_result_path, tmp_path, stage5_repair_path)
    review = review_plan(plan)

    assert plan["stage"] == "stage5_50_sample_confirmation_after_repair"
    assert plan["source_artifacts"]["stage5_repair_smoke"]["stage_status"] == "stage_passed"
    assert review["status"] == "no_blocking_findings"
