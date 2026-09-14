import json

from scripts.run_plotqa_repair_smoke import build_plan, review_plan


def test_stage5_repair_smoke_freezes_wrong_and_degraded_samples(tmp_path) -> None:
    diagnostic_plan = {
        "selected_samples": [{"sample_id": f"s{i}", "question": "q", "image_path": "image.png"} for i in range(10)],
        "frozen_before_paid_run": {"model_provenance": {"display_name": "model"}, "reasoning_model": "model"},
    }
    diagnostic_result = {
        "results": [
            {"sample_id": "s0", "condition": "E0_plotqa_direct", "correct": True},
            {"sample_id": "s1", "condition": "E0_plotqa_direct", "correct": False},
        ]
    }
    measured_records = [{"sample_id": f"s{i}", "schema_violations": [], "measured_points": [{"fact_id": "m0"}]} for i in range(10)]
    stage4_result = {
        "stage_status": "stage_failed_exhausted",
        "results": [
            {"sample_id": "s0", "condition": "G_full_measured_geosym", "correct": False},
            {"sample_id": "s1", "condition": "G_full_measured_geosym", "correct": False},
            *[
                {"sample_id": f"s{i}", "condition": "G_full_measured_geosym", "correct": True}
                for i in range(2, 10)
            ],
        ],
    }
    diagnostic_plan_path = tmp_path / "plan.json"
    diagnostic_result_path = tmp_path / "result.json"
    measured_records_path = tmp_path / "records.json"
    stage4_result_path = tmp_path / "stage4.json"
    diagnostic_plan_path.write_text(json.dumps(diagnostic_plan), encoding="utf-8")
    diagnostic_result_path.write_text(json.dumps(diagnostic_result), encoding="utf-8")
    measured_records_path.write_text(json.dumps(measured_records), encoding="utf-8")
    stage4_result_path.write_text(json.dumps(stage4_result), encoding="utf-8")

    plan = build_plan(
        diagnostic_plan_path,
        diagnostic_result_path,
        measured_records_path,
        stage4_result_path,
        tmp_path,
        repair_iteration=3,
        dominant_bottleneck="m2_bar_sliver_artifact_filter",
    )
    review = review_plan(plan)

    assert plan["selected_ids"][:2] == ["s0", "s1"]
    assert len(plan["selected_ids"]) == 8
    assert plan["repair_iteration"] == 3
    assert plan["dominant_bottleneck"] == "m2_bar_sliver_artifact_filter"
    assert plan["llm_http_attempt_cap"] == 8
    assert review["status"] == "no_blocking_findings"
