import json
from pathlib import Path

from scripts.finalize_plotqa_measured_geosym_evidence import build_audit


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_final_audit_records_projection_success_with_confirmation_gap(tmp_path: Path) -> None:
    confirmation_path = tmp_path / "confirmation.json"
    projection_path = tmp_path / "projection.json"
    repair1_path = tmp_path / "repair1.json"
    repair2_path = tmp_path / "repair2.json"
    output_dir = tmp_path / "audit"

    write_json(
        confirmation_path,
        {
            "stage": "stage5_50_sample_confirmation_after_repair",
            "stage_status": "stage_failed_exhausted",
            "actual_llm_http_attempts": 139,
            "actual_ocr_http_attempts": 0,
            "paid_calls_executed": 139,
            "cached_baseline_correct": {"E0_plotqa_direct": 43, "E1_plotqa_cot": 42},
            "summary": {
                "G_full_measured_geosym": {"completed": 50, "parseable": 50, "correct": 47},
                "Measured_flat_answer_baseline": {"completed": 46, "parseable": 46, "correct": 27},
            },
        },
    )
    write_json(
        projection_path,
        {
            "stage": "stage5_post_repair_projection_zero_cost",
            "stage_status": "projection_passed",
            "paid_calls_executed": 0,
            "cached_baseline_correct": {"E0_plotqa_direct": 43, "E1_plotqa_cot": 42},
            "summary": {
                "G_full_measured_geosym": {"completed": 50, "parseable": 50, "correct": 48},
                "Measured_flat_answer_baseline": {"completed": 46, "parseable": 46, "correct": 27},
            },
        },
    )
    write_json(repair1_path, {"stage": "stage5_repair_smoke", "stage_status": "stage_passed", "actual_llm_http_attempts": 5})
    write_json(repair2_path, {"stage": "stage5_repair_smoke", "stage_status": "stage_passed", "actual_llm_http_attempts": 6})

    audit = build_audit(confirmation_path, projection_path, [repair1_path, repair2_path], output_dir)

    assert audit["stage_status"] == "primary_confirmation_target_met_with_control_gap"
    assert audit["paid_calls_executed"] == 0
    assert audit["gate_assessment"]["confirmation_g_full_correct"] == 47
    assert audit["gate_assessment"]["projection_g_full_correct"] == 48
    assert audit["gate_assessment"]["fresh_50_confirmation_primary_g_full_target_met"] is True
    assert audit["gate_assessment"]["fresh_50_confirmation_stage_gate_met"] is False
    assert audit["gate_assessment"]["strongest_direct_baseline_correct"] == 43
    assert audit["gate_assessment"]["stage5_total_llm_attempts"] == 150
    assert audit["gate_assessment"]["stage5_total_within_314_cap"] is True
    assert [item["label"] for item in audit["blocking_items_for_paper_claim"]] == [
        "fresh_50_confirmation_not_stage_passed",
        "measured_flat_control_parseability_below_plan_gate",
    ]
    assert (output_dir / "final_evidence_audit.json").exists()
    assert "Fresh 50 confirmation G_full: 47/50" in (output_dir / "final_evidence_audit.md").read_text(encoding="utf-8")


def test_final_audit_blocks_when_paid_cap_exceeded(tmp_path: Path) -> None:
    confirmation_path = tmp_path / "confirmation.json"
    projection_path = tmp_path / "projection.json"
    repair_path = tmp_path / "repair.json"

    base_summary = {
        "G_full_measured_geosym": {"completed": 50, "parseable": 50, "correct": 48},
        "Measured_flat_answer_baseline": {"completed": 50, "parseable": 50, "correct": 30},
    }
    write_json(
        confirmation_path,
        {
            "stage_status": "stage_passed",
            "actual_llm_http_attempts": 251,
            "cached_baseline_correct": {"E0_plotqa_direct": 43, "E1_plotqa_cot": 42},
            "summary": base_summary,
        },
    )
    write_json(
        projection_path,
        {
            "stage_status": "projection_passed",
            "cached_baseline_correct": {"E0_plotqa_direct": 43, "E1_plotqa_cot": 42},
            "summary": base_summary,
        },
    )
    write_json(repair_path, {"actual_llm_http_attempts": 64})

    audit = build_audit(confirmation_path, projection_path, [repair_path], tmp_path / "audit")

    labels = {item["label"] for item in audit["blocking_items_for_paper_claim"]}
    assert "stage5_total_paid_cap_exceeded" in labels
    assert "stage5_confirmation_paid_cap_exceeded" in labels
    assert audit["gate_assessment"]["stage5_total_llm_attempts"] == 315


def test_final_audit_marks_gate_met_as_ready_for_paper_update(tmp_path: Path) -> None:
    confirmation_path = tmp_path / "confirmation.json"
    projection_path = tmp_path / "projection.json"
    repair_path = tmp_path / "repair.json"
    summary = {
        "G_full_measured_geosym": {"completed": 50, "parseable": 50, "correct": 48},
        "Measured_flat_answer_baseline": {"completed": 49, "parseable": 49, "correct": 26},
    }
    write_json(
        confirmation_path,
        {
            "stage_status": "stage_passed",
            "actual_llm_http_attempts": 141,
            "cached_baseline_correct": {"E0_plotqa_direct": 43, "E1_plotqa_cot": 42},
            "summary": summary,
        },
    )
    write_json(
        projection_path,
        {
            "stage_status": "projection_passed",
            "cached_baseline_correct": {"E0_plotqa_direct": 43, "E1_plotqa_cot": 42},
            "summary": summary,
        },
    )
    write_json(repair_path, {"actual_llm_http_attempts": 17})

    audit = build_audit(confirmation_path, projection_path, [repair_path], tmp_path / "audit")

    assert audit["stage_status"] == "fresh_50_confirmation_stage_gate_met"
    assert audit["blocking_items_for_paper_claim"] == []
    assert audit["next_required_action"] == "ready_for_paper_package_update_with_m1m2_caveat"
