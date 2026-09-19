import json
from pathlib import Path

from scripts.build_evidence_selection_schema_outputs import build_schema_outputs


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def audit_row(sample_id: str, primary: str = "clean-pass-through") -> dict:
    return {
        "sample_id": sample_id,
        "question_hash": f"question-{sample_id}",
        "source_record_hash": f"record-{sample_id}",
        "primary_classification": primary,
        "classification_flags": {
            "clean_pass_through": primary == "clean-pass-through",
            "candidate_risk": primary == "candidate-risk",
            "evidence_gap": primary == "evidence-gap",
            "deterministic_solver_eligible": primary == "deterministic-solver-eligible",
            "unresolved": primary == "unresolved",
        },
        "risk_reasons": ["ruler_low_or_unknown_confidence"] if primary == "candidate-risk" else [],
    }


def measured_record(sample_id: str, *, gap: bool = False) -> dict:
    return {
        "sample_id": sample_id,
        "question": "What is the maximum value?",
        "answer_type": "numeric",
        "chart_geometry": {"chart_type": "bar"},
        "ruler": {"confidence": "high"},
        "measured_points": [
            {
                "fact_id": f"{sample_id}_good",
                "x_value": 1,
                "y_value": 5,
                "source": "cv_color_component",
                "confidence": "medium",
                "support": ["legend_color_match"],
                "uncertainty": {"y_abs": 0.1},
                "component_bbox": [0, 0, 10, 10],
                "component_area": 100,
                "series_label": "A",
            },
            {
                "fact_id": f"{sample_id}_sliver",
                "x_value": 1,
                "y_value": 0.01,
                "source": "cv_color_component",
                "confidence": "medium",
                "component_bbox": [0, 10, 10, 11],
                "component_area": 10,
                "series_label": "A",
            },
        ],
        "evidence_gaps": [{"reason": "not_detected", "key": "measured_value"}] if gap else [],
    }


def test_schema_outputs_preserve_lineage_and_do_not_modify_default(tmp_path: Path) -> None:
    records = [measured_record(f"s{i:02d}") for i in range(50)]
    audit = {"stage_status": "ready_for_schema_implementation", "rows": [audit_row(record["sample_id"]) for record in records]}
    records_path = tmp_path / "records.json"
    audit_path = tmp_path / "audit.json"
    write_json(records_path, records)
    write_json(audit_path, audit)

    result = build_schema_outputs(records_path, audit_path, tmp_path / "out")

    assert result["stage_status"] == "ready_for_zero_cost_projection"
    assert result["paid_calls_executed"] == 0
    assert result["default_path_modified"] is False
    assert result["summary"]["protected_default_output_changes"] == 0
    # These fixtures carry no OCR row, so the M0' chain cannot fit an axis and
    # M2' withholds the whole record rather than injecting unverifiable values.
    # Both facts per sample are therefore suppressed, not just the sliver.
    assert result["summary"]["suppressed_fact_count"] == 100
    first = result["outputs"][0]
    assert first["selected_evidence_set"]["selected_fact_ids"] == []
    assert first["selected_evidence_set"]["suppressed_fact_ids"] == ["s00_good", "s00_sliver"]
    assert first["decision_audit"]["protected_default_output_changed"] is False


def test_schema_outputs_surface_existing_and_generated_gaps(tmp_path: Path) -> None:
    records = [measured_record("gap", gap=True), measured_record("risk")]
    audit = {"rows": [audit_row("gap", "evidence-gap"), audit_row("risk", "candidate-risk")]}
    records_path = tmp_path / "records.json"
    audit_path = tmp_path / "audit.json"
    write_json(records_path, records)
    write_json(audit_path, audit)

    result = build_schema_outputs(records_path, audit_path, tmp_path / "out")
    rows = {row["sample_id"]: row for row in result["outputs"]}

    assert rows["gap"]["selected_evidence_set"]["surfaced_gap_ids"] == ["gap_000"]
    assert rows["gap"]["evidence_gaps"][0]["needed_for_question"] is True
    assert rows["risk"]["evidence_gaps"][0]["gap_id"] == "risk_000_axis_scale"
    assert rows["risk"]["selected_evidence_set"]["propagation_policy"] == "insufficient_evidence"
    assert result["stage_status"] == "schema_output_failed_closed"


def test_schema_outputs_fail_closed_on_invalid_selected_lineage(tmp_path: Path, monkeypatch) -> None:
    records = [measured_record("s0")]
    audit = {"rows": [audit_row("s0")]}
    records_path = tmp_path / "records.json"
    audit_path = tmp_path / "audit.json"
    write_json(records_path, records)
    write_json(audit_path, audit)

    def bad_measured_prompt_facts(record, locate_row, curated):
        return {"measured_facts": [{"fact_id": "missing"}], "evidence_gaps": []}

    monkeypatch.setattr("scripts.build_evidence_selection_schema_outputs.measured_prompt_facts", bad_measured_prompt_facts)

    result = build_schema_outputs(records_path, audit_path, tmp_path / "out")

    assert result["stage_status"] == "schema_output_failed_closed"
    assert result["next_required_action"] == "repair_schema_output_before_projection"
