import json
from pathlib import Path

from scripts.audit_m1m2_plotqa_guardrail import build_audit


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def result_row(sample_id: str, condition: str, correct: bool = True) -> dict:
    return {
        "sample_id": sample_id,
        "condition": condition,
        "completed": True,
        "parseable": True,
        "correct": correct,
        "normalized_prediction": 1,
    }


def test_guardrail_audit_preserves_frozen_48_of_50_gate(tmp_path: Path) -> None:
    records = []
    protected_results = []
    diagnostic_results = []
    for idx in range(50):
        sample_id = f"s{idx:02d}"
        records.append(
            {
                "sample_id": sample_id,
                "question": "What is the difference between two consecutive major ticks on the Y-axis ?",
                "answer_type": "numeric",
                "ruler": {
                    "confidence": "high",
                    "y_ticks": [{"value": 0}, {"value": 10}],
                },
                "measured_points": [{"fact_id": f"m0_{idx}", "confidence": "medium"}],
                "evidence_gaps": [],
                "schema_violations": [],
                "bottleneck_labels": [],
            }
        )
        protected_results.append(result_row(sample_id, "G_full_measured_geosym", correct=idx >= 2))
        protected_results.append(result_row(sample_id, "G_measured_raw", correct=idx >= 8))
        diagnostic_results.append(result_row(sample_id, "E0_plotqa_direct", correct=idx < 2 or idx >= 7))
        diagnostic_results.append(result_row(sample_id, "E1_plotqa_cot", correct=idx >= 8))

    records_path = tmp_path / "records.json"
    protected_path = tmp_path / "protected.json"
    diagnostic_path = tmp_path / "diagnostic.json"
    write_json(records_path, records)
    write_json(
        protected_path,
        {
            "stage_status": "stage_passed",
            "summary": {
                "G_full_measured_geosym": {"completed": 50, "parseable": 50, "correct": 48},
                "G_measured_raw": {"completed": 50, "parseable": 50, "correct": 42},
            },
            "results": protected_results,
        },
    )
    write_json(diagnostic_path, {"results": diagnostic_results})

    audit = build_audit(records_path, protected_path, diagnostic_path, tmp_path / "out")

    assert audit["stage_status"] == "ready_for_schema_implementation"
    assert audit["paid_calls_executed"] == 0
    assert audit["guardrail"]["passed"] is True
    assert audit["classification_summary"]["manual_labels_used"] == 0
    assert audit["classification_summary"]["direct_baseline_degradations"] == 2
    assert len(audit["rows"]) == 50
    assert (tmp_path / "out" / "plotqa_guardrail_audit.json").exists()


def test_guardrail_audit_prioritizes_gaps_and_candidate_risks(tmp_path: Path) -> None:
    records = [
        {
            "sample_id": "gap",
            "question": "What is the maximum value?",
            "answer_type": "numeric",
            "ruler": {"confidence": "high"},
            "measured_points": [],
            "evidence_gaps": [{"reason": "not_detected"}],
            "schema_violations": [],
            "bottleneck_labels": [],
        },
        {
            "sample_id": "risk",
            "question": "What is the maximum value?",
            "answer_type": "numeric",
            "ruler": {"confidence": "low"},
            "measured_points": [{"fact_id": "m0_1", "confidence": "medium"}],
            "evidence_gaps": [],
            "schema_violations": [],
            "bottleneck_labels": [{"label": "m0_measurement_out_of_tolerance"}],
        },
    ]
    protected_results = [
        result_row("gap", "G_full_measured_geosym", correct=True),
        result_row("gap", "G_measured_raw", correct=False),
        result_row("risk", "G_full_measured_geosym", correct=False),
        result_row("risk", "G_measured_raw", correct=True),
    ]
    records_path = tmp_path / "records.json"
    protected_path = tmp_path / "protected.json"
    write_json(records_path, records)
    write_json(
        protected_path,
        {
            "summary": {"G_full_measured_geosym": {"completed": 50, "parseable": 50, "correct": 48}},
            "results": protected_results,
        },
    )

    audit = build_audit(records_path, protected_path, None, tmp_path / "out")
    rows = {row["sample_id"]: row for row in audit["rows"]}

    assert rows["gap"]["primary_classification"] == "evidence-gap"
    assert rows["gap"]["classification_flags"]["evidence_gap"] is True
    assert rows["risk"]["primary_classification"] == "candidate-risk"
    assert "curation_degraded_raw_correct_answer" in rows["risk"]["risk_reasons"]
    assert rows["risk"]["direct_baseline"]["missing_reason"] == "direct_baseline_row_unavailable"


def test_guardrail_audit_fails_closed_when_protected_gate_not_met(tmp_path: Path) -> None:
    records = [
        {
            "sample_id": "s0",
            "question": "What is the value?",
            "answer_type": "numeric",
            "ruler": {"confidence": "high"},
            "measured_points": [{"fact_id": "m0_0", "confidence": "medium"}],
            "evidence_gaps": [],
            "schema_violations": [],
            "bottleneck_labels": [],
        }
    ]
    records_path = tmp_path / "records.json"
    protected_path = tmp_path / "protected.json"
    write_json(records_path, records)
    write_json(
        protected_path,
        {
            "summary": {"G_full_measured_geosym": {"completed": 50, "parseable": 50, "correct": 47}},
            "results": [result_row("s0", "G_full_measured_geosym", correct=True)],
        },
    )

    audit = build_audit(records_path, protected_path, None, tmp_path / "out")

    assert audit["stage_status"] == "plotqa_guardrail_audit_failed"
    assert audit["next_required_action"] == "repair_or_restore_plotqa_guardrail_before_m1m2_changes"
