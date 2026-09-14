import json
from pathlib import Path

from scripts.analyze_evidence_selection_selective_propagation_pilot import analyze


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def row(stress_id: str, condition: str, success: bool, harm: bool, abstain: bool, expected: str = "insufficient_evidence") -> dict:
    return {
        "stress_id": stress_id,
        "condition": condition,
        "expected_answerability": expected,
        "schema_valid": True,
        "answer_status": "insufficient_evidence" if abstain else "answerable",
        "answerable": not abstain,
        "prediction_parseable": not abstain,
        "normalized_prediction": None if abstain else "1",
        "supporting_fact_ids": [] if abstain else ["f0"],
        "risk_reason": "",
        "task_success": success,
        "harmful_propagation": harm,
        "correct_abstention": abstain and success,
        "correct_answer": (not abstain) and success,
    }


def paid_payload(supported: bool = True) -> dict:
    conditions = [
        "G_selective_propagation",
        "G_selective_no_gaps",
        "G_all_evidence",
        "G_top_confidence",
        "G_random_k",
        "G_no_routing",
    ]
    rows = []
    for idx in range(12):
        sid = f"s{idx:02d}"
        expected = "insufficient_evidence" if idx < 10 else "answerable"
        rows.append(row(sid, "G_selective_propagation", True, False, idx < 10, expected))
        rows.append(row(sid, "G_all_evidence", idx >= 3, idx < 3, idx >= 3 and idx < 10, expected))
        rows.append(row(sid, "G_top_confidence", idx >= 1, idx == 0, idx >= 1 and idx < 10, expected))
        rows.append(row(sid, "G_random_k", idx >= 2, idx < 2, idx >= 2 and idx < 10, expected))
        rows.append(row(sid, "G_selective_no_gaps", idx >= 2, idx < 2, idx >= 2 and idx < 10, expected))
        rows.append(row(sid, "G_no_routing", idx >= 1, idx == 0, idx >= 1 and idx < 10, expected))
    if not supported:
        rows[0]["task_success"] = False
        rows[0]["harmful_propagation"] = True
    summary = {}
    for condition in conditions:
        subset = [item for item in rows if item["condition"] == condition]
        summary[condition] = {
            "schema_valid": sum(1 for item in subset if item["schema_valid"]),
            "answerable": sum(1 for item in subset if item["answerable"]),
            "prediction_parseable": sum(1 for item in subset if item["prediction_parseable"]),
            "correct_answer": sum(1 for item in subset if item["correct_answer"]),
            "correct_abstention": sum(1 for item in subset if item["correct_abstention"]),
            "task_success": sum(1 for item in subset if item["task_success"]),
            "harmful_propagation": sum(1 for item in subset if item["harmful_propagation"]),
            "protocol_violation": 0,
        }
    return {
        "stage_status": "selective_paid_pilot_completed",
        "paid_calls_executed": 72,
        "actual_ocr_http_attempts": 0,
        "input_mode": "evidence_only",
        "summary": summary,
        "results": rows,
    }


def test_selective_analysis_supports_when_all_gates_pass(tmp_path: Path) -> None:
    paid_path = tmp_path / "paid.json"
    zero_path = tmp_path / "zero.json"
    manifest_path = tmp_path / "manifest.json"
    write_json(paid_path, paid_payload())
    write_json(zero_path, {"stage_status": "ready_for_paid_manifest", "plotqa_guardrail": {"passed": True, "protected_output_regressions": 0}})
    write_json(manifest_path, {"stage_status": "selective_paid_pilot_manifest_frozen", "llm_http_attempt_cap": 72, "ocr_http_attempt_cap": 0, "input_mode": "evidence_only", "safe_default": "dry_run_no_paid_calls", "expansion_rule": "No automatic expansion"})

    result = analyze(paid_path, zero_path, manifest_path, tmp_path / "out")

    assert result["stage_status"] == "m1m2_selective_propagation_supported"
    assert result["gates"]["G2_mechanism_signal"]["selective_reduces_harm_vs_all_evidence"] is True
    assert result["answerability_breakdown"]["G_selective_propagation"]["answerable"]["correct_answer"] == 2
    assert "over_abstention_limit_required" in result["confirmation_additional_gates"]


def test_selective_analysis_fails_when_parser_integrity_fails(tmp_path: Path) -> None:
    payload = paid_payload()
    payload["results"][0]["answer_status"] = "insufficient_evidence"
    payload["results"][0]["normalized_prediction"] = "2008"
    paid_path = tmp_path / "paid.json"
    zero_path = tmp_path / "zero.json"
    manifest_path = tmp_path / "manifest.json"
    write_json(paid_path, payload)
    write_json(zero_path, {"stage_status": "ready_for_paid_manifest", "plotqa_guardrail": {"passed": True, "protected_output_regressions": 0}})
    write_json(manifest_path, {"stage_status": "selective_paid_pilot_manifest_frozen", "llm_http_attempt_cap": 72, "ocr_http_attempt_cap": 0, "input_mode": "evidence_only", "safe_default": "dry_run_no_paid_calls", "expansion_rule": "No automatic expansion"})

    result = analyze(paid_path, zero_path, manifest_path, tmp_path / "out")

    assert result["stage_status"] == "m1m2_selective_propagation_not_supported"
    assert result["gates"]["G1_parser_integrity"]["non_answerable_predictions_null"] is False
