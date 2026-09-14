import json
from pathlib import Path

from scripts.analyze_evidence_selection_selective_propagation_confirmation import analyze


CONDITIONS = [
    "G_selective_propagation",
    "G_candidate_unrouted",
    "G_selective_no_gaps",
    "G_all_evidence",
    "G_top_confidence",
    "G_random_k",
    "G_no_routing",
]

FAMILIES = [
    "axis_notation_or_unit_ambiguity",
    "contradictory_duplicate_fact",
    "missing_required_fact",
    "ruler_or_axis_scale_perturbation",
    "series_or_legend_ambiguity",
    "spurious_low_confidence_fact",
]


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def expected_for_family(family: str) -> str:
    if family == "spurious_low_confidence_fact":
        return "answerable"
    if family == "contradictory_duplicate_fact":
        return "contradiction"
    return "insufficient_evidence"


def result_row(stress_id: str, family: str, condition: str, success: bool, harm: bool, expected: str, over_abstain: bool = False) -> dict:
    answer_status = "answerable"
    if expected == "contradiction" and success:
        answer_status = "contradiction"
    elif expected == "insufficient_evidence" and success:
        answer_status = "insufficient_evidence"
    elif expected == "answerable" and over_abstain:
        answer_status = "insufficient_evidence"
    return {
        "stress_id": stress_id,
        "source_sample_id": stress_id.replace("stress", "sample"),
        "perturbation_family": family,
        "condition": condition,
        "expected_answerability": expected,
        "answer_type": "numeric",
        "reference": 1,
        "schema_valid": True,
        "answer_status": answer_status,
        "answerable": answer_status == "answerable",
        "prediction_parseable": answer_status == "answerable",
        "normalized_prediction": "1" if answer_status == "answerable" else None,
        "supporting_fact_ids": ["f0"] if answer_status == "answerable" else [],
        "risk_reason": "",
        "correct_answer": expected == "answerable" and success and answer_status == "answerable",
        "correct_abstention": expected != "answerable" and success and answer_status == expected,
        "task_success": success,
        "harmful_propagation": harm,
    }


def confirmation_payload(too_conservative: bool = False) -> dict:
    rows = []
    idx = 0
    for family in FAMILIES:
        expected = expected_for_family(family)
        for family_idx in range(8):
            stress_id = f"stress_{idx:03d}"
            for condition in CONDITIONS:
                success = True
                harm = False
                over_abstain = False
                if condition == "G_all_evidence" and expected != "answerable" and family_idx in {0, 1, 2, 3, 4, 5}:
                    success = False
                    harm = True
                if condition == "G_top_confidence" and expected != "answerable" and family_idx in {0, 1, 2}:
                    success = False
                    harm = True
                if condition == "G_selective_no_gaps" and expected != "answerable" and family_idx in {0, 1, 2}:
                    success = False
                    harm = True
                if condition == "G_no_routing" and expected != "answerable" and family_idx in {0, 1}:
                    success = False
                    harm = True
                if condition == "G_random_k" and expected != "answerable" and family_idx in {0, 1, 2, 3}:
                    success = False
                    harm = True
                if condition == "G_selective_propagation" and expected == "answerable" and too_conservative and family_idx in {0, 1}:
                    success = False
                    over_abstain = True
                rows.append(result_row(stress_id, family, condition, success, harm, expected, over_abstain))
            idx += 1
    return {
        "stage_status": "selective_confirmation_completed",
        "paid_calls_executed": 48 * len(CONDITIONS),
        "actual_ocr_http_attempts": 0,
        "input_mode": "evidence_only",
        "results": rows,
    }


def manifest_payload() -> dict:
    return {
        "stage_status": "selective_confirmation_manifest_frozen_pending_approval",
        "llm_http_attempt_cap": 48 * len(CONDITIONS),
        "ocr_http_attempt_cap": 0,
        "input_mode": "evidence_only",
    }


def zero_payload() -> dict:
    return {"stage_status": "ready_for_paid_manifest", "plotqa_guardrail": {"passed": True, "protected_output_regressions": 0}}


def test_confirmation_analysis_supports_when_all_frozen_gates_pass(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    manifest_path = tmp_path / "manifest.json"
    zero_path = tmp_path / "zero.json"
    write_json(result_path, confirmation_payload())
    write_json(manifest_path, manifest_payload())
    write_json(zero_path, zero_payload())

    result = analyze(result_path, manifest_path, zero_path, tmp_path / "out")

    assert result["stage_status"] == "m1m2_selective_confirmation_supported"
    assert result["gates"]["G3_answerable_preservation"]["selective_correct_answer_at_least_7_of_8"] is True
    assert result["answerability_breakdown"]["G_selective_propagation"]["answerable"]["correct_answer"] == 8


def test_confirmation_analysis_detects_too_conservative_selective_arm(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    manifest_path = tmp_path / "manifest.json"
    zero_path = tmp_path / "zero.json"
    write_json(result_path, confirmation_payload(too_conservative=True))
    write_json(manifest_path, manifest_payload())
    write_json(zero_path, zero_payload())

    result = analyze(result_path, manifest_path, zero_path, tmp_path / "out")

    assert result["stage_status"] == "m1m2_selective_confirmation_too_conservative"
    assert result["gates"]["G3_answerable_preservation"]["selective_over_abstention_at_most_1_of_8"] is False
