import json
from pathlib import Path

from scripts.analyze_m1m2_stress_paid_pilot import analyze


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def row(stress_id: str, condition: str, correct: bool, text: str = "", parseable: bool = True) -> dict:
    return {
        "stress_id": stress_id,
        "condition": condition,
        "correct": correct,
        "completed": parseable,
        "parseable": parseable,
        "insufficient_evidence_signaled": "insufficient" in text.lower(),
    }


def test_stress_analysis_can_support_mechanism_with_harm_reduction(tmp_path: Path) -> None:
    results = []
    for idx in range(8):
        sid = f"stress_{idx:03d}"
        results.append(row(sid, "G_full_guarded", False, "insufficient evidence", parseable=False))
        results.append(row(sid, "G_all_evidence", False))
        results.append(row(sid, "G_top_confidence", False))
        results.append(row(sid, "G_random_k", False))
    paid_path = tmp_path / "paid.json"
    write_json(
        paid_path,
        {
            "stage_status": "stress_paid_pilot_completed",
            "paid_calls_executed": 32,
            "actual_ocr_http_attempts": 0,
            "summary": {
                "G_full_guarded": {"completed": 0, "parseable": 0, "correct": 0, "insufficient_evidence": 8},
                "G_all_evidence": {"completed": 8, "parseable": 8, "correct": 0, "insufficient_evidence": 0},
                "G_top_confidence": {"completed": 8, "parseable": 8, "correct": 0, "insufficient_evidence": 0},
                "G_random_k": {"completed": 8, "parseable": 8, "correct": 0, "insufficient_evidence": 0},
            },
            "results": results,
        },
    )

    result = analyze(paid_path, tmp_path / "out")

    assert result["stage_status"] == "m1m2_mechanism_supported"
    assert result["gates"]["guarded_reduces_harm_vs_all_evidence"] is True
    assert result["gates"]["guarded_reduces_harm_vs_top_confidence"] is True


def test_stress_analysis_fails_closed_on_guarded_degradation(tmp_path: Path) -> None:
    results = []
    for idx in range(8):
        sid = f"stress_{idx:03d}"
        results.append(row(sid, "G_full_guarded", False))
        results.append(row(sid, "G_all_evidence", True))
        results.append(row(sid, "G_top_confidence", True))
        results.append(row(sid, "G_random_k", True))
    paid_path = tmp_path / "paid.json"
    write_json(
        paid_path,
        {
            "stage_status": "stress_paid_pilot_completed",
            "paid_calls_executed": 32,
            "actual_ocr_http_attempts": 0,
            "summary": {
                "G_full_guarded": {"completed": 8, "parseable": 8, "correct": 0, "insufficient_evidence": 0},
                "G_all_evidence": {"completed": 8, "parseable": 8, "correct": 8, "insufficient_evidence": 0},
                "G_top_confidence": {"completed": 8, "parseable": 8, "correct": 8, "insufficient_evidence": 0},
                "G_random_k": {"completed": 8, "parseable": 8, "correct": 8, "insufficient_evidence": 0},
            },
            "results": results,
        },
    )

    result = analyze(paid_path, tmp_path / "out")

    assert result["stage_status"] == "m1m2_mechanism_not_supported"
    assert result["guarded_degradation_count"] == 24
