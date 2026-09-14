import json
from pathlib import Path

from scripts.analyze_evidence_selection_pilot import analyze


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def row(sample_id: str, condition: str, correct: bool) -> dict:
    return {"sample_id": sample_id, "condition": condition, "completed": True, "parseable": True, "correct": correct}


def test_paid_pilot_analysis_rejects_identical_correctness(tmp_path: Path) -> None:
    results = []
    for idx in range(8):
        correct = idx >= 3
        for condition in ("G_full_guarded", "G_all_evidence", "G_top_confidence", "G_random_k"):
            results.append(row(f"s{idx}", condition, correct))
    paid_path = tmp_path / "paid.json"
    write_json(
        paid_path,
        {
            "stage_status": "paid_pilot_completed",
            "paid_calls_executed": 32,
            "actual_ocr_http_attempts": 0,
            "summary": {
                condition: {"completed": 8, "parseable": 8, "correct": 5}
                for condition in ("G_full_guarded", "G_all_evidence", "G_top_confidence", "G_random_k")
            },
            "results": results,
        },
    )

    analysis = analyze(paid_path, tmp_path / "out")

    assert analysis["stage_status"] == "m1m2_pilot_no_positive_mechanism_signal"
    assert analysis["all_conditions_identical_correctness_by_sample"] is True
    assert analysis["guarded_rescue_count"] == 0
    assert analysis["gates"]["guarded_beats_all_evidence"] is False


def test_paid_pilot_analysis_accepts_guarded_pairwise_wins(tmp_path: Path) -> None:
    results = []
    for condition in ("G_full_guarded", "G_all_evidence", "G_top_confidence", "G_random_k"):
        results.append(row("s0", condition, condition == "G_full_guarded"))
    for idx in range(1, 8):
        for condition in ("G_full_guarded", "G_all_evidence", "G_top_confidence", "G_random_k"):
            results.append(row(f"s{idx}", condition, True))
    paid_path = tmp_path / "paid.json"
    write_json(
        paid_path,
        {
            "stage_status": "paid_pilot_completed",
            "paid_calls_executed": 32,
            "actual_ocr_http_attempts": 0,
            "summary": {
                "G_full_guarded": {"completed": 8, "parseable": 8, "correct": 8},
                "G_all_evidence": {"completed": 8, "parseable": 8, "correct": 7},
                "G_top_confidence": {"completed": 8, "parseable": 8, "correct": 7},
                "G_random_k": {"completed": 8, "parseable": 8, "correct": 7},
            },
            "results": results,
        },
    )

    analysis = analyze(paid_path, tmp_path / "out")

    assert analysis["stage_status"] == "m1m2_pilot_mechanism_supported"
    assert analysis["guarded_rescue_count"] == 3
