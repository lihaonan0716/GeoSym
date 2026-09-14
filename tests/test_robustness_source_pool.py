import json
from pathlib import Path

from scripts.audit_robustness_source_pool import build_audit


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def measured_record(idx: int, question: str) -> dict:
    return {
        "sample_id": f"s{idx:03d}",
        "question": question,
        "answer_type": "numeric",
        "reference": 1,
        "chart_geometry": {"chart_type": "bar"},
        "ruler": {"confidence": "high", "x_ticks": [{"value": 1}], "y_ticks": [{"value": 1}]},
        "series": [{"label": "A"}],
        "measured_points": [
            {
                "fact_id": f"m0_{idx}",
                "x_value": 1,
                "y_value": 1,
                "series_label": "A",
                "pixel": [1, 1],
                "confidence": "medium",
                "source": "cv_color_component",
            }
        ],
        "evidence_gaps": [],
        "schema_violations": [],
    }


def result_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        rows.append(
            {
                "sample_id": record["sample_id"],
                "condition": "G_full_measured_geosym",
                "completed": True,
                "parseable": True,
                "correct": True,
            }
        )
        rows.append(
            {
                "sample_id": record["sample_id"],
                "condition": "G_measured_raw",
                "completed": True,
                "parseable": True,
                "correct": True,
            }
        )
    return rows


def test_source_pool_feasibility_passes_128_with_balanced_operations(tmp_path: Path, monkeypatch) -> None:
    questions = (
        ["What is the value of A in 2012?"] * 11
        + ["How many values are greater than 5?"] * 11
        + ["What is the difference between A and B?"] * 11
        + ["What is the highest value?"] * 11
        + ["What is the difference between highest and lowest?"] * 7
        + ["What is the ratio of A to B?"] * 13
    )
    records = [measured_record(idx, question) for idx, question in enumerate(questions)]
    monkeypatch.setattr("scripts.audit_robustness_source_pool.deterministic_value", lambda record: 1)
    records_path = tmp_path / "records.json"
    result_path = tmp_path / "result.json"
    write_json(records_path, records)
    write_json(result_path, {"stage_status": "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim", "results": result_rows(records)})

    audit = build_audit(records_path, result_path, tmp_path / "out")

    assert audit["stage_status"] == "m1m2_stress_128_source_pool_feasible"
    assert audit["primary_128"]["passed"] is True
    assert audit["paid_calls_executed"] == 0


def test_source_pool_feasibility_fails_closed_when_operation_quota_missing(tmp_path: Path, monkeypatch) -> None:
    records = [measured_record(idx, "What is the value of A in 2012?") for idx in range(80)]
    monkeypatch.setattr("scripts.audit_robustness_source_pool.deterministic_value", lambda record: 1)
    records_path = tmp_path / "records.json"
    result_path = tmp_path / "result.json"
    write_json(records_path, records)
    write_json(result_path, {"stage_status": "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim", "results": result_rows(records)})

    audit = build_audit(records_path, result_path, tmp_path / "out")

    assert audit["stage_status"] == "m1m2_stress_source_pool_insufficient"
    assert audit["primary_128"]["passed"] is False
    assert "stress_96_fallback_source_pool_quota_failed" in audit["blocking_findings"]
