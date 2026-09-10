import json
from pathlib import Path

from scripts.analyze_plotqa_measured_flat_parseability import build_audit


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_flat_parseability_audit_classifies_extractor_gaps(tmp_path: Path) -> None:
    confirmation_path = tmp_path / "confirmation.json"
    records_path = tmp_path / "records.json"

    write_json(
        confirmation_path,
        {
            "stage_status": "stage_failed_exhausted",
            "results": [
                {
                    "sample_id": "s0",
                    "condition": "Measured_flat_answer_baseline",
                    "completed": False,
                    "parseable": False,
                    "raw_text": '{"raw_text":"Cannot determine from the measured facts.","normalized_prediction":"Cannot determine"}',
                    "error": "numeric answer has no parseable number",
                },
                {
                    "sample_id": "s1",
                    "condition": "Measured_flat_answer_baseline",
                    "completed": True,
                    "parseable": True,
                    "normalized_prediction": 3,
                },
            ],
        },
    )
    write_json(
        records_path,
        [
            {
                "sample_id": "s0",
                "target_candidates": [],
                "ruler": {"confidence": "medium"},
                "measured_points": [
                    {"x_value": 2001.0, "y_value": None},
                    {"x_value": 2002.0, "y_value": None},
                ],
                "evidence_gaps": [],
            }
        ],
    )

    audit = build_audit(confirmation_path, records_path, tmp_path / "out")

    assert audit["stage_status"] == "measured_flat_parseability_gap_classified"
    assert audit["summary"]["flat_rows"] == 2
    assert audit["summary"]["parseable"] == 1
    assert audit["summary"]["extractor_gap_failures"] == 1
    assert audit["failures"][0]["failure_label"] == "extractor_gap_no_target_candidates"
    assert audit["claim_boundary"] == "Do not use measured-flat under-parseability as evidence for a reliability-aware fusion gain."
    assert (tmp_path / "out" / "flat_parseability_audit.json").exists()


def test_flat_parseability_audit_classifies_blocking_gap_first(tmp_path: Path) -> None:
    confirmation_path = tmp_path / "confirmation.json"
    records_path = tmp_path / "records.json"
    write_json(
        confirmation_path,
        {
            "results": [
                {
                    "sample_id": "s0",
                    "condition": "Measured_flat_answer_baseline",
                    "completed": False,
                    "parseable": False,
                    "raw_text": "{}",
                }
            ]
        },
    )
    write_json(
        records_path,
        [
            {
                "sample_id": "s0",
                "target_candidates": [{"fact_id": "m0_0"}],
                "measured_points": [],
                "evidence_gaps": [{"reason": "missing_series", "severity": "blocking"}],
            }
        ],
    )

    audit = build_audit(confirmation_path, records_path, tmp_path / "out")

    assert audit["failures"][0]["failure_label"] == "extractor_gap_blocking_evidence_gap"
    assert audit["failures"][0]["gap_reasons"] == ["missing_series"]
