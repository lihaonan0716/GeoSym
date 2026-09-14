import json
from pathlib import Path

from scripts.freeze_robustness_pilot_manifest import build_manifest


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def stress_job(idx: int) -> dict:
    return {
        "stress_id": f"stress_{idx:03d}",
        "source_sample_id": f"s{idx:02d}",
        "perturbation_type": "missing_required_fact" if idx < 6 else "spurious_fact",
        "seed": 20260830,
        "answer_independent_selection": True,
        "stress_schema": {"stress_hash": f"h{idx}"},
    }


def test_stress_paid_manifest_freezes_8_job_cap_and_no_paid_calls(tmp_path: Path) -> None:
    stress_path = tmp_path / "stress.json"
    write_json(stress_path, {"stage_status": "stress_benchmark_frozen", "jobs": [stress_job(i) for i in range(24)]})

    result = build_manifest(stress_path, tmp_path / "out")

    assert result["stage_status"] == "stress_paid_pilot_manifest_frozen"
    assert result["paid_calls_executed"] == 0
    assert result["sample_level_job_cap"] == 8
    assert result["llm_http_attempt_cap"] == 32
    assert result["ocr_http_attempt_cap"] == 0
    assert len(result["selected_roster"]) == 8
    assert result["go_gate"]["answer_independent_selection"] is True


def test_stress_paid_manifest_fails_closed_without_frozen_source(tmp_path: Path) -> None:
    stress_path = tmp_path / "stress.json"
    write_json(stress_path, {"stage_status": "stress_roster_insufficient", "jobs": [stress_job(i) for i in range(24)]})

    result = build_manifest(stress_path, tmp_path / "out")

    assert result["stage_status"] == "stress_paid_pilot_manifest_failed_closed"
    assert result["selected_roster"] == []
    assert result["next_required_action"] == "repair_stress_benchmark_before_paid_manifest"
