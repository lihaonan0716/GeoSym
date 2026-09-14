import json
from pathlib import Path

from scripts.freeze_evidence_selection_pilot_manifest import build_manifest


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def schema_row(idx: int, policy: str) -> dict:
    return {
        "sample_id": f"s{idx:02d}",
        "question_hash": f"q{idx}",
        "selected_evidence_set": {
            "propagation_policy": policy,
            "selected_fact_ids": [f"f{idx}"],
            "suppressed_fact_ids": [],
            "surfaced_gap_ids": [],
        },
    }


def test_paid_pilot_manifest_freezes_exact_bounded_defaults(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    projection_path = tmp_path / "projection.json"
    rows = [schema_row(i, "use_with_caution" if i < 4 else "use_directly") for i in range(50)]
    write_json(schema_path, {"stage_status": "ready_for_zero_cost_projection", "outputs": rows})
    write_json(projection_path, {"stage_status": "plotqa_guardrail_preserved"})

    manifest = build_manifest(schema_path, projection_path, tmp_path / "out")

    assert manifest["stage_status"] == "paid_pilot_manifest_frozen"
    assert manifest["paid_calls_executed"] == 0
    assert manifest["sample_level_job_cap"] == 8
    assert manifest["conditions"] == ["G_full_guarded", "G_all_evidence", "G_top_confidence", "G_random_k"]
    assert manifest["llm_http_attempt_cap"] == 32
    assert manifest["ocr_http_attempt_cap"] == 0
    assert len(manifest["selected_roster"]) == 8
    assert manifest["go_gate"]["plotqa_guardrail_preserved"] is True
    assert manifest["next_required_action"] == "implement_executable_paid_pilot_runner"


def test_paid_pilot_manifest_fails_closed_without_projection_gate(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    projection_path = tmp_path / "projection.json"
    write_json(schema_path, {"stage_status": "ready_for_zero_cost_projection", "outputs": [schema_row(i, "use_directly") for i in range(50)]})
    write_json(projection_path, {"stage_status": "plotqa_guardrail_failed"})

    manifest = build_manifest(schema_path, projection_path, tmp_path / "out")

    assert manifest["stage_status"] == "paid_pilot_manifest_failed_closed"
    assert manifest["selected_roster"] == []
    assert manifest["go_gate"]["plotqa_guardrail_preserved"] is False
