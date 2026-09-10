import json
from pathlib import Path

from scripts.project_m1m2_guarded_zero_cost import build_projection


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def protected_row(sample_id: str, correct: bool) -> dict:
    return {
        "sample_id": sample_id,
        "condition": "G_full_measured_geosym",
        "completed": True,
        "parseable": True,
        "correct": correct,
        "normalized_prediction": sample_id,
    }


def schema_output(sample_id: str, policy: str = "use_directly") -> dict:
    return {
        "sample_id": sample_id,
        "selected_evidence_set": {
            "propagation_policy": policy,
            "selected_fact_ids": [f"{sample_id}_fact"],
            "suppressed_fact_ids": [],
            "surfaced_gap_ids": [],
        },
    }


def test_guarded_projection_preserves_48_of_50_and_zero_regressions(tmp_path: Path) -> None:
    protected_rows = [protected_row(f"s{i:02d}", correct=i >= 2) for i in range(50)]
    schema_rows = [schema_output(f"s{i:02d}", "use_directly" if i % 2 == 0 else "solve_deterministically") for i in range(50)]
    protected_path = tmp_path / "protected.json"
    schema_path = tmp_path / "schema.json"
    write_json(protected_path, {"stage_status": "stage_passed", "results": protected_rows})
    write_json(schema_path, {"stage_status": "ready_for_zero_cost_projection", "condition": "G_full_guarded_schema_only", "outputs": schema_rows})

    projection = build_projection(protected_path, schema_path, tmp_path / "out")

    assert projection["stage_status"] == "plotqa_guardrail_preserved"
    assert projection["paid_calls_executed"] == 0
    assert projection["summary"]["G_full_guarded"]["correct"] == 48
    assert projection["guardrail"]["protected_output_regressions"] == 0
    assert projection["results"][0]["condition"] == "G_full_guarded"
    assert projection["results"][0]["protected_default_output_changed"] is False


def test_guarded_projection_fails_closed_when_schema_not_ready(tmp_path: Path) -> None:
    protected_rows = [protected_row(f"s{i:02d}", correct=True) for i in range(50)]
    protected_path = tmp_path / "protected.json"
    schema_path = tmp_path / "schema.json"
    write_json(protected_path, {"stage_status": "stage_passed", "results": protected_rows})
    write_json(schema_path, {"stage_status": "schema_output_failed_closed", "outputs": [schema_output(f"s{i:02d}") for i in range(50)]})

    projection = build_projection(protected_path, schema_path, tmp_path / "out")

    assert projection["stage_status"] == "plotqa_guardrail_failed"
    assert projection["guardrail"]["passed"] is False
    assert projection["next_required_action"] == "keep_guarded_arm_diagnostic_and_repair_projection"
