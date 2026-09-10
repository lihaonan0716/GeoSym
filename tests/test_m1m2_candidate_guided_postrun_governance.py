import json
from pathlib import Path

from scripts.audit_m1m2_candidate_guided_postrun_governance import audit
from scripts.prepare_public_benchmark_pilot import stable_file_hash


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_package(tmp_path: Path, manifest_hash_override: str | None = None) -> dict[str, Path]:
    manifest = tmp_path / "manifest.json"
    roster = tmp_path / "roster.json"
    preapproval = tmp_path / "preapproval.json"
    dry_run = tmp_path / "dry_run.json"
    paid = tmp_path / "paid.json"
    analysis = tmp_path / "analysis.json"
    goal = tmp_path / "goal.json"

    write_json(
        manifest,
        {
            "stage_status": "candidate_guided_confirmation_manifest_frozen_pending_paid_cap_approval",
            "llm_http_attempt_cap": 96,
            "ocr_http_attempt_cap": 0,
            "model_display_name": "GPT-5.5",
            "model_registry_id_or_config_value": "gpt-5.5",
        },
    )
    write_json(roster, {"stage_status": "candidate_guided_confirmation_roster_frozen"})
    manifest_sha = manifest_hash_override or stable_file_hash(manifest)
    roster_sha = stable_file_hash(roster)
    write_json(
        preapproval,
        {
            "stage_status": "ready_to_request_96_call_paid_cap_approval",
            "paid_calls_executed_by_audit": 0,
            "source_artifacts": {"manifest": {"sha256": manifest_sha}},
        },
    )
    write_json(
        dry_run,
        {
            "stage_status": "candidate_guided_confirmation_dry_run_passed",
            "paid_calls_executed": 0,
            "source_artifacts": {"manifest": {"sha256": manifest_sha}, "confirmation_roster": {"sha256": roster_sha}},
        },
    )
    write_json(
        paid,
        {
            "stage_status": "candidate_guided_confirmation_completed",
            "paid_calls_executed": 96,
            "actual_llm_http_attempts": 96,
            "actual_ocr_http_attempts": 0,
            "model_display_name": "GPT-5.5",
            "source_artifacts": {"manifest": {"sha256": manifest_sha}, "confirmation_roster": {"sha256": roster_sha}},
        },
    )
    write_json(
        analysis,
        {
            "stage_status": "m1m2_candidate_guided_confirmation_supported",
            "source_artifacts": {"paid_result": {"stage_status": "candidate_guided_confirmation_completed", "paid_calls_executed": 96}},
        },
    )
    write_json(
        goal,
        {
            "stage_status": "m1m2_rescue_goal_complete_evidence_supported",
            "plotqa_guardrail": {
                "passed": True,
                "observed_correct": 48,
                "observed_completed": 50,
                "observed_parseable": 50,
                "protected_output_regressions": 0,
            },
        },
    )
    return {
        "manifest": manifest,
        "roster": roster,
        "preapproval": preapproval,
        "dry_run": dry_run,
        "paid": paid,
        "analysis": analysis,
        "goal": goal,
    }


def run_audit(paths: dict[str, Path], out: Path) -> dict:
    return audit(
        paths["manifest"],
        paths["roster"],
        paths["preapproval"],
        paths["dry_run"],
        paths["paid"],
        paths["analysis"],
        paths["goal"],
        out,
    )


def test_postrun_governance_audit_closes_clean_package(tmp_path: Path) -> None:
    paths = write_package(tmp_path)

    result = run_audit(paths, tmp_path / "out")

    assert result["stage_status"] == "candidate_guided_postrun_governance_clean"
    assert result["paid_calls_executed_by_audit"] == 0
    assert result["blocking_findings"] == []


def test_postrun_governance_audit_fails_on_manifest_hash_mismatch(tmp_path: Path) -> None:
    paths = write_package(tmp_path, manifest_hash_override="stale")

    result = run_audit(paths, tmp_path / "out")

    assert result["stage_status"] == "candidate_guided_postrun_governance_failed_closed"
    assert "preapproval_manifest_hash_current" in result["blocking_findings"]
    assert "paid_result_manifest_hash_current" in result["blocking_findings"]
