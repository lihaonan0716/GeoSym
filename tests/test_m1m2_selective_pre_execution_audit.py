import json
from pathlib import Path

from scripts.audit_m1m2_selective_confirmation_pre_execution import audit


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pre_execution_audit_reports_ready_pending_approval(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.md"
    next_plan = tmp_path / "plan.md"
    manifest = tmp_path / "manifest.json"
    dry_run = tmp_path / "dry_run.json"
    zero_cost = tmp_path / "zero_cost.json"
    protocol.write_text("protocol", encoding="utf-8")
    next_plan.write_text("plan", encoding="utf-8")
    write_json(
        manifest,
        {
            "stage_status": "selective_confirmation_manifest_frozen_pending_approval",
            "requires_explicit_paid_cap_approval_before_execution": True,
            "approval_required_for_llm_http_attempt_cap": 288,
            "paid_calls_executed": 0,
            "ocr_http_attempt_cap": 0,
            "input_mode": "evidence_only",
            "estimated_cost_usd_upper_bound": 1.6266,
        },
    )
    write_json(
        dry_run,
        {
            "stage_status": "selective_confirmation_dry_run_ready_pending_approval",
            "paid_calls_executed": 0,
            "actual_llm_http_attempts": 0,
            "actual_ocr_http_attempts": 0,
            "job_count": 288,
            "blocking_findings": [],
            "approved_paid_cap_for_this_dry_run": False,
        },
    )
    write_json(
        zero_cost,
        {
            "stage_status": "ready_for_paid_manifest",
            "plotqa_guardrail": {
                "observed_correct": 48,
                "observed_completed": 50,
                "observed_parseable": 50,
                "protected_output_regressions": 0,
            },
        },
    )

    result = audit(protocol, next_plan, manifest, dry_run, zero_cost, tmp_path / "out")

    assert result["stage_status"] == "ready_pending_explicit_paid_cap_approval"
    assert result["paid_calls_executed_by_audit"] == 0
    assert result["approval_required_for_llm_http_attempt_cap"] == 288
    assert result["gates"]["plotqa_48_of_50_preserved"] is True
    assert result["next_required_action"] == "request_explicit_288_attempt_paid_cap_approval"


def test_pre_execution_audit_fails_closed_when_plotqa_guardrail_missing(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.md"
    next_plan = tmp_path / "plan.md"
    manifest = tmp_path / "manifest.json"
    dry_run = tmp_path / "dry_run.json"
    zero_cost = tmp_path / "zero_cost.json"
    protocol.write_text("protocol", encoding="utf-8")
    next_plan.write_text("plan", encoding="utf-8")
    write_json(
        manifest,
        {
            "stage_status": "selective_confirmation_manifest_frozen_pending_approval",
            "requires_explicit_paid_cap_approval_before_execution": True,
            "approval_required_for_llm_http_attempt_cap": 288,
            "paid_calls_executed": 0,
            "ocr_http_attempt_cap": 0,
            "input_mode": "evidence_only",
            "estimated_cost_usd_upper_bound": 1.6266,
        },
    )
    write_json(
        dry_run,
        {
            "stage_status": "selective_confirmation_dry_run_ready_pending_approval",
            "paid_calls_executed": 0,
            "actual_llm_http_attempts": 0,
            "actual_ocr_http_attempts": 0,
            "job_count": 288,
            "blocking_findings": [],
            "approved_paid_cap_for_this_dry_run": False,
        },
    )
    write_json(
        zero_cost,
        {
            "stage_status": "ready_for_paid_manifest",
            "plotqa_guardrail": {
                "observed_correct": 47,
                "observed_completed": 50,
                "observed_parseable": 50,
                "protected_output_regressions": 1,
            },
        },
    )

    result = audit(protocol, next_plan, manifest, dry_run, zero_cost, tmp_path / "out")

    assert result["stage_status"] == "pre_execution_audit_failed_closed"
    assert "plotqa_48_of_50_preserved" in result["blocking_findings"]
    assert "plotqa_protected_regressions_zero" in result["blocking_findings"]
