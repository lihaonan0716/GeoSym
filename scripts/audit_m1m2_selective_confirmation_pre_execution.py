#!/usr/bin/env python
"""Audit M1/M2 selective confirmation readiness before paid execution."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402

SCHEMA_VERSION = "m1m2_selective_confirmation_pre_execution_audit_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def path_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path.resolve()),
        "exists": exists,
        "sha256": stable_file_hash(path) if exists else None,
    }


def audit(
    protocol_path: Path,
    next_plan_path: Path,
    manifest_path: Path,
    dry_run_path: Path,
    zero_cost_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    dry_run = load_json(dry_run_path) if dry_run_path.exists() else {}
    zero_cost = load_json(zero_cost_path) if zero_cost_path.exists() else {}
    plotqa_guardrail = zero_cost.get("plotqa_guardrail") or {}
    approval_required = int(manifest.get("approval_required_for_llm_http_attempt_cap") or 0)
    gates = {
        "protocol_exists": protocol_path.exists(),
        "next_step_plan_exists": next_plan_path.exists(),
        "manifest_frozen_pending_approval": manifest.get("stage_status") == "selective_confirmation_manifest_frozen_pending_approval",
        "manifest_requires_explicit_paid_cap_approval": manifest.get("requires_explicit_paid_cap_approval_before_execution") is True,
        "approval_required_for_288_llm_attempts": approval_required == 288,
        "manifest_paid_calls_zero": manifest.get("paid_calls_executed") == 0,
        "manifest_ocr_cap_zero": manifest.get("ocr_http_attempt_cap") == 0,
        "manifest_input_mode_evidence_only": manifest.get("input_mode") == "evidence_only",
        "manifest_estimated_cost_available": float(manifest.get("estimated_cost_usd_upper_bound") or 0) > 0,
        "dry_run_ready_pending_approval": dry_run.get("stage_status") == "selective_confirmation_dry_run_ready_pending_approval",
        "dry_run_paid_calls_zero": dry_run.get("paid_calls_executed") == 0,
        "dry_run_llm_attempts_zero": dry_run.get("actual_llm_http_attempts") == 0,
        "dry_run_ocr_attempts_zero": dry_run.get("actual_ocr_http_attempts") == 0,
        "dry_run_job_count_288": dry_run.get("job_count") == 288,
        "dry_run_has_no_blocking_findings": dry_run.get("blocking_findings") == [],
        "dry_run_does_not_approve_paid_cap": dry_run.get("approved_paid_cap_for_this_dry_run") is False,
        "plotqa_guardrail_ready": zero_cost.get("stage_status") == "ready_for_paid_manifest",
        "plotqa_48_of_50_preserved": plotqa_guardrail.get("observed_correct") == 48,
        "plotqa_completed_50_of_50": plotqa_guardrail.get("observed_completed") == 50,
        "plotqa_parseable_50_of_50": plotqa_guardrail.get("observed_parseable") == 50,
        "plotqa_protected_regressions_zero": plotqa_guardrail.get("protected_output_regressions") == 0,
    }
    blocking_findings = [name for name, passed in gates.items() if not passed]
    state = "ready_pending_explicit_paid_cap_approval" if not blocking_findings else "pre_execution_audit_failed_closed"
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_selective_confirmation_pre_execution_audit",
        "stage_status": state,
        "paid_calls_executed_by_audit": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "approval_required_before_paid_execution": True,
        "approval_required_for_llm_http_attempt_cap": approval_required,
        "estimated_cost_usd_upper_bound": manifest.get("estimated_cost_usd_upper_bound"),
        "source_artifacts": {
            "protocol": path_status(protocol_path),
            "next_step_plan": path_status(next_plan_path),
            "manifest": path_status(manifest_path),
            "dry_run": path_status(dry_run_path),
            "zero_cost_projection": path_status(zero_cost_path),
        },
        "gates": gates,
        "blocking_findings": blocking_findings,
        "next_required_action": "request_explicit_288_attempt_paid_cap_approval" if not blocking_findings else "repair_pre_execution_inputs",
        "interpretation_limits": [
            "This audit does not authorize paid calls.",
            "This audit does not analyze confirmation results because the paid confirmation has not run.",
            "The protected PlotQA 48/50 path remains the default guardrail.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pre_execution_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "pre_execution_audit.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Selective Confirmation Pre-Execution Audit",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed by audit: 0",
        f"- Approval required before paid execution: {result['approval_required_before_paid_execution']}",
        f"- Approval required LLM HTTP attempt cap: {result['approval_required_for_llm_http_attempt_cap']}",
        f"- Estimated cost upper bound USD: {result['estimated_cost_usd_upper_bound']}",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {name}: {passed}" for name, passed in result["gates"].items())
    lines.extend(["", "## Blocking Findings", ""])
    lines.extend(f"- `{item}`" for item in result["blocking_findings"] or ["none"])
    lines.extend(["", "## Next Required Action", "", f"- `{result['next_required_action']}`", ""])
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--next-plan", required=True)
    parser.add_argument("--manifest", default="outputs/m1m2_selective_propagation/confirmation_manifest/frozen_confirmation_manifest.json")
    parser.add_argument("--dry-run", default="outputs/m1m2_selective_propagation/confirmation_run/dry_run.json")
    parser.add_argument("--zero-cost", default="outputs/m1m2_selective_propagation/zero_cost_projection/selective_zero_cost_projection.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_selective_propagation/pre_execution_audit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = audit(
        Path(args.protocol),
        Path(args.next_plan),
        Path(args.manifest),
        Path(args.dry_run),
        Path(args.zero_cost),
        Path(args.output_dir),
    )
    print(f"wrote selective confirmation pre-execution audit: {Path(args.output_dir) / 'pre_execution_audit.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed_by_audit=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
