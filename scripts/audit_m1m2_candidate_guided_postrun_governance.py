#!/usr/bin/env python
"""Audit the candidate-guided confirmation governance chain after the run.

This is a zero-cost audit. It never calls an LLM, OCR service, or judge. Its
purpose is to prove that the current manifest, preapproval audit, dry run, paid
result, analysis, and goal-status audit close over the same frozen package.
"""

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

from scripts.freeze_m1m2_candidate_guided_confirmation_manifest import load_default_model_provenance  # noqa: E402
from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402

SCHEMA_VERSION = "m1m2_candidate_guided_postrun_governance_audit_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def artifact(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    exists = path.exists()
    return {
        "path": str(path.resolve()),
        "exists": exists,
        "sha256": stable_file_hash(path) if exists else None,
        "stage_status": payload.get("stage_status"),
        "paid_calls_executed": payload.get("paid_calls_executed"),
        "actual_llm_http_attempts": payload.get("actual_llm_http_attempts"),
        "actual_ocr_http_attempts": payload.get("actual_ocr_http_attempts"),
    }


def source_hash(payload: dict[str, Any], name: str) -> str | None:
    source = (payload.get("source_artifacts") or {}).get(name) or {}
    return source.get("sha256")


def source_status(payload: dict[str, Any], name: str) -> str | None:
    source = (payload.get("source_artifacts") or {}).get(name) or {}
    return source.get("stage_status")


def audit(
    manifest_path: Path,
    roster_path: Path,
    preapproval_path: Path,
    dry_run_path: Path,
    paid_result_path: Path,
    analysis_path: Path,
    goal_status_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    roster = load_json(roster_path)
    preapproval = load_json(preapproval_path)
    dry_run = load_json(dry_run_path)
    paid_result = load_json(paid_result_path)
    analysis = load_json(analysis_path)
    goal_status = load_json(goal_status_path)

    manifest_sha = stable_file_hash(manifest_path) if manifest_path.exists() else None
    roster_sha = stable_file_hash(roster_path) if roster_path.exists() else None
    paid_sha = stable_file_hash(paid_result_path) if paid_result_path.exists() else None
    config_provenance = load_default_model_provenance()
    goal_guardrail = goal_status.get("plotqa_guardrail") or {}
    manifest_cap = manifest.get("llm_http_attempt_cap")

    gates = {
        "all_artifacts_exist": all(
            path.exists()
            for path in (manifest_path, roster_path, preapproval_path, dry_run_path, paid_result_path, analysis_path, goal_status_path)
        ),
        "manifest_stage_frozen": manifest.get("stage_status") == "candidate_guided_confirmation_manifest_frozen_pending_paid_cap_approval",
        "preapproval_ready": preapproval.get("stage_status") == "ready_to_request_96_call_paid_cap_approval",
        "preapproval_paid_calls_zero": preapproval.get("paid_calls_executed_by_audit") == 0,
        "preapproval_manifest_hash_current": source_hash(preapproval, "manifest") == manifest_sha,
        "dry_run_passed": dry_run.get("stage_status") == "candidate_guided_confirmation_dry_run_passed",
        "dry_run_paid_calls_zero": dry_run.get("paid_calls_executed") == 0,
        "dry_run_manifest_hash_current": source_hash(dry_run, "manifest") == manifest_sha,
        "dry_run_roster_hash_current": source_hash(dry_run, "confirmation_roster") == roster_sha,
        "paid_result_completed": paid_result.get("stage_status") == "candidate_guided_confirmation_completed",
        "paid_result_manifest_hash_current": source_hash(paid_result, "manifest") == manifest_sha,
        "paid_result_roster_hash_current": source_hash(paid_result, "confirmation_roster") == roster_sha,
        "paid_cap_respected": paid_result.get("paid_calls_executed") == paid_result.get("actual_llm_http_attempts") == manifest_cap == 96,
        "ocr_cap_zero": paid_result.get("actual_ocr_http_attempts") == manifest.get("ocr_http_attempt_cap") == 0,
        "analysis_supported": analysis.get("stage_status") == "m1m2_candidate_guided_confirmation_supported",
        "analysis_paid_result_status_current": source_status(analysis, "paid_result") == paid_result.get("stage_status"),
        "analysis_paid_result_calls_current": ((analysis.get("source_artifacts") or {}).get("paid_result") or {}).get("paid_calls_executed")
        == paid_result.get("paid_calls_executed"),
        "goal_status_complete": goal_status.get("stage_status") == "m1m2_rescue_goal_complete_evidence_supported",
        "goal_plotqa_guardrail_48_50_preserved": goal_guardrail.get("passed") is True
        and goal_guardrail.get("observed_correct") == 48
        and goal_guardrail.get("observed_completed") == 50
        and goal_guardrail.get("observed_parseable") == 50
        and goal_guardrail.get("protected_output_regressions") == 0,
        "config_default_model_matches_manifest": config_provenance["model_display_name"] == manifest.get("model_display_name")
        and config_provenance["model_registry_id_or_config_value"] == manifest.get("model_registry_id_or_config_value"),
        "paid_result_model_display_matches_manifest": paid_result.get("model_display_name") == manifest.get("model_display_name"),
        "postrun_audit_paid_calls_zero": True,
    }
    blocking = [name for name, passed in gates.items() if not passed]
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_candidate_guided_postrun_governance_audit",
        "stage_status": "candidate_guided_postrun_governance_clean" if not blocking else "candidate_guided_postrun_governance_failed_closed",
        "paid_calls_executed_by_audit": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "manifest": artifact(manifest_path),
            "confirmation_roster": artifact(roster_path),
            "preapproval_audit": artifact(preapproval_path),
            "dry_run": artifact(dry_run_path),
            "paid_result": artifact(paid_result_path),
            "analysis": artifact(analysis_path),
            "goal_status_audit": artifact(goal_status_path),
        },
        "cross_stage_hashes": {
            "current_manifest_sha256": manifest_sha,
            "current_confirmation_roster_sha256": roster_sha,
            "current_paid_result_sha256": paid_sha,
            "preapproval_manifest_sha256": source_hash(preapproval, "manifest"),
            "dry_run_manifest_sha256": source_hash(dry_run, "manifest"),
            "paid_result_manifest_sha256": source_hash(paid_result, "manifest"),
            "dry_run_roster_sha256": source_hash(dry_run, "confirmation_roster"),
            "paid_result_roster_sha256": source_hash(paid_result, "confirmation_roster"),
        },
        "config_model_provenance": config_provenance,
        "gates": gates,
        "blocking_findings": blocking,
        "next_required_action": "paper_package_ready_for_narrowed_m1m2_claim_review" if not blocking else "repair_candidate_guided_governance_chain",
        "interpretation_limits": [
            "This audit performs no paid calls and does not alter paid_result.json.",
            "The supported claim remains limited to candidate-guided answerability-preserving selective propagation.",
            "The protected PlotQA default remains 48/50 and must not be degraded.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "postrun_governance_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "postrun_governance_audit.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Candidate-Guided Post-Run Governance Audit",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed by audit: 0",
        "",
        "## Hash Closure",
        "",
    ]
    for key, value in result["cross_stage_hashes"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- {key}: {value}" for key, value in result["gates"].items())
    lines.extend(["", "## Blocking Findings", ""])
    lines.extend(f"- `{item}`" for item in result["blocking_findings"] or ["none"])
    lines.extend(["", "## Next Required Action", "", f"- `{result['next_required_action']}`", ""])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/m1m2_candidate_guided_v31/confirmation_manifest/frozen_confirmation_manifest.json")
    parser.add_argument("--confirmation-roster", default="outputs/m1m2_candidate_guided_v31/confirmation_manifest/confirmation_roster.json")
    parser.add_argument("--preapproval", default="outputs/m1m2_candidate_guided_v31/confirmation_preapproval_audit/preapproval_audit.json")
    parser.add_argument("--dry-run", default="outputs/m1m2_candidate_guided_v31/confirmation_run/dry_run.json")
    parser.add_argument("--paid-result", default="outputs/m1m2_candidate_guided_v31/confirmation_run/paid_result.json")
    parser.add_argument("--analysis", default="outputs/m1m2_candidate_guided_v31/confirmation_analysis/analysis.json")
    parser.add_argument("--goal-status", default="outputs/m1m2_candidate_guided_v31/goal_status_audit/goal_status_audit.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_candidate_guided_v31/postrun_governance_audit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit(
        Path(args.manifest),
        Path(args.confirmation_roster),
        Path(args.preapproval),
        Path(args.dry_run),
        Path(args.paid_result),
        Path(args.analysis),
        Path(args.goal_status),
        Path(args.output_dir),
    )
    print(f"wrote post-run governance audit: {Path(args.output_dir) / 'postrun_governance_audit.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed_by_audit=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
