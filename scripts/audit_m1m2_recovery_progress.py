#!/usr/bin/env python
"""Audit M1/M2 identifiability recovery progress and paid-attempt accounting."""

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

RECOVERY_ROOT = ROOT / "outputs" / "m1m2_recovery"
LEDGER_PATH = ROOT / "outputs" / "autonomous_run_log" / "budget_ledger.json"
DEFAULT_PHASE_C_RESULT = RECOVERY_ROOT / "phase_c_pilot_repair_round_4_paid_result.json"
DEFAULT_PHASE_C_RESULTS = [
    RECOVERY_ROOT / "phase_c_pilot_paid_result.json",
    RECOVERY_ROOT / "phase_c_pilot_repair_round_2_paid_result.json",
    RECOVERY_ROOT / "phase_c_pilot_repair_round_3_paid_result.json",
    RECOVERY_ROOT / "phase_c_pilot_repair_round_4_paid_result.json",
]
DEFAULT_REPORT = RECOVERY_ROOT / "recovery_progress_audit.json"
DEFAULT_REPORT_MD = RECOVERY_ROOT / "recovery_progress_audit.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ledger_attempts_for_artifact(ledger: dict[str, Any], artifact_path: Path) -> int:
    target = str(artifact_path.resolve()).lower()
    total = 0
    for entry in ledger.get("entries") or []:
        path = str(entry.get("artifact_path") or "")
        if path and str(Path(path).resolve()).lower() == target:
            total += int(entry.get("llm_attempts") or 0)
    return total


def summarize_phase_c_checkpoint(path: Path, ledger: dict[str, Any], prereg: dict[str, Any]) -> dict[str, Any]:
    payload = load_json(path) if path.exists() else {}
    cap = int(payload.get("pilot_llm_http_attempt_cap") or (prereg.get("cap") or {}).get("pilot_llm_http_attempt_cap") or 0)
    attempts = int(payload.get("actual_llm_http_attempts") or 0)
    ledgered = ledger_attempts_for_artifact(ledger, path) if ledger else 0
    rows = payload.get("results") or []
    chain_gate = (payload.get("governance") or {}).get("chain_hash_distinct_gate") or {}
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "sha256": stable_file_hash(path) if path.exists() else None,
        "stage_status": payload.get("stage_status"),
        "completed_sample_quality_jobs": int(payload.get("completed_sample_quality_jobs") or 0),
        "result_rows": len(rows),
        "success_rows": sum(1 for row in rows if row.get("status") == "success"),
        "error_rows": sum(1 for row in rows if row.get("status") != "success"),
        "in_progress_job": payload.get("in_progress_job"),
        "pilot_llm_http_attempt_cap": cap,
        "actual_llm_http_attempts_in_checkpoint": attempts,
        "ledgered_llm_attempts_for_checkpoint": ledgered,
        "unledgered_llm_attempts_for_checkpoint": max(0, attempts - ledgered),
        "remaining_pilot_llm_attempts": max(0, cap - attempts) if cap else None,
        "chain_hash_distinct_gate": chain_gate,
    }


def build_audit(
    *,
    phase_c_result_path: Path = DEFAULT_PHASE_C_RESULT,
    phase_c_result_paths: list[Path] | None = None,
    ledger_path: Path = LEDGER_PATH,
) -> dict[str, Any]:
    a0_path = RECOVERY_ROOT / "m1_tier_discriminative_validity.json"
    phase_a_path = RECOVERY_ROOT / "secondary_metric_variance_study.json"
    prereg_path = RECOVERY_ROOT / "preregistration.json"
    dry_path = RECOVERY_ROOT / "phase_c_dry_run.json"

    a0 = load_json(a0_path) if a0_path.exists() else {}
    phase_a = load_json(phase_a_path) if phase_a_path.exists() else {}
    prereg = load_json(prereg_path) if prereg_path.exists() else {}
    dry = load_json(dry_path) if dry_path.exists() else {}
    ledger = load_json(ledger_path) if ledger_path.exists() else {}

    paths_to_audit = phase_c_result_paths or [phase_c_result_path]
    checkpoint_summaries = [summarize_phase_c_checkpoint(path, ledger, prereg) for path in paths_to_audit]
    primary_checkpoint = checkpoint_summaries[-1] if checkpoint_summaries else {}
    total_checkpoint_attempts = sum(int(row["actual_llm_http_attempts_in_checkpoint"] or 0) for row in checkpoint_summaries)
    total_ledgered_attempts = sum(int(row["ledgered_llm_attempts_for_checkpoint"] or 0) for row in checkpoint_summaries)
    total_unledgered_attempts = sum(int(row["unledgered_llm_attempts_for_checkpoint"] or 0) for row in checkpoint_summaries)
    completed_jobs = int(primary_checkpoint.get("completed_sample_quality_jobs") or 0)
    expected_jobs = 8 * 3
    rows_present = int(primary_checkpoint.get("result_rows") or 0) > 0
    chain_gate = primary_checkpoint.get("chain_hash_distinct_gate") or {}
    phase_c_18_of_24_clean_stop = (
        Path(str(primary_checkpoint.get("path") or "")).name == "phase_c_pilot_repair_round_4_paid_result.json"
        and primary_checkpoint.get("stage_status") == "partial_paid_pilot"
        and completed_jobs == 18
        and int(primary_checkpoint.get("result_rows") or 0) == 48
        and primary_checkpoint.get("in_progress_job") is None
        and total_unledgered_attempts == 0
    )

    blocking: list[str] = []
    if a0.get("stage_status") != "m1_tier_discriminative_validity_completed":
        blocking.append(f"a0_not_completed:{a0.get('stage_status')}")
    elif ((a0.get("primary_240") or {}).get("prereg_verdict") != "discriminative_validity_supported"):
        blocking.append(f"a0_discriminative_validity_not_supported:{(a0.get('primary_240') or {}).get('prereg_verdict')}")
    if phase_a.get("stage_status") != "phase_a_zero_cost_variance_study_completed":
        blocking.append(f"phase_a_not_completed:{phase_a.get('stage_status')}")
    if prereg.get("stage_status") != "phase_b_preregistered_pending_phase_c_cap":
        blocking.append(f"phase_b_preregistration_unexpected:{prereg.get('stage_status')}")
    if dry.get("stage_status") != "phase_c_dry_run_passed":
        blocking.append(f"phase_c_dry_run_not_passed:{dry.get('stage_status')}")
    if not primary_checkpoint.get("exists"):
        blocking.append("phase_c_pilot_result_missing")
    elif primary_checkpoint.get("stage_status") != "phase_c_pilot_completed" and not phase_c_18_of_24_clean_stop:
        blocking.append(f"phase_c_pilot_not_completed:{primary_checkpoint.get('stage_status')}")
    for checkpoint in checkpoint_summaries:
        if checkpoint["unledgered_llm_attempts_for_checkpoint"]:
            blocking.append(
                "budget_ledger_missing_phase_c_attempts:"
                f"{Path(checkpoint['path']).name}:{checkpoint['unledgered_llm_attempts_for_checkpoint']}"
            )
    if primary_checkpoint.get("in_progress_job") is not None:
        blocking.append("phase_c_checkpoint_has_in_progress_job")
    if rows_present and chain_gate.get("passes_nonempty_every_row") is not True and not phase_c_18_of_24_clean_stop:
        blocking.append("phase_c_chain_hash_nonempty_gate_failed")
    if rows_present and chain_gate.get("passes_distinct_gt_0_9n") is not True and not phase_c_18_of_24_clean_stop:
        blocking.append("phase_c_chain_hash_distinct_gate_failed")

    can_resume_under_existing_cap = (
        bool(primary_checkpoint.get("exists"))
        and primary_checkpoint.get("stage_status") != "phase_c_pilot_completed"
        and primary_checkpoint.get("remaining_pilot_llm_attempts") is not None
        and int(primary_checkpoint.get("remaining_pilot_llm_attempts") or 0) > 0
        and completed_jobs < expected_jobs
    )

    return {
        "schema_version": "m1m2_recovery_progress_audit_v1",
        "created_at_utc": utc_now(),
        "stage": "m1m2_recovery_progress_audit",
        "stage_status": "m1m2_recovery_blocked_before_full_completion"
        if blocking
        else "m1m2_recovery_phase_c_clean_stop_18_of_24"
        if phase_c_18_of_24_clean_stop
        else "m1m2_recovery_complete",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "a0_discriminative_validity": {"path": str(a0_path.resolve()), "exists": a0_path.exists(), "sha256": stable_file_hash(a0_path) if a0_path.exists() else None, "stage_status": a0.get("stage_status")},
            "phase_a_variance_study": {"path": str(phase_a_path.resolve()), "exists": phase_a_path.exists(), "sha256": stable_file_hash(phase_a_path) if phase_a_path.exists() else None, "stage_status": phase_a.get("stage_status")},
            "phase_b_preregistration": {"path": str(prereg_path.resolve()), "exists": prereg_path.exists(), "sha256": stable_file_hash(prereg_path) if prereg_path.exists() else None, "stage_status": prereg.get("stage_status")},
            "phase_c_dry_run": {"path": str(dry_path.resolve()), "exists": dry_path.exists(), "sha256": stable_file_hash(dry_path) if dry_path.exists() else None, "stage_status": dry.get("stage_status")},
            "phase_c_pilot_result": {"path": str(phase_c_result_path.resolve()), "exists": phase_c_result_path.exists(), "sha256": stable_file_hash(phase_c_result_path) if phase_c_result_path.exists() else None, "stage_status": primary_checkpoint.get("stage_status")},
            "budget_ledger": {"path": str(ledger_path.resolve()), "exists": ledger_path.exists(), "sha256": stable_file_hash(ledger_path) if ledger_path.exists() else None},
        },
        "phase_c_summary": {
            "expected_sample_quality_jobs": expected_jobs,
            "completed_sample_quality_jobs": completed_jobs,
            "result_rows": primary_checkpoint.get("result_rows"),
            "success_rows": primary_checkpoint.get("success_rows"),
            "error_rows": primary_checkpoint.get("error_rows"),
            "in_progress_job": primary_checkpoint.get("in_progress_job"),
            "pilot_llm_http_attempt_cap": primary_checkpoint.get("pilot_llm_http_attempt_cap"),
            "actual_llm_http_attempts_in_checkpoint": primary_checkpoint.get("actual_llm_http_attempts_in_checkpoint"),
            "ledgered_llm_attempts_for_checkpoint": primary_checkpoint.get("ledgered_llm_attempts_for_checkpoint"),
            "unledgered_llm_attempts_for_checkpoint": primary_checkpoint.get("unledgered_llm_attempts_for_checkpoint"),
            "remaining_pilot_llm_attempts": primary_checkpoint.get("remaining_pilot_llm_attempts"),
            "audited_checkpoints": checkpoint_summaries,
            "total_checkpoint_llm_attempts": total_checkpoint_attempts,
            "total_ledgered_llm_attempts_for_checkpoints": total_ledgered_attempts,
            "total_unledgered_llm_attempts_for_checkpoints": total_unledgered_attempts,
            "chain_hash_distinct_gate": chain_gate,
            "phase_c_18_of_24_clean_stop": phase_c_18_of_24_clean_stop,
        },
        "can_resume_under_existing_pilot_cap": can_resume_under_existing_cap,
        "full_phase_c_authorized": False,
        "blocking_findings": blocking,
        "next_required_action": (
            "repair_budget_ledger_before_any_further_paid_resume"
            if total_unledgered_attempts
            else "report_phase_c_18_of_24_clean_stop_no_new_round"
            if phase_c_18_of_24_clean_stop
            else "resume_or_freeze_new_repair_round_for_phase_c_pilot"
            if can_resume_under_existing_cap
            else "obtain_explicit_new_global_cap_before_full_phase_c"
        ),
        "interpretation_limits": [
            "This audit performs no paid calls and does not modify raw paid results.",
            "A0 supports M1 tier discriminative validity, not calibrated probabilities.",
            "The Phase C pilot is complete mechanism-curve evidence only if all 24 sample-quality jobs complete; the round4 18/24 clean stop is timeout-limited manipulation evidence.",
            "Full Phase C remains unauthorized under the conservative default until a new global cap is explicit.",
        ],
    }


def repair_ledger(audit: dict[str, Any], ledger_path: Path = LEDGER_PATH) -> dict[str, Any]:
    checkpoints = (audit.get("phase_c_summary") or {}).get("audited_checkpoints") or []
    missing_total = sum(int(row.get("unledgered_llm_attempts_for_checkpoint") or 0) for row in checkpoints)
    if missing_total <= 0:
        return {"repaired": False, "reason": "no_unledgered_attempts"}
    ledger = load_json(ledger_path)
    entries = []
    for checkpoint in checkpoints:
        missing = int(checkpoint.get("unledgered_llm_attempts_for_checkpoint") or 0)
        if missing <= 0:
            continue
        entry = {
            "timestamp": utc_now(),
            "phase": "phase_c",
            "step": "phase_c_unclosed_checkpoint_conservative_accounting",
            "llm_attempts": missing,
            "ocr_attempts": 0,
            "artifact_path": checkpoint["path"],
            "artifact_sha256": checkpoint["sha256"],
        }
        ledger.setdefault("entries", []).append(entry)
        entries.append(entry)
    ledger["cumulative_llm"] = int(ledger.get("cumulative_llm") or 0) + missing_total
    ledger["cumulative_ocr"] = int(ledger.get("cumulative_ocr") or 0)
    ledger.setdefault("interpretation_limits", []).append(
        "Phase C unclosed checkpoints are conservatively ledgered from checkpoint totals across all audited pilot rounds."
    )
    write_json(ledger_path, ledger)
    return {"repaired": True, "entries": entries, "cumulative_llm": ledger["cumulative_llm"]}


def render_md(audit: dict[str, Any]) -> str:
    phase_c = audit["phase_c_summary"]
    return "\n".join(
        [
            "# M1/M2 Recovery Progress Audit",
            "",
            f"- Stage status: `{audit['stage_status']}`",
            f"- Blocking findings: `{len(audit['blocking_findings'])}`",
            f"- A0 verdict: `{((load_json(Path(audit['source_artifacts']['a0_discriminative_validity']['path'])).get('primary_240') or {}).get('prereg_verdict') if audit['source_artifacts']['a0_discriminative_validity']['exists'] else None)}`",
            f"- Phase C completed jobs: `{phase_c['completed_sample_quality_jobs']}/{phase_c['expected_sample_quality_jobs']}`",
            f"- Phase C checkpoint attempts: `{phase_c['actual_llm_http_attempts_in_checkpoint']}/{phase_c['pilot_llm_http_attempt_cap']}`",
            f"- Unledgered checkpoint attempts: `{phase_c['unledgered_llm_attempts_for_checkpoint']}`",
            f"- Next required action: `{audit['next_required_action']}`",
            "",
            "## Blocking Findings",
            "",
            *[f"- `{finding}`" for finding in audit["blocking_findings"]],
            "",
        ]
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-c-result", type=Path, default=DEFAULT_PHASE_C_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--repair-ledger", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    phase_c_paths = DEFAULT_PHASE_C_RESULTS if args.phase_c_result == DEFAULT_PHASE_C_RESULT else [args.phase_c_result]
    audit = build_audit(phase_c_result_path=args.phase_c_result, phase_c_result_paths=phase_c_paths)
    if args.repair_ledger:
        repair = repair_ledger(audit)
        audit = build_audit(phase_c_result_path=args.phase_c_result, phase_c_result_paths=phase_c_paths)
        audit["ledger_repair"] = repair
    write_json(args.output, audit)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.write_text(render_md(audit), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.md_output}")
    print(f"stage_status={audit['stage_status']}")
    print(f"blocking_findings={len(audit['blocking_findings'])}")
    print(f"next_required_action={audit['next_required_action']}")
    print("paid_calls_executed=0")
    return 0 if audit["stage_status"] in {
        "m1m2_recovery_complete",
        "m1m2_recovery_phase_c_clean_stop_18_of_24",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
