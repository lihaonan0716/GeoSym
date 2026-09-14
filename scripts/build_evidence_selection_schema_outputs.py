#!/usr/bin/env python
"""Build schema-only M1/M2 recovery artifacts from measured PlotQA records.

This script is a zero-cost P2 step. It emits EvidenceCandidate, EvidenceGap,
SelectedEvidenceSet, and M1M2DecisionAudit records for the experimental recovery
path without modifying the protected measured-GeoSym default.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_public_benchmark_pilot import stable_file_hash, stable_payload_hash  # noqa: E402
from scripts.run_plotqa_locate import measured_prompt_facts  # noqa: E402

SCHEMA_VERSION = "m1m2_schema_only_outputs_v1"
CONDITION = "G_full_guarded_schema_only"
SUPPORTED_TIERS = {"high", "medium", "low", "unknown"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_tier(value: Any) -> str:
    tier = str(value or "unknown").strip().lower()
    return tier if tier in SUPPORTED_TIERS else "unknown"


def candidate_key(point: dict[str, Any]) -> str:
    if point.get("x_value") is not None or point.get("y_value") is not None:
        return "measured_value"
    if point.get("series_label") is not None:
        return "series_label"
    if point.get("category_label") is not None:
        return "category_label"
    return "measured_value"


def evidence_candidate(point: dict[str, Any]) -> dict[str, Any]:
    fact_id = str(point.get("fact_id") or stable_payload_hash(point)[:12])
    return {
        "fact_id": fact_id,
        "key": candidate_key(point),
        "value": {
            key: point.get(key)
            for key in ("x_value", "y_value", "series_label", "category_label", "pixel")
            if key in point
        },
        "provenance": str(point.get("source") or "cv_measurement"),
        "support": list(point.get("support") or []),
        "reliability_score": None,
        "reliability_tier": normalize_tier(point.get("confidence")),
        "uncertainty": point.get("uncertainty"),
        "status": "present",
    }


def normalize_gap(raw: dict[str, Any], idx: int, *, needed_for_question: bool) -> dict[str, Any]:
    reason = str(raw.get("reason") or raw.get("label") or "not_detected")
    severity = str(raw.get("severity") or ("blocking" if needed_for_question else "warning")).lower()
    if severity == "nonblocking":
        severity = "warning"
    if severity not in {"blocking", "warning", "informational"}:
        severity = "warning"
    return {
        "gap_id": str(raw.get("gap_id") or f"gap_{idx:03d}"),
        "key": str(raw.get("key") or raw.get("needed_for") or "measured_value"),
        "needed_for_question": needed_for_question,
        "reason": reason,
        "recovery_action": str(raw.get("recovery_action") or ("deterministic_solver" if needed_for_question else "pass_through")),
        "severity": severity,
    }


def generated_risk_gaps(record: dict[str, Any], audit_row: dict[str, Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for idx, reason in enumerate(audit_row.get("risk_reasons") or []):
        if "ruler" in reason:
            key = "axis_scale"
        elif "series" in reason:
            key = "series_label"
        else:
            key = "measured_value"
        gaps.append(
            {
                "gap_id": f"risk_{idx:03d}_{key}",
                "key": key,
                "needed_for_question": True,
                "reason": reason,
                "recovery_action": "deterministic_solver" if audit_row["classification_flags"].get("deterministic_solver_eligible") else "inspect_image",
                "severity": "blocking" if audit_row["primary_classification"] == "candidate-risk" else "warning",
            }
        )
    return gaps


def selected_evidence_set(
    record: dict[str, Any],
    audit_row: dict[str, Any],
    candidates: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    curated = measured_prompt_facts(record, None, curated=True)
    curated_ids = {str(fact.get("fact_id")) for fact in curated.get("measured_facts") or [] if fact.get("fact_id")}
    all_ids = {candidate["fact_id"] for candidate in candidates}
    selected = sorted(curated_ids & all_ids)
    suppressed = sorted(all_ids - set(selected))
    blocking_gaps = [gap["gap_id"] for gap in gaps if gap.get("severity") == "blocking"]
    flags = audit_row.get("classification_flags") or {}
    if blocking_gaps and not flags.get("deterministic_solver_eligible"):
        policy = "insufficient_evidence"
    elif flags.get("deterministic_solver_eligible"):
        policy = "solve_deterministically"
    elif audit_row.get("primary_classification") == "clean-pass-through":
        policy = "use_directly"
    elif audit_row.get("primary_classification") == "candidate-risk":
        policy = "use_with_caution"
    else:
        policy = "verify_from_image"
    return {
        "selected_fact_ids": selected,
        "suppressed_fact_ids": suppressed,
        "surfaced_gap_ids": sorted(gap["gap_id"] for gap in gaps),
        "propagation_policy": policy,
        "audit_notes": [
            f"primary_classification={audit_row.get('primary_classification')}",
            f"selected={len(selected)}",
            f"suppressed={len(suppressed)}",
            f"surfaced_gaps={len(gaps)}",
        ],
    }


def build_sample_output(record: dict[str, Any], audit_row: dict[str, Any]) -> dict[str, Any]:
    candidates = [evidence_candidate(point) for point in (record.get("measured_points") or [])[:30]]
    raw_gaps = record.get("evidence_gaps") or []
    existing_gaps = [
        normalize_gap(gap, idx, needed_for_question=audit_row["primary_classification"] == "evidence-gap")
        for idx, gap in enumerate(raw_gaps)
        if isinstance(gap, dict)
    ]
    gaps = existing_gaps + generated_risk_gaps(record, audit_row)
    selected = selected_evidence_set(record, audit_row, candidates, gaps)
    decision_audit = {
        "sample_id": record["sample_id"],
        "question_hash": audit_row["question_hash"],
        "source_record_hash": audit_row["source_record_hash"],
        "condition": CONDITION,
        "selected_fact_ids": selected["selected_fact_ids"],
        "suppressed_fact_ids": selected["suppressed_fact_ids"],
        "surfaced_gap_ids": selected["surfaced_gap_ids"],
        "deterministic_solver_used": selected["propagation_policy"] == "solve_deterministically",
        "pass_through_used": selected["propagation_policy"] == "use_directly",
        "protected_default_output_changed": False,
        "failure_mode": "none" if candidates or gaps else "missing_required_fact",
    }
    return {
        "sample_id": record["sample_id"],
        "question_hash": audit_row["question_hash"],
        "condition": CONDITION,
        "evidence_candidates": candidates,
        "evidence_gaps": gaps,
        "selected_evidence_set": selected,
        "decision_audit": decision_audit,
    }


def validate_sample_output(row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    candidate_ids = {candidate.get("fact_id") for candidate in row.get("evidence_candidates") or []}
    gap_ids = {gap.get("gap_id") for gap in row.get("evidence_gaps") or []}
    selected = row.get("selected_evidence_set") or {}
    if not row.get("sample_id"):
        violations.append("sample_id_missing")
    for fact_id in selected.get("selected_fact_ids") or []:
        if fact_id not in candidate_ids:
            violations.append("selected_fact_id_missing_candidate")
            break
    for fact_id in selected.get("suppressed_fact_ids") or []:
        if fact_id not in candidate_ids:
            violations.append("suppressed_fact_id_missing_candidate")
            break
    for gap_id in selected.get("surfaced_gap_ids") or []:
        if gap_id not in gap_ids:
            violations.append("surfaced_gap_id_missing_gap")
            break
    if (row.get("decision_audit") or {}).get("protected_default_output_changed") is not False:
        violations.append("protected_default_output_changed")
    return violations


def build_schema_outputs(measured_records_path: Path, guardrail_audit_path: Path, output_dir: Path) -> dict[str, Any]:
    records = load_json(measured_records_path)
    audit = load_json(guardrail_audit_path)
    audit_rows = {row["sample_id"]: row for row in audit.get("rows", [])}
    outputs = [
        build_sample_output(record, audit_rows[record["sample_id"]])
        for record in sorted(records, key=lambda row: str(row["sample_id"]))
        if record["sample_id"] in audit_rows
    ]
    violations = {row["sample_id"]: validate_sample_output(row) for row in outputs}
    violations = {sample_id: items for sample_id, items in violations.items() if items}
    policy_counts = Counter(row["selected_evidence_set"]["propagation_policy"] for row in outputs)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "p2_schema_only_m1m2_experimental_output",
        "stage_status": "ready_for_zero_cost_projection" if len(outputs) == 50 and not violations else "schema_output_failed_closed",
        "condition": CONDITION,
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "default_path_modified": False,
        "source_artifacts": {
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
            "guardrail_audit": {
                "path": str(guardrail_audit_path.resolve()),
                "sha256": stable_file_hash(guardrail_audit_path),
                "stage_status": audit.get("stage_status"),
            },
        },
        "sample_count": len(outputs),
        "summary": {
            "candidate_count": sum(len(row["evidence_candidates"]) for row in outputs),
            "gap_count": sum(len(row["evidence_gaps"]) for row in outputs),
            "selected_fact_count": sum(len(row["selected_evidence_set"]["selected_fact_ids"]) for row in outputs),
            "suppressed_fact_count": sum(len(row["selected_evidence_set"]["suppressed_fact_ids"]) for row in outputs),
            "policy_counts": dict(sorted(policy_counts.items())),
            "schema_violation_count": sum(len(items) for items in violations.values()),
            "protected_default_output_changes": 0,
        },
        "schema_violations": violations,
        "outputs": outputs,
        "next_required_action": "run_zero_cost_guarded_projection" if len(outputs) == 50 and not violations else "repair_schema_output_before_projection",
        "interpretation_limits": [
            "This artifact is schema-only and performs no answer generation.",
            "It does not prove an M1/M2 accuracy contribution.",
            "The protected G_full_measured_geosym default path is not modified.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m1m2_schema_outputs.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "m1m2_schema_outputs.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# M1/M2 Schema-Only Outputs",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed: 0",
        f"- Sample rows: {result['sample_count']}",
        f"- Evidence candidates: {summary['candidate_count']}",
        f"- Evidence gaps: {summary['gap_count']}",
        f"- Selected facts: {summary['selected_fact_count']}",
        f"- Suppressed facts: {summary['suppressed_fact_count']}",
        f"- Protected default output changes: {summary['protected_default_output_changes']}",
        "",
        "## Propagation Policies",
        "",
    ]
    for policy, count in summary["policy_counts"].items():
        lines.append(f"- {policy}: {count}")
    lines.extend(["", "## Next Required Action", "", f"- `{result['next_required_action']}`", "", "## Limits", ""])
    lines.extend(f"- {line}" for line in result["interpretation_limits"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured-records", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/m0_measurement_records.json")
    parser.add_argument("--guardrail-audit", default="outputs/m1m2_recovery/plotqa_guardrail_audit/plotqa_guardrail_audit.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_recovery/schema_outputs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = build_schema_outputs(Path(args.measured_records), Path(args.guardrail_audit), Path(args.output_dir))
    print(f"wrote schema outputs: {Path(args.output_dir) / 'm1m2_schema_outputs.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
