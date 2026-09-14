#!/usr/bin/env python
"""Zero-cost P1 guardrail audit for M1/M2 recovery on PlotQA.

The audit reads frozen measured-GeoSym artifacts, classifies every sample with
deterministic rules, and writes machine-readable output for the next recovery
stage. It performs no paid calls and never rewrites paid results.
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
from scripts.run_plotqa_measured_geosym_stage3 import deterministic_prediction_from_measured  # noqa: E402

PROTECTED_CONDITION = "G_full_measured_geosym"
RAW_CONDITION = "G_measured_raw"
DIRECT_CONDITIONS = ("E0_plotqa_direct", "E1_plotqa_cot")
SCHEMA_VERSION = "m1m2_plotqa_guardrail_audit_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def row_by_condition(results: list[dict[str, Any]], condition: str) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in results if row.get("condition") == condition}


def direct_rows_by_sample(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_sample: dict[str, dict[str, dict[str, Any]]] = {}
    for row in results:
        condition = row.get("condition")
        if condition in DIRECT_CONDITIONS:
            by_sample.setdefault(str(row["sample_id"]), {})[str(condition)] = row
    return by_sample


def low_confidence_reasons(record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    ruler = record.get("ruler") or {}
    if ruler.get("confidence") in {"low", "unknown", None}:
        reasons.append("ruler_low_or_unknown_confidence")
    for axis in ("x_ticks", "y_ticks"):
        for idx, tick in enumerate(ruler.get(axis) or []):
            if tick.get("confidence") == "low":
                reasons.append(f"{axis}[{idx}]_low_confidence")
                break
    for idx, point in enumerate(record.get("measured_points") or []):
        if point.get("confidence") == "low":
            reasons.append(f"measured_points[{idx}]_low_confidence")
            break
        if point.get("series_confidence") == "low":
            reasons.append(f"measured_points[{idx}]_series_low_confidence")
            break
    return reasons


def has_nontrivial_bottleneck(record: dict[str, Any]) -> bool:
    labels = [item.get("label") for item in record.get("bottleneck_labels") or []]
    return any(label and label != "unknown" for label in labels)


def deterministic_solver_value(record: dict[str, Any]) -> Any:
    sample = {
        "sample_id": record.get("sample_id"),
        "question": record.get("question", ""),
        "answer_type": record.get("answer_type"),
    }
    try:
        return deterministic_prediction_from_measured(sample, record, None)
    except Exception:
        return None


def classify_sample(
    record: dict[str, Any],
    protected_row: dict[str, Any] | None,
    raw_row: dict[str, Any] | None,
    direct_rows: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    gaps = record.get("evidence_gaps") or []
    schema_violations = record.get("schema_violations") or []
    low_conf = low_confidence_reasons(record)
    deterministic_value = deterministic_solver_value(record)
    deterministic_eligible = deterministic_value is not None or bool(protected_row and protected_row.get("deterministic_m3"))
    risk_reasons = []
    if schema_violations:
        risk_reasons.append("schema_violations_present")
    if low_conf:
        risk_reasons.extend(low_conf)
    if has_nontrivial_bottleneck(record):
        risk_reasons.append("nontrivial_bottleneck_label_present")
    if protected_row and not protected_row.get("correct"):
        risk_reasons.append("protected_answer_incorrect")
    if raw_row and protected_row and raw_row.get("correct") and not protected_row.get("correct"):
        risk_reasons.append("curation_degraded_raw_correct_answer")

    if gaps:
        primary = "evidence-gap"
    elif risk_reasons:
        primary = "candidate-risk"
    elif deterministic_eligible:
        primary = "deterministic-solver-eligible"
    elif protected_row and protected_row.get("correct"):
        primary = "clean-pass-through"
    else:
        primary = "unresolved"

    direct_available = bool(direct_rows)
    direct_best_correct = None
    direct_degradation = None
    if direct_rows:
        direct_best_correct = any(bool(row.get("correct")) for row in direct_rows.values())
        if protected_row is not None:
            direct_degradation = bool(direct_best_correct and not protected_row.get("correct"))

    candidate_count = len(record.get("measured_points") or [])
    gap_count = len(gaps)
    source_payload = {
        "question": record.get("question"),
        "measured_points": record.get("measured_points") or [],
        "evidence_gaps": gaps,
        "ruler": record.get("ruler") or {},
    }
    return {
        "sample_id": record["sample_id"],
        "question_hash": stable_payload_hash({"question": record.get("question", "")}),
        "source_record_hash": stable_payload_hash(source_payload),
        "primary_classification": primary,
        "classification_flags": {
            "clean_pass_through": primary == "clean-pass-through",
            "candidate_risk": bool(risk_reasons),
            "evidence_gap": bool(gaps),
            "deterministic_solver_eligible": deterministic_eligible,
            "unresolved": primary == "unresolved",
        },
        "candidate_count": candidate_count,
        "gap_count": gap_count,
        "schema_violation_count": len(schema_violations),
        "risk_reasons": sorted(set(risk_reasons)),
        "gap_reasons": sorted({str(gap.get("reason", "unknown")) for gap in gaps if isinstance(gap, dict)}),
        "protected_default": {
            "condition": PROTECTED_CONDITION,
            "completed": bool(protected_row and protected_row.get("completed")),
            "parseable": bool(protected_row and protected_row.get("parseable")),
            "correct": bool(protected_row and protected_row.get("correct")),
            "normalized_prediction": protected_row.get("normalized_prediction") if protected_row else None,
            "deterministic_m3": bool(protected_row and protected_row.get("deterministic_m3")),
        },
        "raw_measured": {
            "condition": RAW_CONDITION,
            "correct": bool(raw_row and raw_row.get("correct")),
            "normalized_prediction": raw_row.get("normalized_prediction") if raw_row else None,
        },
        "direct_baseline": {
            "available": direct_available,
            "best_correct": direct_best_correct,
            "degradation_vs_best_direct": direct_degradation,
            "missing_reason": None if direct_available else "direct_baseline_row_unavailable",
        },
        "guarded_policy_projection": {
            "pass_through_required": primary in {"clean-pass-through", "deterministic-solver-eligible"},
            "would_change_protected_default_output": False,
            "default_replacement_allowed": False,
            "next_action": "preserve_protected_default_and_emit_diagnostic_selector_inputs",
        },
        "manual_label_used": False,
    }


def build_audit(
    measured_records_path: Path,
    protected_result_path: Path,
    diagnostic_result_path: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    records = load_json(measured_records_path)
    protected_result = load_json(protected_result_path)
    diagnostic_result = load_json(diagnostic_result_path) if diagnostic_result_path and diagnostic_result_path.exists() else None
    if not isinstance(records, list):
        raise ValueError("measured records must be a list")
    protected_by_id = row_by_condition(protected_result.get("results", []), PROTECTED_CONDITION)
    raw_by_id = row_by_condition(protected_result.get("results", []), RAW_CONDITION)
    direct_by_id = direct_rows_by_sample(diagnostic_result.get("results", [])) if diagnostic_result else {}
    rows = [
        classify_sample(record, protected_by_id.get(record["sample_id"]), raw_by_id.get(record["sample_id"]), direct_by_id.get(record["sample_id"]))
        for record in sorted(records, key=lambda row: str(row["sample_id"]))
    ]
    summary = protected_result.get("summary", {})
    protected_summary = summary.get(PROTECTED_CONDITION, {})
    class_counts = Counter(row["primary_classification"] for row in rows)
    flag_counts = Counter()
    for row in rows:
        for key, value in row["classification_flags"].items():
            if value:
                flag_counts[key] += 1
    direct_degradations = sum(1 for row in rows if row["direct_baseline"]["degradation_vs_best_direct"] is True)
    guardrail_passed = (
        protected_summary.get("correct") == 48
        and protected_summary.get("completed") == 50
        and protected_summary.get("parseable") == 50
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "p1_plotqa_guardrail_audit_zero_cost",
        "stage_status": "ready_for_schema_implementation" if guardrail_passed and len(rows) == 50 else "plotqa_guardrail_audit_failed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "strategy": {
                "path": str((ROOT / "docs" / "M1_M2_RECOVERY_STRATEGY.md").resolve()),
                "sha256": stable_file_hash(ROOT / "docs" / "M1_M2_RECOVERY_STRATEGY.md"),
            },
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
            "protected_result": {
                "path": str(protected_result_path.resolve()),
                "sha256": stable_file_hash(protected_result_path),
                "stage_status": protected_result.get("stage_status"),
            },
            "diagnostic_result": (
                {"path": str(diagnostic_result_path.resolve()), "sha256": stable_file_hash(diagnostic_result_path)}
                if diagnostic_result_path and diagnostic_result_path.exists()
                else {"path": None, "sha256": None, "missing_reason": "diagnostic_result_unavailable"}
            ),
        },
        "guardrail": {
            "protected_condition": PROTECTED_CONDITION,
            "required_correct": 48,
            "required_completed": 50,
            "required_parseable": 50,
            "observed": protected_summary,
            "passed": guardrail_passed,
            "default_output_changes_introduced_by_this_audit": 0,
        },
        "sample_count": len(rows),
        "classification_summary": {
            "primary_classification_counts": dict(sorted(class_counts.items())),
            "flag_counts": dict(sorted(flag_counts.items())),
            "direct_baseline_degradations": direct_degradations,
            "manual_labels_used": 0,
        },
        "rows": rows,
        "next_required_action": "implement_schema_only_m1m2_experimental_output" if guardrail_passed else "repair_or_restore_plotqa_guardrail_before_m1m2_changes",
        "interpretation_limits": [
            "This P1 audit performs no paid calls and does not rewrite paid results.",
            "Classification labels are deterministic heuristics for selecting the next M1/M2 schema work, not evidence that M1/M2 improves accuracy.",
            "The protected PlotQA default remains unchanged; every experimental M1/M2 arm must be opt-in.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plotqa_guardrail_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "plotqa_guardrail_audit.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    guardrail = result["guardrail"]
    lines = [
        "# M1/M2 PlotQA Guardrail Audit",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed: 0",
        f"- Protected condition: `{guardrail['protected_condition']}`",
        f"- Guardrail observed: {guardrail['observed'].get('correct')}/50 correct, {guardrail['observed'].get('completed')}/50 completed, {guardrail['observed'].get('parseable')}/50 parseable",
        f"- Guardrail passed: {guardrail['passed']}",
        f"- Sample rows: {result['sample_count']}",
        "",
        "## Classification Summary",
        "",
    ]
    for key, value in result["classification_summary"]["primary_classification_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            f"- direct baseline degradations: {result['classification_summary']['direct_baseline_degradations']}",
            f"- manual labels used: {result['classification_summary']['manual_labels_used']}",
            "",
            "## Next Required Action",
            "",
            f"- `{result['next_required_action']}`",
            "",
            "## Limits",
            "",
        ]
    )
    lines.extend(f"- {line}" for line in result["interpretation_limits"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured-records", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/m0_measurement_records.json")
    parser.add_argument("--protected-result", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/stage5_measured_flat_deterministic_rescore/rescore_result.json")
    parser.add_argument("--diagnostic-result", default="outputs/public_benchmark_pilot/plotqa_diagnostic_50/paid_result_trace_normalized.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_recovery/plotqa_guardrail_audit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = build_audit(
        Path(args.measured_records),
        Path(args.protected_result),
        Path(args.diagnostic_result) if args.diagnostic_result else None,
        Path(args.output_dir),
    )
    print(f"wrote guardrail audit: {Path(args.output_dir) / 'plotqa_guardrail_audit.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
