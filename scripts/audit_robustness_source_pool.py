#!/usr/bin/env python
"""Audit paper-level M1/M2 stress source-pool feasibility.

This zero-cost audit reads measured PlotQA records and paid outputs, then
checks whether a 128-job or 96-job candidate-guided stress roster can be
constructed without manual labels or adaptive replacement.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_plotqa_public_benchmark_pilot import infer_reasoning_type  # noqa: E402
from scripts.prepare_public_benchmark_pilot import stable_file_hash, stable_payload_hash  # noqa: E402
from scripts.run_plotqa_locate import deterministic_prediction_from_measured  # noqa: E402
from scripts.run_plotqa_measured_locate_smoke import (  # noqa: E402
    label_matches_question,
    normalize_text_answer,
    selected_record_points_for_question,
    value_for_label_year,
)

DEFAULT_RECORDS = ROOT / "outputs" / "paper_level_evidence_package" / "plotqa_public_500" / "m0_measurement_records.json"
DEFAULT_RESULT = ROOT / "outputs" / "paper_level_evidence_package" / "plotqa_public_500" / "measured_confirmation" / "paid_result.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "paper_level_evidence_package" / "m1m2_stress_128"
SUPPORTED_OPERATIONS = {"lookup", "threshold_count", "comparison", "extrema", "range_difference", "ratio"}
PRIMARY_TARGET = 128
FALLBACK_TARGET = 96
MAX_SOURCE_REUSE = 2
SEED = "20260831_track_d_source_pool"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def operation_tag(question: str) -> str:
    q = question.lower()
    if "how many" in q and any(cue in q for cue in ("greater than", "more than", "less than", "fewer than", "above", "below", "at least", "at most")):
        return "threshold_count"
    tag = infer_reasoning_type(question, "", "")
    if tag == "range":
        return "range_difference"
    if tag == "slope":
        return "comparison"
    return tag


def rows_by_condition(result: dict[str, Any], condition: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in result.get("results", []) if row.get("condition") == condition}


def deterministic_value(record: dict[str, Any]) -> Any:
    sample = {
        "sample_id": record.get("sample_id"),
        "question": record.get("question", ""),
        "answer_type": record.get("answer_type"),
    }
    try:
        return deterministic_prediction_from_measured(sample, record, None)
    except Exception:
        return None


def question_matched_points(record: dict[str, Any]) -> list[dict[str, Any]]:
    q = normalize_text_answer(record.get("question", ""))
    matched = []
    for point in record.get("measured_points") or []:
        labels = (
            normalize_text_answer(point.get("series_label")),
            normalize_text_answer(point.get("category_label")),
        )
        if any(label_matches_question(label, q) for label in labels):
            matched.append(point)
    return matched


def ratio_support_resolved(record: dict[str, Any]) -> bool:
    q = normalize_text_answer(record.get("question", ""))
    years = [int(match.group(0)) for match in re.finditer(r"\b(?:18|19|20|21)\d{2}\b", q)]
    labels = sorted(
        {
            normalize_text_answer(point.get("series_label"))
            for point in record.get("measured_points") or []
            if normalize_text_answer(point.get("series_label"))
            and label_matches_question(normalize_text_answer(point.get("series_label")), q)
        }
    )
    if labels and len(years) >= 2:
        return (
            value_for_label_year(record.get("measured_points") or [], labels[0], years[0]) is not None
            and value_for_label_year(record.get("measured_points") or [], labels[0], years[1]) is not None
        )
    selected = selected_record_points_for_question(record, {"question": record.get("question", ""), "answer_type": record.get("answer_type")})
    numeric_values = [point for point in selected if isinstance(point.get("y_value"), (int, float))]
    return len(numeric_values) >= 2


def support_schema_resolved(record: dict[str, Any], tag: str, deterministic: Any) -> bool:
    if deterministic is not None:
        return True
    if tag == "lookup":
        return bool(question_matched_points(record))
    if tag == "ratio":
        return ratio_support_resolved(record)
    return False


def measured_fact_hash(record: dict[str, Any]) -> str:
    return stable_payload_hash(
        {
            "measured_points": record.get("measured_points") or [],
            "evidence_gaps": record.get("evidence_gaps") or [],
            "ruler": record.get("ruler") or {},
            "series": record.get("series") or [],
        }
    )


def classify_source(record: dict[str, Any], full_row: dict[str, Any] | None, raw_row: dict[str, Any] | None, source_hash: str) -> dict[str, Any]:
    tag = operation_tag(str(record.get("question", "")))
    schema_valid = not record.get("schema_violations")
    has_facts = bool(record.get("measured_points"))
    numeric = record.get("answer_type") == "numeric"
    value = deterministic_value(record)
    support_resolved = support_schema_resolved(record, tag, value)
    support_schema_status = "resolved" if tag in SUPPORTED_OPERATIONS and schema_valid and has_facts and numeric and support_resolved else "unresolved"
    source_key = stable_payload_hash(
        {
            "seed": SEED,
            "source_artifact_hash": source_hash,
            "source_record_id": record.get("sample_id"),
            "question_sha256": stable_payload_hash({"question": record.get("question", "")}),
            "measured_fact_hash": measured_fact_hash(record),
        }
    )
    return {
        "sample_id": record.get("sample_id"),
        "stress_source_key": source_key,
        "operation_tag": tag,
        "answer_type": record.get("answer_type"),
        "chart_type": ((record.get("chart_geometry") or {}).get("chart_type")),
        "schema_valid": schema_valid,
        "measured_fact_count": len(record.get("measured_points") or []),
        "evidence_gap_count": len(record.get("evidence_gaps") or []),
        "support_schema_status": support_schema_status,
        "answerable_source_eligible": support_schema_status == "resolved",
        "non_answerable_source_eligible": schema_valid and has_facts and tag in SUPPORTED_OPERATIONS,
        "deterministic_value_available": value is not None,
        "support_fact_match_available": support_resolved,
        "protected_completed": bool(full_row and full_row.get("completed")),
        "protected_parseable": bool(full_row and full_row.get("parseable")),
        "protected_correct": bool(full_row and full_row.get("correct")),
        "raw_correct": bool(raw_row and raw_row.get("correct")),
    }


def quota_report(rows: list[dict[str, Any]], target: int) -> dict[str, Any]:
    answerable_target = target // 2
    non_answerable_target = target - answerable_target
    answerable = [row for row in rows if row["answerable_source_eligible"]]
    non_answerable = [row for row in rows if row["non_answerable_source_eligible"]]
    answerable_ops = Counter(row["operation_tag"] for row in answerable)
    non_answerable_capacity = len(non_answerable) * MAX_SOURCE_REUSE
    answerable_capacity = len(answerable) * MAX_SOURCE_REUSE
    required_answerable = {
        "lookup": 10 if target == PRIMARY_TARGET else 8,
        "threshold_count": 10 if target == PRIMARY_TARGET else 8,
        "comparison": 10 if target == PRIMARY_TARGET else 8,
        "extrema": 10 if target == PRIMARY_TARGET else 8,
        "range_difference": 6 if target == PRIMARY_TARGET else 4,
        "ratio": 6 if target == PRIMARY_TARGET else 4,
    }
    missing_operation_quota = {
        op: {"required": required, "available": answerable_ops.get(op, 0)}
        for op, required in required_answerable.items()
        if answerable_ops.get(op, 0) < required
    }
    passed = (
        answerable_capacity >= answerable_target
        and non_answerable_capacity >= non_answerable_target
        and not missing_operation_quota
    )
    return {
        "target_sample_level_jobs": target,
        "answerable_target": answerable_target,
        "non_answerable_target": non_answerable_target,
        "answerable_unique_sources": len(answerable),
        "non_answerable_unique_sources": len(non_answerable),
        "answerable_capacity_with_reuse_cap_2": answerable_capacity,
        "non_answerable_capacity_with_reuse_cap_2": non_answerable_capacity,
        "answerable_operation_counts": dict(sorted(answerable_ops.items())),
        "missing_answerable_operation_quota": missing_operation_quota,
        "passed": passed,
    }


def build_audit(records_path: Path, result_path: Path, output_dir: Path) -> dict[str, Any]:
    records = load_json(records_path)
    result = load_json(result_path)
    if not isinstance(records, list):
        raise ValueError("measured records must be a JSON list")
    full_rows = rows_by_condition(result, "G_full_measured_geosym")
    raw_rows = rows_by_condition(result, "G_measured_raw")
    source_hash = stable_file_hash(records_path)
    rows = [
        classify_source(record, full_rows.get(str(record.get("sample_id"))), raw_rows.get(str(record.get("sample_id"))), source_hash)
        for record in sorted(records, key=lambda row: str(row.get("sample_id")))
    ]
    primary = quota_report(rows, PRIMARY_TARGET)
    fallback = quota_report(rows, FALLBACK_TARGET)
    blocking = []
    if result.get("stage_status") not in {"track_c_plotqa_500_confirmation_completed", "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim"}:
        blocking.append(f"track_c_measured_confirmation_not_completed:{result.get('stage_status')}")
    if not primary["passed"]:
        blocking.append("stress_128_source_pool_quota_failed")
    if not fallback["passed"]:
        blocking.append("stress_96_fallback_source_pool_quota_failed")
    if primary["passed"]:
        status = "m1m2_stress_128_source_pool_feasible"
        next_action = "freeze_m1m2_stress_128_manifest"
    elif fallback["passed"]:
        status = "m1m2_stress_96_fallback_source_pool_feasible"
        next_action = "freeze_m1m2_stress_96_fallback_manifest"
    else:
        status = "m1m2_stress_source_pool_insufficient"
        next_action = "do_not_execute_track_d_expanded_paid_confirmation_report_v31_narrow_confirmation_only"
    audit = {
        "schema_version": "m1m2_stress_128_source_pool_feasibility_v1",
        "created_at_utc": utc_now(),
        "stage": "track_d_m1m2_stress_source_pool_feasibility",
        "stage_status": status,
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "seed": SEED,
        "source_artifacts": {
            "measured_records": {"path": str(records_path.resolve()), "sha256": source_hash},
            "measured_confirmation_result": {
                "path": str(result_path.resolve()),
                "sha256": stable_file_hash(result_path),
                "stage_status": result.get("stage_status"),
            },
        },
        "record_count": len(rows),
        "operation_counts": dict(sorted(Counter(row["operation_tag"] for row in rows).items())),
        "chart_type_counts": dict(sorted(Counter(str(row["chart_type"]) for row in rows).items())),
        "support_schema_counts": dict(sorted(Counter(row["support_schema_status"] for row in rows).items())),
        "primary_128": primary,
        "fallback_96": fallback,
        "blocking_findings": blocking,
        "source_rows": rows,
        "next_required_action": next_action,
        "interpretation_limits": [
            "This is a zero-cost source-pool feasibility audit, not a frozen paid roster.",
            "Prior outcome fields are used only to identify stress-test source suitability, not to estimate population performance.",
            "No chart images, reference answers, or sample IDs may be exposed as semantic hints in a future paid stress prompt.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_pool_feasibility.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "source_pool_feasibility.md").write_text(render_markdown(audit), encoding="utf-8")
    return audit


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Stress Source-Pool Feasibility",
        "",
        f"- Stage status: `{audit['stage_status']}`",
        f"- Records audited: `{audit['record_count']}`",
        f"- 128 target passed: `{audit['primary_128']['passed']}`",
        f"- 96 fallback passed: `{audit['fallback_96']['passed']}`",
        f"- Blocking findings: `{len(audit['blocking_findings'])}`",
        f"- Paid calls executed: `{audit['paid_calls_executed']}`",
        "",
        "## Operation Counts",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in audit["operation_counts"].items())
    lines.extend(["", "## Next Required Action", "", f"- `{audit['next_required_action']}`", ""])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured-records", default=str(DEFAULT_RECORDS))
    parser.add_argument("--measured-result", default=str(DEFAULT_RESULT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(Path(args.measured_records), Path(args.measured_result), Path(args.output_dir))
    print(f"wrote source-pool feasibility: {Path(args.output_dir) / 'source_pool_feasibility.json'}")
    print(f"stage_status={audit['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
