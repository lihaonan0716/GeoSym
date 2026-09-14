#!/usr/bin/env python
"""Summarize remaining PlotQA measured-GeoSym bottlenecks without paid calls."""

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


FULL_CONDITION = "G_full_measured_geosym"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_full_wrong(row: dict[str, Any], repair_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sample_id = row["sample_id"]
    repair = repair_by_id.get(sample_id)
    if repair and repair.get("correct") is True:
        label = "projection_resolved_not_fresh_confirmed"
    elif "minimum" in str(row.get("raw_text", "")).lower() or row.get("normalized_prediction") == 0:
        label = "m0_minimum_or_axis_scale_gap"
    elif "difference" in str(row.get("raw_text", "")).lower() or isinstance(row.get("normalized_prediction"), (int, float)):
        label = "m0_grouped_series_alignment_or_signed_difference_gap"
    else:
        label = "m3_reasoning_under_incomplete_measured_evidence"
    return {
        "sample_id": sample_id,
        "failure_label": label,
        "prediction": row.get("normalized_prediction"),
        "reference": row.get("reference"),
        "numeric_error": row.get("numeric_error"),
        "repair_smoke3_correct": repair.get("correct") if repair else None,
        "repair_smoke3_prediction": repair.get("normalized_prediction") if repair else None,
    }


def build_audit(
    confirmation_path: Path,
    projection_path: Path,
    repair_smoke3_path: Path,
    flat_parseability_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    confirmation = load_json(confirmation_path)
    projection = load_json(projection_path)
    repair3 = load_json(repair_smoke3_path)
    flat = load_json(flat_parseability_path)
    repair_by_id = {row["sample_id"]: row for row in repair3.get("results", []) if row.get("condition") == FULL_CONDITION}
    full_wrong = [
        classify_full_wrong(row, repair_by_id)
        for row in confirmation.get("results", [])
        if row.get("condition") == FULL_CONDITION and not row.get("correct")
    ]
    flat_failures = flat.get("failures", [])
    full_labels = sorted({item["failure_label"] for item in full_wrong})
    flat_labels = sorted({item["failure_label"] for item in flat_failures})
    flat_summary = flat.get("summary", {})
    stage_gate_met = (
        confirmation.get("stage_status") == "stage_passed"
        and confirmation.get("summary", {}).get(FULL_CONDITION, {}).get("correct", 0) >= 46
        and flat_summary.get("completed", 0) >= 48
        and flat_summary.get("parseable", 0) >= 48
    )
    result = {
        "schema_version": "plotqa_remaining_bottleneck_audit_v1",
        "created_at_utc": utc_now(),
        "stage": "plotqa_remaining_bottleneck_audit",
        "stage_status": "stage_gate_met_with_nonblocking_bottlenecks" if stage_gate_met else "remaining_bottlenecks_require_m0_repair_or_projection_only_claim",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "confirmation": {
                "path": str(confirmation_path.resolve()),
                "sha256": stable_file_hash(confirmation_path),
                "stage_status": confirmation.get("stage_status"),
            },
            "projection": {
                "path": str(projection_path.resolve()),
                "sha256": stable_file_hash(projection_path),
                "stage_status": projection.get("stage_status"),
            },
            "repair_smoke3": {
                "path": str(repair_smoke3_path.resolve()),
                "sha256": stable_file_hash(repair_smoke3_path),
                "stage_status": repair3.get("stage_status"),
            },
            "flat_parseability": {
                "path": str(flat_parseability_path.resolve()),
                "sha256": stable_file_hash(flat_parseability_path),
                "stage_status": flat.get("stage_status"),
            },
        },
        "summary": {
            "fresh_g_full_wrong": len(full_wrong),
            "projection_g_full_correct": projection.get("summary", {}).get(FULL_CONDITION, {}).get("correct"),
            "repair_smoke3_correct": repair3.get("summary", {}).get("correct"),
            "repair_smoke3_previous_correct": repair3.get("summary", {}).get("previous_stage4_correct"),
            "flat_parseability_failures": len(flat_failures),
            "full_failure_labels": full_labels,
            "flat_failure_labels": flat_labels,
        },
        "fresh_g_full_wrong_cases": full_wrong,
        "flat_parseability_failures": flat_failures,
        "recommended_next_direction": "report_gate_met_and_continue_optional_m0_repair" if stage_gate_met else (
            "m0_axis_and_series_repair_before_more_50_sample_confirmation"
            if flat_failures or any("m0_" in label for label in full_labels)
            else "accept_projection_as_diagnostic_only"
        ),
        "stop_or_pivot_signal": (
            "Do not run another 50-sample confirmation until a zero-cost M0 repair changes the affected measured evidence or a separately frozen specialized-chart-model baseline is selected."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "remaining_bottleneck_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "remaining_bottleneck_audit.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# PlotQA Remaining Bottleneck Audit",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed: 0",
        f"- Fresh G_full wrong: {summary['fresh_g_full_wrong']}",
        f"- Projection G_full correct: {summary['projection_g_full_correct']}/50",
        f"- Repair smoke3: {summary['repair_smoke3_correct']}/{summary['repair_smoke3_previous_correct']} on same samples",
        f"- Flat parseability failures: {summary['flat_parseability_failures']}",
        f"- Recommended next direction: `{result['recommended_next_direction']}`",
        "",
        "## Fresh G_full Wrong",
        "",
    ]
    for item in result["fresh_g_full_wrong_cases"]:
        lines.append(f"- {item['sample_id']}: `{item['failure_label']}`, pred={item['prediction']}, ref={item['reference']}")
    lines.extend(["", "## Stop Or Pivot Signal", "", result["stop_or_pivot_signal"], ""])
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-result", required=True)
    parser.add_argument("--projection-result", required=True)
    parser.add_argument("--repair-smoke3-result", required=True)
    parser.add_argument("--flat-parseability-audit", required=True)
    parser.add_argument("--output-dir", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/final_evidence_audit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = build_audit(
        Path(args.confirmation_result),
        Path(args.projection_result),
        Path(args.repair_smoke3_result),
        Path(args.flat_parseability_audit),
        Path(args.output_dir),
    )
    print(f"wrote remaining bottleneck audit: {Path(args.output_dir) / 'remaining_bottleneck_audit.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
