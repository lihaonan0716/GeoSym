#!/usr/bin/env python
"""Audit measured-flat parseability failures without paid calls."""

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


CONDITION = "Measured_flat_answer_baseline"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    rows = data if isinstance(data, list) else data.get("records", [])
    return {row["sample_id"]: row for row in rows}


def classify_failure(row: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    m0 = record.get("m0") or record
    gaps = m0.get("evidence_gaps") or []
    target_candidates = record.get("target_candidates") or []
    measured_points = m0.get("measured_points") or []
    y_values = [point.get("y_value") for point in measured_points if point.get("y_value") is not None]
    x_values = [point.get("x_value") for point in measured_points if point.get("x_value") is not None]
    raw = str(row.get("raw_text") or "").lower()
    if any(gap.get("severity") == "blocking" for gap in gaps):
        label = "extractor_gap_blocking_evidence_gap"
    elif not target_candidates:
        label = "extractor_gap_no_target_candidates"
    elif not measured_points:
        label = "extractor_gap_no_measured_points"
    elif not y_values and x_values:
        label = "extractor_gap_missing_numeric_axis_values"
    elif "cannot determine" in raw or "cannot be determined" in raw:
        label = "flat_contract_honest_abstention"
    else:
        label = "flat_parse_failure_unclassified"
    return {
        "sample_id": row["sample_id"],
        "failure_label": label,
        "completed": bool(row.get("completed")),
        "parseable": bool(row.get("parseable")),
        "error": row.get("error") or "",
        "raw_text_prefix": str(row.get("raw_text") or "")[:240],
        "target_candidate_count": len(target_candidates),
        "measured_point_count": len(measured_points),
        "measured_points_with_y_value": len(y_values),
        "measured_points_with_x_value": len(x_values),
        "ruler_confidence": (m0.get("ruler") or {}).get("confidence"),
        "blocking_gap_count": sum(1 for gap in gaps if gap.get("severity") == "blocking"),
        "gap_reasons": sorted({str(gap.get("reason")) for gap in gaps if gap.get("reason")}),
    }


def build_audit(confirmation_path: Path, measured_records_path: Path, output_dir: Path) -> dict[str, Any]:
    confirmation = load_json(confirmation_path)
    records = records_by_id(measured_records_path)
    flat_rows = [row for row in confirmation.get("results", []) if row.get("condition") == CONDITION]
    failing_rows = [row for row in flat_rows if not row.get("completed") or not row.get("parseable")]
    failures = [classify_failure(row, records.get(row["sample_id"], {})) for row in failing_rows]
    extractor_gap_count = sum(1 for item in failures if item["failure_label"].startswith("extractor_gap_"))
    result = {
        "schema_version": "plotqa_measured_flat_parseability_audit_v1",
        "created_at_utc": utc_now(),
        "stage": "plotqa_measured_flat_parseability_audit",
        "stage_status": "measured_flat_parseability_gap_classified" if failures else "measured_flat_parseability_passed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "confirmation": {
                "path": str(confirmation_path.resolve()),
                "sha256": stable_file_hash(confirmation_path),
                "stage_status": confirmation.get("stage_status"),
            },
            "measured_records": {
                "path": str(measured_records_path.resolve()),
                "sha256": stable_file_hash(measured_records_path),
            },
        },
        "summary": {
            "flat_rows": len(flat_rows),
            "completed": sum(1 for row in flat_rows if row.get("completed")),
            "parseable": sum(1 for row in flat_rows if row.get("parseable")),
            "failures": len(failures),
            "extractor_gap_failures": extractor_gap_count,
            "contract_only_failures": sum(1 for item in failures if item["failure_label"] == "flat_contract_honest_abstention"),
        },
        "failures": failures,
        "interpretation": (
            "Measured-flat parseability is limited by incomplete measured evidence, not by a safe JSON syntax repair alone."
            if extractor_gap_count
            else "No extractor-gap parseability failures were found."
        ),
        "claim_boundary": "Do not use measured-flat under-parseability as evidence for a reliability-aware fusion gain.",
        "next_action": "treat_as_extractor_gap_or_freeze_new_m0_repair_plan",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "flat_parseability_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "flat_parseability_audit.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# PlotQA Measured-Flat Parseability Audit",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed: 0",
        f"- Completed: {summary['completed']}/{summary['flat_rows']}",
        f"- Parseable: {summary['parseable']}/{summary['flat_rows']}",
        f"- Failures: {summary['failures']}",
        f"- Extractor-gap failures: {summary['extractor_gap_failures']}",
        "",
        "## Failures",
        "",
    ]
    for item in result["failures"]:
        lines.append(
            f"- {item['sample_id']}: `{item['failure_label']}`, "
            f"target_candidates={item['target_candidate_count']}, "
            f"measured_points={item['measured_point_count']}, "
            f"y_values={item['measured_points_with_y_value']}, "
            f"gaps={item['gap_reasons']}"
        )
    lines.extend(["", "## Claim Boundary", "", result["claim_boundary"], ""])
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-result", required=True)
    parser.add_argument("--measured-records", required=True)
    parser.add_argument("--output-dir", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/final_evidence_audit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = build_audit(Path(args.confirmation_result), Path(args.measured_records), Path(args.output_dir))
    print(f"wrote flat parseability audit: {Path(args.output_dir) / 'flat_parseability_audit.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
