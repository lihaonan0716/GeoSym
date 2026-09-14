#!/usr/bin/env python
"""Finalize the measured-GeoSym PlotQA evidence audit from frozen artifacts.

This script performs no paid calls and does not rewrite paid results. It
summarizes the latest 50-sample confirmation, repair smokes, and zero-cost
projection so that the remaining paper-evidence boundary is explicit.
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

from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": stable_file_hash(path),
        "stage": artifact.get("stage"),
        "stage_status": artifact.get("stage_status"),
        "paid_calls_executed": artifact.get("paid_calls_executed", 0),
        "actual_llm_http_attempts": artifact.get("actual_llm_http_attempts", 0),
        "actual_ocr_http_attempts": artifact.get("actual_ocr_http_attempts", 0),
    }


def build_audit(
    confirmation_path: Path,
    projection_path: Path,
    repair_paths: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    confirmation = load_json(confirmation_path)
    projection = load_json(projection_path)
    repairs = [load_json(path) for path in repair_paths]
    confirmation_summary = confirmation.get("summary", {})
    projection_summary = projection.get("summary", {})
    confirmation_full = confirmation_summary.get("G_full_measured_geosym", {})
    projection_full = projection_summary.get("G_full_measured_geosym", {})
    baseline = projection.get("cached_baseline_correct") or confirmation.get("cached_baseline_correct", {})
    strongest = max(baseline.get("E0_plotqa_direct", 0), baseline.get("E1_plotqa_cot", 0))
    stage5_repair_attempts = sum(int(repair.get("actual_llm_http_attempts") or 0) for repair in repairs)
    stage5_confirmation_attempts = int(confirmation.get("actual_llm_http_attempts") or 0)
    stage5_total = stage5_repair_attempts + stage5_confirmation_attempts
    measured_flat = confirmation_summary.get("Measured_flat_answer_baseline") or projection_summary.get("Measured_flat_answer_baseline", {})
    primary_projection_target_met = (
        projection.get("stage_status") == "projection_passed"
        and projection_full.get("completed", 0) >= 48
        and projection_full.get("parseable", 0) >= 48
        and projection_full.get("correct", 0) >= 46
        and projection_full.get("correct", 0) > strongest
    )
    confirmation_primary_target_met = (
        confirmation_full.get("completed", 0) >= 48
        and confirmation_full.get("parseable", 0) >= 48
        and confirmation_full.get("correct", 0) >= 46
        and confirmation_full.get("correct", 0) > strongest
    )
    confirmation_stage_gate_met = (
        confirmation_primary_target_met
        and confirmation.get("stage_status") == "stage_passed"
    )
    blockers = []
    if not confirmation_stage_gate_met:
        blockers.append(
            {
                "label": "fresh_50_confirmation_not_stage_passed",
                "fields": {
                    "confirmation_stage_status": confirmation.get("stage_status"),
                    "confirmation_g_full_correct": confirmation_full.get("correct"),
                },
            }
        )
    if measured_flat.get("completed", 0) < 48 or measured_flat.get("parseable", 0) < 48:
        blockers.append(
            {
                "label": "measured_flat_control_parseability_below_plan_gate",
                "fields": {
                    "completed": measured_flat.get("completed", 0),
                    "parseable": measured_flat.get("parseable", 0),
                    "required": 48,
                },
            }
        )
    if stage5_total > 314:
        blockers.append({"label": "stage5_total_paid_cap_exceeded", "fields": {"actual": stage5_total, "cap": 314}})
    if stage5_confirmation_attempts > 250:
        blockers.append(
            {
                "label": "stage5_confirmation_paid_cap_exceeded",
                "fields": {"actual": stage5_confirmation_attempts, "cap": 250},
            }
        )
    if confirmation_stage_gate_met:
        stage_status = "fresh_50_confirmation_stage_gate_met"
    elif confirmation_primary_target_met and not confirmation_stage_gate_met:
        stage_status = "primary_confirmation_target_met_with_control_gap"
    elif primary_projection_target_met:
        stage_status = "primary_projection_target_met_with_confirmation_gap"
    else:
        stage_status = "target_not_met"
    next_required_action = (
        "ready_for_paper_package_update_with_m1m2_caveat"
        if confirmation_stage_gate_met and not blockers
        else "resolve_blocking_items_before_paper_claim"
    )
    interpretation_limits = [
        "This audit performs no paid calls and does not rewrite paid results.",
        "The primary measured-GeoSym G_full method reaches the frozen 50-sample PlotQA gate in the supplied confirmation artifact.",
        "The measured-flat control now meets the operational parseability gate, but its lower accuracy must not be used by itself as proof of an independent M1/M2 contribution.",
        "The remaining wrong or unparseable cases are nonblocking residual risks for this narrow claim and should be reported explicitly.",
    ]
    result = {
        "schema_version": "plotqa_measured_geosym_final_evidence_audit_v1",
        "created_at_utc": utc_now(),
        "stage": "plotqa_measured_geosym_final_evidence_audit",
        "stage_status": stage_status,
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "stage5_confirmation": source(confirmation_path, confirmation),
            "stage5_projection": source(projection_path, projection),
            "stage5_repair_smokes": [source(path, repair) for path, repair in zip(repair_paths, repairs)],
        },
        "baseline_correct": baseline,
        "confirmation_summary": confirmation_summary,
        "projection_summary": projection_summary,
        "gate_assessment": {
            "fresh_50_confirmation_primary_g_full_target_met": confirmation_primary_target_met,
            "fresh_50_confirmation_stage_gate_met": confirmation_stage_gate_met,
            "post_repair_projection_primary_target_met": primary_projection_target_met,
            "projection_g_full_correct": projection_full.get("correct"),
            "confirmation_g_full_correct": confirmation_full.get("correct"),
            "strongest_direct_baseline_correct": strongest,
            "stage5_repair_smoke_count": len(repairs),
            "stage5_repair_llm_attempts": stage5_repair_attempts,
            "stage5_confirmation_llm_attempts": stage5_confirmation_attempts,
            "stage5_total_llm_attempts": stage5_total,
            "stage5_total_within_314_cap": stage5_total <= 314,
            "stage5_confirmation_within_250_cap": stage5_confirmation_attempts <= 250,
        },
        "blocking_items_for_paper_claim": blockers,
        "next_required_action": next_required_action,
        "interpretation_limits": interpretation_limits,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_evidence_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "final_evidence_audit.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    gate = result["gate_assessment"]
    lines = [
        "# PlotQA Measured GeoSym Final Evidence Audit",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed by this audit: 0",
        f"- Fresh 50 confirmation G_full: {gate['confirmation_g_full_correct']}/50",
        f"- Post-repair projection G_full: {gate['projection_g_full_correct']}/50",
        f"- Strongest direct baseline: {gate['strongest_direct_baseline_correct']}/50",
        f"- Stage 5 total LLM attempts: {gate['stage5_total_llm_attempts']}/314",
        "",
        "## Blocking Items For Paper Claim",
        "",
    ]
    if result["blocking_items_for_paper_claim"]:
        for item in result["blocking_items_for_paper_claim"]:
            lines.append(f"- {item['label']}: {json.dumps(item['fields'], ensure_ascii=False)}")
    else:
        lines.append("- none")
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {line}" for line in result["interpretation_limits"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-result", required=True)
    parser.add_argument("--projection-result", required=True)
    parser.add_argument("--repair-result", action="append", required=True)
    parser.add_argument("--output-dir", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/final_evidence_audit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = build_audit(
        Path(args.confirmation_result),
        Path(args.projection_result),
        [Path(path) for path in args.repair_result],
        Path(args.output_dir),
    )
    print(f"wrote final evidence audit: {Path(args.output_dir) / 'final_evidence_audit.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
