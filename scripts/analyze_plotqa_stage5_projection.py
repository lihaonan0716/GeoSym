#!/usr/bin/env python
"""Zero-cost projection audit for Stage 5 PlotQA repairs.

This script never calls paid services. It projects a repair-smoke result onto a
previous 50-sample confirmation only for sample-condition rows actually rerun in
the repair smoke, and labels the output as a projection rather than a new
confirmation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402


CONDITION = "G_full_measured_geosym"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_summary(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_condition = defaultdict(list)
    for row in results:
        by_condition[row["condition"]].append(row)
    return {
        condition: {
            "completed": sum(1 for row in rows if row.get("completed")),
            "parseable": sum(1 for row in rows if row.get("parseable")),
            "correct": sum(1 for row in rows if row.get("correct")),
            "deterministic": sum(1 for row in rows if row.get("deterministic_m3") or row.get("deterministic_m3_rescore")),
        }
        for condition, rows in by_condition.items()
    }


def project_results(confirmation: dict[str, Any], repair: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repair_by_id = {row["sample_id"]: row for row in repair.get("results", []) if row.get("condition") == CONDITION}
    projected = []
    replacements = []
    for row in confirmation.get("results", []):
        if row.get("condition") != CONDITION or row.get("sample_id") not in repair_by_id:
            projected.append(row)
            continue
        repair_row = repair_by_id[row["sample_id"]]
        updated = dict(row)
        for key in (
            "completed",
            "parseable",
            "normalized_prediction",
            "raw_text",
            "error",
            "correct",
            "numeric_error",
            "tolerance",
            "deterministic_m3",
        ):
            if key in repair_row:
                updated[key] = repair_row[key]
        updated["projection_source"] = {
            "source": "stage5_repair_smoke",
            "previous_correct": row.get("correct"),
            "repair_correct": repair_row.get("correct"),
        }
        projected.append(updated)
        replacements.append(
            {
                "sample_id": row["sample_id"],
                "previous_prediction": row.get("normalized_prediction"),
                "previous_correct": row.get("correct"),
                "repair_prediction": repair_row.get("normalized_prediction"),
                "repair_correct": repair_row.get("correct"),
            }
        )
    return projected, replacements


def analyze(confirmation_path: Path, repair_path: Path, output_dir: Path) -> dict[str, Any]:
    confirmation = load_json(confirmation_path)
    repair = load_json(repair_path)
    projected, replacements = project_results(confirmation, repair)
    summary = condition_summary(projected)
    full = summary.get(CONDITION, {})
    flat = summary.get("Measured_flat_answer_baseline", {})
    baseline = confirmation.get("cached_baseline_correct", {})
    strongest = max(baseline.get("E0_plotqa_direct", 0), baseline.get("E1_plotqa_cot", 0))
    full_operational = full.get("completed", 0) >= 48 and full.get("parseable", 0) >= 48
    control_operational = flat.get("completed", 0) >= 48 and flat.get("parseable", 0) >= 48
    benefit = full.get("correct", 0) > strongest and full.get("correct", 0) >= 46
    result = {
        "schema_version": "plotqa_stage5_post_repair_projection_v1",
        "created_at_utc": utc_now(),
        "stage": "stage5_post_repair_projection_zero_cost",
        "stage_status": "projection_passed" if full_operational and benefit else "projection_failed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "confirmation": {"path": str(confirmation_path.resolve()), "sha256": stable_file_hash(confirmation_path), "stage_status": confirmation.get("stage_status")},
            "repair_smoke": {"path": str(repair_path.resolve()), "sha256": stable_file_hash(repair_path), "stage_status": repair.get("stage_status")},
        },
        "summary": summary,
        "cached_baseline_correct": baseline,
        "replacements": replacements,
        "go_gate": {
            "g_full_completed_parseable_at_least_48_of_50_projected": full_operational,
            "g_full_exceeds_e0_and_e1_projected": full.get("correct", 0) > strongest,
            "g_full_correct_at_least_46_of_50_projected": full.get("correct", 0) >= 46,
            "measured_flat_completed_parseable_at_least_48_of_50_projected": control_operational,
            "projection_uses_only_repair_rerun_rows": True,
            "paid_calls_zero": True,
        },
        "nonblocking_caveats": [
            {
                "label": "measured_flat_control_parseability_below_stage4_operational_gate",
                "fields": {"completed": flat.get("completed", 0), "parseable": flat.get("parseable", 0), "required": 48},
            }
        ] if not control_operational else [],
        "interpretation_limits": [
            "This is a zero-cost projection, not a fresh 50-sample confirmation.",
            "Only sample-condition rows actually rerun in the repair smoke are replaced.",
            "The primary G_full method gate is reported separately from the measured-flat diagnostic control gate.",
            "A confirmatory paper claim still requires either accepting this as post-repair projection or freezing a new paid plan.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "projection_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "projection_audit.md").write_text(render_audit(result), encoding="utf-8")
    return result


def render_audit(result: dict[str, Any]) -> str:
    lines = ["# PlotQA Stage 5 Post-Repair Projection", "", f"- Status: `{result['stage_status']}`", "- Paid calls executed: 0", "", "## Summary", ""]
    for condition, row in result["summary"].items():
        lines.append(f"- {condition}: correct {row['correct']}/50, completed {row['completed']}/50, parseable {row['parseable']}/50")
    lines.extend(["", "## Replacements", ""])
    for row in result["replacements"]:
        lines.append(f"- {row['sample_id']}: {row['previous_prediction']} ({row['previous_correct']}) -> {row['repair_prediction']} ({row['repair_correct']})")
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {line}" for line in result["interpretation_limits"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-result", required=True)
    parser.add_argument("--repair-result", required=True)
    parser.add_argument("--output-dir", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/stage5_projection_after_repair2")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = analyze(Path(args.confirmation_result), Path(args.repair_result), Path(args.output_dir))
    print(f"wrote projection result: {Path(args.output_dir) / 'projection_result.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
