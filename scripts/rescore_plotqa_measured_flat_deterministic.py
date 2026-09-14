#!/usr/bin/env python
"""Zero-cost deterministic rescore for measured-flat PlotQA abstentions."""

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
from scripts.run_plotqa_locate import deterministic_prediction_from_measured  # noqa: E402
from scripts.run_plotqa_measure import analyze_result, as_records_by_id, load_json  # noqa: E402
from scripts.run_public_benchmark_pilot import score_prediction  # noqa: E402


CONDITION = "Measured_flat_answer_baseline"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_by_id(diagnostic_plan_path: Path) -> dict[str, dict[str, Any]]:
    plan = load_json(diagnostic_plan_path)
    return {sample["sample_id"]: sample for sample in plan.get("selected_samples", [])}


def rescore(
    confirmation_path: Path,
    diagnostic_plan_path: Path,
    measured_records_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    confirmation = load_json(confirmation_path)
    plan_path = Path(confirmation["plan_path"])
    plan = load_json(plan_path)
    samples = sample_by_id(diagnostic_plan_path)
    records = as_records_by_id(measured_records_path)
    rescored_rows = []
    changes = []
    for row in confirmation.get("results", []):
        updated = dict(row)
        if row.get("condition") == CONDITION and not row.get("parseable") and row.get("sample_id") in samples and row.get("sample_id") in records:
            prediction = deterministic_prediction_from_measured(samples[row["sample_id"]], records[row["sample_id"]], None)
            if prediction is not None:
                score = score_prediction(prediction, row.get("reference"), row.get("answer_type"))
                updated.update(
                    {
                        "completed": True,
                        "parseable": True,
                        "normalized_prediction": prediction,
                        "raw_text": f"deterministic_measured_flat_prediction={prediction}",
                        "error": "",
                        "llm_http_attempts": 0,
                        "deterministic_measured_flat_rescore": True,
                        **score,
                    }
                )
                changes.append(
                    {
                        "sample_id": row["sample_id"],
                        "previous_completed": row.get("completed"),
                        "previous_parseable": row.get("parseable"),
                        "new_prediction": prediction,
                        "new_correct": score["correct"],
                    }
                )
        rescored_rows.append(updated)
    result = analyze_result(plan, plan_path, rescored_rows, int(confirmation.get("actual_llm_http_attempts") or 0))
    result.update(
        {
            "schema_version": "plotqa_measured_flat_deterministic_rescore_v1",
            "created_at_utc": utc_now(),
            "stage": "stage5_measured_flat_deterministic_rescore",
            "rescore_paid_calls_executed": 0,
            "rescore_actual_llm_http_attempts": 0,
            "rescore_actual_ocr_http_attempts": 0,
            "source_artifacts": {
                "confirmation": {
                    "path": str(confirmation_path.resolve()),
                    "sha256": stable_file_hash(confirmation_path),
                    "stage_status": confirmation.get("stage_status"),
                },
                "diagnostic_plan": {
                    "path": str(diagnostic_plan_path.resolve()),
                    "sha256": stable_file_hash(diagnostic_plan_path),
                },
                "measured_records": {
                    "path": str(measured_records_path.resolve()),
                    "sha256": stable_file_hash(measured_records_path),
                },
            },
            "rescore_policy": "Only non-parseable measured-flat rows are filled, and only when the deterministic measured-facts solver returns a value.",
            "deterministic_changes": changes,
            "interpretation_limits": [
                "This is a zero-cost deterministic rescore of already frozen paid outputs.",
                "The rescore does not overwrite parseable paid rows and does not rerun any paid model call.",
                "The deterministic solver uses question text plus measured facts, not reference answers, PlotQA data tables, or sample IDs.",
                "This supports operational parseability of the measured-flat control; it is not independent evidence that M1/M2 alone improve accuracy.",
            ],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rescore_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "rescore_audit.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary", {})
    flat = summary.get(CONDITION, {})
    full = summary.get("G_full_measured_geosym", {})
    lines = [
        "# PlotQA Measured-Flat Deterministic Rescore",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed by rescore: 0",
        f"- G_full correct after rescore: {full.get('correct')}/50",
        f"- Measured-flat parseable after rescore: {flat.get('parseable')}/50",
        f"- Deterministic measured-flat changes: {len(result['deterministic_changes'])}",
        "",
        "## Changes",
        "",
    ]
    for item in result["deterministic_changes"]:
        lines.append(f"- {item['sample_id']}: {item['new_prediction']} ({item['new_correct']})")
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {line}" for line in result["interpretation_limits"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-result", required=True)
    parser.add_argument("--diagnostic-plan", required=True)
    parser.add_argument("--measured-records", required=True)
    parser.add_argument("--output-dir", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/stage5_measured_flat_deterministic_rescore")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = rescore(Path(args.confirmation_result), Path(args.diagnostic_plan), Path(args.measured_records), Path(args.output_dir))
    print(f"wrote measured-flat deterministic rescore: {Path(args.output_dir) / 'rescore_result.json'}")
    print(f"stage_status={result['stage_status']}")
    print("rescore_paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
