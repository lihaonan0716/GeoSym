#!/usr/bin/env python
"""Zero-cost rescore for signed-difference response normalization."""

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
from scripts.run_plotqa_locate import corrected_numeric_prediction_from_response  # noqa: E402
from scripts.run_plotqa_measure import analyze_result, as_records_by_id, load_json  # noqa: E402
from scripts.run_public_benchmark_pilot import score_prediction  # noqa: E402


CONDITION = "G_full_measured_geosym"


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
        if row.get("condition") == CONDITION and row.get("sample_id") in samples and row.get("sample_id") in records:
            new_prediction = corrected_numeric_prediction_from_response(
                samples[row["sample_id"]],
                records[row["sample_id"]],
                row.get("raw_text") or "",
                row.get("normalized_prediction"),
            )
            if new_prediction != row.get("normalized_prediction"):
                score = score_prediction(new_prediction, row.get("reference"), row.get("answer_type"))
                updated.update(
                    {
                        "normalized_prediction": new_prediction,
                        "correct": score["correct"],
                        "numeric_error": score["numeric_error"],
                        "tolerance": score["tolerance"],
                        "signed_difference_normalization": {
                            "previous_prediction": row.get("normalized_prediction"),
                            "previous_correct": row.get("correct"),
                            "normalizer": "label_year_operand_order_v1",
                        },
                    }
                )
                changes.append(
                    {
                        "sample_id": row["sample_id"],
                        "previous_prediction": row.get("normalized_prediction"),
                        "new_prediction": new_prediction,
                        "previous_correct": row.get("correct"),
                        "new_correct": score["correct"],
                    }
                )
        rescored_rows.append(updated)
    result = analyze_result(plan, plan_path, rescored_rows, int(confirmation.get("actual_llm_http_attempts") or 0))
    result.update(
        {
            "schema_version": "plotqa_measured_geosym_signed_difference_rescore_v1",
            "created_at_utc": utc_now(),
            "stage": "stage5_50_sample_confirmation_signed_difference_rescore",
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
            "normalizer": "label_year_operand_order_v1",
            "normalization_changes": changes,
            "interpretation_limits": [
                "This is a zero-cost deterministic rescore of already frozen paid outputs.",
                "The rescore changes only signed difference normalization when response text contains both label-year operands.",
                "It does not repair measured-flat parseability or rerun any paid model call.",
            ],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rescore_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "rescore_audit.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    full = result.get("summary", {}).get(CONDITION, {})
    lines = [
        "# PlotQA Signed Difference Rescore",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed by rescore: 0",
        f"- G_full correct after rescore: {full.get('correct')}/50",
        f"- Normalization changes: {len(result['normalization_changes'])}",
        "",
        "## Changes",
        "",
    ]
    for item in result["normalization_changes"]:
        lines.append(
            f"- {item['sample_id']}: {item['previous_prediction']} ({item['previous_correct']}) -> "
            f"{item['new_prediction']} ({item['new_correct']})"
        )
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {line}" for line in result["interpretation_limits"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-result", required=True)
    parser.add_argument("--diagnostic-plan", required=True)
    parser.add_argument("--measured-records", required=True)
    parser.add_argument("--output-dir", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/stage5_confirmation2_signed_rescore")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = rescore(Path(args.confirmation_result), Path(args.diagnostic_plan), Path(args.measured_records), Path(args.output_dir))
    print(f"wrote signed-difference rescore: {Path(args.output_dir) / 'rescore_result.json'}")
    print(f"stage_status={result['stage_status']}")
    print("rescore_paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
