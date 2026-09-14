#!/usr/bin/env python
"""Analyze the bounded M1/M2 recovery paid pilot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GUARDED = "G_full_guarded"
CONTROLS = ("G_all_evidence", "G_top_confidence", "G_random_k")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def grouped_results(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["sample_id"])][str(row["condition"])] = row
    return dict(grouped)


def analyze(paid_result_path: Path, output_dir: Path) -> dict[str, Any]:
    paid = load_json(paid_result_path)
    grouped = grouped_results(paid.get("results", []))
    pairwise = {}
    guarded_rescues = 0
    guarded_degradations = 0
    for control in CONTROLS:
        wins = []
        losses = []
        ties = []
        for sample_id, rows in grouped.items():
            guarded = rows.get(GUARDED, {}).get("correct")
            other = rows.get(control, {}).get("correct")
            if guarded is True and other is not True:
                wins.append(sample_id)
            elif guarded is not True and other is True:
                losses.append(sample_id)
            else:
                ties.append(sample_id)
        pairwise[control] = {
            "guarded_wins": len(wins),
            "guarded_losses": len(losses),
            "ties": len(ties),
            "guarded_win_sample_ids": wins,
            "guarded_loss_sample_ids": losses,
        }
        guarded_rescues += len(wins)
        guarded_degradations += len(losses)
    all_controls_identical = all(
        rows.get(GUARDED, {}).get("correct") == rows.get("G_all_evidence", {}).get("correct")
        == rows.get("G_top_confidence", {}).get("correct") == rows.get("G_random_k", {}).get("correct")
        for rows in grouped.values()
    )
    summary = paid.get("summary", {})
    guarded_summary = summary.get(GUARDED, {})
    gates = {
        "paid_cap_respected": paid.get("paid_calls_executed") == 32 and paid.get("actual_ocr_http_attempts") == 0,
        "all_conditions_completed_parseable_8_of_8": all(
            row.get("completed") == 8 and row.get("parseable") == 8 for row in summary.values()
        ),
        "guarded_beats_all_evidence": pairwise["G_all_evidence"]["guarded_wins"] >= 1 and pairwise["G_all_evidence"]["guarded_losses"] == 0,
        "guarded_beats_top_confidence": pairwise["G_top_confidence"]["guarded_wins"] >= 1 and pairwise["G_top_confidence"]["guarded_losses"] == 0,
        "guarded_beats_random_k": pairwise["G_random_k"]["guarded_wins"] >= 1 and pairwise["G_random_k"]["guarded_losses"] == 0,
    }
    mechanism_supported = all(gates.values())
    result = {
        "schema_version": "m1m2_recovery_paid_pilot_analysis_v1",
        "created_at_utc": utc_now(),
        "stage": "m1m2_recovery_paid_pilot_analysis",
        "stage_status": "m1m2_pilot_mechanism_supported" if mechanism_supported else "m1m2_pilot_no_positive_mechanism_signal",
        "paid_calls_executed_by_analysis": 0,
        "source_artifact": {
            "path": str(paid_result_path.resolve()),
            "stage_status": paid.get("stage_status"),
            "paid_calls_executed": paid.get("paid_calls_executed"),
        },
        "summary": summary,
        "guarded_correct": guarded_summary.get("correct"),
        "pairwise": pairwise,
        "guarded_rescue_count": guarded_rescues,
        "guarded_degradation_count": guarded_degradations,
        "all_conditions_identical_correctness_by_sample": all_controls_identical,
        "gates": gates,
        "next_required_action": "build_frozen_stress_benchmark" if not mechanism_supported else "review_for_possible_50_sample_confirmation",
        "interpretation_limits": [
            "This is an 8-sample pilot and cannot support a full PlotQA claim.",
            "The pilot shows no positive M1/M2 mechanism signal when guarded correctness is identical to all controls.",
            "Do not reframe tied or neutral results as M1/M2 superiority.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "analysis.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Recovery Paid Pilot Analysis",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed by analysis: 0",
        f"- Guarded correct: {result['guarded_correct']}/8",
        f"- Guarded rescues across controls: {result['guarded_rescue_count']}",
        f"- Guarded degradations across controls: {result['guarded_degradation_count']}",
        f"- Identical correctness across all conditions: {result['all_conditions_identical_correctness_by_sample']}",
        "",
        "## Pairwise",
        "",
    ]
    for control, row in result["pairwise"].items():
        lines.append(f"- {GUARDED} vs {control}: wins {row['guarded_wins']}, losses {row['guarded_losses']}, ties {row['ties']}")
    lines.extend(["", "## Gates", ""])
    for key, value in result["gates"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Next Required Action", "", f"- `{result['next_required_action']}`", "", "## Limits", ""])
    lines.extend(f"- {line}" for line in result["interpretation_limits"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paid-result", default="outputs/m1m2_recovery/paid_pilot/paid_result.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_recovery/paid_pilot_analysis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze(Path(args.paid_result), Path(args.output_dir))
    print(f"wrote paid pilot analysis: {Path(args.output_dir) / 'analysis.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed_by_analysis=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
