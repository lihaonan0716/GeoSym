#!/usr/bin/env python
"""Analyze the bounded M1/M2 stress paid pilot."""

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
        grouped[str(row["stress_id"])][str(row["condition"])] = row
    return dict(grouped)


def harmful(row: dict[str, Any]) -> bool:
    return bool(row.get("parseable")) and row.get("correct") is not True and not row.get("insufficient_evidence_signaled")


def analyze(paid_result_path: Path, output_dir: Path) -> dict[str, Any]:
    paid = load_json(paid_result_path)
    grouped = grouped_results(paid.get("results", []))
    pairwise = {}
    guarded_rescues = 0
    guarded_degradations = 0
    guarded_harm_reductions = 0
    guarded_harm_increases = 0
    for control in CONTROLS:
        wins = []
        losses = []
        ties = []
        harm_reductions = []
        harm_increases = []
        for stress_id, rows in grouped.items():
            guarded = rows.get(GUARDED, {})
            other = rows.get(control, {})
            guarded_correct = guarded.get("correct")
            other_correct = other.get("correct")
            if guarded_correct is True and other_correct is not True:
                wins.append(stress_id)
            elif guarded_correct is not True and other_correct is True:
                losses.append(stress_id)
            else:
                ties.append(stress_id)
            guarded_harm = harmful(guarded)
            other_harm = harmful(other)
            if other_harm and not guarded_harm:
                harm_reductions.append(stress_id)
            elif guarded_harm and not other_harm:
                harm_increases.append(stress_id)
        pairwise[control] = {
            "guarded_wins": len(wins),
            "guarded_losses": len(losses),
            "ties": len(ties),
            "guarded_win_stress_ids": wins,
            "guarded_loss_stress_ids": losses,
            "harmful_propagation_reductions": len(harm_reductions),
            "harmful_propagation_increases": len(harm_increases),
            "harm_reduction_stress_ids": harm_reductions,
            "harm_increase_stress_ids": harm_increases,
        }
        guarded_rescues += len(wins)
        guarded_degradations += len(losses)
        guarded_harm_reductions += len(harm_reductions)
        guarded_harm_increases += len(harm_increases)
    summary = paid.get("summary", {})
    gates = {
        "paid_cap_respected": paid.get("paid_calls_executed") == 32 and paid.get("actual_ocr_http_attempts") == 0,
        "all_conditions_completed_or_explicitly_abstained": all(
            bool(row.get("completed")) or bool(row.get("insufficient_evidence_signaled"))
            for row in paid.get("results", [])
        ),
        "guarded_accuracy_not_lower_than_all_evidence": pairwise["G_all_evidence"]["guarded_losses"] == 0,
        "guarded_accuracy_not_lower_than_top_confidence": pairwise["G_top_confidence"]["guarded_losses"] == 0,
        "guarded_accuracy_not_lower_than_random_k": pairwise["G_random_k"]["guarded_losses"] == 0,
        "guarded_reduces_harm_vs_all_evidence": pairwise["G_all_evidence"]["harmful_propagation_reductions"] >= 1
        and pairwise["G_all_evidence"]["harmful_propagation_increases"] == 0,
        "guarded_reduces_harm_vs_top_confidence": pairwise["G_top_confidence"]["harmful_propagation_reductions"] >= 1
        and pairwise["G_top_confidence"]["harmful_propagation_increases"] == 0,
        "guarded_beats_random_by_accuracy_or_harm": pairwise["G_random_k"]["guarded_wins"] >= 1
        or pairwise["G_random_k"]["harmful_propagation_reductions"] >= 1,
    }
    mechanism_supported = all(gates.values())
    result = {
        "schema_version": "m1m2_recovery_stress_paid_pilot_analysis_v1",
        "created_at_utc": utc_now(),
        "stage": "m1m2_recovery_stress_paid_pilot_analysis",
        "stage_status": "m1m2_mechanism_supported" if mechanism_supported else "m1m2_mechanism_not_supported",
        "paid_calls_executed_by_analysis": 0,
        "source_artifact": {
            "path": str(paid_result_path.resolve()),
            "stage_status": paid.get("stage_status"),
            "paid_calls_executed": paid.get("paid_calls_executed"),
        },
        "summary": summary,
        "pairwise": pairwise,
        "guarded_rescue_count": guarded_rescues,
        "guarded_degradation_count": guarded_degradations,
        "guarded_harm_reduction_count": guarded_harm_reductions,
        "guarded_harm_increase_count": guarded_harm_increases,
        "gates": gates,
        "next_required_action": "downgrade_m1m2_claim" if not mechanism_supported else "freeze_larger_confirmation_before_any_expansion",
        "interpretation_limits": [
            "This is an 8-job stress pilot and cannot support a full PlotQA performance claim.",
            "A supported stress mechanism still cannot replace the protected PlotQA path without the 48/50 guardrail.",
            "A not-supported result must be reported as a limitation, not reframed as M1/M2 superiority.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "analysis.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Recovery Stress Paid Pilot Analysis",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed by analysis: 0",
        f"- Guarded rescues across controls: {result['guarded_rescue_count']}",
        f"- Guarded degradations across controls: {result['guarded_degradation_count']}",
        f"- Guarded harmful-propagation reductions across controls: {result['guarded_harm_reduction_count']}",
        f"- Guarded harmful-propagation increases across controls: {result['guarded_harm_increase_count']}",
        "",
        "## Pairwise",
        "",
    ]
    for control, row in result["pairwise"].items():
        lines.append(
            f"- {GUARDED} vs {control}: wins {row['guarded_wins']}, losses {row['guarded_losses']}, "
            f"ties {row['ties']}, harm reductions {row['harmful_propagation_reductions']}, "
            f"harm increases {row['harmful_propagation_increases']}"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- {key}: {value}" for key, value in result["gates"].items())
    lines.extend(["", "## Next Required Action", "", f"- `{result['next_required_action']}`", "", "## Limits", ""])
    lines.extend(f"- {line}" for line in result["interpretation_limits"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paid-result", default="outputs/m1m2_recovery/stress_paid_pilot/paid_result.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_recovery/stress_paid_pilot_analysis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze(Path(args.paid_result), Path(args.output_dir))
    print(f"wrote stress paid pilot analysis: {Path(args.output_dir) / 'analysis.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed_by_analysis=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
