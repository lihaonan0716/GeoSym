#!/usr/bin/env python
"""Analyze the bounded M1/M2 selective propagation paid pilot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SELECTIVE = "G_selective_propagation"
CONTROLS = ("G_all_evidence", "G_top_confidence", "G_random_k", "G_selective_no_gaps", "G_no_routing")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def grouped_results(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["stress_id"])][str(row["condition"])] = row
    return dict(grouped)


def pairwise(rows_by_stress: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    result = {}
    for control in CONTROLS:
        wins = []
        losses = []
        ties = []
        harm_reductions = []
        harm_increases = []
        abstention_wins = []
        for stress_id, rows in rows_by_stress.items():
            selective = rows.get(SELECTIVE, {})
            other = rows.get(control, {})
            selective_success = bool(selective.get("task_success"))
            other_success = bool(other.get("task_success"))
            if selective_success and not other_success:
                wins.append(stress_id)
            elif other_success and not selective_success:
                losses.append(stress_id)
            else:
                ties.append(stress_id)
            selective_harm = bool(selective.get("harmful_propagation"))
            other_harm = bool(other.get("harmful_propagation"))
            if other_harm and not selective_harm:
                harm_reductions.append(stress_id)
            elif selective_harm and not other_harm:
                harm_increases.append(stress_id)
            if selective.get("correct_abstention") and not other.get("correct_abstention"):
                abstention_wins.append(stress_id)
        result[control] = {
            "selective_wins": len(wins),
            "selective_losses": len(losses),
            "ties": len(ties),
            "selective_win_stress_ids": wins,
            "selective_loss_stress_ids": losses,
            "harmful_propagation_reductions": len(harm_reductions),
            "harmful_propagation_increases": len(harm_increases),
            "harm_reduction_stress_ids": harm_reductions,
            "harm_increase_stress_ids": harm_increases,
            "correct_abstention_wins": len(abstention_wins),
            "correct_abstention_win_stress_ids": abstention_wins,
        }
    return result


def answerability_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    breakdown: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        condition = str(row["condition"])
        expected = str(row.get("expected_answerability") or "unknown")
        condition_block = breakdown.setdefault(condition, {})
        bucket = condition_block.setdefault(
            expected,
            {
                "rows": 0,
                "answerable_outputs": 0,
                "correct_answer": 0,
                "correct_abstention": 0,
                "task_success": 0,
                "harmful_propagation": 0,
                "over_abstention": 0,
            },
        )
        bucket["rows"] += 1
        bucket["answerable_outputs"] += int(bool(row.get("answerable")))
        bucket["correct_answer"] += int(bool(row.get("correct_answer")))
        bucket["correct_abstention"] += int(bool(row.get("correct_abstention")))
        bucket["task_success"] += int(bool(row.get("task_success")))
        bucket["harmful_propagation"] += int(bool(row.get("harmful_propagation")))
        bucket["over_abstention"] += int(expected == "answerable" and row.get("answer_status") != "answerable")
    return {condition: dict(sorted(values.items())) for condition, values in sorted(breakdown.items())}


def analyze(paid_result_path: Path, zero_cost_path: Path, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    paid = load_json(paid_result_path)
    zero_cost = load_json(zero_cost_path)
    manifest = load_json(manifest_path)
    rows = paid.get("results", [])
    rows_by_stress = grouped_results(rows)
    comparisons = pairwise(rows_by_stress)
    parser_gates = {
        "all_rows_schema_valid": all(row.get("schema_valid") for row in rows),
        "all_rows_have_answer_status": all(row.get("answer_status") in {"answerable", "insufficient_evidence", "contradiction", "schema_invalid", "protocol_violation"} for row in rows),
        "non_answerable_predictions_null": all(row.get("normalized_prediction") is None for row in rows if row.get("answer_status") != "answerable"),
        "answerable_rows_supported": all(
            row.get("supporting_fact_ids") or row.get("risk_reason") == "deterministic_solver"
            for row in rows
            if row.get("answer_status") == "answerable"
        ),
    }
    mechanism_gates = {
        "selective_reduces_harm_vs_all_evidence": comparisons["G_all_evidence"]["harmful_propagation_reductions"] >= 2
        and comparisons["G_all_evidence"]["harmful_propagation_increases"] == 0,
        "selective_reduces_harm_vs_top_confidence": comparisons["G_top_confidence"]["harmful_propagation_reductions"] >= 1
        and comparisons["G_top_confidence"]["harmful_propagation_increases"] == 0,
        "selective_has_at_most_one_degradation_vs_each_control": all(row["selective_losses"] <= 1 for row in comparisons.values()),
        "selective_beats_no_gaps_on_abstention_or_harm": comparisons["G_selective_no_gaps"]["correct_abstention_wins"] >= 1
        or comparisons["G_selective_no_gaps"]["harmful_propagation_reductions"] >= 1,
    }
    governance_gates = {
        "paid_cap_respected": paid.get("paid_calls_executed") == manifest.get("llm_http_attempt_cap") == 72,
        "ocr_cap_zero": paid.get("actual_ocr_http_attempts") == manifest.get("ocr_http_attempt_cap") == 0,
        "input_mode_evidence_only": paid.get("input_mode") == manifest.get("input_mode") == "evidence_only",
        "manifest_frozen": manifest.get("stage_status") == "selective_paid_pilot_manifest_frozen",
        "dry_run_safe_default_recorded": manifest.get("safe_default") == "dry_run_no_paid_calls",
    }
    plotqa_gates = {
        "zero_cost_projection_ready": zero_cost.get("stage_status") == "ready_for_paid_manifest",
        "plotqa_guardrail_preserved": (zero_cost.get("plotqa_guardrail") or {}).get("passed") is True,
        "protected_output_regressions_zero": (zero_cost.get("plotqa_guardrail") or {}).get("protected_output_regressions") == 0,
    }
    interpretation_gates = {
        "pilot_not_full_confirmation": len(rows_by_stress) == 12,
        "no_sota_claim": True,
        "larger_confirmation_requires_new_manifest": "No automatic expansion" in str(manifest.get("expansion_rule")),
    }
    gates = {
        "G0_governance": governance_gates,
        "G1_parser_integrity": parser_gates,
        "G2_mechanism_signal": mechanism_gates,
        "G3_plotqa_guardrail": plotqa_gates,
        "G4_interpretation": interpretation_gates,
    }
    supported = all(all(group.values()) for group in gates.values())
    result = {
        "schema_version": "m1m2_selective_paid_pilot_analysis_v1",
        "created_at_utc": utc_now(),
        "stage": "m1m2_selective_paid_pilot_analysis",
        "stage_status": "m1m2_selective_propagation_supported" if supported else "m1m2_selective_propagation_not_supported",
        "paid_calls_executed_by_analysis": 0,
        "source_artifacts": {
            "paid_result": {"path": str(paid_result_path.resolve()), "stage_status": paid.get("stage_status"), "paid_calls_executed": paid.get("paid_calls_executed")},
            "zero_cost_projection": {"path": str(zero_cost_path.resolve()), "stage_status": zero_cost.get("stage_status")},
            "manifest": {"path": str(manifest_path.resolve()), "stage_status": manifest.get("stage_status")},
        },
        "summary": paid.get("summary"),
        "answerability_breakdown": answerability_breakdown(rows),
        "pairwise": comparisons,
        "gates": gates,
        "confirmation_additional_gates": {
            "answerable_preservation_required": "On expected_answerability=answerable rows, G_selective_propagation must preserve at least 7/8 correct answers and must not underperform G_top_confidence or G_no_routing.",
            "over_abstention_limit_required": "On expected_answerability=answerable rows, G_selective_propagation may abstain on at most 1/8 rows.",
            "family_coverage_required": "Every perturbation family must have 8/8 completed rows for every condition.",
        },
        "next_required_action": "freeze_larger_selective_confirmation_manifest" if supported else "write_selective_redesign_note_before_more_paid_calls",
        "interpretation_limits": [
            "This is a 12-job evidence-only pilot, not a full confirmation.",
            "A supported pilot permits a selective-propagation mechanism claim only after a separately frozen confirmation.",
            "The protected PlotQA 48/50 path remains the default and is not replaced by this diagnostic arm.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "analysis.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Selective Propagation Paid Pilot Analysis",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed by analysis: 0",
        "",
        "## Summary",
        "",
    ]
    for condition, row in result["summary"].items():
        lines.append(
            f"- {condition}: task_success {row['task_success']}/12, harmful_propagation {row['harmful_propagation']}/12, "
            f"correct_abstention {row['correct_abstention']}/12"
        )
    lines.extend(["", "## Answerability Breakdown", ""])
    for condition, buckets in result["answerability_breakdown"].items():
        lines.append(f"### {condition}")
        for expected, row in buckets.items():
            lines.append(
                f"- expected={expected}: rows {row['rows']}, task_success {row['task_success']}, "
                f"correct_answer {row['correct_answer']}, correct_abstention {row['correct_abstention']}, "
                f"harmful_propagation {row['harmful_propagation']}, over_abstention {row['over_abstention']}"
            )
        lines.append("")
    lines.extend(["", "## Pairwise", ""])
    for control, row in result["pairwise"].items():
        lines.append(
            f"- {SELECTIVE} vs {control}: wins {row['selective_wins']}, losses {row['selective_losses']}, "
            f"ties {row['ties']}, harm reductions {row['harmful_propagation_reductions']}, harm increases {row['harmful_propagation_increases']}"
        )
    lines.extend(["", "## Gates", ""])
    for group, checks in result["gates"].items():
        lines.append(f"### {group}")
        lines.extend(f"- {key}: {value}" for key, value in checks.items())
        lines.append("")
    lines.extend(["## Confirmation Additional Gates", ""])
    lines.extend(f"- {key}: {value}" for key, value in result["confirmation_additional_gates"].items())
    lines.append("")
    lines.extend(["## Next Required Action", "", f"- `{result['next_required_action']}`", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paid-result", default="outputs/m1m2_selective_propagation/paid_pilot/paid_result.json")
    parser.add_argument("--zero-cost", default="outputs/m1m2_selective_propagation/zero_cost_projection/selective_zero_cost_projection.json")
    parser.add_argument("--manifest", default="outputs/m1m2_selective_propagation/paid_pilot_manifest/frozen_manifest.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_selective_propagation/paid_pilot_analysis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze(Path(args.paid_result), Path(args.zero_cost), Path(args.manifest), Path(args.output_dir))
    print(f"wrote selective paid pilot analysis: {Path(args.output_dir) / 'analysis.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed_by_analysis=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
