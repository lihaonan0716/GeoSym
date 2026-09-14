#!/usr/bin/env python
"""Analyze the 48-job M1/M2 selective confirmation with frozen gates."""

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

from scripts.analyze_m1m2_selective_paid_pilot import (  # noqa: E402
    CONTROLS,
    SELECTIVE,
    answerability_breakdown,
    grouped_results,
    pairwise,
)
from scripts.selective_propagation import SELECTIVE_CONDITIONS  # noqa: E402

CONFIRMATION_SAMPLE_CAP = 48
CONFIRMATION_LLM_CAP = CONFIRMATION_SAMPLE_CAP * len(SELECTIVE_CONDITIONS)

EXPECTED_TERMINAL_STATES = {
    "m1m2_selective_confirmation_supported",
    "m1m2_selective_confirmation_too_conservative",
    "m1m2_selective_confirmation_no_harm_reduction",
    "m1m2_selective_confirmation_no_routing_attribution",
    "m1m2_selective_confirmation_family_limited",
    "m1m2_selective_confirmation_failed_closed",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def family_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        condition = str(row["condition"])
        family = str(row.get("perturbation_family") or "unknown")
        condition_block = result.setdefault(condition, {})
        bucket = condition_block.setdefault(
            family,
            {
                "rows": 0,
                "task_success": 0,
                "correct_answer": 0,
                "correct_abstention": 0,
                "harmful_propagation": 0,
                "over_abstention": 0,
            },
        )
        bucket["rows"] += 1
        bucket["task_success"] += int(bool(row.get("task_success")))
        bucket["correct_answer"] += int(bool(row.get("correct_answer")))
        bucket["correct_abstention"] += int(bool(row.get("correct_abstention")))
        bucket["harmful_propagation"] += int(bool(row.get("harmful_propagation")))
        bucket["over_abstention"] += int(row.get("expected_answerability") == "answerable" and row.get("answer_status") != "answerable")
    return {condition: dict(sorted(values.items())) for condition, values in sorted(result.items())}


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    return {
        condition: {
            "rows": len(subset),
            "schema_valid": sum(1 for item in subset if item.get("schema_valid")),
            "answerable": sum(1 for item in subset if item.get("answerable")),
            "prediction_parseable": sum(1 for item in subset if item.get("prediction_parseable")),
            "correct_answer": sum(1 for item in subset if item.get("correct_answer")),
            "correct_abstention": sum(1 for item in subset if item.get("correct_abstention")),
            "task_success": sum(1 for item in subset if item.get("task_success")),
            "harmful_propagation": sum(1 for item in subset if item.get("harmful_propagation")),
            "protocol_violation": sum(1 for item in subset if item.get("answer_status") == "protocol_violation"),
        }
        for condition, subset in sorted(by_condition.items())
    }


def terminal_state(gates: dict[str, dict[str, bool]], family_limited: bool) -> str:
    if not all(gates["G0_governance"].values()) or not all(gates["G1_parser_integrity"].values()) or not all(gates["G6_plotqa_guardrail"].values()):
        return "m1m2_selective_confirmation_failed_closed"
    if not all(gates["G3_answerable_preservation"].values()):
        return "m1m2_selective_confirmation_too_conservative"
    if not all(gates["G2_harmful_propagation"].values()):
        return "m1m2_selective_confirmation_no_harm_reduction"
    if not all(gates["G4_gap_and_routing_contribution"].values()):
        return "m1m2_selective_confirmation_no_routing_attribution"
    if family_limited or not all(gates["G5_family_coverage"].values()):
        return "m1m2_selective_confirmation_family_limited"
    return "m1m2_selective_confirmation_supported"


def analyze(confirmation_result_path: Path, manifest_path: Path, zero_cost_path: Path, output_dir: Path) -> dict[str, Any]:
    confirmation = load_json(confirmation_result_path)
    manifest = load_json(manifest_path)
    zero_cost = load_json(zero_cost_path)
    rows = confirmation.get("results", [])
    rows_by_stress = grouped_results(rows)
    comparisons = pairwise(rows_by_stress)
    summary = condition_summary(rows)
    answerability = answerability_breakdown(rows)
    families = family_breakdown(rows)
    selective_summary = summary.get(SELECTIVE, {})
    selective_answerable = answerability.get(SELECTIVE, {}).get("answerable", {})
    selective_families = families.get(SELECTIVE, {})
    family_failures = [
        family
        for family, bucket in selective_families.items()
        if bucket.get("task_success", 0) < 6 or bucket.get("harmful_propagation", 0) > 2
    ]
    gates = {
        "G0_governance": {
            "confirmation_completed": confirmation.get("stage_status") == "selective_confirmation_completed",
            "manifest_pending_approval_source": manifest.get("stage_status") == "selective_confirmation_manifest_frozen_pending_approval",
            "paid_cap_respected": confirmation.get("paid_calls_executed") == manifest.get("llm_http_attempt_cap") == CONFIRMATION_LLM_CAP,
            "ocr_cap_zero": confirmation.get("actual_ocr_http_attempts") == manifest.get("ocr_http_attempt_cap") == 0,
            "input_mode_evidence_only": confirmation.get("input_mode") == manifest.get("input_mode") == "evidence_only",
            "row_count_matches_frozen_cap": len(rows) == CONFIRMATION_LLM_CAP,
            "conditions_complete": all(summary.get(condition, {}).get("rows") == CONFIRMATION_SAMPLE_CAP for condition in SELECTIVE_CONDITIONS),
        },
        "G1_parser_integrity": {
            "all_rows_schema_valid": all(row.get("schema_valid") for row in rows),
            "all_rows_have_answer_status": all(row.get("answer_status") in {"answerable", "insufficient_evidence", "contradiction", "schema_invalid", "protocol_violation"} for row in rows),
            "non_answerable_predictions_null": all(row.get("normalized_prediction") is None for row in rows if row.get("answer_status") != "answerable"),
            "answerable_rows_supported": all(
                row.get("supporting_fact_ids") or row.get("risk_reason") == "deterministic_solver"
                for row in rows
                if row.get("answer_status") == "answerable"
            ),
        },
        "G2_harmful_propagation": {
            "selective_reduces_harm_vs_all_evidence_by_6": comparisons["G_all_evidence"]["harmful_propagation_reductions"] >= 6,
            "selective_reduces_harm_vs_top_confidence_by_3": comparisons["G_top_confidence"]["harmful_propagation_reductions"] >= 3,
            "selective_has_no_harm_increase_vs_any_control": all(item["harmful_propagation_increases"] == 0 for item in comparisons.values()),
            "selective_has_at_most_two_degradations_vs_each_control": all(item["selective_losses"] <= 2 for item in comparisons.values()),
        },
        "G3_answerable_preservation": {
            "answerable_rows_8": selective_answerable.get("rows") == 8,
            "selective_correct_answer_at_least_7_of_8": selective_answerable.get("correct_answer", 0) >= 7,
            "selective_over_abstention_at_most_1_of_8": selective_answerable.get("over_abstention", 0) <= 1,
            "selective_not_below_top_confidence_on_answerable": selective_answerable.get("correct_answer", 0) >= answerability.get("G_top_confidence", {}).get("answerable", {}).get("correct_answer", 0),
            "selective_not_below_no_routing_on_answerable": selective_answerable.get("correct_answer", 0) >= answerability.get("G_no_routing", {}).get("answerable", {}).get("correct_answer", 0),
        },
        "G4_gap_and_routing_contribution": {
            "selective_beats_no_gaps_by_3": (
                comparisons["G_selective_no_gaps"]["correct_abstention_wins"]
                + comparisons["G_selective_no_gaps"]["harmful_propagation_reductions"]
            )
            >= 3,
            "selective_beats_no_routing_by_2": (
                comparisons["G_no_routing"]["correct_abstention_wins"]
                + comparisons["G_no_routing"]["harmful_propagation_reductions"]
            )
            >= 2,
        },
        "G5_family_coverage": {
            "six_families_present": len(selective_families) == 6,
            "all_conditions_have_8_rows_per_family": all(
                len(family_rows) == 6 and all(bucket.get("rows") == 8 for bucket in family_rows.values())
                for family_rows in families.values()
            ),
            "fewer_than_three_selective_family_failures": len(family_failures) < 3,
        },
        "G6_plotqa_guardrail": {
            "zero_cost_projection_ready": zero_cost.get("stage_status") == "ready_for_paid_manifest",
            "plotqa_guardrail_preserved": (zero_cost.get("plotqa_guardrail") or {}).get("passed") is True,
            "protected_output_regressions_zero": (zero_cost.get("plotqa_guardrail") or {}).get("protected_output_regressions") == 0,
        },
    }
    state = terminal_state(gates, family_limited=bool(family_failures))
    if state not in EXPECTED_TERMINAL_STATES:
        state = "m1m2_selective_confirmation_failed_closed"
    result = {
        "schema_version": "m1m2_selective_confirmation_analysis_v1",
        "created_at_utc": utc_now(),
        "stage": "m1m2_selective_confirmation_analysis",
        "stage_status": state,
        "paid_calls_executed_by_analysis": 0,
        "source_artifacts": {
            "confirmation_result": {
                "path": str(confirmation_result_path.resolve()),
                "stage_status": confirmation.get("stage_status"),
                "paid_calls_executed": confirmation.get("paid_calls_executed"),
            },
            "manifest": {"path": str(manifest_path.resolve()), "stage_status": manifest.get("stage_status")},
            "zero_cost_projection": {"path": str(zero_cost_path.resolve()), "stage_status": zero_cost.get("stage_status")},
        },
        "summary": summary,
        "answerability_breakdown": answerability,
        "family_breakdown": families,
        "pairwise": comparisons,
        "selective_family_failures": family_failures,
        "gates": gates,
        "interpretation_limits": [
            "This confirmation can support only a selective-propagation mechanism claim.",
            "It cannot replace the protected PlotQA 48/50 default unless the guardrail remains preserved.",
            "Failure of any frozen gate must be reported as the corresponding terminal state.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "analysis.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Selective Propagation Confirmation Analysis",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed by analysis: 0",
        "",
        "## Summary",
        "",
    ]
    for condition, row in result["summary"].items():
        lines.append(
            f"- {condition}: task_success {row['task_success']}/{row['rows']}, "
            f"harmful_propagation {row['harmful_propagation']}/{row['rows']}, "
            f"correct_answer {row['correct_answer']}/{row['rows']}, "
            f"correct_abstention {row['correct_abstention']}/{row['rows']}"
        )
    lines.extend(["", "## Pairwise", ""])
    for control, row in result["pairwise"].items():
        lines.append(
            f"- {SELECTIVE} vs {control}: wins {row['selective_wins']}, losses {row['selective_losses']}, "
            f"ties {row['ties']}, harm reductions {row['harmful_propagation_reductions']}, "
            f"harm increases {row['harmful_propagation_increases']}"
        )
    lines.extend(["", "## Gates", ""])
    for group, checks in result["gates"].items():
        lines.append(f"### {group}")
        lines.extend(f"- {key}: {value}" for key, value in checks.items())
        lines.append("")
    lines.extend(["## Selective Family Failures", ""])
    lines.extend(f"- `{item}`" for item in result["selective_family_failures"] or ["none"])
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-result", default="outputs/m1m2_selective_propagation/confirmation_run/paid_result.json")
    parser.add_argument("--manifest", default="outputs/m1m2_selective_propagation/confirmation_manifest/frozen_confirmation_manifest.json")
    parser.add_argument("--zero-cost", default="outputs/m1m2_selective_propagation/zero_cost_projection/selective_zero_cost_projection.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_selective_propagation/confirmation_analysis")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = analyze(Path(args.confirmation_result), Path(args.manifest), Path(args.zero_cost), Path(args.output_dir))
    print(f"wrote selective confirmation analysis: {Path(args.output_dir) / 'analysis.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed_by_analysis=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
