#!/usr/bin/env python
"""Run zero-cost gates for answerability-preserving M1/M2 selective routing."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402
from scripts.selective_propagation import SELECTIVE_CONDITIONS, facts_for_selective_condition, prompt_hash  # noqa: E402

SCHEMA_VERSION = "m1m2_answerability_preserving_zero_cost_projection_v1"
SEED = 20260830


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_projection(roster_path: Path, guarded_projection_path: Path, output_dir: Path) -> dict[str, Any]:
    roster = load_json(roster_path)
    guarded = load_json(guarded_projection_path)
    jobs = roster.get("jobs", [])
    schema_findings = []
    prompt_hashes: dict[str, list[str]] = {condition: [] for condition in SELECTIVE_CONDITIONS}
    answerable_visible_blocking_gaps = 0
    nonanswerable_missing_blocking_gaps = 0
    answerable_route_ok = 0
    for job in jobs:
        expected = str(job.get("expected_answerability"))
        family = str(job.get("perturbation_family"))
        schema = job.get("stress_schema") or {}
        selective_facts = facts_for_selective_condition(schema, "G_selective_propagation", SEED, str(job.get("stress_id")))
        if expected == "answerable":
            if selective_facts.get("evidence_gaps"):
                answerable_visible_blocking_gaps += 1
            if selective_facts.get("routing_policy") == "answer_with_selected_facts":
                answerable_route_ok += 1
            if selective_facts.get("required_fact_coverage_complete") is not True:
                schema_findings.append({"stress_id": job.get("stress_id"), "finding": "answerable_required_coverage_not_complete"})
        else:
            if expected == "insufficient_evidence" and not selective_facts.get("evidence_gaps"):
                nonanswerable_missing_blocking_gaps += 1
            if expected == "contradiction" and selective_facts.get("routing_policy") != "abstain_contradiction":
                schema_findings.append({"stress_id": job.get("stress_id"), "finding": "contradiction_route_not_preserved"})
        for condition in SELECTIVE_CONDITIONS:
            facts = facts_for_selective_condition(schema, condition, SEED, str(job.get("stress_id")))
            prompt_hashes[condition].append(prompt_hash("placeholder question", facts, condition, family))
    guardrail = guarded.get("guardrail") or {}
    plotqa_preserved = (
        guarded.get("stage_status") == "plotqa_guardrail_preserved"
        and guardrail.get("passed") is True
        and (guarded.get("summary", {}).get("G_full_guarded", {}).get("correct") or 0) >= 48
        and (guarded.get("summary", {}).get("G_full_guarded", {}).get("completed") or 0) == 50
        and (guarded.get("summary", {}).get("G_full_guarded", {}).get("parseable") or 0) == 50
        and guardrail.get("protected_output_regressions") == 0
    )
    family_counts = Counter(job.get("perturbation_family") for job in jobs)
    answerability_counts = Counter(job.get("expected_answerability") for job in jobs)
    prompt_hash_summary = {
        condition: {"count": len(values), "unique_count": len(set(values)), "first_hash": values[0] if values else None}
        for condition, values in prompt_hashes.items()
    }
    gates = {
        "roster_frozen": roster.get("stage_status") == "answerability_preserving_roster_frozen",
        "stress_job_count_48": len(jobs) == 48,
        "selective_condition_count_7": len(SELECTIVE_CONDITIONS) == 7,
        "plotqa_guardrail_preserved": plotqa_preserved,
        "answerable_rows_8": answerability_counts.get("answerable") == 8,
        "answerable_visible_blocking_gaps_zero": answerable_visible_blocking_gaps == 0,
        "answerable_route_answer_with_selected_facts": answerable_route_ok == 8,
        "nonanswerable_missing_blocking_gaps_zero": nonanswerable_missing_blocking_gaps == 0,
        "family_counts_8_each": len(family_counts) == 6 and all(value == 8 for value in family_counts.values()),
        "schema_findings_empty": not schema_findings,
        "paid_calls_zero": True,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_answerability_preserving_zero_cost_projection",
        "stage_status": "ready_for_answerability_preserving_paid_manifest" if all(gates.values()) else "answerability_preserving_zero_cost_failed_closed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "input_mode": "evidence_only",
        "conditions": list(SELECTIVE_CONDITIONS),
        "source_artifacts": {
            "roster": {"path": str(roster_path.resolve()), "sha256": stable_file_hash(roster_path), "stage_status": roster.get("stage_status")},
            "guarded_projection": {"path": str(guarded_projection_path.resolve()), "sha256": stable_file_hash(guarded_projection_path), "stage_status": guarded.get("stage_status")},
        },
        "plotqa_guardrail": {
            "required_correct": 48,
            "observed_correct": guarded.get("summary", {}).get("G_full_guarded", {}).get("correct"),
            "observed_completed": guarded.get("summary", {}).get("G_full_guarded", {}).get("completed"),
            "observed_parseable": guarded.get("summary", {}).get("G_full_guarded", {}).get("parseable"),
            "protected_output_regressions": guardrail.get("protected_output_regressions"),
            "passed": plotqa_preserved,
        },
        "answerability_counts": dict(sorted(answerability_counts.items())),
        "perturbation_family_counts": dict(sorted(family_counts.items())),
        "answerable_visible_blocking_gaps": answerable_visible_blocking_gaps,
        "nonanswerable_missing_blocking_gaps": nonanswerable_missing_blocking_gaps,
        "prompt_hash_summary": prompt_hash_summary,
        "schema_findings": schema_findings,
        "gates": gates,
        "next_required_action": "freeze_answerability_preserving_paid_pilot_manifest" if all(gates.values()) else "repair_answerability_preserving_zero_cost_projection",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "zero_cost_projection.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "zero_cost_projection.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Answerability-Preserving Zero-Cost Projection",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed: 0",
        f"- PlotQA guardrail passed: {result['plotqa_guardrail']['passed']}",
        f"- Answerable visible blocking gaps: {result['answerable_visible_blocking_gaps']}",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in result["gates"].items())
    lines.extend(["", "## Blocking Findings", ""])
    lines.extend(f"- `{item}`" for item in result["schema_findings"] or ["none"])
    lines.extend(["", "## Next Required Action", "", f"- `{result['next_required_action']}`", ""])
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", default="outputs/m1m2_answerability_preserving/roster/answerability_preserving_roster.json")
    parser.add_argument("--guarded-projection", default="outputs/m1m2_recovery/guarded_zero_cost_projection/guarded_projection_result.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_answerability_preserving/zero_cost_projection")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = build_projection(Path(args.roster), Path(args.guarded_projection), Path(args.output_dir))
    print(f"wrote answerability-preserving zero-cost projection: {Path(args.output_dir) / 'zero_cost_projection.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
