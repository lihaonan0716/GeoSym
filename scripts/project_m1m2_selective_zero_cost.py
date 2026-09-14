#!/usr/bin/env python
"""Run the zero-cost gate for M1/M2 selective propagation."""

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

SCHEMA_VERSION = "m1m2_selective_zero_cost_projection_v1"
SEED = 20260830


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_projection(stress_roster_path: Path, guarded_projection_path: Path, output_dir: Path) -> dict[str, Any]:
    roster = load_json(stress_roster_path)
    guarded = load_json(guarded_projection_path)
    jobs = roster.get("jobs", [])
    prompt_hashes: dict[str, list[str]] = {condition: [] for condition in SELECTIVE_CONDITIONS}
    schema_findings = []
    for job in jobs:
        schema = job.get("stress_schema") or {}
        if schema.get("expected_answerability") not in {"answerable", "insufficient_evidence", "contradiction"}:
            schema_findings.append({"stress_id": job.get("stress_id"), "finding": "invalid_expected_answerability"})
        for condition in SELECTIVE_CONDITIONS:
            facts = facts_for_selective_condition(schema, condition, SEED, str(job.get("stress_id")))
            prompt_hashes[condition].append(prompt_hash("placeholder question", facts, condition, str(job.get("perturbation_family"))))
    prompt_hash_summary = {
        condition: {
            "count": len(values),
            "unique_count": len(set(values)),
            "first_hash": values[0] if values else None,
        }
        for condition, values in prompt_hashes.items()
    }
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
    gates = {
        "stress_roster_frozen": roster.get("stage_status") == "selective_stress_roster_frozen",
        "stress_job_count_48": len(jobs) == 48,
        "selective_condition_count_7": len(SELECTIVE_CONDITIONS) == 7,
        "schema_findings_empty": not schema_findings,
        "plotqa_guardrail_preserved": plotqa_preserved,
        "paid_calls_zero": True,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_selective_zero_cost_projection",
        "stage_status": "ready_for_paid_manifest" if all(gates.values()) else "selective_zero_cost_failed_closed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "input_mode": "evidence_only",
        "conditions": list(SELECTIVE_CONDITIONS),
        "source_artifacts": {
            "stress_roster": {
                "path": str(stress_roster_path.resolve()),
                "sha256": stable_file_hash(stress_roster_path),
                "stage_status": roster.get("stage_status"),
            },
            "guarded_projection": {
                "path": str(guarded_projection_path.resolve()),
                "sha256": stable_file_hash(guarded_projection_path),
                "stage_status": guarded.get("stage_status"),
            },
        },
        "plotqa_guardrail": {
            "required_correct": 48,
            "observed_correct": guarded.get("summary", {}).get("G_full_guarded", {}).get("correct"),
            "observed_completed": guarded.get("summary", {}).get("G_full_guarded", {}).get("completed"),
            "observed_parseable": guarded.get("summary", {}).get("G_full_guarded", {}).get("parseable"),
            "protected_output_regressions": guardrail.get("protected_output_regressions"),
            "passed": plotqa_preserved,
        },
        "perturbation_family_counts": dict(sorted(family_counts.items())),
        "expected_answerability_counts": dict(sorted(answerability_counts.items())),
        "prompt_hash_summary": prompt_hash_summary,
        "schema_findings": schema_findings,
        "gates": gates,
        "next_required_action": "freeze_selective_paid_pilot_manifest" if all(gates.values()) else "repair_zero_cost_prerequisites",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selective_zero_cost_projection.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "selective_zero_cost_projection.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Selective Propagation Zero-Cost Projection",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed: 0",
        f"- Input mode: `{result['input_mode']}`",
        f"- Conditions: {', '.join(result['conditions'])}",
        f"- PlotQA guardrail passed: {result['plotqa_guardrail']['passed']}",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in result["gates"].items())
    lines.extend(["", "## Next Required Action", "", f"- `{result['next_required_action']}`", ""])
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-roster", default="outputs/m1m2_selective_propagation/stress_roster/selective_stress_roster.json")
    parser.add_argument("--guarded-projection", default="outputs/m1m2_recovery/guarded_zero_cost_projection/guarded_projection_result.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_selective_propagation/zero_cost_projection")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = build_projection(Path(args.stress_roster), Path(args.guarded_projection), Path(args.output_dir))
    print(f"wrote selective zero-cost projection: {Path(args.output_dir) / 'selective_zero_cost_projection.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
