#!/usr/bin/env python
"""Build the zero-cost v2 roster for answerability-preserving M1/M2 routing."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_public_benchmark_pilot import stable_file_hash, stable_payload_hash  # noqa: E402

SCHEMA_VERSION = "m1m2_answerability_preserving_roster_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def blocking_gap_keys(schema: dict[str, Any], surfaced_gap_ids: set[str]) -> list[str]:
    keys = []
    for gap in schema.get("evidence_gaps") or []:
        if gap.get("gap_id") not in surfaced_gap_ids:
            continue
        if gap.get("needed_for_question", True) is True and str(gap.get("severity", "blocking")) == "blocking":
            key = str(gap.get("key") or "")
            if key and key not in keys:
                keys.append(key)
    return sorted(keys)


def redesign_schema(job: dict[str, Any]) -> dict[str, Any]:
    schema = copy.deepcopy(job["stress_schema"])
    selected = schema.setdefault("selected_evidence_set", {})
    expected = str(job.get("expected_answerability") or schema.get("expected_answerability"))
    family = str(job.get("perturbation_family"))
    surfaced_ids = set(selected.get("surfaced_gap_ids") or [])
    if expected == "answerable" and family == "spurious_low_confidence_fact":
        selected["required_fact_coverage_complete"] = True
        selected["blocking_gap_keys"] = []
        selected["nonblocking_gap_ids"] = sorted(surfaced_ids)
        selected["surfaced_gap_ids"] = []
        selected["propagation_policy"] = "answer_with_selected_facts"
        selected.setdefault("audit_notes", []).append("v2_required_facts_complete_low_confidence_distractor_suppressed")
    else:
        selected["required_fact_coverage_complete"] = False
        selected["blocking_gap_keys"] = blocking_gap_keys(schema, surfaced_ids)
        if expected == "contradiction":
            selected["propagation_policy"] = "abstain_contradiction"
        elif expected == "insufficient_evidence":
            selected["propagation_policy"] = "abstain_insufficient_evidence"
        selected.setdefault("audit_notes", []).append("v2_blocking_gap_or_contradiction_preserved")
    schema["route_precedence"] = [
        "contradiction_on_required_key",
        "missing_required_fact",
        "answer_with_selected_facts",
    ]
    schema["redesign_hypothesis"] = "answerability_preserving_selective_propagation_v2"
    schema["stress_hash"] = stable_payload_hash(schema)
    return schema


def build_roster(source_roster_path: Path, output_dir: Path) -> dict[str, Any]:
    source = load_json(source_roster_path)
    findings = []
    jobs = []
    for job in source.get("jobs", []):
        redesigned = copy.deepcopy(job)
        redesigned["source_stress_hash"] = (job.get("stress_schema") or {}).get("stress_hash")
        redesigned["stress_schema"] = redesign_schema(job)
        redesigned["stress_hash"] = redesigned["stress_schema"]["stress_hash"]
        jobs.append(redesigned)
    family_counts = Counter(job.get("perturbation_family") for job in jobs)
    answerability_counts = Counter(job.get("expected_answerability") for job in jobs)
    answerable_jobs = [job for job in jobs if job.get("expected_answerability") == "answerable"]
    if source.get("stage_status") != "selective_stress_roster_frozen":
        findings.append("source_roster_not_frozen")
    if len(jobs) != 48:
        findings.append("job_count_not_48")
    if any(value != 8 for value in family_counts.values()) or len(family_counts) != 6:
        findings.append("family_counts_not_8_each")
    for job in answerable_jobs:
        selected = (job.get("stress_schema") or {}).get("selected_evidence_set") or {}
        if selected.get("required_fact_coverage_complete") is not True:
            findings.append(f"answerable_required_coverage_not_complete:{job.get('stress_id')}")
        if selected.get("blocking_gap_keys") != []:
            findings.append(f"answerable_blocking_gap_keys_not_empty:{job.get('stress_id')}")
        if selected.get("propagation_policy") != "answer_with_selected_facts":
            findings.append(f"answerable_route_not_answer_with_selected_facts:{job.get('stress_id')}")
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_answerability_preserving_roster",
        "stage_status": "answerability_preserving_roster_frozen" if not findings else "answerability_preserving_roster_failed_closed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "source_roster": {
                "path": str(source_roster_path.resolve()),
                "sha256": stable_file_hash(source_roster_path),
                "stage_status": source.get("stage_status"),
            }
        },
        "sample_level_job_count": len(jobs),
        "perturbation_family_counts": dict(sorted(family_counts.items())),
        "expected_answerability_counts": dict(sorted(answerability_counts.items())),
        "redesign_policy": {
            "hypothesis": "suppress_low_confidence_distractors_without_blocking_complete_required_evidence",
            "route_precedence": ["contradiction_on_required_key", "missing_required_fact", "answer_with_selected_facts"],
            "plotqa_default_modified": False,
        },
        "blocking_findings": findings,
        "jobs": jobs,
        "next_required_action": "run_answerability_preserving_zero_cost_projection" if not findings else "repair_answerability_preserving_roster",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "answerability_preserving_roster.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "answerability_preserving_roster.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Answerability-Preserving Roster",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed: 0",
        f"- Sample-level jobs: {result['sample_level_job_count']}",
        "",
        "## Counts",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in result["perturbation_family_counts"].items())
    lines.extend(["", "## Blocking Findings", ""])
    lines.extend(f"- `{item}`" for item in result["blocking_findings"] or ["none"])
    lines.extend(["", "## Next Required Action", "", f"- `{result['next_required_action']}`", ""])
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roster", default="outputs/m1m2_selective_propagation/stress_roster/selective_stress_roster.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_answerability_preserving/roster")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = build_roster(Path(args.source_roster), Path(args.output_dir))
    print(f"wrote answerability-preserving roster: {Path(args.output_dir) / 'answerability_preserving_roster.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
