#!/usr/bin/env python
"""Build the zero-cost v3 candidate-guided M1/M2 stress roster."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_public_benchmark_pilot import normalize_answer, stable_file_hash, stable_payload_hash  # noqa: E402

SCHEMA_VERSION = "m1m2_candidate_guided_roster_v1"
SEED = 20260830


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_candidate(raw_text: str, answer_type: str) -> Any:
    match = re.fullmatch(r"\s*deterministic_m3_prediction=(.+?)\s*", raw_text or "")
    if not match:
        raise ValueError("not_a_deterministic_m3_prediction")
    return normalize_answer(match.group(1), answer_type)


def by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in rows}


def answerable_jobs(schema_outputs: dict[str, Any], guarded: dict[str, Any], records: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    findings: list[str] = []
    guarded_by_id = by_sample(guarded.get("results") or [])
    rows = []
    for schema in sorted(schema_outputs.get("outputs") or [], key=lambda row: str(row.get("sample_id"))):
        sample_id = str(schema.get("sample_id"))
        record = records.get(sample_id)
        guarded_row = guarded_by_id.get(sample_id)
        if record is None or guarded_row is None:
            continue
        selected = schema.get("selected_evidence_set") or {}
        if selected.get("propagation_policy") != "solve_deterministically":
            continue
        raw_text = str(guarded_row.get("raw_text") or "")
        try:
            candidate = deterministic_candidate(raw_text, str(record.get("answer_type")))
        except Exception:
            continue
        selected_ids = list(selected.get("selected_fact_ids") or [])
        suppressed_ids = list(selected.get("suppressed_fact_ids") or [])
        if not selected_ids or not suppressed_ids:
            continue
        stress_schema = copy.deepcopy(schema)
        stress_schema["expected_answerability"] = "answerable"
        stress_schema["redesign_hypothesis"] = "candidate_guided_selective_propagation_v3_1"
        stress_schema["route_precedence"] = ["contradiction_on_required_key", "missing_required_fact", "answer_with_candidate_and_selected_facts"]
        stress_schema.setdefault("selected_evidence_set", {})["required_fact_coverage_complete"] = True
        stress_schema["selected_evidence_set"]["blocking_gap_keys"] = []
        stress_schema["selected_evidence_set"]["surfaced_gap_ids"] = []
        stress_schema["selected_evidence_set"]["propagation_policy"] = "answer_with_candidate_and_selected_facts"
        stress_schema["selected_evidence_set"]["answer_candidate"] = {
            "normalized_prediction": candidate,
            "answer_type": record.get("answer_type"),
            "source": "deterministic_m3_from_selected_facts",
            "source_raw_text_hash": stable_payload_hash({"raw_text": raw_text}),
            "supporting_fact_ids": selected_ids,
            "selection_rule": "schema_policy_solve_deterministically_and_deterministic_m3_raw_text_present",
        }
        stress_schema["stress_hash"] = stable_payload_hash(stress_schema)
        rows.append(
            {
                "stress_id": f"candidate_guided_answerable_{len(rows):03d}",
                "source": "plotqa_guarded_projection_schema_join",
                "source_sample_id": sample_id,
                "perturbation_family": "spurious_low_confidence_fact",
                "seed": SEED,
                "answer_independent_selection": True,
                "expected_answerability": "answerable",
                "stress_schema": stress_schema,
                "stress_hash": stress_schema["stress_hash"],
            }
        )
    if len(rows) < 6:
        findings.append("candidate_guided_answerable_jobs_underfilled")
    return rows[:6], findings


def nonanswerable_jobs(source_roster: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    findings: list[str] = []
    rows = []
    families_seen: set[str] = set()
    nonanswerable_source = [
        job
        for job in sorted(source_roster.get("jobs") or [], key=lambda row: (str(row.get("perturbation_family")), str(row.get("stress_id"))))
        if job.get("expected_answerability") != "answerable"
    ]
    for job in nonanswerable_source:
        if job.get("expected_answerability") == "answerable":
            continue
        family = str(job.get("perturbation_family"))
        if family in families_seen:
            continue
        rows.append(clone_control_job(job, len(rows)))
        families_seen.add(family)
        if len(rows) == 6:
            break
    selected_source_ids = {str(job.get("source_stress_id")) for job in rows}
    for job in nonanswerable_source:
        if len(rows) == 6:
            break
        if str(job.get("stress_id")) in selected_source_ids:
            continue
        rows.append(clone_control_job(job, len(rows)))
    if len(rows) < 6:
        findings.append("nonanswerable_controls_underfilled")
    return rows, findings


def clone_control_job(job: dict[str, Any], index: int) -> dict[str, Any]:
    cloned = copy.deepcopy(job)
    cloned["stress_id"] = f"candidate_guided_control_{index:03d}_{job['stress_id']}"
    cloned["source_stress_id"] = job.get("stress_id")
    cloned["stress_schema"]["redesign_hypothesis"] = "candidate_guided_selective_propagation_v3_1"
    cloned["stress_schema"]["stress_hash"] = stable_payload_hash(cloned["stress_schema"])
    cloned["stress_hash"] = cloned["stress_schema"]["stress_hash"]
    return cloned


def build_roster(schema_outputs_path: Path, guarded_projection_path: Path, measured_records_path: Path, source_roster_path: Path, output_dir: Path) -> dict[str, Any]:
    schema_outputs = load_json(schema_outputs_path)
    guarded = load_json(guarded_projection_path)
    records = {str(row["sample_id"]): row for row in load_json(measured_records_path)}
    source_roster = load_json(source_roster_path)
    answerable, findings = answerable_jobs(schema_outputs, guarded, records)
    controls, control_findings = nonanswerable_jobs(source_roster)
    findings.extend(control_findings)
    jobs = answerable + controls
    counts = Counter(job.get("expected_answerability") for job in jobs)
    family_counts = Counter(job.get("perturbation_family") for job in jobs)
    gates = {
        "schema_outputs_ready": schema_outputs.get("stage_status") == "ready_for_zero_cost_projection",
        "guarded_projection_preserved": guarded.get("stage_status") == "plotqa_guardrail_preserved",
        "selected_jobs_12": len(jobs) == 12,
        "answerable_jobs_6": counts.get("answerable") == 6,
        "nonanswerable_jobs_6": len(jobs) - counts.get("answerable", 0) == 6,
        "all_answerable_have_candidate": all(((job.get("stress_schema") or {}).get("selected_evidence_set") or {}).get("answer_candidate") for job in answerable),
        "paid_calls_zero": True,
        "blocking_findings_empty": not findings,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_candidate_guided_roster",
        "stage_status": "candidate_guided_roster_frozen" if all(gates.values()) else "candidate_guided_roster_failed_closed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "redesign_name": "candidate_guided_selective_propagation_v3_1",
        "source_artifacts": {
            "schema_outputs": {"path": str(schema_outputs_path.resolve()), "sha256": stable_file_hash(schema_outputs_path), "stage_status": schema_outputs.get("stage_status")},
            "guarded_projection": {"path": str(guarded_projection_path.resolve()), "sha256": stable_file_hash(guarded_projection_path), "stage_status": guarded.get("stage_status")},
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
            "source_roster": {"path": str(source_roster_path.resolve()), "sha256": stable_file_hash(source_roster_path), "stage_status": source_roster.get("stage_status")},
        },
        "sample_level_job_count": len(jobs),
        "expected_answerability_counts": dict(sorted(counts.items())),
        "perturbation_family_counts": dict(sorted(family_counts.items())),
        "frozen_hypothesis": "M1/M2 helps when selected facts carry an explicit deterministic answer candidate while suppressed low-confidence evidence remains hidden from the selective arm but visible to unmanaged controls.",
        "selection_rule": "Sort schema outputs by sample_id; keep rows with solve_deterministically policy, nonempty selected and suppressed facts, and deterministic_m3_prediction raw text; take first six. Add one non-answerable control from each stress family.",
        "forbidden_runtime_inputs": ["reference_answer", "sample_id_specific_patch", "chart_image", "manual_label"],
        "blocking_findings": findings,
        "gates": gates,
        "jobs": jobs,
        "next_required_action": "run_candidate_guided_zero_cost_projection" if all(gates.values()) else "repair_candidate_guided_roster",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_guided_roster.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "candidate_guided_roster.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Candidate-Guided Roster",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed: 0",
        f"- Sample-level jobs: {result['sample_level_job_count']}",
        f"- Hypothesis: {result['frozen_hypothesis']}",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in result["gates"].items())
    lines.extend(["", "## Blocking Findings", ""])
    lines.extend(f"- `{item}`" for item in result["blocking_findings"] or ["none"])
    lines.extend(["", "## Next Required Action", "", f"- `{result['next_required_action']}`", ""])
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-outputs", default="outputs/m1m2_recovery/schema_outputs/m1m2_schema_outputs.json")
    parser.add_argument("--guarded-projection", default="outputs/m1m2_recovery/guarded_zero_cost_projection/guarded_projection_result.json")
    parser.add_argument("--measured-records", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/m0_measurement_records.json")
    parser.add_argument("--source-roster", default="outputs/m1m2_answerability_preserving/roster/answerability_preserving_roster.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_candidate_guided_v31/roster")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = build_roster(
        Path(args.schema_outputs),
        Path(args.guarded_projection),
        Path(args.measured_records),
        Path(args.source_roster),
        Path(args.output_dir),
    )
    print(f"wrote candidate-guided roster: {Path(args.output_dir) / 'candidate_guided_roster.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
