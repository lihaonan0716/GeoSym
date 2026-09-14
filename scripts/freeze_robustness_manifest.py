#!/usr/bin/env python
"""Freeze the paper-level 128-job M1/M2 stress confirmation manifest.

This script is zero-cost. It creates a deterministic roster and an
approval-gated manifest; it never executes paid calls.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_robustness_source_pool import (  # noqa: E402
    PRIMARY_TARGET,
    deterministic_value,
    operation_tag,
    question_matched_points,
)
from scripts.freeze_evidence_selection_candidate_guided_confirmation_manifest import load_default_model_provenance, model_registry_value  # noqa: E402
from scripts.freeze_evidence_selection_selective_propagation_pilot_manifest import (  # noqa: E402
    INPUT_TOKEN_LIMIT,
    LOCAL_PRICE_INPUT_PER_MILLION,
    LOCAL_PRICE_OUTPUT_PER_MILLION,
    MAX_OUTPUT_TOKENS,
    SEED,
    estimate_tokens,
    stable_file_hash_from_text,
)
from scripts.prepare_public_benchmark_pilot import stable_file_hash, stable_payload_hash  # noqa: E402
from scripts.run_plotqa_measured_locate_smoke import selected_record_points_for_question  # noqa: E402
from scripts.selective_propagation import SELECTIVE_CONDITIONS, facts_for_selective_condition, selective_answer_prompt  # noqa: E402

SCHEMA_VERSION = "m1m2_stress_128_manifest_v1"
ROSTER_SCHEMA_VERSION = "m1m2_stress_128_roster_v1"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "paper_level_evidence_package" / "m1m2_stress_128"
DEFAULT_SOURCE_POOL = DEFAULT_OUTPUT_DIR / "source_pool_feasibility.json"
DEFAULT_RECORDS = ROOT / "outputs" / "paper_level_evidence_package" / "plotqa_public_500" / "m0_measurement_records.json"

ANSWERABLE_OPERATION_QUOTAS = {
    "lookup": 10,
    "threshold_count": 10,
    "comparison": 10,
    "extrema": 10,
    "range_difference": 6,
    "ratio": 6,
}
ANSWERABLE_TARGET = 64
NON_ANSWERABLE_TARGET = 64
MAX_SOURCE_REUSE = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fact_from_point(point: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "fact_id": f"p{index:03d}",
        "key": "measured_point",
        "series_label": point.get("series_label"),
        "category_label": point.get("category_label"),
        "x_value": point.get("x_value"),
        "y_value": point.get("y_value"),
        "reliability_tier": "high",
        "source": "plotqa_measured_m0_record",
    }


def candidate_from_record(record: dict[str, Any]) -> Any:
    value = deterministic_value(record)
    if value is not None:
        return value
    tag = operation_tag(str(record.get("question", "")))
    if tag == "lookup":
        points = question_matched_points(record)
        for point in points:
            if isinstance(point.get("y_value"), (int, float)):
                return point.get("y_value")
    if tag == "ratio":
        points = [
            point
            for point in selected_record_points_for_question(
                record,
                {"question": record.get("question", ""), "answer_type": record.get("answer_type")},
            )
            if isinstance(point.get("y_value"), (int, float))
        ]
        if len(points) >= 2 and float(points[0]["y_value"]) != 0.0:
            return float(points[1]["y_value"]) / float(points[0]["y_value"])
    return None


def measured_facts(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [fact_from_point(point, idx) for idx, point in enumerate((record.get("measured_points") or [])[:24])]


def answerable_job(record: dict[str, Any], index: int) -> dict[str, Any] | None:
    facts = measured_facts(record)
    candidate = candidate_from_record(record)
    if not facts or candidate is None:
        return None
    selected_ids = [fact["fact_id"] for fact in facts[: min(8, len(facts))]]
    stress_schema = {
        "condition": "G_candidate_guided_answerable_schema",
        "expected_answerability": "answerable",
        "redesign_hypothesis": "candidate_guided_selective_propagation_v3_1",
        "operation_tag": operation_tag(str(record.get("question", ""))),
        "evidence_candidates": facts,
        "evidence_gaps": [],
        "selected_evidence_set": {
            "selected_fact_ids": selected_ids,
            "suppressed_fact_ids": [fact["fact_id"] for fact in facts if fact["fact_id"] not in selected_ids],
            "surfaced_gap_ids": [],
            "blocking_gap_keys": [],
            "required_fact_coverage_complete": True,
            "propagation_policy": "answer_with_candidate_and_selected_facts",
            "answer_candidate": {
                "normalized_prediction": candidate,
                "answer_type": record.get("answer_type"),
                "source": "deterministic_m3_from_measured_facts",
                "supporting_fact_ids": selected_ids,
                "selection_rule": "answerable_source_eligible_and_generic_measured_fact_candidate",
            },
        },
    }
    stress_schema["stress_hash"] = stable_payload_hash(stress_schema)
    return {
        "stress_id": f"m1m2_stress_128_answerable_{index:03d}",
        "source": "plotqa_public_500_measured_record",
        "source_sample_id": record.get("sample_id"),
        "source_reuse_index": 0,
        "perturbation_family": "candidate_supported_selected_facts",
        "operation_tag": stress_schema["operation_tag"],
        "seed": SEED,
        "answer_independent_selection": True,
        "expected_answerability": "answerable",
        "stress_schema": stress_schema,
        "stress_hash": stress_schema["stress_hash"],
    }


def non_answerable_job(record: dict[str, Any], index: int, reuse_index: int) -> dict[str, Any] | None:
    facts = measured_facts(record)
    if not facts:
        return None
    selected_ids = [facts[0]["fact_id"]]
    gap_id = f"gap_missing_{index:03d}"
    stress_schema = {
        "condition": "G_candidate_guided_missing_required_fact_schema",
        "expected_answerability": "insufficient_evidence",
        "redesign_hypothesis": "candidate_guided_selective_propagation_v3_1",
        "operation_tag": operation_tag(str(record.get("question", ""))),
        "evidence_candidates": facts,
        "evidence_gaps": [
            {
                "gap_id": gap_id,
                "key": "required_measured_value",
                "needed_for_question": True,
                "severity": "blocking",
                "reason": "stress_test_hides_required_support_fact",
            }
        ],
        "selected_evidence_set": {
            "selected_fact_ids": selected_ids,
            "suppressed_fact_ids": [fact["fact_id"] for fact in facts if fact["fact_id"] not in selected_ids],
            "surfaced_gap_ids": [gap_id],
            "blocking_gap_keys": ["required_measured_value"],
            "required_fact_coverage_complete": False,
            "propagation_policy": "abstain_insufficient_evidence",
            "answer_candidate": None,
        },
    }
    stress_schema["stress_hash"] = stable_payload_hash(stress_schema)
    return {
        "stress_id": f"m1m2_stress_128_control_{index:03d}",
        "source": "plotqa_public_500_measured_record",
        "source_sample_id": record.get("sample_id"),
        "source_reuse_index": reuse_index,
        "perturbation_family": "missing_required_fact",
        "operation_tag": stress_schema["operation_tag"],
        "seed": SEED,
        "answer_independent_selection": True,
        "expected_answerability": "insufficient_evidence",
        "stress_schema": stress_schema,
        "stress_hash": stress_schema["stress_hash"],
    }


def select_answerable(records: dict[str, dict[str, Any]], source_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    findings: list[str] = []
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    by_op: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        if row.get("answerable_source_eligible"):
            by_op[str(row.get("operation_tag"))].append(row)
    for op, quota in ANSWERABLE_OPERATION_QUOTAS.items():
        op_rows = sorted(by_op.get(op, []), key=lambda row: str(row.get("stress_source_key")))
        for row in op_rows:
            if len([job for job in selected if job["operation_tag"] == op]) >= quota:
                break
            sample_id = str(row.get("sample_id"))
            if sample_id in used:
                continue
            job = answerable_job(records[sample_id], len(selected))
            if job is None:
                continue
            selected.append(job)
            used.add(sample_id)
        if len([job for job in selected if job["operation_tag"] == op]) < quota:
            findings.append(f"answerable_operation_quota_underfilled:{op}")
    for row in sorted([row for rows in by_op.values() for row in rows], key=lambda row: str(row.get("stress_source_key"))):
        if len(selected) >= ANSWERABLE_TARGET:
            break
        sample_id = str(row.get("sample_id"))
        if sample_id in used:
            continue
        job = answerable_job(records[sample_id], len(selected))
        if job is None:
            continue
        selected.append(job)
        used.add(sample_id)
    if len(selected) != ANSWERABLE_TARGET:
        findings.append(f"answerable_target_underfilled:{len(selected)}")
    return selected, findings


def select_non_answerable(records: dict[str, dict[str, Any]], source_rows: list[dict[str, Any]], excluded_sample_ids: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    findings: list[str] = []
    selected: list[dict[str, Any]] = []
    reuse_counts: Counter[str] = Counter()
    candidates = [
        row
        for row in source_rows
        if row.get("non_answerable_source_eligible") and str(row.get("sample_id")) not in excluded_sample_ids
    ]
    for row in sorted(candidates, key=lambda row: (str(row.get("operation_tag")), str(row.get("stress_source_key")))):
        if len(selected) >= NON_ANSWERABLE_TARGET:
            break
        sample_id = str(row.get("sample_id"))
        if reuse_counts[sample_id] >= MAX_SOURCE_REUSE:
            continue
        job = non_answerable_job(records[sample_id], len(selected), reuse_counts[sample_id])
        if job is None:
            continue
        selected.append(job)
        reuse_counts[sample_id] += 1
    if len(selected) != NON_ANSWERABLE_TARGET:
        findings.append(f"non_answerable_target_underfilled:{len(selected)}")
    return selected, findings


def build_manifest(source_pool_path: Path, measured_records_path: Path, output_dir: Path, model_display_name: str | None = None) -> dict[str, Any]:
    source_pool = load_json(source_pool_path)
    records_list = load_json(measured_records_path)
    records = {str(row["sample_id"]): row for row in records_list}
    findings: list[str] = []
    if source_pool.get("stage_status") != "m1m2_stress_128_source_pool_feasible":
        findings.append(f"source_pool_not_128_feasible:{source_pool.get('stage_status')}")
    answerable, answerable_findings = select_answerable(records, source_pool.get("source_rows") or [])
    findings.extend(answerable_findings)
    non_answerable, non_answerable_findings = select_non_answerable(
        records,
        source_pool.get("source_rows") or [],
        {str(job["source_sample_id"]) for job in answerable},
    )
    findings.extend(non_answerable_findings)
    jobs = answerable + non_answerable
    prompt_token_estimates: dict[str, list[int]] = {condition: [] for condition in SELECTIVE_CONDITIONS}
    prompt_hashes: dict[str, list[str]] = {condition: [] for condition in SELECTIVE_CONDITIONS}
    selected_roster = []
    for job in jobs:
        record = records[str(job["source_sample_id"])]
        for condition in SELECTIVE_CONDITIONS:
            facts = facts_for_selective_condition(job["stress_schema"], condition, SEED, str(job["stress_id"]))
            prompt = selective_answer_prompt(str(record["question"]), facts, condition, str(job["perturbation_family"]))
            if str(record.get("sample_id")) in prompt or "reference" in prompt.lower():
                findings.append(f"forbidden_identifier_or_reference_leak:{job['stress_id']}:{condition}")
            tokens = estimate_tokens(prompt)
            prompt_token_estimates[condition].append(tokens)
            if tokens > INPUT_TOKEN_LIMIT:
                findings.append(f"prompt_token_limit_exceeded:{job['stress_id']}:{condition}")
            prompt_hashes[condition].append(stable_file_hash_from_text(prompt))
        selected_roster.append(
            {
                "stress_id": job["stress_id"],
                "source_sample_id_sha256": stable_payload_hash({"sample_id": job["source_sample_id"]}),
                "perturbation_family": job["perturbation_family"],
                "operation_tag": job["operation_tag"],
                "expected_answerability": job["expected_answerability"],
                "stress_hash": job["stress_hash"],
            }
        )
    llm_cap = len(selected_roster) * len(SELECTIVE_CONDITIONS)
    total_input_tokens = sum(sum(values) for values in prompt_token_estimates.values())
    estimated_cost = (
        total_input_tokens * LOCAL_PRICE_INPUT_PER_MILLION / 1_000_000
        + llm_cap * MAX_OUTPUT_TOKENS * LOCAL_PRICE_OUTPUT_PER_MILLION / 1_000_000
    )
    model_provenance = load_default_model_provenance()
    if model_display_name is not None:
        model_provenance.update(
            {
                "model_display_name": model_display_name,
                "model_registry_id_or_config_value": model_registry_value(model_display_name),
            }
        )
    if not model_provenance.get("model_registry_id_or_config_value"):
        findings.append("model_registry_id_unavailable")
    counts = Counter(job["expected_answerability"] for job in jobs)
    operation_counts = Counter(job["operation_tag"] for job in answerable)
    max_input_tokens = max((max(values) for values in prompt_token_estimates.values() if values), default=0)
    gates = {
        "source_pool_128_feasible": source_pool.get("stage_status") == "m1m2_stress_128_source_pool_feasible",
        "selected_jobs_128": len(selected_roster) == PRIMARY_TARGET,
        "answerable_jobs_64": counts.get("answerable") == ANSWERABLE_TARGET,
        "non_answerable_jobs_64": counts.get("insufficient_evidence") == NON_ANSWERABLE_TARGET,
        "answerable_operation_quotas_met": all(operation_counts.get(op, 0) >= quota for op, quota in ANSWERABLE_OPERATION_QUOTAS.items()),
        "conditions_7": len(SELECTIVE_CONDITIONS) == 7,
        "llm_cap_896": llm_cap == 896,
        "ocr_cap_zero": True,
        "max_prompt_tokens_within_limit": max_input_tokens <= INPUT_TOKEN_LIMIT,
        "blocking_findings_empty": not findings,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_stress_128_manifest",
        "stage_status": "m1m2_stress_128_manifest_frozen_pending_paid_cap_approval" if all(gates.values()) else "m1m2_stress_128_manifest_failed_closed",
        "redesign_name": "candidate_guided_selective_propagation_v3_1_paper_level_128",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "requires_explicit_paid_cap_approval_before_execution": True,
        "approval_required_for_llm_http_attempt_cap": llm_cap,
        "input_mode": "evidence_only",
        "sample_level_job_cap": len(selected_roster),
        "condition_list": list(SELECTIVE_CONDITIONS),
        "llm_http_attempt_cap": llm_cap,
        "ocr_http_attempt_cap": 0,
        "retry_cap_per_sample_condition": 1,
        "model_provenance": model_provenance,
        "temperature_or_decoding_note": model_provenance["temperature_or_decoding_note"],
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "prompt_token_estimates_by_condition": {
            condition: {"max": max(values) if values else 0, "sum": sum(values), "count": len(values)}
            for condition, values in prompt_token_estimates.items()
        },
        "estimated_cost_usd_upper_bound": round(estimated_cost, 4),
        "pricing_source_or_registry_snapshot": {
            "source": "local_protocol_default_estimate",
            "input_usd_per_million_tokens": LOCAL_PRICE_INPUT_PER_MILLION,
            "output_usd_per_million_tokens": LOCAL_PRICE_OUTPUT_PER_MILLION,
        },
        "prompt_template_hashes_by_condition": {
            condition: {"count": len(values), "unique_count": len(set(values)), "first_hash": values[0] if values else None}
            for condition, values in prompt_hashes.items()
        },
        "source_artifact_hashes": {
            "source_pool_feasibility": stable_file_hash(source_pool_path),
            "measured_records": stable_file_hash(measured_records_path),
        },
        "selected_roster": selected_roster,
        "expected_answerability_counts": dict(sorted(counts.items())),
        "answerable_operation_counts": dict(sorted(operation_counts.items())),
        "answerable_operation_quotas": ANSWERABLE_OPERATION_QUOTAS,
        "random_seed": SEED,
        "safe_default": "dry_run_only_unless_execute_paid_and_approved_cap_match",
        "retry_policy": "one HTTP attempt per sample-condition; failed attempts count against the cap",
        "cache_policy": "no semantic cache reuse is assumed for this confirmation; prompt hashes are frozen for audit",
        "stop_rule": "Stop after 128 sample-level jobs and 896 LLM HTTP attempts regardless of interim results.",
        "expansion_rule": "No automatic expansion or sample replacement beyond this manifest.",
        "frozen_hypothesis": "Candidate-guided selective propagation reduces harmful answer propagation under incomplete evidence while preserving answerable cases.",
        "falsification_signal": "If G_selective_propagation does not improve task_success over G_candidate_unrouted and G_no_routing, or increases harmful_propagation on non-answerable jobs, the M1/M2 mechanism claim is not supported by this stress confirmation.",
        "interpretation_limits": [
            "This stress confirmation tests candidate-guided propagation under constructed evidence perturbations, not broad PlotQA leaderboard performance.",
            "It cannot repair the broad GeoSym-Bench M1/M2 ablation by itself.",
            "All prompts are evidence-only and exclude chart images, reference answers, and raw sample identifiers.",
        ],
        "blocking_findings": sorted(set(findings)),
        "gates": gates,
        "next_required_action": "run_m1m2_stress_128_dry_run" if all(gates.values()) else "repair_m1m2_stress_128_manifest_inputs",
    }
    roster = {
        "schema_version": ROSTER_SCHEMA_VERSION,
        "created_at_utc": manifest["created_at_utc"],
        "stage": "m1m2_stress_128_roster",
        "stage_status": "m1m2_stress_128_roster_frozen" if all(gates.values()) else "m1m2_stress_128_roster_failed_closed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_manifest_sha256": stable_payload_hash(manifest),
        "jobs": jobs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stress_128_roster.json").write_text(json.dumps(roster, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "stress_128_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "stress_128_manifest.md").write_text(render_markdown(manifest), encoding="utf-8")
    return manifest


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Stress 128 Manifest",
        "",
        f"- Stage status: `{manifest['stage_status']}`",
        f"- Sample-level jobs: `{manifest['sample_level_job_cap']}`",
        f"- Conditions: `{len(manifest['condition_list'])}`",
        f"- LLM HTTP attempt cap: `{manifest['llm_http_attempt_cap']}`",
        f"- OCR HTTP attempt cap: `{manifest['ocr_http_attempt_cap']}`",
        f"- Estimated cost upper bound: `${manifest['estimated_cost_usd_upper_bound']}`",
        f"- Blocking findings: `{len(manifest['blocking_findings'])}`",
        "",
        "## Conditions",
        "",
    ]
    lines.extend(f"- `{condition}`" for condition in manifest["condition_list"])
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- `{key}`: `{value}`" for key, value in manifest["gates"].items())
    lines.extend(["", "## Next Required Action", "", f"- `{manifest['next_required_action']}`", ""])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pool", default=str(DEFAULT_SOURCE_POOL))
    parser.add_argument("--measured-records", default=str(DEFAULT_RECORDS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-display-name", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(
        Path(args.source_pool),
        Path(args.measured_records),
        Path(args.output_dir),
        args.model_display_name,
    )
    print(f"wrote M1/M2 stress 128 manifest: {Path(args.output_dir) / 'stress_128_manifest.json'}")
    print(f"stage_status={manifest['stage_status']}")
    print(f"llm_http_attempt_cap={manifest['llm_http_attempt_cap']}")
    print("paid_calls_executed=0")
    return 0 if manifest["stage_status"] == "m1m2_stress_128_manifest_frozen_pending_paid_cap_approval" else 1


if __name__ == "__main__":
    raise SystemExit(main())
