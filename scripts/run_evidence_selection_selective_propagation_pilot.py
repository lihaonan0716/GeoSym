#!/usr/bin/env python
"""Run or dry-run the bounded M1/M2 selective propagation paid pilot.

Safe default: dry-run only. Use --execute-paid to spend the frozen cap.
"""

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

import evaluator.adapters.base as adapter_base  # noqa: E402
from app.helpers import _create_llm_adapter, _get_saved_key  # noqa: E402
from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402
from scripts.run_public_benchmark_pilot import AttemptCounter  # noqa: E402
from scripts.selective_propagation import (  # noqa: E402
    SELECTIVE_CONDITIONS,
    answer_record_to_dict,
    estimate_tokens_for_prompt,
    facts_for_selective_condition,
    parse_selective_answer,
    score_selective_answer,
    selective_answer_prompt,
)

SCHEMA_VERSION = "m1m2_selective_paid_pilot_result_v1"
DEFAULT_MODEL_DISPLAY_NAME = "GPT-5.5"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["sample_id"]): record for record in records}


def stress_jobs_by_id(stress_roster: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["stress_id"]): row for row in stress_roster.get("jobs", [])}


def review_manifest(manifest: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    expected_llm_cap = int(manifest.get("sample_level_job_cap") or 0) * len(SELECTIVE_CONDITIONS)
    if manifest.get("stage_status") != "selective_paid_pilot_manifest_frozen":
        findings.append("manifest_not_frozen")
    if manifest.get("paid_calls_executed") != 0:
        findings.append("manifest_already_executed_paid_calls")
    if manifest.get("input_mode") != "evidence_only":
        findings.append("input_mode_not_evidence_only")
    if manifest.get("sample_level_job_cap") != 12:
        findings.append("sample_cap_not_12")
    if tuple(manifest.get("condition_list") or []) != SELECTIVE_CONDITIONS:
        findings.append("conditions_not_frozen")
    if manifest.get("llm_http_attempt_cap") != expected_llm_cap:
        findings.append(f"llm_cap_not_{expected_llm_cap}")
    if manifest.get("ocr_http_attempt_cap") != 0:
        findings.append("ocr_cap_not_zero")
    if manifest.get("retry_cap_per_sample_condition") != 1:
        findings.append("retry_cap_not_1")
    if manifest.get("max_output_tokens") != 256:
        findings.append("max_output_tokens_not_256")
    if not manifest.get("estimated_cost_usd_upper_bound"):
        findings.append("paid_cost_estimate_unavailable")
    if len(manifest.get("selected_roster") or []) != 12:
        findings.append("roster_not_12")
    family_counts = manifest.get("perturbation_family_counts") or {}
    answerability_counts = manifest.get("expected_answerability_counts") or {}
    if manifest.get("redesign_name") in {
        "answerability_preserving_selective_propagation_v2",
        "candidate_guided_selective_propagation_v3",
        "candidate_guided_selective_propagation_v3_1",
    }:
        if answerability_counts.get("answerable") != 6:
            findings.append("redesign_answerable_count_not_6")
        if sum(int(value) for value in family_counts.values()) != 12:
            findings.append("redesign_family_counts_do_not_sum_to_12")
    elif sorted(family_counts.values()) != [2, 2, 2, 2, 2, 2]:
        findings.append("family_counts_not_2_each")
    return findings


def controlled_text_only_call(adapter: Any, prompt: str, max_output_tokens: int) -> str:
    if not hasattr(adapter, "_get_client") or not hasattr(adapter, "model"):
        raise ValueError("adapter_does_not_support_capped_text_only_call")
    client = adapter._get_client()
    response = client.chat.completions.create(
        model=adapter.model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_output_tokens,
        temperature=0.0,
        seed=42,
    )
    return response.choices[0].message.content or ""


def build_jobs(manifest: dict[str, Any], records: dict[str, dict[str, Any]], stress_jobs: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    findings = []
    jobs = []
    for item in manifest.get("selected_roster") or []:
        stress_id = str(item["stress_id"])
        source_sample_id = str(item["source_sample_id"])
        record = records.get(source_sample_id)
        stress_job = stress_jobs.get(stress_id)
        if record is None:
            findings.append(f"missing_measured_record:{source_sample_id}")
            continue
        if stress_job is None:
            findings.append(f"missing_stress_job:{stress_id}")
            continue
        for condition in SELECTIVE_CONDITIONS:
            facts = facts_for_selective_condition(stress_job["stress_schema"], condition, int(manifest["random_seed"]), stress_id)
            prompt = selective_answer_prompt(record["question"], facts, condition, str(stress_job["perturbation_family"]))
            image_path = str(record.get("image_path", "") or "")
            if "image_path" in prompt or (image_path and image_path in prompt):
                findings.append(f"image_path_leaked_into_prompt:{stress_id}:{condition}")
            jobs.append(
                {
                    "stress_id": stress_id,
                    "source_sample_id": source_sample_id,
                    "perturbation_family": stress_job["perturbation_family"],
                    "condition": condition,
                    "expected_answerability": stress_job["expected_answerability"],
                    "answer_type": record.get("answer_type"),
                    "reference": record.get("reference"),
                    "prompt": prompt,
                    "prompt_estimated_tokens": estimate_tokens_for_prompt(prompt),
                }
            )
    if len(jobs) != manifest.get("llm_http_attempt_cap"):
        findings.append("job_count_does_not_match_llm_cap")
    return jobs, findings


def condition_summary(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_condition[str(row["condition"])].append(row)
    return {
        condition: {
            "schema_valid": sum(1 for row in rows if row.get("schema_valid")),
            "answerable": sum(1 for row in rows if row.get("answerable")),
            "prediction_parseable": sum(1 for row in rows if row.get("prediction_parseable")),
            "correct_answer": sum(1 for row in rows if row.get("correct_answer")),
            "correct_abstention": sum(1 for row in rows if row.get("correct_abstention")),
            "task_success": sum(1 for row in rows if row.get("task_success")),
            "harmful_propagation": sum(1 for row in rows if row.get("harmful_propagation")),
            "protocol_violation": sum(1 for row in rows if row.get("answer_status") == "protocol_violation"),
        }
        for condition, rows in sorted(by_condition.items())
    }


def build_dry_run(manifest_path: Path, measured_records_path: Path, stress_roster_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    records = records_by_id(load_json(measured_records_path))
    stress_jobs = stress_jobs_by_id(load_json(stress_roster_path))
    findings = review_manifest(manifest)
    jobs, job_findings = build_jobs(manifest, records, stress_jobs)
    findings.extend(job_findings)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_selective_paid_pilot_dry_run",
        "stage_status": "selective_paid_pilot_dry_run_passed" if not findings else "selective_paid_pilot_dry_run_failed_closed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "input_mode": "evidence_only",
        "source_artifacts": {
            "manifest": {"path": str(manifest_path.resolve()), "sha256": stable_file_hash(manifest_path), "stage_status": manifest.get("stage_status")},
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
            "stress_roster": {"path": str(stress_roster_path.resolve()), "sha256": stable_file_hash(stress_roster_path)},
        },
        "blocking_findings": findings,
        "job_count": len(jobs),
        "max_prompt_estimated_tokens": max((job["prompt_estimated_tokens"] for job in jobs), default=0),
        "jobs": [{key: value for key, value in job.items() if key != "prompt"} for job in jobs],
        "next_required_action": "execute_bounded_selective_paid_pilot" if not findings else "repair_selective_paid_pilot_manifest_or_sources",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dry_run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def run_paid(manifest_path: Path, measured_records_path: Path, stress_roster_path: Path, output_dir: Path, model_display_name: str) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    findings = review_manifest(manifest)
    if findings:
        raise ValueError(f"manifest failed review: {findings}")
    records = records_by_id(load_json(measured_records_path))
    stress_jobs = stress_jobs_by_id(load_json(stress_roster_path))
    jobs, job_findings = build_jobs(manifest, records, stress_jobs)
    if job_findings:
        raise ValueError(f"job construction failed review: {job_findings}")
    key = _get_saved_key(model_display_name)
    adapter = _create_llm_adapter(model_display_name, key) if key else None
    if adapter is None:
        raise ValueError(f"could not create adapter for {model_display_name}")
    max_output_tokens = int(manifest["max_output_tokens"])
    counter = AttemptCounter(int(manifest["llm_http_attempt_cap"]))
    results = []
    original_retries = adapter_base._MAX_API_RETRIES
    adapter_base._MAX_API_RETRIES = 1
    try:
        for job in jobs:
            counter.consume_llm(job["stress_id"], job["condition"], "selective_answer")
            raw_text = ""
            error = ""
            try:
                raw_text = controlled_text_only_call(adapter, job["prompt"], max_output_tokens)
                parsed = parse_selective_answer(raw_text, str(job.get("answer_type")))
            except Exception as exc:
                error = str(exc)
                parsed = parse_selective_answer("", str(job.get("answer_type")))
            score = score_selective_answer(parsed, job.get("reference"), str(job.get("answer_type")), str(job["expected_answerability"]))
            row = {
                "stress_id": job["stress_id"],
                "source_sample_id": job["source_sample_id"],
                "perturbation_family": job["perturbation_family"],
                "condition": job["condition"],
                "expected_answerability": job["expected_answerability"],
                "answer_type": job.get("answer_type"),
                "reference": job.get("reference"),
                "input_mode": "evidence_only",
                "llm_http_attempts": 1,
                "ocr_http_attempts": 0,
                "prompt_estimated_tokens": job["prompt_estimated_tokens"],
                "error": error,
                **answer_record_to_dict(parsed),
                **score,
            }
            results.append(row)
            print(f"completed_selective={job['stress_id']} condition={job['condition']} llm_attempts={counter.llm_attempts}/{manifest['llm_http_attempt_cap']}", flush=True)
    finally:
        adapter_base._MAX_API_RETRIES = original_retries
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_selective_paid_pilot",
        "stage_status": "selective_paid_pilot_completed",
        "paid_calls_executed": counter.llm_attempts,
        "actual_llm_http_attempts": counter.llm_attempts,
        "actual_ocr_http_attempts": 0,
        "model_display_name": model_display_name,
        "input_mode": "evidence_only",
        "max_output_tokens": max_output_tokens,
        "source_artifacts": {
            "manifest": {"path": str(manifest_path.resolve()), "sha256": stable_file_hash(manifest_path)},
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
            "stress_roster": {"path": str(stress_roster_path.resolve()), "sha256": stable_file_hash(stress_roster_path)},
        },
        "summary": condition_summary(results),
        "results": results,
        "interpretation_limits": [
            "This is a 12-job bounded evidence-only pilot and must not be expanded automatically.",
            "No original chart image is passed to the answer model.",
            "OCR HTTP attempts are fixed at 0.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paid_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/m1m2_selective_propagation/paid_pilot_manifest/frozen_manifest.json")
    parser.add_argument("--measured-records", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/m0_measurement_records.json")
    parser.add_argument("--stress-roster", default="outputs/m1m2_selective_propagation/stress_roster/selective_stress_roster.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_selective_propagation/paid_pilot")
    parser.add_argument("--model-display-name", default=DEFAULT_MODEL_DISPLAY_NAME)
    parser.add_argument("--execute-paid", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest_path = Path(args.manifest)
    records_path = Path(args.measured_records)
    stress_path = Path(args.stress_roster)
    output_dir = Path(args.output_dir)
    if not args.execute_paid:
        result = build_dry_run(manifest_path, records_path, stress_path, output_dir)
        print(f"wrote selective paid pilot dry run: {output_dir / 'dry_run.json'}")
        print(f"stage_status={result['stage_status']}")
        print("paid_calls_executed=0")
        return 0
    result = run_paid(manifest_path, records_path, stress_path, output_dir, args.model_display_name)
    print(f"wrote selective paid pilot result: {output_dir / 'paid_result.json'}")
    print(f"stage_status={result['stage_status']}")
    print(f"paid_calls_executed={result['paid_calls_executed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
