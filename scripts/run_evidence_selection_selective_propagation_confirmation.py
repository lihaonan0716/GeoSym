#!/usr/bin/env python
"""Run or dry-run the 48-job M1/M2 selective confirmation.

Safe default: dry-run only. Paid execution additionally requires
--execute-paid and an explicit cap-confirmation flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluator.adapters.base as adapter_base  # noqa: E402
from app.helpers import _create_llm_adapter, _get_saved_key  # noqa: E402
from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402
from scripts.run_evidence_selection_selective_propagation_pilot import (  # noqa: E402
    DEFAULT_MODEL_DISPLAY_NAME,
    build_jobs,
    condition_summary,
    controlled_text_only_call,
    load_json,
    records_by_id,
    stress_jobs_by_id,
    utc_now,
)
from scripts.run_public_benchmark_pilot import AttemptCounter  # noqa: E402
from scripts.selective_propagation import (  # noqa: E402
    SELECTIVE_CONDITIONS,
    answer_record_to_dict,
    parse_selective_answer,
    score_selective_answer,
)

SCHEMA_VERSION = "m1m2_selective_confirmation_result_v1"
CONFIRMATION_SAMPLE_CAP = 48
CONFIRMATION_LLM_CAP = CONFIRMATION_SAMPLE_CAP * len(SELECTIVE_CONDITIONS)


def review_confirmation_manifest(manifest: dict[str, Any], paid_cap_approved: bool = False) -> list[str]:
    findings: list[str] = []
    if manifest.get("stage_status") != "selective_confirmation_manifest_frozen_pending_approval":
        findings.append("confirmation_manifest_not_frozen_pending_approval")
    if manifest.get("paid_calls_executed") != 0:
        findings.append("manifest_already_executed_paid_calls")
    if manifest.get("requires_explicit_paid_cap_approval_before_execution") is not True:
        findings.append("explicit_paid_cap_approval_not_required_by_manifest")
    if manifest.get("approval_required_for_llm_http_attempt_cap") != CONFIRMATION_LLM_CAP:
        findings.append(f"approval_cap_not_{CONFIRMATION_LLM_CAP}")
    if manifest.get("input_mode") != "evidence_only":
        findings.append("input_mode_not_evidence_only")
    if manifest.get("sample_level_job_cap") != CONFIRMATION_SAMPLE_CAP:
        findings.append("sample_cap_not_48")
    if tuple(manifest.get("condition_list") or []) != SELECTIVE_CONDITIONS:
        findings.append("conditions_not_frozen")
    if manifest.get("llm_http_attempt_cap") != CONFIRMATION_LLM_CAP:
        findings.append(f"llm_cap_not_{CONFIRMATION_LLM_CAP}")
    if manifest.get("ocr_http_attempt_cap") != 0:
        findings.append("ocr_cap_not_zero")
    if manifest.get("retry_cap_per_sample_condition") != 1:
        findings.append("retry_cap_not_1")
    if manifest.get("max_output_tokens") != 256:
        findings.append("max_output_tokens_not_256")
    if not manifest.get("estimated_cost_usd_upper_bound"):
        findings.append("paid_cost_estimate_unavailable")
    if len(manifest.get("selected_roster") or []) != CONFIRMATION_SAMPLE_CAP:
        findings.append("roster_not_48")
    family_counts = manifest.get("perturbation_family_counts") or {}
    if sorted(family_counts.values()) != [8, 8, 8, 8, 8, 8]:
        findings.append("family_counts_not_8_each")
    expected_counts = manifest.get("expected_answerability_counts") or {}
    if expected_counts.get("answerable") != 8:
        findings.append("answerable_count_not_8")
    if expected_counts.get("contradiction") != 8:
        findings.append("contradiction_count_not_8")
    if expected_counts.get("insufficient_evidence") != 32:
        findings.append("insufficient_evidence_count_not_32")
    if paid_cap_approved is not True:
        findings.append(f"explicit_{CONFIRMATION_LLM_CAP}_paid_cap_approval_missing")
    return findings


def build_confirmation_dry_run(
    manifest_path: Path,
    measured_records_path: Path,
    stress_roster_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    records = records_by_id(load_json(measured_records_path))
    stress_jobs = stress_jobs_by_id(load_json(stress_roster_path))
    findings = review_confirmation_manifest(manifest, paid_cap_approved=True)
    jobs, job_findings = build_jobs(manifest, records, stress_jobs)
    findings.extend(job_findings)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_selective_confirmation_dry_run",
        "stage_status": "selective_confirmation_dry_run_ready_pending_approval" if not findings else "selective_confirmation_dry_run_failed_closed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "requires_explicit_paid_cap_approval_before_execution": True,
        "approved_paid_cap_for_this_dry_run": False,
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
        "next_required_action": f"request_explicit_{CONFIRMATION_LLM_CAP}_attempt_paid_cap_approval" if not findings else "repair_selective_confirmation_manifest_or_sources",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dry_run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def run_confirmation_paid(
    manifest_path: Path,
    measured_records_path: Path,
    stress_roster_path: Path,
    output_dir: Path,
    model_display_name: str,
    paid_cap_approved: bool,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    findings = review_confirmation_manifest(manifest, paid_cap_approved=paid_cap_approved)
    if findings:
        raise ValueError(f"confirmation manifest failed review: {findings}")
    records = records_by_id(load_json(measured_records_path))
    stress_jobs = stress_jobs_by_id(load_json(stress_roster_path))
    jobs, job_findings = build_jobs(manifest, records, stress_jobs)
    if job_findings:
        raise ValueError(f"confirmation job construction failed review: {job_findings}")
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
            counter.consume_llm(job["stress_id"], job["condition"], "selective_confirmation_answer")
            raw_text = ""
            error = ""
            try:
                raw_text = controlled_text_only_call(adapter, job["prompt"], max_output_tokens)
                parsed = parse_selective_answer(raw_text, str(job.get("answer_type")))
            except Exception as exc:
                error = str(exc)
                parsed = parse_selective_answer("", str(job.get("answer_type")))
            score = score_selective_answer(parsed, job.get("reference"), str(job.get("answer_type")), str(job["expected_answerability"]))
            results.append(
                {
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
            )
            print(f"completed_selective_confirmation={job['stress_id']} condition={job['condition']} llm_attempts={counter.llm_attempts}/{manifest['llm_http_attempt_cap']}", flush=True)
    finally:
        adapter_base._MAX_API_RETRIES = original_retries
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_selective_confirmation",
        "stage_status": "selective_confirmation_completed",
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
            "This is the frozen 48-job evidence-only confirmation.",
            "No original chart image is passed to the answer model.",
            "OCR HTTP attempts are fixed at 0.",
            "No automatic expansion is allowed after this run.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paid_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/m1m2_selective_propagation/confirmation_manifest/frozen_confirmation_manifest.json")
    parser.add_argument("--measured-records", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/m0_measurement_records.json")
    parser.add_argument("--stress-roster", default="outputs/m1m2_selective_propagation/stress_roster/selective_stress_roster.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_selective_propagation/confirmation_run")
    parser.add_argument("--model-display-name", default=DEFAULT_MODEL_DISPLAY_NAME)
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--confirm-paid-cap-288", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest_path = Path(args.manifest)
    records_path = Path(args.measured_records)
    stress_path = Path(args.stress_roster)
    output_dir = Path(args.output_dir)
    if not args.execute_paid:
        result = build_confirmation_dry_run(manifest_path, records_path, stress_path, output_dir)
        print(f"wrote selective confirmation dry run: {output_dir / 'dry_run.json'}")
        print(f"stage_status={result['stage_status']}")
        print("paid_calls_executed=0")
        return 0
    result = run_confirmation_paid(
        manifest_path,
        records_path,
        stress_path,
        output_dir,
        args.model_display_name,
        paid_cap_approved=args.confirm_paid_cap_288,
    )
    print(f"wrote selective confirmation result: {output_dir / 'paid_result.json'}")
    print(f"stage_status={result['stage_status']}")
    print(f"paid_calls_executed={result['paid_calls_executed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
