#!/usr/bin/env python
"""Run or dry-run the candidate-guided M1/M2 confirmation.

Safe default: dry-run only. Paid execution requires both --execute-paid and an
--approved-llm-cap value matching the frozen manifest cap.
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
from scripts.freeze_evidence_selection_candidate_guided_confirmation_manifest import load_default_model_provenance, model_registry_value  # noqa: E402
from scripts.run_evidence_selection_selective_propagation_pilot import (  # noqa: E402
    controlled_text_only_call,
    records_by_id,
    stress_jobs_by_id,
)
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

SCHEMA_VERSION = "m1m2_candidate_guided_confirmation_result_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def review_confirmation_manifest(manifest: dict[str, Any], approved_llm_cap: int | None = None, for_paid: bool = False) -> list[str]:
    findings: list[str] = []
    expected_llm_cap = int(manifest.get("sample_level_job_cap") or 0) * len(SELECTIVE_CONDITIONS)
    if manifest.get("stage_status") != "candidate_guided_confirmation_manifest_frozen_pending_paid_cap_approval":
        findings.append("manifest_not_pending_paid_cap_approval")
    if manifest.get("redesign_name") != "candidate_guided_selective_propagation_v3_1":
        findings.append("redesign_not_v31")
    if manifest.get("paid_calls_executed") != 0:
        findings.append("manifest_already_executed_paid_calls")
    if manifest.get("input_mode") != "evidence_only":
        findings.append("input_mode_not_evidence_only")
    if manifest.get("sample_level_job_cap") != 16:
        findings.append("sample_cap_not_16")
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
    if manifest.get("safe_default") != "do_not_execute_without_explicit_paid_cap_approval":
        findings.append("safe_default_not_approval_gated")
    if len(manifest.get("selected_roster") or []) != 16:
        findings.append("roster_not_16")
    answerability = manifest.get("expected_answerability_counts") or {}
    if answerability.get("answerable") != 8:
        findings.append("answerable_count_not_8")
    if for_paid and approved_llm_cap != manifest.get("llm_http_attempt_cap"):
        findings.append("approved_llm_cap_missing_or_mismatch")
    return findings


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


def build_dry_run(manifest_path: Path, measured_records_path: Path, confirmation_roster_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    records = records_by_id(load_json(measured_records_path))
    stress_jobs = stress_jobs_by_id(load_json(confirmation_roster_path))
    findings = review_confirmation_manifest(manifest)
    jobs, job_findings = build_jobs(manifest, records, stress_jobs)
    findings.extend(job_findings)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_candidate_guided_confirmation_dry_run",
        "stage_status": "candidate_guided_confirmation_dry_run_passed" if not findings else "candidate_guided_confirmation_dry_run_failed_closed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "input_mode": "evidence_only",
        "source_artifacts": {
            "manifest": {"path": str(manifest_path.resolve()), "sha256": stable_file_hash(manifest_path), "stage_status": manifest.get("stage_status")},
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
            "confirmation_roster": {"path": str(confirmation_roster_path.resolve()), "sha256": stable_file_hash(confirmation_roster_path)},
        },
        "blocking_findings": findings,
        "job_count": len(jobs),
        "max_prompt_estimated_tokens": max((job["prompt_estimated_tokens"] for job in jobs), default=0),
        "jobs": [{key: value for key, value in job.items() if key != "prompt"} for job in jobs],
        "next_required_action": f"execute_after_explicit_{manifest.get('llm_http_attempt_cap')}_call_approval" if not findings else "repair_candidate_guided_confirmation_inputs",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dry_run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def resolve_model_provenance(model_display_name: str | None) -> dict[str, Any]:
    provenance = load_default_model_provenance()
    if model_display_name is None:
        return provenance
    provenance.update(
        {
            "model_display_name": model_display_name,
            "model_registry_id_or_config_value": model_registry_value(model_display_name),
        }
    )
    return provenance


def run_paid(
    manifest_path: Path,
    measured_records_path: Path,
    confirmation_roster_path: Path,
    output_dir: Path,
    model_display_name: str | None,
    approved_llm_cap: int | None,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    findings = review_confirmation_manifest(manifest, approved_llm_cap=approved_llm_cap, for_paid=True)
    if findings:
        raise ValueError(f"confirmation manifest failed review: {findings}")
    model_provenance = resolve_model_provenance(model_display_name)
    resolved_model_display_name = model_provenance["model_display_name"]
    if not model_provenance.get("model_registry_id_or_config_value"):
        raise ValueError(f"could not resolve model registry id for {resolved_model_display_name}")
    records = records_by_id(load_json(measured_records_path))
    stress_jobs = stress_jobs_by_id(load_json(confirmation_roster_path))
    jobs, job_findings = build_jobs(manifest, records, stress_jobs)
    if job_findings:
        raise ValueError(f"confirmation job construction failed review: {job_findings}")
    key = _get_saved_key(resolved_model_display_name)
    adapter = _create_llm_adapter(resolved_model_display_name, key) if key else None
    if adapter is None:
        raise ValueError(f"could not create adapter for {resolved_model_display_name}")
    max_output_tokens = int(manifest["max_output_tokens"])
    counter = AttemptCounter(int(manifest["llm_http_attempt_cap"]))
    results = []
    original_retries = adapter_base._MAX_API_RETRIES
    adapter_base._MAX_API_RETRIES = 1
    try:
        for job in jobs:
            counter.consume_llm(job["stress_id"], job["condition"], "candidate_guided_confirmation")
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
            print(f"completed_candidate_guided_confirmation={job['stress_id']} condition={job['condition']} llm_attempts={counter.llm_attempts}/{manifest['llm_http_attempt_cap']}", flush=True)
    finally:
        adapter_base._MAX_API_RETRIES = original_retries
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_candidate_guided_confirmation",
        "stage_status": "candidate_guided_confirmation_completed",
        "paid_calls_executed": counter.llm_attempts,
        "actual_llm_http_attempts": counter.llm_attempts,
        "actual_ocr_http_attempts": 0,
        "model_display_name": resolved_model_display_name,
        "model_registry_id_or_config_value": model_provenance["model_registry_id_or_config_value"],
        "model_provenance": model_provenance,
        "temperature_or_decoding_note": model_provenance["temperature_or_decoding_note"],
        "input_mode": "evidence_only",
        "max_output_tokens": max_output_tokens,
        "source_artifacts": {
            "manifest": {"path": str(manifest_path.resolve()), "sha256": stable_file_hash(manifest_path)},
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
            "confirmation_roster": {"path": str(confirmation_roster_path.resolve()), "sha256": stable_file_hash(confirmation_roster_path)},
        },
        "summary": condition_summary(results),
        "results": results,
        "interpretation_limits": [
            "This is a 16-job evidence-only confirmation and must not be expanded automatically.",
            "No original chart image is passed to the answer model.",
            "OCR HTTP attempts are fixed at 0.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paid_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/m1m2_candidate_guided_v31/confirmation_manifest/frozen_confirmation_manifest.json")
    parser.add_argument("--measured-records", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/m0_measurement_records.json")
    parser.add_argument("--confirmation-roster", default="outputs/m1m2_candidate_guided_v31/confirmation_manifest/confirmation_roster.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_candidate_guided_v31/confirmation_run")
    parser.add_argument("--model-display-name", default=None, help="Override the config-resolved default display name.")
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--approved-llm-cap", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    records_path = Path(args.measured_records)
    roster_path = Path(args.confirmation_roster)
    output_dir = Path(args.output_dir)
    if not args.execute_paid:
        result = build_dry_run(manifest_path, records_path, roster_path, output_dir)
        print(f"wrote candidate-guided confirmation dry run: {output_dir / 'dry_run.json'}")
        print(f"stage_status={result['stage_status']}")
        print("paid_calls_executed=0")
        return 0
    result = run_paid(manifest_path, records_path, roster_path, output_dir, args.model_display_name, args.approved_llm_cap)
    print(f"wrote candidate-guided confirmation result: {output_dir / 'paid_result.json'}")
    print(f"stage_status={result['stage_status']}")
    print(f"paid_calls_executed={result['paid_calls_executed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
