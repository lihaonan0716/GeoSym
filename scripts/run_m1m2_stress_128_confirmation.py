#!/usr/bin/env python
"""Dry-run or execute the paper-level 128-job M1/M2 stress confirmation."""

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
from scripts.freeze_m1m2_stress_128_manifest import DEFAULT_OUTPUT_DIR  # noqa: E402
from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402
from scripts.run_m1m2_candidate_guided_confirmation import resolve_model_provenance  # noqa: E402
from scripts.run_m1m2_selective_paid_pilot import controlled_text_only_call, records_by_id, stress_jobs_by_id  # noqa: E402
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

SCHEMA_VERSION = "m1m2_stress_128_result_v1"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "stress_128_manifest.json"
DEFAULT_ROSTER = DEFAULT_OUTPUT_DIR / "stress_128_roster.json"
DEFAULT_RECORDS = ROOT / "outputs" / "paper_level_evidence_package" / "plotqa_public_500" / "m0_measurement_records.json"
DEFAULT_PAID_RESULT_NAME = "stress_128_paid_result.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def existing_paid_result_requires_reauthorization(result: dict[str, Any]) -> bool:
    status = str(result.get("stage_status") or "")
    blockers = [str(finding) for finding in (result.get("blocking_findings") or [])]
    unknown_attempts = result.get("actual_llm_http_attempts") is None or result.get("actual_ocr_http_attempts") is None
    unknown_blocker = any("unknown" in blocker.lower() or "unaccount" in blocker.lower() for blocker in blockers)
    interrupted_uncounted_status = "uncheckpointed" in status or "unaccount" in status
    return bool(unknown_attempts or unknown_blocker or interrupted_uncounted_status)


def guard_existing_paid_result(output_dir: Path, restart_after_unaccounted_interruption: bool) -> None:
    paid_result_path = output_dir / DEFAULT_PAID_RESULT_NAME
    if not paid_result_path.exists():
        return
    existing = load_json(paid_result_path)
    if existing_paid_result_requires_reauthorization(existing) and not restart_after_unaccounted_interruption:
        raise ValueError(
            "existing_paid_result_has_unknown_attempt_accounting;"
            " explicit_reauthorization_required_before_restart"
        )


def review_manifest(manifest: dict[str, Any], approved_llm_cap: int | None = None, for_paid: bool = False) -> list[str]:
    findings: list[str] = []
    if manifest.get("stage_status") != "m1m2_stress_128_manifest_frozen_pending_paid_cap_approval":
        findings.append("manifest_not_pending_paid_cap_approval")
    if manifest.get("redesign_name") != "candidate_guided_selective_propagation_v3_1_paper_level_128":
        findings.append("redesign_not_paper_level_128")
    if manifest.get("paid_calls_executed") != 0:
        findings.append("manifest_already_executed_paid_calls")
    if manifest.get("sample_level_job_cap") != 128:
        findings.append("sample_cap_not_128")
    if tuple(manifest.get("condition_list") or []) != SELECTIVE_CONDITIONS:
        findings.append("conditions_not_frozen")
    if len(SELECTIVE_CONDITIONS) != 7:
        findings.append("conditions_not_7")
    if manifest.get("llm_http_attempt_cap") != 896:
        findings.append("llm_cap_not_896")
    if manifest.get("ocr_http_attempt_cap") != 0:
        findings.append("ocr_cap_not_zero")
    if manifest.get("retry_cap_per_sample_condition") != 1:
        findings.append("retry_cap_not_1")
    if manifest.get("max_output_tokens") != 256:
        findings.append("max_output_tokens_not_256")
    if manifest.get("safe_default") != "dry_run_only_unless_execute_paid_and_approved_cap_match":
        findings.append("safe_default_not_dry_run_only")
    if len(manifest.get("selected_roster") or []) != 128:
        findings.append("roster_not_128")
    counts = manifest.get("expected_answerability_counts") or {}
    if counts.get("answerable") != 64:
        findings.append("answerable_count_not_64")
    if counts.get("insufficient_evidence") != 64:
        findings.append("insufficient_evidence_count_not_64")
    if for_paid and approved_llm_cap != manifest.get("llm_http_attempt_cap"):
        findings.append("approved_llm_cap_missing_or_mismatch")
    return findings


def build_jobs(manifest: dict[str, Any], records: dict[str, dict[str, Any]], stress_jobs: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    findings: list[str] = []
    jobs: list[dict[str, Any]] = []
    for item in manifest.get("selected_roster") or []:
        stress_id = str(item["stress_id"])
        stress_job = stress_jobs.get(stress_id)
        if stress_job is None:
            findings.append(f"missing_stress_job:{stress_id}")
            continue
        source_sample_id = str(stress_job["source_sample_id"])
        record = records.get(source_sample_id)
        if record is None:
            findings.append(f"missing_measured_record:{source_sample_id}")
            continue
        for condition in SELECTIVE_CONDITIONS:
            facts = facts_for_selective_condition(stress_job["stress_schema"], condition, int(manifest["random_seed"]), stress_id)
            prompt = selective_answer_prompt(str(record["question"]), facts, condition, str(stress_job["perturbation_family"]))
            image_path = str(record.get("image_path", "") or "")
            if "image_path" in prompt or (image_path and image_path in prompt) or source_sample_id in prompt or "reference" in prompt.lower():
                findings.append(f"forbidden_prompt_leak:{stress_id}:{condition}")
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
            "rows": len(rows),
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


def job_key(job: dict[str, Any]) -> tuple[str, str]:
    return str(job["stress_id"]), str(job["condition"])


def timeout_result_for_attempted_job(job: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        **{key: value for key, value in job.items() if key != "prompt"},
        "raw_text": "",
        "answer_status": "transport_failure",
        "schema_valid": False,
        "answerable": False,
        "prediction_parseable": False,
        "normalized_prediction": None,
        "numeric_error": None,
        "tolerance": None,
        "correct_answer": False,
        "correct_abstention": False,
        "task_success": False,
        "harmful_propagation": False,
        "supporting_fact_ids": [],
        "contradictory_fact_ids": [],
        "missing_fact_keys": [],
        "risk_reason": "transport failure before parseable model output",
        "parse_error": error,
    }


def write_paid_checkpoint(
    output_dir: Path,
    manifest_path: Path,
    measured_records_path: Path,
    roster_path: Path,
    manifest: dict[str, Any],
    model_provenance: dict[str, Any],
    results: list[dict[str, Any]],
    attempted_jobs: list[dict[str, Any]],
    counter: AttemptCounter,
    stage_status: str,
    error: str = "",
) -> dict[str, Any]:
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_stress_128_paid_confirmation",
        "stage_status": stage_status,
        "paid_calls_executed": len(results),
        "actual_llm_http_attempts": counter.llm_attempts,
        "actual_ocr_http_attempts": 0,
        "llm_http_attempt_cap": manifest["llm_http_attempt_cap"],
        "ocr_http_attempt_cap": manifest["ocr_http_attempt_cap"],
        "model_provenance": model_provenance,
        "source_artifacts": {
            "manifest": {"path": str(manifest_path.resolve()), "sha256": stable_file_hash(manifest_path), "stage_status": manifest.get("stage_status")},
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
            "roster": {"path": str(roster_path.resolve()), "sha256": stable_file_hash(roster_path)},
        },
        "condition_summary": condition_summary(results),
        "attempted_jobs": attempted_jobs,
        "results": results,
        "blocking_findings": [] if stage_status == "m1m2_stress_128_confirmation_completed" else [stage_status],
        "error": error,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stress_128_paid_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def load_resume_checkpoint(
    output_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    paid_result_path = output_dir / DEFAULT_PAID_RESULT_NAME
    if not paid_result_path.exists():
        return [], [], 0
    checkpoint = load_json(paid_result_path)
    if checkpoint.get("source_artifacts", {}).get("manifest", {}).get("sha256") != stable_file_hash(manifest_path):
        raise ValueError("resume_manifest_hash_mismatch")
    results = list(checkpoint.get("results") or [])
    attempted_jobs = list(checkpoint.get("attempted_jobs") or [])
    attempts = checkpoint.get("actual_llm_http_attempts")
    if not isinstance(attempts, int):
        raise ValueError("resume_attempt_accounting_unknown")
    result_keys = {job_key(row) for row in results}
    dangling = [job for job in attempted_jobs if job_key(job) not in result_keys]
    if len(dangling) > 1:
        raise ValueError("resume_multiple_dangling_attempts")
    if dangling:
        error = str(checkpoint.get("error") or checkpoint.get("stage_status") or "transport_failure")
        results.append(timeout_result_for_attempted_job(dangling[0], error))
    if len(results) > int(manifest["llm_http_attempt_cap"]):
        raise ValueError("resume_results_exceed_cap")
    if attempts > int(manifest["llm_http_attempt_cap"]):
        raise ValueError("resume_attempts_exceed_cap")
    return results, attempted_jobs, attempts


def build_dry_run(manifest_path: Path, measured_records_path: Path, roster_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    records = records_by_id(load_json(measured_records_path))
    stress_jobs = stress_jobs_by_id(load_json(roster_path))
    findings = review_manifest(manifest)
    jobs, job_findings = build_jobs(manifest, records, stress_jobs)
    findings.extend(job_findings)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_stress_128_dry_run",
        "stage_status": "m1m2_stress_128_dry_run_passed" if not findings else "m1m2_stress_128_dry_run_failed_closed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "manifest": {"path": str(manifest_path.resolve()), "sha256": stable_file_hash(manifest_path), "stage_status": manifest.get("stage_status")},
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
            "roster": {"path": str(roster_path.resolve()), "sha256": stable_file_hash(roster_path)},
        },
        "blocking_findings": findings,
        "job_count": len(jobs),
        "max_prompt_estimated_tokens": max((job["prompt_estimated_tokens"] for job in jobs), default=0),
        "jobs": [{key: value for key, value in job.items() if key != "prompt"} for job in jobs],
        "next_required_action": "execute_after_explicit_896_call_approval" if not findings else "repair_m1m2_stress_128_inputs",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stress_128_dry_run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def run_paid(
    manifest_path: Path,
    measured_records_path: Path,
    roster_path: Path,
    output_dir: Path,
    model_display_name: str | None,
    approved_llm_cap: int | None,
    restart_after_unaccounted_interruption: bool = False,
) -> dict[str, Any]:
    guard_existing_paid_result(output_dir, restart_after_unaccounted_interruption)
    manifest = load_json(manifest_path)
    findings = review_manifest(manifest, approved_llm_cap=approved_llm_cap, for_paid=True)
    if findings:
        raise ValueError(f"M1/M2 stress 128 manifest failed review: {findings}")
    model_provenance = resolve_model_provenance(model_display_name)
    resolved_model_display_name = model_provenance["model_display_name"]
    key = _get_saved_key(resolved_model_display_name)
    adapter = _create_llm_adapter(resolved_model_display_name, key) if key else None
    if adapter is None:
        raise ValueError(f"could not create adapter for {resolved_model_display_name}")
    records = records_by_id(load_json(measured_records_path))
    stress_jobs = stress_jobs_by_id(load_json(roster_path))
    jobs, job_findings = build_jobs(manifest, records, stress_jobs)
    if job_findings:
        raise ValueError(f"M1/M2 stress 128 job construction failed review: {job_findings}")
    counter = AttemptCounter(int(manifest["llm_http_attempt_cap"]))
    results, attempted_jobs, prior_attempts = load_resume_checkpoint(output_dir, manifest_path, manifest)
    counter.llm_attempts = prior_attempts
    completed_job_keys = {job_key(row) for row in results}
    max_output_tokens = int(manifest["max_output_tokens"])
    original_retries = adapter_base._MAX_API_RETRIES
    adapter_base._MAX_API_RETRIES = 1
    try:
        for job in jobs:
            if job_key(job) in completed_job_keys:
                continue
            counter.consume_llm(str(job["stress_id"]), str(job["condition"]), "selective_answer")
            attempted_jobs.append({key: value for key, value in job.items() if key != "prompt"})
            write_paid_checkpoint(
                output_dir,
                manifest_path,
                measured_records_path,
                roster_path,
                manifest,
                model_provenance,
                results,
                attempted_jobs,
                counter,
                "m1m2_stress_128_paid_run_in_progress",
            )
            raw_text = controlled_text_only_call(adapter, job["prompt"], max_output_tokens)
            record = parse_selective_answer(raw_text, str(job["answer_type"]))
            score = score_selective_answer(record, job["reference"], str(job["answer_type"]), str(job["expected_answerability"]))
            results.append({**{key: value for key, value in job.items() if key != "prompt"}, **answer_record_to_dict(record), **score})
            completed_job_keys.add(job_key(job))
            write_paid_checkpoint(
                output_dir,
                manifest_path,
                measured_records_path,
                roster_path,
                manifest,
                model_provenance,
                results,
                attempted_jobs,
                counter,
                "m1m2_stress_128_paid_run_in_progress",
            )
    except KeyboardInterrupt:
        return write_paid_checkpoint(
            output_dir,
            manifest_path,
            measured_records_path,
            roster_path,
            manifest,
            model_provenance,
            results,
            attempted_jobs,
            counter,
            "m1m2_stress_128_paid_run_interrupted",
            "KeyboardInterrupt",
        )
    except Exception as exc:
        return write_paid_checkpoint(
            output_dir,
            manifest_path,
            measured_records_path,
            roster_path,
            manifest,
            model_provenance,
            results,
            attempted_jobs,
            counter,
            "m1m2_stress_128_paid_run_failed",
            str(exc),
        )
    finally:
        adapter_base._MAX_API_RETRIES = original_retries
    return write_paid_checkpoint(
        output_dir,
        manifest_path,
        measured_records_path,
        roster_path,
        manifest,
        model_provenance,
        results,
        attempted_jobs,
        counter,
        "m1m2_stress_128_confirmation_completed",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--measured-records", default=str(DEFAULT_RECORDS))
    parser.add_argument("--roster", default=str(DEFAULT_ROSTER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--approved-llm-cap", type=int, default=None)
    parser.add_argument("--model-display-name", default=None)
    parser.add_argument(
        "--restart-after-unaccounted-interruption",
        action="store_true",
        help="Allow a paid restart after an existing result records unknown attempt accounting. Use only after explicit cap reauthorization.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute_paid:
        result = run_paid(
            Path(args.manifest),
            Path(args.measured_records),
            Path(args.roster),
            Path(args.output_dir),
            args.model_display_name,
            args.approved_llm_cap,
            args.restart_after_unaccounted_interruption,
        )
        print(f"wrote M1/M2 stress 128 paid result: {Path(args.output_dir) / 'stress_128_paid_result.json'}")
    else:
        result = build_dry_run(Path(args.manifest), Path(args.measured_records), Path(args.roster), Path(args.output_dir))
        print(f"wrote M1/M2 stress 128 dry run: {Path(args.output_dir) / 'stress_128_dry_run.json'}")
    print(f"stage_status={result['stage_status']}")
    print(f"job_count={result.get('job_count', len(result.get('results', [])))}")
    print(f"paid_calls_executed={result['paid_calls_executed']}")
    return 0 if not result.get("blocking_findings") else 1


if __name__ == "__main__":
    raise SystemExit(main())
