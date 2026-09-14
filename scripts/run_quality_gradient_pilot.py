#!/usr/bin/env python
"""Run the bounded Phase C M1/M2 recovery pilot.

Safe default: write a dry-run artifact only. Paid execution requires
``--run-paid`` and is capped by the frozen preregistration plus the global
budget ledger before the first network call.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluator.adapters.base as adapter_base  # noqa: E402
from app.helpers import _create_llm_adapter, _get_saved_key, load_image_and_gt  # noqa: E402
from app.reasoning_pipeline import (  # noqa: E402
    CONDITION_MAP,
    build_chain_text,
    compute_visual_facts_from_coords,
    load_layer_config,
    run_judge,
    run_single_condition,
)
from evaluator.metrics.reasoning import evaluate_comprehensive  # noqa: E402
from recognizer.core.clean_grid_fallback import extract_clean_cartesian_grid  # noqa: E402
from scripts.analyze_evidence_selection_identifiability import load_ground_truth, normalize_points  # noqa: E402
from scripts.m1m2_recovery_quality_gradient import QUALITY_LEVELS, degrade_coords  # noqa: E402
from scripts.run_evidence_selection_variance_study import objective9  # noqa: E402
from scripts.prepare_evidence_selection_preregistration import quality_counts  # noqa: E402
from scripts.prepare_public_benchmark_pilot import canonical_json_bytes, sha256_hex_bytes, stable_file_hash  # noqa: E402

RECOVERY_ROOT = ROOT / "outputs" / "m1m2_recovery"
PREREG = RECOVERY_ROOT / "preregistration.json"
PHASE_C_DRY_RUN = RECOVERY_ROOT / "phase_c_dry_run.json"
BUDGET_LEDGER = ROOT / "outputs" / "autonomous_run_log" / "budget_ledger.json"
TRACK_B_MANIFEST = ROOT / "outputs" / "paper_level_evidence_package" / "geosym_bench_controlled" / "post_remediation_rerun_manifest.json"
DEFAULT_OUTPUT = RECOVERY_ROOT / "phase_c_pilot_paid_result.json"
DEFAULT_RUN_DRY = RECOVERY_ROOT / "phase_c_pilot_runner_dry_run.json"
DEFAULT_MANIFEST = RECOVERY_ROOT / "phase_c_pilot_manifest.json"

CONDITIONS = ["E5", "E5_raw"]
CONDITION_CALL_CAPS = {"E5": 5, "E5_raw": 4}
TRANSPORT_RETRY_CAP_PER_SEMANTIC_CALL = 1
OPENAI_TIMEOUT_SECONDS = 60
QWEN_TIMEOUT_SECONDS = 60
OPENAI_MAX_COMPLETION_TOKENS = 1024
QWEN_MAX_TOKENS = 1024
SDK_MAX_RETRIES = 0


class AttemptCounter:
    def __init__(self, cap: int) -> None:
        self.http_attempts = 0
        self.cap = cap
        self.cap_exhausted = False


class LLMHttpAttemptCapExhausted(RuntimeError):
    pass


@contextmanager
def count_llm_http_attempts(counter: AttemptCounter) -> Iterator[None]:
    """Count and cap actual LLM transport attempts through BaseLLMAdapter."""
    original_call_with_retry = adapter_base.BaseLLMAdapter._call_with_retry

    def counted_call_with_retry(self, image_b64: str, prompt: str) -> str:
        last_exc = None
        for attempt in range(1, adapter_base._MAX_API_RETRIES + 1):
            if counter.http_attempts >= counter.cap:
                counter.cap_exhausted = True
                raise LLMHttpAttemptCapExhausted("llm_http_attempt_cap_exhausted")
            counter.http_attempts += 1
            try:
                return self._call_api(image_b64, prompt)
            except Exception as exc:
                last_exc = exc
                if attempt >= adapter_base._MAX_API_RETRIES or not adapter_base.is_transient_error(exc):
                    raise
                time.sleep(0.1)
        raise last_exc

    adapter_base.BaseLLMAdapter._call_with_retry = counted_call_with_retry
    try:
        yield
    finally:
        adapter_base.BaseLLMAdapter._call_with_retry = original_call_with_retry


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def configure_paid_call_environment() -> None:
    os.environ["GEOSYM_OPENAI_TIMEOUT_SECONDS"] = str(OPENAI_TIMEOUT_SECONDS)
    os.environ["GEOSYM_QWEN_TIMEOUT_SECONDS"] = str(QWEN_TIMEOUT_SECONDS)
    os.environ["GEOSYM_OPENAI_MAX_COMPLETION_TOKENS"] = str(OPENAI_MAX_COMPLETION_TOKENS)
    os.environ["GEOSYM_QWEN_MAX_TOKENS"] = str(QWEN_MAX_TOKENS)
    os.environ["GEOSYM_OPENAI_MAX_RETRIES"] = str(SDK_MAX_RETRIES)
    os.environ["GEOSYM_QWEN_MAX_RETRIES"] = str(SDK_MAX_RETRIES)
    os.environ["GEOSYM_GLM_TIMEOUT_SECONDS"] = str(OPENAI_TIMEOUT_SECONDS)
    os.environ["GEOSYM_GLM_MAX_TOKENS"] = str(OPENAI_MAX_COMPLETION_TOKENS)
    os.environ["GEOSYM_GLM_MAX_RETRIES"] = str(SDK_MAX_RETRIES)


def plan_hash(prereg: dict[str, Any]) -> str:
    material = {k: v for k, v in prereg.items() if k != "created_at_utc"}
    return sha256_hex_bytes(canonical_json_bytes(material))


def pilot_roster(dry_run: dict[str, Any]) -> list[dict[str, str]]:
    seen: set[str] = set()
    roster: list[dict[str, str]] = []
    for row in dry_run.get("rows") or []:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            continue
        seen.add(sample_id)
        family = str(row["family"])
        roster.append({
            "sample_id": sample_id,
            "family": family,
            "image_name": f"{family}-clean_render/{sample_id}",
        })
    return roster


def load_gt_data(image_name: str) -> dict[str, Any]:
    _, gt = load_image_and_gt(image_name)
    if gt is None:
        raise FileNotFoundError(f"ground truth not found for {image_name}")
    gt_data = (gt.get("question") or {}).get("ground_truth") or {}
    if not gt_data:
        raise ValueError(f"ground truth question missing for {image_name}")
    return gt_data


def model_manifest(reasoning_model: str | None = None, judge_model: str | None = None) -> dict[str, Any]:
    manifest = load_json(TRACK_B_MANIFEST)
    reasoning = dict(manifest["reasoning_model_provenance"])
    judge = dict(manifest["judge_model_provenance"])
    if reasoning_model:
        reasoning["display_name"] = reasoning_model
        reasoning["override_source"] = "cli"
    if judge_model:
        judge["display_name"] = judge_model
        judge["override_source"] = "cli"
    return {
        "reasoning_model_provenance": reasoning,
        "judge_model_provenance": judge,
        "temperature": manifest.get("temperature", 0.0),
        "prompt_hashes": manifest.get("prompt_hashes", {}),
    }


def load_experiment_manifest() -> dict[str, Any]:
    """Load the currently frozen manifest, allowing explicit repair rounds."""
    manifest_path = Path(os.environ.get("GEOSYM_PHASE_C_MANIFEST", str(DEFAULT_MANIFEST)))
    if not manifest_path.exists():
        return {"path": str(PREREG), "sha256": stable_file_hash(PREREG), "cap": {}, "round_label": "pilot_round_1"}
    manifest = load_json(manifest_path)
    return {"path": str(manifest_path.resolve()), "sha256": stable_file_hash(manifest_path), "cap": manifest.get("cap") or {}, "round_label": manifest.get("round_label", "explicit_round")}


def current_budget_remaining(ledger: dict[str, Any]) -> dict[str, int]:
    return {
        "llm": int(ledger.get("llm_cap") or 0) - int(ledger.get("cumulative_llm") or 0),
        "ocr": int(ledger.get("ocr_cap") or 0) - int(ledger.get("cumulative_ocr") or 0),
    }


def build_dry_run(
    prereg: dict[str, Any],
    dry_run: dict[str, Any],
    ledger: dict[str, Any],
    checkpoint_path: Path = DEFAULT_OUTPUT,
    reasoning_model: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    model_info = model_manifest(reasoning_model, judge_model)
    experiment_manifest = load_experiment_manifest()
    roster = pilot_roster(dry_run)
    expected_cap = len(roster) * len(QUALITY_LEVELS) * sum(CONDITION_CALL_CAPS.values())
    prereg_cap = int((prereg.get("cap") or {}).get("pilot_llm_http_attempt_cap") or 0)
    manifest_cap = int((experiment_manifest.get("cap") or {}).get("llm_http_attempt_cap") or 0)
    round_cap = manifest_cap or prereg_cap
    remaining = current_budget_remaining(ledger)
    findings: list[str] = []
    if dry_run.get("stage_status") != "phase_c_dry_run_passed":
        findings.append(f"phase_c_dry_run_not_passed:{dry_run.get('stage_status')}")
    if prereg.get("conditions") != CONDITIONS:
        findings.append("conditions_not_frozen_to_e5_e5raw")
    if prereg_cap != expected_cap:
        findings.append(f"pilot_llm_cap_mismatch:prereg={prereg_cap}:expected={expected_cap}")
    if round_cap <= 0:
        findings.append("round_llm_cap_missing")
    if round_cap > expected_cap:
        findings.append(f"round_llm_cap_exceeds_preregistered_pilot_cap:round={round_cap}:pilot={expected_cap}")
    checkpoint_attempts = 0
    if checkpoint_path.exists():
        checkpoint_attempts = int((load_json(checkpoint_path) or {}).get("actual_llm_http_attempts") or 0)
    remaining_pilot_cap = round_cap - checkpoint_attempts
    if remaining_pilot_cap <= 0:
        findings.append(f"pilot_llm_cap_exhausted:used={checkpoint_attempts}:cap={expected_cap}")
    if remaining["llm"] < max(0, remaining_pilot_cap):
        findings.append(f"global_llm_cap_insufficient:remaining={remaining['llm']}:required={remaining_pilot_cap}")
    ocr_cap = (prereg.get("cap") or {}).get("pilot_ocr_http_attempt_cap")
    if ocr_cap is None or int(ocr_cap) != 0:
        findings.append("pilot_ocr_cap_must_be_zero")
    if model_info["reasoning_model_provenance"].get("display_name") == model_info["judge_model_provenance"].get("display_name"):
        findings.append("judge_model_matches_reasoning_model")
    return {
        "schema_version": "m1m2_recovery_phase_c_pilot_runner_dry_run_v1",
        "created_at_utc": utc_now(),
        "stage": "m1m2_recovery_phase_c_pilot_runner_dry_run",
        "stage_status": "ready_for_paid_pilot" if not findings else "blocking_findings_present",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "preregistration_path": str(PREREG.resolve()),
        "preregistration_sha256": stable_file_hash(PREREG),
        "phase_c_dry_run_path": str(PHASE_C_DRY_RUN.resolve()),
        "phase_c_dry_run_sha256": stable_file_hash(PHASE_C_DRY_RUN),
        "budget_ledger_path": str(BUDGET_LEDGER.resolve()),
        "experiment_manifest": experiment_manifest,
        "budget_remaining_before_pilot": remaining,
        "plan_sha256": plan_hash(prereg),
        "sample_count": len(roster),
        "sample_quality_jobs": len(roster) * len(QUALITY_LEVELS),
        "conditions": CONDITIONS,
        "quality_levels": QUALITY_LEVELS,
        "condition_call_caps": CONDITION_CALL_CAPS,
        "pilot_llm_http_attempt_cap": round_cap,
        "original_preregistered_pilot_llm_http_attempt_cap": expected_cap,
        "pilot_ocr_http_attempt_cap": 0,
        "transport_retry_cap_per_semantic_call": TRANSPORT_RETRY_CAP_PER_SEMANTIC_CALL,
        "decoding_overrides": {
            "temperature": 0.0,
            "openai_timeout_seconds": OPENAI_TIMEOUT_SECONDS,
            "qwen_timeout_seconds": QWEN_TIMEOUT_SECONDS,
            "openai_max_completion_tokens": OPENAI_MAX_COMPLETION_TOKENS,
            "qwen_max_tokens": QWEN_MAX_TOKENS,
            "openai_sdk_max_retries": SDK_MAX_RETRIES,
            "qwen_sdk_max_retries": SDK_MAX_RETRIES,
        },
        "model_provenance": model_info,
        "blocking_findings": findings,
    }


def apply_checkpoint_guard(dry_run: dict[str, Any], output_path: Path) -> dict[str, Any]:
    """Mark dry-run blocked when an existing paid checkpoint is unsafe to resume."""
    guarded = dict(dry_run)
    findings = list(guarded.get("blocking_findings") or [])
    checkpoint_summary = None
    if output_path.exists():
        checkpoint = load_json(output_path)
        checkpoint_summary = {
            "path": str(output_path.resolve()),
            "sha256": stable_file_hash(output_path),
            "stage_status": checkpoint.get("stage_status"),
            "actual_llm_http_attempts": checkpoint.get("actual_llm_http_attempts"),
            "actual_ocr_http_attempts": checkpoint.get("actual_ocr_http_attempts"),
            "completed_sample_quality_jobs": checkpoint.get("completed_sample_quality_jobs"),
            "in_progress_job": checkpoint.get("in_progress_job"),
        }
        if checkpoint.get("in_progress_job") is not None and checkpoint.get("stage_status") == "running_paid_pilot":
            findings.append("existing_paid_checkpoint_has_in_progress_job_reauthorization_required")
    guarded["checkpoint_summary"] = checkpoint_summary
    guarded["blocking_findings"] = findings
    guarded["stage_status"] = "ready_for_paid_pilot" if not findings else "blocking_findings_present"
    return guarded


def row_key(sample_id: str, quality_level: str) -> str:
    return f"{sample_id}::{quality_level}"


def condition_result_row(
    sample: dict[str, str],
    quality_level: str,
    condition: str,
    degraded_coords: list[tuple[int, int]],
    m0_quality: dict[str, Any],
    pipeline: dict[str, Any] | None,
    judge: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    *,
    judge_reused: bool,
    attempts_before: int,
    attempts_after: int,
    error: str | None = None,
) -> dict[str, Any]:
    chain_text = build_chain_text(pipeline or {}) if pipeline else ""
    dims = (evaluation or {}).get("dimension_scores") or {}
    status = "success" if error is None else ("cap_exhausted" if "llm_http_attempt_cap_exhausted" in error else "runner_error")
    return {
        "sample_id": sample["sample_id"],
        "family": sample["family"],
        "image_name": sample["image_name"],
        "quality_level": quality_level,
        "condition": condition,
        "status": status,
        "degraded_coords": degraded_coords,
        "degraded_point_count": len(degraded_coords),
        "m0_quality": m0_quality,
        "total_score": (evaluation or {}).get("total_score"),
        "dimension_scores": dims or None,
        "objective9_mean_score": objective9({"dimension_scores": dims}) if dims else None,
        "chain_text_sha256": sha256_hex_bytes(canonical_json_bytes(chain_text)) if chain_text else None,
        "raw_judge_response_sha256": sha256_hex_bytes(canonical_json_bytes((judge or {}).get("raw_judge_response") or "")) if judge else None,
        "judge_result": judge,
        "judge_reused_from_identical_chain_text": judge_reused,
        "llm_http_attempts": attempts_after - attempts_before,
        "error": error,
    }


def replace_job_rows(rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace all prior rows for one sample-quality job with its latest attempt."""
    if not new_rows:
        return rows
    sample_id = str(new_rows[0]["sample_id"])
    quality_level = str(new_rows[0]["quality_level"])
    retained = [
        row for row in rows
        if not (str(row.get("sample_id")) == sample_id and str(row.get("quality_level")) == quality_level)
    ]
    return retained + new_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for quality in [item["level"] for item in QUALITY_LEVELS]:
        q_rows = [row for row in rows if row.get("quality_level") == quality]
        by_condition: dict[str, Any] = {}
        for condition in CONDITIONS:
            items = [row for row in q_rows if row.get("condition") == condition]
            scores = [float(row["objective9_mean_score"]) for row in items if row.get("objective9_mean_score") is not None]
            by_condition[condition] = {
                "rows": len(items),
                "success": sum(1 for row in items if row.get("status") == "success"),
                "mean_objective9": round(sum(scores) / len(scores), 6) if scores else None,
            }
        paired = []
        by_sample: dict[str, dict[str, Any]] = {}
        for row in q_rows:
            by_sample.setdefault(str(row["sample_id"]), {})[str(row["condition"])] = row
        for pair in by_sample.values():
            e5 = pair.get("E5")
            raw = pair.get("E5_raw")
            if e5 and raw and e5.get("objective9_mean_score") is not None and raw.get("objective9_mean_score") is not None:
                paired.append(float(e5["objective9_mean_score"]) - float(raw["objective9_mean_score"]))
        summary[quality] = {
            "by_condition": by_condition,
            "paired_delta_e5_minus_e5raw_objective9_mean": round(sum(paired) / len(paired), 6) if paired else None,
            "paired_delta_n": len(paired),
        }
    return summary


def write_result(
    output_path: Path,
    prereg: dict[str, Any],
    runner_dry_run: dict[str, Any],
    rows: list[dict[str, Any]],
    completed_jobs: list[dict[str, str]],
    status: str,
    started_at_utc: str,
    attempts: int,
    initial_attempts: int,
    in_progress_job: dict[str, str] | None = None,
) -> dict[str, Any]:
    chain_hashes = [row.get("chain_text_sha256") for row in rows]
    nonempty_chain_hashes = [value for value in chain_hashes if value]
    distinct_chain_hashes = len(set(nonempty_chain_hashes))
    result = {
        "schema_version": "m1m2_recovery_phase_c_pilot_paid_result_v1",
        "created_at_utc": utc_now(),
        "stage": "m1m2_recovery_phase_c_pilot_paid_execution",
        "stage_status": status,
        "started_at_utc": started_at_utc,
        "preregistration_path": str(PREREG.resolve()),
        "preregistration_sha256": stable_file_hash(PREREG),
        "plan_sha256": plan_hash(prereg),
        "experiment_manifest": load_experiment_manifest(),
        "paid_calls_executed": attempts - initial_attempts,
        "paid_call_accounting_mode": "actual_llm_http_attempts_with_retry_cap_1",
        "actual_llm_http_attempts": attempts,
        "new_llm_http_attempts_this_invocation": attempts - initial_attempts,
        "actual_ocr_http_attempts": 0,
        "pilot_llm_http_attempt_cap": runner_dry_run["pilot_llm_http_attempt_cap"],
        "pilot_ocr_http_attempt_cap": 0,
        "cap_exhausted": status == "stopped_at_paid_cap",
        "cumulative_attempts_against_original_prereg_cap": {
            "spent": attempts,
            "original_cap": runner_dry_run["original_preregistered_pilot_llm_http_attempt_cap"],
            "ratio": round(
                attempts / float(runner_dry_run["original_preregistered_pilot_llm_http_attempt_cap"]),
                6,
            ) if runner_dry_run["original_preregistered_pilot_llm_http_attempt_cap"] else None,
            "justification_chain": ["round2_manifest", "round3_manifest", "round4_manifest"],
            "note": "Row-level llm_http_attempts are per-row deltas and are not valid for cross-round cumulative accounting.",
        },
        "completed_sample_quality_jobs": len(completed_jobs),
        "in_progress_job": in_progress_job,
        "conditions": CONDITIONS,
        "quality_levels": QUALITY_LEVELS,
        "governance": {
            "within_pilot_llm_http_attempt_cap": attempts <= int(runner_dry_run["pilot_llm_http_attempt_cap"]),
            "within_pilot_ocr_http_attempt_cap": True,
            "sample_replacement": False,
            "conditions_frozen": prereg.get("conditions") == CONDITIONS,
            "judge_model_differs_from_reasoning_model": (
                runner_dry_run["model_provenance"]["reasoning_model_provenance"].get("display_name")
                != runner_dry_run["model_provenance"]["judge_model_provenance"].get("display_name")
            ),
            "chain_hash_distinct_gate": {
                "distinct": distinct_chain_hashes,
                "n_rows": len(chain_hashes),
                "n_rows_with_chain": len(nonempty_chain_hashes),
                "passes_nonempty_every_row": len(nonempty_chain_hashes) == len(chain_hashes) if chain_hashes else False,
                "passes_distinct_gt_0_9n": distinct_chain_hashes > 0.9 * len(chain_hashes) if chain_hashes else False,
            },
        },
        "model_provenance": runner_dry_run["model_provenance"],
        "summary": summarize(rows),
        "completed_jobs": completed_jobs,
        "results": rows,
    }
    write_json(output_path, result)
    return result


def update_budget_ledger(output_path: Path, result: dict[str, Any]) -> None:
    ledger = load_json(BUDGET_LEDGER)
    entry = {
        "timestamp": utc_now(),
        "phase": "phase_c",
        "step": "bounded_m1m2_recovery_pilot",
        "llm_attempts": int(result.get("new_llm_http_attempts_this_invocation") or 0),
        "ocr_attempts": 0,
        "artifact_path": str(output_path.resolve()),
        "artifact_sha256": stable_file_hash(output_path),
    }
    ledger.setdefault("entries", []).append(entry)
    ledger["cumulative_llm"] = int(ledger.get("cumulative_llm") or 0) + entry["llm_attempts"]
    ledger["cumulative_ocr"] = int(ledger.get("cumulative_ocr") or 0)
    ledger.setdefault("interpretation_limits", []).append("Phase C pilot only; full Phase C still requires an explicit new cap.")
    write_json(BUDGET_LEDGER, ledger)


def run_paid(
    output_path: Path,
    max_new_samples: int | None,
    max_new_jobs: int | None,
    reasoning_model_override: str | None = None,
    judge_model_override: str | None = None,
    restart_after_unaccounted_interruption: bool = False,
    resume_source: Path | None = None,
) -> dict[str, Any]:
    prereg = load_json(PREREG)
    phase_c_dry = load_json(PHASE_C_DRY_RUN)
    ledger = load_json(BUDGET_LEDGER)
    configure_paid_call_environment()
    runner_dry = build_dry_run(prereg, phase_c_dry, ledger, output_path, reasoning_model_override, judge_model_override)
    if runner_dry["stage_status"] != "ready_for_paid_pilot":
        raise ValueError(f"phase_c_pilot_dry_run_blocking_findings:{runner_dry['blocking_findings']}")

    model_info = runner_dry["model_provenance"]
    reasoning_model = model_info["reasoning_model_provenance"]["display_name"]
    judge_model = model_info["judge_model_provenance"]["display_name"]
    reasoning_key = _get_saved_key(reasoning_model)
    judge_key = _get_saved_key(judge_model)
    if not reasoning_key:
        raise ValueError(f"saved API key not found for reasoning model {reasoning_model}")
    if not judge_key:
        raise ValueError(f"saved API key not found for judge model {judge_model}")
    reasoning_adapter = _create_llm_adapter(reasoning_model, reasoning_key)
    judge_adapter = _create_llm_adapter(judge_model, judge_key)
    if reasoning_adapter is None or judge_adapter is None:
        raise ValueError("could not create reasoning or judge adapter")

    rows: list[dict[str, Any]] = []
    completed_jobs: list[dict[str, str]] = []
    initial_attempts = 0
    counter_start_attempts = 0
    if output_path.exists():
        checkpoint = load_json(output_path)
        if checkpoint.get("plan_sha256") != plan_hash(prereg):
            raise ValueError("checkpoint_plan_hash_mismatch")
        if checkpoint.get("in_progress_job") is not None and not restart_after_unaccounted_interruption:
            raise ValueError("checkpoint_has_in_progress_job_reauthorization_required")
        rows = list(checkpoint.get("results") or [])
        completed_jobs = list(checkpoint.get("completed_jobs") or [])
        initial_attempts = int(checkpoint.get("actual_llm_http_attempts") or 0)
        counter_start_attempts = initial_attempts
        if checkpoint.get("in_progress_job") is not None and restart_after_unaccounted_interruption:
            counter_start_attempts += 1
    elif resume_source is not None and resume_source.exists():
        checkpoint = load_json(resume_source)
        if checkpoint.get("plan_sha256") != plan_hash(prereg):
            raise ValueError("resume_source_plan_hash_mismatch")
        rows = list(checkpoint.get("results") or [])
        completed_jobs = list(checkpoint.get("completed_jobs") or [])

    counter = AttemptCounter(int(runner_dry["pilot_llm_http_attempt_cap"]))
    counter.http_attempts = counter_start_attempts
    job_status: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        job_status.setdefault(row_key(str(row["sample_id"]), str(row["quality_level"])), {})[str(row["condition"])] = row
    completed_keys = {
        key for key, condition_rows in job_status.items()
        if all(condition_rows.get(condition, {}).get("status") == "success" for condition in CONDITIONS)
    }
    completed_jobs = [
        job for job in completed_jobs
        if row_key(job["sample_id"], job["quality_level"]) in completed_keys
    ]
    judge_cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("chain_text_sha256") and row.get("judge_result"):
            judge_cache[str(row["chain_text_sha256"])] = dict(row["judge_result"])

    roster = pilot_roster(phase_c_dry)
    started_at_utc = utc_now()
    layer_config = load_layer_config()
    original_retries = adapter_base._MAX_API_RETRIES
    adapter_base._MAX_API_RETRIES = TRANSPORT_RETRY_CAP_PER_SEMANTIC_CALL
    samples_started = 0
    jobs_started = 0
    in_progress: dict[str, str] | None = None
    try:
        with count_llm_http_attempts(counter):
            for sample in roster:
                if hasattr(reasoning_adapter, "reset_client"):
                    reasoning_adapter.reset_client()
                if hasattr(judge_adapter, "reset_client"):
                    judge_adapter.reset_client()
                if max_new_samples is not None and samples_started >= max_new_samples:
                    break
                sample_has_new_job = False
                image_path = ROOT / "dataset" / "images" / f"{sample['image_name']}.png"
                image = cv2.imread(str(image_path))
                if image is None:
                    raise FileNotFoundError(f"image not found: {image_path}")
                extraction = extract_clean_cartesian_grid(image)
                gt_data = load_gt_data(sample["image_name"])
                gt_points = normalize_points(load_ground_truth(sample["image_name"]).get("coordinates"))
                for level_spec in QUALITY_LEVELS:
                    quality_level = str(level_spec["level"])
                    job_key = row_key(sample["sample_id"], quality_level)
                    if job_key in completed_keys:
                        continue
                    if max_new_jobs is not None and jobs_started >= max_new_jobs:
                        break
                    sample_has_new_job = True
                    degraded = degrade_coords(extraction.coords, sample_id=sample["sample_id"], level=quality_level)
                    m0_quality = quality_counts(degraded, gt_points)
                    in_progress = {
                        "sample_id": sample["sample_id"],
                        "family": sample["family"],
                        "quality_level": quality_level,
                    }
                    write_result(
                        output_path, prereg, runner_dry, rows, completed_jobs,
                        "running_paid_pilot", started_at_utc, counter.http_attempts,
                        initial_attempts, in_progress,
                    )
                    existing_condition_rows = job_status.get(job_key, {})
                    job_rows: list[dict[str, Any]] = [
                        existing_condition_rows[condition]
                        for condition in CONDITIONS
                        if existing_condition_rows.get(condition, {}).get("status") == "success"
                    ]
                    for condition in CONDITIONS:
                        if existing_condition_rows.get(condition, {}).get("status") == "success":
                            continue
                        if counter.http_attempts + CONDITION_CALL_CAPS[condition] > counter.cap:
                            return write_result(
                                output_path, prereg, runner_dry, rows, completed_jobs,
                                "stopped_at_paid_cap", started_at_utc, counter.http_attempts,
                                initial_attempts, in_progress,
                            )
                        before = counter.http_attempts
                        try:
                            visual_facts = compute_visual_facts_from_coords(degraded) if condition == "E5" else None
                            pipeline = run_single_condition(
                                reasoning_adapter,
                                image_path,
                                gt_data,
                                CONDITION_MAP[condition],
                                geosym_coords=degraded,
                                visual_facts=visual_facts,
                            )
                            chain_text = build_chain_text(pipeline)
                            chain_hash = sha256_hex_bytes(canonical_json_bytes(chain_text))
                            if chain_hash in judge_cache and judge_cache[chain_hash]:
                                judge = judge_cache[chain_hash]
                                judge_reused = True
                            else:
                                judge = run_judge(judge_adapter, image_path, gt_data, pipeline, max_retries=1)
                                judge_cache[chain_hash] = judge
                                judge_reused = False
                            evaluation = evaluate_comprehensive(pipeline["final_parsed"], gt_data, layer_config, judge)
                            row = condition_result_row(
                                sample, quality_level, condition, degraded, m0_quality,
                                pipeline, judge, evaluation, judge_reused=judge_reused,
                                attempts_before=before, attempts_after=counter.http_attempts,
                            )
                        except Exception as exc:
                            row = condition_result_row(
                                sample, quality_level, condition, degraded, m0_quality,
                                None, None, None, judge_reused=False,
                                attempts_before=before, attempts_after=counter.http_attempts,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        job_rows.append(row)
                        job_status.setdefault(job_key, {})[condition] = row
                        rows = replace_job_rows(rows, job_rows)
                        write_result(
                            output_path, prereg, runner_dry, rows, completed_jobs,
                            "running_paid_pilot", started_at_utc, counter.http_attempts,
                            initial_attempts, in_progress,
                        )
                        if row.get("status") == "cap_exhausted":
                            return write_result(
                                output_path, prereg, runner_dry, rows, completed_jobs,
                                "stopped_at_paid_cap", started_at_utc, counter.http_attempts,
                                initial_attempts, in_progress,
                            )
                    job_succeeded = len(job_rows) == len(CONDITIONS) and all(
                        existing.get("status") == "success" for existing in job_rows
                    )
                    if job_succeeded:
                        completed_jobs = [
                            existing for existing in completed_jobs
                            if row_key(existing["sample_id"], existing["quality_level"]) != job_key
                        ]
                        completed_jobs.append(in_progress)
                        completed_keys.add(job_key)
                    jobs_started += 1
                    write_result(
                        output_path, prereg, runner_dry, rows, completed_jobs,
                        "partial_paid_pilot", started_at_utc, counter.http_attempts,
                        initial_attempts,
                    )
                    in_progress = None
                if sample_has_new_job:
                    samples_started += 1
                if max_new_jobs is not None and jobs_started >= max_new_jobs:
                    break
    except KeyboardInterrupt:
        return write_result(
            output_path, prereg, runner_dry, rows, completed_jobs,
            "interrupted_paid_pilot", started_at_utc, counter.http_attempts,
            initial_attempts, in_progress,
        )
    finally:
        adapter_base._MAX_API_RETRIES = original_retries

    expected_jobs = len(roster) * len(QUALITY_LEVELS)
    status = "phase_c_pilot_completed" if len(completed_jobs) == expected_jobs else "partial_paid_pilot"
    result = write_result(
        output_path, prereg, runner_dry, rows, completed_jobs, status,
        started_at_utc, counter.http_attempts, initial_attempts,
    )
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-output", default=str(DEFAULT_RUN_DRY))
    parser.add_argument("--run-paid", action="store_true")
    parser.add_argument("--reasoning-model", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--max-new-samples", type=int)
    parser.add_argument("--max-new-jobs", type=int, help="Limit new sample-quality jobs for smoke runs.")
    parser.add_argument(
        "--restart-after-unaccounted-interruption",
        action="store_true",
        help="Resume after an in-progress checkpoint by conservatively charging one unknown LLM attempt.",
    )
    parser.add_argument(
        "--resume-source",
        type=Path,
        help="Seed a new round checkpoint with rows from a prior checkpoint without carrying its paid-attempt total.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    prereg = load_json(PREREG)
    phase_c_dry = load_json(PHASE_C_DRY_RUN)
    ledger = load_json(BUDGET_LEDGER)
    configure_paid_call_environment()
    output_path = Path(args.output)
    dry = build_dry_run(prereg, phase_c_dry, ledger, output_path, args.reasoning_model, args.judge_model)
    dry = apply_checkpoint_guard(dry, output_path)
    write_json(Path(args.dry_output), dry)
    print(f"wrote Phase C pilot runner dry-run: {args.dry_output}")
    print(f"dry_run_status={dry['stage_status']}")
    print(f"pilot_llm_http_attempt_cap={dry['pilot_llm_http_attempt_cap']}")
    print("paid_calls_executed=0")
    if not args.run_paid:
        return 0 if dry["stage_status"] == "ready_for_paid_pilot" else 2
    result = run_paid(
        Path(args.output),
        args.max_new_samples,
        args.max_new_jobs,
        args.reasoning_model,
        args.judge_model,
        args.restart_after_unaccounted_interruption,
        args.resume_source,
    )
    update_budget_ledger(Path(args.output), result)
    print(f"stage_status={result['stage_status']}")
    print(f"completed_sample_quality_jobs={result['completed_sample_quality_jobs']}")
    print(f"actual_llm_http_attempts={result['actual_llm_http_attempts']}")
    print(f"new_llm_http_attempts_this_invocation={result['new_llm_http_attempts_this_invocation']}")
    print(f"actual_ocr_http_attempts={result['actual_ocr_http_attempts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
