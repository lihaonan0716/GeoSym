#!/usr/bin/env python
"""Run the paper-level PlotQA 500 measured-data confirmation.

Safe default: write a frozen plan, dry-run, and prepaid review with
``paid_calls_executed=0``. Use ``--run-paid`` only after the generated plan and
review are inspected and the per-track cap is still appropriate. Paid mode
supports checkpoint/resume and ``--max-new-jobs`` for bounded execution.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluator.adapters.base as adapter_base  # noqa: E402
from app.helpers import _create_llm_adapter, _get_saved_key  # noqa: E402
from scripts.prepare_public_benchmark_pilot import canonical_json_bytes, sha256_hex_bytes, stable_file_hash  # noqa: E402
from scripts.prepare_plotqa_public_benchmark_pilot import SCORER_VERSION  # noqa: E402
from scripts.run_plotqa_locate import (  # noqa: E402
    CONDITIONS,
    answer_prompt,
    corrected_numeric_prediction_from_response,
    measured_prompt_facts,
    prompt_hashes,
)
from scripts.run_plotqa_measure import as_records_by_id  # noqa: E402
from scripts.run_public_benchmark_pilot import AttemptCounter, extract_prediction, score_prediction  # noqa: E402


PAPER_SAMPLE_COUNT = 500
PAPER_LLM_ATTEMPT_CAP = 1500
PAPER_OCR_ATTEMPT_CAP = 0
# Frozen PaddleOCR boxes for the 500-sample roster. Read from disk by the M0'
# chain; no OCR inference runs on the answering path.
DEFAULT_OCR_PATH = (
    ROOT / "outputs" / "paper_level_evidence_package" / "plotqa_public_500"
    / "openocr_survey_paddleocr" / "raw_ocr.json"
)
COMPLETION_GATE = 490
PER_CALL_TIMEOUT_SECONDS = 90.0
GLM_DISPLAY_NAME = "GLM-4.6V"
GLM_API_MODEL_ID = "glm-4.6v"
# GLM-4.6V spends its token budget on internal thinking before emitting visible
# text. At 4096 the GeoSym-Bench run returned finish_reason='length' with an
# EMPTY body on 90.9% of truncated calls, which scores as a failure and confounds
# every condition mean. See docs/GLM46V_240X7_RERUN_PLAN_2026-09-17.md section 1.
GLM_MAX_TOKENS = 16384
# The GeoSym-Bench pilot lost ~17% of calls to APITimeoutError at 180s, with the
# failures tracking problem difficulty. Same backbone, same constraint here.
GLM_TIMEOUT_SECONDS = 600.0
CONFIRMATION_STATUSES = {
    "track_c_plotqa_500_confirmation_completed",
    "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_samples(plan: dict[str, Any], measured_records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [sample for sample in plan["selected_samples"] if sample["sample_id"] in measured_records]


def build_plan(
    plotqa_plan_path: Path,
    measured_records_path: Path,
    stage1_status_path: Path,
    output_dir: Path,
    public_baseline_result_path: Path | None = None,
    reasoning_model: str | None = None,
    per_call_timeout_seconds: float = PER_CALL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    plotqa_plan = load_json(plotqa_plan_path)
    measured_records = as_records_by_id(measured_records_path)
    stage1_status = load_json(stage1_status_path)
    samples = selected_samples(plotqa_plan, measured_records)
    sample_ids = [sample["sample_id"] for sample in samples]
    schema_valid = sum(1 for sample_id in sample_ids if not measured_records[sample_id].get("schema_violations"))
    stop_reason = None
    if plotqa_plan.get("benchmark") != "PlotQA":
        stop_reason = "source_plan_not_plotqa"
    elif len(plotqa_plan.get("selected_samples") or []) != PAPER_SAMPLE_COUNT:
        stop_reason = "source_plan_not_500_samples"
    elif len(samples) != PAPER_SAMPLE_COUNT:
        stop_reason = "missing_measured_records_for_selected_samples"
    elif stage1_status.get("stage_status") != "stage_passed":
        stop_reason = "measured_m0_stage1_not_passed"
    elif stage1_status.get("sample_count") != PAPER_SAMPLE_COUNT:
        stop_reason = "measured_m0_stage1_not_500_samples"
    elif schema_valid != PAPER_SAMPLE_COUNT:
        stop_reason = "measured_record_schema_invalid"
    baseline_result = load_json(public_baseline_result_path) if public_baseline_result_path else None
    if baseline_result is not None:
        baseline_conditions = {
            row.get("condition")
            for row in baseline_result.get("results", [])
            if row.get("sample_id") in set(sample_ids)
        }
        required_conditions = {"E0_plotqa_direct", "E1_plotqa_cot"}
        baseline_rows = [
            row
            for row in baseline_result.get("results", [])
            if row.get("sample_id") in set(sample_ids)
            and row.get("condition") in required_conditions
        ]
        if len(baseline_rows) != PAPER_SAMPLE_COUNT * len(required_conditions):
            stop_reason = stop_reason or "public_baseline_result_missing_required_rows"
        cached_conditions = sorted(required_conditions | ({"G0_plotqa_evidence_trace"} if "G0_plotqa_evidence_trace" in baseline_conditions else set()))
    else:
        cached_conditions = []
    frozen = plotqa_plan.get("frozen_before_paid_run") or {}
    source_model_provenance = dict(frozen.get("model_provenance", {}))
    display_name = reasoning_model or source_model_provenance.get("display_name")
    if reasoning_model == GLM_DISPLAY_NAME:
        model_provenance = {
            "display_name": GLM_DISPLAY_NAME,
            "api_model_id": GLM_API_MODEL_ID,
            "adapter": "evaluator.adapters.glm.GLMAdapter",
            "timeout_seconds": per_call_timeout_seconds,
            "hard_timeout_seconds": per_call_timeout_seconds,
            "max_tokens": GLM_MAX_TOKENS,
            "temperature": 0.0,
            "sdk_retries": 0,
        }
    else:
        model_provenance = source_model_provenance
    plan = {
        "schema_version": "plotqa_measured_public_500_confirmation_plan_v1",
        "created_at_utc": utc_now(),
        "stage": "track_c_plotqa_500_measured_data_confirmation",
        "paid_run_approved": stop_reason is None,
        "paid_calls_executed": 0,
        "hypothesis": "On the frozen 500-sample official PlotQA public roster, measured-data GeoSym improves over matched direct prompting and evidence-trace baselines while preserving measured-flat control visibility.",
        "falsification_signal": "If G_full_measured_geosym does not exceed the strongest matched baseline, or measured-flat matches/exceeds it, do not claim public benchmark benefit from reliability-aware fusion.",
        "selected_ids": sample_ids,
        "selected_samples": samples,
        "conditions": list(CONDITIONS),
        "cached_conditions_reused": cached_conditions,
        "source_artifacts": {
            "plotqa_500_plan": {"path": str(plotqa_plan_path.resolve()), "sha256": stable_file_hash(plotqa_plan_path)},
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
            "stage1_status": {
                "path": str(stage1_status_path.resolve()),
                "sha256": stable_file_hash(stage1_status_path),
                "stage_status": stage1_status.get("stage_status"),
            },
            "public_baseline_result": {
                "path": str(public_baseline_result_path.resolve()) if public_baseline_result_path else None,
                "sha256": stable_file_hash(public_baseline_result_path) if public_baseline_result_path else None,
                "required_for_public_benefit_claim": True,
                "available": public_baseline_result_path is not None and public_baseline_result_path.exists(),
            },
        },
        "model_provenance": model_provenance,
        "reasoning_model": display_name,
        "per_call_timeout_seconds": per_call_timeout_seconds,
        "hard_timeout_seconds": per_call_timeout_seconds,
        "glm_environment": {
            "GEOSYM_GLM_TIMEOUT_SECONDS": str(int(GLM_TIMEOUT_SECONDS)) if reasoning_model == GLM_DISPLAY_NAME else None,
            "GEOSYM_GLM_MAX_TOKENS": str(GLM_MAX_TOKENS) if reasoning_model == GLM_DISPLAY_NAME else None,
            "GEOSYM_GLM_MAX_RETRIES": "0" if reasoning_model == GLM_DISPLAY_NAME else None,
        },
        "source_sample_selection": frozen.get("sample_selection"),
        "temperature": 0.0,
        "prompt_hashes": prompt_hashes(),
        "scorer_version": frozen.get("scorer_version", SCORER_VERSION),
        "sample_condition_job_cap": PAPER_SAMPLE_COUNT * len(CONDITIONS),
        "llm_http_attempt_cap": PAPER_LLM_ATTEMPT_CAP,
        "ocr_http_attempt_cap": PAPER_OCR_ATTEMPT_CAP,
        "per_condition_attempt_caps": {
            "G_measured_raw": {"llm": PAPER_SAMPLE_COUNT, "ocr": 0},
            "G_full_measured_geosym": {"llm": PAPER_SAMPLE_COUNT, "ocr": 0},
            "Measured_flat_answer_baseline": {"llm": PAPER_SAMPLE_COUNT, "ocr": 0},
        },
        "retry_policy": "one counted HTTP attempt per sample-condition LLM fallback; no adaptive retries or sample replacement",
        "cache_policy": "cache key must include plan hash, model ID, condition, prompt hash, sample ID, scorer version, and measured-record hash",
        "stop_rule": "stop after the frozen 500 samples and at most 1500 LLM HTTP attempts; OCR HTTP attempts fixed at 0; no sample replacement",
        "cost_estimate": "maximum 1500 LLM HTTP attempts, one per sample-condition; OCR HTTP attempts fixed at 0 (cached boxes only)",
        "dry_run": {
            "paid_calls_executed": 0,
            "cap_enforced": PAPER_LLM_ATTEMPT_CAP == PAPER_SAMPLE_COUNT * len(CONDITIONS),
            "stopped_before_paid_calls": stop_reason is not None,
            "stop_reason": stop_reason,
        },
        "interpretation_limits": [
            "This is a frozen 500-sample PlotQA confirmation, not the full PlotQA benchmark or leaderboard.",
            "Measured-flat receives the same measured facts as full measured GeoSym.",
            "Public benefit claims require matched E0/E1 baseline rows on the same roster; G0 is optional diagnostic context under the Track C cap.",
            "The current M1/M2 curation remains heuristic and not calibrated.",
            "Both measured arms consume the same M0' chain output and the same prompt apart from one sentence of reliability semantics; the arms differ by payload only.",
        ],
    }
    plan["plan_sha256"] = sha256_hex_bytes(canonical_json_bytes({k: v for k, v in plan.items() if k != "plan_sha256"}))
    return plan


def review_plan(plan: dict[str, Any]) -> dict[str, Any]:
    findings = []
    if len(plan.get("selected_ids") or []) != PAPER_SAMPLE_COUNT:
        findings.append("selected_roster_not_500")
    if plan.get("sample_condition_job_cap") != PAPER_SAMPLE_COUNT * len(CONDITIONS):
        findings.append("sample_condition_cap_not_1500")
    if plan.get("llm_http_attempt_cap") != PAPER_LLM_ATTEMPT_CAP:
        findings.append("llm_cap_not_1500")
    if plan.get("ocr_http_attempt_cap") != PAPER_OCR_ATTEMPT_CAP:
        findings.append("ocr_cap_not_zero")
    if plan.get("dry_run", {}).get("paid_calls_executed") != 0:
        findings.append("dry_run_executed_paid_calls")
    if set(plan.get("conditions") or []) != set(CONDITIONS):
        findings.append("measured_conditions_not_frozen")
    if plan.get("source_artifacts", {}).get("stage1_status", {}).get("stage_status") != "stage_passed":
        findings.append("measured_m0_stage1_not_passed")
    if not plan.get("model_provenance", {}).get("display_name"):
        findings.append("model_display_name_missing")
    if plan.get("reasoning_model") == GLM_DISPLAY_NAME:
        if plan.get("model_provenance", {}).get("api_model_id") != GLM_API_MODEL_ID:
            findings.append("glm_api_model_id_mismatch")
        if float(plan.get("per_call_timeout_seconds") or 0) != GLM_TIMEOUT_SECONDS:
            findings.append("glm_timeout_mismatch")
        if (plan.get("glm_environment") or {}).get("GEOSYM_GLM_TIMEOUT_SECONDS") != str(int(GLM_TIMEOUT_SECONDS)):
            findings.append("glm_env_timeout_mismatch")
        env = plan.get("glm_environment") or {}
        if env.get("GEOSYM_GLM_MAX_RETRIES") != "0":
            findings.append("glm_retries_not_zero")
        # A plan frozen at 4096 reproduces the empty-output failure mode.
        if plan.get("model_provenance", {}).get("max_tokens") != GLM_MAX_TOKENS:
            findings.append("glm_max_tokens_mismatch")
        if env.get("GEOSYM_GLM_MAX_TOKENS") != str(GLM_MAX_TOKENS):
            findings.append("glm_env_max_tokens_mismatch")
        source_selection = plan.get("source_sample_selection")
        if source_selection != (load_json(Path(plan["source_artifacts"]["plotqa_500_plan"]["path"])).get("frozen_before_paid_run") or {}).get("sample_selection"):
            findings.append("source_sample_selection_not_carried_verbatim")
    return {
        "schema_version": "plotqa_measured_public_500_confirmation_prepaid_review_v1",
        "created_at_utc": utc_now(),
        "status": "no_blocking_findings" if not findings else "blocking_findings_present",
        "blocking_findings": findings,
        "paid_calls_executed": 0,
    }


def call_with_hard_timeout(
    display_name: str,
    key: str,
    image_path: Path | None,
    prompt: str,
    timeout_seconds: float = PER_CALL_TIMEOUT_SECONDS,
) -> str:
    responses: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            adapter = _create_llm_adapter(display_name, key)
            if adapter is None:
                raise ValueError(f"could not create adapter for {display_name}")
            responses.put(("ok", adapter.call_with_prompt(image_path, prompt)))
        except BaseException as exc:  # Propagate API and timeout-layer failures to the result row.
            responses.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise TimeoutError(f"llm_call_hard_timeout_after_{timeout_seconds:.0f}s")
    status, payload = responses.get_nowait()
    if status == "error":
        raise payload
    return str(payload)


def write_result(
    plan: dict[str, Any],
    plan_path: Path,
    output_path: Path,
    results: list[dict[str, Any]],
    attempts: int,
    status: str | None = None,
) -> dict[str, Any]:
    result = analyze_result(plan, plan_path, results, attempts)
    if status is not None:
        result["stage_status"] = status
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def run_paid(
    plan: dict[str, Any],
    measured_records_path: Path,
    plan_path: Path,
    output_path: Path,
    max_new_jobs: int | None = None,
    ocr_path: Path | None = None,
) -> dict[str, Any]:
    if plan.get("paid_run_approved") is not True:
        raise ValueError(plan.get("dry_run", {}).get("stop_reason") or "plan_not_approved")
    display_name = plan.get("model_provenance", {}).get("display_name")
    key = _get_saved_key(display_name)
    if not key:
        raise ValueError(f"saved API key not found for {display_name}")
    records = as_records_by_id(measured_records_path)
    # Cached OCR boxes drive the M0' chain's axis refit and category axis. They
    # are read from disk, so this adds no OCR inference and no HTTP attempts.
    ocr_rows = {
        row["sample_id"]: row
        for row in load_json(ocr_path or DEFAULT_OCR_PATH)
        if isinstance(row, dict) and row.get("sample_id")
    }
    results = []
    completed_keys = set()
    if output_path.exists():
        checkpoint = load_json(output_path)
        checkpoint_hash = checkpoint.get("plan_sha256")
        if checkpoint_hash != plan.get("plan_sha256"):
            raise ValueError(f"checkpoint plan hash mismatch: {checkpoint_hash} != {plan.get('plan_sha256')}")
        results = list(checkpoint.get("results") or [])
        completed_keys = {(row.get("sample_id"), row.get("condition")) for row in results}
    counter = AttemptCounter(plan["llm_http_attempt_cap"])
    counter.llm_attempts = sum(int(row.get("llm_http_attempts") or 0) for row in results)
    new_jobs = 0
    original_retries = adapter_base._MAX_API_RETRIES
    adapter_base._MAX_API_RETRIES = 1
    try:
        for sample in plan["selected_samples"]:
            record = records[sample["sample_id"]]
            for condition in CONDITIONS:
                if (sample["sample_id"], condition) in completed_keys:
                    continue
                if max_new_jobs is not None and new_jobs >= max_new_jobs:
                    return write_result(plan, plan_path, output_path, results, counter.llm_attempts, "partial_paid_run")
                prediction = None
                raw_text = ""
                error = ""
                # The deterministic solver that used to short-circuit this arm is
                # gone. It fired only for G_full_measured_geosym, which made the
                # arms differ by code path as well as evidence (invariant 6), and
                # it was a net loss on its own terms: across the 170 samples it
                # took over it scored 56 correct where the same samples' LLM arm
                # scored 130. Both arms now answer the same way, and the only
                # difference between them is the payload.
                deterministic = False
                facts = measured_prompt_facts(
                    record,
                    None,
                    curated=condition == "G_full_measured_geosym",
                    ocr_row=ocr_rows.get(sample["sample_id"]),
                )
                counter.consume_llm(sample["sample_id"], condition, "answer")
                image_path = None if condition == "Measured_flat_answer_baseline" else Path(sample["image_path"])
                try:
                    raw_text = call_with_hard_timeout(
                        display_name,
                        key,
                        image_path,
                        answer_prompt(sample["question"], facts, condition),
                        timeout_seconds=float(plan.get("per_call_timeout_seconds") or PER_CALL_TIMEOUT_SECONDS),
                    )
                    prediction = extract_prediction(raw_text, sample.get("answer_type"))
                    prediction = corrected_numeric_prediction_from_response(sample, record, raw_text, prediction)
                except Exception as exc:
                    error = str(exc)
                parseable = prediction is not None
                try:
                    score = score_prediction(prediction, sample.get("normalized_answer"), sample.get("answer_type")) if parseable else {"correct": False, "numeric_error": None, "tolerance": None}
                except Exception:
                    score = {"correct": False, "numeric_error": None, "tolerance": None}
                results.append(
                    {
                        "sample_id": sample["sample_id"],
                        "condition": condition,
                        "answer_type": sample.get("answer_type"),
                        "reference": sample.get("normalized_answer"),
                        "completed": parseable,
                        "parseable": parseable,
                        "normalized_prediction": prediction,
                        "raw_text": raw_text,
                        "error": error,
                        "llm_http_attempts": 0 if deterministic else 1,
                        "ocr_http_attempts": 0,
                        "deterministic_m3": deterministic,
                        **score,
                    }
                )
                new_jobs += 1
                print(f"completed_sample={sample['sample_id']} condition={condition} llm_attempts={counter.llm_attempts}/{plan['llm_http_attempt_cap']}", flush=True)
                write_result(plan, plan_path, output_path, results, counter.llm_attempts, "partial_paid_run")
    finally:
        adapter_base._MAX_API_RETRIES = original_retries
    return write_result(plan, plan_path, output_path, results, counter.llm_attempts)


def summarize_by_condition(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_condition = defaultdict(list)
    for row in results:
        by_condition[row["condition"]].append(row)
    return {
        condition: {
            "completed": sum(1 for row in rows if row.get("completed")),
            "parseable": sum(1 for row in rows if row.get("parseable")),
            "correct": sum(1 for row in rows if row.get("correct")),
            "deterministic": sum(1 for row in rows if row.get("deterministic_m3")),
        }
        for condition, rows in by_condition.items()
    }


def cached_baseline_correct(plan: dict[str, Any]) -> dict[str, int]:
    source = plan.get("source_artifacts", {}).get("public_baseline_result", {})
    if not source.get("path"):
        return {}
    baseline = load_json(Path(source["path"]))
    selected = set(plan["selected_ids"])
    return {
        condition: sum(
            1
            for row in baseline.get("results", [])
            if row.get("sample_id") in selected and row.get("condition") == condition and row.get("correct")
        )
        for condition in plan.get("cached_conditions_reused", [])
    }


def analyze_result(plan: dict[str, Any], plan_path: Path, results: list[dict[str, Any]], attempts: int) -> dict[str, Any]:
    summary = summarize_by_condition(results)
    baseline_correct = cached_baseline_correct(plan)
    full_correct = summary.get("G_full_measured_geosym", {}).get("correct", 0)
    strongest_baseline = max(baseline_correct.values()) if baseline_correct else None
    operational = all(
        summary.get(condition, {}).get("completed", 0) >= COMPLETION_GATE
        and summary.get(condition, {}).get("parseable", 0) >= COMPLETION_GATE
        for condition in CONDITIONS
    )
    benefit_gate = strongest_baseline is not None and full_correct > strongest_baseline
    flat_correct = summary.get("Measured_flat_answer_baseline", {}).get("correct", 0)
    flat_separation = full_correct > flat_correct
    if operational and benefit_gate and flat_separation:
        status = "track_c_plotqa_500_confirmation_completed"
    else:
        status = "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim"
    return {
        "schema_version": "plotqa_measured_public_500_confirmation_result_v1",
        "created_at_utc": utc_now(),
        "stage": plan["stage"],
        "stage_status": status,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": plan["plan_sha256"],
        "sample_count": len(plan["selected_ids"]),
        "conditions": list(CONDITIONS),
        "actual_llm_http_attempts": attempts,
        "actual_ocr_http_attempts": 0,
        "paid_calls_executed": attempts,
        "summary": summary,
        "cached_baseline_correct": baseline_correct,
        "go_gate": {
            "new_conditions_completed_at_least_490_of_500": operational,
            "new_conditions_parseable_at_least_490_of_500": operational,
            "matched_public_baselines_available": bool(baseline_correct),
            "g_full_exceeds_strongest_matched_baseline": benefit_gate,
            "g_full_exceeds_measured_flat": flat_separation,
            "within_attempt_cap": attempts <= plan["llm_http_attempt_cap"],
            "ocr_http_attempts_zero": True,
        },
        "results": results,
        "interpretation_limits": plan["interpretation_limits"],
    }


def render_audit(result: dict[str, Any]) -> str:
    lines = [
        "# PlotQA Measured Public 500 Confirmation",
        "",
        f"- Stage status: `{result['stage_status']}`",
        f"- Paid calls executed: `{result['paid_calls_executed']}`",
        f"- Actual LLM HTTP attempts: `{result['actual_llm_http_attempts']}`",
        f"- Actual OCR HTTP attempts: `{result['actual_ocr_http_attempts']}`",
        "",
        "## Summary",
        "",
    ]
    for condition, row in result["summary"].items():
        lines.append(
            f"- {condition}: correct {row['correct']}/{result['sample_count']}, completed {row['completed']}/{result['sample_count']}, parseable {row['parseable']}/{result['sample_count']}, deterministic {row['deterministic']}"
        )
    lines.extend(["", "## Go Gate", ""])
    for key, value in result["go_gate"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plotqa-plan", default="outputs/paper_level_evidence_package/plotqa_public_500/plotqa_500_zero_cost_plan.json")
    parser.add_argument("--measured-records", default="outputs/paper_level_evidence_package/plotqa_public_500/m0_measurement_records.json")
    parser.add_argument("--stage1-status", default="outputs/paper_level_evidence_package/plotqa_public_500/stage1_status.json")
    parser.add_argument("--public-baseline-result")
    parser.add_argument("--output-dir", default="outputs/paper_level_evidence_package/plotqa_public_500/measured_confirmation")
    parser.add_argument("--reasoning-model")
    parser.add_argument("--per-call-timeout-seconds", type=float, default=PER_CALL_TIMEOUT_SECONDS)
    parser.add_argument("--run-paid", action="store_true")
    parser.add_argument("--max-new-jobs", type=int)
    parser.add_argument("--ocr", type=Path, default=DEFAULT_OCR_PATH,
                        help="cached OCR boxes feeding the M0' chain; read from disk, never re-inferred")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "frozen_plan.json"
    if args.run_paid and plan_path.exists():
        plan = load_json(plan_path)
    else:
        plan = build_plan(
            Path(args.plotqa_plan),
            Path(args.measured_records),
            Path(args.stage1_status),
            output_dir,
            Path(args.public_baseline_result) if args.public_baseline_result else None,
            reasoning_model=args.reasoning_model,
            per_call_timeout_seconds=args.per_call_timeout_seconds,
        )
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    review = review_plan(plan)
    (output_dir / "prepaid_review.json").write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote frozen PlotQA measured 500 plan: {plan_path}")
    print(f"llm_http_attempt_cap={plan['llm_http_attempt_cap']}")
    print("ocr_http_attempt_cap=0")
    if not args.run_paid:
        (output_dir / "dry_run.json").write_text(json.dumps(plan["dry_run"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"prepaid_review_status={review['status']}")
        print("dry-run complete; paid_calls_executed=0")
        return 0 if plan["paid_run_approved"] and review["status"] == "no_blocking_findings" else 2
    if review["status"] != "no_blocking_findings":
        raise SystemExit("prepaid review has blocking findings")
    if plan.get("reasoning_model") == GLM_DISPLAY_NAME:
        os.environ["GEOSYM_GLM_TIMEOUT_SECONDS"] = str(int(GLM_TIMEOUT_SECONDS))
        os.environ["GEOSYM_GLM_MAX_TOKENS"] = str(GLM_MAX_TOKENS)
        os.environ["GEOSYM_GLM_MAX_RETRIES"] = "0"
    os.environ.setdefault("GEOSYM_OPENAI_TIMEOUT_SECONDS", "90")
    result = run_paid(plan, Path(args.measured_records), plan_path, output_dir / "paid_result.json", max_new_jobs=args.max_new_jobs, ocr_path=Path(args.ocr))
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "analysis.md").write_text(render_audit(result), encoding="utf-8")
    print(f"stage_status={result['stage_status']}")
    print(f"actual_llm_http_attempts={result['actual_llm_http_attempts']}")
    print("actual_ocr_http_attempts=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
