#!/usr/bin/env python
"""Stage 5 bounded repair smoke for measured-GeoSym G_full only.

Safe default: write a frozen plan and exit with paid_calls_executed=0.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluator.adapters.base as adapter_base  # noqa: E402
from app.helpers import _create_llm_adapter, _get_saved_key  # noqa: E402
from scripts.prepare_public_benchmark_pilot import canonical_json_bytes, sha256_hex_bytes, stable_file_hash  # noqa: E402
from scripts.run_plotqa_measured_geosym_stage3 import answer_prompt, corrected_numeric_prediction_from_response, deterministic_prediction_from_measured, measured_prompt_facts, prompt_hashes  # noqa: E402
from scripts.run_plotqa_measured_geosym_stage4 import as_records_by_id  # noqa: E402
from scripts.run_public_benchmark_pilot import AttemptCounter, extract_prediction, score_prediction  # noqa: E402


CONDITION = "G_full_measured_geosym"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def select_repair_samples(stage4_result: dict[str, Any], diagnostic_plan: dict[str, Any]) -> list[dict[str, Any]]:
    sample_by_id = {sample["sample_id"]: sample for sample in diagnostic_plan["selected_samples"]}
    diagnostic_result = load_json(Path(stage4_result["source_diagnostic_result_path"])) if "source_diagnostic_result_path" in stage4_result else None
    e0_correct = set()
    if diagnostic_result:
        e0_correct = {row["sample_id"] for row in diagnostic_result.get("results", []) if row.get("condition") == "E0_plotqa_direct" and row.get("correct")}
    full_rows = [row for row in stage4_result.get("results", []) if row.get("condition") == CONDITION]
    wrong_ids = [row["sample_id"] for row in full_rows if not row.get("correct")]
    degraded_ids = [row["sample_id"] for row in full_rows if row["sample_id"] in e0_correct and not row.get("correct")]
    ordered = []
    for sample_id in [*degraded_ids, *wrong_ids]:
        if sample_id in sample_by_id and sample_id not in ordered:
            ordered.append(sample_id)
    for row in full_rows:
        if row["sample_id"] in sample_by_id and row["sample_id"] not in ordered:
            ordered.append(row["sample_id"])
        if len(ordered) >= 8:
            break
    return [sample_by_id[sample_id] for sample_id in ordered[:8]]


def build_plan(
    diagnostic_plan_path: Path,
    diagnostic_result_path: Path,
    measured_records_path: Path,
    stage4_result_path: Path,
    output_dir: Path,
    repair_iteration: int = 1,
    dominant_bottleneck: str = "m3_solver_overrides_on_unreliable_measured_extrema",
) -> dict[str, Any]:
    diagnostic_plan = load_json(diagnostic_plan_path)
    stage4_result = load_json(stage4_result_path)
    stage4_result["source_diagnostic_result_path"] = str(diagnostic_result_path.resolve())
    measured_records = as_records_by_id(measured_records_path)
    samples = select_repair_samples(stage4_result, diagnostic_plan)
    sample_ids = [sample["sample_id"] for sample in samples]
    stop_reason = None
    if stage4_result.get("stage_status") != "stage_failed_exhausted":
        stop_reason = "stage5_repair_requires_failed_stage4"
    elif len(samples) != 8:
        stop_reason = "stage5_repair_requires_8_samples"
    elif any(sample_id not in measured_records for sample_id in sample_ids):
        stop_reason = "missing_measured_records"
    cap = 8
    plan = {
        "schema_version": "plotqa_measured_geosym_stage5_repair_smoke_plan_v1",
        "created_at_utc": utc_now(),
        "stage": "stage5_m3_solver_repair_smoke",
        "paid_run_approved": stop_reason is None,
        "paid_calls_executed": 0,
        "repair_iteration": repair_iteration,
        "dominant_bottleneck": dominant_bottleneck,
        "hypothesis": "Failing closed on low-confidence deterministic extrema reduces G_full degradations and improves the Stage 4 wrong/degraded repair set.",
        "falsification_signal": "If repaired G_full correct count does not exceed the previous Stage 4 G_full count on the same 8 samples, keep Stage 5 focused on M0 ruler/measurement repair.",
        "selected_ids": sample_ids,
        "selected_samples": samples,
        "condition": CONDITION,
        "source_artifacts": {
            "diagnostic_plan": {"path": str(diagnostic_plan_path.resolve()), "sha256": stable_file_hash(diagnostic_plan_path)},
            "diagnostic_result": {"path": str(diagnostic_result_path.resolve()), "sha256": stable_file_hash(diagnostic_result_path)},
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
            "stage4_result": {"path": str(stage4_result_path.resolve()), "sha256": stable_file_hash(stage4_result_path), "stage_status": stage4_result.get("stage_status")},
        },
        "model_provenance": diagnostic_plan.get("frozen_before_paid_run", {}).get("model_provenance", {}),
        "reasoning_model": diagnostic_plan.get("frozen_before_paid_run", {}).get("reasoning_model"),
        "temperature": 0.0,
        "prompt_hashes": prompt_hashes(),
        "llm_http_attempt_cap": cap,
        "ocr_http_attempt_cap": 0,
        "per_sample_attempt_cap": 1,
        "retry_policy": "one counted HTTP attempt per sample only when deterministic G_full does not fire; no adaptive retries or sample replacement",
        "stop_rule": "stop after the frozen 8 repair samples and at most 8 LLM HTTP attempts; OCR HTTP attempts fixed at 0",
        "cost_estimate": "maximum 8 LLM HTTP attempts; deterministic G_full answers reduce actual attempts; OCR HTTP attempts fixed at 0",
        "dry_run": {
            "paid_calls_executed": 0,
            "cap_enforced": cap == 8,
            "stopped_before_paid_calls": stop_reason is not None,
            "stop_reason": stop_reason,
        },
        "interpretation_limits": [
            "This is a post-Stage4 repair smoke over wrong/degraded cases, not confirmatory public-benchmark evidence.",
            "Success permits one 50-sample confirmation under Stage 5; it does not replace that confirmation.",
        ],
    }
    plan["plan_sha256"] = sha256_hex_bytes(canonical_json_bytes({k: v for k, v in plan.items() if k != "plan_sha256"}))
    return plan


def review_plan(plan: dict[str, Any]) -> dict[str, Any]:
    findings = []
    if len(plan.get("selected_ids") or []) != 8:
        findings.append("selected_roster_not_8")
    if plan.get("llm_http_attempt_cap") != 8:
        findings.append("llm_cap_not_8")
    if plan.get("ocr_http_attempt_cap") != 0:
        findings.append("ocr_cap_not_zero")
    if plan.get("source_artifacts", {}).get("stage4_result", {}).get("stage_status") != "stage_failed_exhausted":
        findings.append("stage4_source_not_failed")
    return {
        "schema_version": "plotqa_measured_geosym_stage5_repair_smoke_prepaid_review_v1",
        "created_at_utc": utc_now(),
        "blocking_findings": findings,
        "status": "no_blocking_findings" if not findings else "blocking_findings_present",
        "paid_calls_executed": 0,
    }


def run_paid(plan: dict[str, Any], measured_records_path: Path, stage4_result_path: Path, plan_path: Path) -> dict[str, Any]:
    if plan.get("paid_run_approved") is not True:
        raise ValueError(plan.get("dry_run", {}).get("stop_reason") or "plan_not_approved")
    display_name = plan.get("model_provenance", {}).get("display_name")
    key = _get_saved_key(display_name)
    adapter = _create_llm_adapter(display_name, key) if key else None
    if adapter is None:
        raise ValueError(f"could not create adapter for {display_name}")
    records = as_records_by_id(measured_records_path)
    previous = {
        row["sample_id"]: row
        for row in load_json(stage4_result_path).get("results", [])
        if row.get("condition") == CONDITION
    }
    counter = AttemptCounter(plan["llm_http_attempt_cap"])
    results = []
    original_retries = adapter_base._MAX_API_RETRIES
    adapter_base._MAX_API_RETRIES = 1
    try:
        for sample in plan["selected_samples"]:
            record = records[sample["sample_id"]]
            prediction = deterministic_prediction_from_measured(sample, record, None)
            deterministic = prediction is not None
            raw_text = f"deterministic_m3_prediction={prediction}" if deterministic else ""
            error = ""
            if prediction is None:
                facts = measured_prompt_facts(record, None, curated=True)
                counter.consume_llm(sample["sample_id"], CONDITION, "answer")
                try:
                    raw_text = adapter.call_with_prompt(Path(sample["image_path"]), answer_prompt(sample["question"], facts, CONDITION))
                    prediction = extract_prediction(raw_text, sample.get("answer_type"))
                    prediction = corrected_numeric_prediction_from_response(sample, record, raw_text, prediction)
                except Exception as exc:
                    error = str(exc)
            parseable = prediction is not None
            score = score_prediction(prediction, sample.get("normalized_answer"), sample.get("answer_type")) if parseable else {"correct": False, "numeric_error": None, "tolerance": None}
            prev = previous.get(sample["sample_id"], {})
            results.append(
                {
                    "sample_id": sample["sample_id"],
                    "condition": CONDITION,
                    "reference": sample.get("normalized_answer"),
                    "answer_type": sample.get("answer_type"),
                    "completed": parseable,
                    "parseable": parseable,
                    "normalized_prediction": prediction,
                    "raw_text": raw_text,
                    "error": error,
                    "llm_http_attempts": 0 if deterministic else 1,
                    "ocr_http_attempts": 0,
                    "deterministic_m3": deterministic,
                    "previous_stage4_prediction": prev.get("normalized_prediction"),
                    "previous_stage4_correct": prev.get("correct"),
                    **score,
                }
            )
            print(f"completed_sample={sample['sample_id']} llm_attempts={counter.llm_attempts}/{plan['llm_http_attempt_cap']}", flush=True)
    finally:
        adapter_base._MAX_API_RETRIES = original_retries
    return analyze_result(plan, plan_path, results, counter.llm_attempts)


def analyze_result(plan: dict[str, Any], plan_path: Path, results: list[dict[str, Any]], attempts: int) -> dict[str, Any]:
    correct = sum(1 for row in results if row.get("correct"))
    previous_correct = sum(1 for row in results if row.get("previous_stage4_correct"))
    completed = sum(1 for row in results if row.get("completed"))
    status = "stage_passed" if completed == len(results) and correct > previous_correct else "stage_failed_exhausted"
    return {
        "schema_version": "plotqa_measured_geosym_stage5_repair_smoke_result_v1",
        "created_at_utc": utc_now(),
        "stage": plan["stage"],
        "stage_status": status,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": plan["plan_sha256"],
        "sample_count": len(results),
        "condition": CONDITION,
        "actual_llm_http_attempts": attempts,
        "actual_ocr_http_attempts": 0,
        "paid_calls_executed": attempts,
        "summary": {"completed": completed, "correct": correct, "previous_stage4_correct": previous_correct},
        "go_gate": {"correct_exceeds_previous_stage4_on_same_samples": correct > previous_correct, "within_attempt_cap": attempts <= plan["llm_http_attempt_cap"], "ocr_http_attempts_zero": True},
        "results": results,
        "interpretation_limits": plan["interpretation_limits"],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-plan", required=True)
    parser.add_argument("--diagnostic-result", required=True)
    parser.add_argument("--measured-records", required=True)
    parser.add_argument("--stage4-result", required=True)
    parser.add_argument("--output-dir", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/stage5_repair_smoke1")
    parser.add_argument("--repair-iteration", type=int, default=1)
    parser.add_argument("--dominant-bottleneck", default="m3_solver_overrides_on_unreliable_measured_extrema")
    parser.add_argument("--run-paid", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(
        Path(args.diagnostic_plan),
        Path(args.diagnostic_result),
        Path(args.measured_records),
        Path(args.stage4_result),
        output_dir,
        repair_iteration=args.repair_iteration,
        dominant_bottleneck=args.dominant_bottleneck,
    )
    plan_path = output_dir / "frozen_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote frozen Stage 5 repair plan: {plan_path}")
    print(f"llm_http_attempt_cap={plan['llm_http_attempt_cap']}")
    print("ocr_http_attempt_cap=0")
    if not args.run_paid:
        (output_dir / "dry_run.json").write_text(json.dumps(plan["dry_run"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output_dir / "prepaid_review.json").write_text(json.dumps(review_plan(plan), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("dry-run complete; paid_calls_executed=0")
        return 0 if plan["paid_run_approved"] else 2
    result = run_paid(plan, Path(args.measured_records), Path(args.stage4_result), plan_path)
    (output_dir / "paid_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "stage_status.json").write_text(json.dumps({"stage": result["stage"], "stage_status": result["stage_status"]}, indent=2) + "\n", encoding="utf-8")
    print(f"stage_status={result['stage_status']}")
    print(f"actual_llm_http_attempts={result['actual_llm_http_attempts']}")
    print("actual_ocr_http_attempts=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
