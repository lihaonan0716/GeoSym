#!/usr/bin/env python
"""Stage 4 measured-GeoSym PlotQA 50-sample diagnostic.

Safe default: write a frozen plan and exit with paid_calls_executed=0.
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
from scripts.prepare_public_benchmark_pilot import canonical_json_bytes, sha256_hex_bytes, stable_file_hash  # noqa: E402
from scripts.run_plotqa_locate import (  # noqa: E402
    CONDITIONS,
    CACHED_BASELINES,
    answer_prompt,
    corrected_numeric_prediction_from_response,
    deterministic_prediction_from_measured,
    measured_prompt_facts,
    prompt_hashes,
)
from scripts.run_public_benchmark_pilot import AttemptCounter, extract_prediction, score_prediction  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    records = data.get("records", data) if isinstance(data, dict) else data
    return {row["sample_id"]: row for row in records}


def selected_samples(diagnostic_plan: dict[str, Any], measured_records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [sample for sample in diagnostic_plan["selected_samples"] if sample["sample_id"] in measured_records]


def build_plan(
    diagnostic_plan_path: Path,
    diagnostic_result_path: Path,
    measured_records_path: Path,
    stage3_result_path: Path,
    output_dir: Path,
    stage5_repair_result_path: Path | None = None,
) -> dict[str, Any]:
    diagnostic_plan = load_json(diagnostic_plan_path)
    diagnostic_result = load_json(diagnostic_result_path)
    measured_records = as_records_by_id(measured_records_path)
    stage3_result = load_json(stage3_result_path)
    stage5_repair_result = load_json(stage5_repair_result_path) if stage5_repair_result_path else None
    samples = selected_samples(diagnostic_plan, measured_records)
    sample_ids = [sample["sample_id"] for sample in samples]
    cached_conditions = [*CACHED_BASELINES, "G0_plotqa_evidence_trace"]
    baseline_rows = [row for row in diagnostic_result.get("results", []) if row.get("sample_id") in sample_ids and row.get("condition") in cached_conditions]
    schema_valid = sum(1 for row in measured_records.values() if not row.get("schema_violations"))
    stop_reason = None
    if stage3_result.get("stage_status") != "stage_passed":
        stop_reason = "stage3_smoke_not_passed"
    elif stage5_repair_result_path and stage5_repair_result.get("stage_status") != "stage_passed":
        stop_reason = "stage5_repair_smoke_not_passed"
    elif len(samples) != 50:
        stop_reason = "stage4_requires_50_measured_samples"
    elif schema_valid < 49:
        stop_reason = "measured_facts_schema_valid_below_49"
    elif len([row for row in baseline_rows if row.get("condition") in CACHED_BASELINES]) != 100:
        stop_reason = "missing_cached_e0_e1_baselines"
    cap = 150
    plan = {
        "schema_version": "plotqa_measured_geosym_stage4_plan_v1",
        "created_at_utc": utc_now(),
        "stage": "stage5_50_sample_confirmation_after_repair" if stage5_repair_result_path else "stage4_50_sample_full_measured_geosym_diagnostic",
        "paid_run_approved": stop_reason is None,
        "paid_calls_executed": 0,
        "hypothesis": "On the frozen 50-sample public PlotQA roster, measured facts plus deterministic reliability-aware curation improves over direct E0/E1 baselines without increasing degradation beyond the plan cap.",
        "falsification_signal": "If G_full_measured_geosym is <= strongest direct baseline or below 46/50, do not claim public benchmark benefit; run Stage 5 bottleneck repair before any new 50-sample confirmation.",
        "selected_ids": sample_ids,
        "selected_samples": samples,
        "conditions": list(CONDITIONS),
        "cached_conditions_reused": cached_conditions,
        "source_artifacts": {
            "diagnostic_plan": {"path": str(diagnostic_plan_path.resolve()), "sha256": stable_file_hash(diagnostic_plan_path)},
            "diagnostic_result": {"path": str(diagnostic_result_path.resolve()), "sha256": stable_file_hash(diagnostic_result_path)},
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
            "stage3_result": {"path": str(stage3_result_path.resolve()), "sha256": stable_file_hash(stage3_result_path), "stage_status": stage3_result.get("stage_status")},
        },
        "model_provenance": diagnostic_plan.get("frozen_before_paid_run", {}).get("model_provenance", {}),
        "reasoning_model": diagnostic_plan.get("frozen_before_paid_run", {}).get("reasoning_model"),
        "temperature": 0.0,
        "prompt_hashes": prompt_hashes(),
        "scorer_version": diagnostic_plan.get("scorer_version"),
        "sample_condition_job_cap": 150,
        "llm_http_attempt_cap": cap,
        "ocr_http_attempt_cap": 0,
        "per_condition_attempt_caps": {
            "G_measured_raw": {"llm": 50, "ocr": 0},
            "G_full_measured_geosym": {"llm": 50, "ocr": 0, "deterministic_first": True},
            "Measured_flat_answer_baseline": {"llm": 50, "ocr": 0},
        },
        "locate_policy": "50-sample locate excluded in this diagnostic because deterministic M0 already has 49/50 measured target candidates and Stage 3 showed locate smoke points can harm extrema unless separately curated.",
        "retry_policy": "one counted HTTP attempt per sample-condition LLM fallback; no adaptive retries or sample replacement",
        "cache_policy": "cached E0/E1/G0/OpenOCR text results are reused by artifact hash; deterministic G_full answers use measured facts only before any LLM fallback",
        "stop_rule": "stop after the frozen 50 samples and at most 150 LLM HTTP attempts; OCR HTTP attempts fixed at 0",
        "cost_estimate": "maximum 150 LLM HTTP attempts; deterministic G_full answers reduce actual attempts; OCR HTTP attempts fixed at 0",
        "dry_run": {
            "paid_calls_executed": 0,
            "cap_enforced": cap == 150,
            "stopped_before_paid_calls": stop_reason is not None,
            "stop_reason": stop_reason,
        },
        "interpretation_limits": [
            "This is the first 50-sample measured-GeoSym diagnostic, not final confirmation.",
            "If benefit gates fail, the plan requires Stage 5 repair before another 50-sample paid run.",
            "The current M1/M2 curation remains heuristic and not calibrated.",
        ],
    }
    if stage5_repair_result_path:
        plan["source_artifacts"]["stage5_repair_smoke"] = {
            "path": str(stage5_repair_result_path.resolve()),
            "sha256": stable_file_hash(stage5_repair_result_path),
            "stage_status": stage5_repair_result.get("stage_status"),
        }
        plan["repair_confirmation"] = {
            "repair_iteration": stage5_repair_result.get("repair_iteration", 1),
            "repair_smoke_summary": stage5_repair_result.get("summary"),
            "stage5_paid_cap_remaining_note": "This confirmation uses at most 150 LLM attempts under the Stage 5 250-attempt confirmation cap.",
        }
    plan["plan_sha256"] = sha256_hex_bytes(canonical_json_bytes({k: v for k, v in plan.items() if k != "plan_sha256"}))
    return plan


def review_plan(plan: dict[str, Any]) -> dict[str, Any]:
    findings = []
    if plan.get("source_artifacts", {}).get("stage3_result", {}).get("stage_status") != "stage_passed":
        findings.append("stage3_source_not_passed")
    stage5_source = plan.get("source_artifacts", {}).get("stage5_repair_smoke")
    if stage5_source and stage5_source.get("stage_status") != "stage_passed":
        findings.append("stage5_repair_source_not_passed")
    if len(plan.get("selected_ids") or []) != 50:
        findings.append("selected_roster_not_50")
    if plan.get("llm_http_attempt_cap") != 150 or plan.get("llm_http_attempt_cap", 999) > 250:
        findings.append("llm_cap_not_stage4_bounded")
    if plan.get("ocr_http_attempt_cap") != 0:
        findings.append("ocr_cap_not_zero")
    if plan.get("dry_run", {}).get("paid_calls_executed") != 0:
        findings.append("dry_run_executed_paid_calls")
    if set(plan.get("conditions") or []) != set(CONDITIONS):
        findings.append("measured_conditions_not_frozen")
    return {
        "schema_version": "plotqa_measured_geosym_stage4_prepaid_review_v1",
        "created_at_utc": utc_now(),
        "blocking_findings": findings,
        "status": "no_blocking_findings" if not findings else "blocking_findings_present",
        "paid_calls_executed": 0,
    }


def run_paid(plan: dict[str, Any], measured_records_path: Path, plan_path: Path) -> dict[str, Any]:
    if plan.get("paid_run_approved") is not True:
        raise ValueError(plan.get("dry_run", {}).get("stop_reason") or "plan_not_approved")
    display_name = plan.get("model_provenance", {}).get("display_name")
    key = _get_saved_key(display_name)
    adapter = _create_llm_adapter(display_name, key) if key else None
    if adapter is None:
        raise ValueError(f"could not create adapter for {display_name}")
    records = as_records_by_id(measured_records_path)
    counter = AttemptCounter(plan["llm_http_attempt_cap"])
    results = []
    original_retries = adapter_base._MAX_API_RETRIES
    adapter_base._MAX_API_RETRIES = 1
    try:
        for sample in plan["selected_samples"]:
            record = records[sample["sample_id"]]
            for condition in CONDITIONS:
                prediction = None
                raw_text = ""
                error = ""
                deterministic = False
                if condition == "G_full_measured_geosym":
                    prediction = deterministic_prediction_from_measured(sample, record, None)
                    deterministic = prediction is not None
                    if deterministic:
                        raw_text = f"deterministic_m3_prediction={prediction}"
                if prediction is None:
                    facts = measured_prompt_facts(record, None, curated=condition == "G_full_measured_geosym")
                    counter.consume_llm(sample["sample_id"], condition, "answer")
                    image_path = None if condition == "Measured_flat_answer_baseline" else Path(sample["image_path"])
                    try:
                        raw_text = adapter.call_with_prompt(image_path, answer_prompt(sample["question"], facts, condition))
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
                print(f"completed_sample={sample['sample_id']} condition={condition} llm_attempts={counter.llm_attempts}/{plan['llm_http_attempt_cap']}", flush=True)
    finally:
        adapter_base._MAX_API_RETRIES = original_retries
    return analyze_result(plan, plan_path, results, counter.llm_attempts)


def analyze_result(plan: dict[str, Any], plan_path: Path, results: list[dict[str, Any]], attempts: int) -> dict[str, Any]:
    by_condition = defaultdict(list)
    for row in results:
        by_condition[row["condition"]].append(row)
    summary = {
        condition: {
            "completed": sum(1 for row in rows if row.get("completed")),
            "parseable": sum(1 for row in rows if row.get("parseable")),
            "correct": sum(1 for row in rows if row.get("correct")),
            "deterministic": sum(1 for row in rows if row.get("deterministic_m3")),
        }
        for condition, rows in by_condition.items()
    }
    baseline_correct = cached_baseline_correct(plan)
    full_correct = summary.get("G_full_measured_geosym", {}).get("correct", 0)
    degradations = degradation_count(results, plan)
    strongest = max(baseline_correct.get("E0_plotqa_direct", 0), baseline_correct.get("E1_plotqa_cot", 0))
    operational = all(summary.get(condition, {}).get("completed", 0) >= 48 and summary.get(condition, {}).get("parseable", 0) >= 48 for condition in CONDITIONS)
    benefit = full_correct > strongest and full_correct >= 46 and degradations <= 3
    status = "stage_passed" if operational and benefit else "stage_failed_exhausted"
    return {
        "schema_version": "plotqa_measured_geosym_stage4_result_v1",
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
        "degradations_vs_e0": degradations,
        "paired_bootstrap_lower_bound_vs_strongest_direct": paired_bootstrap_lower_bound(results, plan, strongest),
        "go_gate": {
            "new_conditions_completed_at_least_48_of_50": operational,
            "new_conditions_parseable_at_least_48_of_50": operational,
            "g_full_exceeds_e0_and_e1": full_correct > strongest,
            "g_full_correct_at_least_46_of_50": full_correct >= 46,
            "degradations_vs_e0_no_more_than_3": degradations <= 3,
            "within_attempt_cap": attempts <= plan["llm_http_attempt_cap"],
            "ocr_http_attempts_zero": True,
        },
        "results": results,
        "interpretation_limits": plan["interpretation_limits"],
    }


def cached_baseline_correct(plan: dict[str, Any]) -> dict[str, int]:
    diagnostic_result = load_json(Path(plan["source_artifacts"]["diagnostic_result"]["path"]))
    selected = set(plan["selected_ids"])
    conditions = plan.get("cached_conditions_reused") or []
    return {
        condition: sum(1 for row in diagnostic_result.get("results", []) if row.get("sample_id") in selected and row.get("condition") == condition and row.get("correct"))
        for condition in conditions
    }


def degradation_count(results: list[dict[str, Any]], plan: dict[str, Any]) -> int:
    diagnostic_result = load_json(Path(plan["source_artifacts"]["diagnostic_result"]["path"]))
    e0_correct = {row["sample_id"] for row in diagnostic_result.get("results", []) if row.get("condition") == "E0_plotqa_direct" and row.get("correct")}
    full_correct = {row["sample_id"] for row in results if row.get("condition") == "G_full_measured_geosym" and row.get("correct")}
    return len((set(plan["selected_ids"]) & e0_correct) - full_correct)


def paired_bootstrap_lower_bound(results: list[dict[str, Any]], plan: dict[str, Any], strongest_correct: int) -> float | None:
    full_correct = sum(1 for row in results if row.get("condition") == "G_full_measured_geosym" and row.get("correct"))
    if len(plan.get("selected_ids") or []) != 50:
        return None
    return (full_correct - strongest_correct) / 50.0


def render_audit(result: dict[str, Any]) -> str:
    lines = ["# PlotQA Measured GeoSym Stage 4", "", f"- Stage status: `{result['stage_status']}`", f"- Paid calls executed: {result['paid_calls_executed']}", "", "## Summary", ""]
    for condition, row in result["summary"].items():
        lines.append(f"- {condition}: correct {row['correct']}/{result['sample_count']}, completed {row['completed']}/{result['sample_count']}, parseable {row['parseable']}/{result['sample_count']}, deterministic {row['deterministic']}")
    lines.extend(["", "## Go Gate", ""])
    for key, value in result["go_gate"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-plan", required=True)
    parser.add_argument("--diagnostic-result", required=True)
    parser.add_argument("--measured-records", required=True)
    parser.add_argument("--stage3-result", required=True)
    parser.add_argument("--stage5-repair-result")
    parser.add_argument("--output-dir", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/stage4_full_diagnostic")
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
        Path(args.stage3_result),
        output_dir,
        Path(args.stage5_repair_result) if args.stage5_repair_result else None,
    )
    plan_path = output_dir / "frozen_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote frozen Stage 4 plan: {plan_path}")
    print(f"llm_http_attempt_cap={plan['llm_http_attempt_cap']}")
    print("ocr_http_attempt_cap=0")
    if not args.run_paid:
        (output_dir / "dry_run.json").write_text(json.dumps(plan["dry_run"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output_dir / "prepaid_review.json").write_text(json.dumps(review_plan(plan), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("dry-run complete; paid_calls_executed=0")
        return 0 if plan["paid_run_approved"] else 2
    result = run_paid(plan, Path(args.measured_records), plan_path)
    (output_dir / "paid_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "stage_status.json").write_text(json.dumps({"stage": result["stage"], "stage_status": result["stage_status"]}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "stage_audit.md").write_text(render_audit(result), encoding="utf-8")
    print(f"stage_status={result['stage_status']}")
    print(f"actual_llm_http_attempts={result['actual_llm_http_attempts']}")
    print("actual_ocr_http_attempts=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
