#!/usr/bin/env python
"""Run direct E0/E1 baselines for the frozen PlotQA 500 roster.

Safe default: write a dry-run plan with ``paid_calls_executed=0``. The runner
supports checkpoint/resume and ``--max-new-jobs`` for bounded execution.
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
from scripts.prepare_plotqa_public_benchmark_pilot import SCORER_VERSION, e0_prompt, e1_prompt, prompt_hashes  # noqa: E402
from scripts.run_public_benchmark_pilot import AttemptCounter, extract_prediction, score_prediction  # noqa: E402


CONDITIONS = ("E0_plotqa_direct", "E1_plotqa_cot")
SAMPLE_COUNT = 500
LLM_ATTEMPT_CAP = 1000
OCR_ATTEMPT_CAP = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_plan(plotqa_plan_path: Path) -> dict[str, Any]:
    source = load_json(plotqa_plan_path)
    samples = source.get("selected_samples") or []
    stop_reason = None
    if source.get("benchmark") != "PlotQA":
        stop_reason = "source_plan_not_plotqa"
    elif len(samples) != SAMPLE_COUNT:
        stop_reason = "source_plan_not_500_samples"
    frozen = source.get("frozen_before_paid_run") or {}
    plan = {
        "schema_version": "plotqa_direct_public_500_baseline_plan_v1",
        "created_at_utc": utc_now(),
        "stage": "track_c_plotqa_500_direct_baselines",
        "paid_run_approved": stop_reason is None,
        "paid_calls_executed": 0,
        "hypothesis": "Matched E0/E1 direct prompting on the same frozen PlotQA 500 roster provides the public baseline for measured-data GeoSym confirmation.",
        "falsification_signal": "If measured-data GeoSym does not exceed the strongest E0/E1 direct baseline, do not claim public benchmark benefit.",
        "selected_ids": [sample["sample_id"] for sample in samples],
        "selected_samples": samples,
        "conditions": list(CONDITIONS),
        "source_artifacts": {
            "plotqa_500_plan": {"path": str(plotqa_plan_path.resolve()), "sha256": stable_file_hash(plotqa_plan_path)}
        },
        "model_provenance": frozen.get("model_provenance", {}),
        "reasoning_model": frozen.get("reasoning_model"),
        "temperature": 0.0,
        "prompt_hashes": {
            key: value
            for key, value in prompt_hashes().items()
            if key in {"shared_final_answer_instruction", "e0_prompt_template", "e1_prompt_template"}
        },
        "scorer_version": frozen.get("scorer_version", SCORER_VERSION),
        "sample_condition_job_cap": LLM_ATTEMPT_CAP,
        "llm_http_attempt_cap": LLM_ATTEMPT_CAP,
        "ocr_http_attempt_cap": OCR_ATTEMPT_CAP,
        "retry_policy": "one counted HTTP attempt per sample-condition; no adaptive retries or sample replacement",
        "cache_policy": "checkpoint key is sample_id plus condition; full cache key must include plan hash, model ID, prompt hash, sample ID, and scorer version",
        "stop_rule": "stop after the frozen 500 samples and at most 1000 E0/E1 LLM calls",
        "cost_estimate": "maximum 1000 LLM HTTP attempts; OCR HTTP attempts fixed at 0",
        "dry_run": {
            "paid_calls_executed": 0,
            "cap_enforced": LLM_ATTEMPT_CAP == SAMPLE_COUNT * len(CONDITIONS),
            "stopped_before_paid_calls": stop_reason is not None,
            "stop_reason": stop_reason,
        },
        "interpretation_limits": [
            "This is a frozen 500-sample direct baseline, not a full PlotQA leaderboard run.",
            "It provides matched E0/E1 comparison rows for Track C measured-data confirmation.",
            "G0 evidence trace is not included in this direct-only baseline cap.",
        ],
    }
    plan["plan_sha256"] = sha256_hex_bytes(canonical_json_bytes({k: v for k, v in plan.items() if k != "plan_sha256"}))
    return plan


def review_plan(plan: dict[str, Any]) -> dict[str, Any]:
    findings = []
    if len(plan.get("selected_ids") or []) != SAMPLE_COUNT:
        findings.append("selected_roster_not_500")
    if set(plan.get("conditions") or []) != set(CONDITIONS):
        findings.append("direct_conditions_not_frozen")
    if plan.get("llm_http_attempt_cap") != LLM_ATTEMPT_CAP:
        findings.append("llm_cap_not_1000")
    if plan.get("ocr_http_attempt_cap") != 0:
        findings.append("ocr_cap_not_zero")
    if not plan.get("model_provenance", {}).get("display_name"):
        findings.append("model_display_name_missing")
    return {
        "schema_version": "plotqa_direct_public_500_baseline_prepaid_review_v1",
        "created_at_utc": utc_now(),
        "status": "no_blocking_findings" if not findings else "blocking_findings_present",
        "blocking_findings": findings,
        "paid_calls_executed": 0,
    }


def run_paid(plan: dict[str, Any], output_path: Path, max_new_jobs: int | None = None) -> dict[str, Any]:
    if plan.get("paid_run_approved") is not True:
        raise ValueError(plan.get("dry_run", {}).get("stop_reason") or "plan_not_approved")
    display_name = plan.get("model_provenance", {}).get("display_name")
    key = _get_saved_key(display_name)
    adapter = _create_llm_adapter(display_name, key) if key else None
    if adapter is None:
        raise ValueError(f"could not create adapter for {display_name}")
    results = []
    completed_keys = set()
    if output_path.exists():
        checkpoint = load_json(output_path)
        results = list(checkpoint.get("results") or [])
        completed_keys = {(row.get("sample_id"), row.get("condition")) for row in results}
    counter = AttemptCounter(plan["llm_http_attempt_cap"])
    counter.llm_attempts = sum(int(row.get("llm_http_attempts") or 0) for row in results)
    new_jobs = 0
    original_retries = adapter_base._MAX_API_RETRIES
    adapter_base._MAX_API_RETRIES = 1
    try:
        for sample in plan["selected_samples"]:
            image_path = Path(sample["image_path"])
            for condition in CONDITIONS:
                if (sample["sample_id"], condition) in completed_keys:
                    continue
                if max_new_jobs is not None and new_jobs >= max_new_jobs:
                    return write_result(plan, output_path, results, counter.llm_attempts, "partial_paid_run")
                prompt = e0_prompt(sample["question"]) if condition == "E0_plotqa_direct" else e1_prompt(sample["question"])
                print(f"starting_sample={sample['sample_id']} condition={condition}", flush=True)
                counter.consume_llm(sample["sample_id"], condition, "answer")
                raw_text = ""
                prediction = None
                error = ""
                try:
                    raw_text = adapter.call_with_prompt(image_path, prompt)
                    prediction = extract_prediction(raw_text, sample["answer_type"])
                    score = score_prediction(prediction, sample["normalized_answer"], sample["answer_type"])
                    parseable = True
                except Exception as exc:
                    score = {"correct": False, "numeric_error": None, "tolerance": None}
                    parseable = False
                    error = f"{type(exc).__name__}: {exc}"
                results.append(
                    {
                        "sample_id": sample["sample_id"],
                        "condition": condition,
                        "answer_type": sample["answer_type"],
                        "reference": sample["normalized_answer"],
                        "completed": parseable,
                        "parseable": parseable,
                        "normalized_prediction": prediction,
                        "raw_text": raw_text,
                        "error": error,
                        "llm_http_attempts": 1,
                        "ocr_http_attempts": 0,
                        **score,
                    }
                )
                new_jobs += 1
                print(f"completed_sample={sample['sample_id']} condition={condition} llm_attempts={counter.llm_attempts}/{counter.cap}", flush=True)
                write_result(plan, output_path, results, counter.llm_attempts, "partial_paid_run")
    finally:
        adapter_base._MAX_API_RETRIES = original_retries
    return write_result(plan, output_path, results, counter.llm_attempts, "completed_paid_run")


def write_result(plan: dict[str, Any], output_path: Path, results: list[dict[str, Any]], attempts: int, status: str) -> dict[str, Any]:
    summary = {}
    for condition in CONDITIONS:
        rows = [row for row in results if row.get("condition") == condition]
        summary[condition] = {
            "completed": sum(1 for row in rows if row.get("completed")),
            "parseable": sum(1 for row in rows if row.get("parseable")),
            "correct": sum(1 for row in rows if row.get("correct")),
        }
    result = {
        "schema_version": "plotqa_direct_public_500_baseline_result_v1",
        "created_at_utc": utc_now(),
        "stage": plan["stage"],
        "stage_status": status,
        "plan_sha256": plan["plan_sha256"],
        "sample_count": SAMPLE_COUNT,
        "conditions": list(CONDITIONS),
        "actual_llm_http_attempts": attempts,
        "actual_ocr_http_attempts": 0,
        "paid_calls_executed": attempts,
        "summary": summary,
        "results": results,
        "interpretation_limits": plan["interpretation_limits"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plotqa-plan", default="outputs/paper_level_evidence_package/plotqa_public_500/plotqa_500_zero_cost_plan.json")
    parser.add_argument("--output-dir", default="outputs/paper_level_evidence_package/plotqa_public_500/direct_baselines")
    parser.add_argument("--run-paid", action="store_true")
    parser.add_argument("--max-new-jobs", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(Path(args.plotqa_plan))
    plan_path = output_dir / "frozen_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    review = review_plan(plan)
    (output_dir / "prepaid_review.json").write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote frozen PlotQA direct baseline plan: {plan_path}")
    print(f"llm_http_attempt_cap={plan['llm_http_attempt_cap']}")
    print("ocr_http_attempt_cap=0")
    if not args.run_paid:
        (output_dir / "dry_run.json").write_text(json.dumps(plan["dry_run"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"prepaid_review_status={review['status']}")
        print("dry-run complete; paid_calls_executed=0")
        return 0 if plan["paid_run_approved"] and review["status"] == "no_blocking_findings" else 2
    if review["status"] != "no_blocking_findings":
        raise SystemExit("prepaid review has blocking findings")
    result = run_paid(plan, output_dir / "paid_result.json", max_new_jobs=args.max_new_jobs)
    print(f"stage_status={result['stage_status']}")
    print(f"actual_llm_http_attempts={result['actual_llm_http_attempts']}")
    print("actual_ocr_http_attempts=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
