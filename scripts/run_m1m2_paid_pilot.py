#!/usr/bin/env python
"""Run or dry-run the bounded M1/M2 recovery paid pilot.

Safe default: dry-run only. Use --execute-paid to spend the frozen cap.
"""

from __future__ import annotations

import argparse
import json
import random
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
from scripts.prepare_public_benchmark_pilot import stable_file_hash, stable_payload_hash  # noqa: E402
from scripts.prepare_plotqa_public_benchmark_pilot import SHARED_FINAL_ANSWER_INSTRUCTION  # noqa: E402
from scripts.run_public_benchmark_pilot import AttemptCounter, extract_prediction, score_prediction  # noqa: E402

CONDITIONS = ("G_full_guarded", "G_all_evidence", "G_top_confidence", "G_random_k")
SCHEMA_VERSION = "m1m2_recovery_paid_pilot_result_v1"
DEFAULT_MODEL_DISPLAY_NAME = "GPT-5.5"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["sample_id"]): record for record in records}


def schema_by_id(schema_outputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in schema_outputs.get("outputs", [])}


def sorted_by_confidence(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    return sorted(candidates, key=lambda row: (rank.get(str(row.get("reliability_tier")), 3), str(row.get("fact_id"))))


def random_k(candidates: list[dict[str, Any]], k: int, seed: int, sample_id: str) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}:{sample_id}")
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return sorted(shuffled[:k], key=lambda row: str(row.get("fact_id")))


def facts_for_condition(schema_row: dict[str, Any], condition: str, seed: int) -> dict[str, Any]:
    candidates = list(schema_row.get("evidence_candidates") or [])
    gaps = list(schema_row.get("evidence_gaps") or [])
    selected = schema_row.get("selected_evidence_set") or {}
    selected_ids = set(selected.get("selected_fact_ids") or [])
    budget = max(1, len(selected_ids)) if candidates else 0
    if condition == "G_full_guarded":
        facts = [row for row in candidates if row.get("fact_id") in selected_ids]
        surfaced_gap_ids = set(selected.get("surfaced_gap_ids") or [])
        visible_gaps = [row for row in gaps if row.get("gap_id") in surfaced_gap_ids]
        policy = selected.get("propagation_policy")
    elif condition == "G_all_evidence":
        facts = candidates
        visible_gaps = []
        policy = "all_evidence_no_curation"
    elif condition == "G_top_confidence":
        facts = sorted_by_confidence(candidates)[:budget]
        visible_gaps = []
        policy = "top_confidence_same_budget"
    elif condition == "G_random_k":
        facts = random_k(candidates, budget, seed, str(schema_row["sample_id"]))
        visible_gaps = []
        policy = "random_k_same_budget"
    else:
        raise ValueError(f"unsupported condition: {condition}")
    return {
        "schema_version": "m1m2_recovery_prompt_facts_v1",
        "condition": condition,
        "curation_policy": policy,
        "measured_facts": facts,
        "evidence_gaps": visible_gaps,
        "selected_fact_count": len(facts),
        "source_schema_condition": schema_row.get("condition"),
    }


def answer_prompt(question: str, facts: dict[str, Any], condition: str) -> str:
    if condition == "G_full_guarded":
        instruction = (
            "Use the selected measured facts, reliability tiers, uncertainty fields, and surfaced evidence gaps. "
            "Do not treat missing evidence as a negative fact. If the selected evidence is insufficient, say so in raw_text "
            "and still return the best normalized_prediction only when supported."
        )
    elif condition == "G_all_evidence":
        instruction = "Use all supplied measured facts without curation. Do not use reference answers or PlotQA templates."
    elif condition == "G_top_confidence":
        instruction = "Use the supplied top-confidence measured facts only. Do not use reference answers or PlotQA templates."
    else:
        instruction = "Use the supplied deterministic random-k measured facts only. Do not use reference answers or PlotQA templates."
    return (
        f"{SHARED_FINAL_ANSWER_INSTRUCTION}\n"
        f"Condition: {condition}\n"
        f"Use the image and measured evidence. {instruction}\n"
        f"Question: {question}\n"
        f"Measured evidence JSON: {json.dumps(facts, ensure_ascii=False)}"
    )


def condition_summary(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_condition[str(row["condition"])].append(row)
    return {
        condition: {
            "completed": sum(1 for row in rows if row.get("completed")),
            "parseable": sum(1 for row in rows if row.get("parseable")),
            "correct": sum(1 for row in rows if row.get("correct")),
        }
        for condition, rows in sorted(by_condition.items())
    }


def review_manifest(manifest: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    if manifest.get("stage_status") != "paid_pilot_manifest_frozen":
        findings.append("manifest_not_frozen")
    if manifest.get("paid_calls_executed") != 0:
        findings.append("manifest_already_executed_paid_calls")
    if manifest.get("sample_level_job_cap") != 8:
        findings.append("sample_cap_not_8")
    if tuple(manifest.get("conditions") or []) != CONDITIONS:
        findings.append("conditions_not_frozen")
    if manifest.get("llm_http_attempt_cap") != 32:
        findings.append("llm_cap_not_32")
    if manifest.get("ocr_http_attempt_cap") != 0:
        findings.append("ocr_cap_not_zero")
    if len(manifest.get("selected_roster") or []) != 8:
        findings.append("roster_not_8")
    return findings


def build_dry_run(manifest_path: Path, measured_records_path: Path, schema_outputs_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    records = records_by_id(load_json(measured_records_path))
    schemas = schema_by_id(load_json(schema_outputs_path))
    findings = review_manifest(manifest)
    jobs = []
    for item in manifest.get("selected_roster") or []:
        sample_id = item["sample_id"]
        for condition in CONDITIONS:
            jobs.append(
                {
                    "sample_id": sample_id,
                    "condition": condition,
                    "has_measured_record": sample_id in records,
                    "has_schema_output": sample_id in schemas,
                    "would_execute_paid_call": True,
                }
            )
    if len(jobs) != manifest.get("llm_http_attempt_cap"):
        findings.append("job_count_does_not_match_llm_cap")
    if any(not job["has_measured_record"] or not job["has_schema_output"] for job in jobs):
        findings.append("job_missing_source_record")
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_recovery_paid_pilot_dry_run",
        "stage_status": "paid_pilot_dry_run_passed" if not findings else "paid_pilot_dry_run_failed_closed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "manifest": {"path": str(manifest_path.resolve()), "sha256": stable_file_hash(manifest_path), "stage_status": manifest.get("stage_status")},
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
            "schema_outputs": {"path": str(schema_outputs_path.resolve()), "sha256": stable_file_hash(schema_outputs_path)},
        },
        "blocking_findings": findings,
        "jobs": jobs,
        "next_required_action": "execute_bounded_paid_pilot" if not findings else "repair_paid_pilot_manifest_or_sources",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dry_run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def run_paid(manifest_path: Path, measured_records_path: Path, schema_outputs_path: Path, output_dir: Path, model_display_name: str) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    findings = review_manifest(manifest)
    if findings:
        raise ValueError(f"manifest failed review: {findings}")
    key = _get_saved_key(model_display_name)
    adapter = _create_llm_adapter(model_display_name, key) if key else None
    if adapter is None:
        raise ValueError(f"could not create adapter for {model_display_name}")
    records = records_by_id(load_json(measured_records_path))
    schemas = schema_by_id(load_json(schema_outputs_path))
    counter = AttemptCounter(int(manifest["llm_http_attempt_cap"]))
    results = []
    original_retries = adapter_base._MAX_API_RETRIES
    adapter_base._MAX_API_RETRIES = 1
    try:
        for item in manifest["selected_roster"]:
            sample_id = item["sample_id"]
            record = records[sample_id]
            schema_row = schemas[sample_id]
            for condition in CONDITIONS:
                counter.consume_llm(sample_id, condition, "answer")
                facts = facts_for_condition(schema_row, condition, int(manifest["random_seed"]))
                raw_text = ""
                prediction = None
                parseable = False
                error = ""
                try:
                    raw_text = adapter.call_with_prompt(Path(record["image_path"]), answer_prompt(record["question"], facts, condition))
                    prediction = extract_prediction(raw_text, record.get("answer_type"))
                    parseable = prediction is not None
                    score = score_prediction(prediction, record.get("reference"), record.get("answer_type"))
                except Exception as exc:
                    error = str(exc)
                    score = {"correct": False, "numeric_error": None, "tolerance": None}
                results.append(
                    {
                        "sample_id": sample_id,
                        "condition": condition,
                        "answer_type": record.get("answer_type"),
                        "reference": record.get("reference"),
                        "completed": prediction is not None,
                        "parseable": parseable,
                        "normalized_prediction": prediction,
                        "raw_text": raw_text,
                        "error": error,
                        "llm_http_attempts": 1,
                        "ocr_http_attempts": 0,
                        "prompt_hash": stable_payload_hash({"condition": condition, "prompt": answer_prompt(record["question"], facts, condition)}),
                        **score,
                    }
                )
                print(f"completed_sample={sample_id} condition={condition} llm_attempts={counter.llm_attempts}/{manifest['llm_http_attempt_cap']}", flush=True)
    finally:
        adapter_base._MAX_API_RETRIES = original_retries
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_recovery_paid_pilot",
        "stage_status": "paid_pilot_completed",
        "paid_calls_executed": counter.llm_attempts,
        "actual_llm_http_attempts": counter.llm_attempts,
        "actual_ocr_http_attempts": 0,
        "model_display_name": model_display_name,
        "source_artifacts": {
            "manifest": {"path": str(manifest_path.resolve()), "sha256": stable_file_hash(manifest_path)},
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
            "schema_outputs": {"path": str(schema_outputs_path.resolve()), "sha256": stable_file_hash(schema_outputs_path)},
        },
        "summary": condition_summary(results),
        "results": results,
        "interpretation_limits": [
            "This is an 8-sample bounded pilot and must not be expanded automatically.",
            "OCR HTTP attempts are fixed at 0.",
            "A positive result here is not a full 50-sample confirmation.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paid_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/m1m2_recovery/paid_pilot_manifest/frozen_manifest.json")
    parser.add_argument("--measured-records", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/m0_measurement_records.json")
    parser.add_argument("--schema-outputs", default="outputs/m1m2_recovery/schema_outputs/m1m2_schema_outputs.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_recovery/paid_pilot")
    parser.add_argument("--model-display-name", default=DEFAULT_MODEL_DISPLAY_NAME)
    parser.add_argument("--execute-paid", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest_path = Path(args.manifest)
    records_path = Path(args.measured_records)
    schema_path = Path(args.schema_outputs)
    output_dir = Path(args.output_dir)
    if not args.execute_paid:
        result = build_dry_run(manifest_path, records_path, schema_path, output_dir)
        print(f"wrote paid pilot dry run: {output_dir / 'dry_run.json'}")
        print(f"stage_status={result['stage_status']}")
        print("paid_calls_executed=0")
        return 0
    result = run_paid(manifest_path, records_path, schema_path, output_dir, args.model_display_name)
    print(f"wrote paid pilot result: {output_dir / 'paid_result.json'}")
    print(f"stage_status={result['stage_status']}")
    print(f"paid_calls_executed={result['paid_calls_executed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
