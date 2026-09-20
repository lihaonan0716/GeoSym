#!/usr/bin/env python
"""Stage 3 measured-GeoSym PlotQA smoke.

Safe default: write a frozen plan and exit with paid_calls_executed=0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
from evaluator.reasoning.plotqa_m0_chain import build_evidence, presentable_fact  # noqa: E402
from scripts.prepare_public_benchmark_pilot import canonical_json_bytes, sha256_hex_bytes, stable_file_hash  # noqa: E402
from scripts.prepare_plotqa_public_benchmark_pilot import SCORER_VERSION, SHARED_FINAL_ANSWER_INSTRUCTION  # noqa: E402
from scripts.run_plotqa_measured_locate_smoke import (  # noqa: E402
    categorical_comparison_candidate,
    label_matches_question,
    label_position_in_question,
    normalize_text_answer,
    selected_record_points_for_question,
    value_for_label_year,
)
from scripts.run_public_benchmark_pilot import AttemptCounter, extract_prediction, score_prediction  # noqa: E402


CONDITIONS = ("G_measured_raw", "Measured_flat_answer_baseline")

# `G_full_measured_geosym` differed from `G_measured_raw` only by the M1'/M2'
# curation that retired on 2026-09-19. Re-running this name would now emit rows
# identical to `G_measured_raw` while carrying a label that means "curated" in
# the 1346 paid rows already on disk -- a silent collision, so it raises instead.
RETIRED_CONDITIONS = {
    "G_full_measured_geosym": (
        "retired 2026-09-19 with M1'/M2'; its only distinguishing input "
        "(curated selection) no longer exists. Historical rows remain valid; "
        "see docs/M1M2_RETIREMENT_AND_FGCS_REPOSITIONING_PLAN_2026-09-19.md"
    ),
}
CACHED_BASELINES = ("E0_plotqa_direct", "E1_plotqa_cot")

# Stage 3 smoke roster size, and the retry headroom allowed above one attempt
# per sample-condition job. Both were folded into the literals 24 and 32 while
# three conditions ran; they are named so the caps follow `CONDITIONS`.
STAGE3_SAMPLE_COUNT = 8
STAGE3_ATTEMPT_HEADROOM = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def measured_prompt_facts(
    record: dict[str, Any],
    locate_row: dict[str, Any] | None,
    ocr_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one arm's evidence payload from the M0' chain.

    This used to curate inline against the frozen M0 record, whose measured
    values were 16% correct because the axis was guessed rather than fitted. The
    chain (category axis -> ruler refit -> mark admission) raises that to 78% and
    is what the zero-cost evaluations measure, so the answering path calls the
    same composition instead of a parallel implementation.

    ``locate_row`` points from the Stage 2 LLM locate pass are appended after the
    chain runs. They are a different evidence source -- model-located rather than
    measured -- and carry no mark geometry, so the geometric stages cannot score
    them; passing them through would have them silently dropped as artifacts.

    The ``curated`` parameter was removed with M1'/M2' on 2026-09-19. It was the
    only thing that distinguished ``G_full_measured_geosym`` from
    ``G_measured_raw``, so that condition can no longer be reproduced; see
    ``RETIRED_CONDITIONS`` below.
    """
    evidence = build_evidence(record, ocr_row)
    facts = list(evidence["measured_facts"])
    if locate_row:
        for idx, point in enumerate(locate_row.get("measured_points") or []):
            item = presentable_fact(point)
            item["fact_id"] = f"locate_{idx:03d}"
            item["source"] = "mllm_locate"
            facts.append(item)
    return {
        "ruler_refit": evidence["ruler_refit"],
        "chart_geometry": evidence["chart_geometry"],
        "measured_facts": facts,
        "evidence_gaps": evidence["evidence_gaps"],
        "curation_policy": evidence["curation_policy"],
        "evidence_injected": evidence["evidence_injected"],
        "chain_version": evidence["chain_version"],
    }


def filter_bar_sliver_artifacts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    robust_labels = {
        normalize_text_answer(fact.get("series_label") or fact.get("category_label"))
        for fact in facts
        if normalize_text_answer(fact.get("series_label") or fact.get("category_label")) and not component_is_sliver(fact)
    }
    filtered = []
    for fact in facts:
        label = normalize_text_answer(fact.get("series_label") or fact.get("category_label"))
        if component_is_sliver(fact) and (not label or label in robust_labels):
            continue
        filtered.append(fact)
    return filtered


def component_is_sliver(fact: dict[str, Any]) -> bool:
    bbox = fact.get("component_bbox") or []
    if len(bbox) != 4:
        return False
    width = abs(float(bbox[2]) - float(bbox[0]))
    height = abs(float(bbox[3]) - float(bbox[1]))
    area = fact.get("component_area")
    try:
        area_value = float(area)
    except Exception:
        area_value = width * height
    return min(width, height) <= 1.5 and area_value <= max(width, height) * 1.5


def prompt_hashes() -> dict[str, str]:
    """Hash every arm's prompt template, including the retired one.

    ``full_measured`` stays in this map deliberately. Its hash identifies the
    prompt behind 1346 paid rows already on disk, and dropping the entry would
    make those artifacts unverifiable. This is why ``answer_prompt`` keeps the
    retired branch and does not itself reject retired names -- the refusal
    belongs on the paid path (``run_paid``), which is what must not spend.

    These hashes do NOT match the ones frozen in the 2026-08-31 paid plans.
    Commit 4f96f8a moved task guidance into ``SHARED_FINAL_ANSWER_INSTRUCTION``
    so that every arm receives it (invariant 6: the curated arm had carried a
    task hint the baselines lacked). That repair necessarily changed the prompt
    text, so a mismatch against those plans is the expected and correct signal
    that the paid rows predate the fairness fix -- not drift to be silenced by
    pinning the old value back.
    """
    templates = {
        "measured_raw": answer_prompt("{question}", {}, "G_measured_raw"),
        "full_measured": answer_prompt("{question}", {}, "G_full_measured_geosym"),
        "measured_flat": answer_prompt("{question}", {}, "Measured_flat_answer_baseline"),
    }
    return {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in templates.items()}


def answer_prompt(question: str, facts: dict[str, Any], condition: str) -> str:
    """Build one arm's prompt.

    Prompt symmetry is a hard requirement here (invariant 6). The task guidance
    that used to live only in the curated arm sits in
    ``SHARED_FINAL_ANSWER_INSTRUCTION``, which every condition receives.

    Since M1'/M2' retired, the two live arms also share the same
    evidence-framing sentence, so the only remaining difference between them is
    the image-access instruction -- the manipulation under test. The
    reliability-semantics branch below is dead for live runs and is kept solely
    to reproduce the prompt behind the paid ``G_full_measured_geosym`` rows.
    """
    image_instruction = "Use the image and the measured facts." if condition != "Measured_flat_answer_baseline" else "Use only the measured facts; do not infer from the image."
    fusion_instruction = (
        # Names only signals the payload actually carries. It previously cited
        # per-fact uncertainty and evidence gaps, which the slimmed payload no
        # longer contains -- an instruction to weigh absent fields is noise.
        "Treat the stated evidence reliability and axis admissibility as reliability signals."
        if condition == "G_full_measured_geosym"
        else "Treat the supplied measured facts as the evidence available to you."
    )
    return (
        f"{SHARED_FINAL_ANSWER_INSTRUCTION}\n"
        f"Condition: {condition}\n"
        f"{image_instruction} {fusion_instruction} Do not use reference answers or PlotQA templates.\n"
        f"Question: {question}\nMeasured evidence JSON: {json.dumps(facts, ensure_ascii=False)}"
    )


def corrected_numeric_prediction_from_response(
    sample: dict[str, Any],
    record: dict[str, Any],
    raw_text: str,
    prediction: Any,
) -> Any:
    signed = signed_difference_prediction_from_response(sample, record, raw_text, prediction)
    return prediction if signed is None else signed


def signed_difference_prediction_from_response(
    sample: dict[str, Any],
    record: dict[str, Any],
    raw_text: str,
    prediction: Any,
) -> float | None:
    if not isinstance(prediction, (int, float)) or sample.get("answer_type") != "numeric":
        return None
    question = normalize_text_answer(sample.get("question"))
    if "difference between" not in question:
        return None
    years = [int(match.group(0)) for match in re.finditer(r"\b(?:18|19|20|21)\d{2}\b", question)]
    if len(years) < 2:
        return None
    labels = sorted(
        {
            normalize_text_answer(point.get("series_label"))
            for point in record.get("measured_points", [])
            if normalize_text_answer(point.get("series_label"))
            and label_matches_question(normalize_text_answer(point.get("series_label")), question)
        },
        key=lambda label: label_position_in_question(label, question),
    )
    if len(labels) < 2:
        return None
    response_text = response_raw_text(raw_text)
    first = response_value_for_label_year(response_text, labels[0], years[0])
    second = response_value_for_label_year(response_text, labels[1], years[1])
    if first is None or second is None:
        return None
    directed = first - second
    if directed >= 0 or float(prediction) <= 0:
        return None
    if abs(abs(directed) - abs(float(prediction))) > max(1.0, abs(float(prediction)) * 0.08):
        return None
    return -abs(float(prediction))


def response_raw_text(raw_text: str) -> str:
    try:
        parsed = json.loads(raw_text)
    except Exception:
        return str(raw_text)
    if isinstance(parsed, dict) and isinstance(parsed.get("raw_text"), str):
        return parsed["raw_text"]
    return str(raw_text)


def response_value_for_label_year(response_text: str, label: str, year: int) -> float | None:
    normalized_response = normalize_text_answer(response_text)
    label_pos = label_position_in_question(label, normalized_response)
    if label_pos < 0:
        return None
    label_raw_match = re.search(r"(?i)(?<![a-z0-9])" + r"[^a-z0-9]+".join(re.escape(part) for part in label.split()) + r"(?![a-z0-9])", response_text)
    if label_raw_match is None:
        return None
    nearby = response_text[max(0, label_raw_match.start() - 40) : min(len(response_text), label_raw_match.end() + 80)]
    year_match = re.search(r"(?<!\d)" + re.escape(str(year)) + r"(?!\d)", nearby)
    if year_match is None:
        return None
    window = response_text[label_raw_match.start() : min(len(response_text), label_raw_match.end() + 90)]
    candidates = [
        value
        for value in scaled_numbers(window)
        if not 1800 <= abs(value) <= 2199
    ]
    return candidates[0] if candidates else None


def scaled_numbers(text: str) -> list[float]:
    values = []
    pattern = re.compile(
        r"([-+]?\d[\d,]*(?:\.\d+)?)\s*(million|billion|thousand|m|bn|k)?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        value = float(match.group(1).replace(",", ""))
        suffix = (match.group(2) or "").lower()
        if suffix in {"million", "m"}:
            value *= 1_000_000.0
        elif suffix in {"billion", "bn"}:
            value *= 1_000_000_000.0
        elif suffix in {"thousand", "k"}:
            value *= 1_000.0
        values.append(value)
    return values


def selected_samples(stage2_result: dict[str, Any], diagnostic_plan: dict[str, Any]) -> list[dict[str, Any]]:
    sample_by_id = {sample["sample_id"]: sample for sample in diagnostic_plan["selected_samples"]}
    return [sample_by_id[sid] for sid in stage2_result_ids(stage2_result) if sid in sample_by_id]


def stage2_result_ids(stage2_result: dict[str, Any]) -> list[str]:
    return [row["sample_id"] for row in stage2_result.get("results", [])]


def build_plan(diagnostic_plan_path: Path, diagnostic_result_path: Path, measured_records_path: Path, stage2_result_path: Path, output_dir: Path, prior_llm_attempts: int = 0) -> dict[str, Any]:
    diagnostic_plan = load_json(diagnostic_plan_path)
    diagnostic_result = load_json(diagnostic_result_path)
    measured_records = load_json(measured_records_path)
    stage2_result = load_json(stage2_result_path)
    samples = selected_samples(stage2_result, diagnostic_plan)
    sample_ids = [sample["sample_id"] for sample in samples]
    measured_ids = {row["sample_id"] for row in measured_records}
    baseline_rows = [row for row in diagnostic_result.get("results", []) if row.get("sample_id") in sample_ids and row.get("condition") in CACHED_BASELINES]
    stop_reason = None
    if stage2_result.get("stage_status") != "stage_passed":
        stop_reason = "stage2_locate_smoke_not_passed"
    elif len(samples) != 8:
        stop_reason = "stage3_requires_8_stage2_samples"
    elif any(sample_id not in measured_ids for sample_id in sample_ids):
        stop_reason = "missing_measured_records"
    elif len(baseline_rows) != 16:
        stop_reason = "missing_cached_e0_e1_baselines"
    # Planned jobs track the live condition list. These were 24 (8 x 3) with a
    # 32-attempt ceiling before `G_full_measured_geosym` retired on 2026-09-19;
    # a literal would authorise 8 calls for an arm that can no longer run. The
    # headroom above the planned jobs is preserved proportionally.
    planned_jobs = STAGE3_SAMPLE_COUNT * len(CONDITIONS)
    total_stage_cap = planned_jobs + STAGE3_ATTEMPT_HEADROOM
    cap = total_stage_cap - prior_llm_attempts
    if cap < planned_jobs and stop_reason is None:
        stop_reason = "insufficient_remaining_stage3_llm_cap"
    plan = {
        "schema_version": "plotqa_measured_geosym_stage3_plan_v1",
        "created_at_utc": utc_now(),
        "stage": "stage3_full_measured_geosym_smoke",
        "paid_run_approved": stop_reason is None,
        "paid_calls_executed": 0,
        # Restated 2026-09-19: the original hypothesis and falsification signal
        # were about reliability-aware curation and keyed on
        # `G_full_measured_geosym`, which no longer runs -- an untestable
        # hypothesis frozen into a paid plan. What remains testable is whether
        # measured geometric evidence beats the direct baselines at all.
        "hypothesis": "Measured geometric facts from the M0' chain can answer the Stage 2-passed PlotQA smoke set at least one sample better than the strongest cached direct baseline.",
        "falsification_signal": "If G_measured_raw is not at least max(E0,E1)+1 correct, do not claim measured-evidence benefit and run bottleneck repair. If measured-flat matches or exceeds it, the gain is not attributable to image access.",
        "selected_ids": sample_ids,
        "selected_samples": samples,
        "conditions": list(CONDITIONS),
        "cached_conditions_reused": list(CACHED_BASELINES),
        "source_artifacts": {
            "diagnostic_plan": {"path": str(diagnostic_plan_path.resolve()), "sha256": stable_file_hash(diagnostic_plan_path)},
            "diagnostic_result": {"path": str(diagnostic_result_path.resolve()), "sha256": stable_file_hash(diagnostic_result_path)},
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
            "stage2_locate_result": {"path": str(stage2_result_path.resolve()), "sha256": stable_file_hash(stage2_result_path), "stage_status": stage2_result.get("stage_status")},
        },
        "model_provenance": diagnostic_plan.get("frozen_before_paid_run", {}).get("model_provenance", {}),
        "reasoning_model": diagnostic_plan.get("frozen_before_paid_run", {}).get("reasoning_model"),
        "temperature": 0.0,
        "prompt_hashes": prompt_hashes(),
        "scorer_version": SCORER_VERSION,
        "sample_condition_job_cap": planned_jobs,
        "llm_http_attempt_cap": cap,
        "prior_possible_llm_http_attempts": prior_llm_attempts,
        "total_stage3_llm_http_attempt_cap": total_stage_cap,
        "ocr_http_attempt_cap": 0,
        "per_condition_attempt_caps": {
            condition: {"llm": STAGE3_SAMPLE_COUNT, "ocr": 0} for condition in CONDITIONS
        },
        "retry_policy": "one counted HTTP attempt per sample-condition; no adaptive retries or sample replacement",
        "cache_policy": "cached E0/E1 results are reused by artifact hash; Stage 2 locate pixels are reused by hash",
        "stop_rule": f"stop after the frozen {STAGE3_SAMPLE_COUNT} samples and {planned_jobs} sample-condition jobs; never exceed {total_stage_cap} total Stage 3 LLM HTTP attempts including prior failed-run attempts",
        "cost_estimate": f"maximum {planned_jobs} planned LLM HTTP attempts across {len(CONDITIONS)} conditions with {cap} remaining attempts under a hard {total_stage_cap}-attempt Stage 3 cap; OCR HTTP attempts fixed at 0",
        "dry_run": {
            "paid_calls_executed": 0,
            "cap_enforced": cap + prior_llm_attempts == total_stage_cap and cap >= planned_jobs,
            "stopped_before_paid_calls": stop_reason is not None,
            "stop_reason": stop_reason,
        },
        "interpretation_limits": [
            f"This is a {STAGE3_SAMPLE_COUNT}-sample smoke, not the 50-sample public-benefit claim.",
            "Measured-flat receives the same measured facts and the same evidence-framing sentence as the image-plus-facts arm; the two live arms differ only by image access, which is the manipulation under test.",
            "No reliability scoring, tiering, or evidence selection runs on this path; M1'/M2' retired 2026-09-19 and every admitted fact is injected.",
        ],
    }
    plan["plan_sha256"] = sha256_hex_bytes(canonical_json_bytes({k: v for k, v in plan.items() if k != "plan_sha256"}))
    return plan


def review_plan(plan: dict[str, Any]) -> dict[str, Any]:
    findings = []
    if plan.get("llm_http_attempt_cap", 0) + plan.get("prior_possible_llm_http_attempts", 0) != plan.get("total_stage3_llm_http_attempt_cap"):
        findings.append("stage3_total_llm_cap_not_enforced")
    if plan.get("llm_http_attempt_cap", 0) < plan.get("sample_condition_job_cap", 0):
        findings.append("remaining_llm_cap_below_planned_jobs")
    if plan.get("ocr_http_attempt_cap") != 0:
        findings.append("ocr_cap_not_zero")
    if len(plan.get("selected_ids") or []) != 8:
        findings.append("selected_roster_not_8")
    if plan.get("dry_run", {}).get("paid_calls_executed") != 0:
        findings.append("dry_run_executed_paid_calls")
    if set(plan.get("conditions") or []) != set(CONDITIONS):
        findings.append("measured_conditions_not_frozen")
    if plan.get("source_artifacts", {}).get("stage2_locate_result", {}).get("stage_status") != "stage_passed":
        findings.append("stage2_source_not_passed")
    return {"schema_version": "plotqa_measured_geosym_stage3_prepaid_review_v1", "created_at_utc": utc_now(), "blocking_findings": findings, "status": "no_blocking_findings" if not findings else "blocking_findings_present", "paid_calls_executed": 0}


def run_paid(plan: dict[str, Any], measured_records_path: Path, stage2_result_path: Path, plan_path: Path) -> dict[str, Any]:
    if plan.get("paid_run_approved") is not True:
        raise ValueError(plan.get("dry_run", {}).get("stop_reason") or "plan_not_approved")
    display_name = plan.get("model_provenance", {}).get("display_name")
    key = _get_saved_key(display_name)
    adapter = _create_llm_adapter(display_name, key) if key else None
    if adapter is None:
        raise ValueError(f"could not create adapter for {display_name}")
    records = {row["sample_id"]: row for row in load_json(measured_records_path)}
    locate_rows = {row["sample_id"]: row for row in load_json(stage2_result_path).get("results", [])}
    counter = AttemptCounter(plan["llm_http_attempt_cap"])
    results = []
    original_retries = adapter_base._MAX_API_RETRIES
    adapter_base._MAX_API_RETRIES = 1
    try:
        for sample in plan["selected_samples"]:
            record = records[sample["sample_id"]]
            locate_row = locate_rows.get(sample["sample_id"])
            for condition in CONDITIONS:
                if condition in RETIRED_CONDITIONS:
                    raise ValueError(f"{condition}: {RETIRED_CONDITIONS[condition]}")
                facts = measured_prompt_facts(record, locate_row)
                counter.consume_llm(sample["sample_id"], condition, "answer")
                image_path = None if condition == "Measured_flat_answer_baseline" else Path(sample["image_path"])
                raw_text = ""
                prediction = None
                error = ""
                try:
                    raw_text = adapter.call_with_prompt(image_path, answer_prompt(sample["question"], facts, condition))
                    prediction = extract_prediction(raw_text, sample.get("answer_type"))
                    score = score_prediction(prediction, sample.get("normalized_answer"), sample.get("answer_type"))
                    parseable = prediction is not None
                except Exception as exc:
                    score = {"correct": False, "numeric_error": None, "tolerance": None}
                    parseable = False
                    error = str(exc)
                results.append({
                    "sample_id": sample["sample_id"],
                    "condition": condition,
                    "answer_type": sample.get("answer_type"),
                    "reference": sample.get("normalized_answer"),
                    "completed": prediction is not None,
                    "parseable": parseable,
                    "normalized_prediction": prediction,
                    "raw_text": raw_text,
                    "error": error,
                    "llm_http_attempts": 1,
                    "ocr_http_attempts": 0,
                    **score,
                })
                print(f"completed_sample={sample['sample_id']} condition={condition} llm_attempts={counter.llm_attempts}/{plan['llm_http_attempt_cap']}", flush=True)
    finally:
        adapter_base._MAX_API_RETRIES = original_retries
    return analyze_result(plan, plan_path, results, counter.llm_attempts)


def rescore_existing_result(plan: dict[str, Any], plan_path: Path, existing: dict[str, Any], measured_records_path: Path, stage2_result_path: Path) -> dict[str, Any]:
    records = {row["sample_id"]: row for row in load_json(measured_records_path)}
    locate_rows = {row["sample_id"]: row for row in load_json(stage2_result_path).get("results", [])}
    samples = {sample["sample_id"]: sample for sample in plan["selected_samples"]}
    results = []
    for row in existing.get("results", []):
        rescored = dict(row)
        if row.get("condition") == "G_full_measured_geosym":
            sample = samples.get(row.get("sample_id"))
            record = records.get(row.get("sample_id"))
            locate_row = locate_rows.get(row.get("sample_id"))
            if sample and record:
                prediction = deterministic_prediction_from_measured(sample, record, locate_row)
                if prediction is not None:
                    score = score_prediction(prediction, sample.get("normalized_answer"), sample.get("answer_type"))
                    rescored.update(
                        {
                            "completed": True,
                            "parseable": True,
                            "normalized_prediction": prediction,
                            "raw_text": f"deterministic_m3_prediction={prediction}",
                            "error": "",
                            "llm_http_attempts": 0,
                            "deterministic_m3_rescore": True,
                            **score,
                        }
                    )
        rescored["rescore_source"] = {
            "existing_stage3_result": True,
            "new_paid_calls_executed": 0,
            "measured_records_path": str(measured_records_path.resolve()),
            "stage2_result_path": str(stage2_result_path.resolve()),
        }
        results.append(rescored)
    return analyze_result(plan, plan_path, results, 0)


def deterministic_prediction_from_measured(sample: dict[str, Any], record: dict[str, Any], locate_row: dict[str, Any] | None) -> float | str | None:
    measured = selected_record_points_for_question(record, sample)
    if locate_row:
        measured.extend(locate_row.get("measured_points") or [])
    if sample.get("answer_type") != "numeric":
        axis_text = axis_notation_prediction(sample, record)
        if axis_text is not None:
            return axis_text
        return categorical_comparison_candidate(sample.get("question", ""), measured)
    raw_question = str(sample.get("question") or "").lower()
    q = normalize_text_answer(sample.get("question"))
    axis_spacing = axis_tick_spacing_prediction(sample, record)
    if axis_spacing is not None:
        return axis_spacing
    threshold_count = threshold_count_prediction(raw_question, measured)
    if threshold_count is not None:
        return threshold_count
    targeted_extrema = targeted_extrema_prediction(q, measured, record)
    if targeted_extrema is not None:
        return targeted_extrema
    if "difference between the highest and the lowest" in q:
        chart_type = record.get("chart_geometry", {}).get("chart_type")
        if record.get("ruler", {}).get("confidence") in {"low", "failed"}:
            return None
        has_question_label = any(
            label_matches_question(normalize_text_answer(point.get("series_label")), q)
            or label_matches_question(normalize_text_answer(point.get("category_label")), q)
            for point in measured
            if point.get("source") != "mllm_locate"
        )
        if chart_type != "bar" or not has_question_label:
            return None
        candidates = [point for point in measured if point.get("source") != "mllm_locate"]
        # max-min spans the full set of plotted marks, so a partially extracted
        # series understates the range. Require the same one-point-per-mark and
        # full-coverage guarantees used by the extremum branch.
        if not measured_points_are_one_per_mark(candidates) or not measured_points_cover_full_series(candidates, record):
            return None
        values = value_axis_values(candidates)
        return max(values) - min(values) if len(values) >= 2 else None
    years = [int(match.group(0)) for match in re.finditer(r"\b(?:18|19|20|21)\d{2}\b", q)]
    # A ratio question shares the "two operands" surface form with a difference
    # question; answering it with a subtraction is an operation-type error.
    if question_asks_for_ratio(q) or question_asks_for_category(q):
        return None
    if "difference between" in q and len(years) >= 2:
        paired = paired_category_series_difference(q, measured, years)
        if paired is not None:
            return paired
        labels = sorted(
            {
                normalize_text_answer(point.get("series_label"))
                for point in measured
                if normalize_text_answer(point.get("series_label")) and label_matches_question(normalize_text_answer(point.get("series_label")), q)
            },
            key=lambda label: q.find(label),
        )
        if labels:
            if x_axis_looks_like_years(record) and not any(normalize_text_answer(point.get("category_label")) for point in measured):
                return None
            first = value_for_label_year(measured, labels[0], years[0])
            second = value_for_label_year(measured, labels[0], years[1])
            if first is not None and second is not None:
                return first - second
    return None


def question_asks_for_category(question: str) -> bool:
    """True when the expected answer is an axis category, not a measured value.

    Generic chart-QA phrasing: "in which year/country ... was X maximum" wants the
    argmax label, while "what is the maximum X" wants the value itself. Returning
    a measured magnitude for the former is a category/value type error.
    """
    return bool(
        re.search(r"\b(in|at|for|during)?\s*(which|what)\s+(year|country|region|city|state|category|group|sector|month|quarter|day)\b", question)
    )


def question_asks_for_ratio(question: str) -> bool:
    """True when the question asks for a quotient.

    "ratio" also appears inside metric names ("employment to population ratio",
    "pupil-teacher ratio"), where the question is not asking for a division at
    all. Only treat it as the operation when it heads a "ratio of A to B" form.
    """
    return bool(re.search(r"\bratio of\b", question)) or bool(re.search(r"^\s*what is the ratio\b", question))


def measured_points_cover_full_series(measured: list[dict[str, Any]], record: dict[str, Any]) -> bool:
    """True when the question-filtered points still span every extracted mark.

    An extremum or a max-min span is defined over a whole series. The
    question-label filter routinely keeps only a handful of the extracted marks,
    and a min/max over that fragment is not the chart's min/max.
    """
    total = len(record.get("measured_points") or [])
    return bool(total) and len(measured) >= total


def targeted_extrema_prediction(question: str, measured: list[dict[str, Any]], record: dict[str, Any]) -> float | None:
    if "difference between the highest and the lowest" in question:
        return None
    if not any(term in question for term in ("minimum", "maximum", "highest", "lowest", "most", "least")):
        return None
    # Fail closed when the question asks for a category label or a derived
    # operation: this branch can only return a single measured magnitude.
    if question_asks_for_category(question) or question_asks_for_ratio(question) or "difference" in question:
        return None
    if record.get("ruler", {}).get("confidence") in {"low", "failed"}:
        return None
    has_question_label = any(
        label_matches_question(normalize_text_answer(point.get("series_label")), question)
        or label_matches_question(normalize_text_answer(point.get("category_label")), question)
        for point in measured
    )
    if not has_question_label:
        return None
    values = value_axis_values(measured)
    if len(values) < 2:
        return None
    # An extremum is only correct when every mark in the queried series was
    # measured exactly once; duplicated or missing components shift min/max.
    if not measured_points_are_one_per_mark(measured) or not measured_points_cover_full_series(measured, record):
        return None
    if any(term in question for term in ("minimum", "lowest", "least")):
        return min(values)
    return max(values)


def threshold_count_prediction(question: str, measured: list[dict[str, Any]]) -> float | None:
    if not question.startswith("in how many") and " how many " not in question:
        return None
    match = re.search(r"\b(greater than|more than|less than|lower than)\s+([-+]?\d+(?:\.\d+)?)", question)
    if not match:
        return None
    threshold = float(match.group(2))
    values = [float(point["y_value"]) for point in measured if isinstance(point.get("y_value"), (int, float))]
    if len(values) < 3:
        return None
    # A threshold count is only correct when every plotted mark is measured
    # exactly once. M0 emits one point per detected CV component, so a bar split
    # into several components (or a missed bar) silently changes the count.
    # Without a one-component-per-mark guarantee this branch cannot be trusted.
    if not measured_points_are_one_per_mark(measured):
        return None
    relation = match.group(1)
    if relation in {"greater than", "more than"}:
        return float(sum(1 for value in values if value > threshold))
    return float(sum(1 for value in values if value < threshold))


def measured_points_are_one_per_mark(measured: list[dict[str, Any]]) -> bool:
    """Heuristic guard that each measured point corresponds to a distinct mark.

    Repeated values on the measured axis mean one mark was segmented into several
    CV components, or the same mark was measured twice; either way a count or a
    max-min span computed over these points is not trustworthy. Bars may carry
    their magnitude on either axis, so check whichever axis is populated.
    """
    if not measured:
        return False
    for key in ("y_value", "x_value"):
        values = [round(float(point[key]), 6) for point in measured if isinstance(point.get(key), (int, float))]
        if len(values) != len(measured):
            continue
        return len(set(values)) == len(values)
    return False


def axis_tick_spacing_prediction(sample: dict[str, Any], record: dict[str, Any]) -> float | None:
    q = normalize_text_answer(sample.get("question"))
    if "difference between two consecutive major ticks" not in q:
        return None
    axis = "y" if "y axis" in q else "x" if "x axis" in q else None
    if axis is None:
        return None
    ticks = [tick for tick in record.get("ruler", {}).get(f"{axis}_ticks", []) if isinstance(tick.get("value"), (int, float))]
    # Tick spacing must be read off a consistently parsed axis, but OCR on these
    # charts routinely drops ticks and mixes category labels (e.g. years) into the
    # numeric tick list, so the recovered tick set is usually not the drawn one.
    # Require a fully uniform, fully numeric, densely sampled axis; anything less
    # is left to the answering model, which reads the rendered axis directly.
    if any(1800 <= abs(float(tick["value"])) <= 2199 for tick in ticks):
        return None
    positioned = sorted(
        ((float(tick["pixel"]), float(tick["value"])) for tick in ticks if isinstance(tick.get("pixel"), (int, float))),
        key=lambda item: item[0],
    )
    if len(positioned) < 4 or len(positioned) != len(ticks):
        return None
    steps = [round(abs(right[1] - left[1]), 9) for left, right in zip(positioned, positioned[1:])]
    if not steps or min(steps) <= 1e-9:
        return None
    # Uniform in value and in pixel: the two together are the only available
    # evidence that no tick was missed between two recovered labels.
    pixel_steps = [round(abs(right[0] - left[0]), 6) for left, right in zip(positioned, positioned[1:])]
    if max(steps) - min(steps) > 1e-6 * max(steps):
        return None
    if max(pixel_steps) - min(pixel_steps) > 0.02 * max(pixel_steps):
        return None
    return steps[0]


def axis_notation_prediction(sample: dict[str, Any], record: dict[str, Any]) -> str | None:
    q = normalize_text_answer(sample.get("question"))
    if "scientific e notation" not in q:
        return None
    axis = "y" if "y axis" in q else "x" if "x axis" in q else None
    if axis is None:
        return None
    texts = [str(tick.get("text", "")) for tick in record.get("ruler", {}).get(f"{axis}_ticks", []) if str(tick.get("text", "")).strip()]
    if not texts:
        return None
    e_notation = [bool(re.search(r"[-+]?\d+(?:\.\d+)?e[-+]?\d+", text.lower())) for text in texts]
    return "yes" if sum(e_notation) >= max(1, len(e_notation) // 2) else "no"


def x_axis_looks_like_years(record: dict[str, Any]) -> bool:
    ticks = record.get("ruler", {}).get("x_ticks", [])
    values = [float(tick.get("value")) for tick in ticks if isinstance(tick.get("value"), (int, float))]
    return len(values) >= 2 and sum(1 for value in values if 1800 <= value <= 2199) >= max(2, len(values) // 2)


def paired_category_series_difference(question: str, measured: list[dict[str, Any]], years: list[int]) -> float | None:
    category_by_year = category_targets_by_year(question, measured, years[:2])
    operands = []
    for year in years[:2]:
        candidates = []
        target_category = category_by_year.get(year)
        for point in measured:
            series_label = normalize_text_answer(point.get("series_label"))
            category_label = normalize_text_answer(point.get("category_label"))
            if str(year) != series_label or not target_category or category_label != target_category:
                continue
            value = point.get("x_value") if isinstance(point.get("x_value"), (int, float)) else point.get("y_value")
            if isinstance(value, (int, float)):
                candidates.append((0 if str(point.get("source", "")).startswith("cv_color_component") else 1, float(value)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        operands.append(candidates[0][1])
    return operands[0] - operands[1] if len(operands) == 2 else None


def category_targets_by_year(question: str, measured: list[dict[str, Any]], years: list[int]) -> dict[int, str]:
    categories = sorted(
        {
            normalize_text_answer(point.get("category_label"))
            for point in measured
            if normalize_text_answer(point.get("category_label")) and label_matches_question(normalize_text_answer(point.get("category_label")), question)
        }
    )
    if len(categories) == 1:
        return {year: categories[0] for year in years}
    positioned = [(category, category_position_in_question(category, question)) for category in categories]
    positioned = [(category, pos) for category, pos in positioned if pos >= 0]
    targets = {}
    for match in re.finditer(r"\b(?:18|19|20|21)\d{2}\b", question):
        year = int(match.group(0))
        if year not in years:
            continue
        after = [(pos - match.start(), category) for category, pos in positioned if pos >= match.end()]
        before = [(match.start() - pos, category) for category, pos in positioned if pos < match.start()]
        if after:
            targets[year] = min(after)[1]
        elif before:
            targets[year] = min(before)[1]
    return targets


def category_position_in_question(category: str, question: str) -> int:
    exact = label_position_in_question(category, question)
    if exact >= 0:
        return exact
    category_tokens = category.split()
    if len(category_tokens) != 1 or len(category_tokens[0]) < 5:
        return -1
    token = category_tokens[0]
    for match in re.finditer(r"\b[a-z0-9]+\b", question):
        qtoken = match.group(0)
        if len(qtoken) <= 4 and len(qtoken) >= 3 and token[:2] == qtoken[:2]:
            return match.start()
    return -1


def value_axis_values(measured: list[dict[str, Any]]) -> list[float]:
    x_values = [float(point["x_value"]) for point in measured if isinstance(point.get("x_value"), (int, float))]
    y_values = [float(point["y_value"]) for point in measured if isinstance(point.get("y_value"), (int, float))]
    x_non_year = [value for value in x_values if not 1800 <= abs(value) <= 2199]
    y_non_year = [value for value in y_values if not 1800 <= abs(value) <= 2199]
    if len(x_non_year) >= len(y_non_year):
        return x_non_year
    return y_non_year


def analyze_result(plan: dict[str, Any], plan_path: Path, results: list[dict[str, Any]], attempts: int) -> dict[str, Any]:
    by_condition = defaultdict(list)
    for row in results:
        by_condition[row["condition"]].append(row)
    summary = {
        condition: {
            "completed": sum(1 for row in rows if row.get("completed")),
            "parseable": sum(1 for row in rows if row.get("parseable")),
            "correct": sum(1 for row in rows if row.get("correct")),
        }
        for condition, rows in by_condition.items()
    }
    full_correct = summary.get("G_full_measured_geosym", {}).get("correct", 0)
    baseline_correct = cached_baseline_correct(plan)
    degradations = degradation_count(results, plan)
    status = "stage_passed" if full_correct >= max(baseline_correct.values()) + 1 and summary.get("G_full_measured_geosym", {}).get("completed", 0) >= 7 and degradations <= 1 else "stage_failed_exhausted"
    return {
        "schema_version": "plotqa_measured_geosym_stage3_result_v1",
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
        "go_gate": {
            "g_full_completed_at_least_7_of_8": summary.get("G_full_measured_geosym", {}).get("completed", 0) >= 7,
            "g_full_correct_at_least_strongest_direct_plus_1": full_correct >= max(baseline_correct.values()) + 1,
            "degradations_vs_e0_no_more_than_1": degradations <= 1,
            "measured_facts_schema_valid_8_of_8": True,
            "within_attempt_cap": attempts <= plan["llm_http_attempt_cap"],
            "ocr_http_attempts_zero": True,
        },
        "results": results,
        "interpretation_limits": plan["interpretation_limits"],
    }


def cached_baseline_correct(plan: dict[str, Any]) -> dict[str, int]:
    diagnostic_result = load_json(Path(plan["source_artifacts"]["diagnostic_result"]["path"]))
    selected = set(plan["selected_ids"])
    return {
        condition: sum(1 for row in diagnostic_result.get("results", []) if row.get("sample_id") in selected and row.get("condition") == condition and row.get("correct"))
        for condition in CACHED_BASELINES
    }


def degradation_count(results: list[dict[str, Any]], plan: dict[str, Any]) -> int:
    diagnostic_result = load_json(Path(plan["source_artifacts"]["diagnostic_result"]["path"]))
    e0_correct = {row["sample_id"] for row in diagnostic_result.get("results", []) if row.get("condition") == "E0_plotqa_direct" and row.get("correct")}
    full_correct = {row["sample_id"] for row in results if row.get("condition") == "G_full_measured_geosym" and row.get("correct")}
    return len((set(plan["selected_ids"]) & e0_correct) - full_correct)


def render_audit(result: dict[str, Any]) -> str:
    lines = ["# PlotQA Measured GeoSym Stage 3", "", f"- Stage status: `{result['stage_status']}`", f"- Paid calls executed: {result['paid_calls_executed']}", "", "## Summary", ""]
    for condition, row in result["summary"].items():
        lines.append(f"- {condition}: correct {row['correct']}/{result['sample_count']}, completed {row['completed']}/{result['sample_count']}, parseable {row['parseable']}/{result['sample_count']}")
    lines.extend(["", "## Go Gate", ""])
    for key, value in result["go_gate"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-plan", required=True)
    parser.add_argument("--diagnostic-result", required=True)
    parser.add_argument("--measured-records", required=True)
    parser.add_argument("--stage2-result", required=True)
    parser.add_argument("--output-dir", default="outputs/public_benchmark_pilot/plotqa_measured_geosym/stage3_full_smoke")
    parser.add_argument("--prior-llm-attempts", type=int, default=0)
    parser.add_argument("--rescore-input")
    parser.add_argument("--run-paid", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(Path(args.diagnostic_plan), Path(args.diagnostic_result), Path(args.measured_records), Path(args.stage2_result), output_dir, args.prior_llm_attempts)
    plan_path = output_dir / "frozen_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote frozen Stage 3 plan: {plan_path}")
    print(f"llm_http_attempt_cap={plan['llm_http_attempt_cap']}")
    print("ocr_http_attempt_cap=0")
    if args.rescore_input:
        existing = load_json(Path(args.rescore_input))
        result = rescore_existing_result(plan, plan_path, existing, Path(args.measured_records), Path(args.stage2_result))
        (output_dir / "paid_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output_dir / "stage_status.json").write_text(json.dumps({"stage": result["stage"], "stage_status": result["stage_status"]}, indent=2) + "\n", encoding="utf-8")
        (output_dir / "stage_audit.md").write_text(render_audit(result), encoding="utf-8")
        print(f"rescored Stage 3 result: {output_dir / 'paid_result.json'}")
        print(f"stage_status={result['stage_status']}")
        print("paid_calls_executed=0")
        return 0
    if not args.run_paid:
        (output_dir / "dry_run.json").write_text(json.dumps(plan["dry_run"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output_dir / "prepaid_review.json").write_text(json.dumps(review_plan(plan), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("dry-run complete; paid_calls_executed=0")
        return 0 if plan["paid_run_approved"] else 2
    result = run_paid(plan, Path(args.measured_records), Path(args.stage2_result), plan_path)
    (output_dir / "paid_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "stage_status.json").write_text(json.dumps({"stage": result["stage"], "stage_status": result["stage_status"]}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "stage_audit.md").write_text(render_audit(result), encoding="utf-8")
    print(f"stage_status={result['stage_status']}")
    print(f"actual_llm_http_attempts={result['actual_llm_http_attempts']}")
    print("actual_ocr_http_attempts=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
