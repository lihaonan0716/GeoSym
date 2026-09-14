#!/usr/bin/env python
"""Stage 2 measured-M0 MLLM pixel-locate smoke for PlotQA.

Safe default: write a frozen plan and exit with paid_calls_executed=0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluator.adapters.base as adapter_base  # noqa: E402
from app.helpers import _create_llm_adapter, _get_saved_key  # noqa: E402
from scripts.prepare_public_benchmark_pilot import (  # noqa: E402
    canonical_json_bytes,
    sha256_hex_bytes,
    stable_file_hash,
)
from scripts.run_plotqa_measured_m0 import axis_model_is_noisy, pixel_to_value  # noqa: E402
from scripts.run_public_benchmark_pilot import AttemptCounter, first_json_object, score_prediction  # noqa: E402


LOCATE_PROMPT_TEMPLATE = """Return JSON only with key located_points.
Each located point must contain role, series_label, pixel_x, pixel_y, and confidence.
Use absolute image pixel coordinates. The top-left image corner is (0,0).
Forbidden: do not provide the final chart value, arithmetic result, normalized answer, or reference answer.
If the requested mark is ambiguous, return the best pixel coordinate and set confidence to low.

Question: {question}
Computed ruler context: {ruler_context}
Existing deterministic candidate pixels, if any: {candidate_pixels}
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prompt_hash() -> str:
    return hashlib.sha256(LOCATE_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def ruler_context(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "plot_area_bbox": record.get("chart_geometry", {}).get("plot_area_bbox"),
        "x_axis": record.get("ruler", {}).get("x_model"),
        "y_axis": record.get("ruler", {}).get("y_model"),
        "x_ticks": record.get("ruler", {}).get("x_ticks"),
        "y_ticks": record.get("ruler", {}).get("y_ticks"),
        "confidence": record.get("ruler", {}).get("confidence"),
        "evidence_gaps": record.get("evidence_gaps", []),
    }


def build_prompt(sample: dict[str, Any], record: dict[str, Any]) -> str:
    facts = [
        {
            "fact_id": point.get("fact_id"),
            "pixel": point.get("pixel"),
            "confidence": point.get("confidence"),
            "source": point.get("source"),
        }
        for point in record.get("measured_points", [])[:20]
    ]
    return LOCATE_PROMPT_TEMPLATE.format(
        question=sample["question"],
        ruler_context=json.dumps(ruler_context(record), ensure_ascii=False),
        candidate_pixels=json.dumps(facts, ensure_ascii=False),
    )


def selected_samples(diagnostic_plan: dict[str, Any], measured_records: list[dict[str, Any]], requested_ids: list[str] | None = None) -> list[dict[str, Any]]:
    records_by_id = {row["sample_id"]: row for row in measured_records}
    eligible = [sample for sample in diagnostic_plan["selected_samples"] if sample["sample_id"] in records_by_id]
    if requested_ids:
        sample_by_id = {sample["sample_id"]: sample for sample in eligible}
        return [sample_by_id[sample_id] for sample_id in requested_ids if sample_id in sample_by_id]
    return sorted(eligible, key=lambda row: hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest())[:8]


def build_plan(diagnostic_plan_path: Path, measured_records_path: Path, output_dir: Path, requested_ids: list[str] | None = None) -> dict[str, Any]:
    diagnostic_plan = load_json(diagnostic_plan_path)
    records = load_json(measured_records_path)
    samples = selected_samples(diagnostic_plan, records, requested_ids=requested_ids)
    stop_reason = None
    if len(samples) != 8:
        stop_reason = "stage2_requires_8_measured_samples"
    cap = 8
    model_provenance = diagnostic_plan.get("frozen_before_paid_run", {}).get("model_provenance", {})
    plan = {
        "schema_version": "plotqa_measured_m0_locate_smoke_plan_v1",
        "created_at_utc": utc_now(),
        "stage": "stage2_mllm_pixel_locate_smoke",
        "paid_run_approved": stop_reason is None,
        "paid_calls_executed": 0,
        "hypothesis": "A locate-only MLLM call can recover question-relevant mark pixels while deterministic code performs value conversion.",
        "falsification_signal": "If the model returns chart values instead of pixels for more than one sample or fewer than seven samples yield valid measured facts, repair the locate prompt before any full measured run.",
        "selected_ids": [sample["sample_id"] for sample in samples],
        "selected_samples": samples,
        "condition": "M0_measured_facts_only_locate",
        "llm_http_attempt_cap": cap,
        "ocr_http_attempt_cap": 0,
        "per_sample_attempt_cap": 1,
        "prompt_hashes": {"locate_pixel_prompt_template": prompt_hash()},
        "model_provenance": model_provenance,
        "reasoning_model": diagnostic_plan.get("frozen_before_paid_run", {}).get("reasoning_model"),
        "source_artifacts": {
            "diagnostic_plan": {"path": str(diagnostic_plan_path.resolve()), "sha256": stable_file_hash(diagnostic_plan_path)},
            "measured_records": {"path": str(measured_records_path.resolve()), "sha256": stable_file_hash(measured_records_path)},
        },
        "stop_rule": "stop after exactly the frozen 8 samples and at most one locate call per sample; no sample replacement",
        "interpretation_limits": [
            "This stage tests pixel localization only, not final-answer accuracy.",
            "Reference answers are used only after outputs are frozen for post-hoc tolerance diagnostics.",
            "No PlotQA templates, data tables, or answer fields are used in prompts.",
        ],
        "dry_run": {
            "paid_calls_executed": 0,
            "cap_enforced": cap == 8,
            "stopped_before_paid_calls": stop_reason is not None,
            "stop_reason": stop_reason,
        },
    }
    plan["plan_sha256"] = sha256_hex_bytes(canonical_json_bytes({k: v for k, v in plan.items() if k != "plan_sha256"}))
    return plan


def review_plan(plan: dict[str, Any]) -> dict[str, Any]:
    findings = []
    if plan.get("llm_http_attempt_cap") != 8:
        findings.append("llm_cap_not_8")
    if plan.get("ocr_http_attempt_cap") != 0:
        findings.append("ocr_http_cap_not_0")
    if len(plan.get("selected_ids") or []) != 8:
        findings.append("selected_roster_not_8")
    if plan.get("dry_run", {}).get("paid_calls_executed") != 0:
        findings.append("dry_run_executed_paid_calls")
    if "normalized_prediction" in LOCATE_PROMPT_TEMPLATE:
        findings.append("prompt_mentions_output_answer_key")
    if "{answer" in LOCATE_PROMPT_TEMPLATE or "{reference" in LOCATE_PROMPT_TEMPLATE:
        findings.append("prompt_template_has_answer_or_reference_slot")
    if "measured_facts" in LOCATE_PROMPT_TEMPLATE or "x_value" in LOCATE_PROMPT_TEMPLATE or "y_value" in LOCATE_PROMPT_TEMPLATE:
        findings.append("prompt_includes_chart_value_fact_channel")
    for key in ("diagnostic_plan", "measured_records"):
        source = plan.get("source_artifacts", {}).get(key, {})
        if not source.get("path") or not source.get("sha256"):
            findings.append(f"missing_source_hash:{key}")
    return {
        "schema_version": "plotqa_measured_m0_stage2_prepaid_review_v1",
        "created_at_utc": utc_now(),
        "reviewed_items": [
            "GEOSYM_PLOTQA_PERFORMANCE_IMPROVEMENT_PLAN.md",
            "scripts/run_plotqa_measured_m0.py",
            "scripts/run_plotqa_measured_locate_smoke.py",
            "stage2_locate_smoke/frozen_plan.json",
            "stage2_locate_smoke/dry_run.json",
        ],
        "blocking_findings": findings,
        "status": "no_blocking_findings" if not findings else "blocking_findings_present",
        "paid_calls_executed": 0,
    }


def extract_points(text: str) -> tuple[list[dict[str, Any]], bool]:
    try:
        data = first_json_object(text)
        raw = data.get("located_points")
        if not isinstance(raw, list):
            return [], False
        points = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                px = float(item["pixel_x"])
                py = float(item["pixel_y"])
            except Exception:
                continue
            value_leak = any(key in item for key in ("value", "chart_value", "answer", "normalized_prediction"))
            points.append(
                {
                    "role": str(item.get("role", "target")),
                    "series_label": str(item.get("series_label", "")),
                    "pixel_x": px,
                    "pixel_y": py,
                    "confidence": normalize_confidence(item.get("confidence")),
                    "value_leak": value_leak,
                }
            )
        return points, True
    except Exception:
        return fallback_points(text), False


def fallback_points(text: str) -> list[dict[str, Any]]:
    points = []
    for match in re.finditer(r"\(?\s*(?:x|pixel_x)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*[,;]\s*(?:y|pixel_y)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*\)?", text):
        points.append({"role": "target", "series_label": "", "pixel_x": float(match.group(1)), "pixel_y": float(match.group(2)), "confidence": "low", "value_leak": False})
    return points[:3]


def normalize_confidence(value: Any) -> str:
    text = str(value or "").lower()
    if text in {"high", "medium", "low"}:
        return text
    return "unknown"


def measured_from_locate(point: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    x_model = record.get("ruler", {}).get("x_model", {})
    y_model = record.get("ruler", {}).get("y_model", {})
    return {
        "role": point["role"],
        "series_label": point["series_label"],
        "pixel": [point["pixel_x"], point["pixel_y"]],
        "x_value": pixel_to_value(x_model, point["pixel_x"]),
        "y_value": pixel_to_value(y_model, point["pixel_y"]),
        "x_axis_model": x_model,
        "y_axis_model": y_model,
        "confidence": point["confidence"],
        "source": "mllm_locate",
        "uncertainty": {"pixel_abs": 6.0},
    }


def run_paid(plan: dict[str, Any], measured_records_path: Path, plan_path: Path) -> dict[str, Any]:
    if plan.get("paid_run_approved") is not True:
        raise ValueError(plan.get("dry_run", {}).get("stop_reason") or "plan_not_approved")
    display_name = plan.get("model_provenance", {}).get("display_name")
    if not display_name:
        raise ValueError("missing model display_name in frozen plan")
    key = _get_saved_key(display_name)
    if not key:
        raise ValueError(f"missing API key for {display_name}")
    adapter = _create_llm_adapter(display_name, key)
    if adapter is None:
        raise ValueError(f"could not create adapter for {display_name}")
    records_by_id = {row["sample_id"]: row for row in load_json(measured_records_path)}
    counter = AttemptCounter(plan["llm_http_attempt_cap"])
    results = []
    original_retries = adapter_base._MAX_API_RETRIES
    adapter_base._MAX_API_RETRIES = 1
    try:
        for sample in plan["selected_samples"]:
            record = records_by_id[sample["sample_id"]]
            counter.consume_llm(sample["sample_id"], plan["condition"], "locate")
            prompt = build_prompt(sample, record)
            raw_text = adapter.call_with_prompt(Path(sample["image_path"]), prompt)
            points, strict_json = extract_points(raw_text)
            measured = selected_record_points_for_question(record, sample) + [measured_from_locate(point, record) for point in points]
            results.append(
                {
                    "sample_id": sample["sample_id"],
                    "condition": plan["condition"],
                    "completed": bool(points),
                    "strict_json": strict_json,
                    "located_points": points,
                    "measured_points": measured,
                    "returned_value_leak": any(point.get("value_leak") for point in points),
                    "raw_text": raw_text,
                    "llm_http_attempts": 1,
                    "ocr_http_attempts": 0,
                    "posthoc_direct_tolerance": posthoc_direct_tolerance(sample, measured),
                }
            )
            print(f"completed_sample={sample['sample_id']} llm_attempts={counter.llm_attempts}/{plan['llm_http_attempt_cap']}", flush=True)
    finally:
        adapter_base._MAX_API_RETRIES = original_retries
    return summarize(plan, plan_path, results, counter.llm_attempts)


def rescore_existing_result(plan: dict[str, Any], plan_path: Path, existing: dict[str, Any], measured_records_path: Path) -> dict[str, Any]:
    records_by_id = {row["sample_id"]: row for row in load_json(measured_records_path)}
    sample_by_id = {sample["sample_id"]: sample for sample in plan["selected_samples"]}
    results = []
    for row in existing.get("results", []):
        sample_id = row.get("sample_id")
        record = records_by_id.get(sample_id)
        sample = sample_by_id.get(sample_id)
        if not record or not sample:
            continue
        points = row.get("located_points") or []
        measured = selected_record_points_for_question(record, sample) + [measured_from_locate(point, record) for point in points]
        rescored = dict(row)
        rescored["measured_points"] = measured
        rescored["posthoc_direct_tolerance"] = posthoc_direct_tolerance(sample, measured)
        rescored["rescore_source"] = {
            "located_points_reused": True,
            "measured_records_path": str(measured_records_path.resolve()),
            "new_paid_calls_executed": 0,
        }
        results.append(rescored)
    return summarize(plan, plan_path, results, existing.get("actual_llm_http_attempts", 0))


def posthoc_direct_tolerance(sample: dict[str, Any], measured: list[dict[str, Any]]) -> dict[str, Any]:
    if sample.get("answer_type") != "numeric":
        return posthoc_categorical_tolerance(sample, measured)
    y_values = [point.get("y_value") for point in measured if point.get("y_value") is not None]
    x_values = [point.get("x_value") for point in measured if point.get("x_value") is not None]
    values = operation_posthoc_candidates(sample.get("question", ""), y_values, x_values)
    hits = []
    for value in values:
        try:
            score = score_prediction(value, sample["normalized_answer"], "numeric")
        except Exception:
            continue
        hits.append({"value": value, **score})
    return {"eligible": bool(hits), "within_tolerance": any(hit["correct"] for hit in hits), "candidate_scores": hits}


def selected_record_points_for_question(record: dict[str, Any], sample: dict[str, Any]) -> list[dict[str, Any]]:
    points = list(record.get("measured_points") or [])
    question = normalize_text_answer(sample.get("question"))
    scored = []
    for point in points:
        label_hits = sum(
            1
            for label in (normalize_text_answer(point.get("series_label")), normalize_text_answer(point.get("category_label")))
            if label_matches_question(label, question)
        )
        if label_hits:
            source_bonus = 1 if str(point.get("source", "")).startswith("cv_color_component") else 0
            scored.append((label_hits, source_bonus, point))
    if scored:
        best_score = max(score for score, _source_bonus, _point in scored)
        selected = [point for score, _source_bonus, point in scored if score == best_score]
        if len(selected) < 2:
            selected = [point for score, _source_bonus, point in scored if score >= max(1, best_score - 1)]
        selected = sorted(
            selected,
            key=lambda point: (
                0 if str(point.get("source", "")).startswith("cv_color_component") else 1,
                str(point.get("fact_id", "")),
            ),
        )
    else:
        selected = points[:20]
    return [dict(point, source=f"{point.get('source', 'cv')}_deterministic_posthoc") for point in selected[:30]]


def label_matches_question(label: str, question: str) -> bool:
    if not label:
        return False
    if label_position_in_question(label, question) >= 0:
        return True
    label_tokens = [token for token in label.split() if token]
    question_tokens = set(question.split())
    if len(label_tokens) == 1 and len(label_tokens[0]) >= 5:
        token = label_tokens[0]
        return any(len(qtoken) <= 4 and len(qtoken) >= 3 and token[:2] == qtoken[:2] for qtoken in question_tokens)
    return False


def label_position_in_question(label: str, question: str) -> int:
    match = re.search(r"(?<![a-z0-9])" + re.escape(label) + r"(?![a-z0-9])", question)
    return -1 if match is None else match.start()


def posthoc_categorical_tolerance(sample: dict[str, Any], measured: list[dict[str, Any]]) -> dict[str, Any]:
    reference = normalize_text_answer(sample.get("normalized_answer"))
    if not reference:
        return {"eligible": False, "within_tolerance": None}
    candidates = []
    comparison = categorical_comparison_candidate(sample.get("question", ""), measured)
    if comparison:
        candidates.append({"value": comparison, "correct": comparison == reference})
    for point in measured:
        for key in ("series_label", "role"):
            value = normalize_text_answer(point.get(key))
            if value:
                candidates.append({"value": value, "correct": value == reference})
    return {"eligible": bool(candidates), "within_tolerance": any(item["correct"] for item in candidates), "candidate_scores": candidates}


def categorical_comparison_candidate(question: str, measured: list[dict[str, Any]]) -> str | None:
    q = normalize_text_answer(question)
    if not q.startswith("is ") or " difference between " not in q or " greater than " not in q:
        return None
    years = [int(match.group(0)) for match in re.finditer(r"\b(?:18|19|20|21)\d{2}\b", q)]
    if len(years) < 4:
        return None
    labels = sorted(
        {
            normalize_text_answer(point.get("series_label"))
            for point in measured
            if normalize_text_answer(point.get("series_label")) and label_matches_question(normalize_text_answer(point.get("series_label")), q)
        },
        key=lambda label: label_position_in_question(label, q),
    )
    if len(labels) < 2:
        return None
    left = signed_difference_for_label_years(measured, labels[0], years[0], years[1])
    right = signed_difference_for_label_years(measured, labels[1], years[2], years[3])
    if left is None or right is None:
        return None
    return "yes" if left > right else "no"


def signed_difference_for_label_years(measured: list[dict[str, Any]], label: str, first_year: int, second_year: int) -> float | None:
    first = value_for_label_year(measured, label, first_year)
    second = value_for_label_year(measured, label, second_year)
    if first is None or second is None:
        return None
    return first - second


def value_for_label_year(measured: list[dict[str, Any]], label: str, year: int) -> float | None:
    candidates = []
    for point in measured:
        point_label = normalize_text_answer(point.get("series_label"))
        if not label_matches_question(label, point_label):
            continue
        x_value = point.get("x_value")
        y_value = point.get("y_value")
        if isinstance(y_value, (int, float)) and isinstance(x_value, (int, float)):
            candidates.append((abs(float(y_value) - float(year)), float(x_value)))
        if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
            candidates.append((abs(float(x_value) - float(year)), float(y_value)))
    close = [(distance, value) for distance, value in candidates if distance <= 1.0]
    if not close:
        return None
    close.sort(key=lambda item: item[0])
    return close[0][1]


def normalize_text_answer(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def operation_posthoc_candidates(question: str, y_values: list[float], x_values: list[float]) -> list[float]:
    q = question.lower()
    candidates: list[float] = []
    if "ratio" in q and len(y_values) >= 2:
        denominator = y_values[1]
        if abs(float(denominator)) > 1e-12:
            candidates.append(float(y_values[0]) / float(denominator))
    if ("highest" in q and "lowest" in q) or "difference" in q:
        for values in (y_values, x_values):
            numeric = [float(v) for v in values if v is not None]
            if len(numeric) >= 2:
                ordered = sorted(numeric, reverse=True)
                candidates.append(max(numeric) - min(numeric))
                candidates.append(ordered[0] - ordered[1])
                if len(numeric) <= 12:
                    for left_idx, left in enumerate(numeric):
                        for right_idx, right in enumerate(numeric):
                            if left_idx != right_idx:
                                candidates.append(left - right)
                                candidates.append(abs(left - right))
    candidates.extend(float(value) for value in y_values if value is not None)
    candidates.extend(float(value) for value in x_values if value is not None)
    return candidates


def summarize(plan: dict[str, Any], plan_path: Path, results: list[dict[str, Any]], attempts: int) -> dict[str, Any]:
    for row in results:
        row["bottleneck_labels"] = stage2_bottleneck_labels(row)
    completed = sum(1 for row in results if row["completed"])
    strict_json = sum(1 for row in results if row["strict_json"])
    leaks = sum(1 for row in results if row["returned_value_leak"])
    direct = [row["posthoc_direct_tolerance"] for row in results if row["posthoc_direct_tolerance"]["eligible"]]
    within = sum(1 for row in direct if row["within_tolerance"])
    direct_tolerance_gate = within >= 6
    status = "stage_passed" if completed >= 7 and leaks <= 1 and direct_tolerance_gate else "stage_failed_exhausted"
    return {
        "schema_version": "plotqa_measured_m0_locate_smoke_result_v1",
        "created_at_utc": utc_now(),
        "stage": plan["stage"],
        "stage_status": status,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": plan["plan_sha256"],
        "condition": plan["condition"],
        "sample_count": len(results),
        "llm_http_attempt_cap": plan["llm_http_attempt_cap"],
        "ocr_http_attempt_cap": 0,
        "actual_llm_http_attempts": attempts,
        "actual_ocr_http_attempts": 0,
        "paid_calls_executed": attempts,
        "summary": {
            "completed": completed,
            "strict_json": strict_json,
            "returned_value_leak": leaks,
            "posthoc_direct_eligible": len(direct),
            "posthoc_direct_within_tolerance": within,
        },
        "go_gate": {
            "valid_measured_facts_at_least_7_of_8": completed >= 7,
            "posthoc_direct_tolerance_at_least_6_of_8": direct_tolerance_gate,
            "value_leak_no_more_than_1_of_8": leaks <= 1,
            "within_attempt_cap": attempts <= plan["llm_http_attempt_cap"],
            "ocr_http_attempts_zero": True,
        },
        "results": results,
        "interpretation_limits": plan["interpretation_limits"],
    }


def stage2_bottleneck_labels(row: dict[str, Any]) -> list[dict[str, Any]]:
    labels = []
    measured = row.get("measured_points") or []
    bad_ruler_points = [
        point for point in measured
        if axis_model_is_noisy(point.get("x_axis_model", {})) or axis_model_is_noisy(point.get("y_axis_model", {}))
    ]
    if not row.get("completed"):
        labels.append({"label": "m0_missing_mark", "fields": {"completed": False}})
    if row.get("returned_value_leak"):
        labels.append({"label": "unknown", "fields": {"reason": "locate_prompt_value_leak"}})
    if bad_ruler_points:
        labels.append(
            {
                "label": "m0_bad_ruler",
                "fields": {
                    "bad_point_count": len(bad_ruler_points),
                    "residual_cap_px": 12.0,
                },
            }
        )
    posthoc = row.get("posthoc_direct_tolerance") or {}
    if posthoc.get("eligible") is False:
        labels.append({"label": "m0_no_ruler", "fields": {"reason": "no_numeric_locate_value_for_posthoc_check"}})
    elif posthoc.get("within_tolerance") is False:
        labels.append({"label": "m0_measurement_out_of_tolerance", "fields": {"candidate_scores": posthoc.get("candidate_scores", [])}})
    return labels or [{"label": "unknown", "fields": {"reason": "stage2_locate_output_not_directly_diagnostic"}}]


def write_bottleneck_audit(output_dir: Path, result: dict[str, Any]) -> None:
    labels = [
        {"sample_id": row["sample_id"], "labels": row.get("bottleneck_labels", [])}
        for row in result.get("results", [])
    ]
    counts: dict[str, int] = {}
    for row in labels:
        for label in row["labels"]:
            counts[label["label"]] = counts.get(label["label"], 0) + 1
    audit = {
        "schema_version": "plotqa_measured_m0_stage2_bottleneck_audit_v1",
        "created_at_utc": utc_now(),
        "stage": result.get("stage"),
        "stage_status": result.get("stage_status"),
        "labels": labels,
        "label_counts": counts,
        "dominant_bottleneck": max(counts, key=counts.get) if counts else "unknown",
        "next_required_action": "repair_stage1_ruler_and_measurement" if counts.get("m0_measurement_out_of_tolerance", 0) else "review_stage2_locate_prompt",
    }
    (output_dir / "bottleneck_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_audit(result: dict[str, Any]) -> str:
    lines = [
        "# PlotQA Measured M0 Stage 2 Locate Smoke",
        "",
        f"- Stage status: `{result['stage_status']}`",
        f"- Completed: {result['summary']['completed']}/{result['sample_count']}",
        f"- Strict JSON: {result['summary']['strict_json']}/{result['sample_count']}",
        f"- Value leaks: {result['summary']['returned_value_leak']}/{result['sample_count']}",
        f"- Paid calls executed: {result['paid_calls_executed']}",
        "",
        "## Go Gate",
        "",
    ]
    for key, value in result["go_gate"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-plan", required=True)
    parser.add_argument("--measured-records", required=True)
    parser.add_argument("--output-dir", default="outputs/public_benchmark_pilot/plotqa_measured_geosym/stage2_locate_smoke")
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--run-paid", action="store_true")
    parser.add_argument("--rescore-input", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_ids = [item.strip() for item in args.sample_ids.split(",") if item.strip()]
    plan = build_plan(Path(args.diagnostic_plan), Path(args.measured_records), output_dir, requested_ids=requested_ids or None)
    plan_path = output_dir / "frozen_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote frozen Stage 2 locate plan: {plan_path}")
    print(f"llm_http_attempt_cap={plan['llm_http_attempt_cap']}")
    print("ocr_http_attempt_cap=0")
    if args.rescore_input:
        existing = load_json(Path(args.rescore_input))
        result = rescore_existing_result(plan, plan_path, existing, Path(args.measured_records))
        (output_dir / "paid_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output_dir / "stage_status.json").write_text(json.dumps({"stage": result["stage"], "stage_status": result["stage_status"]}, indent=2) + "\n", encoding="utf-8")
        (output_dir / "stage_audit.md").write_text(render_audit(result), encoding="utf-8")
        write_bottleneck_audit(output_dir, result)
        print(f"rescored Stage 2 locate result: {args.rescore_input}")
        print(f"stage_status={result['stage_status']}")
        print("paid_calls_executed=0")
        return 0
    if not args.run_paid:
        (output_dir / "dry_run.json").write_text(json.dumps(plan["dry_run"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output_dir / "prepaid_review.json").write_text(json.dumps(review_plan(plan), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("dry-run complete; paid_calls_executed=0")
        return 0 if plan["paid_run_approved"] else 2
    result = run_paid(plan, Path(args.measured_records), plan_path)
    (output_dir / "paid_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "stage_status.json").write_text(json.dumps({"stage": result["stage"], "stage_status": result["stage_status"]}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "stage_audit.md").write_text(render_audit(result), encoding="utf-8")
    write_bottleneck_audit(output_dir, result)
    print(f"stage_status={result['stage_status']}")
    print(f"actual_llm_http_attempts={result['actual_llm_http_attempts']}")
    print("actual_ocr_http_attempts=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
