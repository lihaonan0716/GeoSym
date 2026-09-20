#!/usr/bin/env python
"""Run a capped public benchmark pilot from a frozen plan.

Safe default: without ``--run-paid`` the script validates the plan and prints a
dry-run summary with ``paid_calls_executed=0``.
"""

from __future__ import annotations

import argparse
import copy
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
from app.helpers import MODEL_REGISTRY, _create_llm_adapter, _get_saved_key  # noqa: E402
from scripts.prepare_public_benchmark_pilot import (  # noqa: E402
    DEFAULT_CONDITIONS,
    SCORER_VERSION,
    e0_prompt,
    e1_prompt,
    g0_answer_prompt,
    g0_curate_prompt,
    g0_extract_prompt,
    normalize_answer,
)
from scripts import prepare_plotqa_public_benchmark_pilot as plotqa_prompts  # noqa: E402


PLOTQA_CONDITIONS = set(plotqa_prompts.DEFAULT_CONDITIONS)
SUPPORTED_PAID_CONDITIONS = set(DEFAULT_CONDITIONS) | PLOTQA_CONDITIONS
NUMERIC_ABS_TOLERANCE = 1.0
NUMERIC_REL_TOLERANCE = 0.05

# A year answer is an identifying label, not a magnitude: 2006 is not 2007, and
# no relative tolerance on it is meaningful. The generic numeric rule
# `max(1.0, |ref| * 0.05)` yields +/-100.35 years at ref=2007, so a prediction of
# 1907 or 2107 scored correct. Measured on the frozen 500-sample roster, this
# inflated every arm and -- because it inflated them by *different* amounts
# (E0 -7, G_measured_raw -10, Measured_flat -14 under the strict rule) -- it
# distorted precisely the between-arm comparison the artifacts exist to support.
#
# Detection uses two independent signals that must agree: the question asks
# which year, and the reference is an integral year-range value. On the frozen
# roster the two signals agree on all 37 samples with zero disagreement in
# either direction, so neither is carrying the rule alone. This is a property of
# the question type, not of any answer key -- no benchmark-specific constant
# selects a sample (invariant 5).
YEAR_QUESTION_PATTERN = re.compile(r"\b(?:in |during )?which year\b", re.IGNORECASE)
YEAR_VALUE_MIN, YEAR_VALUE_MAX = 1800, 2100

CONFIDENCE_LABELS = {"high", "medium", "low", "unknown"}
REJECT_REASON_LABELS = {"low_confidence", "contradiction_risk", "low_utility", "unsupported"}


def model_display_name(api_model_id: str) -> str:
    for display_name, (_, _, registry_model_id) in MODEL_REGISTRY.items():
        if registry_model_id == api_model_id:
            return display_name
    raise ValueError(f"model id is not present in registry: {api_model_id}")


def first_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("no JSON object found in model response")


# Phrases a model uses to decline rather than answer. Matched only inside the
# model's own ``normalized_prediction`` field, never against free-form
# reasoning, so a chain of thought that mentions uncertainty on the way to an
# answer is unaffected.
ABSTENTION_MARKERS = (
    "cannot be determined",
    "can not be determined",
    "cannot determine",
    "not determinable",
    "indeterminable",
    "unmeasurable",
    "not measurable",
    "unanswerable",
    "cannot be answered",
    "not available",
    "not provided",
    "insufficient data",
    "insufficient information",
    "no data",
    "unknown",
    "n/a",
)


def is_abstention(value: Any) -> bool:
    """True when the model declined to answer rather than answering wrongly.

    A declined answer must not become a number. ``normalize_answer`` scans for
    the first numeric token anywhere in the string, so "Cannot be determined
    (data for 1988 and 1990 is missing)" yields 1988.0 and a refusal is scored
    as a confident wrong answer. That misreads the model in the one direction
    that matters: it credits evidence-aware abstention as failure, and only the
    evidence arms can abstain, so the distortion is not symmetric between arms
    even though this code is (invariant 6).

    A bare numeric string is never an abstention, so "0" stays 0.0.
    """
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    lowered = text.lower()
    if any(marker in lowered for marker in ABSTENTION_MARKERS):
        # "unknown" inside a longer sentence that also states a number is an
        # answer with a caveat, not a refusal; require the phrase to dominate.
        return not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    return False


def extract_prediction(raw_text: str, answer_type: str) -> Any:
    """Parse the model's answer, or return None when it declined or failed.

    Returning None rather than raising keeps the caller's contract: an
    unparseable or declined response is recorded as not answered, which scores
    as incorrect without inventing a number the model never gave.
    """
    text = raw_text.strip()
    try:
        parsed = first_json_object(text)
        for field in ("normalized_prediction", "raw_text"):
            if field not in parsed:
                continue
            candidate = parsed[field]
            # Checked before normalisation: afterwards the refusal has already
            # collapsed into whichever digits it happened to contain.
            if answer_type == "numeric" and is_abstention(candidate):
                return None
            try:
                return normalize_answer(candidate, answer_type)
            except Exception:
                # This field carried no usable answer. Fall through to the next
                # one, then to the free-text paths below.
                continue
    except Exception:
        pass
    tag_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if tag_match:
        text = tag_match.group(1).strip()
    final_match = re.search(r"final answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        text = final_match.group(1).strip()
    if answer_type in {"categorical", "text"}:
        text = text.splitlines()[0] if text.splitlines() else text
    # The same guard on the free-text path. Only applied once the answer-bearing
    # span has been isolated: an unfenced refusal is still a refusal, but a
    # narrowed "<answer>7</answer>" is not, whatever the surrounding prose said.
    if answer_type == "numeric" and is_abstention(text):
        return None
    return normalize_answer(text, answer_type)


def is_year_answer(question: Any, reference: Any) -> bool:
    """True when the answer identifies a year and must therefore match exactly.

    Both signals must hold: the question asks *which year*, and the reference is
    an integral value inside the year range. Requiring both keeps a genuine
    magnitude that happens to land on 2010.0 from being scored as a label, and
    keeps a year question with an unexpected reference from being scored as a
    magnitude. See the note on ``YEAR_QUESTION_PATTERN`` for why this is a
    property of the question type rather than a benchmark-specific constant.
    """
    if not isinstance(question, str) or not YEAR_QUESTION_PATTERN.search(question):
        return False
    try:
        value = float(reference)
    except (TypeError, ValueError):
        return False
    return value == int(value) and YEAR_VALUE_MIN <= value <= YEAR_VALUE_MAX


def score_prediction(prediction: Any, reference: Any, answer_type: str,
                     question: Any = None) -> dict[str, Any]:
    """Score one prediction.

    ``question`` is optional so that the existing call sites keep working; when
    it is absent the year rule cannot fire and scoring falls back to the generic
    numeric tolerance. That default is deliberate: silently guessing that a
    reference *looks like* a year would change historical numbers for callers
    that never opted in. Callers that score year questions must pass it.
    """
    # A declined or unparseable answer scores as incorrect, explicitly. It used
    # to reach here only as an exception from `float(None)`, which callers
    # caught alongside genuine transport errors -- so "the model refused" and
    # "the request failed" became the same recorded outcome.
    if prediction is None:
        return {"correct": False, "numeric_error": None, "tolerance": None, "abstained": True}
    if answer_type == "numeric":
        pred = float(prediction)
        ref = float(reference)
        numeric_error = abs(pred - ref)
        if is_year_answer(question, reference):
            # Exact match only. Tolerance 0.0 is recorded rather than None so
            # that a reader can tell the strict rule fired from missing data.
            return {"correct": pred == ref, "numeric_error": numeric_error,
                    "tolerance": 0.0, "year_answer": True}
        tolerance = max(NUMERIC_ABS_TOLERANCE, abs(ref) * NUMERIC_REL_TOLERANCE)
        return {"correct": numeric_error <= tolerance, "numeric_error": numeric_error, "tolerance": tolerance}
    return {"correct": prediction == reference, "numeric_error": None, "tolerance": None}


def numeric_entities(text: str) -> list[float]:
    values = []
    for match in re.finditer(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text.replace(",", "")):
        try:
            values.append(float(match.group(0)))
        except ValueError:
            continue
    return values


def normalize_numeric_entity_list(value: Any, fallback_text: str = "") -> list[float]:
    if not isinstance(value, list):
        return numeric_entities(fallback_text)
    normalized = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            normalized.append(float(item))
        elif isinstance(item, str) and re.fullmatch(r"\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*", item.replace(",", "")):
            normalized.append(float(item.replace(",", "")))
    return normalized


def normalize_confidence(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in CONFIDENCE_LABELS:
        return text
    try:
        score = float(text)
    except ValueError:
        return "unknown"
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def normalize_categorical_entity_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict) and item.get("value") is not None:
            text = str(item.get("value")).strip()
        else:
            text = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if text:
            normalized.append(text)
    return normalized


def normalize_fact_list(value: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized = []
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            item = {"text": str(item)}
        text = str(item.get("text", "")).strip()
        entities = normalize_numeric_entity_list(item.get("numeric_entities"), text)
        fact_id = str(item.get("fact_id") or f"raw_{idx:03d}")
        normalized.append(
            {
                "fact_id": fact_id,
                "source_step": str(item.get("source_step") or key),
                "text": text,
                "numeric_entities": entities,
                "categorical_entities": normalize_categorical_entity_list(item.get("categorical_entities")),
                "confidence": normalize_confidence(item.get("confidence")),
            }
        )
    return normalized


def normalize_rejected_fact_list(value: Any) -> list[dict[str, Any]]:
    normalized = normalize_fact_list(value, "curate")
    raw_items = value if isinstance(value, list) else []
    for idx, fact in enumerate(normalized):
        raw_item = raw_items[idx] if idx < len(raw_items) and isinstance(raw_items[idx], dict) else {}
        reason = str(raw_item.get("reason") or "unsupported").strip().lower()
        fact["reason"] = reason if reason in REJECT_REASON_LABELS else "unsupported"
    return normalized


def validate_fact_schema(fact: Any, path: str, require_reason: bool = False) -> list[str]:
    violations = []
    if not isinstance(fact, dict):
        return [f"{path}:not_object"]
    for key in ("fact_id", "source_step", "text"):
        if not isinstance(fact.get(key), str) or not fact.get(key).strip():
            violations.append(f"{path}:missing_or_invalid_{key}")
    if not isinstance(fact.get("numeric_entities"), list) or any(
        not isinstance(item, (int, float)) or isinstance(item, bool) for item in fact.get("numeric_entities", [])
    ):
        violations.append(f"{path}:numeric_entities_not_number_array")
    if not isinstance(fact.get("categorical_entities"), list) or any(
        not isinstance(item, str) for item in fact.get("categorical_entities", [])
    ):
        violations.append(f"{path}:categorical_entities_not_string_array")
    if fact.get("confidence") not in CONFIDENCE_LABELS:
        violations.append(f"{path}:confidence_not_ordinal")
    if require_reason and fact.get("reason") not in REJECT_REASON_LABELS:
        violations.append(f"{path}:reason_not_supported")
    return violations


def validate_trace_record(record: dict[str, Any]) -> list[str]:
    violations = []
    raw_observations = record.get("raw_observations")
    retained_facts = record.get("retained_facts")
    rejected = record.get("rejected_or_low_confidence_facts")
    if not isinstance(raw_observations, list):
        violations.append("raw_observations_not_list")
        raw_observations = []
    if not isinstance(retained_facts, list):
        violations.append("retained_facts_not_list")
        retained_facts = []
    if rejected is None:
        rejected = []
    if not isinstance(rejected, list):
        violations.append("rejected_or_low_confidence_facts_not_list")
        rejected = []
    for idx, fact in enumerate(raw_observations):
        violations.extend(validate_fact_schema(fact, f"raw_observations[{idx}]"))
    raw_ids = {fact.get("fact_id") for fact in raw_observations if isinstance(fact, dict)}
    for idx, fact in enumerate(retained_facts):
        violations.extend(validate_fact_schema(fact, f"retained_facts[{idx}]"))
        if isinstance(fact, dict) and fact.get("fact_id") not in raw_ids:
            violations.append(f"retained_facts[{idx}]:fact_id_not_in_raw_observations")
    for idx, fact in enumerate(rejected):
        violations.extend(validate_fact_schema(fact, f"rejected_or_low_confidence_facts[{idx}]", require_reason=True))
    answer = record.get("answer")
    if not isinstance(answer, dict):
        violations.append("answer_not_object")
    else:
        if "raw_text" not in answer:
            violations.append("answer_missing_raw_text")
        if "normalized_prediction" not in answer:
            violations.append("answer_missing_normalized_prediction")
    if not record.get("attribution_label"):
        violations.append("missing_attribution_label")
    return violations


def normalize_trace_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(record)
    if "raw_observations" in normalized:
        normalized["raw_observations"] = normalize_fact_list(normalized.get("raw_observations"), "extract")
    if "retained_facts" in normalized:
        normalized["retained_facts"] = normalize_fact_list(normalized.get("retained_facts"), "curate")
    if "rejected_or_low_confidence_facts" in normalized:
        normalized["rejected_or_low_confidence_facts"] = normalize_rejected_fact_list(
            normalized.get("rejected_or_low_confidence_facts")
        )
    return normalized


def _reference_matches_fact(reference: Any, answer_type: str, fact: dict[str, Any]) -> bool:
    if answer_type == "numeric":
        try:
            ref = float(reference)
        except Exception:
            return False
        tolerance = max(NUMERIC_ABS_TOLERANCE, abs(ref) * NUMERIC_REL_TOLERANCE)
        for value in fact.get("numeric_entities") or []:
            try:
                if abs(float(value) - ref) <= tolerance:
                    return True
            except Exception:
                continue
        return False
    ref_text = normalize_answer(reference, "text")
    fact_text = normalize_answer(fact.get("text", ""), "text")
    return bool(ref_text) and ref_text in fact_text


def _proxy_eligible(metadata: dict[str, Any]) -> bool:
    reasoning_type = metadata.get("reasoning_type")
    return reasoning_type in {"lookup", "extrema"} or (
        reasoning_type == "comparison" and metadata.get("answer_derivation") == "direct"
    )


def assign_attribution(record: dict[str, Any], sample: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    metadata = sample.get("metadata") or {}
    proxy_prefix = "proxy_" if metadata.get("benchmark") == "PlotQA" else ""
    reference = record.get("reference")
    answer_type = record.get("answer_type")
    proxy = {
        "reference_value_parseable": True,
        "prediction_value_parseable": record.get("parseable") is True,
        "answer_value_mentioned_in_raw_observations": False,
        "answer_value_mentioned_in_retained_facts": False,
        "numeric_entities_overlap_raw": False,
        "numeric_entities_overlap_retained": False,
    }
    try:
        normalize_answer(reference, answer_type)
    except Exception:
        proxy["reference_value_parseable"] = False
        return "ambiguous_ground_truth", proxy, "reference answer is not parseable under the frozen scorer"
    if record.get("correct") is True:
        return "correct", proxy, "G0 answer is correct under the frozen scorer"
    if record.get("completed") and record.get("raw_answer_text") and not record.get("parseable"):
        return "normalization_failure", proxy, "raw answer was present but normalized prediction was not parseable"
    if not _proxy_eligible(metadata):
        return "attribution_unknown", proxy, "derived or unsupported reasoning type has no frozen operand schema"

    raw_matches = [
        fact for fact in record.get("raw_observations") or [] if _reference_matches_fact(reference, answer_type, fact)
    ]
    retained_matches = [
        fact for fact in record.get("retained_facts") or [] if _reference_matches_fact(reference, answer_type, fact)
    ]
    proxy["answer_value_mentioned_in_raw_observations"] = bool(raw_matches)
    proxy["answer_value_mentioned_in_retained_facts"] = bool(retained_matches)
    proxy["numeric_entities_overlap_raw"] = bool(raw_matches) if answer_type == "numeric" else False
    proxy["numeric_entities_overlap_retained"] = bool(retained_matches) if answer_type == "numeric" else False
    if not raw_matches:
        return f"{proxy_prefix}extraction_failure", proxy, "reference value or token was absent from raw observations"
    if not retained_matches:
        return f"{proxy_prefix}curation_failure", proxy, "reference-matching raw observation was not retained"
    return f"{proxy_prefix}reasoning_failure", proxy, "reference-matching retained fact was present but final answer was wrong"


class AttemptCounter:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.llm_attempts = 0
        self.ocr_attempts = 0

    def consume_llm(self, sample_id: str, condition: str, step: str) -> None:
        if self.llm_attempts >= self.cap:
            raise RuntimeError(f"LLM attempt cap exceeded before {sample_id}/{condition}/{step}")
        self.llm_attempts += 1


def load_plan(path: Path, allow_paper_level: bool = False) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    frozen = plan["frozen_before_paid_run"]
    samples = plan["selected_samples"]
    conditions = frozen["conditions"]
    if set(conditions) - SUPPORTED_PAID_CONDITIONS:
        unsupported = sorted(set(conditions) - SUPPORTED_PAID_CONDITIONS)
        raise ValueError(f"unsupported Phase 1 conditions: {unsupported}")
    expected_jobs = len(samples) * len(conditions)
    if expected_jobs != frozen["sample_condition_job_cap"]:
        raise ValueError("selected sample-condition count does not match frozen cap")
    expected_attempts = len(samples) * sum(3 if condition in {"G0_public_evidence_trace", "G0_plotqa_evidence_trace"} else 1 for condition in conditions)
    if expected_attempts != frozen["llm_http_attempt_cap"]:
        raise ValueError("selected sample-step count does not match frozen LLM attempt cap")
    is_plotqa_paper_level_500 = (
        allow_paper_level
        and plan.get("benchmark") == "PlotQA"
        and plan.get("schema_version") == "plotqa_g0_smoke_plan_v1"
        and len(samples) == 500
        and frozen.get("sample_selection", {}).get("sampling_frame") == "official_plotqa_full_split_hash_pool"
    )
    if plan.get("benchmark") == "PlotQA":
        max_llm_cap = 2500 if is_plotqa_paper_level_500 else 250
    else:
        max_llm_cap = 40
    if frozen["llm_http_attempt_cap"] > max_llm_cap:
        raise ValueError(f"public benchmark cap exceeded: LLM attempts must be <= {max_llm_cap}")
    if frozen.get("ocr_http_attempt_cap") != 0:
        raise ValueError("Phase 1 public smoke requires OCR disabled")
    if plan.get("benchmark") == "PlotQA":
        if is_plotqa_paper_level_500:
            pass
        elif len(samples) not in {8, 50}:
            raise ValueError("PlotQA public run requires 8 smoke samples, 50 diagnostic samples, or explicit --allow-paper-level for the frozen 500-sample plan")
    elif len(samples) != 8:
        raise ValueError("Phase 1 public smoke requires exactly 8 selected samples")
    if plan.get("benchmark") == "PlotQA":
        if plan.get("schema_version") != "plotqa_g0_smoke_plan_v1":
            raise ValueError("PlotQA runner requires plotqa_g0_smoke_plan_v1")
        if frozen.get("sample_selection", {}).get("sampling_frame") != "official_plotqa_full_split_hash_pool":
            raise ValueError("PlotQA runner requires official full-split hash pool")
        if len({sample.get("metadata", {}).get("image_index") for sample in samples}) != len(samples):
            raise ValueError("PlotQA runner requires unique image_index values")
        model_provenance = frozen.get("model_provenance")
        if not isinstance(model_provenance, dict):
            raise ValueError("PlotQA runner requires model_provenance")
        if not model_provenance.get("display_name") or not model_provenance.get("api_model_id"):
            raise ValueError("PlotQA runner requires display_name and api_model_id in model_provenance")
        if frozen.get("reasoning_model") != model_provenance.get("api_model_id"):
            raise ValueError("PlotQA reasoning_model does not match model_provenance api_model_id")
    return plan


def _call_model(adapter: Any, image: Path | None, prompt: str) -> str:
    return adapter.call_with_prompt(image, prompt)


def _record_error(record: dict[str, Any], exc: Exception) -> dict[str, Any]:
    record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def run_direct(adapter: Any, image_path: Path, sample: dict[str, Any], condition: str, counter: AttemptCounter) -> dict[str, Any]:
    if condition == "E0_public_direct":
        prompt = e0_prompt(sample["question"])
    elif condition == "E1_public_cot":
        prompt = e1_prompt(sample["question"])
    elif condition == "E0_plotqa_direct":
        prompt = plotqa_prompts.e0_prompt(sample["question"])
    elif condition == "E1_plotqa_cot":
        prompt = plotqa_prompts.e1_prompt(sample["question"])
    else:
        raise ValueError(f"unsupported direct condition: {condition}")
    record = {
        "sample_id": sample["sample_id"],
        "condition": condition,
        "answer_type": sample["answer_type"],
        "reference": sample["normalized_answer"],
        "completed": False,
        "parseable": False,
        "correct": None,
        "numeric_error": None,
        "tolerance": None,
        "llm_http_attempts": 0,
        "ocr_http_attempts": 0,
        "error": "",
    }
    try:
        counter.consume_llm(sample["sample_id"], condition, "answer")
        record["llm_http_attempts"] = 1
        raw_text = _call_model(adapter, image_path, prompt)
        prediction = extract_prediction(raw_text, sample["answer_type"])
        record.update(
            {
                "completed": True,
                "parseable": True,
                "raw_text": raw_text,
                "normalized_prediction": prediction,
                **score_prediction(prediction, sample["normalized_answer"], sample["answer_type"]),
            }
        )
    except Exception as exc:
        _record_error(record, exc)
    return record


def run_g0(adapter: Any, image_path: Path, sample: dict[str, Any], condition: str, counter: AttemptCounter) -> dict[str, Any]:
    if condition == "G0_public_evidence_trace":
        extract_prompt = g0_extract_prompt
        curate_prompt = g0_curate_prompt
        answer_prompt = g0_answer_prompt
    elif condition == "G0_plotqa_evidence_trace":
        extract_prompt = plotqa_prompts.g0_extract_prompt
        curate_prompt = plotqa_prompts.g0_curate_prompt
        answer_prompt = plotqa_prompts.g0_answer_prompt
    else:
        raise ValueError(f"unsupported G0 condition: {condition}")
    record = {
        "sample_id": sample["sample_id"],
        "condition": condition,
        "answer_type": sample["answer_type"],
        "reference": sample["normalized_answer"],
        "completed": False,
        "parseable": False,
        "correct": None,
        "numeric_error": None,
        "tolerance": None,
        "llm_http_attempts": 0,
        "ocr_http_attempts": 0,
        "raw_observations": [],
        "retained_facts": [],
        "rejected_or_low_confidence_facts": [],
        "answer": {"raw_text": "", "normalized_prediction": None, "correct": None},
        "attribution_label": "attribution_unknown",
        "attribution_rationale": "",
        "attribution_proxy": {},
        "error": "",
    }
    try:
        counter.consume_llm(sample["sample_id"], record["condition"], "extract")
        record["llm_http_attempts"] += 1
        raw_extract = _call_model(adapter, image_path, extract_prompt(sample["question"]))
        extract_json = first_json_object(raw_extract)
        raw_observations = normalize_fact_list(extract_json.get("raw_observations"), "extract")
        record["raw_observations"] = raw_observations

        counter.consume_llm(sample["sample_id"], record["condition"], "curate")
        record["llm_http_attempts"] += 1
        raw_curate = _call_model(adapter, None, curate_prompt(sample["question"], raw_observations))
        curate_json = first_json_object(raw_curate)
        retained = normalize_fact_list(curate_json.get("retained_facts"), "curate")
        raw_ids = {fact["fact_id"] for fact in raw_observations}
        if any(fact["fact_id"] not in raw_ids for fact in retained):
            raise ValueError("curation output introduced fact_id values not present in raw observations")
        record["retained_facts"] = retained
        record["rejected_or_low_confidence_facts"] = normalize_rejected_fact_list(
            curate_json.get("rejected_or_low_confidence_facts")
        )

        counter.consume_llm(sample["sample_id"], record["condition"], "answer")
        record["llm_http_attempts"] += 1
        raw_answer = _call_model(adapter, None, answer_prompt(sample["question"], retained))
        prediction = extract_prediction(raw_answer, sample["answer_type"])
        score = score_prediction(prediction, sample["normalized_answer"], sample["answer_type"])
        record.update(
            {
                "completed": True,
                "parseable": True,
                "raw_answer_text": raw_answer,
                "normalized_prediction": prediction,
                **score,
            }
        )
        record["answer"] = {"raw_text": raw_answer, "normalized_prediction": prediction, "correct": score["correct"]}
    except Exception as exc:
        _record_error(record, exc)
    label, proxy, rationale = assign_attribution(record, sample)
    record["attribution_label"] = label
    record["attribution_proxy"] = proxy
    record["attribution_rationale"] = rationale
    return record


def run_paid(
    plan: dict[str, Any],
    plan_path: Path,
    checkpoint_path: Path | None = None,
    max_new_jobs: int | None = None,
) -> dict[str, Any]:
    if plan.get("paid_run_approved") is not True:
        raise ValueError("frozen plan is not approved for paid execution")
    frozen = plan["frozen_before_paid_run"]
    model_provenance = frozen.get("model_provenance") or {}
    display_name = model_provenance.get("display_name") or model_display_name(frozen["reasoning_model"])
    key = _get_saved_key(display_name)
    if not key:
        raise ValueError(f"missing API key for {display_name}")
    adapter = _create_llm_adapter(display_name, key)
    if adapter is None:
        raise ValueError(f"could not create adapter for {display_name}")

    manifest_dir = Path(plan.get("manifest_path", plan_path)).resolve().parent
    counter = AttemptCounter(frozen["llm_http_attempt_cap"])
    results = []
    completed_keys = set()
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        results = list(checkpoint.get("results") or [])
        completed_keys = {(row.get("sample_id"), row.get("condition")) for row in results}
        counter.llm_attempts = int(checkpoint.get("actual_llm_http_attempts") or sum(int(row.get("llm_http_attempts") or 0) for row in results))
        counter.ocr_attempts = int(checkpoint.get("actual_ocr_http_attempts") or sum(int(row.get("ocr_http_attempts") or 0) for row in results))
    new_jobs = 0
    original_retries = adapter_base._MAX_API_RETRIES
    adapter_base._MAX_API_RETRIES = 1
    try:
        for sample in plan["selected_samples"]:
            image_path = Path(sample["image_path"])
            resolved_image = image_path if image_path.is_absolute() else manifest_dir / image_path
            for condition in frozen["conditions"]:
                key_tuple = (sample["sample_id"], condition)
                if key_tuple in completed_keys:
                    continue
                if max_new_jobs is not None and new_jobs >= max_new_jobs:
                    result = summarize_result(plan, plan_path, results, counter.llm_attempts, counter.ocr_attempts)
                    result["stage_status"] = "partial_paid_run"
                    result["partial_run_stop_reason"] = "max_new_jobs_reached"
                    if checkpoint_path:
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        checkpoint_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    return result
                print(f"starting_sample={sample['sample_id']} condition={condition}", flush=True)
                if condition in {"G0_public_evidence_trace", "G0_plotqa_evidence_trace"}:
                    row = run_g0(adapter, resolved_image, sample, condition, counter)
                else:
                    row = run_direct(adapter, resolved_image, sample, condition, counter)
                results.append(row)
                completed_keys.add(key_tuple)
                new_jobs += 1
                print(
                    f"completed_sample={sample['sample_id']} condition={condition} llm_attempts={counter.llm_attempts}/{counter.cap}",
                    flush=True,
                )
                if checkpoint_path:
                    checkpoint = summarize_result(plan, plan_path, results, counter.llm_attempts, counter.ocr_attempts)
                    checkpoint["stage_status"] = "partial_paid_run"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    finally:
        adapter_base._MAX_API_RETRIES = original_retries

    result = summarize_result(plan, plan_path, results, counter.llm_attempts, counter.ocr_attempts)
    result["stage_status"] = "completed_paid_run"
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def summarize_result(
    plan: dict[str, Any], plan_path: Path, results: list[dict[str, Any]], llm_attempts: int, ocr_attempts: int
) -> dict[str, Any]:
    frozen = plan["frozen_before_paid_run"]
    g0_conditions = {"G0_public_evidence_trace", "G0_plotqa_evidence_trace"}
    direct_conditions = {"E0_public_direct", "E1_public_cot", "E0_plotqa_direct", "E1_plotqa_cot"}
    g0 = [row for row in results if row["condition"] in g0_conditions]
    direct = [row for row in results if row["condition"] in direct_conditions]
    return {
        "schema_version": "plotqa_g0_smoke_result_v1" if plan.get("benchmark") == "PlotQA" else "chartqa_g0_smoke_result_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan_path),
        "plan_sha256": plan.get("plan_sha256"),
        "benchmark": plan["benchmark"],
        "reasoning_model": frozen["reasoning_model"],
        "conditions": frozen["conditions"],
        "sample_count": len(plan["selected_samples"]),
        "sample_condition_job_cap": frozen["sample_condition_job_cap"],
        "llm_http_attempt_cap": frozen["llm_http_attempt_cap"],
        "ocr_http_attempt_cap": frozen["ocr_http_attempt_cap"],
        "actual_llm_http_attempts": llm_attempts,
        "actual_ocr_http_attempts": ocr_attempts,
        "scorer_version": frozen.get("scorer_version", SCORER_VERSION),
        "prompt_hashes": frozen["prompt_hashes"],
        "interpretation_limits": frozen["interpretation_limits"],
        "results": results,
        "summary": {
            "completed_jobs": sum(1 for row in results if row.get("completed")),
            "parseable_jobs": sum(1 for row in results if row.get("parseable")),
            "correct_jobs": sum(1 for row in results if row.get("correct") is True),
            "g0_completed": sum(1 for row in g0 if row.get("completed")),
            "g0_parseable_trace_records": sum(1 for row in g0 if not validate_trace_record(row)),
            "direct_completed": sum(1 for row in direct if row.get("completed")),
            "attribution_labels": {
                label: sum(1 for row in g0 if row.get("attribution_label") == label)
                for label in sorted({row.get("attribution_label") for row in g0})
            },
        },
    }


def rescore_existing(result: dict[str, Any]) -> dict[str, Any]:
    rescored = copy.deepcopy(result)
    for record in rescored.get("results", []):
        if record.get("condition") in {"G0_public_evidence_trace", "G0_plotqa_evidence_trace"}:
            record.update(normalize_trace_record(record))
        if not record.get("completed"):
            continue
        raw_text = record.get("raw_answer_text") or record.get("raw_text") or record.get("answer", {}).get("raw_text")
        if not raw_text:
            continue
        try:
            prediction = extract_prediction(raw_text, record["answer_type"])
            score = score_prediction(prediction, record["reference"], record["answer_type"])
        except Exception as exc:
            record["parseable"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            continue
        record["parseable"] = True
        record["normalized_prediction"] = prediction
        record.update(score)
    rescored["rescored_at_utc"] = datetime.now(timezone.utc).isoformat()
    rescored["rescore_note"] = "zero-cost rescore from cached raw text; no additional paid calls"
    results = rescored.get("results", [])
    g0 = [row for row in results if row.get("condition") in {"G0_public_evidence_trace", "G0_plotqa_evidence_trace"}]
    rescored["summary"] = {
        "completed_jobs": sum(1 for row in results if row.get("completed")),
        "parseable_jobs": sum(1 for row in results if row.get("parseable")),
        "correct_jobs": sum(1 for row in results if row.get("correct") is True),
        "g0_completed": sum(1 for row in g0 if row.get("completed")),
        "g0_parseable_trace_records": sum(1 for row in g0 if not validate_trace_record(row)),
    }
    return rescored


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Frozen public benchmark pilot plan JSON.")
    parser.add_argument("--output", required=True, help="Result JSON output path.")
    parser.add_argument("--run-paid", action="store_true", help="Execute the frozen paid calls.")
    parser.add_argument("--rescore-input", help="Existing result JSON to rescore without paid calls.")
    parser.add_argument("--allow-paper-level", action="store_true", help="Allow the explicitly frozen PlotQA 500 paper-level plan and its 2500-attempt cap.")
    parser.add_argument("--max-new-jobs", type=int, help="Optional resumable paid-run throttle for transport smoke or chunking.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    plan_path = Path(args.plan).resolve()
    output_path = Path(args.output)
    try:
        plan = load_plan(plan_path, allow_paper_level=args.allow_paper_level)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    frozen = plan["frozen_before_paid_run"]
    if args.rescore_input:
        result = json.loads(Path(args.rescore_input).read_text(encoding="utf-8"))
        rescored = rescore_existing(result)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rescored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote rescored public benchmark pilot result: {output_path}")
        print(f"completed_jobs={rescored['summary']['completed_jobs']}")
        print(f"parseable_jobs={rescored['summary']['parseable_jobs']}")
        print(f"correct_jobs={rescored['summary']['correct_jobs']}")
        print("paid_calls_executed=0")
        return 0

    if not args.run_paid:
        print("frozen paid plan loaded; no paid calls executed")
        print(f"benchmark={plan['benchmark']}")
        print(f"conditions={','.join(frozen['conditions'])}")
        print(f"sample_condition_job_cap={frozen['sample_condition_job_cap']}")
        print(f"llm_http_attempt_cap={frozen['llm_http_attempt_cap']}")
        print(f"allow_paper_level={args.allow_paper_level}")
        print("paid_calls_executed=0")
        return 0

    try:
        result = run_paid(plan, plan_path, checkpoint_path=output_path, max_new_jobs=args.max_new_jobs)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote public benchmark pilot result: {output_path}")
    print(f"completed_jobs={result['summary']['completed_jobs']}")
    print(f"parseable_jobs={result['summary']['parseable_jobs']}")
    print(f"correct_jobs={result['summary']['correct_jobs']}")
    print(f"actual_llm_http_attempts={result['actual_llm_http_attempts']}")
    print(f"actual_ocr_http_attempts={result['actual_ocr_http_attempts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
