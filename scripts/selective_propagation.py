"""Utilities for the M1/M2 selective propagation protocol."""

from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Any

from scripts.prepare_public_benchmark_pilot import normalize_answer, stable_payload_hash
from scripts.run_public_benchmark_pilot import NUMERIC_ABS_TOLERANCE, NUMERIC_REL_TOLERANCE, first_json_object

ANSWER_STATUSES = {
    "answerable",
    "insufficient_evidence",
    "contradiction",
    "schema_invalid",
    "protocol_violation",
}
SELECTIVE_CONDITIONS = (
    "G_selective_propagation",
    "G_candidate_unrouted",
    "G_selective_no_gaps",
    "G_all_evidence",
    "G_top_confidence",
    "G_random_k",
    "G_no_routing",
)
TIER_RANK = {"high": 0, "medium": 1, "low": 2, "unknown": 3}


@dataclass(frozen=True)
class SelectiveAnswerRecord:
    schema_valid: bool
    answer_status: str
    answerable: bool
    prediction_parseable: bool
    normalized_prediction: Any
    supporting_fact_ids: list[str]
    missing_fact_keys: list[str]
    contradictory_fact_ids: list[str]
    risk_reason: str
    raw_text: str
    parse_error: str


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _schema_invalid(raw_text: str, parse_error: str) -> SelectiveAnswerRecord:
    return SelectiveAnswerRecord(
        schema_valid=False,
        answer_status="schema_invalid",
        answerable=False,
        prediction_parseable=False,
        normalized_prediction=None,
        supporting_fact_ids=[],
        missing_fact_keys=[],
        contradictory_fact_ids=[],
        risk_reason="schema_invalid",
        raw_text=raw_text,
        parse_error=parse_error,
    )


def parse_selective_answer(raw_text: str, answer_type: str) -> SelectiveAnswerRecord:
    """Parse selective-propagation answers without numeric fallback leakage."""
    try:
        parsed = first_json_object(raw_text)
    except Exception as exc:
        return _schema_invalid(raw_text, str(exc))
    status = str(parsed.get("answer_status", "")).strip().lower()
    if status not in ANSWER_STATUSES or status == "schema_invalid":
        return _schema_invalid(raw_text, f"invalid answer_status: {status}")
    prediction = parsed.get("normalized_prediction")
    if status != "answerable":
        if prediction is not None:
            return _schema_invalid(raw_text, "non-answerable response must use null normalized_prediction")
        return SelectiveAnswerRecord(
            schema_valid=True,
            answer_status=status,
            answerable=False,
            prediction_parseable=False,
            normalized_prediction=None,
            supporting_fact_ids=_string_list(parsed.get("supporting_fact_ids")),
            missing_fact_keys=_string_list(parsed.get("missing_fact_keys")),
            contradictory_fact_ids=_string_list(parsed.get("contradictory_fact_ids")),
            risk_reason=str(parsed.get("risk_reason", "") or ""),
            raw_text=raw_text,
            parse_error="",
        )
    try:
        normalized = normalize_answer(prediction, answer_type)
    except Exception as exc:
        return SelectiveAnswerRecord(
            schema_valid=True,
            answer_status="answerable",
            answerable=True,
            prediction_parseable=False,
            normalized_prediction=None,
            supporting_fact_ids=_string_list(parsed.get("supporting_fact_ids")),
            missing_fact_keys=_string_list(parsed.get("missing_fact_keys")),
            contradictory_fact_ids=_string_list(parsed.get("contradictory_fact_ids")),
            risk_reason=str(parsed.get("risk_reason", "") or ""),
            raw_text=raw_text,
            parse_error=str(exc),
        )
    return SelectiveAnswerRecord(
        schema_valid=True,
        answer_status="answerable",
        answerable=True,
        prediction_parseable=True,
        normalized_prediction=normalized,
        supporting_fact_ids=_string_list(parsed.get("supporting_fact_ids")),
        missing_fact_keys=_string_list(parsed.get("missing_fact_keys")),
        contradictory_fact_ids=_string_list(parsed.get("contradictory_fact_ids")),
        risk_reason=str(parsed.get("risk_reason", "") or ""),
        raw_text=raw_text,
        parse_error="",
    )


def score_selective_answer(record: SelectiveAnswerRecord, reference: Any, answer_type: str, expected_answerability: str) -> dict[str, Any]:
    correct_answer = False
    numeric_error = None
    tolerance = None
    if record.answer_status == "answerable" and record.prediction_parseable:
        if answer_type == "numeric":
            pred = float(record.normalized_prediction)
            ref = float(reference)
            numeric_error = abs(pred - ref)
            tolerance = max(NUMERIC_ABS_TOLERANCE, abs(ref) * NUMERIC_REL_TOLERANCE)
            correct_answer = numeric_error <= tolerance
        else:
            correct_answer = record.normalized_prediction == reference
    correct_abstention = expected_answerability != "answerable" and record.answer_status == expected_answerability
    harmful_propagation = (
        record.answer_status == "answerable"
        and expected_answerability != "answerable"
        and not correct_abstention
    ) or (
        record.answer_status == "answerable"
        and expected_answerability == "answerable"
        and record.prediction_parseable
        and not correct_answer
    )
    return {
        "correct_answer": correct_answer,
        "correct_abstention": correct_abstention,
        "task_success": correct_answer if expected_answerability == "answerable" else correct_abstention,
        "harmful_propagation": harmful_propagation,
        "numeric_error": numeric_error,
        "tolerance": tolerance,
    }


def selected_ids(stress_schema: dict[str, Any]) -> set[str]:
    return set(((stress_schema.get("selected_evidence_set") or {}).get("selected_fact_ids")) or [])


def sorted_by_confidence(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda row: (TIER_RANK.get(str(row.get("reliability_tier")), 3), str(row.get("fact_id"))))


def random_k(candidates: list[dict[str, Any]], k: int, seed: int, stress_id: str) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}:{stress_id}")
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return sorted(shuffled[:k], key=lambda row: str(row.get("fact_id")))


def facts_for_selective_condition(stress_schema: dict[str, Any], condition: str, seed: int, stress_id: str) -> dict[str, Any]:
    candidates = copy.deepcopy(list(stress_schema.get("evidence_candidates") or []))
    gaps = copy.deepcopy(list(stress_schema.get("evidence_gaps") or []))
    selected = copy.deepcopy(stress_schema.get("selected_evidence_set") or {})
    selected_fact_ids = selected_ids(stress_schema)
    selected_gaps = set(selected.get("surfaced_gap_ids") or [])
    required_fact_coverage_complete = selected.get("required_fact_coverage_complete")
    answer_candidate = copy.deepcopy(selected.get("answer_candidate"))
    blocking_gap_keys = set(str(key) for key in selected.get("blocking_gap_keys") or [])
    budget = max(1, len(selected_fact_ids)) if candidates else 0
    routing_policy = selected.get("propagation_policy", "answer_with_selected_facts")
    selected_visible_gaps = [
        row
        for row in gaps
        if row.get("gap_id") in selected_gaps
        and (blocking_gap_keys == set() or str(row.get("key")) in blocking_gap_keys)
        and row.get("needed_for_question", True) is True
        and str(row.get("severity", "blocking")) == "blocking"
    ]
    if condition == "G_selective_propagation":
        facts = [row for row in candidates if row.get("fact_id") in selected_fact_ids]
        visible_gaps = selected_visible_gaps
        visible_route = routing_policy
        visible_answer_candidate = answer_candidate
    elif condition == "G_candidate_unrouted":
        facts = [row for row in candidates if row.get("fact_id") in selected_fact_ids]
        visible_gaps = []
        visible_route = "hidden"
        visible_answer_candidate = answer_candidate
    elif condition == "G_selective_no_gaps":
        facts = [row for row in candidates if row.get("fact_id") in selected_fact_ids]
        visible_gaps = []
        visible_route = "answerable_by_default"
        visible_answer_candidate = None
    elif condition == "G_all_evidence":
        facts = candidates
        visible_gaps = []
        visible_route = "answerable_by_default"
        visible_answer_candidate = None
    elif condition == "G_top_confidence":
        facts = sorted_by_confidence(candidates)[:budget]
        visible_gaps = []
        visible_route = "answerable_by_default"
        visible_answer_candidate = None
    elif condition == "G_random_k":
        facts = random_k(candidates, budget, seed, stress_id)
        visible_gaps = []
        visible_route = "answerable_by_default"
        visible_answer_candidate = None
    elif condition == "G_no_routing":
        facts = [row for row in candidates if row.get("fact_id") in selected_fact_ids]
        visible_gaps = selected_visible_gaps
        visible_route = "hidden"
        visible_answer_candidate = None
    else:
        raise ValueError(f"unsupported selective condition: {condition}")
    return {
        "schema_version": "m1m2_selective_prompt_facts_v1",
        "condition": condition,
        "input_mode": "evidence_only",
        "routing_policy": visible_route,
        "measured_facts": facts,
        "evidence_gaps": visible_gaps,
        "answer_candidate": visible_answer_candidate,
        "required_fact_coverage_complete": required_fact_coverage_complete,
        "blocking_gap_keys": sorted(blocking_gap_keys),
        "suppressed_fact_ids": list(selected.get("suppressed_fact_ids") or []),
        "selected_fact_count": len(facts),
        "expected_answerability": stress_schema.get("expected_answerability"),
        "source_schema_condition": stress_schema.get("condition"),
        "stress_hash": stress_schema.get("stress_hash"),
    }


def selective_answer_prompt(question: str, facts: dict[str, Any], condition: str, perturbation_family: str) -> str:
    schema = {
        "answer_status": "answerable|insufficient_evidence|contradiction|schema_invalid|protocol_violation",
        "normalized_prediction": "final answer string/number when answerable, else null",
        "supporting_fact_ids": ["fact_id"],
        "missing_fact_keys": ["key"],
        "contradictory_fact_ids": ["fact_id"],
        "risk_reason": "short machine-readable text",
    }
    return (
        "Return JSON only using the exact schema below. Do not include markdown.\n"
        f"Schema: {json.dumps(schema, ensure_ascii=False)}\n"
        "This is an evidence-only selective propagation test. No chart image is available. "
        "Use only the supplied measured evidence JSON, evidence gaps, and routing policy. "
        "If the evidence is insufficient or contradictory, set answer_status accordingly and set normalized_prediction to null. "
        "Candidate-guided rule: if routing_policy is answer_with_candidate_and_selected_facts and answer_candidate is present with no visible blocking evidence gap, you must set answer_status to answerable, return answer_candidate.normalized_prediction exactly as normalized_prediction, and cite answer_candidate.supporting_fact_ids. "
        "Under the candidate-guided rule, do not invent missing facts, re-audit hidden facts, or override the candidate from general chart knowledge. "
        "Do not infer missing visual facts.\n"
        f"Condition: {condition}\n"
        f"Perturbation family: {perturbation_family}\n"
        f"Question: {question}\n"
        f"Evidence JSON: {json.dumps(facts, ensure_ascii=False, sort_keys=True)}"
    )


def estimate_tokens_for_prompt(prompt: str) -> int:
    return max(1, (len(prompt) + 3) // 4)


def prompt_hash(question: str, facts: dict[str, Any], condition: str, perturbation_family: str) -> str:
    return stable_payload_hash({"prompt": selective_answer_prompt(question, facts, condition, perturbation_family)})


def answer_record_to_dict(record: SelectiveAnswerRecord) -> dict[str, Any]:
    return asdict(record)
