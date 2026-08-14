"""Shared schema helpers for single-image reasoning result records.

This module centralizes the persisted JSON contract so future tools
(for example, a private batch dashboard) can reuse the same structure
without re-implementing ad-hoc field assembly.
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.reasoning_pipeline import (
    CONDITION_MAP,
    CONDITION_ORDER,
    EXTREMA_CHANNEL_CONDITIONS,
)

SCHEMA_VERSION = 2
RUN_TYPE_SINGLE_IMAGE_REASONING = "single_image_reasoning"


def build_condition_record(
    *,
    condition_id: str,
    label: str,
    status: str,
    cached: bool,
    elapsed_ms: float,
    retry_count: int,
    error: dict | None,
    eval_result: dict | None,
    parsed: dict | None,
    judge: dict | None,
    pipeline: dict | None,
    md: str,
    extrema_channel_enabled: bool = False,
) -> Dict[str, Any]:
    """Build one persisted condition record.

    ``extrema_channel_enabled`` is the flag state that produced *this* condition,
    which is not necessarily the state of the process writing the file: a run may
    reuse cached conditions from an earlier run made under the opposite setting.
    Recording it per condition is what makes such a mixed record readable.
    """
    return {
        "condition_id": condition_id,
        "label": label,
        "mode": CONDITION_MAP.get(condition_id, ""),
        "status": status,
        "cached": cached,
        "extrema_channel_enabled": extrema_channel_enabled,
        "elapsed_ms": elapsed_ms,
        "retry_count": retry_count,
        "error": error,
        "summary": {
            "total_score": eval_result.get("total_score") if eval_result else None,
            "dimension_scores": eval_result.get("dimension_scores") if eval_result else None,
            "judge_inference_score": (
                eval_result.get("sub_scores", {}).get("judge_inference_score") if eval_result else None
            ),
        },
        "eval": eval_result,
        "parsed": parsed,
        "judge": judge,
        "pipeline": pipeline,
        "md": md,
    }


def condition_extrema_flag(cond_data: Dict[str, Any] | None) -> bool:
    """Read the extrema-channel state a cached condition was produced under.

    Records written before the flag existed have no such field.  ``False`` is the
    correct default rather than a guess: the extrema slot was hard-coded to ``[]``
    at the time, so every one of those runs was flag-off by construction.

    Deliberately *not* added to :func:`validate_reasoning_record`'s required
    fields — making it mandatory would fail validation on every archived record
    and silently discard caches that cost real API spend to produce.

    Only a real JSON boolean counts.  Truthiness would be actively dangerous
    here: ``bool("false")`` is ``True``, so a record hand-edited or produced by
    some future writer that stored the raw env-var string would be read as
    flag-ON, and a flag-off run would then judge every such condition stale and
    re-run it.  Anything that is not a bool is treated as absent.
    """
    if not isinstance(cond_data, dict):
        return False
    flag = cond_data.get("extrema_channel_enabled")
    if not isinstance(flag, bool):
        return False
    return flag


def stale_extrema_conditions(
    cached: Dict[str, Any] | None,
    current_flag: bool,
) -> set:
    """Conditions cached under the opposite extrema-channel state.

    Only flag-sensitive conditions can go stale (see
    :data:`app.reasoning_pipeline.EXTREMA_CHANNEL_CONDITIONS`); the rest never see
    GeoSym facts, so reusing them across a flag change is sound and saves spend.

    Reusing a stale condition is not merely untidy: the resulting record would
    carry an E5 produced with the channel off while claiming the opposite, which
    breaks experiment provenance in a way no downstream reader could detect.
    """
    if not isinstance(cached, dict):
        return set()
    conds = cached.get("conditions") or {}
    return {
        ck for ck in EXTREMA_CHANNEL_CONDITIONS
        if ck in conds and condition_extrema_flag(conds.get(ck)) != current_flag
    }


def compute_overall_status(condition_records: Dict[str, Dict[str, Any]]) -> str:
    """Summarize run status from all persisted condition records."""
    statuses = [cond.get("status") for cond in condition_records.values()]
    if statuses and all(status == "success" for status in statuses):
        return "success"
    if any(status == "success" for status in statuses):
        return "partial_success"
    return "failed"


def build_reasoning_record(
    *,
    image_name: str,
    image_path: str,
    reasoning_model: str,
    judge_model: str,
    started_at: str,
    finished_at: str,
    duration_ms: float,
    requested_conditions: List[str],
    run_settings: Dict[str, Any],
    ground_truth_snapshot: dict,
    score_table_markdown: str,
    cached_conditions: List[str],
    geosym_artifacts: dict,
    conditions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the full persisted single-image reasoning record."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_type": RUN_TYPE_SINGLE_IMAGE_REASONING,
        "image_name": image_name,
        "image_path": image_path,
        "reasoning_model": reasoning_model,
        "judge_model": judge_model,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "status": compute_overall_status(conditions),
        "condition_order": list(CONDITION_ORDER),
        "requested_conditions": requested_conditions,
        "run_settings": run_settings,
        "ground_truth_snapshot": ground_truth_snapshot,
        "summary": {
            "score_table_markdown": score_table_markdown,
            "cached_conditions": cached_conditions,
            "condition_totals": {
                ck: (conditions.get(ck, {}).get("summary", {}) or {}).get("total_score")
                for ck in CONDITION_ORDER
            },
        },
        "geosym_artifacts": geosym_artifacts,
        "conditions": conditions,
    }


def validate_reasoning_record(record: Dict[str, Any]) -> List[str]:
    """Return human-readable schema validation errors."""
    errors: List[str] = []
    required_top = [
        "schema_version",
        "run_type",
        "image_name",
        "image_path",
        "reasoning_model",
        "judge_model",
        "started_at",
        "finished_at",
        "duration_ms",
        "status",
        "condition_order",
        "requested_conditions",
        "run_settings",
        "ground_truth_snapshot",
        "summary",
        "geosym_artifacts",
        "conditions",
    ]
    for key in required_top:
        if key not in record:
            errors.append(f"missing top-level field: {key}")

    if record.get("run_type") not in {RUN_TYPE_SINGLE_IMAGE_REASONING}:
        errors.append(f"invalid run_type: {record.get('run_type')}")

    conditions = record.get("conditions")
    if not isinstance(conditions, dict):
        errors.append("conditions must be a dict")
        return errors

    for cond_key in CONDITION_ORDER:
        cond = conditions.get(cond_key)
        if cond is None:
            errors.append(f"missing condition record: {cond_key}")
            continue
        for field in (
            "condition_id",
            "label",
            "mode",
            "status",
            "cached",
            "elapsed_ms",
            "retry_count",
            "error",
            "summary",
            "eval",
            "parsed",
            "judge",
            "pipeline",
            "md",
        ):
            if field not in cond:
                errors.append(f"{cond_key} missing field: {field}")

    summary = record.get("summary")
    if isinstance(summary, dict):
        for field in ("score_table_markdown", "cached_conditions", "condition_totals"):
            if field not in summary:
                errors.append(f"summary missing field: {field}")
    else:
        errors.append("summary must be a dict")

    return errors
