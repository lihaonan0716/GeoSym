"""Shared reasoning pipeline: condition dispatch, Judge LLM, and common helpers.

Single source of truth for logic shared across:
  - app/reasoning.py         (Math Reasoning tab — single image UI)

Extracted to eliminate code duplication and ensure identical behaviour
(especially Judge LLM evaluation) in the single-image reasoning entry point.
"""
from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.helpers import _root, _images_dir, _get_saved_key, _create_llm_adapter
from generator.config_loader import load_config
from evaluator.reasoning.parser import parse_comprehensive_answer

logger = logging.getLogger(__name__)

# ── Condition label ↔ internal mode mapping ────────────────────────────

CONDITION_MAP: Dict[str, str] = {
    "E0": "direct_answer",
    "E1": "direct_cot",
    "E2": "direct_self_consistency",
    "E3": "ocr_answer",
    "E4": "ocr_cot",
    # `E5` (full_geosym) retired 2026-09-19 with M1/M2. `E5_raw` is GeoSym
    # itself now; never rename the key -- see CLAUDE.md "Experiment matrix".
    "E5_raw": "raw_geosym",
}

MODE_TO_SHORT: Dict[str, str] = {v: k for k, v in CONDITION_MAP.items()}

CONDITION_ORDER: List[str] = list(CONDITION_MAP.keys())

_PROMPT_DIRECT = _root / "prompts" / "reasoning_direct_single_prompt.txt"
_PROMPT_DIRECT_COT = _root / "prompts" / "reasoning_direct_cot_prompt.txt"
_PROMPT_OCR = _root / "prompts" / "reasoning_ocr_single_prompt.txt"
_PROMPT_OCR_COT = _root / "prompts" / "reasoning_ocr_cot_prompt.txt"

SELF_CONSISTENCY_RUNS = 3


def resolve_conditions(raw: str) -> List[str]:
    """Parse comma-separated condition labels into internal mode keys."""
    labels = [s.strip() for s in raw.split(",") if s.strip()]
    modes: List[str] = []
    for label in labels:
        mode = CONDITION_MAP.get(label)
        if mode is None:
            if label in CONDITION_MAP.values():
                mode = label
            else:
                logger.warning("Unknown condition label '%s', skipping", label)
                continue
        modes.append(mode)
    return modes


# ── Image path resolution ──────────────────────────────────────────────

def resolve_image_path(image_name: str) -> Optional[Path]:
    """Resolve image name to actual file path, checking subset dirs."""
    for suffix in (".png", ".jpg"):
        p = _images_dir / f"{image_name}{suffix}"
        if p.exists():
            return p
        stem, has_index = image_name.rsplit("_", 1) if "_" in image_name else (image_name, "")
        if has_index:
            subset_id = stem
            p = _images_dir / subset_id / f"{image_name}{suffix}"
            if p.exists():
                return p
    return None


# ── GeoSym coordinate extraction ──────────────────────────────────────

def extract_geosym_coords(
    model_name: str,
    api_key: str,
    img_path: Path,
    gt: dict,
    *,
    gemini_enabled: bool = False,
) -> Tuple[list, Optional[dict], bool, Optional[str]]:
    """Run GeoSym coordinate extraction via RecognizerAdapter.

    Returns ``(coords_list, visual_facts_or_None, success_flag, failure_reason)``.

    Visual facts are no longer computed by the CV pipeline.  Use
    :func:`infer_visual_facts_with_llm` to obtain them from an LLM.
    """
    from evaluator.adapters.recognizer import RecognizerAdapter

    try:
        recognizer = RecognizerAdapter()
        detected, result, _annotated = recognizer.recognize_intersections(
            str(img_path), gt, gemini_enabled=gemini_enabled,
        )
        success = result.get("success", False) if isinstance(result, dict) else bool(detected)
        reason = None
        if not success and isinstance(result, dict):
            error = result.get("error")
            if isinstance(error, str) and error.strip():
                reason = error.strip()
            elif error:
                reason = repr(error)
            else:
                reason = "recognizer_returned_success_false_without_error"

        return detected or [], None, bool(success), reason
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "GeoSym extraction failed for %s: %s",
            getattr(img_path, "name", img_path), exc,
        )
        return [], None, False, reason


# ── LLM-based visual fact inference ──────────────────────────────────

_VF_INFERENCE_PROMPT_PATH = _root / "prompts" / "visual_fact_inference_prompt.txt"


# ── Extrema channel feature flag ──────────────────────────────────────
#
# The extrema slot of `compute_visual_facts_from_coords` has always been
# hard-coded to `[]`, so no extrema evidence ever reached the prompts even
# though `evaluator.reasoning.intervention` contains a working candidate
# generator.  This flag wires that generator in *without* changing any
# recorded behaviour by default.
#
# Default OFF, deliberately: the channel was absent from every historical
# reported run, and turning it on is therefore a new experiment rather than a
# retroactive fix.  Every historical number was produced with it off, and
# keeping it off keeps the pipeline output structurally identical to those runs.
EXTREMA_CHANNEL_ENV_VAR = "GEOSYM_ENABLE_EXTREMA_CHANNEL"


def extrema_channel_enabled() -> bool:
    """Whether the extrema evidence channel is enabled (env-var flag, default off)."""
    import os

    return os.environ.get(EXTREMA_CHANNEL_ENV_VAR, "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


# Conditions whose prompt content depends on the flag. E5 was the only one,
# because it was the sole condition that received `visual_facts`: E5_raw passes
# `visual_facts=None` and E0-E4 never see GeoSym output at all.
#
# E5 retired on 2026-09-19, so NO LIVE CONDITION consumes the extrema channel.
# The entry is kept because caches on disk still hold E5 records and must stay
# correctly invalidated. Do not read this set as "the extrema channel is wired
# into the current pipeline" -- it is not; re-wiring it into E5_raw would be a
# prompt change to the main condition and needs its own decision.
EXTREMA_CHANNEL_CONDITIONS: frozenset = frozenset({"E5"})


def compute_visual_facts_from_coords(coords: list) -> Optional[Dict[str, Any]]:
    """Compute visual facts deterministically from M0 coordinate points.

    Replaces LLM-based inference with computation-grounded derivation,
    ensuring that M1 facts are as reliable as M0 coordinates themselves.

    The extrema slot is populated only when :func:`extrema_channel_enabled`
    returns True; otherwise it stays ``[]`` as it always has.  The returned
    field set, field order, and every other category are identical either way.
    """
    if not coords or len(coords) < 2:
        return None

    # Sort by x
    sorted_pts = sorted(coords, key=lambda c: (c[0], c[1]))

    # --- X-intercepts: only exact coordinate points with y=0 ---
    x_intercepts = []
    for x, y in sorted_pts:
        if y == 0:
            val = round(x, 2)
            if val not in x_intercepts:
                x_intercepts.append(val)

    # --- Y-intercept: only exact coordinate point at x=0 ---
    y_intercept = None
    for x, y in sorted_pts:
        if x == 0:
            y_intercept = round(y, 2)
            break

    # --- Extrema: gated on the feature flag (see above) ---
    # Flag off  -> `[]`, matching every run recorded so far.
    # Flag on   -> filtered candidates from the single shared implementation.
    if extrema_channel_enabled():
        from evaluator.reasoning.intervention import infer_extrema_candidates

        extrema = infer_extrema_candidates(coords)
    else:
        extrema = []

    # --- Monotonicity: group consecutive slopes into intervals ---
    monotonicity = []
    if len(sorted_pts) >= 2:
        current_dir = None
        interval_start = sorted_pts[0][0]
        for i in range(1, len(sorted_pts)):
            dy = sorted_pts[i][1] - sorted_pts[i - 1][1]
            dx = sorted_pts[i][0] - sorted_pts[i - 1][0]
            if dx == 0:
                continue
            if dy > 0:
                direction = "increasing"
            elif dy < 0:
                direction = "decreasing"
            else:
                direction = current_dir  # flat: continue current

            if direction != current_dir and current_dir is not None:
                monotonicity.append({
                    "interval": [round(interval_start, 2), round(sorted_pts[i - 1][0], 2)],
                    "direction": current_dir,
                })
                interval_start = sorted_pts[i - 1][0]
            current_dir = direction

        if current_dir is not None:
            monotonicity.append({
                "interval": [round(interval_start, 2), round(sorted_pts[-1][0], 2)],
                "direction": current_dir,
            })

    return {
        "intercepts": {
            "x_intercepts": x_intercepts,
            "y_intercept": y_intercept,
        },
        "extrema": extrema,
        "monotonicity": monotonicity,
        "source": "computation_grounded",
    }


def infer_visual_facts_with_llm(
    adapter,
    img_path: Path,
    coords: list,
) -> Optional[Dict[str, Any]]:
    """Infer visual facts (intercepts, extrema, monotonicity) via LLM.

    The LLM receives the chart image together with the coordinate points
    detected by M0 and reasons about the mathematical properties of the
    curve.  This replaces the former algorithmic ``compute_visual_facts``
    and better demonstrates a *visual-enhancement* fusion paradigm.

    Returns a dict matching the ``visual_facts`` schema, or ``None`` on
    failure (downstream code will fall back to ``infer_chart_facts``).
    """
    import json as _json
    import re as _re

    if not coords:
        logger.warning("infer_visual_facts_with_llm: empty coords, skipping")
        return None

    # Format coordinates
    coords_text = ", ".join(f"({x}, {y})" for x, y in coords)

    # Load and fill prompt template
    try:
        prompt_tpl = _load_prompt(_VF_INFERENCE_PROMPT_PATH)
    except FileNotFoundError:
        logger.error("Visual-fact inference prompt not found: %s", _VF_INFERENCE_PROMPT_PATH)
        return None
    prompt = prompt_tpl.replace("{coords_text}", coords_text)

    # Call LLM
    try:
        raw = adapter.call_with_prompt(str(img_path), prompt)
    except Exception as exc:
        logger.warning("LLM visual-fact inference call failed: %s", exc)
        return None

    # Parse JSON from response
    try:
        src = raw.strip()
        m = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", src, _re.DOTALL)
        if m:
            src = m.group(1)
        else:
            start = src.find("{")
            end = src.rfind("}")
            if start != -1 and end != -1 and end > start:
                src = src[start:end + 1]
        data = _json.loads(src)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object")
    except Exception as exc:
        logger.warning("Failed to parse LLM visual-fact response: %s", exc)
        return None

    # Validate schema
    intercepts = data.get("intercepts")
    if not isinstance(intercepts, dict):
        intercepts = {}
    extrema = data.get("extrema")
    if not isinstance(extrema, list):
        extrema = []
    monotonicity = data.get("monotonicity")
    if not isinstance(monotonicity, list):
        monotonicity = []

    return {
        "intercepts": {
            "x_intercepts": intercepts.get("x_intercepts", []),
            "y_intercept": intercepts.get("y_intercept"),
        },
        "extrema": extrema,
        "monotonicity": monotonicity,
        "source": "llm_inferred",
    }


def _load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _inject_prediction_placeholders(prompt: str, gt_data: dict) -> str:
    pred_targets = gt_data.get("prediction_targets", {})
    rate_info = gt_data.get("rate_of_change", {})

    xs_str = ", ".join(str(x) for x in pred_targets.keys()) if pred_targets else "N/A"
    prompt = prompt.replace("{prediction_xs}", xs_str)

    if pred_targets:
        example_json = "{ " + ", ".join(
            f'"{x}": value{idx + 1}' for idx, x in enumerate(pred_targets.keys())
        ) + " }"
    else:
        example_json = "{}"
    prompt = prompt.replace("{prediction_json_example}", example_json)

    prompt = prompt.replace("{rate_x1}", str(rate_info.get("x1", "N/A")))
    prompt = prompt.replace("{rate_x2}", str(rate_info.get("x2", "N/A")))
    return prompt


def _format_ocr_evidence(ocr_coords: list) -> str:
    if not ocr_coords:
        return "No OCR coordinates were extracted."
    return "Detected coordinate candidates: " + ", ".join(f"({x},{y})" for x, y in ocr_coords)


def _build_single_step_pipeline(
    *,
    step_name: str,
    prompt: str,
    raw_response: str,
    parsed: dict,
    image_mode: str,
    extra: Optional[dict] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "pipeline_type": "single_step",
        "image_mode": image_mode,
        "steps": {
            step_name: {
                "prompt": prompt,
                "raw": raw_response,
                "parsed": parsed,
            }
        },
        "final_parsed": parsed,
        "judge_chain_text": f"=== {step_name} ===\n{raw_response}",
    }
    if extra:
        payload.update(extra)
    return payload


def _canonical_answer_key(parsed: dict) -> str:
    normalized = {
        "expression": str(parsed.get("expression", "")).strip(),
        "domain": str(parsed.get("domain", "")).strip(),
        "range": str(parsed.get("range", "")).strip(),
        "coordinates": str(parsed.get("coordinates", "")).strip(),
        "predictions": parsed.get("predictions", {}) or {},
        "rate_of_change": parsed.get("rate_of_change"),
        "intercepts": parsed.get("intercepts", {}) or {},
        "extrema": parsed.get("extrema", []) or [],
        "monotonicity": parsed.get("monotonicity", []) or [],
        "symmetry": str(parsed.get("symmetry", "")).strip(),
    }
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def _run_single_step_reasoning(
    adapter,
    img_path: Path,
    gt_data: dict,
    *,
    prompt_path: Path,
    step_name: str,
    image_mode: str,
    ocr_evidence: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = _inject_prediction_placeholders(_load_prompt(prompt_path), gt_data)
    if ocr_evidence is not None:
        prompt = prompt.replace("{ocr_evidence}", ocr_evidence)
    raw = adapter.call_with_prompt(str(img_path), prompt)
    parsed = parse_comprehensive_answer(raw)
    return _build_single_step_pipeline(
        step_name=step_name,
        prompt=prompt,
        raw_response=raw,
        parsed=parsed,
        image_mode=image_mode,
        extra={
            "finish_reasons": {step_name: getattr(adapter, "last_finish_reason", None)},
            "response_metadata": {step_name: getattr(adapter, "last_response_metadata", {})},
        },
    )


def _run_self_consistency_reasoning(adapter, img_path: Path, gt_data: dict) -> Dict[str, Any]:
    prompt = _inject_prediction_placeholders(_load_prompt(_PROMPT_DIRECT_COT), gt_data)
    samples: List[dict] = []
    counter: Counter[str] = Counter()

    for sample_idx in range(SELF_CONSISTENCY_RUNS):
        sampled_prompt = (
            f"{prompt}\n\n"
            f"Self-consistency sample id: {sample_idx + 1}. "
            "Independently solve the task and still output only the final JSON object."
        )
        raw = adapter.call_with_prompt(str(img_path), sampled_prompt)
        parsed = parse_comprehensive_answer(raw)
        key = _canonical_answer_key(parsed)
        counter[key] += 1
        samples.append({
            "prompt": sampled_prompt,
            "raw": raw,
            "parsed": parsed,
            "vote_key": key,
            "finish_reason": getattr(adapter, "last_finish_reason", None),
            "response_metadata": getattr(adapter, "last_response_metadata", {}),
        })

    best_key = counter.most_common(1)[0][0] if counter else ""
    chosen = next((item for item in samples if item["vote_key"] == best_key), samples[0])
    return {
        "pipeline_type": "self_consistency",
        "image_mode": "direct_image",
        "steps": {
            "SelfConsistency": {
                "prompt": prompt,
                "raw": chosen["raw"],
                "parsed": chosen["parsed"],
                "finish_reason": chosen.get("finish_reason"),
                "response_metadata": chosen.get("response_metadata", {}),
            }
        },
        "samples": samples,
        "self_consistency_votes": counter,
        "final_parsed": chosen["parsed"],
        "judge_chain_text": (
            f"Aggregated self-consistency final answer (votes={counter.get(best_key, 0)}/{len(samples)}):\n"
            f"{chosen['raw']}"
        ),
    }


# ── Unified condition dispatcher ───────────────────────────────────────

def run_single_condition(
    adapter,
    img_path: Path,
    gt_data: dict,
    mode: str,
    *,
    geosym_coords: Optional[list] = None,
    visual_facts: Optional[dict] = None,
    ocr_baseline=None,
    plain_baseline=None,
) -> Dict[str, Any]:
    """Run one condition from the unified ten-condition experiment matrix."""
    from app.reasoning import _run_reasoning_flow

    if mode == "direct_answer":
        return _run_single_step_reasoning(
            adapter,
            img_path,
            gt_data,
            prompt_path=_PROMPT_DIRECT,
            step_name="DirectQA",
            image_mode="direct_image",
        )

    if mode == "direct_cot":
        return _run_single_step_reasoning(
            adapter,
            img_path,
            gt_data,
            prompt_path=_PROMPT_DIRECT_COT,
            step_name="DirectQA-CoT",
            image_mode="direct_image",
        )

    if mode == "direct_self_consistency":
        return _run_self_consistency_reasoning(adapter, img_path, gt_data)

    if mode == "ocr_answer":
        if ocr_baseline is None:
            from evaluator.baselines.ocr_rule_baseline import OCRRuleBaseline
            ocr_baseline = OCRRuleBaseline()
        ocr_coords = ocr_baseline.extract_coordinates(str(img_path))
        return _run_single_step_reasoning(
            adapter,
            img_path,
            gt_data,
            prompt_path=_PROMPT_OCR,
            step_name="OCRRulesQA",
            image_mode="ocr_assisted",
            ocr_evidence=_format_ocr_evidence(ocr_coords or []),
        )

    if mode == "ocr_cot":
        if ocr_baseline is None:
            from evaluator.baselines.ocr_rule_baseline import OCRRuleBaseline
            ocr_baseline = OCRRuleBaseline()
        ocr_coords = ocr_baseline.extract_coordinates(str(img_path))
        return _run_single_step_reasoning(
            adapter,
            img_path,
            gt_data,
            prompt_path=_PROMPT_OCR_COT,
            step_name="OCRRules-CoT",
            image_mode="ocr_assisted",
            ocr_evidence=_format_ocr_evidence(ocr_coords or []),
        )

    # `full_geosym` (condition E5) was the sole production entry point into
    # M1/M2 and retired with them on 2026-09-19. `raw_geosym` (E5_raw) is
    # GeoSym itself now; it never touched that path.
    if mode == "raw_geosym":
        return _run_reasoning_flow(
            adapter, img_path, gt_data,
            geosym_coords=geosym_coords or [],
            visual_facts=None,
            l1_prompt_path=_root / "prompts" / "ablation_coords_only_L1_prompt.txt",
        )

    raise ValueError(f"Unknown reasoning mode: {mode}")


# ── Layer config ───────────────────────────────────────────────────────

def load_layer_config() -> dict:
    """Load main experiment scoring-group config from YAML."""
    from evaluator.metrics.reasoning import DEFAULT_LAYER_CONFIG
    cfg = load_config()
    return cfg.get("reasoning", {}).get("layers", DEFAULT_LAYER_CONFIG)


# ── Judge LLM ──────────────────────────────────────────────────────────

_JUDGE_PROMPT_PATH = _root / "prompts" / "judge_prompt.txt"


def load_judge_config() -> dict:
    """Load judge configuration from YAML."""
    cfg = load_config()
    return cfg.get("reasoning", {}).get("judge", {"enabled": False})


def build_gt_summary(gt_data: dict) -> str:
    """Build a ground-truth summary string for the Judge prompt."""
    lines = [
        f"Expression: {gt_data.get('expression', 'N/A')}",
        f"Domain: {gt_data.get('domain', 'N/A')}",
        f"Range: {gt_data.get('range', 'N/A')}",
        f"Coordinates: {gt_data.get('coordinates', 'N/A')}",
    ]
    pred_tgt = gt_data.get("prediction_targets", {})
    if pred_tgt:
        lines.append(f"Prediction targets: {pred_tgt}")
    roc = gt_data.get("rate_of_change", {})
    if roc:
        lines.append(f"Rate of change: {roc.get('gt_value', 'N/A')} "
                      f"(x∈[{roc.get('x1', '?')},{roc.get('x2', '?')}])")
    intercepts = gt_data.get("intercepts", {})
    if intercepts:
        lines.append(f"Intercepts: {intercepts}")
    extrema = gt_data.get("extrema", [])
    if extrema:
        lines.append(f"Extrema: {extrema}")
    mono = gt_data.get("monotonicity", [])
    if mono:
        lines.append(f"Monotonicity: {mono}")
    sym = gt_data.get("symmetry", "")
    if sym:
        lines.append(f"Symmetry: {sym}")
    return "\n".join(lines)


def build_chain_text(pipeline_result: dict) -> str:
    """Build Judge-visible mathematical inference text for any pipeline style."""
    override = pipeline_result.get("judge_chain_text")
    if override:
        return override
    parsed = pipeline_result.get("final_parsed", {}) or {}
    lines = [
        f"Expression: {parsed.get('expression', '')}",
        f"Domain: {parsed.get('domain', '')}",
        f"Range: {parsed.get('range', '')}",
        f"Inference explanation: {parsed.get('inference_explanation', '')}",
    ]
    support_keys = ("coordinates", "intercepts", "extrema", "monotonicity", "symmetry")
    for key in support_keys:
        val = parsed.get(key)
        if val not in (None, "", [], {}):
            lines.append(f"{key}: {val}")
    return "\n".join(lines)


def ensure_judge_adapter(
    judge_model_name: str,
    judge_api_key: str,
    fallback_model_name: str,
    fallback_api_key: str,
):
    """Create a Judge adapter, falling back to the main reasoning model
    when no dedicated Judge model is configured."""
    judge_cfg = load_judge_config()
    if not judge_cfg.get("enabled", False):
        return None

    j_name = (judge_model_name or "").strip()
    j_key = (judge_api_key or "").strip()

    if not j_name:
        j_name = fallback_model_name
    if not j_key:
        j_key = _get_saved_key(j_name)
    if not j_key:
        j_key = fallback_api_key

    if j_name and j_key:
        adapter = _create_llm_adapter(j_name, j_key)
        if adapter is not None:
            logger.info("Judge adapter ready: %s", j_name)
            return adapter

    logger.warning("Failed to create Judge adapter (model=%s)", j_name)
    return None


def run_judge(
    judge_adapter,
    img_path: Path,
    gt_data: dict,
    pipeline_result: dict,
    *,
    max_retries: int = 5,
) -> dict:
    """Invoke the Judge LLM to evaluate mathematical inference quality.

    The underlying ``call_with_prompt`` already retries transient API errors.
    This outer retry handles higher-level failures (unparseable response, etc.)
    and gives the LLM another chance to produce valid output.

    Returns the output of ``evaluate_judge_response``.
    """
    from evaluator.reasoning.parser import parse_judge_response
    from evaluator.metrics.reasoning import evaluate_judge_response
    from evaluator.adapters.base import is_transient_error

    gt_summary = build_gt_summary(gt_data)
    chain_text = build_chain_text(pipeline_result)

    if _JUDGE_PROMPT_PATH.exists():
        prompt_tpl = _JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    else:
        prompt_tpl = (
            "Evaluate the model's mathematical inference against ground truth.\n"
            "Ground truth:\n{ground_truth_summary}\n"
            "Model response:\n{model_response}\n"
            "Output JSON with 4 dimensions (0-5 each): factual_accuracy, "
            "inference_quality, reasoning_coherence, evidence_support, comments."
        )

    prompt = prompt_tpl.replace("{ground_truth_summary}", gt_summary)
    prompt = prompt.replace("{model_response}", chain_text)

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = judge_adapter.call_with_prompt(str(img_path), prompt)
            parsed = parse_judge_response(raw)
            result = evaluate_judge_response(parsed)
            result["raw_judge_response"] = raw
            result["finish_reason"] = getattr(judge_adapter, "last_finish_reason", None)
            result["response_metadata"] = getattr(judge_adapter, "last_response_metadata", {})
            return result
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries and is_transient_error(exc):
                delay = min(3.0 * (2 ** (attempt - 1)), 30.0)
                delay *= 0.5 + random.random()
                logger.warning(
                    "Judge attempt %d/%d failed (%s: %s), retry in %.1fs",
                    attempt, max_retries, type(exc).__name__,
                    str(exc)[:200], delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Judge attempt %d/%d final failure: %s",
                    attempt, max_retries, exc,
                )
                break

    raise RuntimeError(f"Judge failed after {attempt} attempts: {last_exc}")
