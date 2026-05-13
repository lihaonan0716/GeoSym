"""
GeoSym structured visual-symbolic fusion module.

This module implements the fusion bridge between GeoSym's visual coordinate
extraction and downstream LLM reasoning.  It converts reliable coordinates
and visually-computed facts into structured representations for the
experiment to compare different *fusion levels*:

    E3 / E4 – OCR + rule evidence baselines
    E5      – GeoSym fusion pipeline

The retired factorized naming is no longer part of the active project narrative.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


DEFAULT_INTERVENTION_MODE = "visual_facts_flat"

INTERVENTION_PROFILES: Dict[str, Dict[str, Any]] = {
    # Active GeoSym fusion modes used by the current six-condition matrix.
    "visual_facts_flat": {
        "label": "GeoSym Structured Fusion (Flat Facts)",
        "role": "experimental",
        "visual_fact_keys": ["intercepts", "extrema", "monotonicity"],
        "tiered_injection": False,
        "llm_verify": False,
        "description": "GeoSym coordinates plus structured visual facts injected in a flat format.",
    },
    "visual_facts_full": {
        "label": "Full GeoSym Pipeline",
        "role": "main",
        "visual_fact_keys": ["intercepts", "extrema", "monotonicity"],
        "tiered_injection": True,
        "llm_verify": True,
        "description": "Full pipeline: LLM error detection + completeness-aware tiered injection.",
    },
}

BASELINE_PROFILES: Dict[str, Dict[str, str]] = {
    "ocr_baseline": {
        "label": "OCR + Rules Baseline",
        "role": "baseline",
        "prompt_template": "reasoning_L1_baseline_ocr_prompt.txt",
        "description": "OCR text extraction + rule-based coordinate parsing, plain injection.",
    },
    "plain_table": {
        "label": "Plain Coordinate Table Baseline",
        "role": "baseline",
        "prompt_template": "reasoning_L1_baseline_table_prompt.txt",
        "description": "GeoSym-extracted coordinates injected as a plain list without structured framing.",
    },
}


def normalize_coords(coords: Iterable[Any]) -> List[Tuple[int, int]]:
    """Normalise mixed coordinate representations into sorted unique integer pairs."""
    result: List[Tuple[int, int]] = []
    seen = set()
    for item in coords or []:
        pair = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                pair = (int(round(float(item[0]))), int(round(float(item[1]))))
            except (TypeError, ValueError):
                pair = None
        elif isinstance(item, dict):
            try:
                x_val = item.get("coordinate_x", item.get("x"))
                y_val = item.get("coordinate_y", item.get("y"))
                pair = (int(round(float(x_val))), int(round(float(y_val))))
            except (TypeError, ValueError):
                pair = None
        if pair is not None and pair not in seen:
            seen.add(pair)
            result.append(pair)
    return sorted(result, key=lambda p: (p[0], p[1]))


def get_intervention_profile(mode: str | None) -> Dict[str, str]:
    """Return a known intervention profile, falling back to the default one."""
    key = (mode or DEFAULT_INTERVENTION_MODE).strip().lower()
    if key not in INTERVENTION_PROFILES:
        key = DEFAULT_INTERVENTION_MODE
    profile = dict(INTERVENTION_PROFILES[key])
    profile["mode"] = key
    return profile


_FILTER_PROMPT = """You are verifying candidate visual facts for a math function graph.

You are given:
1. The graph image.
2. Reliable coordinate evidence extracted by GeoSym.
3. Candidate visual facts extracted from the same image.

Your job is NOT to compute new facts. Your job is ONLY to decide which candidate
facts are visually supported and useful enough to keep.

STRICT RULES:
- Do NOT invent new numeric values.
- Do NOT modify candidate values.
- Only keep or reject existing indexed candidates.
- If a candidate is uncertain, reject it.
- Be conservative: precision matters more than recall.
- Use the image as the ground truth; use the reliable coordinates as anchors.

Reliable coordinates:
{coords_text}

Candidate visual facts (indexed):
{candidate_facts}

Return ONLY JSON in this exact schema:
{
  "keep": {
    "x_intercepts": [0, 2],
    "keep_y_intercept": false,
    "extrema": [1],
    "monotonicity": [0]
  },
  "notes": [
    "short reason 1",
    "short reason 2"
  ]
}

Meaning:
- x_intercepts: indices of x-intercept candidates to keep
- keep_y_intercept: whether to keep the single y-intercept candidate
- extrema: indices of extrema candidates to keep
- monotonicity: indices of monotonicity candidates to keep

Reject candidates that are contradicted by the image, unsupported by the curve shape,
or too imprecise / too broad to trust.
"""


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from an LLM response."""
    src = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", src, re.DOTALL)
    if m:
        src = m.group(1)
    else:
        start = src.find("{")
        end = src.rfind("}")
        if start != -1 and end != -1 and end > start:
            src = src[start:end + 1]
    data = json.loads(src)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object from LLM filter")
    return data


def _build_filter_candidates(
    facts: Dict[str, Any],
    *,
    fact_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build an indexed candidate structure for the LLM filter."""
    keys = fact_keys if fact_keys is not None else list(_VISUAL_FACT_KEYS)
    payload: Dict[str, Any] = {}

    if "intercepts" in keys:
        raw = facts.get("intercepts", {}) or {}
        x_ints = raw.get("x_intercepts", []) or []
        y_int = raw.get("y_intercept")
        payload["intercepts"] = {
            "x_intercepts": [
                {"idx": i, "value": x_val}
                for i, x_val in enumerate(x_ints)
            ],
            "y_intercept": (
                {"idx": 0, "value": y_int}
                if y_int is not None else None
            ),
        }

    if "extrema" in keys:
        raw = facts.get("extrema", []) or []
        payload["extrema"] = [
            {"idx": i, "x": item.get("x"), "y": item.get("y"), "type": item.get("type")}
            for i, item in enumerate(raw)
            if isinstance(item, dict)
        ]

    if "monotonicity" in keys:
        raw = facts.get("monotonicity", []) or []
        payload["monotonicity"] = [
            {
                "idx": i,
                "interval": item.get("interval"),
                "direction": item.get("direction"),
                "scope": item.get("scope", "visible_range"),
            }
            for i, item in enumerate(raw)
            if isinstance(item, dict)
        ]

    return payload


def _coerce_index_list(raw: Any, upper_bound: int) -> List[int]:
    """Parse a list of integer indices safely."""
    if not isinstance(raw, list):
        return []
    out: List[int] = []
    for item in raw:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < upper_bound and idx not in out:
            out.append(idx)
    return out


def filter_visual_facts_with_llm(
    adapter,
    image: Any,
    coords: Iterable[Any],
    facts: Dict[str, Any],
    *,
    mode: str | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Use the current reasoning model to conservatively filter visual facts.

    The model is only allowed to keep/reject indexed candidates; numeric values
    are copied verbatim from the CV pipeline so the filter cannot hallucinate
    new facts.
    """
    profile = get_intervention_profile(mode)
    fact_keys: List[str] = list(profile.get("visual_fact_keys") or _VISUAL_FACT_KEYS)
    candidates = _build_filter_candidates(facts, fact_keys=fact_keys)
    coords_text = format_coords(coords)
    prompt = _FILTER_PROMPT.replace("{coords_text}", coords_text)
    prompt = prompt.replace(
        "{candidate_facts}",
        json.dumps(candidates, ensure_ascii=False, indent=2),
    )

    metadata: Dict[str, Any] = {
        "applied": True,
        "mode": profile.get("mode"),
        "raw_candidate_counts": {
            "x_intercepts": len(candidates.get("intercepts", {}).get("x_intercepts", [])),
            "y_intercept": 1 if candidates.get("intercepts", {}).get("y_intercept") is not None else 0,
            "extrema": len(candidates.get("extrema", [])),
            "monotonicity": len(candidates.get("monotonicity", [])),
        },
        "notes": [],
        "status": "ok",
    }

    try:
        raw = adapter.call_with_prompt(image, prompt)
        decision = _extract_json_object(raw)
        keep = decision.get("keep", {}) if isinstance(decision.get("keep"), dict) else {}
        metadata["raw_response"] = raw
        if isinstance(decision.get("notes"), list):
            metadata["notes"] = [str(x) for x in decision["notes"][:8]]

        raw_intercepts = facts.get("intercepts", {}) or {}
        raw_x = raw_intercepts.get("x_intercepts", []) or []
        keep_x = _coerce_index_list(keep.get("x_intercepts"), len(raw_x))
        keep_y = bool(keep.get("keep_y_intercept")) and raw_intercepts.get("y_intercept") is not None

        raw_extrema = facts.get("extrema", []) or []
        keep_extrema = _coerce_index_list(keep.get("extrema"), len(raw_extrema))

        raw_mono = facts.get("monotonicity", []) or []
        keep_mono = _coerce_index_list(keep.get("monotonicity"), len(raw_mono))

        filtered: Dict[str, Any] = {
            "intercepts": {
                "x_intercepts": [raw_x[i] for i in keep_x],
                "y_intercept": raw_intercepts.get("y_intercept") if keep_y else None,
            },
            "extrema": [raw_extrema[i] for i in keep_extrema],
            "monotonicity": [raw_mono[i] for i in keep_mono],
            "source": "visual_computation_llm_filtered",
        }

        metadata["kept_counts"] = {
            "x_intercepts": len(filtered["intercepts"]["x_intercepts"]),
            "y_intercept": 1 if filtered["intercepts"]["y_intercept"] is not None else 0,
            "extrema": len(filtered["extrema"]),
            "monotonicity": len(filtered["monotonicity"]),
        }
        return filtered, metadata
    except Exception as exc:
        logger.warning("LLM fact filter failed for mode=%s: %s", mode, exc)
        fallback = dict(facts)
        fallback["source"] = "visual_computation_filter_failed"
        metadata["status"] = "failed_open"
        metadata["notes"] = [f"LLM filter failed: {type(exc).__name__}"]
        return fallback, metadata


def infer_chart_facts(coords: Iterable[Any]) -> Dict[str, Any]:
    """Infer compact chart facts from detected integer coordinates.

    Each fact category is annotated with a ``_confidence`` field
    (``"high"`` / ``"medium"`` / ``"low"``) so downstream formatting
    can decide what to inject versus what to omit.
    """
    points = normalize_coords(coords)
    intercepts = _infer_intercepts(points)
    monotonicity = _infer_monotonicity(points)
    extrema = _infer_extrema(points)
    symmetry = _infer_symmetry(points)

    # ---- Confidence scoring ----
    intercepts["_confidence"] = _score_intercept_confidence(intercepts, points)
    _score_monotonicity_confidence(monotonicity, points)
    extrema_conf = "high" if len(points) >= 5 else ("medium" if len(points) >= 3 else "low")
    symmetry_conf = _score_symmetry_confidence(points, symmetry)

    notes = []
    if not points:
        notes.append("No reliable coordinate points were available.")
    else:
        notes.append("Coordinates are high-confidence (integer-grid detection).")

    return {
        "coordinates": [[x, y] for x, y in points],
        "intercepts": intercepts,
        "extrema": extrema,
        "extrema_confidence": extrema_conf,
        "monotonicity": monotonicity,
        "symmetry": symmetry,
        "symmetry_confidence": symmetry_conf,
        "notes": notes,
    }


def _score_intercept_confidence(intercepts: dict, points: list) -> str:
    """Score overall intercept confidence."""
    y_exact = intercepts.get("y_intercept_exact", False)
    y_val = intercepts.get("y_intercept")
    has_x_exact = bool(intercepts.get("x_intercepts_exact"))
    has_x_interp = bool(intercepts.get("x_intercepts_interpolated"))

    if y_exact and has_x_exact and not has_x_interp:
        return "high"
    if y_val is not None and y_val != "undetermined":
        return "medium"
    return "low"


def _score_monotonicity_confidence(segments: list, points: list) -> None:
    """Add per-segment confidence in place."""
    n = len(points)
    for seg in segments:
        interval = seg.get("interval", [])
        lo = interval[0] if interval else None
        hi = interval[1] if interval else None
        has_inf = any(isinstance(v, str) and "inf" in str(v) for v in interval)

        lo_num = lo if isinstance(lo, (int, float)) else None
        hi_num = hi if isinstance(hi, (int, float)) else None
        if lo_num is not None and hi_num is not None:
            span_pts = sum(1 for x, _ in points if lo_num <= x <= hi_num)
        else:
            span_pts = n

        if span_pts >= 4 and not has_inf:
            seg["_confidence"] = "high"
        elif span_pts >= 3:
            seg["_confidence"] = "medium"
        else:
            seg["_confidence"] = "low"


def _score_symmetry_confidence(points: list, symmetry: str) -> str:
    """Score symmetry confidence."""
    if len(points) < 4:
        return "low"
    point_set = set((x, y) for x, y in points)
    pos_pts = [(x, y) for x, y in points if x > 0]
    if not pos_pts:
        return "low"
    if symmetry == "neither":
        return "medium"
    match_count = 0
    for x, y in pos_pts:
        if symmetry == "even" and (-x, y) in point_set:
            match_count += 1
        elif symmetry == "odd" and (-x, -y) in point_set:
            match_count += 1
    ratio = match_count / len(pos_pts) if pos_pts else 0
    if ratio >= 0.9:
        return "high"
    return "medium"


def build_intervention_payload(
    coords: Iterable[Any],
    mode: str | None = None,
    *,
    visual_facts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the prompt payload for a specific GeoSym fusion level.

    When ``visual_facts`` (from the CV pipeline) is provided, they are used
    directly — these are grounded in pixel-level evidence and far more
    reliable than the heuristic ``infer_chart_facts``.

    For tiered-injection fusion modes, the payload includes a
    ``tiered_text`` field with the 3-tier structured evidence block.
    """
    profile = get_intervention_profile(mode)
    points = normalize_coords(coords)

    if visual_facts is not None:
        facts = visual_facts
    else:
        facts = infer_chart_facts(points)

    vf_keys: Optional[List[str]] = profile.get("visual_fact_keys")
    use_tiered = profile.get("tiered_injection", False)

    if use_tiered and vf_keys:
        tiered_text = format_tiered_facts_for_prompt(facts, coords, fact_keys=vf_keys)
        structured_facts = tiered_text
    elif vf_keys and len(vf_keys) > 0:
        structured_facts = format_visual_facts_for_prompt(facts, fact_keys=vf_keys)
        tiered_text = ""
    else:
        structured_facts = "Not provided in this intervention profile."
        tiered_text = ""

    return {
        "mode": profile["mode"],
        "label": profile["label"],
        "description": profile["description"],
        "coordinates": points,
        "coordinate_text": format_coords(points),
        "facts": facts,
        "facts_text": structured_facts,
        "tiered_text": tiered_text,
        "use_tiered": use_tiered,
        "summary": build_summary(profile["mode"], points, facts),
    }


def format_coords(coords: Iterable[Any]) -> str:
    """Format coordinates for prompt injection."""
    points = normalize_coords(coords)
    if not points:
        return "(no points detected)"
    return ", ".join(f"({x}, {y})" for x, y in points)


# ── Visual fact formatting (Module M2 output → prompt text) ──────────

_VISUAL_FACT_KEYS = ("intercepts", "extrema", "monotonicity")


def format_visual_facts_for_prompt(
    facts: Dict[str, Any],
    *,
    fact_keys: Optional[List[str]] = None,
) -> str:
    """Format visually-computed facts into a structured JSON for prompt injection.

    Unlike the old ``format_structured_facts`` which had confidence-gating
    for heuristic inferences, visual facts from the CV pipeline are treated
    as high-confidence by default because they are grounded in pixel-level
    evidence (same reliability tier as coordinate extraction).
    """
    keys = fact_keys if fact_keys is not None else list(_VISUAL_FACT_KEYS)
    payload: Dict[str, Any] = {}

    for k in keys:
        raw = facts.get(k)
        if raw is None:
            continue

        if k == "intercepts":
            x_ints = raw.get("x_intercepts", [])
            y_int = raw.get("y_intercept")
            entry: Dict[str, Any] = {}
            if x_ints:
                entry["x_intercepts"] = x_ints
            else:
                entry["x_intercepts"] = "none detected (curve may not cross x-axis in visible range)"
            if y_int is not None:
                entry["y_intercept"] = y_int
            else:
                entry["y_intercept"] = "none detected (curve may not cross y-axis in visible range)"
            payload["intercepts"] = entry

        elif k == "extrema":
            if isinstance(raw, list) and len(raw) > 0:
                payload["extrema"] = [
                    {"x": e["x"], "y": e["y"], "type": e["type"]}
                    for e in raw
                ]
            else:
                payload["extrema"] = "none detected in visible range"

        elif k == "monotonicity":
            if isinstance(raw, list) and len(raw) > 0:
                payload["monotonicity"] = [
                    {"interval": m["interval"], "direction": m["direction"],
                     "scope": m.get("scope", "visible_range")}
                    for m in raw
                ]
            else:
                payload["monotonicity"] = "insufficient data"

    payload["data_source"] = "GeoSym visual-computational pipeline (pixel-level detection)"
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ── Tiered fact formatting (completeness-aware injection) ────────────

def format_tiered_facts_for_prompt(
    facts: Dict[str, Any],
    coords: Iterable[Any],
    *,
    fact_keys: Optional[List[str]] = None,
) -> str:
    """Format GeoSym output as a 3-tier structured text block.

    Tier 1 — Verified Anchors: coordinates (fully reliable)
    Tier 2 — Detected Features: intercepts, extrema (reliable but possibly incomplete)
    Tier 3 — Inferred Patterns: monotonicity (reference only, verify from image)
    """
    keys = fact_keys if fact_keys is not None else list(_VISUAL_FACT_KEYS)
    points = normalize_coords(coords)
    lines: List[str] = []

    # ── Tier 1: Verified Anchors ──
    lines.append("== TIER 1: VERIFIED COORDINATE ANCHORS (fully reliable) ==")
    if points:
        coord_str = ", ".join(f"({x}, {y})" for x, y in points)
        lines.append(f"GeoSym detected {len(points)} integer-valued points on the curve:")
        lines.append(coord_str)
        lines.append("These are pixel-level detections. You may rely on them fully.")
    else:
        lines.append("No coordinate points detected.")
    lines.append("")

    # ── Tier 2: Detected Features (only exact observations) ──
    tier2_items: List[str] = []
    if "intercepts" in keys:
        raw = facts.get("intercepts") or {}
        x_ints = raw.get("x_intercepts", [])
        y_int = raw.get("y_intercept")
        if x_ints:
            tier2_items.append(
                f"  - X-intercepts at grid points: {x_ints}\n"
                f"    Note: There may be additional x-intercepts at non-integer positions. Check the image."
            )
        if y_int is not None:
            tier2_items.append(f"  - Y-intercept at grid point: {y_int}")

    if tier2_items:
        lines.append("== TIER 2: DETECTED FEATURES (exact grid observations) ==")
        lines.append("The following features were observed at exact grid intersections.")
        lines.append("Additional features may exist at non-integer positions — check the image.")
        lines.extend(tier2_items)
        lines.append("")

    # ── Tier 3: Inferred Patterns ──
    if "monotonicity" in keys:
        raw_mono = facts.get("monotonicity", [])
        lines.append("== TIER 3: INFERRED PATTERNS (reference only — verify from image) ==")
        if isinstance(raw_mono, list) and raw_mono:
            n_pts = len(points)
            mono_parts = []
            for m in raw_mono:
                interval = m.get("interval", [])
                direction = m.get("direction", "?")
                mono_parts.append(f"[{interval[0]}, {interval[1]}] {direction}")
            lines.append(f"  - Monotonicity (inferred from {n_pts} detected points):")
            for part in mono_parts:
                lines.append(f"    {part}")
            lines.append("    WARNING: These intervals are derived from limited coordinate data.")
            lines.append("    The actual intervals may be WIDER. Use the image as your primary")
            lines.append("    reference for monotonicity — extend intervals as needed.")
        else:
            lines.append("  - Insufficient data to infer monotonicity.")
            lines.append("    Please determine monotonicity directly from the image.")

    return "\n".join(lines)


# ── LLM fact verification (conservative error detection) ─────────────

_VERIFY_PROMPT_PATH = "prompts/fact_verification_prompt.txt"


def verify_facts_with_llm(
    adapter,
    image: Any,
    coords: Iterable[Any],
    facts: Dict[str, Any],
    *,
    fact_keys: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Use the LLM to identify and remove INCORRECT facts.

    Unlike the old ``filter_visual_facts_with_llm`` which selected
    *useful* facts (aggressive filtering), this is a conservative error
    detector that only removes facts clearly contradicted by the image.
    """
    import os
    keys = fact_keys if fact_keys is not None else list(_VISUAL_FACT_KEYS)
    coords_text = format_coords(coords)
    candidates = _build_filter_candidates(facts, fact_keys=keys)
    candidate_text = json.dumps(candidates, ensure_ascii=False, indent=2)

    prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                               _VERIFY_PROMPT_PATH)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_tpl = f.read()
    except FileNotFoundError:
        prompt_tpl = (
            "Review these visual facts against the image. "
            "Mark each as correct or incorrect.\n"
            "Coords: {coords_text}\nFacts: {candidate_facts}\n"
            "Return JSON with status for each category."
        )

    prompt = prompt_tpl.replace("{coords_text}", coords_text)
    prompt = prompt.replace("{candidate_facts}", candidate_text)

    metadata: Dict[str, Any] = {
        "applied": True, "type": "verification",
        "status": "ok", "notes": [],
    }

    try:
        raw = adapter.call_with_prompt(image, prompt)
        decision = _extract_json_object(raw)
        metadata["raw_response"] = raw

        filtered = dict(facts)
        removed: List[str] = []

        for category in keys:
            cat_result = decision.get(category, {})
            if isinstance(cat_result, dict):
                status = cat_result.get("status", "correct").lower()
                reason = cat_result.get("reason", "")
            else:
                status = "correct"
                reason = ""

            if status == "incorrect":
                removed.append(f"{category}: {reason}")
                if category == "intercepts":
                    filtered["intercepts"] = {
                        "x_intercepts": [],
                        "y_intercept": None,
                    }
                elif category == "extrema":
                    filtered["extrema"] = []
                elif category == "monotonicity":
                    filtered["monotonicity"] = []

        metadata["removed_categories"] = removed
        metadata["notes"] = [f"Removed: {c}" for c in removed] if removed else ["All facts verified"]
        return filtered, metadata

    except Exception as exc:
        logger.warning("LLM fact verification failed: %s", exc)
        metadata["status"] = "failed_open"
        metadata["notes"] = [f"Verification failed: {type(exc).__name__}"]
        return dict(facts), metadata


_ALL_FACT_KEYS = ("intercepts", "extrema", "monotonicity", "symmetry")
_FACT_DEFAULTS: Dict[str, Any] = {
    "intercepts": {},
    "extrema": [],
    "monotonicity": [],
    "symmetry": "neither",
}


_VERIFY_IMAGE = "⚠ verify from the image (low-confidence estimate)"

_CONF_THRESHOLDS = {
    "inject": {"high", "medium"},
    "hint_only": {"low"},
}


def format_structured_facts(
    facts: Dict[str, Any],
    *,
    fact_keys: Optional[List[str]] = None,
) -> str:
    """Return a confidence-gated JSON string for prompt injection.

    Confidence levels:
      - **high** → injected as reliable data
      - **medium** → injected with a "verify" note
      - **low** → replaced with an instruction to check the image

    This prevents incomplete inferences from overriding the LLM's own
    visual reasoning, which is the root cause of injection-induced
    performance degradation.
    """
    keys = fact_keys if fact_keys is not None else list(_ALL_FACT_KEYS)
    reliable: Dict[str, Any] = {}
    verify_hints: List[str] = []

    for k in keys:
        raw = facts.get(k, _FACT_DEFAULTS.get(k))
        if k == "intercepts":
            reliable[k] = _gate_intercepts(raw, verify_hints)
        elif k == "monotonicity":
            reliable[k] = _gate_monotonicity(raw, verify_hints)
        elif k == "extrema":
            conf = facts.get("extrema_confidence", "medium")
            if conf == "low":
                verify_hints.append("Extrema: not enough points to determine reliably — check image.")
            else:
                reliable[k] = raw
        elif k == "symmetry":
            conf = facts.get("symmetry_confidence", "medium")
            if conf == "low":
                verify_hints.append("Symmetry: insufficient data — determine from image.")
            else:
                reliable[k] = raw
        else:
            reliable[k] = raw

    if verify_hints:
        reliable["⚠ verify_from_image"] = verify_hints

    reliable["notes"] = facts.get("notes", [])
    return json.dumps(reliable, ensure_ascii=False, indent=2)


def _gate_intercepts(intercepts: dict, hints: List[str]) -> dict:
    """Gate intercept facts by confidence."""
    conf = intercepts.get("_confidence", "medium")
    out = {}

    # x-intercepts
    x_exact = intercepts.get("x_intercepts_exact", [])
    x_interp = intercepts.get("x_intercepts_interpolated", [])
    if x_exact:
        out["x_intercepts"] = x_exact
    if x_interp:
        out["x_intercepts_estimated"] = x_interp
    if not x_exact and not x_interp:
        hints.append("x-intercepts: none on integer grid — check image for zero-crossings.")

    # y-intercept
    y_val = intercepts.get("y_intercept")
    y_exact = intercepts.get("y_intercept_exact", False)
    if y_exact and y_val is not None and y_val != "undetermined":
        out["y_intercept"] = y_val
    elif isinstance(y_val, (int, float)):
        out["y_intercept_estimated"] = y_val
        out["y_intercept_note"] = "interpolated — verify from image"
    else:
        hints.append("y-intercept: not on integer grid — read it from the image.")

    return out


def _gate_monotonicity(segments: list, hints: List[str]) -> list:
    """Gate monotonicity segments by per-segment confidence."""
    gated = []
    for seg in segments:
        conf = seg.get("_confidence", "medium")
        clean_seg = {
            "interval": seg["interval"],
            "direction": seg["direction"],
        }
        if conf == "low":
            clean_seg["note"] = _VERIFY_IMAGE
        gated.append(clean_seg)

    if not gated:
        hints.append("Monotonicity: not enough data — determine from the image.")

    return gated


def build_summary(mode: str, points: List[Tuple[int, int]], facts: Dict[str, Any]) -> List[str]:
    """Build short display lines for UI/debug output."""
    lines = [f"Fusion level: {mode}", f"Reliable coordinates: {len(points)}"]
    intercepts = facts.get("intercepts", {})
    x_ints = intercepts.get("x_intercepts", [])
    x_exact = intercepts.get("x_intercepts_exact", [])
    x_interp = intercepts.get("x_intercepts_interpolated", [])
    y_int = intercepts.get("y_intercept")
    if isinstance(y_int, str) and "undetermined" in y_int:
        y_int = None
    x_desc = f"{x_ints}" if x_ints else "none"
    if x_interp:
        x_desc += f" (exact={x_exact}, interpolated={x_interp})"
    lines.append(f"x-intercepts: {x_desc}")
    lines.append(f"y-intercept: {y_int if y_int is not None else 'none'}")
    extrema = facts.get("extrema", [])
    lines.append(f"extrema hints: {len(extrema)}")
    mono = facts.get("monotonicity", [])
    lines.append(f"monotonicity segments: {len(mono)}")
    lines.append(f"symmetry hint: {facts.get('symmetry', 'neither')}")
    return lines


def _infer_intercepts(points: List[Tuple[int, int]]) -> Dict[str, Any]:
    """Infer intercepts from detected coordinate points.

    Exact intercepts (integer grid) are reported directly.  For adjacent
    points with opposite-sign y-values, linear interpolation estimates
    the zero-crossing x-coordinate (rounded to 2 decimal places).

    The y-intercept is reported as:
      - An exact integer if a point at x=0 is detected.
      - An interpolated float if adjacent points span x=0.
      - The string ``"undetermined"`` otherwise, so the LLM knows to
        check the image rather than assuming no y-intercept.
    """
    exact_x = {float(x) for x, y in points if y == 0}

    interpolated_x: set[float] = set()
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if y1 == 0 or y2 == 0:
            continue
        if (y1 > 0) != (y2 > 0):
            x_zero = x1 - y1 * (x2 - x1) / (y2 - y1)
            interpolated_x.add(round(x_zero, 2))

    all_x = sorted(exact_x | interpolated_x)

    # y-intercept: exact → interpolated → undetermined
    y_candidates = sorted({y for x, y in points if x == 0})
    y_intercept_exact = True
    if len(y_candidates) == 1:
        y_intercept = y_candidates[0]
    else:
        # Try interpolation between adjacent points that span x=0
        y_intercept = _interpolate_y_at_zero(points)
        y_intercept_exact = False

    return {
        "x_intercepts": all_x,
        "y_intercept": y_intercept,
        "y_intercept_exact": y_intercept_exact,
        "x_intercepts_exact": sorted(exact_x),
        "x_intercepts_interpolated": sorted(interpolated_x - exact_x),
    }


def _interpolate_y_at_zero(points: List[Tuple[int, int]]):
    """Estimate y-intercept via linear interpolation between adjacent
    points that straddle x=0.  Returns a rounded float or
    ``"undetermined"`` when no spanning pair exists."""
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if x1 < 0 < x2 and x2 != x1:
            y_at_zero = y1 + (0 - x1) * (y2 - y1) / (x2 - x1)
            return round(y_at_zero, 2)
        if x1 == 0:
            return y1
    return "undetermined"


def _infer_monotonicity(points: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """Infer monotonicity segments from detected coordinate points.

    At integer-grid resolution, slowly-changing functions (e.g. exponentials
    near an asymptote) appear "constant".  Constant regions at the left/right
    boundary are absorbed into the adjacent monotonic segment so the LLM is
    not misled into thinking there is a turning point.
    """
    if len(points) < 2:
        return []

    # Step 1: classify each consecutive pair
    pair_dirs: List[Tuple[int, int, str]] = []  # (x_start, x_end, direction)
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if x2 == x1:
            continue
        if y2 > y1:
            d = "increasing"
        elif y2 < y1:
            d = "decreasing"
        else:
            d = "constant"
        pair_dirs.append((x1, x2, d))

    if not pair_dirs:
        return []

    # Step 2: merge consecutive same-direction pairs into segments,
    # treating "constant" as compatible with an adjacent direction
    # at the boundaries (left/right edge).
    segments: List[Dict[str, Any]] = []
    seg_start = pair_dirs[0][0]
    seg_end = pair_dirs[0][1]
    seg_dir = pair_dirs[0][2]

    for _, x_end, d in pair_dirs[1:]:
        if d == seg_dir or d == "constant":
            seg_end = x_end
        elif seg_dir == "constant":
            seg_dir = d
            seg_end = x_end
        else:
            segments.append({"interval": [seg_start, seg_end], "direction": seg_dir})
            seg_start = seg_end
            seg_end = x_end
            seg_dir = d

    segments.append({"interval": [seg_start, seg_end], "direction": seg_dir})

    # Step 3: absorb boundary "constant" segments into their neighbor
    # (a constant region at the very edge likely means the function is
    # changing too slowly for integer detection, not a true flat region)
    if len(segments) > 1 and segments[0]["direction"] == "constant":
        segments[1]["interval"][0] = segments[0]["interval"][0]
        segments.pop(0)
    if len(segments) > 1 and segments[-1]["direction"] == "constant":
        segments[-2]["interval"][1] = segments[-1]["interval"][1]
        segments.pop()
    # If the only segment is "constant", report it as-is (truly flat)

    x_min = points[0][0]
    x_max = points[-1][0]
    cleaned = []
    for seg in segments:
        lo, hi = seg["interval"]
        if hi > lo and seg["direction"] != "constant":
            if lo == x_min:
                seg["interval"][0] = "-inf (observed from {})".format(lo)
            if hi == x_max:
                seg["interval"][1] = "+inf (observed to {})".format(hi)
            cleaned.append(seg)
        elif hi > lo:
            cleaned.append(seg)
    return cleaned


def _infer_extrema(points: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    extrema: List[Dict[str, Any]] = []
    seen = set()
    for idx in range(1, len(points) - 1):
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        x2, y2 = points[idx + 1]
        if not (x0 < x1 < x2):
            continue
        ext_type = None
        if y1 > y0 and y1 > y2:
            ext_type = "max"
        elif y1 < y0 and y1 < y2:
            ext_type = "min"
        if ext_type and (x1, y1, ext_type) not in seen:
            seen.add((x1, y1, ext_type))
            extrema.append({"x": x1, "y": y1, "type": ext_type})
    return extrema


def _infer_symmetry(points: List[Tuple[int, int]]) -> str:
    if len(points) < 3:
        return "neither"

    point_set = set(points)
    pair_total = 0
    even_hits = 0
    odd_hits = 0

    for x_val, y_val in points:
        if x_val <= 0:
            continue
        pair_total += 1
        if (-x_val, y_val) in point_set:
            even_hits += 1
        if (-x_val, -y_val) in point_set:
            odd_hits += 1

    if pair_total == 0:
        return "neither"

    even_score = even_hits / pair_total
    odd_score = odd_hits / pair_total

    if even_score >= 0.8 and even_score > odd_score:
        return "even"
    if odd_score >= 0.8 and odd_score > even_score:
        return "odd"
    return "neither"
