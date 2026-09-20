"""
GeoSym chart-fact construction and prompt injection.

Converts reliable coordinates into structured facts (intercepts, extrema,
monotonicity, symmetry) and formats them for injection into the reasoning
prompts. Used by the `E5_raw` condition — the GeoSym method itself.

Confidence scoring (M1) and utility-aware selection (M2) were removed on
2026-09-19 after eight independent tests found no discriminative signal;
see `docs/M1M2_RETIREMENT_AND_FGCS_REPOSITIONING_PLAN_2026-09-19.md`.
What remains is M0's inference capability plus flat prompt formatting.
"""

from __future__ import annotations

import json
import logging
import statistics
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


DEFAULT_INTERVENTION_MODE = "visual_facts_flat"

EXTREMA_FILTER_RULE_VERSION = "p0-2-discontinuity-bridge-v1"
EXTREMA_FILTER_X_GAP_FACTOR = 1.5
EXTREMA_FILTER_Y_JUMP_FACTOR = 2.5

INTERVENTION_PROFILES: Dict[str, Dict[str, Any]] = {
    # Flat injection is the only surviving profile. `visual_facts_full`
    # (LLM verification + tiered injection) was removed with M1/M2 on
    # 2026-09-19; it was the sole production entry point into that path.
    "visual_facts_flat": {
        "label": "GeoSym Structured Fusion (Flat Facts)",
        "role": "experimental",
        "visual_fact_keys": ["intercepts", "extrema", "monotonicity"],
        "description": "GeoSym coordinates plus structured visual facts injected in a flat format.",
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


def infer_chart_facts(coords: Iterable[Any]) -> Dict[str, Any]:
    """Infer compact chart facts from detected integer coordinates.

    Facts are emitted unannotated. The heuristic ``_confidence`` tiers this
    function used to attach were removed with M1 (2026-09-19): their only
    consumers were the confidence gates that shipped with M1/M2, and eight
    independent tests found the tiers carried no discriminative signal.
    """
    points = normalize_coords(coords)

    notes = []
    if not points:
        notes.append("No reliable coordinate points were available.")
    else:
        notes.append("Coordinates are high-confidence (integer-grid detection).")

    return {
        "coordinates": [[x, y] for x, y in points],
        "intercepts": _infer_intercepts(points),
        "extrema": _infer_extrema(points),
        "monotonicity": _infer_monotonicity(points),
        "symmetry": _infer_symmetry(points),
        "notes": notes,
    }


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

    Facts are injected flat. The tiered-injection variant was removed with
    M2 (2026-09-19).
    """
    profile = get_intervention_profile(mode)
    points = normalize_coords(coords)

    if visual_facts is not None:
        facts = visual_facts
    else:
        facts = infer_chart_facts(points)

    vf_keys: Optional[List[str]] = profile.get("visual_fact_keys")

    if vf_keys:
        structured_facts = format_visual_facts_for_prompt(facts, fact_keys=vf_keys)
    else:
        structured_facts = "Not provided in this intervention profile."

    return {
        "mode": profile["mode"],
        "label": profile["label"],
        "description": profile["description"],
        "coordinates": points,
        "coordinate_text": format_coords(points),
        "facts": facts,
        "facts_text": structured_facts,
        "summary": build_summary(profile["mode"], points, facts),
    }


def format_coords(coords: Iterable[Any]) -> str:
    """Format coordinates for prompt injection."""
    points = normalize_coords(coords)
    if not points:
        return "(no points detected)"
    return ", ".join(f"({x}, {y})" for x, y in points)


# ── Visual fact formatting (chart facts → prompt text) ──────────

_VISUAL_FACT_KEYS = ("intercepts", "extrema", "monotonicity")


def format_visual_facts_for_prompt(
    facts: Dict[str, Any],
    *,
    fact_keys: Optional[List[str]] = None,
) -> str:
    """Format visually-computed facts into a structured JSON for prompt injection.

    Every fact is injected as-is. Visual facts from the CV pipeline are
    grounded in pixel-level evidence, the same reliability tier as the
    coordinate extraction they come from.
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


# The tiered-formatting and LLM-verification sections that stood here were
# removed with M1/M2 on 2026-09-19, along with their last consumers. The
# constants they left behind (`_VERIFY_PROMPT_PATH`, `_VERIFY_IMAGE`,
# `_CONF_THRESHOLDS`, `_ALL_FACT_KEYS`, `_FACT_DEFAULTS`) went with them: they
# described a confidence-gating policy this module no longer applies, and
# leaving them readable invited a future caller to reimplement against them.
# `prompts/fact_verification_prompt.txt` stays on disk -- its SHA256 is part of
# the paid `code_state` records under outputs/.


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


def infer_extrema_candidates(coords: Iterable[Any]) -> List[Dict[str, Any]]:
    """Derive filtered extrema candidates from raw M0 coordinates.

    Narrow public entry point for the single extrema implementation in this
    module, so callers outside this file do not have to reach for private
    helpers. P0-2 applies a deterministic, family-agnostic filter before
    returning candidates.

    Returns the existing candidate schema — a list of
    ``{"x": int, "y": int, "type": "max" | "min"}`` — and nothing else: no
    symmetry, no reliability tier, no other fact category.

    These are *candidates*, not confirmed facts.  The three-point discrete test
    used underneath can confuse a genuine turning point with a jump across a
    discontinuity. The filter drops only candidates adjacent to an anomalous
    bridge segment: a wider-than-typical x gap together with a larger-than-
    typical y jump. This is a generic sampling-continuity rule, not a function
    family rule.
    """
    decisions = classify_extrema_candidates(coords)
    return [
        decision["candidate"]
        for decision in decisions
        if decision["decision"] == "KEEP"
    ]


def classify_extrema_candidates(coords: Iterable[Any]) -> List[Dict[str, Any]]:
    """Classify raw extrema candidates as KEEP or DROP.

    The classification is deterministic and uses only local coordinate geometry:
    it never reads function family, ground truth, expression labels, scores, or
    sample identifiers. It is intended for P0-2/P0-3 auditability, where the
    same frozen rule must be applied to discovery and unseen validation data.
    """
    points = normalize_coords(coords)
    raw_candidates = _infer_extrema(points)
    if not raw_candidates:
        return []

    candidates_by_point = {
        (candidate["x"], candidate["y"], candidate["type"]): candidate
        for candidate in raw_candidates
    }
    typical_x_gap = _median_positive(
        abs(points[index + 1][0] - points[index][0])
        for index in range(len(points) - 1)
    )
    typical_y_jump = _median_positive(
        abs(points[index + 1][1] - points[index][1])
        for index in range(len(points) - 1)
    )

    decisions: List[Dict[str, Any]] = []
    for point_index in range(1, len(points) - 1):
        previous_x, previous_y = points[point_index - 1]
        current_x, current_y = points[point_index]
        next_x, next_y = points[point_index + 1]
        matching_candidate = candidates_by_point.get(
            (current_x, current_y, "max")
        ) or candidates_by_point.get((current_x, current_y, "min"))
        if not matching_candidate:
            continue

        left_x_gap = abs(current_x - previous_x)
        left_y_jump = abs(current_y - previous_y)
        right_x_gap = abs(next_x - current_x)
        right_y_jump = abs(next_y - current_y)
        left_bridge = _is_discontinuity_bridge(
            x_gap=left_x_gap,
            y_jump=left_y_jump,
            typical_x_gap=typical_x_gap,
            typical_y_jump=typical_y_jump,
        )
        right_bridge = _is_discontinuity_bridge(
            x_gap=right_x_gap,
            y_jump=right_y_jump,
            typical_x_gap=typical_x_gap,
            typical_y_jump=typical_y_jump,
        )
        metrics = {
            "typical_x_gap": typical_x_gap,
            "typical_y_jump": typical_y_jump,
            "left_x_gap": left_x_gap,
            "left_y_jump": left_y_jump,
            "right_x_gap": right_x_gap,
            "right_y_jump": right_y_jump,
        }
        if left_bridge or right_bridge:
            bridge_side = "left" if left_bridge else "right"
            decisions.append(
                {
                    "candidate": matching_candidate,
                    "decision": "DROP",
                    "rule": EXTREMA_FILTER_RULE_VERSION,
                    "reason": f"adjacent_{bridge_side}_discontinuity_bridge",
                    "metrics": metrics,
                }
            )
        else:
            decisions.append(
                {
                    "candidate": matching_candidate,
                    "decision": "KEEP",
                    "rule": EXTREMA_FILTER_RULE_VERSION,
                    "reason": "local_turn_without_discontinuity_bridge",
                    "metrics": metrics,
                }
            )
    return decisions


def _median_positive(values: Iterable[float]) -> float:
    positive_values = [float(value) for value in values if float(value) > 0]
    if not positive_values:
        return 0.0
    return float(statistics.median(positive_values))


def _is_discontinuity_bridge(
    *,
    x_gap: float,
    y_jump: float,
    typical_x_gap: float,
    typical_y_jump: float,
) -> bool:
    if typical_x_gap <= 0 or typical_y_jump <= 0:
        return False
    return (
        x_gap > EXTREMA_FILTER_X_GAP_FACTOR * typical_x_gap
        and y_jump > EXTREMA_FILTER_Y_JUMP_FACTOR * typical_y_jump
    )


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
