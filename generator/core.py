"""
Orchestrate layout, stem, chart, curves; apply digital degradations; write image and ground-truth JSON.
Integer ticks only (step=1). All curves/axes/grid black. Origin labeled once.
"""

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .rendering.augmentation import apply_render_degradation
from .rendering.chart import draw_axes_and_grid, draw_chart_border
from .config_loader import load_config
from .rendering.curves import (
    FUNCTION_COORD_POOL,
    TYPE_INTERSECTION_CAP,
    draw_curves,
    generate_random_function,
    get_richness_spec,
    sample_function,
    summarize_function_candidate,
)
from .rendering.layout import chart_bbox, choose_layout, stem_bbox, estimate_stem_width
from .math.properties import compute_properties, compute_extended_properties
from .math.question import generate_question, SUB_QUESTION_KEYS
from .rendering.stem import draw_stem, estimate_wrapped_line_count


# ============================================================
# Module-level constants
# ============================================================

GRID_STEP = 1
MAX_RETRIES = 80
MAX_AXIS_SPAN = 14
STRICT_EPS = 1e-9
ORIGIN_GUARD = 2          # [-2, 2]² always visible
CURVE_SAMPLE_STEP = 0.02
ADAPTIVE_NUM_SAMPLES = 800
GENERATION_GUARD_RETRIES = 6
COORD_READABILITY_TARGET = 0.18
GENERIC_FRAGILITY_THRESHOLD = 0.58
GENERIC_HARD_FRAGILITY_THRESHOLD = 0.66
DEFAULT_CANDIDATE_ATTEMPTS = 120
DEFAULT_DIVERSITY_RETRIES = 18
DEFAULT_FAMILY_SOFT_CAP = 8
DEFAULT_SELECTION_POOL_SIZE = 10
DEFAULT_SELECTION_SCORE_MARGIN = 3.0


# ============================================================
# Low-level helpers
# ============================================================

def _tick_values(lo: int, hi: int) -> List[int]:
    """All integer tick positions from lo to hi (inclusive)."""
    return list(range(lo, hi + 1))


def _pick_recommended_range(func_type: str) -> Tuple[list, list]:
    """Return a randomly-chosen (x_range, y_range) from the per-type pool."""
    pool = FUNCTION_COORD_POOL.get(func_type)
    if not pool:
        pool = [{"x_range": [-5, 5], "y_range": [-5, 5]}]
    choice = random.choice(pool)
    return list(choice["x_range"]), list(choice["y_range"])


def _budget_pad(feat_lo: float, feat_hi: float) -> Tuple[int, int]:
    """Compute an integer range that covers *both* the feature interval
    [feat_lo, feat_hi] and the origin guard [-ORIGIN_GUARD, ORIGIN_GUARD],
    then distributes remaining budget as balanced whitespace.

    Guarantees: returned span <= MAX_AXIS_SPAN when the must-cover region
    fits; otherwise returns the minimal enclosing range (will be rejected
    by _finalize_range).
    """
    must_lo = int(math.floor(min(feat_lo, -ORIGIN_GUARD)))
    must_hi = int(math.ceil(max(feat_hi, ORIGIN_GUARD)))
    span = must_hi - must_lo

    if span >= MAX_AXIS_SPAN:
        return must_lo, must_hi

    budget = MAX_AXIS_SPAN - span
    pad_lo = budget // 2
    pad_hi = budget - pad_lo
    return must_lo - pad_lo, must_hi + pad_hi


def _finalize_range(
    x_min: int, x_max: int, y_min: int, y_max: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Validate that origin is visible and both axis spans fit within
    MAX_AXIS_SPAN.  Returns the (possibly expanded) range or *None* if
    the range is incompatible — the caller should discard this attempt
    and retry with new parameters.

    No trimming is ever performed: this avoids clipping curve features
    that would produce visual artefacts like trapezoid shapes.
    """
    x_min = min(x_min, -ORIGIN_GUARD)
    x_max = max(x_max, ORIGIN_GUARD)
    y_min = min(y_min, -ORIGIN_GUARD)
    y_max = max(y_max, ORIGIN_GUARD)

    if (x_max - x_min) > MAX_AXIS_SPAN or (y_max - y_min) > MAX_AXIS_SPAN:
        return None

    return x_min, x_max, y_min, y_max


# ============================================================
# Adaptive coordinate range refinement
# ============================================================

def _compute_adaptive_range(
    sampler,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Constraint-aware adaptive range.

    Samples the curve densely, detects feature extent via percentiles,
    and uses _budget_pad to compute a balanced viewing window.

    For unbounded functions whose P2/P98 range exceeds MAX_AXIS_SPAN,
    falls back to the original pool range — which is specifically
    designed to frame the curve's interesting region.
    """
    x_lo_init, x_hi_init = float(x_range[0]), float(x_range[1])
    y_lo_init, y_hi_init = float(y_range[0]), float(y_range[1])

    xs = np.linspace(x_lo_init, x_hi_init, ADAPTIVE_NUM_SAMPLES)
    raw = [sampler(float(x)) for x in xs]

    valid_mask = np.array([
        y is not None and np.isfinite(y) and abs(y) < 1e6
        for y in raw
    ])

    if valid_mask.sum() < 20:
        return (int(x_lo_init), int(x_hi_init)), (int(y_lo_init), int(y_hi_init))

    valid_xs = xs[valid_mask]
    valid_ys = np.array([float(raw[i]) for i in range(len(raw)) if valid_mask[i]])

    # --- X: detect the function's defined region ---
    x_def_lo = float(valid_xs[0])
    x_def_hi = float(valid_xs[-1])
    x_init_span = x_hi_init - x_lo_init
    x_coverage = (x_def_hi - x_def_lo) / x_init_span if x_init_span > 0 else 1.0

    if x_coverage < 0.6:
        margin = max((x_def_hi - x_def_lo) * 0.15, 1.0)
        x_feat_lo = x_def_lo - margin
        x_feat_hi = x_def_hi + margin
    else:
        x_feat_lo = x_lo_init
        x_feat_hi = x_hi_init

    new_x_min, new_x_max = _budget_pad(x_feat_lo, x_feat_hi)
    if (new_x_max - new_x_min) > MAX_AXIS_SPAN:
        new_x_min, new_x_max = int(x_lo_init), int(x_hi_init)

    # --- Y: P2/P98 percentile as feature extent ---
    y_feat_lo = float(np.percentile(valid_ys, 2))
    y_feat_hi = float(np.percentile(valid_ys, 98))

    new_y_min, new_y_max = _budget_pad(y_feat_lo, y_feat_hi)
    if (new_y_max - new_y_min) > MAX_AXIS_SPAN:
        new_y_min, new_y_max = int(y_lo_init), int(y_hi_init)

    return (new_x_min, new_x_max), (new_y_min, new_y_max)


# ============================================================
# Grid intersection detection
# ============================================================

def _compute_exact_grid_points(
    sampler,
    x_min: int, x_max: int,
    y_min: int, y_max: int,
) -> List[dict]:
    """
    Find *strict* grid intersection points (step=1): evaluate f(x) at
    every integer x and keep only those where f(x) is an exact integer
    (within floating-point epsilon).
    """
    y_set = set(range(y_min, y_max + 1))
    seen: set = set()
    out: List[dict] = []

    for xv in range(x_min, x_max + 1):
        y = sampler(float(xv))
        if y is None or not np.isfinite(y):
            continue
        iy = round(y)
        if abs(y - iy) > STRICT_EPS:
            continue
        iy = int(iy)
        if iy not in y_set:
            continue
        # Treat origin as a valid GT coordinate when it lies on the curve.
        # This keeps coordinate GT consistent with intercept facts
        # (x_intercept=0 and y_intercept=0 for curves through origin).
        # Keep boundary points as valid GT coordinates.
        # Previously these were dropped (strict interior only), which can
        # create false "extra predictions" in facts diagnosis when the
        # recognizer correctly detects visible edge points.
        if (xv, iy) not in seen:
            seen.add((xv, iy))
            out.append({
                "coordinate_x": xv,
                "coordinate_y": iy,
                "type": "curve_axis" if (xv == 0 or iy == 0) else "curve_grid",
            })

    return out


def _generation_cfg(config: dict) -> dict:
    """Return generation-specific config with safe defaults."""
    return config.get("generation", {})


def _intersection_targets(config: dict, func_type: str, user_min: int) -> Dict[str, int]:
    """Resolve per-type hard/preferred intersection targets."""
    spec = get_richness_spec(func_type)
    cfg = _generation_cfg(config)
    target_map = cfg.get("intersection_targets", {})
    target_override = target_map.get(func_type, {}) if isinstance(target_map, dict) else {}

    hard_min = int(target_override.get("hard_min", spec["hard_min"]))
    preferred_min = int(target_override.get("preferred_min", spec["preferred_min"]))
    preferred_max = int(target_override.get("preferred_max", spec["preferred_max"]))

    type_cap = TYPE_INTERSECTION_CAP.get(func_type, 99)
    hard_min = min(type_cap, max(user_min, hard_min))
    preferred_min = min(type_cap, max(hard_min, preferred_min))
    preferred_max = min(type_cap, max(preferred_min, preferred_max))

    return {
        "hard_min": hard_min,
        "preferred_min": preferred_min,
        "preferred_max": preferred_max,
        "type_cap": type_cap,
    }


def _score_generation_candidate(
    generated_functions: List[Dict[str, Any]],
    intersections: List[dict],
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    readability: float,
    targets: Dict[str, int],
) -> Tuple[float, Dict[str, Any]]:
    """Score a candidate by GT richness first, readability second."""
    if not generated_functions:
        return -1e9, {
            "intersection_count": 0,
            "x_axis_count": 0,
            "y_axis_count": 0,
            "x_coverage": 0.0,
            "y_coverage": 0.0,
            "family_signature": "invalid",
            "exact_signature": "invalid",
        }

    summary = summarize_function_candidate(
        generated_functions[0],
        intersections,
        x_min,
        x_max,
        y_min,
        y_max,
    )
    count = summary["intersection_count"]
    hard_min = targets["hard_min"]
    preferred_min = targets["preferred_min"]
    preferred_max = targets["preferred_max"]
    func_type = summary["func_type"]

    score = 8.0 * float(readability)
    score += 6.0 * float(summary["x_coverage"])
    score += 4.0 * float(summary["y_coverage"])
    score += 1.2 * float(summary["x_axis_count"])
    score += 0.8 * float(summary["y_axis_count"])

    if count < hard_min:
        score -= 30.0 * float(hard_min - count)
    elif count < preferred_min:
        score += 5.0 * float(count)
        score -= 8.0 * float(preferred_min - count)
    elif count <= preferred_max:
        score += 18.0 + 1.5 * float(count)
        score -= 1.5 * abs(float(count - preferred_min))
    else:
        score += 18.0 + 1.5 * float(preferred_max)
        score += 0.35 * float(count - preferred_max)

    params = generated_functions[0].get("parameters", {}) if generated_functions else {}
    if func_type == "linear":
        a = abs(float(params.get("a", 0.0)))
        b = abs(float(params.get("b", 0.0)))
        if a >= 2.0:
            score += 4.0
        if a >= 3.0:
            score += 3.0
        if b >= 2.0:
            score += 2.0
        if b == 0.0:
            score -= 3.0

    if func_type == "logarithmic":
        a = abs(float(params.get("a", 0.0)))
        h = abs(float(params.get("h", 0.0)))
        score += 0.8 * min(a, 4.0) + 0.3 * min(h, 8.0)

    if func_type == "exponential":
        a = abs(float(params.get("a", 0.0)))
        b = abs(float(params.get("b", 0.0)))
        h = abs(float(params.get("h", 0.0)))
        score += 0.8 * min(a, 3.0) + 0.6 * max(0.0, b - 2.0) + 0.2 * min(h, 5.0)

    if func_type == "rational":
        a = abs(float(params.get("a", 0.0)))
        h = abs(float(params.get("h", 0.0)))
        k = abs(float(params.get("k", 0.0)))
        score += 0.4 * min(a, 10.0) + 0.3 * min(h + k, 8.0)

    return score, summary


# ============================================================
# Extra evaluation tasks (prediction + rate of change)
# ============================================================

def _generate_extra_tasks(
    intersections: List[dict],
    sampler,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Prediction targets: midpoints between consecutive intersection
    x-values evaluated via the sampler.

    Rate of change: (y_last - y_first) / (x_last - x_first).
    """
    pred_targets: Dict[str, float] = {}
    rate_info: Dict[str, Any] = {"x1": 0, "x2": 1, "gt_value": 0.0}

    if not intersections or sampler is None:
        return pred_targets, rate_info

    sorted_pts = sorted(intersections, key=lambda p: p["coordinate_x"])
    xs = [p["coordinate_x"] for p in sorted_pts]
    ys = [p["coordinate_y"] for p in sorted_pts]

    for i in range(len(sorted_pts) - 1):
        mid_x = (xs[i] + xs[i + 1]) / 2.0
        y_val = sampler(mid_x)
        if y_val is not None and np.isfinite(y_val):
            pred_targets[str(round(mid_x, 1))] = round(float(y_val), 2)
        if len(pred_targets) >= 3:
            break

    if len(sorted_pts) >= 2:
        x1, y1 = xs[0], ys[0]
        x2, y2 = xs[-1], ys[-1]
        if x2 != x1:
            rate_info = {
                "x1": x1, "x2": x2,
                "gt_value": round((y2 - y1) / (x2 - x1), 4),
            }

    return pred_targets, rate_info


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _mask_true_runs(mask: List[bool] | np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive [start, end] runs where mask is True."""
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, flag in enumerate(mask):
        if bool(flag):
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _key_shape_points(
    fn: Dict[str, Any],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> List[Tuple[float, float]]:
    """Return visible shape-defining points that should not hug chart edges."""
    sampler = fn.get("sampler")
    if sampler is None:
        return []

    ext = compute_extended_properties(
        fn.get("type", ""), fn.get("parameters", {}), sampler, x_min, x_max,
    )

    points: List[Tuple[float, float]] = []
    for item in ext.get("extrema") or []:
        if not isinstance(item, dict):
            continue
        try:
            px = float(item.get("x"))
            py = float(item.get("y"))
        except (TypeError, ValueError):
            continue
        if x_min <= px <= x_max and y_min <= py <= y_max:
            points.append((px, py))

    # For piecewise linear functions, the breakpoint is visually important even
    # when it is not an extremum.
    if fn.get("type") == "piecewise_linear":
        params = fn.get("parameters", {})
        try:
            px = float(params.get("h"))
            py = float(params.get("k"))
        except (TypeError, ValueError):
            px = py = None
        if px is not None and py is not None and x_min <= px <= x_max and y_min <= py <= y_max:
            points.append((px, py))

    deduped: List[Tuple[float, float]] = []
    seen = set()
    for point in points:
        key = (round(point[0], 6), round(point[1], 6))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(point)
    return deduped


def _has_internal_visible_clipping(
    sampler,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    sample_count: int = 801,
) -> bool:
    """
    Detect truncation where a single continuous branch disappears beyond the
    top/bottom boundary and re-enters later, which is what causes cut-off
    shapes such as "flat-topped" cubics.
    """
    xs = np.linspace(float(x_min), float(x_max), sample_count)
    ys_raw = [sampler(float(x)) for x in xs]
    finite_mask = np.array([
        y is not None and np.isfinite(y) and abs(float(y)) < 1e6 for y in ys_raw
    ], dtype=bool)
    if finite_mask.sum() < 8:
        return True

    ys = np.array([
        float(y) if y is not None and np.isfinite(y) else np.nan for y in ys_raw
    ], dtype=np.float64)
    visible_mask = finite_mask & (ys >= y_min - 0.35) & (ys <= y_max + 0.35)

    for start, end in _mask_true_runs(finite_mask):
        branch_visible_runs = _mask_true_runs(visible_mask[start:end + 1])
        if len(branch_visible_runs) > 1:
            return True
    return False


def _shape_points_have_margin(
    fn: Dict[str, Any],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> bool:
    """Ensure visible shape-defining points are not squeezed against boundaries."""
    points = _key_shape_points(fn, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    if not points:
        return True

    x_span = max(1e-6, float(x_max - x_min))
    y_span = max(1e-6, float(y_max - y_min))
    x_margin = max(0.45, 0.06 * x_span)
    y_margin = max(0.65, 0.09 * y_span)

    return all(
        (x_min + x_margin) < px < (x_max - x_margin)
        and (y_min + y_margin) < py < (y_max - y_margin)
        for px, py in points
    )


def _passes_visibility_guard(
    fn: Dict[str, Any],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> bool:
    """Generic pre-render guard against visually truncated function shapes."""
    sampler = fn.get("sampler")
    if sampler is None:
        return False
    if _has_internal_visible_clipping(sampler, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max):
        return False
    return _shape_points_have_margin(fn, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def _math_to_pixel_from_gt(
    x: float,
    y: float,
    gt: Dict[str, Any],
    padding: int,
) -> Tuple[float, float]:
    """Map math coordinates into the current chart bbox recorded in GT."""
    cx, cy, cw, ch = [float(v) for v in gt["chart_bbox"]]
    x_min, x_max = [float(v) for v in gt["x_range"]]
    y_min, y_max = [float(v) for v in gt["y_range"]]

    ix = cx + padding
    iy = cy + padding
    iw = max(1.0, cw - 2.0 * padding)
    ih = max(1.0, ch - 2.0 * padding)

    if abs(x_max - x_min) < 1e-9 or abs(y_max - y_min) < 1e-9:
        return ix + iw * 0.5, iy + ih * 0.5

    px = ix + (float(x) - x_min) / (x_max - x_min) * iw
    py = iy + (y_max - float(y)) / (y_max - y_min) * ih
    return float(px), float(py)


def _curve_feature_points(
    generated_functions: List[Dict[str, Any]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> List[Tuple[float, float]]:
    """Visible intercept / extremum points for generic readability scoring."""
    if not generated_functions:
        return []

    fn = generated_functions[0]
    ext = compute_extended_properties(
        fn["type"], fn["parameters"], fn["sampler"], x_min, x_max,
    )

    pts: List[Tuple[float, float]] = []
    intercepts = (ext.get("intercepts") or {})

    for xv in intercepts.get("x_intercepts") or []:
        try:
            xf = float(xv)
        except (TypeError, ValueError):
            continue
        if x_min <= xf <= x_max and y_min <= 0.0 <= y_max:
            pts.append((xf, 0.0))

    y_int = intercepts.get("y_intercept")
    if y_int is not None:
        try:
            yf = float(y_int)
        except (TypeError, ValueError):
            yf = None
        if yf is not None and x_min <= 0.0 <= x_max and y_min <= yf <= y_max:
            pts.append((0.0, yf))

    for item in ext.get("extrema") or []:
        if not isinstance(item, dict):
            continue
        try:
            xf = float(item.get("x"))
            yf = float(item.get("y"))
        except (TypeError, ValueError):
            continue
        if x_min <= xf <= x_max and y_min <= yf <= y_max:
            pts.append((xf, yf))

    deduped: List[Tuple[float, float]] = []
    seen = set()
    for x_val, y_val in pts:
        key = (round(x_val, 5), round(y_val, 5))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((x_val, y_val))
    return deduped


def _estimate_structure_readability(
    generated_functions: List[Dict[str, Any]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> float:
    """
    Generic pre-render readability score for all function types.

    Higher is better. We reward broad visible coverage and penalize charts whose
    curve mass or key features are crowded around the origin / label corridor.
    """
    if not generated_functions:
        return -1e9

    pts = sample_function(
        generated_functions[0], x_min, x_max, y_min, y_max, step=0.05,
    )
    if len(pts) < 3:
        return -1e9

    x_span = max(1e-6, float(x_max - x_min))
    y_span = max(1e-6, float(y_max - y_min))
    xs = [float(x) for x, _ in pts]
    ys = [float(y) for _, y in pts]
    x_cov = _clip01((max(xs) - min(xs)) / x_span)
    y_cov = _clip01((max(ys) - min(ys)) / y_span)

    center_x_band = max(0.9, 0.18 * x_span)
    center_y_band = max(0.9, 0.22 * y_span)
    edge_x_band = max(0.6, 0.08 * x_span)
    edge_y_band = max(0.6, 0.08 * y_span)
    axis_x_band = max(0.5, 0.08 * x_span)
    axis_y_band = max(0.5, 0.08 * y_span)

    center_density = sum(
        1 for x_val, y_val in pts
        if abs(float(x_val)) <= center_x_band and abs(float(y_val)) <= center_y_band
    ) / len(pts)

    edge_density = sum(
        1 for x_val, y_val in pts
        if (
            abs(float(x_val) - x_min) <= edge_x_band
            or abs(float(x_val) - x_max) <= edge_x_band
            or abs(float(y_val) - y_min) <= edge_y_band
            or abs(float(y_val) - y_max) <= edge_y_band
        )
    ) / len(pts)

    axis_density = sum(
        1 for x_val, y_val in pts
        if (
            (x_min <= 0.0 <= x_max and abs(float(x_val)) <= axis_x_band)
            or (y_min <= 0.0 <= y_max and abs(float(y_val)) <= axis_y_band)
        )
    ) / len(pts)

    feature_pts = _curve_feature_points(
        generated_functions, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
    )
    near_origin_features = 0.0
    if feature_pts:
        near_origin_features = sum(
            1 for x_val, y_val in feature_pts
            if abs(float(x_val)) <= center_x_band and abs(float(y_val)) <= center_y_band
        ) / len(feature_pts)

    return (
        0.95 * x_cov
        + 0.55 * y_cov
        - 0.90 * center_density
        - 0.45 * edge_density
        - 0.35 * axis_density
        - 0.30 * near_origin_features
    )


def _label_boxes(labels: List[Dict[str, Any]], pad: float) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    for label in labels:
        try:
            boxes.append((
                float(label["left"]) - pad,
                float(label["top"]) - pad,
                float(label["right"]) + pad,
                float(label["bottom"]) + pad,
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return boxes


def _fraction_points_in_boxes(
    points: List[Tuple[float, float]],
    boxes: List[Tuple[float, float, float, float]],
) -> float:
    if not points or not boxes:
        return 0.0
    hits = 0
    for px, py in points:
        for left, top, right, bottom in boxes:
            if left <= px <= right and top <= py <= bottom:
                hits += 1
                break
    return hits / len(points)


def _estimate_render_ocr_fragility(
    generated_functions: List[Dict[str, Any]],
    gt: Dict[str, Any],
    config: dict,
) -> float:
    """
    Cross-function OCR fragility estimate based on rendered geometry.

    This deliberately avoids function-type-specific branches so the same metric
    can steer linears, quadratics, exponentials, rationals, piecewise curves,
    and future function families.
    """
    if not generated_functions:
        return 0.0

    x_min, x_max = [float(v) for v in gt["x_range"]]
    y_min, y_max = [float(v) for v in gt["y_range"]]
    padding = int(config.get("chart", {}).get("padding_px", 80))
    pts = sample_function(
        generated_functions[0], x_min, x_max, y_min, y_max, step=0.03,
    )
    if len(pts) < 3:
        return 1.0

    pixel_points = [
        _math_to_pixel_from_gt(x_val, y_val, gt, padding)
        for x_val, y_val in pts
    ]

    all_label_boxes = _label_boxes(
        gt.get("x_tick_labels", []) + gt.get("y_tick_labels", []), pad=8.0,
    )
    x_label_boxes = _label_boxes(gt.get("x_tick_labels", []), pad=12.0)
    y_label_boxes = _label_boxes(gt.get("y_tick_labels", []), pad=12.0)

    label_overlap = _fraction_points_in_boxes(pixel_points, all_label_boxes)
    label_band_overlap = max(
        _fraction_points_in_boxes(pixel_points, x_label_boxes),
        _fraction_points_in_boxes(pixel_points, y_label_boxes),
    )

    cx, cy, cw, ch = [float(v) for v in gt["chart_bbox"]]
    ix = cx + padding
    iy = cy + padding
    iw = max(1.0, cw - 2.0 * padding)
    ih = max(1.0, ch - 2.0 * padding)
    edge_pad = max(10.0, min(iw, ih) * 0.025)

    edge_overlap = sum(
        1 for px, py in pixel_points
        if (
            px <= ix + edge_pad
            or px >= ix + iw - edge_pad
            or py <= iy + edge_pad
            or py >= iy + ih - edge_pad
        )
    ) / len(pixel_points)

    axis_overlap = 0.0
    origin_overlap = 0.0
    origin_box: Tuple[float, float, float, float] | None = None
    if x_min <= 0.0 <= x_max and y_min <= 0.0 <= y_max:
        origin_px, origin_py = _math_to_pixel_from_gt(0.0, 0.0, gt, padding)
        origin_half = max(26.0, min(iw, ih) * 0.04)
        origin_box = (
            origin_px - origin_half,
            origin_py - origin_half,
            origin_px + origin_half,
            origin_py + origin_half,
        )
        origin_overlap = _fraction_points_in_boxes(pixel_points, [origin_box])

        axis_half = max(8.0, min(iw, ih) * 0.012)
        x_axis_overlap = sum(
            1 for _, py in pixel_points if abs(py - origin_py) <= axis_half
        ) / len(pixel_points)
        y_axis_overlap = sum(
            1 for px, _ in pixel_points if abs(px - origin_px) <= axis_half
        ) / len(pixel_points)
        axis_overlap = max(x_axis_overlap, y_axis_overlap)

    feature_penalty = 0.0
    feature_points = _curve_feature_points(
        generated_functions, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
    )
    if feature_points:
        feature_pixels = [
            _math_to_pixel_from_gt(x_val, y_val, gt, padding)
            for x_val, y_val in feature_points
        ]
        label_hits = _fraction_points_in_boxes(feature_pixels, all_label_boxes)
        origin_hits = (
            _fraction_points_in_boxes(feature_pixels, [origin_box])
            if origin_box is not None else 0.0
        )
        feature_penalty = 0.35 * label_hits + 0.25 * origin_hits

    return _clip01(
        1.15 * label_overlap
        + 0.60 * label_band_overlap
        + 0.70 * origin_overlap
        + 0.32 * axis_overlap
        + 0.22 * edge_overlap
        + feature_penalty
    )


def _x_tick_tilt_deg(x_tick_labels: List[Dict[str, Any]]) -> float:
    """Signed tilt angle of the x-axis tick-label row."""
    if len(x_tick_labels) < 2:
        return 0.0
    first = x_tick_labels[0]
    last = x_tick_labels[-1]
    dx = float(last["center_x"] - first["center_x"])
    dy = float(last["center_y"] - first["center_y"])
    if abs(dx) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def _is_ocr_fragile_generic(
    pre_aug_fragility: float,
    gt: Dict[str, Any],
) -> bool:
    """
    Cross-function fallback guard.

    We only reject charts that are already structurally crowded *and* become
    noticeably tilted after augmentation, which is the regime where OCR is most
    likely to lose tick labels entirely.
    """
    if pre_aug_fragility >= GENERIC_HARD_FRAGILITY_THRESHOLD:
        return True
    if pre_aug_fragility < GENERIC_FRAGILITY_THRESHOLD:
        return False
    return abs(_x_tick_tilt_deg(gt.get("x_tick_labels", []))) >= 0.85


def _is_ocr_fragile_cubic(
    generated_functions: List[Dict[str, Any]],
    gt: Dict[str, Any],
) -> bool:
    """
    Empirical guard for a small family of cubic charts that repeatedly cause
    OCR to return a bare chart with no children_ids.

    The failures observed in the dataset share two traits:
      1. the cubic's turning structure is packed against one side of the y-axis
      2. the final photo-style warp tilts the x tick row noticeably downward

    We only reject these narrow cases to avoid over-pruning valid diversity.
    """
    if not generated_functions:
        return False
    fn = generated_functions[0]
    if fn.get("type") != "cubic":
        return False

    x_tilt = _x_tick_tilt_deg(gt.get("x_tick_labels", []))
    if x_tilt > -0.8:
        return False

    q = gt.get("question", {}).get("ground_truth", {})
    extrema = q.get("extrema") or []
    if len(extrema) != 2:
        return False

    try:
        ext_x = [float(e["x"]) for e in extrema]
        ext_y = [float(e["y"]) for e in extrema]
    except (KeyError, TypeError, ValueError):
        return False

    max_x = max(ext_x)
    min_x = min(ext_x)
    max_y = max(ext_y)
    min_y = min(ext_y)

    # Family A: the whole turning structure sits to the right of the y-axis
    # and remains below the x-axis, which is structurally brittle under degradation.
    if min_x > -0.25 and max_y < 0.0:
        return True

    # Family B: the turning structure sits on the left half-plane, straddles
    # the x-axis, and becomes brittle when the rendered chart tilts down
    # to the right.
    if max_x < 0.25 and min_y < 0.0 < max_y:
        return True

    return False


# ============================================================
# Phase 1 — Resolve coordinates and function parameters
# ============================================================

def _resolve_coordinates(
    config: dict,
    coords_cfg: dict,
    curves_cfg: dict,
    func_types: List[str],
) -> Tuple[List[Dict], List[dict], int, int, int, int, Dict[str, Any]]:
    """
    Select a function type, repeatedly generate random parameters and
    coordinate ranges, compute strict grid intersections, and return
    the richest readable candidate, preferring high-GT-intersection samples.

    Returns:
        (generated_functions, intersections,
         x_min, x_max, y_min, y_max, generation_meta)
    """
    coord_mode = coords_cfg.get("coord_mode", "manual")
    user_min = int(coords_cfg.get("min_intersections", 6))
    gen_cfg = _generation_cfg(config)
    candidate_attempts = int(gen_cfg.get("candidate_attempts", DEFAULT_CANDIDATE_ATTEMPTS))
    selection_pool_size = int(gen_cfg.get("selection_pool_size", DEFAULT_SELECTION_POOL_SIZE))
    selection_score_margin = float(gen_cfg.get("selection_score_margin", DEFAULT_SELECTION_SCORE_MARGIN))

    chosen_type = random.choice(func_types)
    targets = _intersection_targets(config, chosen_type, user_min)

    best_score = -1e12
    best_state: Optional[tuple] = None
    candidate_pool: List[Dict[str, Any]] = []

    generated_functions: List[Dict] = []
    intersections: List[dict] = []
    x_min = x_max = y_min = y_max = 0
    best_meta: Dict[str, Any] = {}

    for _attempt in range(max(MAX_RETRIES, candidate_attempts)):
        if coord_mode in ("static", "adaptive"):
            x_min, x_max, y_min, y_max, generated_functions = (
                _attempt_smart_coords(chosen_type, coord_mode)
            )
        else:
            x_min, x_max, y_min, y_max, generated_functions = (
                _attempt_manual_coords(coords_cfg, chosen_type)
            )

        result = _finalize_range(x_min, x_max, y_min, y_max)
        if result is None:
            continue
        x_min, x_max, y_min, y_max = result

        # Reject candidates whose visible shape would still be truncated after
        # rendering even if the function itself is mathematically valid.
        if generated_functions:
            _fn = generated_functions[0]
            _params = _fn.get("parameters", {})
            _ftype = _fn.get("type", "")
            if _ftype in ("quadratic", "absolute"):
                _k = _params.get("k", 0)
                if _k < y_min or _k > y_max:
                    continue
            if not _passes_visibility_guard(_fn, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max):
                continue

        sampler = generated_functions[0]["sampler"] if generated_functions else None
        intersections = (
            _compute_exact_grid_points(sampler, x_min, x_max, y_min, y_max)
            if sampler is not None else []
        )

        # Reject attempts where the curve has no visible points at all
        if sampler is not None:
            _vis_pts = sample_function(
                generated_functions[0], x_min, x_max, y_min, y_max, step=0.5,
            )
            if len(_vis_pts) < 3:
                continue

        attempt_readability = _estimate_structure_readability(
            generated_functions, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        )
        attempt_score, attempt_summary = _score_generation_candidate(
            generated_functions,
            intersections,
            x_min,
            x_max,
            y_min,
            y_max,
            attempt_readability,
            targets,
        )

        if attempt_score > best_score:
            best_score = attempt_score
            best_state = (
                list(generated_functions), list(intersections),
                x_min, x_max, y_min, y_max,
            )
            best_meta = {
                "candidate_score": round(float(attempt_score), 5),
                "structure_readability": round(float(attempt_readability), 5),
                "targets": dict(targets),
                "summary": dict(attempt_summary),
            }
        candidate_pool.append({
            "score": float(attempt_score),
            "state": (
                list(generated_functions), list(intersections),
                x_min, x_max, y_min, y_max,
            ),
            "meta": {
                "candidate_score": round(float(attempt_score), 5),
                "structure_readability": round(float(attempt_readability), 5),
                "targets": dict(targets),
                "summary": dict(attempt_summary),
            },
        })

        if (
            len(intersections) >= targets["preferred_min"]
            and attempt_readability >= COORD_READABILITY_TARGET
            and attempt_summary.get("x_coverage", 0.0) >= 0.35
        ):
            break

    # Fallback: use the best valid attempt, or a safe default range
    if candidate_pool:
        candidate_pool.sort(key=lambda item: item["score"], reverse=True)
        threshold = candidate_pool[0]["score"] - selection_score_margin
        top_pool = [item for item in candidate_pool if item["score"] >= threshold][:max(1, selection_pool_size)]
        chosen_item = random.choice(top_pool)
        (generated_functions, intersections,
         x_min, x_max, y_min, y_max) = chosen_item["state"]
        best_meta = chosen_item["meta"]
    elif best_state is not None:
        (generated_functions, intersections,
         x_min, x_max, y_min, y_max) = best_state
    else:
        x_min, x_max, y_min, y_max = -7, 7, -7, 7
        fn = generate_random_function(chosen_type, x_min, x_max, y_min, y_max)
        generated_functions = [fn]
        intersections = _compute_exact_grid_points(
            fn["sampler"], x_min, x_max, y_min, y_max,
        )
        fallback_readability = _estimate_structure_readability(
            generated_functions, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        )
        fallback_score, fallback_summary = _score_generation_candidate(
            generated_functions,
            intersections,
            x_min,
            x_max,
            y_min,
            y_max,
            fallback_readability,
            targets,
        )
        best_meta = {
            "candidate_score": round(float(fallback_score), 5),
            "structure_readability": round(float(fallback_readability), 5),
            "targets": dict(targets),
            "summary": dict(fallback_summary),
        }

    return generated_functions, intersections, x_min, x_max, y_min, y_max, best_meta


def _attempt_smart_coords(
    chosen_type: str,
    coord_mode: str,
) -> Tuple[int, int, int, int, List[Dict]]:
    """One attempt at generating coordinates in 'static' or 'adaptive' mode.

    Static:   pool range  → quick feature scan → _budget_pad
    Adaptive: pool range  → dense sampling      → _budget_pad
    """
    init_x, init_y = _pick_recommended_range(chosen_type)
    x_min, x_max = init_x[0], init_x[1]
    y_min, y_max = init_y[0], init_y[1]

    fn = generate_random_function(chosen_type, x_min, x_max, y_min, y_max)

    if coord_mode == "adaptive":
        (x_min, x_max), (y_min, y_max) = _compute_adaptive_range(
            fn["sampler"], (x_min, x_max), (y_min, y_max),
        )
    else:
        # Static: sample near origin only to detect bounded peaks/troughs
        # without being misled by unbounded tails at range edges.
        y_vals = []
        for x in range(max(x_min, -ORIGIN_GUARD), min(x_max, ORIGIN_GUARD) + 1):
            v = fn["sampler"](float(x))
            if v is not None and np.isfinite(v) and abs(v) < 1e6:
                y_vals.append(v)
        if y_vals:
            feat_lo, feat_hi = min(y_vals), max(y_vals)
            if feat_lo < y_min or feat_hi > y_max:
                adj_lo, adj_hi = _budget_pad(feat_lo, feat_hi)
                if (adj_hi - adj_lo) <= MAX_AXIS_SPAN:
                    y_min, y_max = adj_lo, adj_hi

    return x_min, x_max, y_min, y_max, [fn]


def _attempt_manual_coords(
    coords_cfg: dict,
    chosen_type: str,
) -> Tuple[int, int, int, int, List[Dict]]:
    """One attempt at generating coordinates in 'manual' mode."""
    x_min = int(coords_cfg["x_range"][0])
    x_max = int(coords_cfg["x_range"][1])
    y_min = int(coords_cfg["y_range"][0])
    y_max = int(coords_cfg["y_range"][1])

    fn = generate_random_function(chosen_type, x_min, x_max, y_min, y_max)
    return x_min, x_max, y_min, y_max, [fn]


# ============================================================
# Phase 2 — Render the image
# ============================================================

def _render_image(
    config: dict,
    generated_functions: List[Dict],
    x_min: int, x_max: int, y_min: int, y_max: int,
    intersections: List[dict],
    layout: str,
) -> Tuple[np.ndarray, Tuple, list, list, dict]:
    """
    Create the canvas and draw stem, axes, grid, and curves.

    Returns:
        (img, chart_rect, x_tick_labels, y_tick_labels, question_data)
    """
    page_cfg = config.get("page", {})
    chart_cfg = config.get("chart", {})
    curves_cfg = config.get("curves", {})
    stem_cfg = config.get("stem", {})

    page_w = int(page_cfg.get("width", 1920))
    page_h = int(page_cfg.get("height", 1200))
    margin = int(page_cfg.get("margin_px", 40))
    stem_font_size = int(stem_cfg.get("font_size", 28))

    # --- Question / stem text ---
    first_fn = generated_functions[0] if generated_functions else None
    fn_meta_for_q = {}
    if first_fn:
        fn_props = compute_properties(
            first_fn["type"], first_fn["parameters"],
            first_fn.get("expression_sympy", ""),
        )
        fn_meta_for_q = {
            "expression_latex": first_fn.get("expression_latex", ""),
            "expression_sympy": first_fn.get("expression_sympy", ""),
            "math_properties": fn_props,
        }
    question_lang = config.get("stem", {}).get("lang", "en")
    question_data = generate_question(fn_meta_for_q, intersections=None, lang=question_lang)
    stem_lines = question_data.get("stem_lines", [""])
    stem_max_lines = int(stem_cfg.get("max_lines", 5))

    if layout == "top_bottom":
        estimated_stem_w = page_w - 2 * margin
    else:
        avail_w = page_w - 2 * margin
        estimated_stem_w = estimate_stem_width(stem_font_size, 30)
        estimated_stem_w = max(120, min(estimated_stem_w, avail_w // 3))
    estimated_text_w = max(80, estimated_stem_w - 20)
    stem_text_lines = max(
        1,
        estimate_wrapped_line_count(
            stem_lines,
            stem_font_size,
            estimated_text_w,
            max_lines=stem_max_lines,
        ),
    )

    cx, cy, cw, ch = chart_bbox(
        page_w, page_h, layout,
        float(chart_cfg["area_ratio_range"][0]),
        float(chart_cfg["area_ratio_range"][1]),
        float(chart_cfg["aspect_ratio_range"][0]),
        float(chart_cfg["aspect_ratio_range"][1]),
        margin,
        stem_text_lines=stem_text_lines,
        stem_font_size=stem_font_size,
    )
    stem_x, stem_y, stem_w, stem_h = stem_bbox(
        page_w, page_h, layout, cx, cy, cw, ch, margin,
    )
    chart_rect = (cx, cy, cw, ch)

    # --- Draw ---
    img = np.ones((page_h, page_w, 3), dtype=np.uint8) * 255

    draw_stem(
        img, (stem_x, stem_y, stem_w, stem_h),
        stem_lines,
        stem_font_size,
        tuple(int(v) for v in stem_cfg.get("color", [30, 30, 30])),
        stem_max_lines,
    )
    draw_chart_border(img, chart_rect, config)

    x_ticks = _tick_values(x_min, x_max)
    y_ticks = _tick_values(y_min, y_max)

    x_tick_labels, y_tick_labels = draw_axes_and_grid(
        img, chart_rect,
        (x_min, x_max), (y_min, y_max),
        x_ticks, y_ticks, config,
    )

    curves_pts = [
        sample_function(fn, x_min, x_max, y_min, y_max, CURVE_SAMPLE_STEP)
        for fn in generated_functions
    ]
    curve_color = tuple(int(v) for v in curves_cfg.get("color", [0, 0, 0]))
    line_width = int(curves_cfg.get("line_width", 3))
    padding = int(chart_cfg.get("padding_px", 50))

    draw_curves(
        img, chart_rect,
        (x_min, x_max), (y_min, y_max),
        curves_pts, curve_color, line_width, padding,
    )

    return img, chart_rect, x_tick_labels, y_tick_labels, question_data


def _clip_monotonicity_to_visible(
    intervals: List[Dict[str, Any]],
    x_min: float,
    x_max: float,
) -> List[Dict[str, Any]]:
    """Clip monotonicity intervals to visible x-range [x_min, x_max]."""
    out: List[Dict[str, Any]] = []
    for item in intervals or []:
        if not isinstance(item, dict):
            continue
        interval = item.get("interval")
        direction = item.get("direction")
        if not isinstance(interval, (list, tuple)) or len(interval) < 2:
            continue
        if direction not in ("increasing", "decreasing"):
            continue
        try:
            lo = float(interval[0])
            hi = float(interval[1])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(lo):
            lo = float(x_min)
        if not np.isfinite(hi):
            hi = float(x_max)
        lo = max(lo, float(x_min))
        hi = min(hi, float(x_max))
        if hi - lo < 1e-9:
            continue
        out.append({
            "interval": [round(lo, 6), round(hi, 6)],
            "direction": direction,
        })
    return out


def _build_visible_facts(
    ext: Dict[str, Any],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Dict[str, Any]:
    """Build strict visible-range facts aligned to rendered chart bounds."""
    src_ints = (ext or {}).get("intercepts", {}) or {}

    x_ints = []
    if y_min <= 0 <= y_max:
        x_ints = [
            float(v) for v in (src_ints.get("x_intercepts") or [])
            if x_min <= float(v) <= x_max
        ]
    x_ints = sorted(set(round(v, 6) for v in x_ints))

    y_int = src_ints.get("y_intercept")
    if y_int is not None:
        try:
            yv = float(y_int)
            if not (x_min <= 0 <= x_max and y_min <= yv <= y_max):
                y_int = None
            else:
                y_int = round(yv, 6)
        except (TypeError, ValueError):
            y_int = None

    extrema_vis: List[Dict[str, Any]] = []
    for e in (ext or {}).get("extrema", []) or []:
        if not isinstance(e, dict):
            continue
        try:
            xv = float(e.get("x"))
            yv = float(e.get("y"))
        except (TypeError, ValueError):
            continue
        if x_min <= xv <= x_max and y_min <= yv <= y_max:
            extrema_vis.append({
                "x": round(xv, 6),
                "y": round(yv, 6),
                "type": e.get("type"),
            })

    mono_vis = _clip_monotonicity_to_visible(
        (ext or {}).get("monotonicity", []) or [],
        x_min=x_min,
        x_max=x_max,
    )
    return {
        "intercepts": {"x_intercepts": x_ints, "y_intercept": y_int},
        "extrema": extrema_vis,
        "monotonicity": mono_vis,
    }


# ============================================================
# Phase 3 — Assemble ground truth
# ============================================================

def _assemble_ground_truth(
    generated_functions: List[Dict],
    intersections: List[dict],
    x_min: int, x_max: int, y_min: int, y_max: int,
    chart_rect: Tuple, page_size: Tuple[int, int], layout: str,
    x_tick_labels: list, y_tick_labels: list,
    question_data: dict,
) -> Dict[str, Any]:
    """Build the complete ground-truth dictionary."""

    def _label_dict(L: dict) -> dict:
        return {
            "value": L["value"], "text": L["text"],
            "center_x": L["center_x"], "center_y": L["center_y"],
            "left": L["left"], "top": L["top"],
            "right": L["right"], "bottom": L["bottom"],
        }

    functions_meta = []
    for fn in generated_functions:
        fn_props = compute_properties(
            fn["type"], fn["parameters"], fn.get("expression_sympy", ""),
        )
        functions_meta.append({
            "type": fn["type"],
            "parameters": fn["parameters"],
            "expression_latex": fn["expression_latex"],
            "expression_sympy": fn["expression_sympy"],
            "math_properties": fn_props,
        })

    gt_data = question_data["ground_truth"]
    gt_data["coordinates"] = [
        [p["coordinate_x"], p["coordinate_y"]] for p in intersections
    ]

    sampler = generated_functions[0]["sampler"] if generated_functions else None
    pred_targets, rate_of_change = _generate_extra_tasks(intersections, sampler)
    gt_data["prediction_targets"] = pred_targets
    gt_data["rate_of_change"] = rate_of_change

    # Extended properties for three-layer evaluation (L1/L2)
    if generated_functions:
        fn = generated_functions[0]
        ext = compute_extended_properties(
            fn["type"], fn["parameters"], fn["sampler"], x_min, x_max,
        )
        gt_data["intercepts"] = ext["intercepts"]
        gt_data["extrema"] = ext["extrema"]
        gt_data["monotonicity"] = ext["monotonicity"]
        gt_data["symmetry"] = ext["symmetry"]
        # Strict visible-range fact GT for GeoSym fact-diagnosis experiments.
        # These are clipped to rendered chart bounds to match recognizer outputs
        # point-by-point (especially for monotonicity interval endpoints).
        gt_data["visible_facts"] = _build_visible_facts(
            ext,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
        )

    return {
        "chart_bbox": list(chart_rect),
        "page_size": list(page_size),
        "layout": layout,
        "x_range": [x_min, x_max],
        "y_range": [y_min, y_max],
        "x_tick_values": _tick_values(x_min, x_max),
        "y_tick_values": _tick_values(y_min, y_max),
        "intersections": intersections,
        "x_tick_labels": [_label_dict(L) for L in x_tick_labels],
        "y_tick_labels": [_label_dict(L) for L in y_tick_labels],
        "functions": functions_meta,
        "question": {
            "question_type": "comprehensive",
            "question_text": question_data["question_text"],
            "question_text_zh": question_data["question_text_zh"],
            "ground_truth": gt_data,
        },
    }


# ============================================================
# Public API
# ============================================================

def generate_one(
    config: dict,
    output_image_path: str | Path | None = None,
    output_gt_path: str | Path | None = None,
    layout_override: str | None = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate one full-page image and its ground truth.

    Three phases:
      1. _resolve_coordinates  — pick function, retry for intersections
      2. _render_image         — draw stem, axes, grid, curves
      3. _assemble_ground_truth — build the GT JSON structure
    """
    coords_cfg = config.get("coords", {})
    curves_cfg = config.get("curves", {})
    page_cfg = config.get("page", {})

    func_types = curves_cfg.get(
        "function_types",
        ["linear", "quadratic", "absolute", "cubic", "piecewise_linear",
         "exponential", "logarithmic", "rational"],
    )

    best_result: Tuple[np.ndarray, Dict[str, Any]] | None = None
    best_guard_risk = float("inf")

    for attempt in range(GENERATION_GUARD_RETRIES):
        attempt_seed = None if seed is None else seed + attempt
        if attempt_seed is not None:
            random.seed(attempt_seed)

        layout = layout_override or choose_layout(config)

        (generated_functions, intersections,
         x_min, x_max, y_min, y_max, generation_meta) = _resolve_coordinates(
            config, coords_cfg, curves_cfg, func_types,
        )

        page_w = int(page_cfg.get("width", 1920))
        page_h = int(page_cfg.get("height", 1200))

        img, chart_rect, x_tick_labels, y_tick_labels, question_data = (
            _render_image(
                config, generated_functions,
                x_min, x_max, y_min, y_max,
                intersections, layout,
            )
        )

        gt = _assemble_ground_truth(
            generated_functions, intersections,
            x_min, x_max, y_min, y_max,
            chart_rect, (page_w, page_h), layout,
            x_tick_labels, y_tick_labels,
            question_data,
        )
        gt["generation_meta"] = generation_meta

        pre_aug_fragility = _estimate_render_ocr_fragility(
            generated_functions, gt, config,
        )
        img, gt = apply_render_degradation(
            img, gt, config, seed=attempt_seed, fragility=pre_aug_fragility,
        )

        tilt_penalty = max(0.0, abs(_x_tick_tilt_deg(gt.get("x_tick_labels", []))) - 0.8)
        guard_risk = pre_aug_fragility + 0.10 * tilt_penalty
        if best_result is None or guard_risk < best_guard_risk:
            best_result = (img, gt)
            best_guard_risk = guard_risk

        if (
            _is_ocr_fragile_generic(pre_aug_fragility, gt)
            or _is_ocr_fragile_cubic(generated_functions, gt)
        ):
            continue
        break

    if best_result is None:
        raise RuntimeError("Image generation failed to produce a valid chart")

    img, gt = best_result

    if output_image_path is not None:
        Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image_path), img)
    if output_gt_path is not None:
        Path(output_gt_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_gt_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=2, ensure_ascii=False)

    return img, gt


class ImageGenerator:
    def __init__(self, config_path: str | Path | None = None):
        self.config = load_config(config_path)

    def generate(
        self,
        output_image_path: str | Path | None = None,
        output_gt_path: str | Path | None = None,
        layout_override: str | None = None,
        seed: int | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return generate_one(
            self.config,
            output_image_path=output_image_path,
            output_gt_path=output_gt_path,
            layout_override=layout_override,
            seed=seed,
        )
