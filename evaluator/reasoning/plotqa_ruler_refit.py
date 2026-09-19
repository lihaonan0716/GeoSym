"""Refit the PlotQA value axis from OCR'd tick labels (M0' work item W1).

Why this module exists
----------------------
Diagnosis against the official PlotQA annotations located the accuracy
bottleneck precisely (see ``docs/PLOTQA_M0_REBUILD_PLAN_2026-09-18.md``):

- mark detection recall is 0.750 -- segmentation mostly finds the marks;
- but the value accuracy of *detected* marks is 0.160;
- and for 132/274 samples the measured/truth ratio is **constant within the
  sample**, which is the signature of a mis-calibrated axis rather than
  per-mark read-out noise.

Tracing those constant-ratio samples showed the axis scale came from guessing
paths (``nice_grid_top_from_ocr_digits``, ``unit_grid_heuristic``,
``small_tick_step_2_grid_heuristic``) that exist only because the previous OCR
backend recovered the printed ticks unreliably (numeric recall 0.422). Under
PaddleOCR that recall is 0.961, so the axis can be fitted from the ticks that
are actually printed on the chart.

The design rule this module follows
-----------------------------------
**Fit from read ticks, or declare failure. Never guess.** A guessed axis
produces a confident-looking wrong number, and evidence that looks trustworthy
but is wrong is worse than absent evidence -- it displaces a correct visual
reading. Every returned ruler therefore states its provenance and whether it is
admissible.

Answer independence: nothing here reads a reference answer. Inputs are the OCR
boxes, the plot geometry, and the measured pixel positions.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# A tick label must sit within this fraction of the plot height/width of the
# axis it belongs to, measured from the axis line outward.
Y_AXIS_BAND_FRACTION = 0.14
X_AXIS_BAND_FRACTION = 0.22
# How far *inside* the fitted plot area a tick label may sit and still belong to
# the axis. The plot-area bbox is itself estimated, and OCR centres a label on
# its tick, so a label can land a little inside the nominal edge.
Y_AXIS_INWARD_FRACTION = 0.02
X_AXIS_INWARD_FRACTION = 0.04
# Two ticks define a line exactly, so the fit carries no evidence of its own
# quality below three. Samples with fewer are reported unfittable rather than
# fitted, which is the whole point of this module.
MIN_TICKS_FOR_FIT = 3
# Maximum tolerated median residual of the tick fit, as a fraction of the axis
# value span. Above this the ticks disagree among themselves and the axis is
# not admissible.
MAX_RESIDUAL_FRACTION = 0.02
# Ticks closer than this in pixels describe one label read twice.
TICK_DEDUPE_PX = 6.0

# The exponent branch is required, not cosmetic. PlotQA labels large-magnitude
# axes in scientific E-notation ("2.000e+8"), and without it such a label splits
# into two tokens and is rejected as ambiguous. That silently emptied the value
# axis on 38 of the frozen roster's samples while OCR had read every tick
# correctly -- the ticks were discarded after being read, not missed.
NUMERIC_TOKEN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _numeric(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _bbox(raw: Any) -> Optional[List[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return [float(v) for v in raw]
    except (TypeError, ValueError):
        return None


def _center(bbox: Sequence[float]) -> Tuple[float, float]:
    return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0


def parse_tick_value(text: str) -> Optional[float]:
    """Parse a tick label into a single number, or None.

    A label carrying more than one number is ambiguous as an axis anchor (it is
    usually a wrapped axis title or a mis-merged pair of labels), so it is
    rejected rather than guessed at. Thousands separators are handled because
    PlotQA renders large values with commas.
    """
    cleaned = str(text or "").strip().replace(",", "")
    if not cleaned:
        return None
    # Percent and currency decorations are axis units, not part of the number.
    cleaned = cleaned.replace("%", "").replace("$", "").strip()
    matches = NUMERIC_TOKEN.findall(cleaned)
    if len(matches) != 1:
        return None
    # Reject labels that are mostly letters with an incidental digit.
    letters = sum(1 for char in cleaned if char.isalpha())
    if letters > 1:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def collect_axis_ticks(
    ocr_boxes: Iterable[Dict[str, Any]],
    plot_area: Sequence[float],
    axis: str,
) -> List[Dict[str, Any]]:
    """Collect numeric tick labels belonging to one axis.

    ``axis`` is ``"y"`` for the left value axis or ``"x"`` for the bottom axis.
    Selection is geometric: a y tick sits left of the plot and vertically inside
    it, an x tick sits below the plot and horizontally inside it.
    """
    if len(plot_area) != 4:
        return []
    left, top, right, bottom = (float(v) for v in plot_area)
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)

    ticks: List[Dict[str, Any]] = []
    for box in ocr_boxes or []:
        bbox = _bbox(box.get("bbox"))
        if bbox is None:
            continue
        value = parse_tick_value(box.get("text", ""))
        if value is None:
            continue
        cx, cy = _center(bbox)
        if axis == "y":
            # Left of the plot area, within a band, and vertically inside. The
            # inward tolerance is a fraction of the plot width rather than a few
            # pixels: OCR centres labels on the tick, so a two-digit label on a
            # tight left margin sits slightly inside the fitted plot-area edge.
            if not (left - width * Y_AXIS_BAND_FRACTION <= cx <= left + width * Y_AXIS_INWARD_FRACTION):
                continue
            if not (top - 6.0 <= cy <= bottom + 6.0):
                continue
            pixel = cy
        else:
            if not (bottom - height * X_AXIS_INWARD_FRACTION <= cy <= bottom + height * X_AXIS_BAND_FRACTION):
                continue
            if not (left - 6.0 <= cx <= right + 6.0):
                continue
            pixel = cx
        ticks.append({
            "value": value,
            "pixel": pixel,
            "bbox": bbox,
            "text": str(box.get("text", "")).strip(),
            "confidence": box.get("confidence"),
        })

    # One printed label can be detected twice; keep the higher-confidence read.
    ticks.sort(key=lambda t: t["pixel"])
    deduped: List[Dict[str, Any]] = []
    for tick in ticks:
        if deduped and abs(tick["pixel"] - deduped[-1]["pixel"]) < TICK_DEDUPE_PX:
            previous = deduped[-1]
            if (_numeric(tick.get("confidence")) or 0.0) > (_numeric(previous.get("confidence")) or 0.0):
                deduped[-1] = tick
            continue
        deduped.append(tick)
    return deduped


def _least_squares(pairs: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    n = len(pairs)
    if n < 2:
        return None
    sum_p = sum(p for p, _ in pairs)
    sum_v = sum(v for _, v in pairs)
    sum_pp = sum(p * p for p, _ in pairs)
    sum_pv = sum(p * v for p, v in pairs)
    denominator = n * sum_pp - sum_p * sum_p
    if abs(denominator) < 1e-12:
        return None
    slope = (n * sum_pv - sum_p * sum_v) / denominator
    intercept = (sum_v - slope * sum_p) / n
    return slope, intercept


def fit_value_axis(ticks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Fit pixel -> value from tick labels, dropping at most one outlier.

    A single mis-read tick is common and would otherwise tilt the whole axis, so
    one leave-one-out retry is allowed when it materially improves the residual.
    Dropping more than one would let the fit choose its own data, which is how a
    wrong axis starts looking well-supported.
    """
    pairs = [(float(t["pixel"]), float(t["value"])) for t in ticks]
    if len(pairs) < MIN_TICKS_FOR_FIT:
        return {
            "admissible": False,
            "reason": "insufficient_ticks",
            "tick_count": len(pairs),
        }

    def evaluate(subset: Sequence[Tuple[float, float]]) -> Optional[Dict[str, Any]]:
        fit = _least_squares(subset)
        if fit is None:
            return None
        slope, intercept = fit
        values = [v for _, v in subset]
        span = max(values) - min(values)
        if span <= 0:
            return None
        residuals = [abs(v - (slope * p + intercept)) for p, v in subset]
        return {
            "slope": slope,
            "intercept": intercept,
            "residual_fraction": statistics.median(residuals) / span,
            "max_residual_fraction": max(residuals) / span,
            "tick_count": len(subset),
        }

    best = evaluate(pairs)
    dropped: Optional[float] = None
    if best is not None and best["max_residual_fraction"] > MAX_RESIDUAL_FRACTION and len(pairs) > MIN_TICKS_FOR_FIT:
        for index in range(len(pairs)):
            subset = pairs[:index] + pairs[index + 1:]
            candidate = evaluate(subset)
            if candidate is None:
                continue
            if candidate["max_residual_fraction"] < best["max_residual_fraction"]:
                best, dropped = candidate, pairs[index][1]

    if best is None:
        return {"admissible": False, "reason": "degenerate_tick_geometry", "tick_count": len(pairs)}

    admissible = best["max_residual_fraction"] <= MAX_RESIDUAL_FRACTION
    return {
        "admissible": admissible,
        "reason": "fitted_from_ocr_ticks" if admissible else "tick_residual_too_high",
        "slope": best["slope"],
        "intercept": best["intercept"],
        "residual_fraction": best["residual_fraction"],
        "max_residual_fraction": best["max_residual_fraction"],
        "tick_count": best["tick_count"],
        "dropped_outlier_value": dropped,
        "provenance": "ocr_tick_least_squares",
    }


def pixel_to_value(axis_fit: Dict[str, Any], pixel: float) -> Optional[float]:
    """Map a pixel coordinate through an admissible axis fit.

    Returns None for an inadmissible fit. That is the mechanism by which this
    module refuses to guess: callers get no number rather than a wrong one.
    """
    if not axis_fit.get("admissible"):
        return None
    slope = _numeric(axis_fit.get("slope"))
    intercept = _numeric(axis_fit.get("intercept"))
    if slope is None or intercept is None:
        return None
    return slope * float(pixel) + intercept


# A mark whose bbox is taller than this multiple of its width is a bar grown
# from the axis baseline rather than a point marker. Bars encode their value at
# the growing edge, point markers at their centre; reading a bar at its centre
# halves the value, which is exactly the 0.499 ratio the first W1 evaluation
# showed for vertical bars.
BAR_ASPECT_RATIO = 1.6
# Separates horizontal bars (which share a value-axis baseline, so their left
# edges barely spread) from polyline components (whose segments start all across
# the plot). See ``value_axis_orientation``.
BASELINE_SPREAD_MAX_RATIO = 2.5


def value_pixel_for_mark(
    point: Dict[str, Any],
    orientation: str = "y",
    axis_fit: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Pick the pixel coordinate along ``orientation`` that carries the value.

    Point markers (dots, line samples, short marks) are read at their centre.
    Bars are read at their growing edge -- the one away from the value-axis
    baseline. Reading a bar at its centre halves its value, which is exactly the
    0.499 ratio the first W1 evaluation showed for vertical bars.

    Choosing the growing edge is done geometrically, from where value zero falls
    on the fitted axis: the growing edge is whichever bbox edge is *farther*
    from the zero line. This handles bars that hang the other way -- 33 samples
    on the frozen roster plot negative values -- without needing to know the
    sign in advance.

    The extractor's ``pixel`` anchor is deliberately not consulted. It was tried
    first and is actively misleading on horizontal bars, where it sits on the
    baseline for 3748 of 3829 components; snapping to the nearest edge therefore
    picked the zero end, giving a median measured/truth ratio of 0.0039 against
    1.003 for the correct edge, and hbar value accuracy of 0.032.

    Without a usable fit the fallback is the conventional direction: vertical
    bars rise from a bottom baseline (top edge, ``low`` in image coordinates),
    horizontal bars extend rightward (right edge, ``high``).
    """
    index = 1 if orientation == "y" else 0
    bbox = _bbox(point.get("component_bbox"))
    center = point.get("component_center_pixel") or point.get("pixel")
    center_value = None
    if isinstance(center, (list, tuple)) and len(center) == 2:
        center_value = _numeric(center[index])

    if bbox is None:
        return center_value

    low, high = float(bbox[index]), float(bbox[index + 2])
    extent = high - low
    cross = float(bbox[1 - index + 2]) - float(bbox[1 - index])
    if extent <= 0 or cross <= 0 or extent < cross * BAR_ASPECT_RATIO:
        # Point-like along this axis: the centre is the measurement.
        return center_value if center_value is not None else (low + high) / 2.0

    zero_pixel = _zero_pixel(axis_fit)
    if zero_pixel is not None:
        return low if abs(low - zero_pixel) >= abs(high - zero_pixel) else high
    return low if orientation == "y" else high


def _zero_pixel(axis_fit: Optional[Dict[str, Any]]) -> Optional[float]:
    """Pixel coordinate where the fitted axis reads zero, if it is admissible."""
    if not axis_fit or not axis_fit.get("admissible"):
        return None
    slope = _numeric(axis_fit.get("slope"))
    intercept = _numeric(axis_fit.get("intercept"))
    if slope is None or intercept is None or slope == 0:
        return None
    return -intercept / slope


def value_axis_orientation(record: Dict[str, Any]) -> str:
    """Which axis carries the measured quantity: ``"y"`` or ``"x"``.

    On a horizontal bar chart the categories run down the left side and the
    value axis is along the bottom, so reading the left axis yields category
    positions rather than values. Orientation is inferred from mark geometry --
    bars wider than they are tall are horizontal -- which works even when the
    recorded ``chart_type`` is coarse or absent.
    """
    points = list(record.get("measured_points") or [])

    # A horizontal bar starts on the value-axis baseline, so every bar in a
    # series shares a left edge and differs at the right. A line chart's
    # connected component is also a wide blob, but its left edges are spread
    # across the plot -- that difference is what separates the two, whereas
    # aspect ratio alone misreads a polyline as a bar.
    wide_boxes = []
    tall_boxes = []
    for point in points:
        bbox = _bbox(point.get("component_bbox"))
        if bbox is None:
            continue
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        if width <= 0 or height <= 0:
            continue
        if width >= height * BAR_ASPECT_RATIO:
            wide_boxes.append(bbox)
        elif height >= width * BAR_ASPECT_RATIO:
            tall_boxes.append(bbox)

    if len(tall_boxes) >= len(wide_boxes) or len(wide_boxes) < 2:
        return "y"

    left_edges = [float(b[0]) for b in wide_boxes]
    spread = max(left_edges) - min(left_edges)
    widths = [float(b[2]) - float(b[0]) for b in wide_boxes]
    median_width = statistics.median(widths)
    if median_width <= 0:
        return "y"
    # Measured on the frozen roster, this ratio separates the two populations
    # cleanly: horizontal bars run 0.00-2.53 (p10-p90) because they share a
    # baseline, while polyline components run 2.90-88.0 because their segments
    # start all across the plot. The cut sits in the empty gap between them.
    return "x" if spread / median_width <= BASELINE_SPREAD_MAX_RATIO else "y"


def ticks_look_like_years(values: Sequence[float]) -> bool:
    """True when a tick set is a year category axis rather than a value axis.

    PlotQA labels its category axis with years far more often than not, and a
    year axis fits a straight line just as well as a value axis does -- so fit
    quality alone cannot tell them apart. The value range can: a quantity axis
    does not consist mostly of integers in 1800-2200.
    """
    if not values:
        return False
    yearlike = sum(1 for v in values if 1800.0 <= float(v) <= 2200.0)
    return yearlike >= max(2, int(len(values) * 0.6))


def resolve_value_axis(
    record: Dict[str, Any],
    ocr_row: Dict[str, Any],
    plot_area: Sequence[float],
) -> tuple[str, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Choose the value axis, then fit it. Returns (axis, ticks, fit, trace).

    Mark geometry supplies the prior because it is right 93.8% of the time on
    the frozen roster. Tick evidence is consulted only as a fallback: if the
    prior's axis yields no admissible *value* fit, the other axis is tried.

    The ordering matters and was measured. Deciding purely on tick evidence --
    fit both axes, keep the better -- scores 84.8%, worse than geometry alone,
    because a year category axis fits beautifully and wins on residual.
    Geometry-first with an evidence fallback scores 98.8%.
    """
    prior = value_axis_orientation(record)
    other = "x" if prior == "y" else "y"
    boxes = ocr_row.get("ocr_boxes") or []

    prior_ticks = collect_axis_ticks(boxes, plot_area, axis=prior)
    prior_fit = fit_value_axis(prior_ticks)
    prior_usable = prior_fit.get("admissible") and not ticks_look_like_years(
        [t["value"] for t in prior_ticks])
    if prior_usable:
        return prior, prior_ticks, prior_fit, {
            "prior": prior, "resolved": prior, "fallback_used": False}

    other_ticks = collect_axis_ticks(boxes, plot_area, axis=other)
    other_fit = fit_value_axis(other_ticks)
    other_usable = other_fit.get("admissible") and not ticks_look_like_years(
        [t["value"] for t in other_ticks])
    if other_usable:
        return other, other_ticks, other_fit, {
            "prior": prior, "resolved": other, "fallback_used": True,
            "prior_rejected_because": (
                "year_category_axis" if prior_fit.get("admissible") else prior_fit.get("reason"))}

    # Neither axis yields a usable value fit. Report the prior's failure, since
    # that is the axis the geometry says should have carried the value.
    return prior, prior_ticks, prior_fit, {
        "prior": prior, "resolved": prior, "fallback_used": False,
        "both_axes_unusable": True}


def refit_record(
    record: Dict[str, Any],
    ocr_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a copy of ``record`` with values recomputed from a refitted axis.

    The frozen input is never mutated: it is an input to a frozen experiment.
    Points whose value cannot be recomputed admissibly are kept but have their
    ``y_value`` set to None and are marked, so a downstream stage can tell
    "unmeasured" from "measured as zero".
    """
    geometry = record.get("chart_geometry") or {}
    plot_area = geometry.get("plot_area_bbox")
    if not ocr_row or not plot_area or len(plot_area) != 4:
        return {
            **record,
            "ruler_refit": {"admissible": False, "reason": "missing_ocr_or_geometry"},
        }

    orientation, ticks, axis_fit, axis_trace = resolve_value_axis(record, ocr_row, plot_area)

    points: List[Dict[str, Any]] = []
    recomputed = 0
    for original in record.get("measured_points") or []:
        point = dict(original)
        pixel = value_pixel_for_mark(point, orientation, axis_fit)
        new_value = pixel_to_value(axis_fit, pixel) if pixel is not None else None
        if new_value is None:
            point["y_value"] = None
            point["measured_value"] = None
            point["value_status"] = "unmeasurable_axis_not_admissible"
        else:
            # ``measured_value`` is the orientation-independent field: it holds
            # the quantity regardless of which axis encodes it. ``y_value`` is
            # kept in step for consumers written against the frozen schema,
            # including on horizontal charts where the value came from x.
            point["measured_value"] = new_value
            point["y_value"] = new_value
            point["value_status"] = "refit_from_ocr_ticks"
            recomputed += 1
        points.append(point)

    return {
        **record,
        "measured_points": points,
        "ruler_refit": {
            **axis_fit,
            "value_axis": orientation,
            "axis_resolution": axis_trace,
            "tick_values": [t["value"] for t in ticks],
            "recomputed_point_count": recomputed,
            "total_point_count": len(points),
        },
    }
