"""Category-axis construction for public charts (the M0' extraction gap).

Failure attribution that motivates this module
----------------------------------------------
``category_label`` coverage on the frozen PlotQA 500 roster splits sharply by
chart orientation:

===========  =========================
``qa_type``  samples with any label
===========  =========================
``hbar``     75/168  (0.45)
``vbar``      1/176  (0.01)
``dot``       0/99   (0.00)
``line``      0/57   (0.00)
===========  =========================

The cause is not a missing feature but a mis-scoped one:
``run_plotqa_measured_m0.category_label_candidates`` only accepts OCR boxes
satisfying ``cx <= left + 100``, i.e. text to the *left* of the plot area. That
is where categories sit on a horizontal bar chart. On vertical bars, dot plots
and line charts the categories run along the *bottom* x axis, so they were never
collected -- 332/500 samples had no category axis for a structural reason.

This module adds the missing orientation. It is deliberately a separate module
rather than an edit to the frozen extractor: the recorded
``m0_measurement_records.json`` is an input to the frozen PlotQA confirmation,
and changing the extractor in place would invalidate it. Callers run this over
existing records to produce an *augmented* record set, leaving the original on
disk untouched.

Answer independence holds throughout: every function reads only the OCR row, the
plot geometry, and the measured points. None reads a reference answer.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# A category label sits under the axis line, but "under" has to tolerate
# multi-row tick text and rotated labels; PlotQA renders both.
BOTTOM_BAND_ABOVE_AXIS = 12.0
BOTTOM_BAND_BELOW_AXIS = 140.0
# Axis titles ("Year", "Number of ...") also live in the bottom band. They span a
# large share of the plot width and sit centred, while category labels are
# narrow and repeat. Anything wider than this fraction of the plot is a title.
MAX_LABEL_WIDTH_FRACTION = 0.42
# Horizontal slack for assigning a mark to a label column, as a multiple of the
# mark's own width. Grouped bars sit off-centre from their category tick.
COLUMN_SLACK_WIDTH_MULTIPLE = 1.6
COLUMN_SLACK_FLOOR_PX = 18.0
# A category axis labels several columns. One surviving box is an axis title or
# merged text, never a category set -- see the guard in
# ``x_axis_category_candidates``.
MIN_CATEGORY_COLUMNS = 2
# How far below the label row a box must sit to read as the axis title. Scaled
# by the labels' own height, with this floor so a tight axis still separates.
TITLE_ROW_GAP_FLOOR_PX = 20.0

# Axis-title and unit words that are never category values. Matched on the
# normalised form, and only as a whole phrase or a leading token, so a genuine
# category such as "Year 2000" is not discarded.
AXIS_TITLE_TOKENS = frozenset({
    "year", "years", "number", "value", "values", "percent", "percentage",
    "amount", "total", "count", "index", "rate", "ratio", "share",
})


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


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _spans_overlap(a: Sequence[float], b: Sequence[float], min_fraction: float = 0.5) -> bool:
    """True when two boxes share at least ``min_fraction`` of the narrower span.

    Used to recognise the separate OCR boxes of one wrapped category label,
    which stack vertically inside a single column.
    """
    a_left, a_right = float(a[0]), float(a[2])
    b_left, b_right = float(b[0]), float(b[2])
    overlap = min(a_right, b_right) - max(a_left, b_left)
    if overlap <= 0:
        return False
    narrower = min(a_right - a_left, b_right - b_left)
    return narrower > 0 and (overlap / narrower) >= min_fraction


def _is_axis_title(normalized: str, box_width: float, plot_width: float) -> bool:
    """True when a bottom-band box is the axis title rather than a category.

    Two independent signals: the text is a bare measurement word, or the box is
    too wide to be one category among several.
    """
    if not normalized:
        return True
    if plot_width > 0 and box_width > plot_width * MAX_LABEL_WIDTH_FRACTION:
        return True
    tokens = normalized.split()
    if len(tokens) <= 2 and all(token in AXIS_TITLE_TOKENS for token in tokens):
        return True
    # "number of x", "percentage of y" -- a title phrase, not a category value.
    if len(tokens) >= 2 and tokens[0] in AXIS_TITLE_TOKENS and tokens[1] == "of":
        return True
    return False


def x_axis_category_candidates(
    ocr_row: Dict[str, Any],
    plot_area: Sequence[float],
    exclude_numeric: bool = True,
) -> List[Dict[str, Any]]:
    """Collect category labels along the bottom x axis.

    ``exclude_numeric`` drops boxes parsing as pure numbers, because those are
    value ticks handled by the ruler. It is a parameter rather than a constant
    because a category axis can legitimately be numeric (years as categories),
    and the caller knows which case it is from the fitted axis model.
    """
    if len(plot_area) != 4:
        return []
    left, _top, right, bottom = (float(v) for v in plot_area)
    plot_width = max(0.0, right - left)
    candidates: List[Dict[str, Any]] = []

    for box in ocr_row.get("ocr_boxes") or []:
        text = str(box.get("text", "")).strip()
        if not text:
            continue
        bbox = _bbox(box.get("bbox"))
        if bbox is None:
            continue
        cx, cy = _center(bbox)
        # Must sit in the band just below the axis line, horizontally inside the
        # plot. The small upward tolerance catches labels whose box overlaps the
        # axis line itself.
        if not (bottom - BOTTOM_BAND_ABOVE_AXIS <= cy <= bottom + BOTTOM_BAND_BELOW_AXIS):
            continue
        if not (left - COLUMN_SLACK_FLOOR_PX <= cx <= right + COLUMN_SLACK_FLOOR_PX):
            continue
        normalized = _normalize(text)
        if exclude_numeric and re.fullmatch(r"[-+]?[\d\s.,]+", text.strip()):
            continue
        box_width = float(bbox[2]) - float(bbox[0])
        if _is_axis_title(normalized, box_width, plot_width):
            continue
        candidates.append({
            "label": text,
            "normalized": normalized,
            "cx": cx,
            "cy": cy,
            "bbox": bbox,
            "confidence": str(box.get("confidence") or "medium"),
        })

    # Drop the axis title before deduping, not after. The title is wider than a
    # category label, and the dedupe below keeps the wider of two overlapping
    # boxes -- so a title left in the pool swallows whichever real label shares
    # its column ("Chile" lost to "Country"), removing a category rather than
    # merely adding a spurious one.
    candidates = _without_title_row(candidates)

    # Two OCR boxes can describe one category (wrapped text, stacked on separate
    # rows). Centres are not a reliable test for this: wrapping shifts the centre
    # of each line independently, so "St. Kitts" and "St. Kitts and Nevis" sit
    # 21px apart. Horizontal span overlap is the robust test, since the lines of
    # one wrapped label always occupy the same column.
    candidates.sort(key=lambda c: c["bbox"][0])
    deduped: List[Dict[str, Any]] = []
    for candidate in candidates:
        merged = False
        for index, kept in enumerate(deduped):
            if _spans_overlap(kept["bbox"], candidate["bbox"]):
                kept_width = float(kept["bbox"][2]) - float(kept["bbox"][0])
                cur_width = float(candidate["bbox"][2]) - float(candidate["bbox"][0])
                if cur_width > kept_width:
                    deduped[index] = candidate
                merged = True
                break
        if not merged:
            deduped.append(candidate)
    deduped.sort(key=lambda c: c["cx"])
    return deduped


def _without_title_row(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop an axis title sitting on its own row below the category labels.

    Category labels share a row -- rendered at one offset under the axis, and
    rotated when long. The axis title sits on a separate line beneath it, so it
    survives the width and wordlist filters whenever it is short: "Country"
    appeared as a bogus category on 39 frozen-roster samples, at cy~646 while
    the real labels sat at cy~550-575.

    The test is positional rather than lexical because the offending strings
    have nothing in common as text. Exactly one box below the row is a title;
    several means the labels themselves are staggered and all must be kept.
    """
    if len(candidates) <= MIN_CATEGORY_COLUMNS:
        return candidates
    main_row_cy = statistics.median([c["cy"] for c in candidates])
    heights = [max(1.0, float(c["bbox"][3]) - float(c["bbox"][1])) for c in candidates]
    gap = max(TITLE_ROW_GAP_FLOOR_PX, statistics.median(heights))

    kept = [c for c in candidates if c["cy"] <= main_row_cy + gap]
    if len(candidates) - len(kept) != 1 or len(kept) < MIN_CATEGORY_COLUMNS:
        return candidates
    return kept


def admissible_category_columns(
    candidates: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Reject a candidate set that is not a category axis at all.

    A category axis labels several columns. One surviving box is the axis
    title, a merged run of legend text, or a stray tick that the per-box
    filters missed ("Cost (US$ per container)", "AngolaHong KongMacedonia",
    "0.000e+0"). Attaching it stamped one identical label onto every mark in
    the chart on 85 of the frozen 500 samples -- worse than no label, because
    it asserts an association the chart does not make.

    A genuinely single-category chart loses nothing here: one label shared by
    all marks cannot distinguish them anyway.
    """
    return list(candidates) if len(candidates) >= MIN_CATEGORY_COLUMNS else []


def attach_x_axis_categories(
    points: List[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assign each mark the nearest category column, in place.

    A mark is assigned only when its horizontal distance to the column centre is
    within its own width plus a floor. Marks are the width of their category
    slot, so this keeps a mark from borrowing a neighbour's label when the label
    it belongs to was not read by OCR -- an unlabelled point is strictly better
    than a confidently mislabelled one.

    Existing ``category_label`` values are never overwritten: the frozen
    left-side extractor is authoritative where it fired.
    """
    if not candidates:
        return {"candidate_count": 0, "assigned_count": 0, "status": "no_candidates"}

    assigned = 0
    for point in points:
        if point.get("category_label"):
            continue
        bbox = _bbox(point.get("component_bbox"))
        center = point.get("component_center_pixel") or point.get("pixel")
        if bbox is None or not isinstance(center, (list, tuple)) or len(center) != 2:
            continue
        px = _numeric(center[0])
        if px is None:
            continue
        mark_width = max(1.0, float(bbox[2]) - float(bbox[0]))
        slack = max(mark_width * COLUMN_SLACK_WIDTH_MULTIPLE, COLUMN_SLACK_FLOOR_PX)
        best = min(candidates, key=lambda c: abs(float(c["cx"]) - px))
        if abs(float(best["cx"]) - px) <= slack:
            point["category_label"] = best["label"]
            point["category_confidence"] = best["confidence"]
            point.setdefault("support", []).append("x_axis_category_column")
            assigned += 1

    return {
        "candidate_count": len(candidates),
        "assigned_count": assigned,
        "status": "attached" if assigned else "no_assignment",
    }


def numeric_labels_are_categories(record: Dict[str, Any]) -> bool:
    """True when numeric bottom-axis text names categories rather than ticks.

    ``x_axis_category_candidates`` drops numeric boxes by default because on a
    horizontal bar chart the bottom axis *is* the value axis, so its numbers are
    ruler ticks. On every other orientation the bottom axis is the category
    axis, and PlotQA labels it with years far more often than not -- so the
    default discarded the whole category axis on 199 of the frozen 500 samples,
    reporting ``no_candidates`` with nothing read.

    The distinction needs no new geometry: it is exactly the orientation
    ``value_axis_orientation`` already computes from mark shape. Imported here
    rather than at module scope because the ruler module is the heavier of the
    two and only this one predicate needs it.
    """
    from evaluator.reasoning.plotqa_ruler_refit import value_axis_orientation

    return value_axis_orientation(record) != "x"


def augment_record(record: Dict[str, Any], ocr_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a shallow-copied record with x-axis categories attached.

    The input record is not mutated, so the frozen artifact stays valid; the
    measured points are deep-enough copied that attaching a label cannot leak
    back into the caller's originals.

    Measured offline on the frozen 500 before this rule was written into the
    pipeline (``scripts/probe_plotqa_numeric_category_axis.py``): labelled
    samples 133 -> 346, labelled marks 1877 -> 5555, precision of the newly
    attached labels 0.957 against the official annotation, and **zero samples
    lost a label they already had** in any chart family.
    """
    if not ocr_row:
        return {**record, "category_axis": {"status": "no_ocr_row", "assigned_count": 0}}

    geometry = record.get("chart_geometry") or {}
    plot_area = geometry.get("plot_area_bbox")
    if not plot_area or len(plot_area) != 4:
        return {**record, "category_axis": {"status": "no_plot_area", "assigned_count": 0}}

    points = [dict(point) for point in (record.get("measured_points") or [])]
    numeric_categories = numeric_labels_are_categories(record)
    candidates = admissible_category_columns(
        x_axis_category_candidates(
            ocr_row, plot_area, exclude_numeric=not numeric_categories))
    summary = attach_x_axis_categories(points, candidates)
    summary["labels"] = [c["label"] for c in candidates]
    summary["numeric_labels_admitted"] = numeric_categories
    return {**record, "measured_points": points, "category_axis": summary}


def coverage_report(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate category-label coverage, for before/after comparison."""
    materialised = list(records)
    total_points = 0
    labelled_points = 0
    samples_with_label = 0
    for record in materialised:
        points = record.get("measured_points") or []
        total_points += len(points)
        labelled = sum(1 for p in points if p.get("category_label"))
        labelled_points += labelled
        if labelled:
            samples_with_label += 1
    return {
        "sample_count": len(materialised),
        "samples_with_any_category_label": samples_with_label,
        "total_measured_points": total_points,
        "points_with_category_label": labelled_points,
        "point_label_rate": (labelled_points / total_points) if total_points else 0.0,
        "sample_label_rate": (samples_with_label / len(materialised)) if materialised else 0.0,
    }
