"""Reject non-mark artifacts from measured chart evidence (M0' work item W2).

Why this module exists
----------------------
Measured against the official PlotQA annotations, only 40.5% of the points the
extractor emits correspond to a real plotted mark. The other 59.5% are axis
fragments, grid remnants, text edges and anti-aliasing slivers. That matters
because injected evidence is not inert: 45% of the samples where evidence made
the answer *worse* were cases where the model adopted an injected number that
was wrong (see ``PLOTQA_IDEAL_CHAIN_ZERO_COST_ANALYSIS_2026-09-18.md``).

Ground-truth characterisation of the two populations:

===============  ==================  ==============
Feature          spurious (p10/50/90)  true (p10/50/90)
===============  ==================  ==============
bbox width       1 / 8 / 68          12 / 22 / 46
bbox min side    --                  --
component area   25 / 136 / 906      97 / 520 / 15674
carries a label  0.390               0.847
===============  ==================  ==============

The discriminating feature is the **minimum bbox side**: real marks are at
least a few pixels across in both directions, while artifacts are one- or
two-pixel slivers. A threshold sweep on the frozen roster gives:

=====  =========  ======
thr    precision  recall
=====  =========  ======
0      0.405      1.000
4      0.725      0.952
**5**  **0.735**  **0.948**
8      0.743      0.916
12     0.770      0.836
=====  =========  ======

5 px is taken as the operating point: it is at the F1 maximum (0.828) and the
precision curve flattens after it, so paying further recall buys almost nothing.

What this module deliberately does *not* do
-------------------------------------------
Label presence separates the populations more sharply (0.390 vs 0.847) but
filtering on it costs far more recall than the min-side rule for a comparable
precision gain, and it would bias the surviving evidence toward whichever series
OCR happened to read -- a systematic distortion rather than a random loss.

Line charts are excluded from filtering entirely. Their components are polyline
fragments rather than discrete marks, so the sliver rule does not separate
signal from noise there (precision 0.149 -> 0.139, i.e. no better than chance)
and would only shrink the evidence. Line charts need a different extractor, not
a different filter; that is recorded as W3, not papered over here.

Answer independence: every feature reads geometry and provenance only.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Minimum extent, in pixels, that a real plotted mark spans in *both*
# directions. See the sweep in the module docstring.
MIN_MARK_SIDE_PX = 5.0
# A component whose area is a vanishing fraction of the sample's median mark is
# a fragment even when its bounding box passes the side test.
MIN_AREA_FRACTION_OF_MEDIAN = 0.02
# Sources whose points are polyline samples rather than discrete marks. The
# sliver rule does not apply to them.
POLYLINE_SOURCES = frozenset({"cv_colored_line_sample"})
# If fewer than this fraction of a sample's components would survive the sliver
# rule, the rule is not separating marks from artifacts on this sample -- it is
# deleting the evidence. The filter then stands down. See
# ``filtering_would_gut_evidence`` for why this is keyed on the outcome rather
# than on a guess at the chart type.
SURVIVAL_FLOOR = 0.25


def _bbox(raw: Any) -> Optional[List[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return [float(v) for v in raw]
    except (TypeError, ValueError):
        return None


def _area(point: Dict[str, Any], bbox: Sequence[float]) -> float:
    recorded = point.get("component_area")
    if isinstance(recorded, (int, float)) and not isinstance(recorded, bool) and recorded > 0:
        return float(recorded)
    return max(0.0, (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])))


def filtering_would_gut_evidence(
    points: Sequence[Dict[str, Any]],
    min_side_px: float,
    survival_floor: float = SURVIVAL_FLOOR,
) -> bool:
    """True when the sliver rule would remove nearly all of a sample's evidence.

    This is the abstention guard, and it is deliberately a property of the
    *outcome* rather than a guess at the chart type. Two earlier attempts keyed
    on chart identity and both failed on the frozen roster: point provenance
    identifies only 8 of 57 line-chart samples, and an elongation test misread
    151 of 176 vertical-bar samples as polylines because a bar is elongated too.
    Edge-sharing separates bars from dots but leaves line charts spread across
    the whole range (p10 0.17, p90 1.00), so no geometric proxy is clean.

    What *is* unambiguous is the effect. Measured per chart type, the rule is
    strongly positive on bars (precision 0.332 -> 0.766 for a recall cost of
    0.826 -> 0.794) and clearly negative on line charts (precision 0.149 ->
    0.181 while recall collapses 0.194 -> 0.040). A sample about to lose almost
    all its components is in the second regime whatever it is plotting, so the
    filter stands down and the evidence is passed through unfiltered.
    """
    materialised = list(points)
    if not materialised:
        return False
    measurable = 0
    survivors = 0
    for point in materialised:
        bbox = _bbox(point.get("component_bbox"))
        if bbox is None:
            continue
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        if width <= 0 or height <= 0:
            continue
        measurable += 1
        if min(width, height) >= min_side_px:
            survivors += 1
    if measurable == 0:
        return False
    return (survivors / measurable) < survival_floor


def is_polyline_evidence(points: Iterable[Dict[str, Any]]) -> bool:
    """True when a sample's points are declared polyline samples.

    Kept as an explicit provenance check. It is reliable when it fires but fires
    rarely, so ``filtering_would_gut_evidence`` is what actually protects
    stroke-based evidence.
    """
    materialised = list(points)
    if not materialised:
        return False
    polyline = sum(1 for p in materialised if p.get("source") in POLYLINE_SOURCES)
    return polyline * 2 >= len(materialised)


def admit_marks(
    points: Sequence[Dict[str, Any]],
    min_side_px: float = MIN_MARK_SIDE_PX,
) -> Dict[str, Any]:
    """Partition points into admitted marks and rejected artifacts.

    Returns the admitted list alongside per-reason rejection counts, so a
    downstream report can state what was dropped instead of silently shrinking
    the evidence.

    Points are annotated in place on their copies with ``admission_status``;
    nothing is deleted from the caller's objects.
    """
    if is_polyline_evidence(points):
        return {
            "admitted": [dict(p, admission_status="admitted_polyline_unfiltered")
                         for p in points],
            "rejected": [],
            "rejected_counts": {},
            "policy": "polyline_evidence_not_filtered",
            "input_count": len(points),
        }
    if filtering_would_gut_evidence(points, min_side_px):
        return {
            "admitted": [dict(p, admission_status="admitted_filter_stood_down")
                         for p in points],
            "rejected": [],
            "rejected_counts": {},
            "policy": "filter_stood_down_survival_below_floor",
            "input_count": len(points),
        }

    areas = []
    for point in points:
        bbox = _bbox(point.get("component_bbox"))
        if bbox is not None:
            areas.append(_area(point, bbox))
    median_area = statistics.median(areas) if areas else 0.0

    admitted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    def reject(point: Dict[str, Any], reason: str) -> None:
        counts[reason] = counts.get(reason, 0) + 1
        rejected.append(dict(point, admission_status=f"rejected_{reason}"))

    for point in points:
        bbox = _bbox(point.get("component_bbox"))
        if bbox is None:
            # No geometry to judge by. Keep it: absence of evidence about a
            # point is not evidence it is an artifact.
            admitted.append(dict(point, admission_status="admitted_no_geometry"))
            continue
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        if width <= 0 or height <= 0:
            reject(point, "degenerate_bbox")
            continue
        if min(width, height) < min_side_px:
            reject(point, "sliver_below_min_side")
            continue
        if median_area > 0 and _area(point, bbox) < median_area * MIN_AREA_FRACTION_OF_MEDIAN:
            reject(point, "area_fragment")
            continue
        admitted.append(dict(point, admission_status="admitted"))

    return {
        "admitted": admitted,
        "rejected": rejected,
        "rejected_counts": counts,
        "policy": f"min_side_{min_side_px:g}px_plus_area_fragment",
        "input_count": len(points),
    }


def apply_admission(
    record: Dict[str, Any],
    min_side_px: float = MIN_MARK_SIDE_PX,
) -> Dict[str, Any]:
    """Return a copy of ``record`` with artifact points removed.

    The frozen input is never mutated. The rejection summary is attached under
    ``mark_admission`` so a report can state the drop rate rather than present
    a quietly smaller evidence set.
    """
    points = list(record.get("measured_points") or [])
    result = admit_marks(points, min_side_px=min_side_px)
    return {
        **record,
        "measured_points": result["admitted"],
        "mark_admission": {
            "policy": result["policy"],
            "input_count": result["input_count"],
            "admitted_count": len(result["admitted"]),
            "rejected_count": len(result["rejected"]),
            "rejected_counts": result["rejected_counts"],
        },
    }
