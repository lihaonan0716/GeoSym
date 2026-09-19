"""Answer-independent reliability scoring and admission gating for measured
chart evidence (the public-chart instantiation of M1 and M2).

This module is the PlotQA-family counterpart to
:mod:`evaluator.reasoning.intervention`. The two share an interface, not an
algorithm: ``intervention`` scores *inferred* facts (intercepts, monotonicity,
extrema, symmetry) on clean synthetic charts, while this module scores *measured
points* extracted from real-world charts, where the dominant failure mode is
mis-attribution rather than arithmetic slips.

Two properties are load-bearing and must be preserved by any change:

1. **Answer independence.** Every feature reads only the M0 record and the
   question text. None reads the reference answer. This is what makes a gate
   built on these scores a legitimate pipeline component rather than an oracle.
2. **Ordinal, not calibrated.** The scores are heuristic reliability signals
   combined with fixed, hand-set cutpoints. They are not probabilities and carry
   no calibration evidence. Describe them as ordinal tiers only.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Fixed scales for normalising raw feature values into [0, 1]. These are set
# from the physical meaning of each quantity, not fitted to any outcome:
# a tick that sits half an axis-range away from its own fitted position is
# wholly untrustworthy, so 0.5 is the natural saturation point.
RULER_RESIDUAL_SATURATION = 0.5
# Two marks extracted at one x position whose values differ by more than this
# fraction of their median cannot both describe the same quantity.
VALUE_CONFLICT_RATIO = 0.10
# Ordinal tier cutpoints on the composite score.
TIER_HIGH_MIN = 0.70
TIER_MEDIUM_MIN = 0.40

RELIABILITY_TIERS = ("high", "medium", "low")


def _numeric(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _point_label(point: Dict[str, Any]) -> str:
    for key in ("series_label", "category_label"):
        raw = point.get(key)
        if raw:
            return str(raw).strip().lower()
    return ""


def value_conflict_ratio(points: Iterable[Dict[str, Any]]) -> Optional[float]:
    """Fraction of x positions carrying mutually inconsistent values.

    Several marks sharing an x position is normal for a grouped or multi-series
    chart, so co-location alone is not evidence of a defect. It becomes one only
    when the co-located marks cannot be told apart: if they carry distinct series
    labels the grouping is legitimate, but if they are unlabelled (or share a
    label) and disagree in value, then M0 has extracted competing readings for
    one quantity and cannot say which is the real one.

    Returns ``None`` when no x position holds more than one point, i.e. when the
    feature is not measurable rather than when it is clean.
    """
    buckets: Dict[float, List[Dict[str, Any]]] = {}
    for point in points:
        x_value = _numeric(point.get("x_value"))
        if x_value is None:
            continue
        buckets.setdefault(round(x_value, 3), []).append(point)

    measurable = 0
    conflicting = 0
    for group in buckets.values():
        values = [(_numeric(p.get("y_value")), _point_label(p)) for p in group]
        values = [(v, label) for v, label in values if v is not None]
        if len(values) < 2:
            continue
        measurable += 1
        distinct_labels = {label for _, label in values if label}
        if len(distinct_labels) == len(values):
            # Every co-located mark is separately identified: legitimate grouping.
            continue
        magnitudes = [v for v, _ in values]
        median = statistics.median(magnitudes)
        if median and max(abs(v - median) for v in magnitudes) > abs(median) * VALUE_CONFLICT_RATIO:
            conflicting += 1
    return (conflicting / measurable) if measurable else None


def ruler_residual(ruler: Optional[Dict[str, Any]]) -> Optional[float]:
    """Worst normalised residual of the pixel-to-value tick fit, over both axes.

    The ruler is the mapping every measured value depends on, so a poor fit
    discredits the whole record rather than any individual point. Residuals are
    normalised by the axis value range so the number is unit-free and comparable
    across charts.
    """
    if not ruler:
        return None
    worst: List[float] = []
    for axis in ("x_ticks", "y_ticks"):
        ticks = ruler.get(axis) or []
        pairs: List[Tuple[float, float]] = []
        for tick in ticks:
            pixel = _numeric(tick.get("pixel"))
            value = _numeric(tick.get("value"))
            if pixel is not None and value is not None:
                pairs.append((pixel, value))
        # Two points define a line exactly, so a residual only carries
        # information from the third tick onward.
        if len(pairs) < 3:
            continue
        n = len(pairs)
        sum_p = sum(p for p, _ in pairs)
        sum_v = sum(v for _, v in pairs)
        sum_pp = sum(p * p for p, _ in pairs)
        sum_pv = sum(p * v for p, v in pairs)
        denominator = n * sum_pp - sum_p * sum_p
        if denominator == 0:
            continue
        slope = (n * sum_pv - sum_p * sum_v) / denominator
        intercept = (sum_v - slope * sum_p) / n
        value_range = max(v for _, v in pairs) - min(v for _, v in pairs)
        if value_range <= 0:
            continue
        worst.append(max(abs(v - (slope * p + intercept)) / value_range for p, v in pairs))
    return max(worst) if worst else None


def label_coverage(points: Iterable[Dict[str, Any]]) -> Optional[float]:
    """Fraction of measured points that carry a series or category label.

    An unlabelled point has a value but no statement of *what* it measures, so
    it cannot be matched to the entity a question names.
    """
    materialised = list(points)
    if not materialised:
        return None
    return sum(1 for p in materialised if _point_label(p)) / len(materialised)


def score_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Score one M0 record's evidence reliability. Never reads the answer.

    The three sub-scores are averaged with equal weight over whichever are
    measurable. Equal weighting is deliberate: fitting weights against the
    outcome on the evaluation roster is exactly the tuning-on-the-test-set
    failure the project forbids, and with three features there is no headroom
    for a weighting to earn its extra degrees of freedom honestly.
    """
    points = list(record.get("measured_points") or [])
    conflict = value_conflict_ratio(points)
    residual = ruler_residual(record.get("ruler"))
    coverage = label_coverage(points)

    components: Dict[str, float] = {}
    if conflict is not None:
        components["consistency"] = 1.0 - conflict
    if residual is not None:
        components["ruler_fit"] = 1.0 - min(1.0, residual / RULER_RESIDUAL_SATURATION)
    if coverage is not None:
        components["attribution"] = coverage

    composite = sum(components.values()) / len(components) if components else None
    if composite is None:
        tier = "low"
    elif composite >= TIER_HIGH_MIN:
        tier = "high"
    elif composite >= TIER_MEDIUM_MIN:
        tier = "medium"
    else:
        tier = "low"

    return {
        "composite": composite,
        "tier": tier,
        "components": components,
        "raw_features": {
            "value_conflict_ratio": conflict,
            "ruler_residual": residual,
            "label_coverage": coverage,
        },
        "measured_point_count": len(points),
        "scoring_policy": "plotqa_reliability_v1_ordinal_uncalibrated",
    }


def admit_evidence(record: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Decide whether measured evidence should be injected at all (M2').

    Returns the decision alongside the score that produced it. A record scoring
    below ``threshold`` is withheld and the caller is expected to fall back to a
    purely visual answer: injecting evidence that cannot be trusted is worse
    than injecting none, because a confident-looking wrong number displaces a
    correct visual reading.

    ``threshold`` is the single tuned knob in this module and must be fixed on
    data disjoint from whatever roster the result is reported on.
    """
    score = score_record(record)
    composite = score["composite"]
    admitted = composite is not None and composite >= threshold
    return {
        "admitted": admitted,
        "threshold": threshold,
        "reason": "evidence_admitted" if admitted else "withheld_low_reliability",
        "reliability": score,
    }
