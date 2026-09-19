"""Evidence reliability scoring for public charts (M1', work item W4).

What changed since the first attempt
------------------------------------
An earlier reliability module (``plotqa_reliability.py``) scored the *frozen* M0
evidence and was falsified: its composite separated help from harm at AUC 0.454
and 0.439, i.e. slightly worse than chance. The cause is now known. At that time
the value accuracy of detected marks was 0.160 -- the measurements were
systematically wrong because the axis was guessed rather than fitted, so any
reliability signal was swamped by calibration error. There was nothing to score.

After W1 (axis refit from OCR ticks) and W2 (artifact rejection), value accuracy
is 0.781 and a signal exists. Measured against official per-mark ground truth on
the frozen roster, with "reliable" defined as *every* ground-truth cell present
with a correct value:

===============================  =====
Feature                          AUC
===============================  =====
usable value count               0.710
usable fraction of admitted       0.652
axis admissible                   0.624
tick count                        0.618
tick fit residual                 0.469
artifact rejection rate           0.461
===============================  =====

``axis_admissible * log1p(usable_count)`` reaches the same 0.710 as the best
single feature while being the one with a defensible reading: evidence is
trustworthy when the axis was actually fitted **and** enough marks survived to
answer with. Products of more features did not improve on it, so the simpler
form is kept.

A confound was checked and ruled out. ``usable_count`` could have been proxying
chart size rather than quality; within chart-size strata its AUC is 0.705-0.912,
*higher* than pooled, so the signal is not a size artifact.

Honest limits
-------------
AUC 0.710 is a usable ordinal signal, not a calibrated probability. The tiers
below are heuristic cutpoints. Do not describe them as calibrated, probabilistic,
or verified, and do not read a tier as a likelihood.

Answer independence holds: every input is derived from the OCR boxes, the mark
geometry, and the fitted axis. None reads a reference answer.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

# Tier cutpoints on the composite. Set from the score's own construction rather
# than tuned against an outcome: log1p(6) ~ 1.95 is "the axis was fitted and at
# least six marks survived", log1p(2) ~ 1.10 is "fitted but barely any evidence".
TIER_HIGH_MIN = 1.95
TIER_MEDIUM_MIN = 1.10

RELIABILITY_TIERS = ("high", "medium", "low")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def usable_values(points: Sequence[Dict[str, Any]]) -> List[float]:
    """Values a downstream stage can actually compute with.

    Reads ``measured_value`` -- the orientation-independent field written by the
    W1 refit -- and falls back to ``y_value`` for records that predate it. A
    point whose axis was inadmissible carries None and is excluded here, which is
    the intended behaviour: it was never measured.
    """
    values: List[float] = []
    for point in points:
        raw = point.get("measured_value") if "measured_value" in point else point.get("y_value")
        if _is_number(raw):
            values.append(float(raw))
    return values


def score_evidence(record: Dict[str, Any]) -> Dict[str, Any]:
    """Score one post-W1/W2 record's evidence reliability. Never reads the answer."""
    points = list(record.get("measured_points") or [])
    refit = record.get("ruler_refit") or {}
    admission = record.get("mark_admission") or {}

    axis_admissible = bool(refit.get("admissible"))
    values = usable_values(points)
    usable_count = len(values)

    composite = (1.0 if axis_admissible else 0.0) * math.log1p(usable_count)
    if composite >= TIER_HIGH_MIN:
        tier = "high"
    elif composite >= TIER_MEDIUM_MIN:
        tier = "medium"
    else:
        tier = "low"

    return {
        "composite": composite,
        "tier": tier,
        "signals": {
            "axis_admissible": axis_admissible,
            "usable_value_count": usable_count,
            "usable_fraction_of_admitted": (usable_count / len(points)) if points else 0.0,
            "tick_count": refit.get("tick_count"),
            "tick_residual_fraction": refit.get("residual_fraction"),
            "axis_fallback_used": bool((refit.get("axis_resolution") or {}).get("fallback_used")),
            "artifacts_rejected": admission.get("rejected_count"),
            "admission_policy": admission.get("policy"),
        },
        "scoring_policy": "plotqa_evidence_reliability_v1_ordinal_uncalibrated",
        "discrimination_auc_on_frozen_roster": 0.710,
    }


def select_evidence(
    record: Dict[str, Any],
    max_facts: int = 20,
) -> Dict[str, Any]:
    """Choose which facts to inject, and whether to inject at all (M2').

    Two decisions, both driven by the reliability score:

    1. **Admission.** A ``low`` tier record has either no fitted axis or almost
       no surviving marks. Injecting it supplies numbers that look measured but
       are not, and the frozen evidence shows that is actively harmful -- 45% of
       the samples where evidence made the answer worse were cases where the
       model adopted a wrong injected number. Such a record is withheld and the
       caller is expected to answer visually.

    2. **Ranking.** Facts carrying a usable value and an identifying label come
       first, because a value without a label cannot be matched to the entity a
       question names. Truncation is by this ranking rather than by input order,
       so the budget is spent on the most answerable facts. Ranking is stable and
       answer-independent -- it never inspects the question.
    """
    reliability = score_evidence(record)
    points = list(record.get("measured_points") or [])

    if reliability["tier"] == "low":
        return {
            "inject": False,
            "reason": "withheld_low_reliability_evidence",
            "reliability": reliability,
            "selected_facts": [],
            "input_fact_count": len(points),
        }

    def rank(point: Dict[str, Any]) -> tuple:
        raw = point.get("measured_value") if "measured_value" in point else point.get("y_value")
        has_value = _is_number(raw)
        has_label = bool(point.get("series_label") or point.get("category_label"))
        # Sort key: valued-and-labelled first, then valued, then the rest.
        return (0 if (has_value and has_label) else 1 if has_value else 2,)

    ranked = sorted(points, key=rank)
    return {
        "inject": True,
        "reason": "admitted",
        "reliability": reliability,
        "selected_facts": ranked[:max_facts],
        "input_fact_count": len(points),
        "selection_policy": f"reliability_tier_gate_plus_value_label_ranking_top{max_facts}",
    }
