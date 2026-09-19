"""The M0' evidence chain for public charts, as one callable path.

Until now W1 (ruler refit), W2 (mark admission), W4 (reliability scoring) and the
category-axis builder existed only as modules that the ``evaluate_*`` scripts
composed ad hoc. Nothing on the answering path called them, so the evidence
actually injected into a prompt was still the frozen M0 output whose measured
values were 16% correct. This module is the single composition those scripts and
the answering path share, so "what we evaluated" and "what we inject" cannot
drift apart.

Stage order is forced by data dependencies, not preference:

1. **category axis** attaches labels to marks. It must run before selection,
   because ranking prefers facts that carry both a value and a label, and before
   admission only in the sense that it is harmless there -- it adds fields and
   removes nothing.
2. **ruler refit** recomputes every mark's value from OCR-read ticks and writes
   ``measured_value``. Marks on an inadmissible axis get ``None`` rather than a
   guess.
3. **mark admission** drops sliver artifacts, with the survival floor that keeps
   line charts from losing all their evidence.
4. **reliability scoring / selection** (M1'/M2') reads the result of 1-3. It must
   be last: it scores the evidence as it will actually be injected.

Answer independence holds throughout. Every stage reads OCR boxes, mark
geometry, and the fitted axis; none reads a reference answer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from evaluator.reasoning.plotqa_category_axis import augment_record
from evaluator.reasoning.plotqa_evidence_reliability import score_evidence, select_evidence
from evaluator.reasoning.plotqa_mark_admission import apply_admission
from evaluator.reasoning.plotqa_ruler_refit import refit_record

CHAIN_VERSION = "plotqa_m0_prime_chain_v1"

# Fields a reasoning stage can answer a question with. Everything else on a
# measured point is extractor bookkeeping -- BGR colours, component areas,
# bboxes, pixel coordinates, per-axis uncertainties -- which inflated the
# injected payload to a median of 12.2k characters, over 90% of it unusable.
# Volume is not neutral: it buries the handful of numbers the question needs
# and invites the model to reason from pixel coordinates instead of values.
PRESENTABLE_FACT_FIELDS = (
    "fact_id",
    "series_label",
    "category_label",
    "measured_value",
    "value_status",
)


def presentable_fact(fact: Dict[str, Any]) -> Dict[str, Any]:
    """Project a measured point onto the fields that can answer a question.

    Keys whose value is None are dropped rather than emitted as nulls: an
    absent label should read as absent, not as a field the model must reason
    about. ``fact_id`` is kept so an answer remains traceable to its evidence.
    """
    return {
        key: fact[key]
        for key in PRESENTABLE_FACT_FIELDS
        if fact.get(key) is not None
    }


def _axis_summary(refit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Report the fitted axis, without the tick-by-tick fitting record.

    The frozen record also carries a ``ruler`` block from the original
    extractor. It is deliberately not forwarded: after the refit its tick values
    contradict the ones the measurements were actually computed from -- one
    observed sample offers ``"I" -> 1.0`` alongside a correct refit -- so
    injecting both hands the model two disagreeing axes.
    """
    refit = refit or {}
    return {
        "value_axis": refit.get("value_axis"),
        "admissible": bool(refit.get("admissible")),
        "tick_count": refit.get("tick_count"),
        "provenance": refit.get("provenance"),
    }


def build_m0_prime_record(
    record: Dict[str, Any],
    ocr_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run stages 1-3 and return a new record. The frozen input is not mutated.

    ``ocr_row`` may be None -- each stage then records why it could not run and
    leaves values unmeasured, rather than falling back to a heuristic. A guessed
    axis produces numbers that look measured but are not, which the frozen
    evidence showed to be worse than supplying nothing.
    """
    chained = augment_record(record, ocr_row)
    chained = refit_record(chained, ocr_row)
    chained = apply_admission(chained)
    return {**chained, "m0_prime_chain": CHAIN_VERSION}


def build_evidence(
    record: Dict[str, Any],
    ocr_row: Optional[Dict[str, Any]],
    curated: bool,
    max_facts: int = 20,
) -> Dict[str, Any]:
    """Build the evidence payload for one arm.

    Both arms consume the *same* M0' record; the only difference is whether
    M1'/M2' may withhold or re-rank it:

    - ``curated=True``  -- M2' gates on the reliability tier and ranks the
      surviving facts, so a low-reliability sample is answered visually.
    - ``curated=False`` -- every admitted fact is injected in extraction order,
      truncated to the same budget. This is the uncurated-injection control, not
      a weaker pipeline: it sees identical evidence.

    Keeping the budget identical matters. If the curated arm also got a larger
    fact budget, any gain would confound selection with quantity.
    """
    chained = build_m0_prime_record(record, ocr_row)
    points = list(chained.get("measured_points") or [])
    reliability = score_evidence(chained)

    if curated:
        selection = select_evidence(chained, max_facts=max_facts)
        facts = selection["selected_facts"]
        injected = selection["inject"]
        policy = selection.get("selection_policy") or selection["reason"]
    else:
        selection = None
        facts = points[:max_facts]
        injected = True
        policy = f"uncurated_all_admitted_top{max_facts}"

    return {
        "ruler_refit": _axis_summary(chained.get("ruler_refit")),
        "chart_geometry": chained.get("chart_geometry"),
        "measured_facts": [presentable_fact(fact) for fact in facts],
        "evidence_gaps": chained.get("evidence_gaps") or [],
        "curation_policy": policy,
        # Reported for both arms so an analysis can stratify the control arm by
        # the same reliability tier the curated arm gated on.
        "evidence_reliability": {
            "tier": reliability["tier"],
            "composite": reliability["composite"],
            "signals": reliability["signals"],
        },
        "evidence_injected": injected,
        "chain_version": CHAIN_VERSION,
        "_selection": selection,
        "_chained_record": chained,
    }
