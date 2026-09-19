"""Tests for the M0' chain -- the single composition the answering path uses.

The chain exists so that "what the evaluations measure" and "what a model is
shown" cannot drift apart. Before it, W1/W2/W4 were composed ad hoc inside the
``evaluate_*`` scripts while ``measured_prompt_facts`` still curated the frozen
M0 record, so the measured improvements reached no prompt.

The payload contract is pinned here because it is a performance surface, not
just plumbing: the injected JSON was a median 12.2k characters of extractor
bookkeeping (BGR colours, component areas, bboxes, pixel coordinates), which
buries the few numbers a question needs and invites reasoning from pixel
coordinates instead of values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.reasoning.plotqa_m0_chain import (  # noqa: E402
    CHAIN_VERSION,
    build_evidence,
    build_m0_prime_record,
    presentable_fact,
)

PLOT_AREA = [100.0, 50.0, 500.0, 400.0]


def _ocr_box(text, x0, y0, x1, y1, confidence=0.99):
    return {"text": text, "bbox": [x0, y0, x1, y1], "confidence": confidence}


def _bar(cx, top, label="Brazil"):
    """A vertical bar: tall, anchored on the bottom axis at y=400."""
    return {
        "fact_id": f"m0_{int(cx)}",
        "series_label": label,
        "component_bbox": [cx - 15.0, top, cx + 15.0, 400.0],
        "component_center_pixel": [cx, (top + 400.0) / 2.0],
        "pixel": [cx, top],
        "component_color_bgr": [112, 160, 240],
        "component_area": 4500,
        "uncertainty": {"y_abs": 0.02, "pixel_abs": 4.0},
        "source": "cv_color_component",
        "confidence": "medium",
        "y_value": 1.0,
    }


def _record():
    return {
        "sample_id": "s1",
        "chart_geometry": {"plot_area_bbox": PLOT_AREA, "chart_type": "bar"},
        "ruler": {"y_ticks": [{"text": "I", "value": 1.0, "confidence": "low"}]},
        "measured_points": [_bar(170.0, 200.0), _bar(270.0, 100.0), _bar(370.0, 300.0)],
    }


def _ocr():
    # A y axis reading 0 at the baseline and 40 at the top, plus category labels.
    return {"ocr_boxes": [
        _ocr_box("0", 80, 392, 95, 408),
        _ocr_box("20", 76, 217, 95, 233),
        _ocr_box("40", 76, 42, 95, 58),
        _ocr_box("Brazil", 150, 410, 190, 430),
        _ocr_box("Chile", 250, 410, 290, 430),
        _ocr_box("Peru", 350, 410, 390, 430),
    ]}


class TestPresentableFact:
    def test_keeps_only_answer_relevant_fields(self):
        projected = presentable_fact({
            "fact_id": "m0_000",
            "series_label": "Brazil",
            "category_label": "2013",
            "measured_value": 12.5,
            "value_status": "refit_from_ocr_ticks",
            "component_color_bgr": [1, 2, 3],
            "component_bbox": [0, 0, 10, 10],
            "component_area": 100,
            "pixel": [5, 5],
            "uncertainty": {"y_abs": 0.1},
        })
        assert projected == {
            "fact_id": "m0_000",
            "series_label": "Brazil",
            "category_label": "2013",
            "measured_value": 12.5,
            "value_status": "refit_from_ocr_ticks",
        }

    def test_absent_fields_are_omitted_not_nulled(self):
        """An absent label should read as absent, not as a null to reason about."""
        projected = presentable_fact({"fact_id": "m0_000", "measured_value": 1.0,
                                      "series_label": None})
        assert "series_label" not in projected


class TestBuildEvidence:
    def test_payload_carries_no_extractor_bookkeeping(self):
        evidence = build_evidence(_record(), _ocr(), curated=False)
        assert evidence["measured_facts"]
        for fact in evidence["measured_facts"]:
            for noisy in ("component_color_bgr", "component_bbox", "component_area",
                          "pixel", "uncertainty", "component_center_pixel"):
                assert noisy not in fact

    def test_stale_extractor_ruler_is_not_forwarded(self):
        """After the refit the frozen ``ruler`` block contradicts the measurements.

        One observed sample carries ``"I" -> 1.0`` in the frozen ruler next to a
        correct refit. Injecting both hands the model two disagreeing axes.
        """
        evidence = build_evidence(_record(), _ocr(), curated=False)
        assert "ruler" not in evidence
        assert evidence["ruler_refit"]["value_axis"] == "y"

    def test_both_arms_receive_the_same_underlying_evidence(self):
        raw = build_evidence(_record(), _ocr(), curated=False)
        curated = build_evidence(_record(), _ocr(), curated=True)
        raw_values = {f["measured_value"] for f in raw["measured_facts"]}
        curated_values = {f["measured_value"] for f in curated["measured_facts"]}
        assert curated_values <= raw_values

    def test_arms_share_one_fact_budget(self):
        record = _record()
        record["measured_points"] = [_bar(100.0 + 12 * i, 200.0) for i in range(40)]
        raw = build_evidence(record, _ocr(), curated=False, max_facts=20)
        curated = build_evidence(record, _ocr(), curated=True, max_facts=20)
        assert len(raw["measured_facts"]) <= 20
        assert len(curated["measured_facts"]) <= 20

    def test_reliability_is_reported_for_both_arms(self):
        """The control arm must be stratifiable by the tier the curated arm gated on."""
        for curated in (True, False):
            evidence = build_evidence(_record(), _ocr(), curated=curated)
            assert evidence["evidence_reliability"]["tier"] in {"high", "medium", "low"}

    def test_chain_version_is_stamped(self):
        assert build_evidence(_record(), _ocr(), curated=True)["chain_version"] == CHAIN_VERSION

    def test_without_ocr_the_axis_is_declared_inadmissible_not_guessed(self):
        evidence = build_evidence(_record(), None, curated=False)
        assert evidence["ruler_refit"]["admissible"] is False


class TestBuildM0PrimeRecord:
    def test_does_not_mutate_the_frozen_input(self):
        record = _record()
        build_m0_prime_record(record, _ocr())
        assert "measured_value" not in record["measured_points"][0]
        assert "category_label" not in record["measured_points"][0]

    def test_runs_every_stage(self):
        chained = build_m0_prime_record(_record(), _ocr())
        assert chained["category_axis"]["status"] == "attached"
        assert chained["ruler_refit"]["admissible"] is True
        assert chained["mark_admission"]["policy"]
        assert chained["m0_prime_chain"] == CHAIN_VERSION

    def test_values_are_recomputed_from_the_fitted_axis(self):
        """Bars are read at the growing edge against the refitted ticks."""
        chained = build_m0_prime_record(_record(), _ocr())
        values = sorted(p["measured_value"] for p in chained["measured_points"])
        # Tops at y=300/200/100 map to 20/40/60% of a 0..40 axis.
        assert values == [
            pytest.approx(11.4, abs=1.0),
            pytest.approx(22.9, abs=1.0),
            pytest.approx(34.3, abs=1.0),
        ]
