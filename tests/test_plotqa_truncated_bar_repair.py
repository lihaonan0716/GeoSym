"""Regression tests for the truncated-bar repair (2026-09-20).

`infer_plot_area` puts the right edge at the last x tick *label*, but a
horizontal bar routinely runs past it -- an axis labelled to 80 carries a bar of
77.5 whose pixels end beyond the "80" text. The ROI crop truncated every such
bar to the same boundary pixel, so a whole series read as one identical value,
stamped `refit_from_ocr_ticks` like a real measurement.

A first attempt widened the detection window instead of repairing bars. It
admitted legend swatches and stray marks into every chart type and dropped value
accuracy 0.873 -> 0.760, including on `vbar` and `dot`, which have no cropping
at all. That failure is why the tests below weigh so heavily towards negative
controls and towards proving the mark count cannot change: repairing existing
bars is safe in a way that re-detecting them is not.

Measured on the frozen roster with the strict rule: 608 bars repaired, hbar
value accuracy 0.866 -> 0.902, other chart types bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.run_plotqa_measured_m0 import (
    BAR_BASELINE_TOLERANCE_FRACTION,
    BAR_ELONGATION_RATIO,
    BAR_TRUNCATION_TOLERANCE_PX,
    _extend_truncated_bars,
)

PLOT_AREA = [60.0, 39.0, 776.0, 554.0]
BAR_COLOUR = (224, 144, 96)


def _image(width: int = 882, height: int = 650) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _draw_bar(image: np.ndarray, x_lo: int, y_lo: int, x_hi: int, y_hi: int,
              colour: tuple[int, int, int] = BAR_COLOUR) -> None:
    image[y_lo:y_hi, x_lo:x_hi] = colour


def _point(x_lo: float, y_lo: float, x_hi: float, y_hi: float) -> dict:
    return {"fact_id": "m0_000", "component_bbox": [x_lo, y_lo, x_hi, y_hi], "x_value": 71.29}


X_MODEL = {"type": "linear", "params": [0.0998, -6.1526]}


class TestTruncatedBarsAreRepaired:
    def test_a_bar_cut_off_by_the_plot_edge_is_extended(self):
        image = _image()
        _draw_bar(image, 64, 58, 838, 80)          # true extent runs to 838
        points = [_point(64, 58, 776, 80)]          # recorded as cut at 776
        assert _extend_truncated_bars(image, points, PLOT_AREA) == 1
        assert points[0]["component_bbox"][2] == pytest.approx(837, abs=2)

    def test_the_pre_repair_edge_is_preserved(self):
        image = _image()
        _draw_bar(image, 64, 58, 838, 80)
        points = [_point(64, 58, 776, 80)]
        _extend_truncated_bars(image, points, PLOT_AREA)
        assert points[0]["right_edge_before_truncation_repair"] == 776.0

    def test_the_stale_x_value_is_recomputed(self):
        """The bbox and the stored value must not disagree after a repair."""
        image = _image()
        _draw_bar(image, 64, 58, 838, 80)
        points = [_point(64, 58, 776, 80)]
        _extend_truncated_bars(image, points, PLOT_AREA, X_MODEL)
        assert points[0]["x_value_before_truncation_repair"] == 71.29
        assert points[0]["x_value"] > 75.0

    def test_bars_of_different_lengths_get_different_values(self):
        """The defect's signature was a whole series collapsing to one value."""
        image = _image()
        _draw_bar(image, 64, 58, 787, 80)
        _draw_bar(image, 64, 181, 838, 204)
        points = [_point(64, 58, 776, 80), _point(64, 181, 776, 204)]
        _extend_truncated_bars(image, points, PLOT_AREA, X_MODEL)
        assert points[0]["x_value"] != points[1]["x_value"]


class TestNegativeControls:
    """What the repair must refuse to touch. The loose version failed these."""

    def test_a_bar_ending_inside_the_plot_is_untouched(self):
        image = _image()
        _draw_bar(image, 64, 58, 400, 80)
        points = [_point(64, 58, 400, 80)]
        assert _extend_truncated_bars(image, points, PLOT_AREA) == 0
        assert points[0]["component_bbox"][2] == 400

    def test_a_legend_swatch_is_not_extended(self):
        """Bar-shaped, coloured, but not grown from the baseline."""
        image = _image()
        _draw_bar(image, 790, 58, 860, 80)
        points = [_point(790, 58, 860, 80)]
        assert _extend_truncated_bars(image, points, PLOT_AREA) == 0

    def test_a_point_marker_is_not_extended(self):
        image = _image()
        _draw_bar(image, 770, 300, 782, 312)
        points = [_point(770, 300, 782, 312)]
        assert _extend_truncated_bars(image, points, PLOT_AREA) == 0

    def test_a_vertical_bar_is_not_extended(self):
        """Vertical bars encode value on y; cropping never affected them."""
        image = _image()
        _draw_bar(image, 100, 100, 130, 554)
        points = [_point(100, 100, 130, 554)]
        assert _extend_truncated_bars(image, points, PLOT_AREA) == 0

    def test_a_floating_mid_plot_rule_is_not_extended(self):
        image = _image()
        _draw_bar(image, 400, 100, 776, 130)
        points = [_point(400, 100, 776, 130)]
        assert _extend_truncated_bars(image, points, PLOT_AREA) == 0


class TestRepairCannotInventMarks:
    def test_the_mark_count_never_changes(self):
        """Repairing is not detecting: the list length is invariant."""
        image = _image()
        _draw_bar(image, 64, 58, 838, 80)
        _draw_bar(image, 790, 300, 860, 320)  # a legend swatch, must stay out
        points = [_point(64, 58, 776, 80)]
        before = len(points)
        _extend_truncated_bars(image, points, PLOT_AREA)
        assert len(points) == before

    def test_an_edge_that_would_shrink_is_rejected(self):
        """A repair may only extend; shrinking would fabricate a smaller value."""
        image = _image()
        _draw_bar(image, 64, 58, 700, 80)
        points = [_point(64, 58, 776, 80)]
        _extend_truncated_bars(image, points, PLOT_AREA)
        assert points[0]["component_bbox"][2] == 776.0


class TestDegradesSafely:
    @pytest.mark.parametrize("plot_area", [None, [], [1.0, 2.0]])
    def test_missing_plot_area_is_a_no_op(self, plot_area):
        points = [_point(64, 58, 776, 80)]
        assert _extend_truncated_bars(_image(), points, plot_area) == 0

    def test_missing_image_is_a_no_op(self):
        points = [_point(64, 58, 776, 80)]
        assert _extend_truncated_bars(None, points, PLOT_AREA) == 0

    def test_malformed_bbox_is_skipped(self):
        points = [{"component_bbox": [1.0, 2.0]}, {"component_bbox": None}, {}]
        assert _extend_truncated_bars(_image(), points, PLOT_AREA) == 0

    def test_without_an_axis_model_geometry_is_still_repaired(self):
        image = _image()
        _draw_bar(image, 64, 58, 838, 80)
        points = [_point(64, 58, 776, 80)]
        assert _extend_truncated_bars(image, points, PLOT_AREA, None) == 1


class TestThresholdsAgreeAcrossModules:
    def test_elongation_matches_the_refit_module(self):
        from evaluator.reasoning.plotqa_ruler_refit import BAR_ASPECT_RATIO

        assert BAR_ELONGATION_RATIO == BAR_ASPECT_RATIO

    def test_thresholds_are_bounded(self):
        assert 0 < BAR_TRUNCATION_TOLERANCE_PX <= 5
        assert 0 < BAR_BASELINE_TOLERANCE_FRACTION <= 0.5
