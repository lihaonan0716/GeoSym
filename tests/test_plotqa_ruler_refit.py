"""Tests for the W1 value-axis refit.

The module's contract is narrow and load-bearing: fit the axis from tick labels
that were actually read, or declare the axis inadmissible. A guessed axis is what
produced the original failure -- measured values off by a constant factor, with
nothing in the record to signal they were untrustworthy.

Two bugs found during W1 development are pinned here as regressions:

- reading a bar at its bbox centre instead of its growing edge halves the value
  (observed as a 0.499 measured/truth ratio on every vertical bar);
- assuming the value axis is always y makes horizontal bar charts read category
  positions as values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.reasoning.plotqa_ruler_refit import (  # noqa: E402
    collect_axis_ticks,
    fit_value_axis,
    parse_tick_value,
    pixel_to_value,
    refit_record,
    resolve_value_axis,
    ticks_look_like_years,
    value_axis_orientation,
    value_pixel_for_mark,
)

# Plot area spanning x 100..500, y 50..400.
PLOT_AREA = [100.0, 50.0, 500.0, 400.0]


def _ocr_box(text, x0, y0, x1, y1, confidence=0.99):
    return {"text": text, "bbox": [x0, y0, x1, y1], "confidence": confidence}


def _y_tick(value, pixel_y):
    """A y-axis tick label: sits left of the plot, centred on its tick."""
    return _ocr_box(str(value), 70.0, pixel_y - 8.0, 95.0, pixel_y + 8.0)


class TestParseTickValue:
    @pytest.mark.parametrize("text,expected", [
        ("0", 0.0), ("25", 25.0), ("-10", -10.0), ("1,500", 1500.0),
        ("3.5", 3.5), ("40%", 40.0), ("$20", 20.0),
    ])
    def test_parses_numeric_labels(self, text, expected):
        assert parse_tick_value(text) == expected

    @pytest.mark.parametrize("text", ["", "   ", "Year", "Number of countries", None])
    def test_rejects_non_numeric(self, text):
        assert parse_tick_value(text) is None

    @pytest.mark.parametrize("text,expected", [
        ("2.000e+8", 2.0e8), ("0.000e+0", 0.0), ("1.000e+10", 1.0e10),
        ("5.000e+9", 5.0e9), ("-2.5e+3", -2500.0), ("1.5E+7", 1.5e7),
        ("3e8", 3.0e8),
    ])
    def test_parses_scientific_e_notation(self, text, expected):
        """Regression: E-notation ticks were read by OCR and then discarded.

        PlotQA labels large-magnitude axes as "2.000e+8". Without an exponent
        branch the label split into two numeric tokens and was rejected as
        ambiguous, emptying the value axis on 38 frozen-roster samples whose
        ticks OCR had read perfectly. Value accuracy rose 0.786 -> 0.831 once
        they parsed.
        """
        assert parse_tick_value(text) == pytest.approx(expected)

    def test_rejects_multi_number_labels(self):
        """Two numbers in one box means a merged or wrapped label, not a tick."""
        assert parse_tick_value("10 20") is None

    def test_rejects_two_e_notation_numbers_in_one_box(self):
        assert parse_tick_value("1.2e+3 4.5e+6") is None

    def test_rejects_mostly_alphabetic_labels(self):
        assert parse_tick_value("Q1 2020") is None


class TestCollectAxisTicks:
    def test_collects_left_axis_ticks_in_order(self):
        ocr = {"ocr_boxes": [_y_tick(0, 390), _y_tick(10, 290), _y_tick(20, 190)]}
        ticks = collect_axis_ticks(ocr["ocr_boxes"], PLOT_AREA, axis="y")
        assert [t["value"] for t in ticks] == [20.0, 10.0, 0.0]
        assert [t["pixel"] for t in ticks] == [190.0, 290.0, 390.0]

    def test_tolerates_a_label_slightly_inside_the_plot_edge(self):
        """The plot-area bbox is itself estimated; labels can land just inside.

        This is the regression for the original `cx <= left + 4` rule, which
        dropped ticks at cx=38 when the fitted left edge was 32.
        """
        ocr = {"ocr_boxes": [_ocr_box("15", 96.0, 120.0, 106.0, 138.0)]}
        assert len(collect_axis_ticks(ocr["ocr_boxes"], PLOT_AREA, axis="y")) == 1

    def test_ignores_bottom_axis_labels_when_reading_y(self):
        ocr = {"ocr_boxes": [_ocr_box("2013", 200.0, 410.0, 240.0, 430.0)]}
        assert collect_axis_ticks(ocr["ocr_boxes"], PLOT_AREA, axis="y") == []

    def test_collects_bottom_axis_when_reading_x(self):
        ocr = {"ocr_boxes": [
            _ocr_box("0", 100.0, 405.0, 112.0, 425.0),
            _ocr_box("20", 200.0, 405.0, 216.0, 425.0),
        ]}
        ticks = collect_axis_ticks(ocr["ocr_boxes"], PLOT_AREA, axis="x")
        assert [t["value"] for t in ticks] == [0.0, 20.0]

    def test_dedupes_one_label_read_twice_keeping_higher_confidence(self):
        ocr = {"ocr_boxes": [
            _ocr_box("10", 70.0, 282.0, 95.0, 298.0, confidence=0.4),
            _ocr_box("10", 71.0, 283.0, 96.0, 299.0, confidence=0.9),
        ]}
        ticks = collect_axis_ticks(ocr["ocr_boxes"], PLOT_AREA, axis="y")
        assert len(ticks) == 1
        assert ticks[0]["confidence"] == 0.9


class TestFitValueAxis:
    def test_refuses_to_fit_below_three_ticks(self):
        """Two points fit a line exactly, so they cannot evidence their own fit."""
        result = fit_value_axis([{"pixel": 100.0, "value": 10.0},
                                 {"pixel": 200.0, "value": 20.0}])
        assert result["admissible"] is False
        assert result["reason"] == "insufficient_ticks"

    def test_fits_a_clean_linear_axis(self):
        ticks = [{"pixel": 390.0, "value": 0.0},
                 {"pixel": 290.0, "value": 10.0},
                 {"pixel": 190.0, "value": 20.0}]
        result = fit_value_axis(ticks)
        assert result["admissible"] is True
        assert result["reason"] == "fitted_from_ocr_ticks"
        assert pixel_to_value(result, 340.0) == pytest.approx(5.0, abs=1e-6)

    def test_drops_a_single_mis_read_tick(self):
        ticks = [{"pixel": 390.0, "value": 0.0},
                 {"pixel": 340.0, "value": 5.0},
                 {"pixel": 290.0, "value": 10.0},
                 {"pixel": 240.0, "value": 7777.0}]
        result = fit_value_axis(ticks)
        assert result["admissible"] is True
        assert result["dropped_outlier_value"] == 7777.0
        assert pixel_to_value(result, 190.0) == pytest.approx(20.0, abs=0.5)

    def test_rejects_an_axis_whose_ticks_disagree(self):
        ticks = [{"pixel": 390.0, "value": 0.0},
                 {"pixel": 340.0, "value": 900.0},
                 {"pixel": 290.0, "value": 10.0},
                 {"pixel": 240.0, "value": -400.0}]
        result = fit_value_axis(ticks)
        assert result["admissible"] is False
        assert result["reason"] == "tick_residual_too_high"

    def test_inadmissible_fit_yields_no_value(self):
        """Refusing to guess is the mechanism, so this must return None."""
        assert pixel_to_value({"admissible": False}, 300.0) is None


class TestValuePixelForMark:
    def test_point_marker_is_read_at_its_centre(self):
        point = {"component_bbox": [200.0, 100.0, 213.0, 113.0],
                 "component_center_pixel": [206.5, 106.5]}
        assert value_pixel_for_mark(point, "y") == pytest.approx(106.5)

    def test_vertical_bar_is_read_at_its_top_edge(self):
        """Regression: reading a bar at its centre halves its value.

        This is the 0.499 measured/truth ratio the first W1 evaluation showed
        across every vertical bar sample.
        """
        point = {"component_bbox": [200.0, 118.0, 240.0, 631.0],
                 "component_center_pixel": [220.0, 374.5],
                 "pixel": [220.0, 118.0]}
        assert value_pixel_for_mark(point, "y") == pytest.approx(118.0)

    def test_bar_hanging_from_above_is_read_at_its_bottom_edge(self):
        """A bar plotting a negative value grows downward from the zero line.

        33 samples on the frozen roster plot negative values. The growing edge
        is identified as the one farther from where the fitted axis reads zero,
        so both directions work without knowing the sign beforehand.
        """
        # Axis reading zero at y=60: value = -0.1 * pixel + 6.0.
        axis_fit = {"admissible": True, "slope": -0.1, "intercept": 6.0}
        point = {"component_bbox": [200.0, 60.0, 240.0, 300.0],
                 "component_center_pixel": [220.0, 180.0],
                 "pixel": [220.0, 300.0]}
        assert value_pixel_for_mark(point, "y", axis_fit) == pytest.approx(300.0)

    def test_bar_rising_from_below_is_read_at_its_top_edge_using_the_zero_line(self):
        # Same axis, zero at y=400: a bar spanning 118..400 grows upward.
        axis_fit = {"admissible": True, "slope": -0.1, "intercept": 40.0}
        point = {"component_bbox": [200.0, 118.0, 240.0, 400.0],
                 "component_center_pixel": [220.0, 259.0],
                 "pixel": [220.0, 400.0]}
        assert value_pixel_for_mark(point, "y", axis_fit) == pytest.approx(118.0)

    def test_without_a_usable_fit_the_conventional_edge_is_used(self):
        point = {"component_bbox": [200.0, 118.0, 240.0, 631.0],
                 "component_center_pixel": [220.0, 374.5],
                 "pixel": [220.0, 631.0]}
        assert value_pixel_for_mark(point, "y") == pytest.approx(118.0)
        assert value_pixel_for_mark(
            point, "y", {"admissible": False}) == pytest.approx(118.0)

    def test_horizontal_bar_is_read_at_its_right_edge_on_the_x_axis(self):
        point = {"component_bbox": [100.0, 200.0, 460.0, 226.0],
                 "component_center_pixel": [280.0, 213.0],
                 "pixel": [460.0, 213.0]}
        assert value_pixel_for_mark(point, "x") == pytest.approx(460.0)

    def test_horizontal_bar_anchored_at_its_baseline_is_still_read_at_the_right_edge(self):
        """Regression: the extractor anchors horizontal bars at the baseline.

        Edge choice used to follow whichever bbox edge the extractor's ``pixel``
        anchor was nearest. On horizontal bars that anchor sits at the *left*
        baseline for 3748 of 3829 components on the frozen roster, so the rule
        read every bar at its zero end -- measured/truth median 0.0039 against
        1.003 for the right edge, and hbar value accuracy of 0.032.

        Which edge grows is a property of the chart's orientation, not of an
        individual mark, so it is no longer inferred from the anchor.
        """
        point = {"component_bbox": [100.0, 200.0, 460.0, 226.0],
                 "component_center_pixel": [280.0, 213.0],
                 "pixel": [100.0, 213.0]}
        assert value_pixel_for_mark(point, "x") == pytest.approx(460.0)

    def test_missing_bbox_falls_back_to_the_centre(self):
        assert value_pixel_for_mark({"component_center_pixel": [10.0, 20.0]}, "y") == 20.0


class TestValueAxisOrientation:
    def _bar(self, x0, y0, x1, y1):
        return {"component_bbox": [x0, y0, x1, y1]}

    def test_vertical_bars_imply_a_y_value_axis(self):
        record = {"measured_points": [self._bar(200, 118, 240, 631),
                                      self._bar(260, 200, 300, 631)]}
        assert value_axis_orientation(record) == "y"

    def test_horizontal_bars_sharing_a_baseline_imply_an_x_value_axis(self):
        record = {"measured_points": [self._bar(100, 200, 460, 226),
                                      self._bar(100, 240, 380, 266),
                                      self._bar(100, 280, 420, 306)]}
        assert value_axis_orientation(record) == "x"

    def test_polyline_components_do_not_imply_an_x_value_axis(self):
        """Regression: a wide polyline blob is not a horizontal bar.

        Line components are wide but start all across the plot, unlike bars
        which share a baseline. Aspect ratio alone misclassified 42/57 line
        samples.
        """
        # Segment starts spread across 600px while each segment is ~60px long,
        # a ratio of 10 -- inside the measured line population (p10 = 2.9) and
        # far above the horizontal-bar population (p90 = 2.53).
        record = {"measured_points": [self._bar(100, 200, 160, 210),
                                      self._bar(300, 180, 360, 190),
                                      self._bar(700, 260, 760, 270)]}
        assert value_axis_orientation(record) == "y"

    def test_bars_of_differing_length_still_read_as_horizontal(self):
        """Bars share a left baseline even when their lengths differ widely."""
        record = {"measured_points": [self._bar(100, 200, 460, 226),
                                      self._bar(100, 240, 160, 266),
                                      self._bar(100, 280, 300, 306)]}
        assert value_axis_orientation(record) == "x"

    def test_point_markers_default_to_y(self):
        record = {"measured_points": [{"component_bbox": [200, 100, 213, 113]}]}
        assert value_axis_orientation(record) == "y"

    def test_empty_record_defaults_to_y(self):
        assert value_axis_orientation({"measured_points": []}) == "y"


class TestTicksLookLikeYears:
    def test_year_axis_is_recognised(self):
        assert ticks_look_like_years([2010.0, 2011.0, 2012.0, 2013.0]) is True

    def test_value_axis_is_not(self):
        assert ticks_look_like_years([0.0, 20.0, 40.0, 60.0]) is False

    def test_empty_is_not_a_year_axis(self):
        assert ticks_look_like_years([]) is False

    def test_a_value_axis_reaching_into_year_range_is_not_flagged(self):
        """Large quantities can exceed 1800 without being years."""
        assert ticks_look_like_years([0.0, 1000.0, 2000.0, 3000.0]) is False


class TestResolveValueAxis:
    def _record(self, bars):
        return {"measured_points": bars}

    def _vertical_bars(self):
        return [{"component_bbox": [100 + i * 60, 200, 140 + i * 60, 400]} for i in range(3)]

    def _horizontal_bars(self):
        return [{"component_bbox": [100, 200 + i * 40, 300 + i * 50, 226 + i * 40]}
                for i in range(3)]

    def test_uses_the_geometry_prior_when_it_yields_a_value_fit(self):
        ocr = {"ocr_boxes": [_y_tick(0, 390), _y_tick(10, 290), _y_tick(20, 190)]}
        axis, _ticks, fit, trace = resolve_value_axis(
            self._record(self._vertical_bars()), ocr, PLOT_AREA)
        assert axis == "y"
        assert fit["admissible"] is True
        assert trace["fallback_used"] is False

    def test_falls_back_to_the_other_axis_when_the_prior_holds_years(self):
        """Regression: a year category axis fits a line as well as a value axis.

        Deciding on fit quality alone scored 84.8% on the frozen roster, worse
        than geometry alone (93.8%); geometry-first with this fallback scored
        98.8%.
        """
        ocr = {"ocr_boxes": [
            _y_tick(2010, 390), _y_tick(2011, 290), _y_tick(2012, 190),
            _ocr_box("0", 100.0, 405.0, 112.0, 425.0),
            _ocr_box("20", 200.0, 405.0, 216.0, 425.0),
            _ocr_box("40", 300.0, 405.0, 316.0, 425.0),
        ]}
        axis, _ticks, fit, trace = resolve_value_axis(
            self._record(self._vertical_bars()), ocr, PLOT_AREA)
        assert axis == "x"
        assert fit["admissible"] is True
        assert trace["fallback_used"] is True
        assert trace["prior_rejected_because"] == "year_category_axis"

    def test_falls_back_when_the_prior_has_too_few_ticks(self):
        ocr = {"ocr_boxes": [
            _y_tick(0, 390),
            _ocr_box("0", 100.0, 405.0, 112.0, 425.0),
            _ocr_box("20", 200.0, 405.0, 216.0, 425.0),
            _ocr_box("40", 300.0, 405.0, 316.0, 425.0),
        ]}
        axis, _ticks, _fit, trace = resolve_value_axis(
            self._record(self._vertical_bars()), ocr, PLOT_AREA)
        assert axis == "x"
        assert trace["fallback_used"] is True

    def test_horizontal_bars_resolve_to_the_bottom_axis(self):
        ocr = {"ocr_boxes": [
            _ocr_box("0", 100.0, 405.0, 112.0, 425.0),
            _ocr_box("20", 200.0, 405.0, 216.0, 425.0),
            _ocr_box("40", 300.0, 405.0, 316.0, 425.0),
        ]}
        axis, _ticks, fit, _trace = resolve_value_axis(
            self._record(self._horizontal_bars()), ocr, PLOT_AREA)
        assert axis == "x"
        assert fit["admissible"] is True

    def test_reports_when_neither_axis_is_usable(self):
        ocr = {"ocr_boxes": [_y_tick(2010, 390), _y_tick(2011, 290), _y_tick(2012, 190)]}
        _axis, _ticks, fit, trace = resolve_value_axis(
            self._record(self._vertical_bars()), ocr, PLOT_AREA)
        assert fit["admissible"] is False or trace.get("both_axes_unusable")


class TestRefitRecord:
    def _record(self):
        return {
            "sample_id": "s1",
            "answer_type": "numeric",
            "chart_geometry": {"plot_area_bbox": PLOT_AREA},
            "measured_points": [{
                "component_bbox": [200.0, 290.0, 240.0, 390.0],
                "component_center_pixel": [220.0, 340.0],
                "pixel": [220.0, 290.0],
                "y_value": 999.0,
            }],
        }

    def _ocr(self):
        return {"ocr_boxes": [_y_tick(0, 390), _y_tick(10, 290), _y_tick(20, 190)]}

    def test_recomputes_the_value_from_the_fitted_axis(self):
        refitted = refit_record(self._record(), self._ocr())
        point = refitted["measured_points"][0]
        assert point["measured_value"] == pytest.approx(10.0, abs=1e-6)
        assert point["value_status"] == "refit_from_ocr_ticks"

    def test_keeps_y_value_in_step_for_frozen_schema_consumers(self):
        refitted = refit_record(self._record(), self._ocr())
        point = refitted["measured_points"][0]
        assert point["y_value"] == point["measured_value"]

    def test_does_not_mutate_the_frozen_input(self):
        record = self._record()
        refit_record(record, self._ocr())
        assert record["measured_points"][0]["y_value"] == 999.0
        assert "measured_value" not in record["measured_points"][0]

    def test_inadmissible_axis_blanks_the_value_rather_than_guessing(self):
        record = self._record()
        refitted = refit_record(record, {"ocr_boxes": [_y_tick(0, 390)]})
        point = refitted["measured_points"][0]
        assert point["measured_value"] is None
        assert point["y_value"] is None
        assert point["value_status"] == "unmeasurable_axis_not_admissible"
        assert refitted["ruler_refit"]["admissible"] is False

    def test_missing_ocr_is_reported(self):
        refitted = refit_record(self._record(), None)
        assert refitted["ruler_refit"]["reason"] == "missing_ocr_or_geometry"

    def test_records_which_axis_carried_the_value(self):
        refitted = refit_record(self._record(), self._ocr())
        assert refitted["ruler_refit"]["value_axis"] == "y"

    def test_never_reads_the_reference_answer(self):
        """Answer independence keeps the refit a pipeline stage, not an oracle."""
        ocr = self._ocr()
        first = refit_record({**self._record(), "reference": 1.0}, ocr)
        second = refit_record({**self._record(), "reference": 12345.0}, ocr)
        assert first["measured_points"] == second["measured_points"]
