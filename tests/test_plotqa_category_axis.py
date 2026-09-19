"""Tests for x-axis category construction (the M0' extraction gap).

The module exists because the frozen extractor only looked for categories to the
*left* of the plot area, which is correct for horizontal bars and wrong for the
other three chart orientations. These tests pin the new orientation's behaviour
and, just as importantly, the guards that stop it from inventing labels: a
mislabelled point is worse than an unlabelled one, because it makes wrong
evidence look attributable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.reasoning.plotqa_category_axis import (  # noqa: E402
    attach_x_axis_categories,
    augment_record,
    coverage_report,
    x_axis_category_candidates,
)

# A plot area spanning x 100..500, with the axis line at y=400.
PLOT_AREA = [100.0, 50.0, 500.0, 400.0]


def _ocr_box(text, x0, y0, x1, y1, confidence=0.9):
    return {"text": text, "bbox": [x0, y0, x1, y1], "confidence": confidence}


def _mark(cx, bbox_width=20.0, top=200.0, bottom=400.0):
    half = bbox_width / 2.0
    return {
        "component_bbox": [cx - half, top, cx + half, bottom],
        "component_center_pixel": [cx, (top + bottom) / 2.0],
    }


class TestCandidateCollection:
    def test_collects_labels_in_the_bottom_band(self):
        ocr = {"ocr_boxes": [
            _ocr_box("Brazil", 140, 410, 200, 430),
            _ocr_box("Chile", 240, 410, 290, 430),
        ]}
        labels = [c["label"] for c in x_axis_category_candidates(ocr, PLOT_AREA)]
        assert labels == ["Brazil", "Chile"]

    def test_ignores_text_above_the_plot(self):
        """A chart title must not become a category."""
        ocr = {"ocr_boxes": [_ocr_box("Broad money growth", 140, 10, 300, 34)]}
        assert x_axis_category_candidates(ocr, PLOT_AREA) == []

    def test_ignores_text_far_below_the_axis(self):
        ocr = {"ocr_boxes": [_ocr_box("Source: World Bank", 140, 600, 300, 624)]}
        assert x_axis_category_candidates(ocr, PLOT_AREA) == []

    def test_ignores_y_axis_tick_text_left_of_the_plot(self):
        ocr = {"ocr_boxes": [_ocr_box("Angola", 10, 410, 60, 430)]}
        assert x_axis_category_candidates(ocr, PLOT_AREA) == []

    def test_drops_pure_numeric_ticks_by_default(self):
        ocr = {"ocr_boxes": [
            _ocr_box("2013", 140, 410, 180, 430),
            _ocr_box("Peru", 240, 410, 280, 430),
        ]}
        labels = [c["label"] for c in x_axis_category_candidates(ocr, PLOT_AREA)]
        assert labels == ["Peru"]

    def test_numeric_categories_can_be_kept_when_the_caller_knows(self):
        ocr = {"ocr_boxes": [_ocr_box("2013", 140, 410, 180, 430)]}
        labels = x_axis_category_candidates(ocr, PLOT_AREA, exclude_numeric=False)
        assert [c["label"] for c in labels] == ["2013"]

    @pytest.mark.parametrize("title", ["Year", "Years", "Number of countries", "percentage"])
    def test_drops_axis_titles(self, title):
        ocr = {"ocr_boxes": [_ocr_box(title, 260, 410, 330, 430)]}
        assert x_axis_category_candidates(ocr, PLOT_AREA) == []

    def test_drops_boxes_too_wide_to_be_one_category(self):
        """A box spanning most of the plot width is a title, whatever it says."""
        ocr = {"ocr_boxes": [_ocr_box("Gross domestic savings", 110, 410, 480, 430)]}
        assert x_axis_category_candidates(ocr, PLOT_AREA) == []

    def test_keeps_a_genuine_category_containing_a_title_word(self):
        """'Year' as a token must not blacklist 'New Year Island'."""
        ocr = {"ocr_boxes": [_ocr_box("New Year Island", 200, 410, 280, 430)]}
        labels = [c["label"] for c in x_axis_category_candidates(ocr, PLOT_AREA)]
        assert labels == ["New Year Island"]

    def test_dedupes_wrapped_labels_keeping_the_widest(self):
        ocr = {"ocr_boxes": [
            _ocr_box("St. Kitts", 200, 410, 250, 428),
            _ocr_box("St. Kitts and Nevis", 202, 430, 290, 448),
        ]}
        labels = x_axis_category_candidates(ocr, PLOT_AREA)
        assert len(labels) == 1
        assert labels[0]["label"] == "St. Kitts and Nevis"


class TestAttachment:
    def test_assigns_nearest_column(self):
        candidates = x_axis_category_candidates(
            {"ocr_boxes": [_ocr_box("Brazil", 140, 410, 200, 430),
                           _ocr_box("Chile", 300, 410, 350, 430)]}, PLOT_AREA)
        points = [_mark(170.0), _mark(325.0)]
        summary = attach_x_axis_categories(points, candidates)
        assert summary["assigned_count"] == 2
        assert points[0]["category_label"] == "Brazil"
        assert points[1]["category_label"] == "Chile"

    def test_leaves_distant_marks_unlabelled(self):
        """Better an unlabelled point than a confidently mislabelled one."""
        candidates = x_axis_category_candidates(
            {"ocr_boxes": [_ocr_box("Brazil", 140, 410, 200, 430)]}, PLOT_AREA)
        points = [_mark(480.0, bbox_width=6.0)]
        summary = attach_x_axis_categories(points, candidates)
        assert summary["assigned_count"] == 0
        assert "category_label" not in points[0]

    def test_never_overwrites_an_existing_label(self):
        candidates = x_axis_category_candidates(
            {"ocr_boxes": [_ocr_box("Brazil", 140, 410, 200, 430)]}, PLOT_AREA)
        point = _mark(170.0)
        point["category_label"] = "PreExisting"
        attach_x_axis_categories([point], candidates)
        assert point["category_label"] == "PreExisting"

    def test_records_its_support_provenance(self):
        candidates = x_axis_category_candidates(
            {"ocr_boxes": [_ocr_box("Brazil", 140, 410, 200, 430)]}, PLOT_AREA)
        points = [_mark(170.0)]
        attach_x_axis_categories(points, candidates)
        assert "x_axis_category_column" in points[0]["support"]

    def test_no_candidates_is_reported_not_crashed(self):
        summary = attach_x_axis_categories([_mark(170.0)], [])
        assert summary["assigned_count"] == 0
        assert summary["status"] == "no_candidates"


class TestAugmentRecord:
    def _record(self):
        return {
            "sample_id": "s1",
            "chart_geometry": {"plot_area_bbox": PLOT_AREA},
            "measured_points": [_mark(170.0)],
        }

    def test_does_not_mutate_the_frozen_input(self):
        """The frozen artifact is an input to a frozen experiment."""
        record = self._record()
        augment_record(record, {"ocr_boxes": [_ocr_box("Brazil", 140, 410, 200, 430)]})
        assert "category_label" not in record["measured_points"][0]

    def test_attaches_on_the_copy(self):
        augmented = augment_record(self._record(), {"ocr_boxes": [
            _ocr_box("Brazil", 140, 410, 200, 430),
            _ocr_box("Chile", 240, 410, 300, 430),
        ]})
        assert augmented["measured_points"][0]["category_label"] == "Brazil"
        assert augmented["category_axis"]["assigned_count"] == 1

    def test_a_lone_label_is_not_treated_as_a_category_axis(self):
        """Regression: a single bottom-band box is an axis title, not a category.

        A category axis labels several columns. Attaching a lone survivor
        stamped one identical label onto every mark in the chart on 85 of the
        frozen 500 samples -- "Cost (US$ per container)", "AngolaHong
        KongMacedonia", "0.000e+0". That is worse than no label: it asserts an
        association the chart does not make.
        """
        augmented = augment_record(self._record(), {"ocr_boxes": [
            _ocr_box("Cost (US$ per container)", 140, 410, 200, 430)]})
        assert "category_label" not in augmented["measured_points"][0]
        assert augmented["category_axis"]["assigned_count"] == 0

    def test_axis_title_below_the_label_row_is_dropped(self):
        """Regression: 'Country' sat a row below the real labels on 39 samples.

        Category labels share a row under the axis; the title sits on its own
        line beneath it. The test is positional, not lexical, because the
        offending strings have nothing in common as text.
        """
        record = {
            "sample_id": "s1",
            "chart_geometry": {"plot_area_bbox": PLOT_AREA},
            "measured_points": [_mark(170.0), _mark(270.0), _mark(370.0)],
        }
        augmented = augment_record(record, {"ocr_boxes": [
            _ocr_box("Brazil", 150, 405, 190, 425),
            _ocr_box("Chile", 250, 405, 290, 425),
            _ocr_box("Peru", 350, 405, 390, 425),
            # The axis title, one row below the labels.
            _ocr_box("Country", 235, 470, 305, 490),
        ]})
        labels = {p.get("category_label") for p in augmented["measured_points"]}
        assert "Country" not in labels
        assert labels == {"Brazil", "Chile", "Peru"}

    def test_missing_ocr_row_is_reported(self):
        augmented = augment_record(self._record(), None)
        assert augmented["category_axis"]["status"] == "no_ocr_row"

    def test_missing_plot_area_is_reported(self):
        record = {"sample_id": "s1", "chart_geometry": {}, "measured_points": []}
        augmented = augment_record(record, {"ocr_boxes": []})
        assert augmented["category_axis"]["status"] == "no_plot_area"

    def test_never_reads_the_reference_answer(self):
        ocr = {"ocr_boxes": [_ocr_box("Brazil", 140, 410, 200, 430)]}
        first = augment_record({**self._record(), "reference": 1.0}, ocr)
        second = augment_record({**self._record(), "reference": 999.0}, ocr)
        assert first["measured_points"] == second["measured_points"]


class TestCoverageReport:
    def test_counts_samples_and_points(self):
        records = [
            {"measured_points": [{"category_label": "a"}, {}]},
            {"measured_points": [{}]},
        ]
        report = coverage_report(records)
        assert report["sample_count"] == 2
        assert report["samples_with_any_category_label"] == 1
        assert report["total_measured_points"] == 3
        assert report["points_with_category_label"] == 1

    def test_empty_input_does_not_divide_by_zero(self):
        report = coverage_report([])
        assert report["point_label_rate"] == 0.0
        assert report["sample_label_rate"] == 0.0
