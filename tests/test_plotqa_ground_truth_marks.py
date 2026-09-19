"""Tests for the official-annotation reader used by every PlotQA evaluation.

This function decides which samples an evaluation can see at all: a sample whose
annotation yields no marks is skipped silently by each ``evaluate_*`` script. A
defect here therefore does not produce a wrong number, it produces a number
computed on a subset -- which is harder to notice and easier to believe.

That is exactly what happened. PlotQA stores the value axis as ``y`` for
vertical charts but as ``x`` for horizontal bars, where ``y`` holds the category
strings instead. Reading ``y`` unconditionally dropped all 168 ``hbar`` samples
of the frozen 500, so W1/W2/W4 were all reported on 331 samples while appearing
to cover the roster.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_plotqa_mark_segmentation import ground_truth_marks  # noqa: E402


def _box(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h}


class TestVerticalCharts:
    def test_reads_values_from_the_y_series(self):
        annotation = {"models": [{
            "name": "Export",
            "bboxes": [_box(100, 300, 40, 200), _box(200, 250, 40, 250)],
            "x": ["2007", "2008"],
            "y": [12.5, 18.0],
        }]}
        marks = ground_truth_marks(annotation)
        assert [m["value"] for m in marks] == [12.5, 18.0]
        assert [m["series_label"] for m in marks] == ["Export", "Export"]

    def test_mark_centres_come_from_the_bbox(self):
        annotation = {"models": [{
            "name": "S", "bboxes": [_box(100, 300, 40, 200)], "y": [1.0],
        }]}
        mark = ground_truth_marks(annotation)[0]
        assert mark["center_x"] == 120.0
        assert mark["center_y"] == 400.0


class TestHorizontalBars:
    """Regression: hbar annotations put values in ``x`` and categories in ``y``."""

    def test_reads_values_from_the_x_series_when_y_holds_categories(self):
        annotation = {"models": [{
            "name": "Export",
            "bboxes": [_box(69, 541, 543, 30), _box(69, 390, 799, 30)],
            "y": ["2007", "2008"],
            "x": [1235, 1816],
        }]}
        marks = ground_truth_marks(annotation)
        assert [m["value"] for m in marks] == [1235.0, 1816.0]

    def test_every_series_of_a_grouped_horizontal_chart_is_read(self):
        annotation = {"models": [
            {"name": "Export", "bboxes": [_box(69, 541, 543, 30)],
             "y": ["2007"], "x": [1235]},
            {"name": "Import", "bboxes": [_box(69, 511, 635, 30)],
             "y": ["2007"], "x": [1445]},
        ]}
        marks = ground_truth_marks(annotation)
        assert [(m["series_label"], m["value"]) for m in marks] == [
            ("Export", 1235.0), ("Import", 1445.0)]

    def test_y_is_still_preferred_when_both_axes_are_numeric(self):
        """A numeric category axis -- years -- must not displace the values."""
        annotation = {"models": [{
            "name": "S",
            "bboxes": [_box(100, 300, 40, 200), _box(200, 250, 40, 250)],
            "x": [2007, 2008],
            "y": [12.5, 18.0],
        }]}
        assert [m["value"] for m in ground_truth_marks(annotation)] == [12.5, 18.0]


class TestDegenerateInput:
    def test_annotation_without_models_yields_nothing(self):
        assert ground_truth_marks({}) == []

    def test_series_with_no_numeric_axis_is_skipped(self):
        annotation = {"models": [{
            "name": "S", "bboxes": [_box(0, 0, 10, 10)],
            "x": ["a", "b"], "y": ["c", "d"],
        }]}
        assert ground_truth_marks(annotation) == []

    def test_values_are_paired_with_boxes_by_index_and_extras_ignored(self):
        annotation = {"models": [{
            "name": "S",
            "bboxes": [_box(0, 0, 10, 10)],
            "y": [1.0, 2.0, 3.0],
        }]}
        assert [m["value"] for m in ground_truth_marks(annotation)] == [1.0]

    def test_booleans_do_not_count_as_numeric_values(self):
        annotation = {"models": [{
            "name": "S", "bboxes": [_box(0, 0, 10, 10)], "y": [True],
        }]}
        assert ground_truth_marks(annotation) == []
