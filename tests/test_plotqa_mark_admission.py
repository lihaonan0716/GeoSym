"""Tests for the W2 mark-admission filter.

The filter removes sliver artifacts from measured evidence. Its risk is the
mirror image of its purpose: a rule aggressive enough to drop one-pixel
fragments can also delete a line chart's entire stroke-based evidence. Both
behaviours are pinned here, including the stand-down guard and the two
chart-type heuristics that were tried and rejected on the frozen roster.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.reasoning.plotqa_mark_admission import (  # noqa: E402
    MIN_MARK_SIDE_PX,
    SURVIVAL_FLOOR,
    admit_marks,
    apply_admission,
    filtering_would_gut_evidence,
    is_polyline_evidence,
)


def _point(x0, y0, x1, y1, source="cv_color_component", area=None):
    point = {"component_bbox": [x0, y0, x1, y1], "source": source}
    if area is not None:
        point["component_area"] = area
    return point


def _bar(index=0):
    """A compact, mark-sized component."""
    return _point(100 + index * 40, 200, 130 + index * 40, 400)


def _sliver(index=0):
    """A one-pixel artifact: grid line, axis edge, anti-aliasing."""
    return _point(100 + index * 10, 200, 101 + index * 10, 400)


class TestAdmitMarks:
    def test_keeps_mark_sized_components(self):
        result = admit_marks([_bar(0), _bar(1), _bar(2)])
        assert len(result["admitted"]) == 3
        assert result["rejected"] == []

    def test_rejects_slivers(self):
        result = admit_marks([_bar(0), _bar(1), _bar(2), _sliver(0)])
        assert len(result["admitted"]) == 3
        assert result["rejected_counts"]["sliver_below_min_side"] == 1

    def test_rejects_degenerate_boxes(self):
        result = admit_marks([_bar(0), _bar(1), _bar(2), _point(10, 10, 10, 10)])
        assert result["rejected_counts"]["degenerate_bbox"] == 1

    def test_keeps_points_without_geometry(self):
        """Absence of evidence about a point is not evidence it is an artifact."""
        result = admit_marks([_bar(0), _bar(1), _bar(2), {"source": "x"}])
        assert any(p.get("admission_status") == "admitted_no_geometry"
                   for p in result["admitted"])

    def test_rejects_area_fragments_that_pass_the_side_test(self):
        big = [_point(100 + i * 60, 100, 150 + i * 60, 500, area=20000) for i in range(3)]
        fragment = _point(400, 100, 406, 106, area=36)
        result = admit_marks(big + [fragment])
        assert result["rejected_counts"].get("area_fragment") == 1

    def test_annotates_rather_than_deletes(self):
        original = _bar(0)
        result = admit_marks([original, _bar(1), _bar(2)])
        assert "admission_status" not in original
        assert result["admitted"][0]["admission_status"] == "admitted"

    def test_reports_the_policy_it_applied(self):
        result = admit_marks([_bar(0), _bar(1), _bar(2)])
        assert "min_side" in result["policy"]


class TestStandDownGuard:
    def test_stands_down_when_almost_everything_would_be_dropped(self):
        """Regression: the sliver rule collapsed line-chart recall 0.194 -> 0.040.

        A sample about to lose nearly all components is not being cleaned, it is
        being emptied, so the filter passes the evidence through instead.
        """
        points = [_sliver(i) for i in range(10)] + [_bar(0)]
        assert filtering_would_gut_evidence(points, MIN_MARK_SIDE_PX) is True
        result = admit_marks(points)
        assert len(result["admitted"]) == len(points)
        assert result["policy"] == "filter_stood_down_survival_below_floor"
        assert all(p["admission_status"] == "admitted_filter_stood_down"
                   for p in result["admitted"])

    def test_does_not_stand_down_when_most_components_survive(self):
        points = [_bar(i) for i in range(8)] + [_sliver(0)]
        assert filtering_would_gut_evidence(points, MIN_MARK_SIDE_PX) is False
        result = admit_marks(points)
        assert result["rejected_counts"]["sliver_below_min_side"] == 1

    def test_survival_floor_is_the_documented_value(self):
        """The floor is an operating point, not an incidental constant."""
        assert 0.0 < SURVIVAL_FLOOR < 1.0

    def test_empty_evidence_does_not_trigger_the_guard(self):
        assert filtering_would_gut_evidence([], MIN_MARK_SIDE_PX) is False

    def test_points_without_geometry_do_not_trigger_the_guard(self):
        assert filtering_would_gut_evidence([{"source": "x"}], MIN_MARK_SIDE_PX) is False


class TestPolylineProvenance:
    def test_detects_declared_polyline_samples(self):
        points = [_point(10, 10, 20, 20, source="cv_colored_line_sample") for _ in range(3)]
        assert is_polyline_evidence(points) is True

    def test_mixed_evidence_below_majority_is_not_polyline(self):
        points = [_point(10, 10, 20, 20, source="cv_colored_line_sample")]
        points += [_bar(0), _bar(1), _bar(2)]
        assert is_polyline_evidence(points) is False

    def test_declared_polyline_evidence_is_left_unfiltered(self):
        points = [_point(10 + i, 10, 11 + i, 200, source="cv_colored_line_sample")
                  for i in range(4)]
        result = admit_marks(points)
        assert result["policy"] == "polyline_evidence_not_filtered"
        assert len(result["admitted"]) == 4

    def test_empty_input_is_not_polyline(self):
        assert is_polyline_evidence([]) is False


class TestApplyAdmission:
    def _record(self):
        return {
            "sample_id": "s1",
            "measured_points": [_bar(0), _bar(1), _bar(2), _sliver(0)],
        }

    def test_does_not_mutate_the_frozen_input(self):
        record = self._record()
        apply_admission(record)
        assert len(record["measured_points"]) == 4

    def test_filters_on_the_copy(self):
        filtered = apply_admission(self._record())
        assert len(filtered["measured_points"]) == 3

    def test_reports_what_was_dropped(self):
        """A silently smaller evidence set reads as 'nothing was removed'."""
        summary = apply_admission(self._record())["mark_admission"]
        assert summary["input_count"] == 4
        assert summary["admitted_count"] == 3
        assert summary["rejected_count"] == 1
        assert summary["rejected_counts"]["sliver_below_min_side"] == 1

    def test_empty_record_is_handled(self):
        filtered = apply_admission({"sample_id": "s", "measured_points": []})
        assert filtered["measured_points"] == []
        assert filtered["mark_admission"]["input_count"] == 0

    def test_never_reads_the_reference_answer(self):
        first = apply_admission({**self._record(), "reference": 1.0})
        second = apply_admission({**self._record(), "reference": 9999.0})
        assert first["measured_points"] == second["measured_points"]

    def test_threshold_is_configurable_for_sensitivity_analysis(self):
        record = {"sample_id": "s", "measured_points": [
            _point(100, 100, 107, 300), _point(200, 100, 207, 300),
            _point(300, 100, 307, 300), _point(400, 100, 407, 300)]}
        assert len(apply_admission(record, min_side_px=5.0)["measured_points"]) == 4
        # At a 10px threshold every component is a sliver, so the guard stands down
        # rather than returning empty evidence.
        stood_down = apply_admission(record, min_side_px=10.0)
        assert stood_down["mark_admission"]["policy"] == (
            "filter_stood_down_survival_below_floor")
