"""Tests for M1'/M2' evidence reliability scoring and selection (W4).

The predecessor module was falsified at AUC 0.454 because it scored evidence
whose values were systematically wrong. This one scores post-W1/W2 evidence and
reaches 0.690 on a held-out fold. The tests pin the properties that make that
result meaningful rather than incidental: answer independence, monotone tiers, a
gate that withholds rather than injects when evidence is thin, and ranking that
puts answerable facts inside the budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.reasoning.plotqa_evidence_reliability import (  # noqa: E402
    RELIABILITY_TIERS,
    score_evidence,
    select_evidence,
    usable_values,
)


def _point(value=None, label=None, key="measured_value"):
    point = {}
    if value is not None:
        point[key] = value
    if label is not None:
        point["series_label"] = label
    return point


def _record(points, admissible=True, **refit):
    return {
        "measured_points": points,
        "ruler_refit": {"admissible": admissible, **refit},
        "mark_admission": {"rejected_count": 0, "input_count": len(points)},
    }


class TestUsableValues:
    def test_reads_the_orientation_independent_field(self):
        assert usable_values([_point(1.0), _point(2.0)]) == [1.0, 2.0]

    def test_falls_back_to_y_value_for_pre_w1_records(self):
        assert usable_values([_point(3.0, key="y_value")]) == [3.0]

    def test_excludes_unmeasured_points(self):
        """An inadmissible axis writes None; that point was never measured."""
        points = [_point(1.0), {"measured_value": None}, {}]
        assert usable_values(points) == [1.0]

    def test_excludes_booleans(self):
        assert usable_values([{"measured_value": True}]) == []


class TestScoreEvidence:
    def test_plentiful_evidence_on_a_fitted_axis_scores_high(self):
        record = _record([_point(float(i)) for i in range(12)])
        assert score_evidence(record)["tier"] == "high"

    def test_an_inadmissible_axis_scores_low_regardless_of_count(self):
        """Without a fitted axis the values are not measurements at all."""
        record = _record([_point(float(i)) for i in range(50)], admissible=False)
        score = score_evidence(record)
        assert score["composite"] == 0.0
        assert score["tier"] == "low"

    def test_a_fitted_axis_with_almost_no_evidence_scores_low(self):
        assert score_evidence(_record([_point(1.0)]))["tier"] == "low"

    def test_score_is_monotone_in_usable_count(self):
        few = score_evidence(_record([_point(float(i)) for i in range(3)]))
        many = score_evidence(_record([_point(float(i)) for i in range(20)]))
        assert many["composite"] > few["composite"]

    def test_tier_is_one_of_the_declared_ordinal_levels(self):
        assert score_evidence(_record([_point(1.0)]))["tier"] in RELIABILITY_TIERS

    def test_does_not_claim_calibration(self):
        """Invariant: M1 tiers are ordinal heuristics, never probabilities."""
        policy = score_evidence(_record([_point(1.0)]))["scoring_policy"]
        assert "uncalibrated" in policy
        assert "calibrated" not in policy.replace("uncalibrated", "")

    def test_reports_its_supporting_signals(self):
        record = _record([_point(1.0), _point(2.0)], tick_count=5)
        signals = score_evidence(record)["signals"]
        assert signals["axis_admissible"] is True
        assert signals["usable_value_count"] == 2
        assert signals["tick_count"] == 5

    def test_empty_evidence_does_not_crash(self):
        score = score_evidence({"measured_points": []})
        assert score["tier"] == "low"
        assert score["signals"]["usable_fraction_of_admitted"] == 0.0

    def test_never_reads_the_reference_answer(self):
        points = [_point(float(i)) for i in range(8)]
        first = score_evidence({**_record(points), "reference": 1.0})
        second = score_evidence({**_record(points), "reference": 98765.0})
        assert first == second


class TestSelectEvidence:
    def test_withholds_low_reliability_evidence(self):
        """Injecting numbers that look measured but are not is actively harmful."""
        result = select_evidence(_record([_point(1.0)], admissible=False))
        assert result["inject"] is False
        assert result["reason"] == "withheld_low_reliability_evidence"
        assert result["selected_facts"] == []

    def test_injects_reliable_evidence(self):
        result = select_evidence(_record([_point(float(i)) for i in range(12)]))
        assert result["inject"] is True
        assert result["selected_facts"]

    def test_ranks_valued_and_labelled_facts_first(self):
        points = [
            _point(),                       # neither
            _point(1.0),                    # value only
            _point(2.0, label="Brazil"),    # value and label
        ] + [_point(float(i)) for i in range(10)]
        selected = select_evidence(_record(points))["selected_facts"]
        assert selected[0].get("series_label") == "Brazil"

    def test_truncates_to_the_budget(self):
        points = [_point(float(i)) for i in range(50)]
        result = select_evidence(_record(points), max_facts=20)
        assert len(result["selected_facts"]) == 20
        assert result["input_fact_count"] == 50

    def test_truncation_keeps_the_most_answerable_facts(self):
        """The budget must be spent on facts a question can be matched against."""
        points = [_point(float(i)) for i in range(30)]
        points.append(_point(99.0, label="Tonga"))
        selected = select_evidence(_record(points), max_facts=5)["selected_facts"]
        assert any(p.get("series_label") == "Tonga" for p in selected)

    def test_reports_the_reliability_that_drove_the_decision(self):
        result = select_evidence(_record([_point(float(i)) for i in range(12)]))
        assert result["reliability"]["tier"] == "high"

    def test_selection_never_inspects_a_question(self):
        """Ranking is answer- and question-independent, so it cannot leak.

        A question-conditioned ranking would be a different mechanism with a
        different failure mode; this one must stay stable across questions.
        """
        points = [_point(float(i), label=f"s{i}") for i in range(12)]
        first = select_evidence({**_record(points), "question": "what is X in 2013"})
        second = select_evidence({**_record(points), "question": "which country is max"})
        assert first["selected_facts"] == second["selected_facts"]

    def test_never_reads_the_reference_answer(self):
        points = [_point(float(i)) for i in range(12)]
        first = select_evidence({**_record(points), "reference": 1.0})
        second = select_evidence({**_record(points), "reference": 4242.0})
        assert first["selected_facts"] == second["selected_facts"]
