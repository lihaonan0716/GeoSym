"""The extrema and intercept scores are RECALL, and must stay that way.

An audit found that neither dimension penalises an invented answer: a curve
with no extrema scores 100 whether the model reports none or three, and a
prediction listing every true extremum plus several invented ones also scores
100. That is a real property of the measure, not a bug to be silently patched
-- all 29425 historical paid rows were scored under it, so changing the basis
would make new numbers incomparable with them (CLAUDE.md invariant 13).

What these tests lock in is the disclosure, not a penalty:

- the score stays recall-only, so historical comparability holds;
- ``precision`` and ``surplus`` are always returned, so no report can quote the
  score as accuracy without the number that contradicts it being right there.

Measured on the 240-sample GLM-4.6V run: E2 precision 0.803 / recall 0.777,
E5_raw precision 0.615 / recall 0.979. A high extrema score means "missed
almost nothing", not "was right".
"""

import pytest

from evaluator.metrics.reasoning import evaluate_extrema, evaluate_intercepts


class TestExtremaIsRecallOnly:
    GT_ONE = [{"x": 2.0, "y": -4.0, "type": "min"}]

    def test_exact_answer_scores_full_marks(self):
        assert evaluate_extrema(list(self.GT_ONE), self.GT_ONE)["score"] == 100.0

    def test_surplus_does_not_reduce_the_score(self):
        """Historical basis: finding everything scores 100 even with extras."""
        result = evaluate_extrema(
            [{"x": 2.0, "y": -4.0, "type": "min"},
             {"x": 5.0, "y": 1.0, "type": "max"},
             {"x": -5.0, "y": 1.0, "type": "max"}],
            self.GT_ONE,
        )
        assert result["score"] == 100.0

    def test_but_the_surplus_is_reported(self):
        result = evaluate_extrema(
            [{"x": 2.0, "y": -4.0, "type": "min"},
             {"x": 5.0, "y": 1.0, "type": "max"},
             {"x": -5.0, "y": 1.0, "type": "max"}],
            self.GT_ONE,
        )
        assert result["surplus"] == 2
        assert result["precision"] == pytest.approx(1 / 3, abs=1e-3)
        assert result["scoring_basis"] == "recall_only"
        assert "surplus" in result["details"]

    def test_invented_extrema_on_a_monotone_curve_score_100_but_precision_0(self):
        clean = evaluate_extrema([], [])
        assert clean["score"] == 100.0
        assert clean["surplus"] == 0

        invented = evaluate_extrema(
            [{"x": 0.0, "y": 0.0, "type": "min"},
             {"x": 1.0, "y": 1.0, "type": "max"}], [])
        assert invented["score"] == 100.0        # recall-only basis
        assert invented["surplus"] == 2          # but it is visible
        assert invented["precision"] == 0.0
        assert "not penalised" in invented["details"]

    def test_missing_extrema_still_scores_zero(self):
        result = evaluate_extrema([], self.GT_ONE)
        assert result["score"] == 0.0
        assert result["precision"] is None

    def test_position_and_type_scoring_is_preserved(self):
        wrong_type = evaluate_extrema(
            [{"x": 2.0, "y": -4.0, "type": "max"}], self.GT_ONE)
        assert 0.0 < wrong_type["score"] < 100.0
        far_off = evaluate_extrema(
            [{"x": -8.0, "y": 9.0, "type": "min"}], self.GT_ONE)
        assert far_off["score"] < wrong_type["score"]

    def test_two_true_extrema_matched_exactly(self):
        gt = [{"x": -2.0, "y": 4.0, "type": "max"},
              {"x": 2.0, "y": -4.0, "type": "min"}]
        result = evaluate_extrema(list(gt), gt)
        assert result["score"] == 100.0
        assert result["surplus"] == 0
        assert result["precision"] == pytest.approx(1.0)

    def test_every_result_carries_the_basis_label(self):
        """No caller should be able to read a score without the caveat."""
        for pred, gt in (
            ([], []),
            ([{"x": 0.0, "y": 0.0, "type": "min"}], []),
            ([], [{"x": 1.0, "y": 1.0, "type": "max"}]),
            ([{"x": 1.0, "y": 1.0, "type": "max"}], [{"x": 1.0, "y": 1.0, "type": "max"}]),
        ):
            assert evaluate_extrema(pred, gt)["scoring_basis"] == "recall_only"


class TestInterceptsIsRecallOnly:
    TRUTH_NO_X = {"x_intercepts": [], "y_intercept": 5.0}

    def test_correct_answer_scores_full_marks(self):
        result = evaluate_intercepts(
            {"x_intercepts": [], "y_intercept": 5.0}, self.TRUTH_NO_X)
        assert result["score"] == 100.0
        assert result["x_intercepts_surplus"] == 0

    def test_invented_x_intercepts_do_not_reduce_the_score(self):
        result = evaluate_intercepts(
            {"x_intercepts": [-3.0, 2.0], "y_intercept": 5.0}, self.TRUTH_NO_X)
        assert result["score"] == 100.0

    def test_but_the_surplus_is_reported(self):
        result = evaluate_intercepts(
            {"x_intercepts": [-3.0, 2.0], "y_intercept": 5.0}, self.TRUTH_NO_X)
        assert result["x_intercepts_surplus"] == 2
        assert result["scoring_basis"] == "recall_only"
        assert "x_surplus=2" in result["details"]

    def test_y_intercept_hallucination_is_still_penalised(self):
        """This asymmetry is pre-existing and deliberately left intact."""
        result = evaluate_intercepts(
            {"x_intercepts": [1.0], "y_intercept": -7.0},
            {"x_intercepts": [1.0], "y_intercept": None},
        )
        assert result["score"] < 100.0

    def test_real_x_intercepts_are_scored_as_before(self):
        truth = {"x_intercepts": [-2.0, 2.0], "y_intercept": -4.0}
        exact = evaluate_intercepts(
            {"x_intercepts": [-2.0, 2.0], "y_intercept": -4.0}, truth)
        assert exact["score"] == 100.0
        missing = evaluate_intercepts({"x_intercepts": [], "y_intercept": -4.0}, truth)
        assert missing["score"] < exact["score"]

    def test_absent_intercepts_answer_is_not_a_surplus(self):
        result = evaluate_intercepts(
            {"x_intercepts": [], "y_intercept": None},
            {"x_intercepts": [], "y_intercept": None},
        )
        assert result["score"] == 100.0
        assert result["x_intercepts_surplus"] == 0
