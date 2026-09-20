"""A year answer must match exactly, not within a relative tolerance.

Regression guard for a scoring defect on the frozen PlotQA rosters. The generic
numeric rule is ``tolerance = max(1.0, |ref| * 0.05)``, which at ``ref=2007``
admits anything within +/-100.35 years: predicting 1907 or 2107 scored correct.

Why this mattered beyond tidiness: the inflation was *not* uniform across arms.
Under the strict rule the frozen 500-sample artifacts lose 7 (E0), 9 (E1),
10 (G_measured_raw) and 14 (Measured_flat_answer_baseline) correct rows. An
error that shifts arms by different amounts distorts exactly the between-arm
comparison those artifacts exist to support.
"""

import pytest

from scripts.run_public_benchmark_pilot import is_year_answer, score_prediction

YEAR_Q = "In which year was the population minimum ?"
VALUE_Q = "What is the total number of hours required to prepare and pay taxes?"


class TestYearDetection:
    @pytest.mark.parametrize("question", [
        "In which year was the population minimum ?",
        "which year had the highest debt",
        "During which year was the total reserves maximum?",
        "IN WHICH YEAR was it minimum?",
    ])
    def test_year_questions_are_detected(self, question):
        assert is_year_answer(question, 2007) is True

    def test_both_signals_are_required(self):
        """Either signal alone must not trigger the strict rule."""
        # Year-shaped reference, but the question asks for a magnitude.
        assert is_year_answer(VALUE_Q, 2007) is False
        # Year question, but the reference is not a year-range integer.
        assert is_year_answer(YEAR_Q, 45.7) is False
        assert is_year_answer(YEAR_Q, 1200) is False

    def test_non_integral_year_range_value_is_a_magnitude(self):
        """2007.5 cannot be a year label."""
        assert is_year_answer(YEAR_Q, 2007.5) is False

    @pytest.mark.parametrize("reference", [None, "", "not-a-number"])
    def test_unparseable_reference_is_not_a_year(self, reference):
        assert is_year_answer(YEAR_Q, reference) is False

    def test_missing_question_cannot_trigger_the_rule(self):
        """The default must fall back to the generic rule, never guess."""
        assert is_year_answer(None, 2007) is False


class TestStrictScoring:
    def test_adjacent_year_is_wrong(self):
        """The defect: 2006 and 2008 both scored correct against 2007."""
        for pred in (2006, 2008):
            result = score_prediction(pred, 2007, "numeric", question=YEAR_Q)
            assert result["correct"] is False, pred

    def test_the_recorded_real_case_is_now_wrong(self):
        """Predicted 1995 against reference 1998 previously scored correct."""
        result = score_prediction(1995, 1998, "numeric", question=YEAR_Q)
        assert result["correct"] is False

    def test_exact_year_is_correct(self):
        result = score_prediction(2007, 2007, "numeric", question=YEAR_Q)
        assert result["correct"] is True

    def test_tolerance_is_zero_and_flagged(self):
        """Recorded as 0.0, not None, so 'strict rule fired' stays legible."""
        result = score_prediction(2007, 2007, "numeric", question=YEAR_Q)
        assert result["tolerance"] == 0.0
        assert result["year_answer"] is True

    def test_numeric_error_is_still_reported(self):
        result = score_prediction(2004, 2007, "numeric", question=YEAR_Q)
        assert result["numeric_error"] == pytest.approx(3.0)


class TestGenericScoringIsUnchanged:
    """The repair must not move any non-year number."""

    def test_magnitude_still_uses_relative_tolerance(self):
        result = score_prediction(2050, 2007, "numeric", question=VALUE_Q)
        assert result["correct"] is True
        assert result["tolerance"] == pytest.approx(100.35)

    def test_absolute_floor_still_applies_to_small_values(self):
        result = score_prediction(0.6, 1.0, "numeric", question=VALUE_Q)
        assert result["correct"] is True
        assert result["tolerance"] == pytest.approx(1.0)

    def test_callers_that_pass_no_question_keep_prior_behaviour(self):
        """20 existing call sites omit `question`; their numbers must not move."""
        result = score_prediction(2006, 2007, "numeric")
        assert result["correct"] is True
        assert result["tolerance"] == pytest.approx(100.35)
        assert "year_answer" not in result

    def test_abstention_still_reported(self):
        result = score_prediction(None, 2007, "numeric", question=YEAR_Q)
        assert result["correct"] is False
        assert result["abstained"] is True

    def test_text_answers_unaffected(self):
        assert score_prediction("Brazil", "Brazil", "text", question=YEAR_Q)["correct"] is True
