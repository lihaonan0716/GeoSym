"""Regression tests for piecewise expression parsing.

Every spelling here appeared in real model output. Before this was handled,
SymPy returned None for all of them and the scorer awarded 0 to answers that
were mathematically sound -- and it did so unevenly: arms given precise
coordinates describe a curve's flat tail as a literal constant piece roughly
three times more often than the image-only baselines, so the scorer's grammar
coverage was being read as a difference between methods.

The benchmark's own piecewise ground truth is written as LaTeX ``\\begin{cases}``,
which was also unparseable, so that family scored near zero for every condition.
"""

import pytest

from evaluator.metrics.expression import parse_expression, symbolic_equivalence
from evaluator.metrics.reasoning import evaluate_expression


def _eq(a: str, b: str) -> bool:
    pa, pb = parse_expression(a), parse_expression(b)
    assert pa is not None, f"failed to parse: {a!r}"
    assert pb is not None, f"failed to parse: {b!r}"
    return bool(symbolic_equivalence(pa, pb))


class TestRealSpellings:
    """Each case is abridged from a stored run."""

    @pytest.mark.parametrize("text", [
        "f(x) = {x + 8 for x <= -4, -x for x > -4}",
        "f(x) = x + 8 if x <= -4, f(x) = -x if x >= -4",
        "f(x) = x + 8 for x <= -4; f(x) = -x for x >= -4",
        "x if x <= 2 else -x + 4",
        "if(x <= -2, -4, 0.5*2^x - 4)",
        "piecewise function: 0 for x <= -6, 2^(x+3) for x > -6",
        "f(x) = { -1, if x ≤ -6; 2^(x+1) - 2, if x > -6 }",
        "f(x) = -2, for x ≤ -4; f(x) = 2^(x+2) - 2, for x > -4",
        "y = -3 if x <= -4 else 4*2^x - 3",
    ])
    def test_spelling_parses(self, text):
        assert parse_expression(text) is not None

    def test_unicode_comparators_are_accepted(self):
        assert _eq(
            "f(x) = -2, for x ≤ -4; f(x) = 2^(x+2) - 2, for x > -4",
            "f(x) = -2 for x <= -4; f(x) = 2^(x+2) - 2 for x > -4",
        )

    def test_three_branch_chained_bound(self):
        """A chained bound like "-3 < x <= 0" must not be dropped."""
        expr = parse_expression(
            "f(x) = -2 if x <= -3; f(x) = (1/6)x^2 + (7/6)x if -3 < x <= 0; "
            "f(x) = x^2 + x if x > 0"
        )
        assert expr is not None
        assert expr.subs("x", -5) == -2
        assert expr.subs("x", 2) == 6

    def test_spellings_of_one_function_agree(self):
        """The same function written four ways must compare equal."""
        canonical = "f(x) = x + 8 for x <= -4; f(x) = -x for x > -4"
        for variant in (
            "f(x) = {x + 8 for x <= -4, -x for x > -4}",
            "x + 8 if x <= -4 else -x",
            "piecewise: x + 8 for x <= -4, -x for x > -4",
        ):
            assert _eq(variant, canonical), variant


class TestLatexCasesGroundTruth:
    GT = ("y = \\begin{cases} -x - 1 & x < 3 \\\\\\ -4x + 8 & x \\geq 3 "
          "\\end{cases}")

    def test_benchmark_ground_truth_parses(self):
        assert parse_expression(self.GT) is not None

    def test_correct_answer_matches_the_latex_reference(self):
        assert _eq("f(x) = -x - 1 for x < 3; f(x) = -4x + 8 for x >= 3", self.GT)

    def test_correct_answer_scores_full_marks(self):
        result = evaluate_expression(
            "f(x) = -x - 1 for x < 3; f(x) = -4x + 8 for x >= 3", self.GT,
        )
        assert result["correct"] is True
        assert result["score"] == 100.0

    def test_wrong_piecewise_answer_does_not_score_full_marks(self):
        """The repair must not make every piecewise answer correct."""
        result = evaluate_expression(
            "f(x) = 5x + 2 for x < 3; f(x) = -x for x >= 3", self.GT,
        )
        assert result["correct"] is False
        assert result["score"] < 100.0


class TestNonPiecewiseUnaffected:
    @pytest.mark.parametrize("candidate,reference", [
        ("x^2 - 4", "y = x^{2} - 4"),
        ("2^(x-1) - 4", "y = 2^{x - 1} - 4"),
        ("-3*log(x + 2, 2) + 4", "y = -3\\log_{2}(x + 2) + 4"),
        ("-8/(x+2) + 2", "y = \\frac{-8}{x + 2} + 2"),
        ("abs(x + 4) + 3", "y = |x + 4| + 3"),
    ])
    def test_plain_expressions_still_score_full_marks(self, candidate, reference):
        result = evaluate_expression(candidate, reference)
        assert result["correct"] is True, candidate
        assert result["score"] == 100.0, candidate

    def test_a_bare_conditional_word_does_not_trigger_piecewise(self):
        """"for" inside prose must not be read as a piecewise clause."""
        assert parse_expression("x^2 + 2*x + 1") is not None

    def test_unparseable_input_still_returns_none(self):
        assert parse_expression("the vertex is at (0, -1)") is None
        assert parse_expression("") is None


class TestNoSilentFallback:
    def test_malformed_piecewise_returns_none_not_a_guess(self):
        """A clause with an unreadable condition must fail, not be invented."""
        assert parse_expression("f(x) = 2 for x <= ; f(x) = 3 for x > 1") is None

    def test_missing_condition_returns_none(self):
        assert parse_expression("f(x) = 2 for banana; f(x) = 3 for x > 1") is None
