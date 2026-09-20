"""Regression tests for L3 expression extraction and logarithm parsing.

Every payload here is abridged from a real paid run in which a correct answer
scored near zero. Two independent defects produced that:

1. ``parse_L3_answer`` fell back to a positional ``(a)`` pattern that captured
   narration. The L3 prompt numbers its tasks ``(h)/(i)/(j)``, so a model's
   own list markers matched instead of an answer.
2. ``_normalize_expr_str`` turned ``\\log_{2}(u)`` into a symbol named
   ``log_2`` multiplied by ``(u)``, making the logarithmic ground truth itself
   a polynomial. Correct answers could then only match by copying the
   reference spelling.
"""

from evaluator.metrics.expression import parse_expression, symbolic_equivalence
from evaluator.metrics.reasoning import evaluate_expression
from evaluator.reasoning.parser import parse_L3_answer


class TestProseIsNotAnExpression:
    def test_list_marker_narration_is_rejected(self):
        """A leading "- The vertex is at (0, -1)" must not become the answer."""
        raw = (
            "Looking at the coordinates, I see a V-shaped pattern.\n"
            "- The vertex is at (0, -1)\n"
            "- The function has x-intercepts at x = -1 and x = 1\n"
            "Therefore, f(x) = |x| - 1\n"
        )
        assert parse_L3_answer(raw)["expression"] == "|x| - 1"

    def test_introductory_sentence_is_rejected(self):
        raw = (
            "The pattern shows that for any x-value, the y-value equals the "
            "absolute value of x minus 1. This can be written as:\n"
            "f(x) = |x| - 1\n"
        )
        assert parse_L3_answer(raw)["expression"] == "|x| - 1"

    def test_method_narration_without_answer_yields_empty(self):
        """Describing an approach is not answering.

        This previously scored 15.5 -- the words "absolute value" matched the
        family regex, buying credit for a string containing no answer.
        """
        raw = (
            "Using the vertex form of an absolute value function: "
            "f(x) = a|x - h| + k, where (h, k) is the vertex.\n"
            "The vertex is at (2, -4).\n"
        )
        assert parse_L3_answer(raw)["expression"] == ""

    def test_genuine_non_conclusion_stays_empty(self):
        """A model that never concluded must not have an answer invented."""
        raw = (
            "Let me try f(x) = 2^(x+5):\n"
            "- f(-5) = 2^0 = 1\n"
            "I'm still not getting the correct values. Let me try again.\n"
        )
        assert parse_L3_answer(raw)["expression"] == ""


class TestTruncatedJson:
    def test_truncated_json_answer_is_recovered(self):
        """Output cut off mid-object still carries a complete answer."""
        raw = (
            "For the range:\n- The minimum value is -1\n\n"
            '```json\n{\n  "expression": "|x| - 1",\n'
        )
        assert parse_L3_answer(raw)["expression"] == "|x| - 1"

    def test_well_formed_json_still_wins(self):
        raw = (
            "Some narration mentioning (a) a distractor line.\n"
            '```json\n{"expression": "x^2 - 4", "domain": "(-inf, +inf)"}\n```\n'
        )
        parsed = parse_L3_answer(raw)
        assert parsed["expression"] == "x^2 - 4"
        assert parsed["domain"] == "(-inf, +inf)"

    def test_partial_trailing_value_is_not_invented(self):
        """A key whose value was cut mid-write must not be half-captured."""
        raw = '```json\n{\n  "domain": "(-inf, +inf)",\n  "expression": "2^(x'
        parsed = parse_L3_answer(raw)
        assert parsed["domain"] == "(-inf, +inf)"
        assert parsed["expression"] == ""


class TestLabeledFieldsStillWork:
    def test_explicit_label_is_honoured(self):
        raw = "**Expression:** x^3 - 3x\n**Domain:** (-inf, +inf)\n"
        parsed = parse_L3_answer(raw)
        assert parsed["expression"] == "x^3 - 3x"

    def test_lettered_answer_at_line_start_is_honoured(self):
        """A real (a)-style answer line is still accepted."""
        raw = "(a) 2^(x+2) - 2\n(b) (-inf, +inf)\n"
        assert parse_L3_answer(raw)["expression"] == "2^(x+2) - 2"


class TestLogarithmBaseParsing:
    GT = "y = -3\\log_{2}(x + 2) + 4"

    def test_ground_truth_parses_as_a_logarithm(self):
        """The reference itself was previously a polynomial."""
        expr = parse_expression(self.GT.replace("y = ", ""))
        assert expr is not None
        assert "log" in str(expr)
        assert "log_2" not in str(expr)

    def test_equivalent_spellings_all_match(self):
        gt = parse_expression(self.GT.replace("y = ", ""))
        for candidate in (
            "-3*log(x + 2, 2) + 4",
            "-3*log2(x + 2) + 4",
            "-3log_2(x+2)+4",
            "-3*log_{2}(x + 2) + 4",
            "-3 * log(x+2)/log(2) + 4",
        ):
            parsed = parse_expression(candidate)
            assert parsed is not None, candidate
            assert symbolic_equivalence(parsed, gt), candidate

    def test_unicode_subscript_matches(self):
        gt = parse_expression(self.GT.replace("y = ", ""))
        parsed = parse_expression("-3*log₂(x + 2) + 4")
        assert parsed is not None
        assert symbolic_equivalence(parsed, gt)

    def test_correct_logarithm_scores_full_marks(self):
        result = evaluate_expression("-3*log(x + 2, 2) + 4", self.GT)
        assert result["correct"] is True
        assert result["score"] == 100.0

    def test_natural_log_base_e_is_not_mangled(self):
        expr = parse_expression("log_e(x)")
        assert expr is not None
        assert symbolic_equivalence(expr, parse_expression("log(x)"))

    def test_non_logarithmic_families_are_unaffected(self):
        for candidate, gt in (
            ("2^(x-1) - 4", "y = 2^{x - 1} - 4"),
            ("2*2^(x-2) - 3", "y = 2 \\cdot 2^{x - 2} - 3"),
            ("-8/(x+2) + 2", "y = \\frac{-8}{x + 2} + 2"),
            ("x^2 - 4", "y = x^{2} - 4"),
        ):
            result = evaluate_expression(candidate, gt)
            assert result["correct"] is True, candidate
            assert result["score"] == 100.0, candidate
