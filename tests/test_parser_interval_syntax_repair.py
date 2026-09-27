"""Tests for the interval/glyph JSON repairs in the reasoning parser.

Why these exist
---------------
The R3-3 attribution run produced a +13.4 point paired gap that was not
reasoning. 44 of 104 single-pass rows answered CORRECTLY -- coordinates
identical to the layered arm -- but wrote the monotonicity interval in
mathematical notation::

    "monotonicity": [{"interval": (-∞, 1], "direction": "decreasing"}]

Bare parentheses are not JSON and a literal U+221E is not a JSON number, so
the object failed to load and four dimensions scored zero. The failure was
arm-asymmetric: the tuple form appeared in 12 single-pass rows and in 0 rows
of either layered arm, so the parser limitation presented as a layering
effect -- a baseline-fairness defect in our own harness (invariant 6).

After the repair the same contrast reads +0.11 (p=0.94).

The repair must stay strictly syntactic. A repair that alters values would
manufacture the result it is meant to measure, so most of what follows pins
what the repair must NOT do.
"""

from __future__ import annotations

import json

from evaluator.reasoning.parser import (
    _loads_json_with_numeric_fraction_repair,
    parse_comprehensive_answer,
    parse_L2_answer,
)


class TestIntervalTupleRecovery:
    def test_open_left_interval_loads(self):
        loaded = _loads_json_with_numeric_fraction_repair(
            '{"monotonicity": [{"interval": (-5, 1], "direction": "decreasing"}]}')
        assert loaded is not None
        assert loaded["monotonicity"][0]["interval"] == [-5, 1]

    def test_open_right_interval_loads(self):
        loaded = _loads_json_with_numeric_fraction_repair(
            '{"monotonicity": [{"interval": [1, 5), "direction": "increasing"}]}')
        assert loaded is not None
        assert loaded["monotonicity"][0]["interval"] == [1, 5]

    def test_infinity_glyph_inside_an_interval_loads(self):
        # The infinity repair emits the JSON string "-Infinity" rather than a
        # bare literal, and `_parse_monotonicity` floats it. Asserting through
        # the step parser checks the value the metric actually sees.
        parsed = parse_L2_answer(
            '{"predictions": {}, "rate_of_change": 1, '
            '"monotonicity": [{"interval": (-∞, 1], "direction": "decreasing"}]}')
        assert parsed["monotonicity"][0]["interval"][0] == float("-inf")

    def test_the_real_failing_payload_from_the_run(self):
        # Verbatim from outputs/m3_attribution_pilot, sample
        # absolute-clean_render_0041, condition E6_flat_single.
        raw = (
            '```json\n{\n'
            '  "coordinates": "(-5,8), (-4,6), (-1,0), (3,0)",\n'
            '  "intercepts": {"x_intercepts": [-1, 3], "y_intercept": -2},\n'
            '  "extrema": [{"x": 1, "y": -4, "type": "min"}],\n'
            '  "predictions": {"-4.5": 7, "-3.5": 5, "-2.5": 3},\n'
            '  "rate_of_change": 0,\n'
            '  "monotonicity": [{"interval": (-∞, 1], "direction": "decreasing"}, '
            '{"interval": [1, ∞), "direction": "increasing"}],\n'
            '  "symmetry": "neither",\n'
            '  "expression": "2*abs(x-1) -4",\n'
            '  "domain": "(-∞, ∞)",\n'
            '  "range": "[-4, ∞)",\n'
            '  "inference_explanation": "V-shaped with vertex at (1, -4)."\n'
            '}\n```'
        )
        parsed = parse_comprehensive_answer(raw)
        assert parsed["coordinates"]
        assert parsed["predictions"] == {"-4.5": 7.0, "-3.5": 5.0, "-2.5": 3.0}
        assert parsed["rate_of_change"] == 0.0
        assert len(parsed["monotonicity"]) == 2

    def test_layer_two_answers_recover_the_same_way(self):
        parsed = parse_L2_answer(
            '{"predictions": {"1.5": 2}, "rate_of_change": 3, '
            '"monotonicity": [{"interval": (0, ∞), "direction": "increasing"}]}')
        assert parsed["monotonicity"]
        assert parsed["rate_of_change"] == 3.0


class TestRepairIsStrictlySyntactic:
    def test_valid_json_is_returned_unchanged(self):
        text = '{"rate_of_change": 0.25, "predictions": {"1.5": -2.75}}'
        assert _loads_json_with_numeric_fraction_repair(text) == json.loads(text)

    def test_numbers_are_not_altered(self):
        loaded = _loads_json_with_numeric_fraction_repair(
            '{"monotonicity": [{"interval": (-2.5, 7.25], "direction": "up"}]}')
        assert loaded["monotonicity"][0]["interval"] == [-2.5, 7.25]

    def test_an_infinity_glyph_inside_a_string_is_preserved(self):
        # `domain` is a STRING and its brackets carry meaning. Rewriting the
        # glyph there would corrupt the model's own answer text; downstream
        # normalisation already handles the glyph in string context.
        loaded = _loads_json_with_numeric_fraction_repair(
            '{"domain": "(-∞, ∞)", "monotonicity": [{"interval": (0, 1], "direction": "up"}]}')
        assert loaded["domain"] == "(-∞, ∞)"

    def test_a_parenthesised_expression_string_is_preserved(self):
        loaded = _loads_json_with_numeric_fraction_repair(
            '{"expression": "2*(x-1)^2 + 3", "monotonicity": [{"interval": (0, 1], "direction": "up"}]}')
        assert loaded["expression"] == "2*(x-1)^2 + 3"

    def test_prose_containing_an_interval_phrase_is_untouched(self):
        loaded = _loads_json_with_numeric_fraction_repair(
            '{"inference_explanation": "On the interval: (0, 5] it rises.", '
            '"monotonicity": [{"interval": (0, 5], "direction": "up"}]}')
        assert loaded["inference_explanation"] == "On the interval: (0, 5] it rises."

    def test_unicode_minus_outside_a_string_becomes_ascii(self):
        loaded = _loads_json_with_numeric_fraction_repair(
            '{"monotonicity": [{"interval": [−3, 1], "direction": "up"}]}')
        assert loaded["monotonicity"][0]["interval"] == [-3, 1]

    def test_genuinely_broken_json_still_returns_none(self):
        # The repair must not become a salvage-anything fallback: a truncated
        # object has its own recovery path with its own guarantees.
        assert _loads_json_with_numeric_fraction_repair('{"a": ') is None

    def test_prose_is_not_mistaken_for_json(self):
        assert _loads_json_with_numeric_fraction_repair(
            "The function increases on (0, 5].") is None


class TestPreExistingRepairsStillWork:
    """The new repairs run before the old ones; none may regress."""

    def test_fraction_repair(self):
        loaded = _loads_json_with_numeric_fraction_repair(
            '{"rate_of_change": 3/2}')
        assert loaded["rate_of_change"] == 1.5

    def test_bare_infinity_word_repair(self):
        parsed = parse_L2_answer(
            '{"predictions": {}, "rate_of_change": 1, '
            '"monotonicity": [{"interval": [-inf, 1], "direction": "up"}]}')
        assert parsed["monotonicity"][0]["interval"][0] == float("-inf")

    def test_latex_escape_repair(self):
        loaded = _loads_json_with_numeric_fraction_repair(
            r'{"expression": "\infty is not an escape"}')
        assert loaded is not None
