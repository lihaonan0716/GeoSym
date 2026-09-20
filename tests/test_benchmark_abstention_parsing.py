"""Regression tests: a declined answer must never become a number.

Found by the GLM PlotQA pilot on 2026-09-20. A model answered "Cannot be
determined (Sudan's export data for 1988 and 1990 is not available in the
provided measured facts)" -- a correct refusal, since the evidence genuinely
lacked Sudan. `normalize_answer` scans for the first numeric token anywhere in
the string, so the refusal was recorded as the prediction **1988.0** and scored
as a confident wrong answer.

A second response, `"normalized_prediction": "unmeasurable"`, was worse: the
field raised, the exception was swallowed, and the fallback scanned the whole
JSON blob -- picking the `2` out of "CO2" in the echoed question and recording
**2.0**.

Why this mattered beyond tidiness: only an evidence-bearing arm can notice that
the evidence lacks the entity and decline. The image-only baseline always
guesses. So a parser that converts refusals into wrong numbers penalises
exactly the behaviour the evidence pipeline is supposed to produce, while the
code path is identical for both arms (invariant 6).

The most important tests here are the negative ones: the fix must not change a
single answer that previously parsed correctly, or it silently rewrites history
for every cached row.
"""

from __future__ import annotations

import json

import pytest

from scripts.run_public_benchmark_pilot import (
    extract_prediction,
    is_abstention,
    score_prediction,
)


def _response(prediction, raw_text: str = "some question text") -> str:
    return json.dumps({"raw_text": raw_text, "normalized_prediction": prediction})


class TestTheTwoObservedFailures:
    """Verbatim payloads from the pilot's paid rows."""

    def test_cannot_be_determined_does_not_become_the_year(self):
        raw = _response(
            "Cannot be determined (Sudan's export data for 1988 and 1990 is not "
            "available in the provided measured facts)",
            "What is the ratio of the total exports of goods and services in Sudan "
            "in 1988 to that in 1990?",
        )
        assert extract_prediction(raw, "numeric") is None

    def test_unmeasurable_does_not_become_a_digit_from_the_question(self):
        raw = _response(
            "unmeasurable", "What is the amount of CO2 emitted in 1960 in Poland ?"
        )
        assert extract_prediction(raw, "numeric") is None


class TestRealAnswersAreUnchanged:
    """The fix must move no answer that already parsed correctly."""

    @pytest.mark.parametrize(
        "prediction,expected",
        [
            ("0", 0.0),
            ("0.0", 0.0),
            ("-3.5", -3.5),
            ("1988", 1988.0),
            ("1.484809948954022", 1.484809948954022),
            ("2.000e+8", 2.000e8),
            ("1,234.5", 1234.5),
            (0, 0.0),
            (19.43, 19.43),
        ],
    )
    def test_numeric_answers_parse_as_before(self, prediction, expected):
        assert extract_prediction(_response(prediction), "numeric") == expected

    def test_zero_is_an_answer_not_an_abstention(self):
        """The pilot's third regression genuinely answered 0; it must stay 0."""
        raw = _response(
            "0",
            "The measured facts indicate all values are 71.29, so the difference "
            "between highest and lowest is 0.",
        )
        assert extract_prediction(raw, "numeric") == 0.0
        assert is_abstention("0") is False

    def test_a_number_with_hedging_prose_still_parses(self):
        """Uncertainty on the way to an answer is not a refusal."""
        assert is_abstention("about 7.5, though the label is unclear") is False

    def test_categorical_answers_are_untouched(self):
        assert extract_prediction(_response("Brazil"), "categorical") == "brazil"

    def test_answer_tag_still_wins_over_surrounding_prose(self):
        raw = "I could not read the axis at first. <answer>42</answer>"
        assert extract_prediction(raw, "numeric") == 42.0

    def test_final_answer_marker_still_parses(self):
        assert extract_prediction("Working... Final answer: 3.14", "numeric") == 3.14


class TestAbstentionDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "Cannot be determined",
            "cannot determine the value",
            "unmeasurable",
            "not measurable from the facts",
            "N/A",
            "unknown",
            "insufficient data to answer",
            "The value is not available in the evidence",
            "",
            "   ",
        ],
    )
    def test_refusals_are_detected(self, text):
        assert is_abstention(text) is True

    def test_none_is_an_abstention(self):
        assert is_abstention(None) is True

    @pytest.mark.parametrize("text", ["0", "-1", "42", "3.14", "2e8"])
    def test_bare_numbers_are_never_abstentions(self, text):
        assert is_abstention(text) is False


class TestScoringOfDeclinedAnswers:
    def test_none_scores_incorrect_without_raising(self):
        score = score_prediction(None, 5.0, "numeric")
        assert score["correct"] is False
        assert score["abstained"] is True

    def test_declining_is_distinguishable_from_answering_wrongly(self):
        """Both are incorrect, but only one is the model's own judgement."""
        declined = score_prediction(None, 5.0, "numeric")
        wrong = score_prediction(1988.0, 5.0, "numeric")
        assert declined["correct"] is wrong["correct"] is False
        assert declined.get("abstained") is True
        assert wrong.get("abstained") is None

    def test_a_correct_answer_is_still_correct(self):
        assert score_prediction(5.0, 5.0, "numeric")["correct"] is True


class TestSymmetryAcrossArms:
    """One code path serves every condition; nothing keys on the arm."""

    def test_identical_text_parses_identically_regardless_of_arm(self):
        raw = _response("Cannot be determined")
        # There is no arm parameter to pass -- that is the property. If a future
        # change adds one, this test's premise breaks and should be revisited.
        assert extract_prediction(raw, "numeric") is None
        assert extract_prediction(raw, "numeric") is None

    def test_baseline_style_guess_is_unaffected(self):
        """The image-only arm guesses a number; it must parse as that number."""
        assert extract_prediction(_response("0.87"), "numeric") == 0.87
