from evaluator.reasoning.parser import parse_L2_answer, parse_comprehensive_answer, parse_judge_response


def test_l2_parser_accepts_bare_fraction_rate_of_change() -> None:
    parsed = parse_L2_answer(
        """
        {
          "predictions": {"-5.5": 4.5, "-4.5": 3.5},
          "rate_of_change": 1/7,
          "monotonicity": [
            {"interval": [-Infinity, 0], "direction": "decreasing"},
            {"interval": [0, Infinity], "direction": "increasing"}
          ],
          "symmetry": "even"
        }
        """
    )

    assert parsed["predictions"] == {"-5.5": 4.5, "-4.5": 3.5}
    assert parsed["rate_of_change"] == 1 / 7
    assert parsed["monotonicity"] == [
        {"interval": [float("-inf"), 0.0], "direction": "decreasing"},
        {"interval": [0.0, float("inf")], "direction": "increasing"},
    ]
    assert parsed["symmetry"] == "even"


def test_l2_parser_accepts_fenced_negative_fraction_rate_of_change() -> None:
    parsed = parse_L2_answer(
        """```json
        {
          "predictions": {"-5.5": 3.5},
          "rate_of_change": -1/7,
          "monotonicity": [{"interval": [-Infinity, 2], "direction": "decreasing"}],
          "symmetry": "neither"
        }
        ```"""
    )

    assert parsed["predictions"] == {"-5.5": 3.5}
    assert parsed["rate_of_change"] == -1 / 7
    assert parsed["monotonicity"] == [
        {"interval": [float("-inf"), 2.0], "direction": "decreasing"}
    ]
    assert parsed["symmetry"] == "neither"


def test_comprehensive_parser_accepts_bare_inf_tokens_without_labeled_fallback() -> None:
    parsed = parse_comprehensive_answer(
        r"""
        {
          "coordinates": "(-4,6), (-3,3), (-2,0), (-1,1), (0,2), (1,3), (2,4), (3,5), (4,6), (5,7), (6,8)",
          "intercepts": {"x_intercepts": [-2], "y_intercept": 2},
          "extrema": [{"x": -2, "y": 0, "type": "min"}],
          "predictions": {"-3.5": 4.5, "-2.5": 1.5, "-1.5": 0.5},
          "rate_of_change": 0.2,
          "monotonicity": [
            {"interval": [-inf, -2], "direction": "decreasing"},
            {"interval": [-2, +inf], "direction": "increasing"}
          ],
          "symmetry": "neither",
          "expression": "f(x) = -3x - 6 for x < -2; f(x) = x + 2 for x >= -2",
          "domain": "(-inf, +inf)",
          "range": "[0, +inf)",
          "inference_explanation": "Both line segments continue without endpoints or restrictions, so the mathematical domain is all real numbers. The vertex is the global minimum, making the range [0,+inf)."
        }
        """
    )

    assert parsed["expression"] == "f(x) = -3x - 6 for x < -2; f(x) = x + 2 for x >= -2"
    assert parsed["domain"] == "(-inf, +inf)"
    assert parsed["range"] == "[0, +inf)"
    assert parsed["monotonicity"] == [
        {"interval": [float("-inf"), -2.0], "direction": "decreasing"},
        {"interval": [-2.0, float("inf")], "direction": "increasing"},
    ]


def test_judge_parser_accepts_latex_backslash_in_json_comments() -> None:
    parsed = parse_judge_response(
        r"""```json
        {
          "factual_accuracy": 4,
          "inference_quality": 5,
          "reasoning_coherence": 5,
          "evidence_support": 5,
          "comments": "Because the left branch has negative slope, as $x \to -\infty$, $y \to +\infty$."
        }
        ```"""
    )

    assert parsed["factual_accuracy"] == 4.0
    assert parsed["inference_quality"] == 5.0
    assert parsed["reasoning_coherence"] == 5.0
    assert parsed["evidence_support"] == 5.0
    assert r"\infty" in parsed["comments"]


def test_comprehensive_parser_accepts_monotonicity_interval_strings() -> None:
    parsed = parse_comprehensive_answer(
        """
        {
          "monotonicity": [
            {"interval": "(-inf, -2)", "direction": "decreasing"},
            {"interval": "(-2, +inf)", "direction": "increasing"}
          ]
        }
        """
    )

    assert parsed["monotonicity"] == [
        {"interval": [float("-inf"), -2.0], "direction": "decreasing"},
        {"interval": [-2.0, float("inf")], "direction": "increasing"},
    ]
