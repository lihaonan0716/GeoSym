from evaluator.reasoning.parser import parse_L2_answer


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
