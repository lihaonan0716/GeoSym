from evaluator.reasoning.parser import parse_L2_answer


def test_l2_parser_plain_json_behavior_unchanged() -> None:
    parsed = parse_L2_answer(
        '{"predictions": {"1": 2}, "rate_of_change": 1.25, '
        '"monotonicity": [{"interval": [-1, 1], "direction": "increasing"}], '
        '"symmetry": "neither"}'
    )

    assert parsed["predictions"] == {"1": 2.0}
    assert parsed["rate_of_change"] == 1.25
    assert parsed["monotonicity"] == [{"interval": [-1.0, 1.0], "direction": "increasing"}]
    assert parsed["symmetry"] == "neither"


def test_l2_parser_fenced_json_behavior_unchanged() -> None:
    parsed = parse_L2_answer(
        """```json
        {"predictions": {"0": 4}, "rate_of_change": 2,
         "monotonicity": [{"interval": [0, 3], "direction": "decreasing"}],
         "symmetry": "even"}
        ```"""
    )

    assert parsed["predictions"] == {"0": 4.0}
    assert parsed["rate_of_change"] == 2.0
    assert parsed["monotonicity"] == [{"interval": [0.0, 3.0], "direction": "decreasing"}]
    assert parsed["symmetry"] == "even"


def test_l2_parser_preserves_date_like_string() -> None:
    parsed = parse_L2_answer(
        '{"predictions": {"2026/09/13": 7}, "rate_of_change": 0, '
        '"monotonicity": [], "symmetry": "none"}'
    )

    assert parsed["predictions"] == {"2026/09/13": 7.0}
    assert parsed["rate_of_change"] == 0.0


def test_l2_parser_preserves_string_with_slash() -> None:
    parsed = parse_L2_answer(
        '{"predictions": {"1": "not/a/number"}, "rate_of_change": 0, '
        '"monotonicity": [], "symmetry": "none"}'
    )

    assert parsed["predictions"] == {}
    assert parsed["rate_of_change"] == 0.0


def test_l2_parser_repairs_bare_positive_fraction() -> None:
    parsed = parse_L2_answer('{"predictions": {}, "rate_of_change": 3/2, "monotonicity": []}')

    assert parsed["rate_of_change"] == 1.5


def test_l2_parser_repairs_bare_negative_fraction() -> None:
    parsed = parse_L2_answer('{"predictions": {}, "rate_of_change": -3/4, "monotonicity": []}')

    assert parsed["rate_of_change"] == -0.75


def test_l2_parser_accepts_bare_infinity_tokens() -> None:
    parsed = parse_L2_answer(
        '{"predictions": {}, "rate_of_change": 1, '
        '"monotonicity": [{"interval": [-Infinity, Infinity], "direction": "increasing"}]}'
    )

    assert parsed["monotonicity"] == [
        {"interval": [float("-inf"), float("inf")], "direction": "increasing"}
    ]


def test_l2_parser_does_not_repair_division_by_zero() -> None:
    parsed = parse_L2_answer('{"predictions": {}, "rate_of_change": 1/0, "monotonicity": []}')

    assert parsed["rate_of_change"] is None
