import inspect

from scripts.analyze_m1_tier_discriminative_validity import (
    intercept_block_correct,
    monotonicity_segment_correct,
    symmetry_correct,
)


def test_truth_functions_do_not_accept_tier_parameter():
    for fn in (intercept_block_correct, symmetry_correct, monotonicity_segment_correct):
        assert "tier" not in inspect.signature(fn).parameters


def test_intercept_block_correct_requires_whole_block_match():
    gt = {
        "question": {
            "ground_truth": {
                "intercepts": {"x_intercepts": [1.0], "y_intercept": 2.0},
                "zeros": [1.0],
            }
        }
    }

    correct, details = intercept_block_correct({"x_intercepts": [1.0], "y_intercept": 2.0}, gt)
    assert correct is True
    assert details["x_correct"] is True
    assert details["y_correct"] is True

    correct, details = intercept_block_correct({"x_intercepts": [1.0], "y_intercept": "undetermined"}, gt)
    assert correct is False
    assert details["x_correct"] is True
    assert details["y_correct"] is False


def test_symmetry_correct_compares_ground_truth_label_only():
    gt = {"question": {"ground_truth": {"symmetry": "even"}}}

    assert symmetry_correct("even", gt)[0] is True
    assert symmetry_correct("odd", gt)[0] is False
