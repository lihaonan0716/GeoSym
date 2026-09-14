from evaluator.metrics.reasoning import evaluate_domain
from evaluator.reasoning.parser import normalize_interval_str


def test_interval_union_spellings_score_equivalently():
    gt = r"(-\infty,\;-2.0) \cup (-2.0,\;+\infty)"
    variants = [
        "(-inf, -2) U (-2, +inf)",
        r"(-inf, -2) \cup (-2, +inf)",
        "(-inf, -2) ∪ (-2, +inf)",
        "(-inf, -2) union (-2, +inf)",
        "(-inf,-2)u(-2,+inf)",
    ]

    for pred in variants:
        result = evaluate_domain(pred, gt)
        assert result["score"] == 100.0
        assert result["correct"] is True


def test_interval_normalizer_does_not_rewrite_u_inside_words():
    assert "function" in normalize_interval_str("function")
    assert "value" in normalize_interval_str("value")
