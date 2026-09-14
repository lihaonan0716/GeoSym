from evaluator.metrics.reasoning import evaluate_comprehensive, evaluate_intercepts


def test_absent_y_intercept_rewards_absent_answer_with_matching_x_intercepts():
    result = evaluate_intercepts(
        {"x_intercepts": [-5, -4, -3], "y_intercept": None},
        {"x_intercepts": [-5, -4, -3], "y_intercept": None},
    )

    assert result["score"] == 100.0
    assert result["correct"] is True
    assert "yi=100.0" in result["details"]


def test_absent_y_intercept_penalizes_fabricated_numeric_answer():
    result = evaluate_intercepts(
        {"x_intercepts": [-5, -4, -3], "y_intercept": -60.0},
        {"x_intercepts": [-5, -4, -3], "y_intercept": None},
    )

    assert result["score"] == 50.0
    assert result["correct"] is False
    assert "xi=100.0 yi=0.0" in result["details"]


def test_absent_y_intercept_accepts_undetermined_answer():
    result = evaluate_intercepts(
        {"x_intercepts": [], "y_intercept": "undetermined"},
        {"x_intercepts": [], "y_intercept": None},
    )

    assert result["score"] == 100.0
    assert result["correct"] is True


def test_comprehensive_scores_intercepts_against_visible_facts():
    pred = {"intercepts": {"x_intercepts": [-5, -4, -3], "y_intercept": None}}
    gt = {
        "intercepts": {"x_intercepts": [-5, -4, -3], "y_intercept": -60.0},
        "visible_facts": {
            "intercepts": {"x_intercepts": [-5, -4, -3], "y_intercept": None}
        },
    }

    result = evaluate_comprehensive(pred, gt)

    intercepts = result["sub_scores"]["intercepts"]
    assert intercepts["score"] == 100.0
    assert intercepts["gt_display"] == "x:[-5, -4, -3] y:None"


def test_comprehensive_intercepts_falls_back_to_algebraic_gt_without_visible_facts():
    pred = {"intercepts": {"x_intercepts": [-5, -4, -3], "y_intercept": None}}
    gt = {"intercepts": {"x_intercepts": [-5, -4, -3], "y_intercept": -60.0}}

    result = evaluate_comprehensive(pred, gt)

    intercepts = result["sub_scores"]["intercepts"]
    assert intercepts["score"] == 50.0
    assert intercepts["gt_display"] == "x:[-5, -4, -3] y:-60.0"
