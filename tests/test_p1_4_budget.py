from scripts.estimate_p1_4_rerun_budget import estimate_budget


def test_p1_4_budget_matches_480_sample_baseline_rerun():
    budget = estimate_budget()

    assert budget["total_samples"] == 480
    assert budget["condition_jobs"] == 2400
    assert budget["reasoning_calls"] == 3360
    assert budget["judge_calls"] == 2400
    assert budget["nominal_http_calls"] == 5760
    assert budget["worst_case_http_attempt_cap"] == 28800
