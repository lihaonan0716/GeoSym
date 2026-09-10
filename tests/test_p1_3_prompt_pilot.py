from scripts.run_p1_3_prompt_pilot import (
    CONDITIONS,
    MAX_HTTP_ATTEMPTS_PER_JOB,
    MAX_SAMPLE_LEVEL_JOBS,
    frozen_plan,
)


def test_p1_3_pilot_has_safe_frozen_cap():
    plan = frozen_plan()

    assert len(CONDITIONS) <= MAX_SAMPLE_LEVEL_JOBS
    assert MAX_HTTP_ATTEMPTS_PER_JOB == 1
    assert plan["conditions"] == ["E0", "E1", "E3", "E4"]
    assert plan["sample_roster"] == ["quadratic-clean_render/quadratic-clean_render_0000"]
    assert "E2" not in plan["conditions"]
    assert "It does not authorize full reruns or P2." in plan["interpretation_limits"]
