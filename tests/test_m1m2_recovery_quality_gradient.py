import inspect

from scripts.m1m2_recovery_quality_gradient import degrade_coords


def test_degrade_coords_does_not_accept_ground_truth():
    forbidden = {"gt", "ground_truth", "truth", "answer", "label"}
    assert forbidden.isdisjoint(inspect.signature(degrade_coords).parameters)


def test_degrade_coords_is_deterministic_and_changes_by_level():
    coords = [(x, x) for x in range(10)]

    first = degrade_coords(coords, sample_id="s0", level="q1_moderate_degradation")
    second = degrade_coords(coords, sample_id="s0", level="q1_moderate_degradation")
    strong = degrade_coords(coords, sample_id="s0", level="q2_strong_degradation")

    assert first == second
    assert first != strong
