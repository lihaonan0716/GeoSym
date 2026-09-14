from scripts.run_m1m2_recovery_phase_a import paired_stats


def test_paired_stats_reports_nonzero_rate_and_required_pairs():
    row = paired_stats([0.0, 1.0, -1.0, 0.0])

    assert row["n"] == 4
    assert row["nonzero_pair_rate"] == 0.5
    assert row["required_pairs_for_0_19"] > 0
