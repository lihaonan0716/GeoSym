from scripts.run_p0_3_extrema_validation import (
    IMAGES_PER_FAMILY,
    MAX_GENERATION_ATTEMPTS_PER_SLOT,
    MIN_DISTINCT_QUOTA_FAMILIES,
    PROFILES_PER_FAMILY,
    QUOTA_DIVERSITY_RETRY_ATTEMPTS,
    QUOTA_SOURCE_PROFILES,
    SAMPLE_CATEGORY_QUOTAS,
    summarize_manifest,
)


def test_p0_3_generation_contract_is_bounded_and_balanced():
    assert IMAGES_PER_FAMILY == 8
    assert len(PROFILES_PER_FAMILY) == IMAGES_PER_FAMILY
    assert PROFILES_PER_FAMILY.count("digital_mild") == 4
    assert PROFILES_PER_FAMILY.count("digital_hard") == 4
    assert MAX_GENERATION_ATTEMPTS_PER_SLOT == 80


def test_p0_3_summary_reports_duplicate_and_rejection_counts():
    manifest = {
        "rejected_candidates": [{"sample_id": "rejected"}],
        "samples": [
            {
                "family": "quadratic",
                "duplicate_image": False,
                "duplicate_truth": False,
                "classification": {
                    "categories": ["ordinary_true_extrema"],
                    "true_extrema_count": 1,
                    "candidate_count": 1,
                    "keep_true": 1,
                    "keep_false": 0,
                    "drop_true": 0,
                    "drop_false": 0,
                },
            }
        ],
    }

    summary = summarize_manifest(manifest)

    assert summary["sample_count"] == 1
    assert summary["duplicate_image_count"] == 0
    assert summary["duplicate_truth_count"] == 0
    assert summary["rejected_candidate_count"] == 1
    assert summary["sample_category_quotas"]["ordinary_true_extrema"]["observed"] == 1
    assert summary["sample_category_quota_passed"] is False
    assert summary["h1_eligible"] is False


def test_p0_3_category_quota_targets_are_frozen():
    assert SAMPLE_CATEGORY_QUOTAS == {
        "ordinary_true_extrema": 16,
        "true_extrema_large_gap_or_jump": 16,
        "discontinuity_or_asymptote_pseudo_extrema": 16,
        "no_extrema_negative": 16,
    }


def test_p0_3_quota_sources_are_frozen_and_low_cost():
    assert QUOTA_DIVERSITY_RETRY_ATTEMPTS == 1
    assert MIN_DISTINCT_QUOTA_FAMILIES == 7
    assert QUOTA_SOURCE_PROFILES["true_extrema_large_gap_or_jump"] == (
        ("piecewise_linear", "clean_render"),
        ("piecewise_linear", "digital_mild"),
        ("piecewise_linear", "robustness_stress"),
    )
    assert QUOTA_SOURCE_PROFILES["discontinuity_or_asymptote_pseudo_extrema"] == (
        ("rational", "robustness_stress"),
        ("rational", "digital_hard"),
        ("rational", "digital_mild"),
    )
