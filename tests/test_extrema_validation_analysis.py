from scripts.analyze_extrema_validation import analyze_manifest


def test_h1_go_when_frozen_gate_is_satisfied():
    manifest = {
        "name": "unit manifest",
        "sampling_strategy": "quota_stratified_v1",
        "rule_version": "p0-2-discontinuity-bridge-v1",
        "summary": {"h1_eligible": True},
        "samples": [
            {
                "family": "quadratic",
                "classification": {
                    "keep_true": 4,
                    "keep_false": 0,
                    "drop_true": 0,
                    "drop_false": 0,
                },
            },
            {
                "family": "rational",
                "classification": {
                    "keep_true": 0,
                    "keep_false": 0,
                    "drop_true": 0,
                    "drop_false": 4,
                },
            },
        ],
    }

    report = analyze_manifest(manifest)

    assert report["h1_decision"] == "go"
    assert report["delta_keep_precision_minus_drop_true_rate"] == 1.0
    assert report["family_clustered_bootstrap"]["ci95"] == [1.0, 1.0]


def test_h1_stop_when_manifest_is_not_eligible():
    manifest = {
        "name": "unit manifest",
        "summary": {"h1_eligible": False},
        "samples": [
            {
                "family": "quadratic",
                "classification": {
                    "keep_true": 4,
                    "keep_false": 0,
                    "drop_true": 0,
                    "drop_false": 4,
                },
            }
        ],
    }

    report = analyze_manifest(manifest)

    assert report["h1_decision"] == "stop"
    assert any("not H1 eligible" in warning for warning in report["warnings"])
