"""The dimension-exclusion analysis matches what the manuscript prints.

Table 3 claims the margin survives removal of the prompt-sensitive dimensions
(D8 domain, D9 range) and the judge-scored one (D10). These numbers were
previously carried as prose from a retired cohort; they are now recomputed from
the artifact, and this suite pins them so the table cannot drift away from it.

Zero paid calls.
"""

from __future__ import annotations

import pytest

# Dev-local script (`scripts/*` is gitignored): skip rather than abort the
# suite's collection in a clean clone.
_module = pytest.importorskip("scripts.compute_dimension_exclusion")

BENCH = _module.BENCH
HEADLINE_DELTA = _module.HEADLINE_DELTA
compute = _module.compute
load_layer_weights = _module.load_layer_weights
weighted_score = _module.weighted_score

pytestmark = pytest.mark.skipif(
    not BENCH.exists(), reason="bench artifact not present on this host")


@pytest.fixture(scope="module")
def report():
    return compute()


#: What the manuscript's Table 3 prints, to two decimals.
PRINTED = {
    "all_ten": (81.46, 70.80, 10.66),
    "exclude_d8_d9": (77.07, 67.79, 9.27),
    "exclude_d8_d9_d10": (74.33, 63.85, 10.48),
}


@pytest.mark.parametrize("scenario", sorted(PRINTED))
def test_table_matches_the_artifact(report, scenario):
    method, baseline, delta = PRINTED[scenario]
    entry = report["scenarios"][scenario]
    assert entry["method_mean"] == pytest.approx(method, abs=0.005)
    assert entry["baseline_mean"] == pytest.approx(baseline, abs=0.005)
    assert entry["mean_delta"] == pytest.approx(delta, abs=0.005)


def test_every_scenario_uses_the_same_paired_samples(report):
    # A margin that survives exclusion only because the sample set changed
    # would be an artefact of the exclusion, not a robustness result.
    counts = {name: entry["n"] for name, entry in report["scenarios"].items()}
    assert len(set(counts.values())) == 1, counts
    assert counts["all_ten"] == 237


def test_the_unexcluded_case_reproduces_the_headline(report):
    # This script forms a score independently of the writing-pack verifier.
    # Agreement on the unexcluded case is what licenses trusting the others.
    assert report["scenarios"]["all_ten"]["mean_delta"] == pytest.approx(
        HEADLINE_DELTA, abs=0.01)


def test_every_scenario_stays_significant(report):
    for name, entry in report["scenarios"].items():
        assert entry["p_value"] < 1e-7, (name, entry["p_value"])


def test_excluding_a_dimension_renormalises_rather_than_shrinks():
    # Without renormalisation every condition's score would fall, and the
    # contrast would report the exclusion rather than the method.
    layers = load_layer_weights()
    perfect = {d: 100.0 for spec in layers.values()
               for d in (spec.get("dimensions") or {})}
    assert weighted_score(perfect, layers) == pytest.approx(100.0)
    assert weighted_score(perfect, layers,
                          ("domain", "range")) == pytest.approx(100.0)
    assert weighted_score(
        perfect, layers,
        ("domain", "range", "judge_inference_score")) == pytest.approx(100.0)


def test_excluding_every_dimension_returns_none():
    layers = load_layer_weights()
    every = [d for spec in layers.values()
             for d in (spec.get("dimensions") or {})]
    assert weighted_score({d: 100.0 for d in every}, layers, every) is None
