"""Tests for the evidence-complementarity measurement.

This is the one positive PlotQA claim that survived scrutiny, so its arithmetic
is pinned. Two things carry the result:

* the 2x2 orientation -- swapping `rescue` and `risk` would invert the
  conclusion while still producing a plausible-looking table;
* the bootstrap CI, which is what makes the margin a claim rather than an
  observation.

The CI is digest-keyed rather than RNG-seeded so it reproduces exactly across
machines. That property is tested directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_plotqa_evidence_complementarity import (  # noqa: E402
    bootstrap_margin,
    contingency,
)


def _row(sample_id: str, baseline_right: bool, reachable: bool) -> dict:
    return {"sample_id": sample_id, "E0": baseline_right, "reachable": reachable}


class TestContingency:
    def test_cells_are_oriented_correctly(self):
        """rescue = baseline WRONG and evidence reachable; risk is its mirror."""
        rows = [
            _row("a", False, True),    # rescue
            _row("b", False, True),    # rescue
            _row("c", True, False),    # risk
            _row("d", True, True),     # redundant
            _row("e", False, False),   # dead
        ]
        stats = contingency(rows, "E0")
        assert stats["rescue_baseline_wrong_evidence_reachable"] == 2
        assert stats["risk_baseline_right_evidence_unreachable"] == 1
        assert stats["redundant_baseline_right_evidence_reachable"] == 1
        assert stats["dead_baseline_wrong_evidence_unreachable"] == 1
        assert stats["net_ceiling"] == 1

    def test_baseline_accuracy_counts_only_the_baseline(self):
        rows = [_row("a", True, False), _row("b", True, True), _row("c", False, True)]
        assert contingency(rows, "E0")["baseline_accuracy"] == 2 / 3

    def test_ratio_is_none_rather_than_infinite_when_risk_is_zero(self):
        rows = [_row("a", False, True), _row("b", False, True)]
        assert contingency(rows, "E0")["rescue_to_risk_ratio"] is None

    def test_empty_input_does_not_divide_by_zero(self):
        stats = contingency([], "E0")
        assert stats["n"] == 0
        assert stats["baseline_accuracy"] is None


class TestBootstrapMargin:
    def _rows(self, rescue: int, risk: int, filler: int = 0) -> list[dict]:
        rows = [_row(f"resc{i}", False, True) for i in range(rescue)]
        rows += [_row(f"risk{i}", True, False) for i in range(risk)]
        rows += [_row(f"fill{i}", True, True) for i in range(filler)]
        return rows

    def test_is_deterministic_across_calls(self):
        """A digest-keyed resample must reproduce exactly, with no global RNG."""
        rows = self._rows(30, 8, 20)
        first = bootstrap_margin(rows, "E0", resamples=200)
        second = bootstrap_margin(rows, "E0", resamples=200)
        assert first == second

    def test_a_clear_margin_excludes_zero(self):
        result = bootstrap_margin(self._rows(30, 2, 10), "E0", resamples=300)
        assert result["excludes_zero"] is True
        assert result["ci_95"][0] > 0

    def test_a_balanced_margin_does_not_exclude_zero(self):
        """Equal rescue and risk must not read as a positive result."""
        result = bootstrap_margin(self._rows(10, 10, 10), "E0", resamples=300)
        assert result["excludes_zero"] is False
        assert result["ci_95"][0] <= 0 <= result["ci_95"][1]

    def test_an_adverse_margin_is_reported_as_negative(self):
        result = bootstrap_margin(self._rows(2, 30, 10), "E0", resamples=300)
        assert result["median"] < 0
        assert result["excludes_zero"] is False

    def test_the_seed_actually_drives_the_resampling(self):
        """Two seeds must draw different samples, even when both CIs agree.

        Agreement between seeds is a sign the estimate is stable, so comparing
        intervals cannot distinguish "seed is wired in" from "seed is ignored".
        A degenerate row set makes the draw itself observable: with a single
        rescue among many risks, the margin varies draw to draw.
        """
        rows = self._rows(1, 40)
        a = bootstrap_margin(rows, "E0", resamples=40, seed="alpha")
        b = bootstrap_margin(rows, "E0", resamples=40, seed="beta")
        assert a == bootstrap_margin(rows, "E0", resamples=40, seed="alpha")
        assert a["seed"] == "alpha" and b["seed"] == "beta"
        assert a["ci_95"] != b["ci_95"] or a["median"] != b["median"]

    def test_empty_input_is_reported_not_crashed(self):
        result = bootstrap_margin([], "E0")
        assert result["excludes_zero"] is False
        assert result["median"] is None
