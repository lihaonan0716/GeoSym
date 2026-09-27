"""Tests for the writing-pack verifier.

The verifier is what stops a stale figure reaching the manuscript, so its own
failure mode matters: a guard that cannot fire certifies staleness instead of
catching it. These tests prove both halves fire — the numeric comparison and
the prose assertions.

The pack and the paid artifacts live under gitignored directories (`docs/`,
`outputs/`), so a clean clone has neither. These tests skip in that case rather
than fail: the test suite is part of the public artifact and must collect
without the private workspace (CLAUDE.md invariant 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Dev-local script (`scripts/*` is gitignored): skip rather than abort the
# suite's collection in a clean clone.
_module = pytest.importorskip("scripts.verify_writing_pack_numbers")  # noqa: E402

BENCH = _module.BENCH
LAYER_PROFILE = _module.LAYER_PROFILE
PACK = _module.PACK
PLOTQA_ATTR = _module.PLOTQA_ATTR
PLOTQA_M0 = _module.PLOTQA_M0
R3_3 = _module.R3_3
R3_3_PAID = _module.R3_3_PAID
SUBSTITUTABILITY = _module.SUBSTITUTABILITY
build_assertions = _module.build_assertions
prose_assertions = _module.prose_assertions

_ARTIFACTS = (BENCH, PLOTQA_M0, PLOTQA_ATTR, R3_3, R3_3_PAID,
              LAYER_PROFILE, SUBSTITUTABILITY)

requires_artifacts = pytest.mark.skipif(
    not all(p.exists() for p in _ARTIFACTS),
    reason="paid artifacts are gitignored; present only in the dev workspace")
requires_pack = pytest.mark.skipif(
    not PACK.exists(),
    reason="writing pack is gitignored; present only in the dev workspace")


def _pack_text() -> str:
    return PACK.read_text(encoding="utf-8")


@requires_artifacts
class TestNumericChecksPassOnCurrentArtifacts:
    def test_every_numeric_check_matches_its_artifact(self):
        failures = [r for r in build_assertions() if r["status"] != "OK"]
        assert not failures, f"writing pack disagrees with artifacts: {failures}"

    def test_the_check_set_is_not_trivially_small(self):
        # A verifier that checks three numbers gives false confidence.
        assert len(build_assertions()) >= 60


@requires_artifacts
class TestC1StatisticsAreAllPinned:
    """Three statistics for C1 coexist and differ by which samples enter.

    Pinning all three prevents the paper from quoting one figure in the table
    and another in the abstract — the inconsistency a reviewer finds first.
    """

    @staticmethod
    def _value(name):
        return next(r for r in build_assertions() if r["check"] == name)

    def test_difference_of_means_is_pinned(self):
        assert self._value("C1 difference of means")["status"] == "OK"

    def test_paired_statistic_is_pinned(self):
        assert self._value("C1 paired (LOCKED)")["status"] == "OK"

    def test_complete_case_statistic_is_pinned(self):
        assert self._value("C1 complete-case")["status"] == "OK"

    def test_the_three_statistics_are_genuinely_different(self):
        means = self._value("C1 difference of means")["artifact_value"]
        paired = self._value("C1 paired (LOCKED)")["artifact_value"]
        complete = self._value("C1 complete-case")["artifact_value"]
        assert len({round(means, 2), round(paired, 2), round(complete, 2)}) == 3

    def test_the_locked_statistic_is_marked_as_such(self):
        # The lock lives in the check name, so a future edit that swaps the
        # headline statistic has to touch a line that says LOCKED.
        names = [r["check"] for r in build_assertions()]
        assert sum("LOCKED" in n for n in names) == 1

    def test_both_significance_tests_are_pinned(self):
        # A magnitude-driven effect with a near-even win/loss split needs both
        # the parametric and the rank test on record.
        assert self._value("C1 t p")["status"] == "OK"
        assert self._value("C1 Wilcoxon p")["status"] == "OK"

    def test_the_win_loss_asymmetry_is_pinned(self):
        # Wins average ~+28.9 and losses ~-8.1: this is what explains a
        # +10.66 mean alongside a 0.00 median.
        wins = self._value("C1 mean win magnitude")["artifact_value"]
        losses = self._value("C1 mean loss magnitude")["artifact_value"]
        assert wins > 0 > losses
        assert abs(wins) > abs(losses) * 3

    def test_the_normalisation_bound_stays_below_the_measured_value(self):
        # The bound is generous to the baselines by construction, so it must
        # not exceed the measured margin -- if it did, the bound is inverted.
        bound = self._value("C1 normalisation bound")["artifact_value"]
        measured = self._value("C1 paired (LOCKED)")["artifact_value"]
        assert bound < measured


@requires_artifacts
class TestCostIsCountedAsReasoningCalls:
    """The HTTP figure includes a judge call; the architectural cost does not.

    Quoting the raw 2.0x HTTP ratio understates what a deployer pays, because
    a deployment runs no judge. The verifier pins the reasoning-call counts so
    the 3x figure cannot silently revert.
    """

    @staticmethod
    def _value(name):
        return next(r for r in build_assertions() if r["check"] == name)

    def test_layered_arm_makes_three_reasoning_calls(self):
        assert self._value("C9 layered reasoning calls")["artifact_value"] == 3.0

    def test_single_pass_makes_one_reasoning_call(self):
        assert self._value("C9 single-pass reasoning calls")["artifact_value"] == 1.0


@requires_pack
class TestProseGuardsFire:
    @staticmethod
    def _failures(text):
        return [r["check"] for r in prose_assertions(text) if r["status"] != "OK"]

    def test_the_current_pack_passes(self):
        assert not self._failures(_pack_text())

    def test_an_unsupported_plotqa_win_is_caught(self):
        text = _pack_text()
        text += "\n\nGeoSym beats the image-only baseline on PlotQA.\n"
        assert "prose: asserts a PlotQA win" in self._failures(text)

    def test_outperforms_phrasing_is_also_caught(self):
        text = _pack_text()
        text += "\n\nGeoSym outperforms the image-only arm.\n"
        assert "prose: asserts a PlotQA win" in self._failures(text)

    def test_dropping_the_varying_n_warning_is_caught(self):
        text = _pack_text().replace(
            "it is not a uniform 240", "")
        assert "prose: names the varying n" in self._failures(text)

    def test_dropping_the_reasoning_cost_statement_is_caught(self):
        text = _pack_text().replace(
            "3 reasoning calls against 1", "")
        assert "prose: states the 3x reasoning cost" in self._failures(text)

    def test_dropping_the_subtype_breakdown_is_caught(self):
        text = _pack_text().replace(
            "Per-subtype detail", "")
        assert "prose: shows the per-subtype split" in self._failures(text)


class TestPackExists:
    def test_the_pack_is_dev_local(self):
        # docs/ is gitignored precisely so unpublished results can live on
        # disk (CLAUDE.md invariant 2). The pack must stay there.
        assert PACK.parent.name == "docs"
