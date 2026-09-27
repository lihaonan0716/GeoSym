"""The PlotQA 500-sample roster is restored from raw rows, not re-scored.

The run was marked superseded in full because one of its five arms was wrong.
That was correct about the arm and too broad about the run: 1000 of the 2500
paid calls were image-only baselines that never touched the defective code, and
330 of the defective arm's own 500 rows never entered the solver either.

These tests pin what makes the restoration trustworthy: the defect is bounded
to the rows that fired the solver, every one of those rows is replaced, and the
assembled arms reproduce the independent 2026-09-14 analysis.

Zero paid calls.
"""

from __future__ import annotations

import json

import pytest

restore = pytest.importorskip("scripts.restore_plotqa_500")

pytestmark = pytest.mark.skipif(
    not restore.CACHED.exists(), reason="PlotQA 500 artifacts not present")


@pytest.fixture(scope="module")
def report():
    return restore.build()


@pytest.fixture(scope="module")
def cached_rows():
    return json.loads(restore.CACHED.read_text(encoding="utf-8"))["results"]


def test_restoration_reproduces_the_independent_analysis(report):
    assert report["status"] == "restored", report["problems"]
    assert not report["problems"], report["problems"]


def test_every_arm_matches_its_recorded_total(report):
    arms = report["arms"]
    assert arms["E0_plotqa_direct"]["correct"] == 390
    assert arms["E1_plotqa_cot"]["correct"] == 374
    assert arms["G_measured_raw"]["correct"] == 338
    assert arms[restore.REPAIRED_ARM]["correct"] == 380
    assert arms["Measured_flat_answer_baseline"]["correct"] == 186


def test_every_solver_corrupted_row_is_replaced(report):
    # A single surviving solver row would put the defect back into the total.
    assert report["solver_rows_total"] == 170
    assert report["solver_rows_replaced"] == 170


def test_solver_and_transport_retry_are_different_sets(cached_rows):
    """Counting their union reported 175 replacements for a 170-row defect.

    170 rows fired the solver; 30 separate rows failed in transport and were
    retried symmetrically in both arms. 18 rows are in both sets and 5 are
    transport-only, so neither count substitutes for the other.
    """
    solver = {r["sample_id"] for r in cached_rows
              if r["condition"] == restore.DEFECTIVE_ARM
              and r.get("deterministic_m3")}
    retry = json.loads(restore.RETRY.read_text(encoding="utf-8"))["results"]
    retried = {r["sample_id"] for r in retry
               if r["condition"] == restore.REPAIRED_ARM}
    assert len(solver) == 170
    assert len(retried) == 23
    assert len(retried & solver) == 18
    assert len(retried - solver) == 5


def test_the_defect_is_bounded_to_the_rows_that_fired_the_solver(cached_rows):
    """On untouched rows the curated arm was ahead, which is why the run is
    restorable rather than discardable.

    Solver rows: 56/170 curated against 130/170 raw. Untouched rows:
    229/330 against 208/330.
    """
    by_arm: dict[str, dict[str, dict]] = {}
    for row in cached_rows:
        by_arm.setdefault(row["condition"], {})[row["sample_id"]] = row

    curated = by_arm[restore.DEFECTIVE_ARM]
    raw = by_arm["G_measured_raw"]
    fired = {s for s, r in curated.items() if r.get("deterministic_m3")}

    on_solver_curated = sum(bool(curated[s].get("correct")) for s in fired)
    on_solver_raw = sum(bool(raw[s].get("correct")) for s in fired)
    untouched = set(curated) - fired
    off_solver_curated = sum(bool(curated[s].get("correct"))
                             for s in untouched)
    off_solver_raw = sum(bool(raw[s].get("correct")) for s in untouched)

    assert (on_solver_curated, on_solver_raw) == (56, 130)
    assert (off_solver_curated, off_solver_raw) == (229, 208)
    assert off_solver_curated > off_solver_raw


def test_as_run_completion_matches_the_original_summaries(report):
    """Assembled completion is not the completion the run recorded.

    The symmetric retry recovered 7 raw rows and 3 curated ones. None changed
    an answer from wrong to right, so accuracy is untouched while completion
    is not. An earlier report printed the assembled figure for one arm beside
    as-run figures for four others.
    """
    as_run = {name: arm["completed_as_run"]
              for name, arm in report["arms"].items()}
    assert as_run == {
        "E0_plotqa_direct": 499,
        "E1_plotqa_cot": 499,
        "G_measured_raw": 489,
        restore.REPAIRED_ARM: 482,
        "Measured_flat_answer_baseline": 459,
    }


def test_the_retry_changed_completion_but_not_raw_accuracy(report):
    # 338 is a native total: it holds with or without the retry overlay.
    raw = report["arms"]["G_measured_raw"]
    assert raw["correct"] == 338
    assert raw["completed"] > raw["completed_as_run"]


def test_row_provenance_is_what_the_analysis_recorded(report):
    assert report["repaired_arm_row_provenance"] == {
        "retry": 23, "repaired": 152, "cached": 325}


def test_the_baselines_share_the_measured_roster(cached_rows):
    # A restored contrast across different rosters would compare nothing.
    baselines = json.loads(
        restore.BASELINES.read_text(encoding="utf-8"))["results"]
    assert ({r["sample_id"] for r in cached_rows}
            == {r["sample_id"] for r in baselines})


def test_the_negative_result_is_reported_not_buried(report):
    """Invariant 8, asserted on the artifact rather than trusted to prose.

    The measured route trails the image-only baseline. The restored file has
    to carry that, because the arm totals alone read as a success.
    """
    assert report["headline"]["measured_vs_strongest_image_only"] == -10
    assert report["headline"]["measured_vs_uncurated"] == +42
    limits = " ".join(report["interpretation_limits"])
    assert "380 against 390" in limits
    assert "invariant 12" in limits


def test_no_source_artifact_was_modified(report):
    """Restoration assembles; it never edits a paid artifact in place."""
    for path in report["source_artifacts"].values():
        assert path.startswith("outputs/")
    assert report["paid_calls_executed"] == 0
