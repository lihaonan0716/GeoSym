"""Roster composition, retired vocabulary, and extractor parameters.

Three checks added 2026-09-26 after a review found defects that every existing
verifier passed over:

* the manuscript stated "30 per family" for a roster whose families run 19-32;
* the supplementary still routed evidence to ``M1`` and cited a ``Stage 2.5``
  that the prose/code verifier independently reports as absent;
* supplementary Table S1 listed fifteen extractor parameters, fourteen of which
  the executed extractor does not contain.

A verifier with no test can stop checking and still exit zero. These pin the
behaviour each one was written for, including the false-negative that the
first version of the vocabulary scan produced.

Zero paid calls.
"""

from __future__ import annotations

import pytest

# These verifiers are dev-local (`scripts/*` is gitignored). Skip rather than
# fail collection, so a clean clone still collects the public suite.
params = pytest.importorskip("scripts.verify_extractor_parameters")
vocab = pytest.importorskip("scripts.verify_retired_vocabulary")
roster = pytest.importorskip("scripts.verify_roster_composition")


# ---------------------------------------------------------------- roster ---

bench_required = pytest.mark.skipif(
    not roster.BENCH.exists(), reason="bench artifact not present")


@pytest.fixture(scope="module")
def roster_report():
    return roster.build()


@bench_required
def test_roster_totals_match_the_manuscript(roster_report):
    assert not roster_report["problems"], roster_report["problems"]


@bench_required
def test_families_are_not_uniform(roster_report):
    """The defect: 240 = 30 x 8 is arithmetic, not a description of the data.

    If a future roster genuinely becomes uniform this fails, which is the
    right signal -- the prose describing unequal families would then be wrong.
    """
    composition = roster_report["composition"]
    assert composition["per_family_min"] == 19
    assert composition["per_family_max"] == 32
    assert not composition["uniform_per_family"]
    assert composition["roster_total"] == 240


@bench_required
def test_every_purged_row_was_re_measured(roster_report):
    # A purge that was never re-run is a hole in the roster and the arms stop
    # being matched. All 137 came back, and that is asserted rather than
    # assumed.
    for name, record in roster_report["purge_records"]["records"].items():
        assert record["net_loss"] == 0, (name, record)


@bench_required
def test_purge_matching_uses_the_unfiltered_row_set():
    """The retired arm shares the result file with the live ones.

    Matching the purge records against E5-excluded rows reported 28 rows as an
    unrecovered hole. They are E5 rows, re-measured like the rest; the hole
    was an artefact of the filter.
    """
    live, every = roster.load_rows()
    assert len(every) > len(live)
    purged = roster.purges(every)
    assert sum(r["net_loss"] for r in purged["records"].values()) == 0


@bench_required
def test_unscored_rows_are_mostly_the_ocr_baselines(roster_report):
    # The prose attributes n = 229-240 to unscorable answers. Most of the loss
    # is an OCR stage inside E3/E4 declining to build a ruler, which is a
    # property of those baselines rather than of the shared backbone.
    causes = roster_report["unscored_row_causes"]
    baseline_losses = sum(sum(v.values())
                          for k, v in causes.items() if k in ("E3", "E4"))
    total = sum(sum(v.values()) for v in causes.values())
    assert baseline_losses == 22
    assert total == 26


@bench_required
def test_forbidden_roster_phrasing_is_absent(roster_report):
    assert not roster_report["forbidden_prose_hits"], \
        roster_report["forbidden_prose_hits"]


# ------------------------------------------------------------ vocabulary ---

@pytest.fixture(scope="module")
def vocab_report():
    return vocab.build()


@pytest.mark.skipif(not vocab.SUPPLEMENTARY.exists(),
                    reason="writing surface not present")
def test_writing_surface_has_no_retired_vocabulary(vocab_report):
    assert not vocab_report["hits"], vocab_report["hits"]


def test_module_names_are_caught_standalone():
    hits = vocab.scan_text(vocab.ROOT / "probe.tex",
                           "The evidence format consumed by M1.")
    assert [h["match"] for h in hits] == ["M1"]


def test_word_boundaries_prevent_the_false_positives_that_hid_real_hits():
    """A loose search returns so much noise that a real hit reads as noise.

    Two genuine references survived a manual sweep for exactly this reason.
    Neither of these lines is a claim about a retired module.
    """
    benign = (r"where $c_{\text{ruler}}$ denotes the ruler confidence.",
              "The M0PRIME identifier is part of a longer name.",
              "Scores M10 and M12 are unrelated labels.")
    for line in benign:
        assert vocab.scan_text(vocab.ROOT / "probe.tex", line) == []


def test_verbatim_blocks_are_skipped():
    """A prompt is quoted as executed; editing it would misreport the run."""
    quoted = ("\\begin{lstlisting}\n"
              "You are the L1 stage consuming M1 output.\n"
              "\\end{lstlisting}\n")
    assert vocab.scan_text(vocab.ROOT / "probe.tex", quoted) == []


def test_artifact_paths_are_not_claims():
    line = r"recorded in \path{outputs/paper_level/m1m2_stress_128/a.json}"
    assert vocab.scan_text(vocab.ROOT / "probe.tex", line) == []


def test_stage_2_5_is_caught():
    hits = vocab.scan_text(vocab.ROOT / "probe.tex",
                           "mitigated by cross-validation in Stage~2.5.")
    assert hits and hits[0]["match"] == "Stage~2.5"


def test_full_geosym_matches_the_identifier_not_the_english():
    # "the full GeoSym pipeline" is ordinary prose; `full_geosym` is a deleted
    # condition string. The first version of the pattern conflated them.
    assert vocab.scan_text(vocab.ROOT / "p.tex",
                           "We evaluate the full GeoSym pipeline.") == []
    assert vocab.scan_text(vocab.ROOT / "p.tex", "condition full_geosym")


# ------------------------------------------------------------ parameters ---

@pytest.fixture(scope="module")
def params_report():
    return params.build()


@pytest.mark.skipif(not params.EXTRACTOR.exists(),
                    reason="extractor module not present")
def test_tabled_parameters_match_the_source(params_report):
    assert not params_report["problems"], params_report["problems"]


@pytest.mark.skipif(not params.EXTRACTOR.exists(),
                    reason="extractor module not present")
def test_every_parameter_resolves(params_report):
    unresolved = [r for r in params_report["parameters"]
                  if r["status"] != "ok"]
    assert not unresolved, unresolved


@pytest.mark.skipif(not params.EXTRACTOR.exists(),
                    reason="extractor module not present")
def test_the_mechanisms_the_old_table_implied_are_absent(params_report):
    # Hough, Canny, CLAHE, black-hat and perspective correction are the five
    # the prose/code verifier independently reports as unexecuted. The table
    # asserted parameters for them.
    present = [r["mechanism"] for r in params_report["absent_mechanisms"]
               if r["present"]]
    assert not present, present


@pytest.mark.skipif(not params.EXTRACTOR.exists(),
                    reason="extractor module not present")
def test_the_snapping_tolerance_is_a_quarter_unit(params_report):
    """The table said 0.3 while the prose beside it said a quarter unit.

    Both described the same constant, and the constant is 0.25.
    """
    row = next(r for r in params_report["parameters"]
               if "snapping tolerance" in r["parameter"])
    assert row["found"] == "0.25"


@pytest.mark.skipif(not params.EXTRACTOR.exists(),
                    reason="extractor module not present")
def test_the_structural_gate_admits_three_gridlines(params_report):
    """The table said six. `cubic` succeeds with three intersections."""
    row = next(r for r in params_report["parameters"]
               if "min gridlines" in r["parameter"])
    assert row["found"] == "3"


@pytest.mark.skipif(not params.SUPPLEMENTARY.exists(), reason="local manuscript absent")
@pytest.mark.parametrize("old,new,rule", [
    (r"|I_B(p)-I_G(p)|<8", r"\max_{c,c'}|I_c(p)-I_{c'}(p)|<8",
     "B-G mask comparison"),
    (r"\max(80,\lfloor W/14\rfloor)", r"\max(80,\lfloor H/14\rfloor)",
     "horizontal grid count"),
    ("200 uniformly spaced points", "70 uniformly spaced points",
     "expression sample count"),
    ("highest combined score", "nearest prediction", "extrema combined matching"),
    ("Each\nrun contributes its midpoint", "Each strip contributes one value",
     "per-run ordinate"),
])
def test_method_description_guard_rejects_the_observed_drift(old, new, rule):
    text = params.read_metric_text(params.SUPPLEMENTARY)
    assert old in text
    changed = text.replace(old, new)
    checks = {row["rule"]: row["status"]
              for row in params.check_method_descriptions(changed)}
    assert checks[rule] != "ok"
