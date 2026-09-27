"""The artifact registry classifies every directory and blocks stale citations.

Four ledgers described the same tree and disagreed: one was pinned to the
retired M1/M2 framing, one named an artifact authoritative that a later
decision retired, and the evidence bundle cited a run whose own status marker
reads "Do not cite this artifact as current PlotQA state". An agent reading the
tree could not tell which instruction won.

These tests pin the two properties that make the registry an answer rather than
a fifth opinion: nothing on disk escapes classification, and nothing the
registry calls stale can be cited by the bundle.

Zero paid calls.
"""

from __future__ import annotations

import pytest

# The registry script is dev-local (`scripts/*` is gitignored), like the
# bundle builder it checks. Skip rather than fail collection so a clean clone
# still collects the public suite.
registry = pytest.importorskip("scripts.build_artifact_registry")

OUTPUTS = registry.OUTPUTS
REGISTRY = registry.REGISTRY
STATUSES = registry.STATUSES
build = registry.build
discover = registry.discover

pytestmark = pytest.mark.skipif(
    not OUTPUTS.exists(), reason="outputs/ not present on this host")


@pytest.fixture(scope="module")
def report():
    return build()


def test_every_directory_on_disk_is_classified(report):
    # A new experiment must be classified before anything can cite it. Without
    # this, an unlabelled directory looks exactly like an authoritative one.
    unclassified = [k for k, v in report["entries"].items()
                    if v["exists"] and not v["declared"]]
    assert not unclassified, unclassified


def test_no_declared_directory_is_missing(report):
    absent = [k for k, v in report["entries"].items() if not v["exists"]]
    assert not absent, absent


def test_statuses_are_from_the_vocabulary():
    unknown = {k: v["status"] for k, v in REGISTRY.items()
               if v["status"] not in STATUSES}
    assert not unknown, unknown


def test_only_authoritative_artifacts_carry_claims():
    # A claim resting on a superseded artifact is the defect this file exists
    # to prevent; it is how 285/500 reached the manuscript.
    misfiled = {k: v["claims"] for k, v in REGISTRY.items()
                if v.get("claims") and v["status"] != "authoritative"}
    assert not misfiled, misfiled


def test_every_superseded_entry_names_its_successor():
    orphaned = [k for k, v in REGISTRY.items()
                if v["status"] == "superseded" and not v.get("superseded_by")]
    assert not orphaned, orphaned


def test_successors_resolve():
    dangling = {k: v["superseded_by"] for k, v in REGISTRY.items()
                if v.get("superseded_by") and v["superseded_by"] not in REGISTRY}
    assert not dangling, dangling


def test_the_evidence_bundle_cites_nothing_stale(report):
    stale = {k: v for k, v in report["bundle_references"].items()
             if REGISTRY[k]["status"] in ("superseded", "historical")}
    assert not stale, stale


def test_the_bundle_check_reads_the_registry_not_the_source_text():
    """The first version grepped the bundle's source and found nothing.

    ``build_paper_evidence_bundle.py`` wraps long paths across two string
    literals, so the substring it searched for never appeared and the check
    reported "consistent" while the defect was live. Importing the bundle's
    own registry is what makes the check immune to how the file is formatted.
    """
    references = registry.bundle_references()
    assert references, "bundle registry did not resolve to any registry entry"
    assert all(key in REGISTRY for key in references)


def test_withheld_artifacts_carry_no_claim():
    """Sound data the paper deliberately does not report.

    The restored PlotQA 500 roster is trustworthy and cites nothing: the
    maintainer decided on 2026-09-26 not to report the negative external
    result. Attaching a claim would reverse that decision by relabelling,
    which is why the registry fails on it rather than accepting the label.
    """
    for key in registry.WITHHELD:
        assert key in REGISTRY, key
        assert not REGISTRY[key].get("claims"), key


def test_an_authoritative_artifact_without_a_claim_must_say_why():
    """Otherwise it reads as an oversight and the next agent writes it in."""
    unexplained = [k for k, v in REGISTRY.items()
                   if v["status"] == "authoritative" and not v.get("claims")
                   and k not in registry.WITHHELD]
    assert not unexplained, unexplained


def test_the_restored_roster_is_citable_not_removed():
    """Restoration was the instruction; marking it removed was not.

    One arm of five was corrupted and has been replaced row-by-row. The
    directory holding the raw rows stays authoritative so the data remains
    usable, and the decision not to report it is recorded separately from
    whether it is sound.
    """
    for key in ("paper_level_evidence_package/plotqa_public_500",
                "paper_level_evidence_package/plotqa_public_500_restored"):
        assert REGISTRY[key]["status"] == "authoritative", key


def test_claims_cover_the_ledger():
    claimed = {claim for spec in REGISTRY.values()
               for claim in (spec.get("claims") or [])}
    # C6 is a decision record rather than an artifact directory, and C7 was
    # deleted on 2026-09-23 with no replacement.
    for claim in ("C1", "C2", "C3", "C4", "C5", "C8", "C9"):
        assert claim in claimed, claim


def test_discovery_and_declaration_agree():
    assert set(discover()) == set(REGISTRY)
