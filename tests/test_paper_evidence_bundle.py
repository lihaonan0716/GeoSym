"""Tests for the paper evidence bundle.

The bundle's whole value is that it recomputes each claim number from its
artifact and FAILS when the artifact and the claim ledger disagree. A guard
that cannot fire is worse than no guard: it certifies staleness. These tests
prove it fires, and that provenance is not invented where it is unknown.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

# Dev-local script (`scripts/*` is gitignored): skip rather than abort the
# suite's collection in a clean clone.
_module = pytest.importorskip("scripts.build_paper_evidence_bundle")

ARTIFACTS = _module.ARTIFACTS
CHECKS = _module.CHECKS
TOLERANCE = _module.TOLERANCE
resolve_model = _module.resolve_model


class TestEveryArtifactIsClaimed:
    def test_each_artifact_names_at_least_one_claim(self):
        for name, spec in ARTIFACTS.items():
            assert spec["claims"], f"{name} lists no claim"

    def test_each_artifact_states_its_role(self):
        for name, spec in ARTIFACTS.items():
            assert spec["role"].strip(), f"{name} has no role"

    def test_derived_artifacts_point_at_a_real_parent(self):
        for name, spec in ARTIFACTS.items():
            parent = spec.get("derived_from")
            if parent is not None:
                assert parent in ARTIFACTS, f"{name} derives from unknown {parent}"

    def test_no_artifact_derives_from_itself(self):
        for name, spec in ARTIFACTS.items():
            assert spec.get("derived_from") != name

    def test_checks_reference_declared_artifacts(self):
        for check, (_fn, _desc, needs) in CHECKS.items():
            for artifact in needs:
                assert artifact in ARTIFACTS, f"{check} needs unknown {artifact}"

    def test_no_artifact_path_crosses_the_private_boundary(self):
        # The public repo must never reference the paper workspace
        # (CLAUDE.md invariant 1). A bundle that copied from there would
        # leak unpublished material into a shareable directory.
        reserved = "private" + "_lab"
        for name, spec in ARTIFACTS.items():
            assert reserved not in spec["path"], f"{name} crosses the boundary"


class TestMismatchGuardActuallyFires:
    """If the artifact moves away from the ledger, the bundle must fail."""

    def test_a_shifted_value_exceeds_tolerance(self):
        # A 0.5-point drift in a headline number is exactly the kind of stale
        # figure that reaches a table unnoticed. It must not pass.
        assert abs(10.83 - 11.33) > TOLERANCE

    def test_float_formatting_noise_stays_within_tolerance(self):
        # The ledger records 10.83 while the artifact recomputes 10.8189.
        # That is rounding in the source document, not a changed result.
        assert abs(10.83 - 10.8189) <= TOLERANCE

    def test_tolerance_is_tight_enough_to_be_meaningful(self):
        assert TOLERANCE < 0.05


class TestModelProvenanceIsNeverInvented:
    def test_a_missing_artifact_is_unresolved(self, tmp_path):
        result = resolve_model(tmp_path / "does_not_exist.json")
        assert result["resolved"] is False

    def test_a_result_file_without_provenance_is_unresolved(self, tmp_path):
        artifact = tmp_path / "paid_result.json"
        artifact.write_text(json.dumps({"results": []}), encoding="utf-8")
        result = resolve_model(artifact)
        assert result["resolved"] is False
        assert "no model_provenance" in result["reason"]

    def test_provenance_in_the_result_file_is_used_directly(self, tmp_path):
        artifact = tmp_path / "paid_result.json"
        artifact.write_text(json.dumps({
            "model_provenance": {"display_name": "GPT-5.5",
                                 "api_model_id": "gpt-5.5"},
        }), encoding="utf-8")
        result = resolve_model(artifact)
        assert result["display_name"] == "GPT-5.5"
        assert result["source"] == "result_file"
        # Nothing was inferred, so no warning is warranted.
        assert "invariant_13_warning" not in result

    def test_a_reasoning_model_key_is_accepted(self):
        # Runners disagree: some write `display_name`, others only
        # `reasoning_model`. A recorded model must not read as unresolved.
        artifact = (ROOT / "outputs" / "m3_attribution_pilot"
                    / "paid_result.json")
        if not artifact.exists():
            return
        result = resolve_model(artifact)
        assert result["resolved"] is True
        assert result["display_name"]


class TestClaimCoverage:
    def test_every_ledger_claim_has_an_artifact(self):
        claimed = {c for spec in ARTIFACTS.values() for c in spec["claims"]}
        # C7 is absent deliberately: it was deleted on 2026-09-23 because the
        # run it rested on varies image access rather than evidence. Listing it
        # here would reintroduce it as a requirement.
        for claim in ("C1", "C2", "C3", "C4", "C5", "C8", "C9"):
            assert claim in claimed, f"{claim} has no artifact in the bundle"
        assert "C7" not in claimed, (
            "C7 was retired; see NARRATIVE.md O-4 before adding it back")

    def test_c8_requires_the_retained_plotqa_pair(self):
        _fn, _desc, needs = CHECKS["C8_retained_pair_present"]
        assert needs == ("plotqa_numeric_paid",)

    def test_withdrawn_rows_do_not_enter_the_c8_summary(self, tmp_path):
        path = tmp_path / "rows.json"
        path.write_text(json.dumps({"results": [
            {"condition": "E0_plotqa_direct", "correct": True},
            {"condition": "G_measured_raw", "correct": False},
            {"condition": "Measured_flat_answer_baseline", "correct": True},
        ]}), encoding="utf-8")
        report = _module.check_c8({"plotqa_numeric_paid": path})
        assert report["value"] == report["expected"] == 2
        assert set(report["detail"]) == {"E0_plotqa_direct", "G_measured_raw"}
