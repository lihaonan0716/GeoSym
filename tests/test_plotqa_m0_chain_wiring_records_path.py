"""The wiring verifier must report which records file it measured.

Regression guard for a real defect: the verifier hardcoded its input to
``plotqa_public_500/m0_measurement_records.json`` (via ``PACKAGE``), so after the
truncated-bar repair wrote a *new* records file beside the old one, the
per-family coverage rows still described the pre-repair extractor while the
headline numbers quoted elsewhere came from the repaired one. Mixing the two in
one table makes the per-family rows fail to add up to the headline.

These tests lock two properties:

1. the records file is selectable, so a post-repair run is possible at all;
2. every emitted artifact names and hashes the records file it used, so a
   pre/post mix-up is visible in the artifact instead of having to be inferred.
"""

import inspect
from pathlib import Path

import pytest

from scripts import verify_plotqa_m0_chain_wiring as wiring


class TestRecordsPathIsSelectable:
    def test_run_accepts_a_records_path(self):
        """A caller must be able to point the verifier at repaired records.

        Enforced on the signature rather than by running the analysis, because
        the failure mode was a hardcoded constant: if the parameter is absent
        there is no way to express "measure the repaired file".
        """
        params = inspect.signature(wiring.run).parameters
        assert "records_path" in params

    def test_records_path_defaults_to_none_not_a_hardcoded_package_path(self):
        """The default is resolved inside ``run``, keeping one source of truth."""
        default = inspect.signature(wiring.run).parameters["records_path"].default
        assert default is None

    def test_cli_exposes_the_records_flag(self):
        """Without a flag the parameter is unreachable from the command line."""
        source = inspect.getsource(wiring.main)
        assert "--records" in source


class TestProvenanceIsEmitted:
    def test_sha256_helper_hashes_file_bytes(self, tmp_path: Path):
        target = tmp_path / "records.json"
        target.write_bytes(b"[]")
        # sha256 of b"[]" -- pinned so a switch to a different digest is caught.
        assert wiring.sha256_of_file(target) == (
            "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
        )

    def test_two_different_records_files_hash_differently(self, tmp_path: Path):
        """The hash is what makes a silent input swap detectable."""
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_bytes(b'[{"sample_id": "x"}]')
        b.write_bytes(b'[{"sample_id": "y"}]')
        assert wiring.sha256_of_file(a) != wiring.sha256_of_file(b)

    def test_run_records_input_provenance_in_its_result(self):
        """``inputs.records`` / ``inputs.records_sha256`` must be emitted.

        Asserted on the source of ``run`` because executing it requires the
        frozen 500-sample artifacts, which are gitignored; the point being
        locked is that provenance is written at all.
        """
        source = inspect.getsource(wiring.run)
        assert '"inputs"' in source
        assert '"records_sha256"' in source

    def test_report_surfaces_the_records_file(self):
        """A reader of the Markdown must see which records produced the table."""
        source = inspect.getsource(wiring.render)
        assert "inputs" in source and "records" in source


class TestMixingIsCalledOutInLimits:
    def test_limits_warn_against_sharing_a_table_across_records_files(self):
        source = inspect.getsource(wiring.run)
        assert "never share one table" in source
