import json
from pathlib import Path

from scripts.export_paper_results import export_paper_results


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_registry(path: Path, source_path: str, required: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
schema_version: geosym_paper_results_sources_v1
exports:
  current:
    raw_data_manifests:
      - name: raw_manifest
        path: dataset/raw/example/manifest.json
        required: true
        role: Test raw manifest.
    source_artifacts:
      evidence_audit:
        path: {source_path}
        required: {str(required).lower()}
        role: Test evidence artifact.
    tables:
      source_artifacts:
        output: tables/source_artifacts.csv
    claims:
      claim_matrix:
        output: claims/claim_matrix.json
""".lstrip(),
        encoding="utf-8",
    )


def test_export_paper_results_writes_manifest_tables_and_hashes(tmp_path: Path) -> None:
    root = tmp_path
    write_json(root / "dataset/raw/example/manifest.json", {"schema_version": "raw_test_v1"})
    write_json(
        root / "outputs/source/evidence_audit.json",
        {"schema_version": "audit_test_v1", "stage_status": "completed"},
    )
    config_path = root / "config/paper_results_sources.yaml"
    write_registry(config_path, "outputs/source/evidence_audit.json")

    manifest = export_paper_results(config_path, root / "outputs/paper_results/current", root=root)

    output_root = root / "outputs/paper_results/current"
    assert manifest["export_status"] == "ready_for_review"
    assert manifest["blocking_findings"] == []
    assert (output_root / "manifest.json").is_file()
    assert (output_root / "MANIFEST.md").is_file()
    assert (output_root / "tables/source_artifacts.csv").is_file()
    hashes = json.loads((output_root / "provenance/artifact_hashes.json").read_text(encoding="utf-8"))
    assert hashes["evidence_audit"]["sha256"]
    source_ref = json.loads((output_root / "source_artifacts/evidence_audit.json").read_text(encoding="utf-8"))
    assert source_ref["source_stage_status"] == "completed"


def test_export_paper_results_blocks_missing_required_source(tmp_path: Path) -> None:
    root = tmp_path
    write_json(root / "dataset/raw/example/manifest.json", {"schema_version": "raw_test_v1"})
    config_path = root / "config/paper_results_sources.yaml"
    write_registry(config_path, "outputs/source/missing.json")

    manifest = export_paper_results(config_path, root / "outputs/paper_results/current", root=root)

    assert manifest["export_status"] == "blocked"
    assert "missing_required:evidence_audit:outputs/source/missing.json" in manifest["blocking_findings"]


def test_export_paper_results_allows_missing_optional_source(tmp_path: Path) -> None:
    root = tmp_path
    write_json(root / "dataset/raw/example/manifest.json", {"schema_version": "raw_test_v1"})
    config_path = root / "config/paper_results_sources.yaml"
    write_registry(config_path, "outputs/source/missing.json", required=False)

    manifest = export_paper_results(config_path, root / "outputs/paper_results/current", root=root)

    assert manifest["export_status"] == "ready_for_review"
    assert manifest["blocking_findings"] == []
    assert "missing_optional:evidence_audit:outputs/source/missing.json" in manifest["warnings"]


def test_export_paper_results_rejects_reserved_workspace_boundary(tmp_path: Path) -> None:
    root = tmp_path
    write_json(root / "dataset/raw/example/manifest.json", {"schema_version": "raw_test_v1"})
    config_path = root / "config/paper_results_sources.yaml"
    reserved = "private" + "_lab"
    write_registry(config_path, f"{reserved}/source.json")

    try:
        export_paper_results(config_path, root / "outputs/paper_results/current", root=root)
    except ValueError as exc:
        assert "reserved internal workspace boundary" in str(exc)
    else:
        raise AssertionError("reserved workspace path was not rejected")
