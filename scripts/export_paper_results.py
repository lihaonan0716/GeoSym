"""Export paper-facing result inventories from frozen local artifacts.

The exporter is intentionally offline and side-effect limited: it never runs an
experiment, never calls a model service, and only reads paths declared in the
source registry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "geosym_paper_results_sources_v1"
EXPORT_SCHEMA_VERSION = "geosym_paper_results_export_v1"
RESERVED_INTERNAL_DIR = "private" + "_lab"


@dataclass(frozen=True)
class SourceRecord:
    section: str
    name: str
    path: str
    role: str
    required: bool
    exists: bool
    sha256: str | None
    schema_version: str | None
    stage_status: str | None
    status: str


def stable_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML registry must contain a mapping: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def path_has_reserved_component(path_text: str) -> bool:
    return RESERVED_INTERNAL_DIR in Path(path_text).parts


def safe_relative_path(path_text: str) -> None:
    candidate = Path(path_text)
    if candidate.is_absolute():
        raise ValueError(f"registry paths must be relative: {path_text}")
    if ".." in candidate.parts:
        raise ValueError(f"registry paths must not traverse upward: {path_text}")
    if path_has_reserved_component(path_text):
        raise ValueError("registry path crosses the reserved internal workspace boundary")


def resolve_under_root(root: Path, path_text: str) -> Path:
    safe_relative_path(path_text)
    resolved_root = root.resolve()
    resolved_path = (resolved_root / path_text).resolve()
    if os.path.commonpath([str(resolved_root), str(resolved_path)]) != str(resolved_root):
        raise ValueError(f"registry path escapes repository root: {path_text}")
    return resolved_path


def read_json_metadata(path: Path) -> tuple[str | None, str | None]:
    if path.suffix.lower() != ".json":
        return None, None
    try:
        payload = load_json(path)
    except (json.JSONDecodeError, OSError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    schema_version = payload.get("schema_version")
    stage_status = payload.get("stage_status")
    return (
        schema_version if isinstance(schema_version, str) else None,
        stage_status if isinstance(stage_status, str) else None,
    )


def iter_registry_sources(config: dict[str, Any], export_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported registry schema_version: {config.get('schema_version')}")
    exports = config.get("exports")
    if not isinstance(exports, dict) or export_id not in exports:
        raise ValueError(f"export id not found in registry: {export_id}")
    export_config = exports[export_id]
    if not isinstance(export_config, dict):
        raise ValueError(f"export config must be a mapping: {export_id}")

    entries: list[dict[str, Any]] = []
    for raw_entry in export_config.get("raw_data_manifests", []) or []:
        if not isinstance(raw_entry, dict):
            raise ValueError("raw_data_manifests entries must be mappings")
        entries.append({"section": "raw_data_manifests", **raw_entry})

    source_artifacts = export_config.get("source_artifacts", {}) or {}
    if not isinstance(source_artifacts, dict):
        raise ValueError("source_artifacts must be a mapping")
    for name, raw_entry in source_artifacts.items():
        if not isinstance(raw_entry, dict):
            raise ValueError(f"source_artifacts entry must be a mapping: {name}")
        entries.append({"section": "source_artifacts", "name": name, **raw_entry})

    return export_config, entries


def inspect_source(root: Path, entry: dict[str, Any], blocking_findings: list[str], warnings: list[str]) -> SourceRecord:
    name = str(entry.get("name") or "")
    path_text = str(entry.get("path") or "")
    role = str(entry.get("role") or "")
    required = bool(entry.get("required", False))
    section = str(entry.get("section") or "source_artifacts")
    if not name:
        raise ValueError(f"registry source entry is missing name: {entry}")
    if not path_text:
        raise ValueError(f"registry source entry is missing path: {name}")

    resolved = resolve_under_root(root, path_text)
    exists = resolved.is_file()
    if exists:
        sha256 = stable_file_hash(resolved)
        schema_version, stage_status = read_json_metadata(resolved)
        status = "present"
    else:
        sha256 = None
        schema_version = None
        stage_status = None
        status = "missing_required" if required else "missing_optional"
        message = f"{status}:{name}:{path_text}"
        if required:
            blocking_findings.append(message)
        else:
            warnings.append(message)

    if entry.get("status") == "superseded" and not entry.get("replacement_path"):
        blocking_findings.append(f"superseded_source_missing_replacement:{name}")

    return SourceRecord(
        section=section,
        name=name,
        path=path_text,
        role=role,
        required=required,
        exists=exists,
        sha256=sha256,
        schema_version=schema_version,
        stage_status=stage_status,
        status=status,
    )


def write_source_inventory_table(path: Path, records: list[SourceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "section",
        "name",
        "path",
        "role",
        "required",
        "exists",
        "sha256",
        "schema_version",
        "stage_status",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: getattr(record, field) for field in fieldnames})


def write_source_artifact_references(output_dir: Path, records: list[SourceRecord]) -> None:
    for record in records:
        if record.section != "source_artifacts" or not record.exists:
            continue
        write_json(
            output_dir / "source_artifacts" / f"{record.name}.json",
            {
                "schema_version": "geosym_source_artifact_reference_v1",
                "name": record.name,
                "source_path": record.path,
                "role": record.role,
                "sha256": record.sha256,
                "source_schema_version": record.schema_version,
                "source_stage_status": record.stage_status,
            },
        )


def git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def write_manifest_markdown(path: Path, manifest: dict[str, Any], records: list[SourceRecord]) -> None:
    lines = [
        "# Paper Results Export Manifest",
        "",
        f"- Export status: `{manifest['export_status']}`",
        f"- Generated at: `{manifest['generated_at_utc']}`",
        f"- Export id: `{manifest['export_id']}`",
        f"- Blocking findings: `{len(manifest['blocking_findings'])}`",
        f"- Warnings: `{len(manifest['warnings'])}`",
        "",
        "## Sources",
        "",
        "| Section | Name | Required | Exists | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record.section} | `{record.name}` | {str(record.required).lower()} | "
            f"{str(record.exists).lower()} | `{record.status}` |"
        )
    if manifest["blocking_findings"]:
        lines.extend(["", "## Blocking Findings", ""])
        lines.extend(f"- `{item}`" for item in manifest["blocking_findings"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_paper_results(
    config_path: Path,
    output_dir: Path,
    export_id: str = "current",
    root: Path | None = None,
) -> dict[str, Any]:
    root = (root or repo_root_from_script()).resolve()
    config_path = config_path if config_path.is_absolute() else root / config_path
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir

    config = load_yaml(config_path)
    export_config, entries = iter_registry_sources(config, export_id)

    blocking_findings: list[str] = []
    warnings: list[str] = []
    records = [inspect_source(root, entry, blocking_findings, warnings) for entry in entries]

    export_status = "ready_for_review" if not blocking_findings else "blocked"
    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "registry_schema_version": config.get("schema_version"),
        "export_id": export_id,
        "export_status": export_status,
        "generated_at_utc": generated_at,
        "source_registry": str(config_path.relative_to(root)) if config_path.is_relative_to(root) else str(config_path),
        "git_revision": git_revision(root),
        "blocking_findings": blocking_findings,
        "warnings": warnings,
        "record_count": len(records),
        "present_record_count": sum(1 for record in records if record.exists),
        "missing_required_count": sum(1 for record in records if record.status == "missing_required"),
        "tables": export_config.get("tables", {}),
        "claims": export_config.get("claims", {}),
    }

    hashes = {
        record.name: {
            "path": record.path,
            "exists": record.exists,
            "sha256": record.sha256,
            "required": record.required,
            "section": record.section,
        }
        for record in records
    }
    environment = {
        "schema_version": "geosym_export_environment_v1",
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }
    command_log = {
        "schema_version": "geosym_export_command_log_v1",
        "argv": sys.argv,
        "paid_calls_executed": 0,
        "experiment_reruns_executed": 0,
    }
    model_provenance = {
        "schema_version": "geosym_export_model_provenance_v1",
        "paid_model_calls_executed_by_exporter": 0,
        "source_model_metadata_policy": "read from frozen source artifacts only",
    }
    claim_matrix = {
        "schema_version": "geosym_claim_matrix_v1",
        "status": "registry_initialized_pending_claim_population",
        "claims": [],
    }
    audit = {
        "schema_version": "geosym_export_audit_v1",
        "stage_status": "paper_results_export_ready_for_review" if not blocking_findings else "paper_results_export_blocked",
        "blocking_findings": blocking_findings,
        "warnings": warnings,
        "offline_export_only": True,
        "paid_calls_executed": 0,
        "experiment_reruns_executed": 0,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_source_inventory_table(output_dir / "tables" / "source_artifacts.csv", records)
    write_source_artifact_references(output_dir, records)
    write_json(output_dir / "provenance" / "artifact_hashes.json", hashes)
    write_json(output_dir / "provenance" / "environment.json", environment)
    write_json(output_dir / "provenance" / "command_log.json", command_log)
    write_json(output_dir / "provenance" / "model_provenance.json", model_provenance)
    write_json(output_dir / "claims" / "claim_matrix.json", claim_matrix)
    write_json(output_dir / "export_audit.json", audit)
    write_json(output_dir / "manifest.json", manifest)
    write_manifest_markdown(output_dir / "MANIFEST.md", manifest, records)

    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/paper_results_sources.yaml"))
    parser.add_argument("--export-id", default="current")
    parser.add_argument("--output", type=Path, default=Path("outputs/paper_results/current"))
    parser.add_argument("--strict", action="store_true", help="return a non-zero exit code when blocking findings exist")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = export_paper_results(args.config, args.output, export_id=args.export_id)
    print(json.dumps({"export_status": manifest["export_status"], "blocking_findings": manifest["blocking_findings"]}, indent=2))
    if args.strict and manifest["blocking_findings"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
