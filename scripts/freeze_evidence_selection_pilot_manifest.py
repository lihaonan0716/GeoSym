#!/usr/bin/env python
"""Freeze the bounded paid-pilot manifest for M1/M2 recovery.

This P5 script creates a manifest only. It does not execute paid calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402

SCHEMA_VERSION = "m1m2_recovery_paid_pilot_manifest_v1"
CONDITIONS = ["G_full_guarded", "G_all_evidence", "G_top_confidence", "G_random_k"]
SEED = 20260830


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def select_roster(schema_outputs: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    outputs = sorted(schema_outputs.get("outputs", []), key=lambda row: str(row["sample_id"]))
    chosen = []
    for row in outputs:
        policy = (row.get("selected_evidence_set") or {}).get("propagation_policy")
        if policy in {"insufficient_evidence", "use_with_caution", "solve_deterministically"}:
            chosen.append(row)
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        chosen_ids = {row["sample_id"] for row in chosen}
        for row in outputs:
            if row["sample_id"] not in chosen_ids:
                chosen.append(row)
            if len(chosen) >= limit:
                break
    return [
        {
            "sample_id": row["sample_id"],
            "question_hash": row.get("question_hash"),
            "propagation_policy": (row.get("selected_evidence_set") or {}).get("propagation_policy"),
            "selected_fact_count": len((row.get("selected_evidence_set") or {}).get("selected_fact_ids") or []),
            "suppressed_fact_count": len((row.get("selected_evidence_set") or {}).get("suppressed_fact_ids") or []),
            "surfaced_gap_count": len((row.get("selected_evidence_set") or {}).get("surfaced_gap_ids") or []),
        }
        for row in chosen
    ]


def build_manifest(schema_outputs_path: Path, projection_path: Path, output_dir: Path, sample_cap: int = 8) -> dict[str, Any]:
    schema_outputs = load_json(schema_outputs_path)
    projection = load_json(projection_path)
    projection_ready = projection.get("stage_status") == "plotqa_guardrail_preserved"
    schema_ready = schema_outputs.get("stage_status") == "ready_for_zero_cost_projection"
    roster = select_roster(schema_outputs, sample_cap) if projection_ready and schema_ready else []
    llm_cap = sample_cap * len(CONDITIONS)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "p5_m1m2_recovery_paid_pilot_manifest",
        "stage_status": "paid_pilot_manifest_frozen" if len(roster) == sample_cap else "paid_pilot_manifest_failed_closed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "hypothesis": (
            "Gap-aware reliability scoring and utility-aware propagation control reduce harmful evidence propagation "
            "without damaging the protected PlotQA measured-data result."
        ),
        "falsification_signal": (
            "If G_full_guarded fails the 48/50 PlotQA guardrail or does not reduce harmful propagation against "
            "G_all_evidence and G_top_confidence by at least one case, M1/M2 cannot be claimed as an independent "
            "performance contribution."
        ),
        "sample_level_job_cap": sample_cap,
        "conditions": CONDITIONS,
        "llm_http_attempt_cap": llm_cap,
        "ocr_http_attempt_cap": 0,
        "retry_cap_per_sample_condition": 1,
        "decoding": "config-resolved model default with temperature 0 when supported",
        "random_seed": SEED,
        "stopping_rule": "Stop after the frozen jobs regardless of interim success; no automatic expansion.",
        "safe_default": "dry_run_no_paid_calls",
        "source_artifacts": {
            "schema_outputs": {
                "path": str(schema_outputs_path.resolve()),
                "sha256": stable_file_hash(schema_outputs_path),
                "stage_status": schema_outputs.get("stage_status"),
            },
            "guarded_projection": {
                "path": str(projection_path.resolve()),
                "sha256": stable_file_hash(projection_path),
                "stage_status": projection.get("stage_status"),
            },
        },
        "selected_roster": roster,
        "go_gate": {
            "schema_outputs_ready": schema_ready,
            "plotqa_guardrail_preserved": projection_ready,
            "roster_size_equals_cap": len(roster) == sample_cap,
            "llm_cap_equals_32": llm_cap == 32,
            "ocr_cap_zero": True,
            "paid_calls_zero_in_manifest": True,
        },
        "next_required_action": "implement_executable_paid_pilot_runner" if len(roster) == sample_cap else "repair_zero_cost_prerequisites_before_paid_manifest",
        "interpretation_limits": [
            "This manifest freezes a bounded pilot but does not execute it.",
            "The selected roster is deterministic and answer-independent.",
            "No sample replacement or automatic expansion is permitted.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "frozen_manifest.md").write_text(render_markdown(manifest), encoding="utf-8")
    return manifest


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Recovery Paid Pilot Manifest",
        "",
        f"- Stage status: `{manifest['stage_status']}`",
        "- Paid calls executed: 0",
        f"- Sample-level job cap: {manifest['sample_level_job_cap']}",
        f"- Conditions: {', '.join(manifest['conditions'])}",
        f"- LLM HTTP attempt cap: {manifest['llm_http_attempt_cap']}",
        f"- OCR HTTP attempt cap: {manifest['ocr_http_attempt_cap']}",
        f"- Selected roster: {len(manifest['selected_roster'])}",
        "",
        "## Go Gate",
        "",
    ]
    for key, value in manifest["go_gate"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Next Required Action", "", f"- `{manifest['next_required_action']}`", "", "## Limits", ""])
    lines.extend(f"- {line}" for line in manifest["interpretation_limits"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-outputs", default="outputs/m1m2_recovery/schema_outputs/m1m2_schema_outputs.json")
    parser.add_argument("--guarded-projection", default="outputs/m1m2_recovery/guarded_zero_cost_projection/guarded_projection_result.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_recovery/paid_pilot_manifest")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest = build_manifest(Path(args.schema_outputs), Path(args.guarded_projection), Path(args.output_dir))
    print(f"wrote paid pilot manifest: {Path(args.output_dir) / 'frozen_manifest.json'}")
    print(f"stage_status={manifest['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
