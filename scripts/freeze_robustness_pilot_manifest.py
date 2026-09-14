#!/usr/bin/env python
"""Freeze the bounded paid-pilot manifest for the M1/M2 stress benchmark.

This script writes a manifest only. Safe default: no paid calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402

SCHEMA_VERSION = "m1m2_recovery_stress_paid_pilot_manifest_v1"
CONDITIONS = ["G_full_guarded", "G_all_evidence", "G_top_confidence", "G_random_k"]
SEED = 20260830


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def select_stress_roster(stress_benchmark: dict[str, Any], sample_cap: int) -> list[dict[str, Any]]:
    jobs = sorted(stress_benchmark.get("jobs", []), key=lambda row: str(row["stress_id"]))
    chosen = jobs[:sample_cap]
    return [
        {
            "stress_id": row["stress_id"],
            "source_sample_id": row["source_sample_id"],
            "perturbation_type": row["perturbation_type"],
            "seed": row["seed"],
            "stress_hash": (row.get("stress_schema") or {}).get("stress_hash"),
        }
        for row in chosen
    ]


def build_manifest(stress_benchmark_path: Path, output_dir: Path, sample_cap: int = 8) -> dict[str, Any]:
    stress = load_json(stress_benchmark_path)
    stress_ready = stress.get("stage_status") == "stress_benchmark_frozen"
    roster = select_stress_roster(stress, sample_cap) if stress_ready else []
    perturbation_counts = Counter(row["perturbation_type"] for row in roster)
    llm_cap = sample_cap * len(CONDITIONS)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "p5b_m1m2_recovery_stress_paid_pilot_manifest",
        "stage_status": "stress_paid_pilot_manifest_frozen" if len(roster) == sample_cap else "stress_paid_pilot_manifest_failed_closed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "hypothesis": (
            "On frozen perturbed evidence schemas, gap-aware guarded M1/M2 reduces harmful propagation "
            "relative to all-evidence, top-confidence, and random-k controls without lowering final-answer accuracy."
        ),
        "falsification_signal": (
            "If G_full_guarded has lower accuracy than any control, or fails to reduce harmful propagation "
            "against both G_all_evidence and G_top_confidence by at least one case, the stress pilot does not "
            "support an independent M1/M2 performance contribution."
        ),
        "sample_level_job_cap": sample_cap,
        "conditions": CONDITIONS,
        "llm_http_attempt_cap": llm_cap,
        "ocr_http_attempt_cap": 0,
        "retry_cap_per_sample_condition": 1,
        "decoding": "config-resolved model default with temperature 0 when supported",
        "random_seed": SEED,
        "stopping_rule": "Stop after the frozen stress jobs regardless of interim success; no automatic expansion.",
        "safe_default": "dry_run_no_paid_calls",
        "source_artifacts": {
            "stress_benchmark": {
                "path": str(stress_benchmark_path.resolve()),
                "sha256": stable_file_hash(stress_benchmark_path),
                "stage_status": stress.get("stage_status"),
            }
        },
        "selected_roster": roster,
        "selected_perturbation_counts": dict(sorted(perturbation_counts.items())),
        "go_gate": {
            "stress_benchmark_ready": stress_ready,
            "roster_size_equals_cap": len(roster) == sample_cap,
            "llm_cap_equals_32": llm_cap == 32,
            "ocr_cap_zero": True,
            "paid_calls_zero_in_manifest": True,
            "answer_independent_selection": all(row.get("answer_independent_selection") for row in stress.get("jobs", [])[:sample_cap]),
        },
        "next_required_action": "execute_stress_paid_pilot_dry_run" if len(roster) == sample_cap else "repair_stress_benchmark_before_paid_manifest",
        "interpretation_limits": [
            "This manifest freezes an 8-job stress pilot but does not execute it.",
            "The selected roster is deterministic by sorted stress_id and answer-independent.",
            "No sample replacement, prompt editing after outputs, or automatic expansion is permitted.",
            "The stress pilot is diagnostic evidence and cannot be reported as natural PlotQA performance.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "frozen_manifest.md").write_text(render_markdown(manifest), encoding="utf-8")
    return manifest


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Recovery Stress Paid Pilot Manifest",
        "",
        f"- Stage status: `{manifest['stage_status']}`",
        "- Paid calls executed: 0",
        f"- Sample-level job cap: {manifest['sample_level_job_cap']}",
        f"- Conditions: {', '.join(manifest['conditions'])}",
        f"- LLM HTTP attempt cap: {manifest['llm_http_attempt_cap']}",
        f"- OCR HTTP attempt cap: {manifest['ocr_http_attempt_cap']}",
        f"- Selected roster: {len(manifest['selected_roster'])}",
        "",
        "## Perturbations",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in manifest["selected_perturbation_counts"].items())
    lines.extend(["", "## Go Gate", ""])
    lines.extend(f"- {key}: {value}" for key, value in manifest["go_gate"].items())
    lines.extend(["", "## Next Required Action", "", f"- `{manifest['next_required_action']}`", "", "## Limits", ""])
    lines.extend(f"- {line}" for line in manifest["interpretation_limits"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-benchmark", default="outputs/m1m2_recovery/stress_benchmark/stress_benchmark.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_recovery/stress_paid_pilot_manifest")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest = build_manifest(Path(args.stress_benchmark), Path(args.output_dir))
    print(f"wrote stress paid pilot manifest: {Path(args.output_dir) / 'frozen_manifest.json'}")
    print(f"stage_status={manifest['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
