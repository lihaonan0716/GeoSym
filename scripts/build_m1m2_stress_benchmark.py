#!/usr/bin/env python
"""Build the frozen zero-cost M1/M2 stress benchmark artifact."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_public_benchmark_pilot import stable_file_hash, stable_payload_hash  # noqa: E402

SCHEMA_VERSION = "m1m2_recovery_stress_benchmark_v1"
SEED = 20260830
PERTURBATION_PLAN = [
    ("missing_required_fact", 6),
    ("spurious_fact", 4),
    ("ruler_perturbation", 4),
    ("series_ambiguity", 4),
    ("axis_notation_ambiguity", 3),
    ("visual_artifact_or_extrema_ambiguity", 3),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def first_candidate(row: dict[str, Any]) -> dict[str, Any]:
    candidates = row.get("evidence_candidates") or []
    return copy.deepcopy(candidates[0]) if candidates else {}


def perturb(row: dict[str, Any], perturbation_type: str, index: int) -> dict[str, Any]:
    modified = copy.deepcopy(row)
    candidates = modified.setdefault("evidence_candidates", [])
    gaps = modified.setdefault("evidence_gaps", [])
    selected = modified.setdefault("selected_evidence_set", {})
    selected_ids = list(selected.get("selected_fact_ids") or [])
    if perturbation_type == "missing_required_fact":
        removed = selected_ids[:1] or [candidates[0]["fact_id"]] if candidates else []
        candidates[:] = [candidate for candidate in candidates if candidate.get("fact_id") not in set(removed)]
        selected["selected_fact_ids"] = [fact_id for fact_id in selected_ids if fact_id not in set(removed)]
        gaps.append(
            {
                "gap_id": f"stress_missing_{index:03d}",
                "key": "measured_value",
                "needed_for_question": True,
                "reason": "stress_removed_required_fact",
                "recovery_action": "insufficient_evidence",
                "severity": "blocking",
            }
        )
    elif perturbation_type == "spurious_fact":
        candidate = first_candidate(row)
        candidate.update({"fact_id": f"stress_spurious_{index:03d}", "reliability_tier": "low", "status": "present"})
        value = candidate.setdefault("value", {})
        if isinstance(value.get("y_value"), (int, float)):
            value["y_value"] = float(value["y_value"]) * 1.25 + 1
        candidates.append(candidate)
    elif perturbation_type == "ruler_perturbation":
        gaps.append(
            {
                "gap_id": f"stress_ruler_{index:03d}",
                "key": "axis_scale",
                "needed_for_question": True,
                "reason": "stress_axis_scale_perturbation",
                "recovery_action": "verify_from_image",
                "severity": "blocking",
            }
        )
    elif perturbation_type == "series_ambiguity":
        candidate = first_candidate(row)
        candidate.update({"fact_id": f"stress_series_ambiguous_{index:03d}", "reliability_tier": "low"})
        value = candidate.setdefault("value", {})
        value["series_label"] = "ambiguous duplicate"
        candidates.append(candidate)
        gaps.append(
            {
                "gap_id": f"stress_series_gap_{index:03d}",
                "key": "series_label",
                "needed_for_question": True,
                "reason": "stress_ambiguous_series_mapping",
                "recovery_action": "inspect_image",
                "severity": "blocking",
            }
        )
    elif perturbation_type == "axis_notation_ambiguity":
        gaps.append(
            {
                "gap_id": f"stress_axis_notation_{index:03d}",
                "key": "axis_scale",
                "needed_for_question": True,
                "reason": "stress_scientific_notation_or_unit_ambiguity",
                "recovery_action": "verify_from_image",
                "severity": "blocking",
            }
        )
    elif perturbation_type == "visual_artifact_or_extrema_ambiguity":
        candidate = first_candidate(row)
        candidate.update({"fact_id": f"stress_visual_artifact_{index:03d}", "reliability_tier": "low"})
        value = candidate.setdefault("value", {})
        if isinstance(value.get("y_value"), (int, float)):
            value["y_value"] = 0.0
        candidates.append(candidate)
        gaps.append(
            {
                "gap_id": f"stress_visual_gap_{index:03d}",
                "key": "measured_value",
                "needed_for_question": False,
                "reason": "stress_visual_artifact_or_extrema_ambiguity",
                "recovery_action": "pass_through",
                "severity": "warning",
            }
        )
    else:
        raise ValueError(f"unknown perturbation: {perturbation_type}")
    modified["stress_hash"] = stable_payload_hash(modified)
    return modified


def build_stress_benchmark(schema_outputs_path: Path, output_dir: Path) -> dict[str, Any]:
    schema_outputs = load_json(schema_outputs_path)
    source_rows = sorted(schema_outputs.get("outputs", []), key=lambda row: str(row["sample_id"]))
    jobs = []
    cursor = 0
    for perturbation_type, count in PERTURBATION_PLAN:
        for _ in range(count):
            if not source_rows:
                break
            source = source_rows[cursor % len(source_rows)]
            stress_row = perturb(source, perturbation_type, len(jobs))
            jobs.append(
                {
                    "stress_id": f"stress_{len(jobs):03d}",
                    "source": "plotqa_schema_outputs",
                    "source_sample_id": source["sample_id"],
                    "perturbation_type": perturbation_type,
                    "seed": SEED,
                    "answer_independent_selection": True,
                    "stress_schema": stress_row,
                }
            )
            cursor += 1
    counts = Counter(job["perturbation_type"] for job in jobs)
    plan_met = len(jobs) == 24 and all(counts[name] == expected for name, expected in PERTURBATION_PLAN)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_recovery_frozen_stress_benchmark",
        "stage_status": "stress_benchmark_frozen" if plan_met else "stress_roster_insufficient",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "seed": SEED,
        "source_artifacts": {
            "schema_outputs": {
                "path": str(schema_outputs_path.resolve()),
                "sha256": stable_file_hash(schema_outputs_path),
                "stage_status": schema_outputs.get("stage_status"),
            }
        },
        "sample_level_job_count": len(jobs),
        "perturbation_counts": dict(sorted(counts.items())),
        "jobs": jobs,
        "next_required_action": "freeze_stress_paid_pilot_manifest" if plan_met else "repair_stress_roster_sources",
        "interpretation_limits": [
            "This artifact creates perturbed evidence schemas only and performs no answer generation.",
            "Source rows are selected by sorted sample id and do not depend on answer correctness.",
            "Stress perturbations are diagnostic and must not be reported as natural PlotQA performance.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stress_benchmark.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "stress_benchmark.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Recovery Stress Benchmark",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed: 0",
        f"- Sample-level jobs: {result['sample_level_job_count']}",
        f"- Seed: {result['seed']}",
        "",
        "## Perturbations",
        "",
    ]
    for key, value in result["perturbation_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Next Required Action", "", f"- `{result['next_required_action']}`", "", "## Limits", ""])
    lines.extend(f"- {line}" for line in result["interpretation_limits"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-outputs", default="outputs/m1m2_recovery/schema_outputs/m1m2_schema_outputs.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_recovery/stress_benchmark")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_stress_benchmark(Path(args.schema_outputs), Path(args.output_dir))
    print(f"wrote stress benchmark: {Path(args.output_dir) / 'stress_benchmark.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
