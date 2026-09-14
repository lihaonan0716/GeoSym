#!/usr/bin/env python
"""Build the 48-job stress roster for M1/M2 selective propagation."""

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

SCHEMA_VERSION = "m1m2_selective_stress_roster_v1"
SEED = 20260830
PERTURBATION_PLAN = [
    ("missing_required_fact", 8),
    ("spurious_low_confidence_fact", 8),
    ("contradictory_duplicate_fact", 8),
    ("ruler_or_axis_scale_perturbation", 8),
    ("series_or_legend_ambiguity", 8),
    ("axis_notation_or_unit_ambiguity", 8),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def first_candidate(row: dict[str, Any]) -> dict[str, Any]:
    candidates = row.get("evidence_candidates") or []
    return copy.deepcopy(candidates[0]) if candidates else {}


def selected_fact_ids(row: dict[str, Any]) -> list[str]:
    selected = row.get("selected_evidence_set") or {}
    return list(selected.get("selected_fact_ids") or [])


def append_gap(row: dict[str, Any], gap_id: str, key: str, reason: str, severity: str = "blocking") -> None:
    row.setdefault("evidence_gaps", []).append(
        {
            "gap_id": gap_id,
            "key": key,
            "needed_for_question": severity == "blocking",
            "reason": reason,
            "recovery_action": "abstain" if severity == "blocking" else "pass_through",
            "severity": severity,
        }
    )
    selected = row.setdefault("selected_evidence_set", {})
    surfaced = list(selected.get("surfaced_gap_ids") or [])
    if gap_id not in surfaced:
        surfaced.append(gap_id)
    selected["surfaced_gap_ids"] = surfaced


def configure_route(row: dict[str, Any], route: str) -> None:
    row.setdefault("selected_evidence_set", {})["propagation_policy"] = route


def perturb(row: dict[str, Any], family: str, index: int) -> tuple[dict[str, Any], str]:
    modified = copy.deepcopy(row)
    candidates = modified.setdefault("evidence_candidates", [])
    selected = modified.setdefault("selected_evidence_set", {})
    selected_ids = selected_fact_ids(modified)
    expected_answerability = "answerable"
    if family == "missing_required_fact":
        removed = selected_ids[:1] or ([candidates[0]["fact_id"]] if candidates else [])
        removed_set = set(removed)
        candidates[:] = [candidate for candidate in candidates if candidate.get("fact_id") not in removed_set]
        selected["selected_fact_ids"] = [fact_id for fact_id in selected_ids if fact_id not in removed_set]
        append_gap(modified, f"selective_missing_{index:03d}", "measured_value", "selective_removed_required_fact")
        configure_route(modified, "abstain_insufficient_evidence")
        expected_answerability = "insufficient_evidence"
    elif family == "spurious_low_confidence_fact":
        candidate = first_candidate(row)
        candidate.update({"fact_id": f"selective_spurious_{index:03d}", "reliability_tier": "low", "status": "present"})
        value = candidate.setdefault("value", {})
        if isinstance(value.get("y_value"), (int, float)):
            value["y_value"] = float(value["y_value"]) * 1.25 + 1
        candidates.append(candidate)
        selected["suppressed_fact_ids"] = sorted(set((selected.get("suppressed_fact_ids") or []) + [candidate["fact_id"]]))
        configure_route(modified, "answer_with_selected_facts")
    elif family == "contradictory_duplicate_fact":
        candidate = first_candidate(row)
        candidate.update({"fact_id": f"selective_contradiction_{index:03d}", "reliability_tier": "medium", "status": "present"})
        value = candidate.setdefault("value", {})
        if isinstance(value.get("y_value"), (int, float)):
            value["y_value"] = float(value["y_value"]) * -1
        candidates.append(candidate)
        selected["suppressed_fact_ids"] = sorted(set((selected.get("suppressed_fact_ids") or []) + [candidate["fact_id"]]))
        append_gap(modified, f"selective_contradiction_gap_{index:03d}", "measured_value", "selective_contradictory_duplicate")
        configure_route(modified, "abstain_contradiction")
        expected_answerability = "contradiction"
    elif family == "ruler_or_axis_scale_perturbation":
        append_gap(modified, f"selective_axis_scale_{index:03d}", "axis_scale", "selective_axis_scale_perturbation")
        configure_route(modified, "abstain_insufficient_evidence")
        expected_answerability = "insufficient_evidence"
    elif family == "series_or_legend_ambiguity":
        candidate = first_candidate(row)
        candidate.update({"fact_id": f"selective_series_ambiguous_{index:03d}", "reliability_tier": "low"})
        candidate.setdefault("value", {})["series_label"] = "ambiguous duplicate"
        candidates.append(candidate)
        append_gap(modified, f"selective_series_gap_{index:03d}", "series_label", "selective_ambiguous_series_mapping")
        configure_route(modified, "abstain_insufficient_evidence")
        expected_answerability = "insufficient_evidence"
    elif family == "axis_notation_or_unit_ambiguity":
        append_gap(modified, f"selective_axis_notation_{index:03d}", "axis_scale", "selective_axis_notation_or_unit_ambiguity")
        configure_route(modified, "abstain_insufficient_evidence")
        expected_answerability = "insufficient_evidence"
    else:
        raise ValueError(f"unknown perturbation family: {family}")
    modified["expected_answerability"] = expected_answerability
    modified["stress_hash"] = stable_payload_hash(modified)
    return modified, expected_answerability


def build_roster(schema_outputs_path: Path, output_dir: Path) -> dict[str, Any]:
    schema_outputs = load_json(schema_outputs_path)
    source_rows = sorted(schema_outputs.get("outputs", []), key=lambda row: str(row["sample_id"]))
    jobs = []
    cursor = 0
    for family, count in PERTURBATION_PLAN:
        for _ in range(count):
            if cursor >= len(source_rows):
                break
            source = source_rows[cursor]
            stress_schema, expected = perturb(source, family, len(jobs))
            jobs.append(
                {
                    "stress_id": f"selective_stress_{len(jobs):03d}",
                    "source": "plotqa_schema_outputs",
                    "source_sample_id": source["sample_id"],
                    "perturbation_family": family,
                    "seed": SEED,
                    "answer_independent_selection": True,
                    "expected_answerability": expected,
                    "stress_schema": stress_schema,
                }
            )
            cursor += 1
    family_counts = Counter(job["perturbation_family"] for job in jobs)
    answerability_counts = Counter(job["expected_answerability"] for job in jobs)
    plan_met = len(jobs) == 48 and all(family_counts[name] == expected for name, expected in PERTURBATION_PLAN)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_selective_stress_roster",
        "stage_status": "selective_stress_roster_frozen" if plan_met else "stress_roster_insufficient",
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
        "perturbation_family_counts": dict(sorted(family_counts.items())),
        "expected_answerability_counts": dict(sorted(answerability_counts.items())),
        "jobs": jobs,
        "next_required_action": "run_selective_zero_cost_projection" if plan_met else "repair_selective_stress_roster_sources",
        "interpretation_limits": [
            "This roster is generated from sorted source ids and does not inspect model outputs.",
            "Expected answerability is derived from deterministic perturbation operations.",
            "No sample replacement or manual labeling is allowed.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selective_stress_roster.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "selective_stress_roster.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Selective Propagation Stress Roster",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed: 0",
        f"- Sample-level jobs: {result['sample_level_job_count']}",
        f"- Seed: {result['seed']}",
        "",
        "## Perturbation Families",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in result["perturbation_family_counts"].items())
    lines.extend(["", "## Expected Answerability", ""])
    lines.extend(f"- {key}: {value}" for key, value in result["expected_answerability_counts"].items())
    lines.extend(["", "## Next Required Action", "", f"- `{result['next_required_action']}`", "", "## Limits", ""])
    lines.extend(f"- {line}" for line in result["interpretation_limits"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-outputs", default="outputs/m1m2_recovery/schema_outputs/m1m2_schema_outputs.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_selective_propagation/stress_roster")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = build_roster(Path(args.schema_outputs), Path(args.output_dir))
    print(f"wrote selective stress roster: {Path(args.output_dir) / 'selective_stress_roster.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
