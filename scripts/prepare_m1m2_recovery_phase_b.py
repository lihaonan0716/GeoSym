#!/usr/bin/env python
"""Prepare Phase B preregistration and zero-cost dry run."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognizer.core.clean_grid_fallback import extract_clean_cartesian_grid  # noqa: E402
from scripts.analyze_track_b_m1m2_identifiability import normalize_points, load_ground_truth  # noqa: E402
from scripts.m1m2_recovery_quality_gradient import (  # noqa: E402
    QUALITY_LEVELS,
    assert_degradation_does_not_accept_ground_truth,
    degrade_coords,
)
from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402

RECOVERY_ROOT = ROOT / "outputs" / "m1m2_recovery"
FULL_M0_QUALITY = ROOT / "outputs" / "paper_level_evidence_package" / "geosym_bench_controlled" / "clean_grid_fallback_m0_quality_full_240.json"
PHASE_A = RECOVERY_ROOT / "secondary_metric_variance_study.json"
PREREG = RECOVERY_ROOT / "preregistration.json"
DRY_RUN = RECOVERY_ROOT / "phase_c_dry_run.json"
PILOT_SAMPLE_COUNT = 8
CONDITIONS = ["E5", "E5_raw"]
SEMANTIC_CALLS_BY_CONDITION = {
    "E5": 5,      # fact verification + L1 + L2 + L3 + judge
    "E5_raw": 4,  # L1 + L2 + L3 + judge
}
PILOT_LLM_CAP = PILOT_SAMPLE_COUNT * len(QUALITY_LEVELS) * sum(
    SEMANTIC_CALLS_BY_CONDITION[condition] for condition in CONDITIONS
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def select_pilot_roster(roster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in roster:
        by_family[str(row["family"])].append(row)
    return [sorted(rows, key=lambda row: row["sample_id"])[0] for _, rows in sorted(by_family.items())]


def quality_counts(points: list[tuple[int, int]], gt_points: set[tuple[int, int]]) -> dict[str, Any]:
    predicted = set(points)
    hit = len(predicted & gt_points)
    miss = len(gt_points - predicted)
    false = len(predicted - gt_points)
    return {
        "hit": hit,
        "miss": miss,
        "false": false,
        "recall": round(hit / (hit + miss), 6) if hit + miss else None,
        "precision": round(hit / (hit + false), 6) if hit + false else None,
    }


def run_degradation_dry_run(roster: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in select_pilot_roster(roster):
        image_path = ROOT / "dataset" / "images" / f"{item['image_name']}.png"
        image = cv2.imread(str(image_path))
        extraction = extract_clean_cartesian_grid(image)
        gt = normalize_points(load_ground_truth(item["image_name"]).get("coordinates"))
        last_recall = None
        for level in QUALITY_LEVELS:
            degraded = degrade_coords(extraction.coords, sample_id=item["sample_id"], level=level["level"])
            quality = quality_counts(degraded, gt)
            rows.append({
                "sample_id": item["sample_id"],
                "family": item["family"],
                "quality_level": level["level"],
                "native_point_count": len(extraction.coords),
                "degraded_point_count": len(degraded),
                "m0_quality": quality,
                "recall_nonincreasing_vs_previous": last_recall is None or quality["recall"] <= last_recall,
            })
            last_recall = quality["recall"]
    return {
        "schema_version": "m1m2_recovery_phase_c_dry_run_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "m1m2_recovery_phase_c_dry_run",
        "stage_status": "phase_c_dry_run_passed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "sample_count": PILOT_SAMPLE_COUNT,
        "quality_levels": QUALITY_LEVELS,
        "conditions": CONDITIONS,
        "semantic_calls_by_condition": SEMANTIC_CALLS_BY_CONDITION,
        "planned_pilot_llm_cap": PILOT_LLM_CAP,
        "rows": rows,
        "quality_gradient_verified": all(row["recall_nonincreasing_vs_previous"] for row in rows),
    }


def main() -> int:
    assert_degradation_does_not_accept_ground_truth()
    phase_a = load_json(PHASE_A)
    m0 = load_json(FULL_M0_QUALITY)
    roster = m0.get("results") or []
    selected_metric = phase_a["selected_phase_c_primary_metric"]
    selected_summary = phase_a["candidate_metrics"][selected_metric]
    dry_run = run_degradation_dry_run(roster)
    prereg = {
        "schema_version": "m1m2_recovery_preregistration_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "m1m2_recovery_phase_b_preregistration",
        "stage_status": "phase_b_preregistered_pending_phase_c_cap",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "hypotheses": {
            "H1": "M1+M2 utility increases monotonically as input-side M0 evidence quality decreases.",
            "H0": "E5 minus E5_raw is independent of input-side M0 evidence quality.",
        },
        "primary_test": "Trend test/regression slope of paired delta against continuous measured M0 quality, with family-clustered uncertainty.",
        "falsification_signals": [
            "Positive slope with corrected confidence interval excluding zero supports conditional utility.",
            "Confidence interval includes zero and upper bound is below the minimum meaningful effect falsifies H1 with an upper bound.",
            "Confidence interval includes zero and remains wide means the experiment is still not identifiable.",
        ],
        "minimum_meaningful_effect": {
            "metric": selected_metric,
            "effect_points": 0.19,
            "phase_a_mde_80_power": selected_summary["mde_80_power"],
            "phase_a_required_pairs": selected_summary["required_pairs_for_0_19"],
        },
        "roster": {
            "source": str(FULL_M0_QUALITY.resolve()),
            "sha256": stable_file_hash(FULL_M0_QUALITY),
            "sample_count": len(roster),
        },
        "quality_levels": QUALITY_LEVELS,
        "degradation_operator": {
            "function": "scripts.m1m2_recovery_quality_gradient.degrade_coords",
            "seed": 20260911,
            "fairness_constraint": "The operator receives only detected coordinates, sample_id, level, and seed; it does not receive ground truth.",
        },
        "conditions": CONDITIONS,
        "cap": {
            "conservative_default": "pilot_only_until_new_global_cap_is_explicitly_set",
            "pilot_sample_level_jobs": PILOT_SAMPLE_COUNT,
            "pilot_llm_http_attempt_cap": PILOT_LLM_CAP,
            "pilot_ocr_http_attempt_cap": 0,
            "full_required_pairs_from_phase_a": selected_summary["required_pairs_for_0_19"],
        },
        "stopping_rule": "Run only the 8-sample pilot under the current cap; do not start full Phase C without an explicit new global cap.",
        "source_artifacts": {
            "phase_a": {"path": str(PHASE_A.resolve()), "sha256": stable_file_hash(PHASE_A)},
            "dry_run": {"path": str(DRY_RUN.resolve())},
        },
    }
    write_json(DRY_RUN, dry_run)
    prereg["source_artifacts"]["dry_run"]["sha256"] = stable_file_hash(DRY_RUN)
    write_json(PREREG, prereg)
    print(f"wrote {DRY_RUN}")
    print(f"wrote {PREREG}")
    print(json.dumps({"stage_status": prereg["stage_status"], "pilot_llm_cap": PILOT_LLM_CAP, "quality_gradient_verified": dry_run["quality_gradient_verified"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
