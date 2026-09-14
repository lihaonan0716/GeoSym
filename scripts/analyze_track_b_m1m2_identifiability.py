#!/usr/bin/env python
"""Analyze whether repaired Track B can identify an M1/M2 effect.

This script performs only zero-cost local analysis. It joins the repaired
Track B E5/E5_raw rows with a local clean-grid fallback M0 quality audit over
the frozen 240-sample roster. Ground truth is used only after local extraction
for M0 diagnostic scoring; it is not passed to the extractor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognizer.core.clean_grid_fallback import extract_clean_cartesian_grid  # noqa: E402

OUTPUT_ROOT = ROOT / "outputs" / "paper_level_evidence_package"
TRACK_B_FINAL = OUTPUT_ROOT / "geosym_bench_controlled" / "post_remediation_e5_backfill_full_240_final_analysis.json"
TRACK_B_DEDUP = OUTPUT_ROOT / "geosym_bench_controlled" / "metadata_dedup_audit.json"
OUTPUT_PATH = OUTPUT_ROOT / "geosym_bench_controlled" / "track_b_m1m2_identifiability_analysis.json"
M0_FULL_QUALITY_PATH = OUTPUT_ROOT / "geosym_bench_controlled" / "clean_grid_fallback_m0_quality_full_240.json"
ALPHA = 0.05
POWER = 0.80
HISTORICAL_M1M2_EFFECT = 0.19


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def image_name_for_sample(sample_id: str) -> str:
    if "-clean_render_" not in sample_id:
        raise ValueError(f"Cannot derive image path from sample_id={sample_id!r}")
    family = sample_id.split("-clean_render_", 1)[0]
    return f"{family}-clean_render/{sample_id}"


def load_ground_truth(image_name: str) -> dict[str, Any]:
    path = ROOT / "dataset" / "ground_truth" / f"{image_name}.json"
    data = load_json(path)
    return data.get("question", {}).get("ground_truth", {})


def normalize_points(points: Any) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for point in points or []:
        if isinstance(point, dict):
            x = point.get("x", point.get("coordinate_x", point.get("real_x")))
            y = point.get("y", point.get("coordinate_y", point.get("real_y")))
        else:
            x, y = point[:2]
        result.add((int(round(float(x))), int(round(float(y)))))
    return result


def rate(hit: int, denom: int) -> float | None:
    return round(hit / denom, 6) if denom else None


def summarize_counts(counts: dict[str, int]) -> dict[str, Any]:
    hit = counts["hit"]
    miss = counts["miss"]
    false = counts["false"]
    return {
        "hit": hit,
        "miss": miss,
        "false": false,
        "recall": rate(hit, hit + miss),
        "precision": rate(hit, hit + false),
    }


def audit_m0_quality(roster: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    total = {"hit": 0, "miss": 0, "false": 0}
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"hit": 0, "miss": 0, "false": 0})
    for item in roster:
        sample_id = str(item["sample_id"])
        family = str(item["family"])
        image_name = image_name_for_sample(sample_id)
        image_path = ROOT / "dataset" / "images" / f"{image_name}.png"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        extraction = extract_clean_cartesian_grid(image)
        predicted = set(extraction.coords)
        gt = normalize_points(load_ground_truth(image_name).get("coordinates"))
        hits = len(predicted & gt)
        misses = len(gt - predicted)
        false = len(predicted - gt)
        row = {
            "sample_id": sample_id,
            "family": family,
            "image_name": image_name,
            "success": extraction.success,
            "predicted_count": len(predicted),
            "ground_truth_count": len(gt),
            "hit": hits,
            "miss": misses,
            "false": false,
            "recall": rate(hits, hits + misses),
            "precision": rate(hits, hits + false),
        }
        rows.append(row)
        for key, value in (("hit", hits), ("miss", misses), ("false", false)):
            total[key] += value
            by_family[family][key] += value
    return {
        "schema_version": "clean_grid_fallback_m0_quality_full_240_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "track_b_clean_grid_fallback_m0_quality_full_roster",
        "stage_status": "completed_zero_cost_local_cv_diagnostic",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "sample_count": len(rows),
        "selection_rule": "Frozen Track B 240-sample roster from metadata_dedup_audit.json.",
        "ground_truth_usage": "Ground truth is used only after extraction for scoring; it is not passed to the extractor.",
        "overall": summarize_counts(total),
        "by_family": {family: summarize_counts(counts) for family, counts in sorted(by_family.items())},
        "results": rows,
        "interpretation_limits": [
            "This is a point-level M0 diagnostic, not a scored GeoSym-Bench dimension.",
            "Extractor success means a non-empty coordinate set was returned; it does not imply coordinate correctness.",
            "The clean-grid fallback remains scoped to clean synthetic Cartesian grid charts.",
        ],
    }


def paired_stats(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = sum(values) / n if n else None
    if n < 2 or mean is None:
        return {"n": n, "mean": mean, "sd": None, "se": None, "t": None, "p": None, "ci95": None, "cohen_dz": None}
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
    se = sd / math.sqrt(n)
    t_stat = mean / se if se else None
    p_value = float(stats.ttest_1samp(values, 0.0).pvalue)
    tcrit = float(stats.t.ppf(1 - ALPHA / 2, n - 1))
    return {
        "n": n,
        "mean": round(mean, 6),
        "sd": round(sd, 6),
        "se": round(se, 6),
        "t": round(t_stat, 6) if t_stat is not None else None,
        "p": round(p_value, 6),
        "ci95": [round(mean - tcrit * se, 6), round(mean + tcrit * se, 6)],
        "cohen_dz": round(mean / sd, 6) if sd else None,
    }


def normal_approx_mde(sd: float, n: int) -> float:
    z_alpha = float(stats.norm.ppf(1 - ALPHA / 2))
    z_power = float(stats.norm.ppf(POWER))
    return (z_alpha + z_power) * sd / math.sqrt(n)


def normal_approx_required_n(sd: float, effect: float) -> int:
    z_alpha = float(stats.norm.ppf(1 - ALPHA / 2))
    z_power = float(stats.norm.ppf(POWER))
    return math.ceil(((z_alpha + z_power) * sd / abs(effect)) ** 2)


def split_by_m0_precision(m0_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    sorted_rows = sorted(
        m0_rows,
        key=lambda row: (
            float(row["precision"]) if row.get("precision") is not None else -1.0,
            float(row["recall"]) if row.get("recall") is not None else -1.0,
            str(row["sample_id"]),
        ),
    )
    midpoint = len(sorted_rows) // 2
    return {
        "low_precision_half": {str(row["sample_id"]) for row in sorted_rows[:midpoint]},
        "high_precision_half": {str(row["sample_id"]) for row in sorted_rows[midpoint:]},
    }


def build_pairs(track_b_final: dict[str, Any]) -> list[dict[str, Any]]:
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in track_b_final.get("results") or []:
        if row.get("condition") in {"E5", "E5_raw"}:
            by_sample[str(row["sample_id"])][str(row["condition"])] = row
    pairs = []
    for sample_id, rows in sorted(by_sample.items()):
        e5 = rows.get("E5")
        raw = rows.get("E5_raw")
        if not e5 or not raw:
            continue
        dims = sorted(set((e5.get("dimension_scores") or {}).keys()) | set((raw.get("dimension_scores") or {}).keys()))
        dim_delta = {
            dim: float((e5.get("dimension_scores") or {}).get(dim, 0.0))
            - float((raw.get("dimension_scores") or {}).get(dim, 0.0))
            for dim in dims
        }
        pairs.append(
            {
                "sample_id": sample_id,
                "family": e5.get("family"),
                "e5_total": e5.get("total_score"),
                "e5_raw_total": raw.get("total_score"),
                "total_delta_e5_minus_e5_raw": float(e5.get("total_score")) - float(raw.get("total_score")),
                "dimension_delta_e5_minus_e5_raw": dim_delta,
                "dimension_vectors_identical": all(value == 0 for value in dim_delta.values()),
                "chain_text_sha256_equal": e5.get("chain_text_sha256") == raw.get("chain_text_sha256"),
            }
        )
    return pairs


def summarize_dimension_deltas(pairs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    dims = sorted({dim for pair in pairs for dim in pair["dimension_delta_e5_minus_e5_raw"].keys()})
    summary = {}
    for dim in dims:
        values = [float(pair["dimension_delta_e5_minus_e5_raw"].get(dim, 0.0)) for pair in pairs]
        stats_row = paired_stats(values)
        stats_row["nonzero_pairs"] = sum(1 for value in values if value != 0)
        summary[dim] = stats_row
    return summary


def summarize_groups(pairs: list[dict[str, Any]], groups: dict[str, set[str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, sample_ids in groups.items():
        subset = [pair for pair in pairs if pair["sample_id"] in sample_ids]
        result[name] = paired_stats([pair["total_delta_e5_minus_e5_raw"] for pair in subset])
    by_family: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        by_family[str(pair["family"])].append(pair["total_delta_e5_minus_e5_raw"])
    result["by_family"] = {family: paired_stats(values) for family, values in sorted(by_family.items())}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-b-final", type=Path, default=TRACK_B_FINAL)
    parser.add_argument("--dedup-audit", type=Path, default=TRACK_B_DEDUP)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--m0-output", type=Path, default=M0_FULL_QUALITY_PATH)
    args = parser.parse_args()

    track_b_final = load_json(args.track_b_final)
    dedup = load_json(args.dedup_audit)
    roster = dedup.get("selected_subset") or []
    if len(roster) != 240:
        raise ValueError(f"Expected 240 Track B roster samples, found {len(roster)}")

    m0_quality = audit_m0_quality(roster)
    pairs = build_pairs(track_b_final)
    if len(pairs) != 240:
        raise ValueError(f"Expected 240 E5/E5_raw pairs, found {len(pairs)}")

    m0_by_sample = {row["sample_id"]: row for row in m0_quality["results"]}
    for pair in pairs:
        m0 = m0_by_sample[pair["sample_id"]]
        pair["m0_precision"] = m0.get("precision")
        pair["m0_recall"] = m0.get("recall")
        pair["m0_hit"] = m0.get("hit")
        pair["m0_miss"] = m0.get("miss")
        pair["m0_false"] = m0.get("false")

    total_values = [pair["total_delta_e5_minus_e5_raw"] for pair in pairs]
    overall = paired_stats(total_values)
    sd = float(overall["sd"])
    mde = normal_approx_mde(sd, len(total_values))
    required_n = normal_approx_required_n(sd, HISTORICAL_M1M2_EFFECT)
    chain_audit = track_b_final.get("chain_text_reuse_audit") or {}
    group_sets = split_by_m0_precision(m0_quality["results"])
    group_summary = summarize_groups(pairs, group_sets)
    identifiability_blockers = []
    if chain_audit.get("judge_reuse_required_pairs") == 0:
        identifiability_blockers.append("track_b_no_judge_reuse_pairs")
    if mde > abs(HISTORICAL_M1M2_EFFECT):
        identifiability_blockers.append("track_b_min_detectable_effect_exceeds_historical_m1m2_effect")
    if overall.get("ci95") and overall["ci95"][0] < 0 < overall["ci95"][1]:
        identifiability_blockers.append("track_b_total_delta_ci_crosses_zero")

    payload = {
        "schema_version": "track_b_m1m2_identifiability_analysis_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "track_b_m1m2_identifiability_analysis",
        "stage_status": "track_b_m1m2_not_identifiable_from_repaired_rerun"
        if identifiability_blockers
        else "track_b_m1m2_identifiable_from_repaired_rerun",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "track_b_final": str(args.track_b_final.resolve()),
            "dedup_audit": str(args.dedup_audit.resolve()),
            "m0_full_quality": str(args.m0_output.resolve()),
        },
        "predeclared_mechanism_prediction": (
            "If reliability-aware M1/M2 selection is visible in this repaired Track B design, "
            "E5 minus E5_raw should improve more in the low-M0-precision half than in the "
            "high-M0-precision half."
        ),
        "chain_text_reuse_audit": chain_audit,
        "overall_total_delta": overall,
        "power_analysis": {
            "alpha_two_sided": ALPHA,
            "target_power": POWER,
            "observed_delta_sd": overall["sd"],
            "minimum_detectable_effect_points_normal_approx": round(mde, 6),
            "historical_m1m2_effect_points": HISTORICAL_M1M2_EFFECT,
            "required_pairs_for_historical_effect_normal_approx": required_n,
            "interpretation": (
                "At the observed paired-delta variance, the repaired 240-pair Track B rerun "
                "is underpowered for effects on the order of the historical M1/M2 delta."
            ),
        },
        "m0_quality": {
            "sample_count": m0_quality["sample_count"],
            "overall": m0_quality["overall"],
            "by_family": m0_quality["by_family"],
            "stratification_rule": (
                "Sort by per-image M0 precision, then recall, then sample_id; lower 120 samples "
                "form low_precision_half and upper 120 form high_precision_half."
            ),
        },
        "stratified_total_delta": group_summary,
        "dimension_delta": summarize_dimension_deltas(pairs),
        "identical_dimension_vector_pairs": sum(1 for pair in pairs if pair["dimension_vectors_identical"]),
        "nonidentical_dimension_vector_pairs": sum(1 for pair in pairs if not pair["dimension_vectors_identical"]),
        "chain_text_equal_pairs": sum(1 for pair in pairs if pair["chain_text_sha256_equal"]),
        "identifiability_blockers": identifiability_blockers,
        "conclusion": (
            "Track B is valid as repaired end-to-end execution evidence, but it is not valid evidence "
            "for either a positive or negative M1/M2 mechanism effect because no E5/E5_raw pair triggers "
            "judge-score reuse and the detectable effect size is far above the historical M1/M2 delta."
            if identifiability_blockers
            else "Track B passes the configured identifiability checks."
        ),
        "interpretation_limits": [
            "Do not report the repaired Track B E5-E5_raw total-score delta as evidence that M1/M2 works.",
            "Do not report the repaired Track B E5-E5_raw total-score delta as evidence that M1/M2 does not work.",
            "Use Track B as end-to-end repaired pipeline evidence and use Track D as the primary controlled M1/M2 evidence.",
            "The M0 precision stratification is a zero-cost mechanism diagnostic; it does not repair the missing judge-reuse protection.",
        ],
        "paired_rows": pairs,
    }

    args.m0_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.m0_output.write_text(json.dumps(m0_quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.m0_output}")
    print(f"wrote {args.output}")
    print(
        json.dumps(
            {
                "stage_status": payload["stage_status"],
                "overall_total_delta": payload["overall_total_delta"],
                "power_analysis": payload["power_analysis"],
                "identifiability_blockers": payload["identifiability_blockers"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
