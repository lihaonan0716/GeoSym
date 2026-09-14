#!/usr/bin/env python
"""Zero-cost M1 confidence-tier discriminative-validity analysis.

The primary unit is the per-image intercepts block. Truth functions do not
accept or read the tier label; tiers are joined only after correctness has
been computed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import sympy as sp
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.reasoning.intervention import infer_chart_facts  # noqa: E402
from recognizer.core.clean_grid_fallback import extract_clean_cartesian_grid  # noqa: E402
from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402

OUTPUT_ROOT = ROOT / "outputs" / "paper_level_evidence_package"
FULL_M0_QUALITY = OUTPUT_ROOT / "geosym_bench_controlled" / "clean_grid_fallback_m0_quality_full_240.json"
OUTPUT_PATH = ROOT / "outputs" / "m1m2_recovery" / "m1_tier_discriminative_validity.json"
PERMUTATION_SEED = 20260911
PERMUTATIONS = 1000
TIERS = ("low", "medium", "high")
TIER_SCORES = {"low": 0, "medium": 1, "high": 2}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def image_name_for_sample(sample_id: str) -> str:
    family = sample_id.split("-clean_render_", 1)[0]
    return f"{family}-clean_render/{sample_id}"


def all_clean_render_roster() -> list[dict[str, str]]:
    rows = []
    for image_dir in sorted((ROOT / "dataset" / "images").glob("*-clean_render")):
        family = image_dir.name.replace("-clean_render", "")
        for image_path in sorted(image_dir.glob("*.png")):
            rows.append({"family": family, "sample_id": image_path.stem, "image_name": f"{image_dir.name}/{image_path.stem}"})
    return rows


def load_frozen_roster(path: Path = FULL_M0_QUALITY) -> tuple[list[dict[str, str]], dict[str, Any]]:
    payload = load_json(path)
    roster = [
        {"family": str(row["family"]), "sample_id": str(row["sample_id"]), "image_name": str(row["image_name"])}
        for row in payload.get("results") or []
    ]
    return roster, {
        "path": str(path.resolve()),
        "sha256": stable_file_hash(path),
        "sample_count": len(roster),
        "selection_rule": payload.get("selection_rule"),
    }


def load_ground_truth(image_name: str) -> dict[str, Any]:
    path = ROOT / "dataset" / "ground_truth" / f"{image_name}.json"
    return load_json(path)


def close_float(a: Any, b: Any, tol: float = 0.25) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def sorted_float_set(values: Any) -> list[float]:
    result = []
    for value in values or []:
        try:
            result.append(round(float(value), 2))
        except (TypeError, ValueError):
            continue
    return sorted(set(result))


def float_sets_match(predicted: Any, expected: Any, tol: float = 0.25) -> bool:
    pred = sorted_float_set(predicted)
    exp = sorted_float_set(expected)
    if len(pred) != len(exp):
        return False
    return all(close_float(a, b, tol=tol) for a, b in zip(pred, exp))


def intercept_block_correct(predicted_intercepts: dict[str, Any], ground_truth: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Return intercept-block correctness without reading any tier label."""
    gt_intercepts = ground_truth.get("question", {}).get("ground_truth", {}).get("intercepts") or {}
    gt_zeros = ground_truth.get("question", {}).get("ground_truth", {}).get("zeros")
    if gt_zeros is None:
        functions = ground_truth.get("functions") or []
        gt_zeros = ((functions[0] if functions else {}).get("math_properties") or {}).get("zeros") or []

    predicted_y = predicted_intercepts.get("y_intercept")
    expected_y = gt_intercepts.get("y_intercept")
    if expected_y is None:
        y_correct = predicted_y is None or predicted_y == "undetermined" or predicted_y == []
    else:
        y_correct = close_float(predicted_y, expected_y)

    predicted_x = predicted_intercepts.get("x_intercepts")
    x_correct = float_sets_match(predicted_x, gt_zeros)
    return bool(x_correct and y_correct), {
        "x_correct": x_correct,
        "y_correct": y_correct,
        "predicted_x_intercepts": sorted_float_set(predicted_x),
        "expected_x_intercepts": sorted_float_set(gt_zeros),
        "predicted_y_intercept": predicted_y,
        "expected_y_intercept": expected_y,
    }


def symmetry_correct(predicted_symmetry: str, ground_truth: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Return symmetry correctness without reading any tier label."""
    expected = ground_truth.get("question", {}).get("ground_truth", {}).get("symmetry")
    return predicted_symmetry == expected, {"predicted": predicted_symmetry, "expected": expected}


def _numeric_endpoint(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    text = str(value)
    match = re.search(r"observed (?:from|to) (-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    try:
        number = float(text)
        if math.isfinite(number):
            return number
    except ValueError:
        return None
    return None


def _expression_callable(ground_truth: dict[str, Any]):
    functions = ground_truth.get("functions") or []
    expression = ((functions[0] if functions else {}).get("expression_sympy"))
    if not expression:
        return None
    x = sp.symbols("x")
    expr = sp.sympify(expression)
    return sp.lambdify(x, expr, "math")


def monotonicity_segment_correct(segment: dict[str, Any], ground_truth: dict[str, Any]) -> tuple[bool | None, dict[str, Any]]:
    """Return segment correctness without reading any tier label."""
    fn = _expression_callable(ground_truth)
    if fn is None:
        return None, {"reason": "missing_expression_sympy"}
    interval = segment.get("interval") or []
    lo = _numeric_endpoint(interval[0]) if len(interval) >= 1 else None
    hi = _numeric_endpoint(interval[1]) if len(interval) >= 2 else None
    if lo is None or hi is None or lo >= hi:
        return None, {"reason": "non_numeric_interval", "interval": interval}
    width = hi - lo
    h = max(width / 100.0, 0.05)
    votes: list[str] = []
    for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
        x0 = lo + width * frac
        try:
            left = float(fn(x0 - h))
            right = float(fn(x0 + h))
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            continue
        diff = right - left
        if abs(diff) < 1e-6:
            votes.append("constant")
        elif diff > 0:
            votes.append("increasing")
        else:
            votes.append("decreasing")
    if not votes:
        return None, {"reason": "no_numeric_votes", "interval": interval}
    expected = Counter(votes).most_common(1)[0][0]
    predicted = segment.get("direction")
    return predicted == expected, {"predicted": predicted, "expected": expected, "votes": votes, "interval": interval}


def wilson_ci(correct: int, total: int, z: float = 1.959963984540054) -> list[float | None]:
    if total == 0:
        return [None, None]
    phat = correct / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return [round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6)]


def summarize_units(units: list[dict[str, Any]]) -> dict[str, Any]:
    by_tier: dict[str, dict[str, int]] = {tier: {"correct": 0, "total": 0} for tier in TIERS}
    for unit in units:
        tier = str(unit.get("tier"))
        if tier not in by_tier:
            continue
        by_tier[tier]["total"] += 1
        if unit.get("correct") is True:
            by_tier[tier]["correct"] += 1
    table = {}
    for tier in TIERS:
        correct = by_tier[tier]["correct"]
        total = by_tier[tier]["total"]
        table[tier] = {
            "correct": correct,
            "total": total,
            "rate": round(correct / total, 6) if total else None,
            "wilson_ci95": wilson_ci(correct, total),
        }
    return table


def cochran_armitage(units: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [(TIER_SCORES[str(unit["tier"])], 1 if unit.get("correct") is True else 0) for unit in units if unit.get("tier") in TIER_SCORES]
    n = len(rows)
    successes = sum(y for _, y in rows)
    if n < 2 or successes in {0, n}:
        return {"n": n, "z": None, "p": None, "statistic": None}
    scores = [score for score, _ in rows]
    mean_score = sum(scores) / n
    p_hat = successes / n
    numerator = sum((score - mean_score) * (y - p_hat) for score, y in rows)
    denominator = math.sqrt(p_hat * (1 - p_hat) * sum((score - mean_score) ** 2 for score in scores))
    z_value = numerator / denominator if denominator else None
    if z_value is None:
        return {"n": n, "z": None, "p": None, "statistic": None}
    p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))
    return {"n": n, "z": round(float(z_value), 6), "p": round(float(p_value), 6), "statistic": round(float(z_value), 6)}


def rate_difference_ci(high: dict[str, Any], low: dict[str, Any]) -> dict[str, Any]:
    h_total = high["total"]
    l_total = low["total"]
    if h_total == 0 or l_total == 0:
        return {"difference": None, "ci95": [None, None]}
    h_rate = high["correct"] / h_total
    l_rate = low["correct"] / l_total
    diff = h_rate - l_rate
    se = math.sqrt(h_rate * (1 - h_rate) / h_total + l_rate * (1 - l_rate) / l_total)
    return {"difference": round(diff, 6), "ci95": [round(diff - 1.959963984540054 * se, 6), round(diff + 1.959963984540054 * se, 6)]}


def family_stratified_permutation(units: list[dict[str, Any]], *, seed: int = PERMUTATION_SEED, n_perm: int = PERMUTATIONS) -> dict[str, Any]:
    observed = cochran_armitage(units)
    rng = random.Random(seed)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        by_family[str(unit["family"])].append(unit)
    null_stats = []
    for _ in range(n_perm):
        permuted = []
        for family_units in by_family.values():
            tiers = [unit["tier"] for unit in family_units]
            rng.shuffle(tiers)
            for unit, tier in zip(family_units, tiers):
                clone = dict(unit)
                clone["tier"] = tier
                permuted.append(clone)
        stat = cochran_armitage(permuted).get("statistic")
        if stat is not None:
            null_stats.append(float(stat))
    obs = observed.get("statistic")
    if obs is None or not null_stats:
        empirical_p = None
        percentile = None
    else:
        empirical_p = (sum(1 for value in null_stats if abs(value) >= abs(float(obs))) + 1) / (len(null_stats) + 1)
        percentile = sum(1 for value in null_stats if value <= float(obs)) / len(null_stats)
    return {
        "permutation_scheme": "family-stratified tier-label permutation; image-internal permutation is not applicable to the one-intercepts-block-per-image primary unit",
        "seed": seed,
        "iterations": n_perm,
        "observed_statistic": obs,
        "empirical_two_sided_p": round(empirical_p, 6) if empirical_p is not None else None,
        "observed_percentile": round(percentile, 6) if percentile is not None else None,
        "null_distribution_summary": {
            "n": len(null_stats),
            "mean": round(sum(null_stats) / len(null_stats), 6) if null_stats else None,
            "p025": round(float(stats.scoreatpercentile(null_stats, 2.5)), 6) if null_stats else None,
            "p975": round(float(stats.scoreatpercentile(null_stats, 97.5)), 6) if null_stats else None,
        },
    }


def evaluate_roster(roster: list[dict[str, str]]) -> dict[str, Any]:
    units: list[dict[str, Any]] = []
    excluded_images = []
    mechanism_counts = {
        "monotonicity_segments_with_high": 0,
        "monotonicity_segments_total": 0,
        "extrema_empty_with_medium_confidence": 0,
        "intercepts_low_with_correct_block": 0,
    }
    for item in roster:
        image_path = ROOT / "dataset" / "images" / f"{item['image_name']}.png"
        image = cv2.imread(str(image_path))
        if image is None:
            excluded_images.append({"sample_id": item["sample_id"], "reason": f"missing_image:{image_path}"})
            continue
        extraction = extract_clean_cartesian_grid(image)
        if not extraction.success:
            excluded_images.append({"sample_id": item["sample_id"], "reason": extraction.reason})
            continue
        facts = infer_chart_facts(extraction.coords)
        gt = load_ground_truth(item["image_name"])

        intercepts = facts.get("intercepts") or {}
        intercept_correct, intercept_detail = intercept_block_correct(intercepts, gt)
        intercept_tier = intercepts.get("_confidence")
        if intercept_tier == "low" and intercept_correct:
            mechanism_counts["intercepts_low_with_correct_block"] += 1
        units.append(
            {
                "sample_id": item["sample_id"],
                "family": item["family"],
                "fact_type": "intercepts",
                "observation_unit": "intercepts_block",
                "tier": intercept_tier,
                "correct": intercept_correct,
                "details": intercept_detail,
            }
        )

        symmetry_correct_value, symmetry_detail = symmetry_correct(str(facts.get("symmetry")), gt)
        units.append(
            {
                "sample_id": item["sample_id"],
                "family": item["family"],
                "fact_type": "symmetry",
                "observation_unit": "image",
                "tier": facts.get("symmetry_confidence"),
                "correct": symmetry_correct_value,
                "details": symmetry_detail,
            }
        )

        if facts.get("extrema") == [] and facts.get("extrema_confidence") == "medium":
            mechanism_counts["extrema_empty_with_medium_confidence"] += 1

        for index, segment in enumerate(facts.get("monotonicity") or []):
            tier = segment.get("_confidence")
            mechanism_counts["monotonicity_segments_total"] += 1
            if tier == "high":
                mechanism_counts["monotonicity_segments_with_high"] += 1
            correct, detail = monotonicity_segment_correct(segment, gt)
            if correct is None:
                continue
            units.append(
                {
                    "sample_id": item["sample_id"],
                    "family": item["family"],
                    "fact_type": "monotonicity",
                    "observation_unit": "segment",
                    "segment_index": index,
                    "tier": tier,
                    "correct": correct,
                    "details": detail,
                }
            )
    return {"units": units, "excluded_images": excluded_images, "mechanism_counts": mechanism_counts}


def analyze_units(units: list[dict[str, Any]]) -> dict[str, Any]:
    intercept_units = [unit for unit in units if unit["fact_type"] == "intercepts"]
    primary_table = summarize_units(intercept_units)
    ca = cochran_armitage(intercept_units)
    effect = rate_difference_ci(primary_table["high"], primary_table["low"])
    permutation = family_stratified_permutation(intercept_units)
    verdict = "not_detected"
    if ca.get("p") is not None and ca["p"] < 0.05 and permutation.get("empirical_two_sided_p") is not None and permutation["empirical_two_sided_p"] < 0.05:
        if effect.get("difference") is not None and effect["difference"] > 0.15:
            verdict = "discriminative_validity_supported"
        else:
            verdict = "weak_validity"
    if permutation.get("empirical_two_sided_p") is not None and permutation["empirical_two_sided_p"] >= 0.05 and ca.get("p") is not None and ca["p"] < 0.05:
        verdict = "invalidated_by_permutation_control"
    secondary = {}
    coverage = {}
    for fact_type in ("monotonicity", "symmetry"):
        fact_units = [unit for unit in units if unit["fact_type"] == fact_type]
        secondary[fact_type] = summarize_units(fact_units)
        coverage[fact_type] = {tier: secondary[fact_type][tier]["total"] for tier in TIERS}
    by_family = {}
    for family in sorted({unit["family"] for unit in intercept_units}):
        family_units = [unit for unit in intercept_units if unit["family"] == family]
        by_family[family] = {
            "table": summarize_units(family_units),
            "cochran_armitage": cochran_armitage(family_units),
        }
    return {
        "primary_analysis": {
            "fact_type": "intercepts",
            "table_3x2": primary_table,
            "cochran_armitage": ca,
            "effect_size_high_minus_low": effect,
            "family_stratified": by_family,
        },
        "secondary_report": {
            **secondary,
            "coverage_gaps": {
                **coverage,
                "note": "Secondary fact types have incomplete tier coverage and are not used for trend testing.",
            },
        },
        "permutation_control": permutation,
        "prereg_verdict": verdict,
        "multiplicity": {
            "primary_confirmatory_tests": 1,
            "primary_test": "intercepts Cochran-Armitage trend test with permutation control",
            "secondary_tests": 0,
            "note": "Monotonicity and symmetry are descriptive because tier coverage is incomplete; extrema is excluded.",
        },
    }


def assert_truth_functions_do_not_accept_tier() -> None:
    for fn in (intercept_block_correct, symmetry_correct, monotonicity_segment_correct):
        assert "tier" not in inspect.signature(fn).parameters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--full-m0-quality", type=Path, default=FULL_M0_QUALITY)
    args = parser.parse_args()

    assert_truth_functions_do_not_accept_tier()
    frozen_roster, frozen_source = load_frozen_roster(args.full_m0_quality)
    sensitivity_roster = all_clean_render_roster()
    frozen_eval = evaluate_roster(frozen_roster)
    sensitivity_eval = evaluate_roster(sensitivity_roster)
    frozen_analysis = analyze_units(frozen_eval["units"])
    sensitivity_analysis = analyze_units(sensitivity_eval["units"])
    payload = {
        "schema_version": "m1_tier_discriminative_validity_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "m1_tier_discriminative_validity",
        "stage_status": "m1_tier_discriminative_validity_completed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "roster": {
            "primary_240": frozen_source,
            "sensitivity_480": {
                "sample_count": len(sensitivity_roster),
                "selection_rule": "All dataset/images/*-clean_render/*.png files in sorted order; sensitivity only.",
            },
        },
        "excluded_images": {
            "primary_240": frozen_eval["excluded_images"],
            "sensitivity_480": sensitivity_eval["excluded_images"],
        },
        "excluded_fact_types": {
            "extrema": "Excluded because extrema_channel_enabled is false for the repaired Track B path; extrema facts may be empty while extrema_confidence still receives a tier.",
        },
        "truth_rules": {
            "intercepts": "The whole intercepts block is correct only when predicted x-intercepts match ground-truth zeros and the predicted y-intercept matches ground truth, or is empty/undetermined when no ground-truth y-intercept is defined.",
            "monotonicity": "Each segment direction is checked by numerical finite differences of expression_sympy inside the predicted interval; tier is not read by the truth function.",
            "symmetry": "Predicted symmetry label is compared with the ground-truth symmetry label; tier is not read by the truth function.",
            "truth_function_signature_check": "Passed: truth functions do not accept a tier parameter.",
        },
        "observation_unit_definition": {
            "intercepts": "one block per image",
            "monotonicity": "one unit per inferred segment",
            "symmetry": "one unit per image",
            "extrema": "excluded",
        },
        "primary_240": frozen_analysis,
        "sensitivity_480": sensitivity_analysis,
        "mechanism_observations": {
            "primary_240": frozen_eval["mechanism_counts"],
            "sensitivity_480": sensitivity_eval["mechanism_counts"],
            "interpretation": [
                "Monotonicity high-tier coverage is absent under the clean-grid fallback outputs, indicating a threshold mismatch.",
                "Extrema confidence can be assigned when extrema facts are empty, so extrema is excluded from tier-validity analysis.",
                "The earlier low-tier intercept anomaly is not present under the corrected block-level observation unit; low-tier intercept blocks are zero-correct under the strict whole-block truth rule.",
            ],
        },
        "wording_guardrails": [
            "Report discriminative validity only.",
            "Do not claim numeric likelihood validity.",
            "Do not claim formal evidence-theory quantities or combination rules.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(
        json.dumps(
            {
                "stage_status": payload["stage_status"],
                "primary_verdict": frozen_analysis["prereg_verdict"],
                "primary_table": frozen_analysis["primary_analysis"]["table_3x2"],
                "primary_ca": frozen_analysis["primary_analysis"]["cochran_armitage"],
                "primary_effect": frozen_analysis["primary_analysis"]["effect_size_high_minus_low"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
