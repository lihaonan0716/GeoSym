"""Generate and evaluate the P0-3 extrema mechanism validation set.

This is an offline, zero-API script. It generates new-seed / noisy /
transformed within-family charts, applies the frozen P0-2 extrema
KEEP/DROP rule, classifies validation examples, and writes an auditable
manifest. It must not tune thresholds or alter sampling after seeing results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator.benchmark import (  # noqa: E402
    PAGE_SIZE_POOL,
    PROFILE_REGISTRY,
    build_subset_config,
    generate_and_save_diverse_image,
    subset_id,
)
from generator.config_loader import load_config  # noqa: E402
from generator.rendering.curves import ALL_FUNCTION_TYPES  # noqa: E402
from evaluator.reasoning.intervention import (  # noqa: E402
    EXTREMA_FILTER_RULE_VERSION,
    classify_extrema_candidates,
)


BASE_SEED = 8811
IMAGES_PER_FAMILY = 8
PROFILES_PER_FAMILY = ("digital_mild",) * 4 + ("digital_hard",) * 4
MATCH_TOLERANCE = 0.5
MAX_GENERATION_ATTEMPTS_PER_SLOT = 80
TOTAL_VALIDATION_SAMPLES = 64
MIN_SAMPLES_PER_FAMILY = 1
MIN_DISTINCT_QUOTA_FAMILIES = 7
MAX_TOTAL_CANDIDATE_ATTEMPTS = 2200
QUOTA_DIVERSITY_RETRY_ATTEMPTS = 1
QUOTA_PROFILES = ("digital_mild", "digital_hard", "clean_render", "robustness_stress")
QUOTA_SOURCE_PROFILES = {
    "ordinary_true_extrema": (
        ("quadratic", "digital_mild"),
        ("quadratic", "digital_hard"),
        ("absolute", "digital_mild"),
        ("absolute", "digital_hard"),
        ("cubic", "digital_mild"),
        ("cubic", "digital_hard"),
    ),
    "true_extrema_large_gap_or_jump": (
        ("piecewise_linear", "clean_render"),
        ("piecewise_linear", "digital_mild"),
        ("piecewise_linear", "robustness_stress"),
    ),
    "discontinuity_or_asymptote_pseudo_extrema": (
        ("rational", "robustness_stress"),
        ("rational", "digital_hard"),
        ("rational", "digital_mild"),
    ),
    "no_extrema_negative": (
        ("linear", "digital_mild"),
        ("linear", "digital_hard"),
        ("exponential", "digital_mild"),
        ("exponential", "digital_hard"),
        ("logarithmic", "digital_mild"),
        ("logarithmic", "digital_hard"),
    ),
}
TAIL_SOURCE_PROFILES = tuple(
    (family, profile)
    for family in ALL_FUNCTION_TYPES
    for profile in QUOTA_PROFILES
)
OUTPUT_DIR = ROOT / "outputs" / "p0_3_extrema_validation"
SHIPPED_BENCHMARK_SEEDS = (42, 8801, 8802)
REFERENCE_DIRS = (
    ROOT / "dataset",
    ROOT / "outputs",
    ROOT / "reports",
)
SAMPLE_CATEGORY_QUOTAS = {
    "ordinary_true_extrema": 16,
    "true_extrema_large_gap_or_jump": 16,
    "discontinuity_or_asymptote_pseudo_extrema": 16,
    "no_extrema_negative": 16,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_truth_payload(ground_truth: dict[str, Any]) -> dict[str, Any] | None:
    question_truth = ground_truth.get("question", {}).get("ground_truth", {})
    functions = ground_truth.get("functions", [])
    if not functions or not question_truth:
        return None
    return {
        "functions": [
            {
                "type": function.get("type"),
                "parameters": function.get("parameters", {}),
                "expression": function.get("expression_latex"),
                "math_properties": function.get("math_properties", {}),
            }
            for function in functions
        ],
        "question_ground_truth": {
            "expression": question_truth.get("expression"),
            "domain": question_truth.get("domain"),
            "range": question_truth.get("range"),
            "coordinates": question_truth.get("coordinates", []),
            "extrema": question_truth.get("extrema", []),
        },
    }


def _semantic_truth_hash(ground_truth: dict[str, Any]) -> str | None:
    payload = _semantic_truth_payload(ground_truth)
    if payload is None:
        return None
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reference_hashes(excluded_output_dir: Path) -> dict[str, set[str]]:
    image_hashes: set[str] = set()
    truth_hashes: set[str] = set()
    for reference_dir in REFERENCE_DIRS:
        if not reference_dir.exists():
            continue
        for image_path in reference_dir.rglob("*.png"):
            if excluded_output_dir in image_path.parents:
                continue
            image_hashes.add(_sha256_file(image_path))
        for truth_path in reference_dir.rglob("*.json"):
            if excluded_output_dir in truth_path.parents:
                continue
            try:
                truth_hash = _semantic_truth_hash(_load_json(truth_path))
                if truth_hash is not None:
                    truth_hashes.add(truth_hash)
            except Exception:
                continue
    return {"image": image_hashes, "truth": truth_hashes}


def _coords_from_ground_truth(ground_truth: dict[str, Any]) -> list[tuple[float, float]]:
    question_truth = ground_truth.get("question", {}).get("ground_truth", {})
    coords = question_truth.get("coordinates") or []
    return [(float(coord[0]), float(coord[1])) for coord in coords if len(coord) >= 2]


def _true_extrema_from_ground_truth(ground_truth: dict[str, Any]) -> list[dict[str, Any]]:
    question_truth = ground_truth.get("question", {}).get("ground_truth", {})
    extrema = question_truth.get("extrema") or []
    return [
        {
            "x": float(item["x"]),
            "y": float(item["y"]),
            "type": str(item["type"]),
        }
        for item in extrema
        if {"x", "y", "type"} <= set(item)
    ]


def _matches_true_extremum(
    candidate: dict[str, Any],
    true_extrema: list[dict[str, Any]],
) -> bool:
    for true_item in true_extrema:
        if candidate.get("type") != true_item.get("type"):
            continue
        if abs(float(candidate["x"]) - float(true_item["x"])) > MATCH_TOLERANCE:
            continue
        if abs(float(candidate["y"]) - float(true_item["y"])) > MATCH_TOLERANCE:
            continue
        return True
    return False


def _has_large_regular_change(decision: dict[str, Any]) -> bool:
    metrics = decision.get("metrics", {})
    typical_x_gap = float(metrics.get("typical_x_gap") or 0)
    typical_y_jump = float(metrics.get("typical_y_jump") or 0)
    if typical_x_gap <= 0 or typical_y_jump <= 0:
        return False
    left_large = (
        float(metrics.get("left_x_gap") or 0) >= 1.5 * typical_x_gap
        or float(metrics.get("left_y_jump") or 0) >= 2.5 * typical_y_jump
    )
    right_large = (
        float(metrics.get("right_x_gap") or 0) >= 1.5 * typical_x_gap
        or float(metrics.get("right_y_jump") or 0) >= 2.5 * typical_y_jump
    )
    return left_large or right_large


def classify_sample(ground_truth: dict[str, Any]) -> dict[str, Any]:
    coords = _coords_from_ground_truth(ground_truth)
    true_extrema = _true_extrema_from_ground_truth(ground_truth)
    decisions = classify_extrema_candidates(coords)

    true_decisions = [
        decision
        for decision in decisions
        if _matches_true_extremum(decision["candidate"], true_extrema)
    ]
    false_decisions = [
        decision
        for decision in decisions
        if not _matches_true_extremum(decision["candidate"], true_extrema)
    ]

    categories: list[str] = []
    if true_decisions:
        categories.append("ordinary_true_extrema")
    if any(_has_large_regular_change(decision) for decision in true_decisions):
        categories.append("true_extrema_large_gap_or_jump")
    if any(decision["decision"] == "DROP" for decision in false_decisions):
        categories.append("discontinuity_or_asymptote_pseudo_extrema")
    if not true_extrema and not decisions:
        categories.append("no_extrema_negative")

    keep_true = sum(
        1
        for decision in true_decisions
        if decision["decision"] == "KEEP"
    )
    keep_false = sum(
        1
        for decision in false_decisions
        if decision["decision"] == "KEEP"
    )
    drop_true = sum(
        1
        for decision in true_decisions
        if decision["decision"] == "DROP"
    )
    drop_false = sum(
        1
        for decision in false_decisions
        if decision["decision"] == "DROP"
    )

    return {
        "coords_count": len(coords),
        "true_extrema_count": len(true_extrema),
        "candidate_count": len(decisions),
        "categories": categories,
        "keep_true": keep_true,
        "keep_false": keep_false,
        "drop_true": drop_true,
        "drop_false": drop_false,
        "decisions": decisions,
    }


def _sample_seed(base_seed: int, family: str, profile: str, sample_index: int) -> int:
    family_index = ALL_FUNCTION_TYPES.index(family)
    profile_index = ("digital_mild", "digital_hard", "clean_render", "robustness_stress").index(profile)
    return int(base_seed) * 100000 + family_index * 10000 + profile_index * 1000 + sample_index


def _attempt_seed(base_seed: int, family: str, profile: str, sample_index: int, attempt: int) -> int:
    return _sample_seed(base_seed, family, profile, sample_index) + attempt * 997_003


def generate_validation_set(
    *,
    output_dir: Path,
    base_seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    base_config = load_config()
    reference_hashes = _reference_hashes(output_dir)
    accepted_image_hashes = set(reference_hashes["image"])
    accepted_truth_hashes = set(reference_hashes["truth"])
    manifest: dict[str, Any] = {
        "name": "P0-3 extrema mechanism validation",
        "created_by": "scripts/run_p0_3_extrema_validation.py",
        "base_seed": base_seed,
        "rule_version": EXTREMA_FILTER_RULE_VERSION,
        "images_per_family": IMAGES_PER_FAMILY,
        "profiles_per_family": list(PROFILES_PER_FAMILY),
        "match_tolerance": {"abs_dx": MATCH_TOLERANCE, "abs_dy": MATCH_TOLERANCE},
        "reference_seed_disjointness": list(SHIPPED_BENCHMARK_SEEDS),
        "max_generation_attempts_per_slot": MAX_GENERATION_ATTEMPTS_PER_SLOT,
        "samples": [],
        "rejected_candidates": [],
        "summary": {},
    }

    for family in ALL_FUNCTION_TYPES:
        exact_seen: set[str] = set()
        family_counts: dict[str, int] = {}
        for sample_index, profile in enumerate(PROFILES_PER_FAMILY):
            sample_subset_id = subset_id(family, profile)
            sample_name = f"{sample_subset_id}_{sample_index:04d}"
            image_path = output_dir / "images" / sample_subset_id / f"{sample_name}.png"
            truth_path = output_dir / "ground_truth" / sample_subset_id / f"{sample_name}.json"
            accepted = None
            last_reject: dict[str, Any] | None = None
            for attempt in range(MAX_GENERATION_ATTEMPTS_PER_SLOT):
                sample_seed = _attempt_seed(base_seed, family, profile, sample_index, attempt)
                config = build_subset_config(family, profile, deepcopy(base_config))
                rng = random.Random(sample_seed)
                page_choice = rng.choice(PAGE_SIZE_POOL)
                config["page"] = page_choice
                exact_seen_candidate = set(exact_seen)
                family_counts_candidate = dict(family_counts)
                result = generate_and_save_diverse_image(
                    config,
                    str(image_path),
                    str(truth_path),
                    sample_seed,
                    sample_subset_id,
                    family,
                    profile,
                    sample_index,
                    config.get("coords", {}).get("coord_mode", "manual"),
                    page_choice,
                    exact_seen_candidate,
                    family_counts_candidate,
                )
                ground_truth = _load_json(truth_path)
                classification = classify_sample(ground_truth)
                image_hash = _sha256_file(image_path)
                truth_hash = _semantic_truth_hash(ground_truth)
                if truth_hash is None:
                    raise ValueError(f"Generated ground truth is missing semantic fields: {truth_path}")
                duplicate_image = image_hash in accepted_image_hashes
                duplicate_truth = truth_hash in accepted_truth_hashes
                if duplicate_image or duplicate_truth:
                    last_reject = {
                        "sample_id": sample_name,
                        "family": family,
                        "profile": profile,
                        "seed": sample_seed,
                        "attempt": attempt,
                        "duplicate_image": duplicate_image,
                        "duplicate_truth": duplicate_truth,
                        "image_sha256": image_hash,
                        "semantic_truth_sha256": truth_hash,
                    }
                    manifest["rejected_candidates"].append(last_reject)
                    image_path.unlink(missing_ok=True)
                    truth_path.unlink(missing_ok=True)
                    continue

                exact_seen = exact_seen_candidate
                family_counts = family_counts_candidate
                accepted_image_hashes.add(image_hash)
                accepted_truth_hashes.add(truth_hash)
                accepted = {
                    "sample_id": sample_name,
                    "family": family,
                    "profile": profile,
                    "seed": sample_seed,
                    "attempt": attempt,
                    "image_path": image_path.relative_to(ROOT).as_posix(),
                    "ground_truth_path": truth_path.relative_to(ROOT).as_posix(),
                    "image_sha256": image_hash,
                    "semantic_truth_sha256": truth_hash,
                    "duplicate_image": False,
                    "duplicate_truth": False,
                    "generation": result,
                    "classification": classification,
                }
                break
            if accepted is None:
                raise RuntimeError(
                    "Failed to generate a non-duplicate sample for "
                    f"{sample_name}; last rejection={last_reject}"
                )
            manifest["samples"].append(accepted)

    manifest["summary"] = summarize_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _categories_help_quota(categories: list[str], category_counts: dict[str, int]) -> bool:
    return any(
        category in SAMPLE_CATEGORY_QUOTAS
        and category_counts.get(category, 0) < SAMPLE_CATEGORY_QUOTAS[category]
        for category in categories
    )


def _next_quota_target(category_counts: dict[str, int]) -> str | None:
    for category, required in SAMPLE_CATEGORY_QUOTAS.items():
        if category_counts.get(category, 0) < required:
            return category
    return None


def _source_for_attempt(
    target_category: str | None,
    attempt: int,
) -> tuple[str, str]:
    if target_category is None:
        return TAIL_SOURCE_PROFILES[attempt % len(TAIL_SOURCE_PROFILES)]
    sources = QUOTA_SOURCE_PROFILES[target_category]
    return sources[attempt % len(sources)]


def generate_quota_validation_set(
    *,
    output_dir: Path,
    base_seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    base_config = load_config()
    reference_hashes = _reference_hashes(output_dir)
    accepted_image_hashes = set(reference_hashes["image"])
    accepted_truth_hashes = set(reference_hashes["truth"])
    category_counts = {category: 0 for category in SAMPLE_CATEGORY_QUOTAS}
    family_counts = {family: 0 for family in ALL_FUNCTION_TYPES}
    exact_seen_by_subset: dict[str, set[str]] = {}
    family_signature_counts_by_subset: dict[str, dict[str, int]] = {}
    candidate_dir = output_dir / "_candidate_work"
    samples: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []

    for attempt in range(MAX_TOTAL_CANDIDATE_ATTEMPTS):
        quotas_met = all(
            category_counts[category] >= required
            for category, required in SAMPLE_CATEGORY_QUOTAS.items()
        )
        families_covered = (
            sum(1 for count in family_counts.values() if count >= MIN_SAMPLES_PER_FAMILY)
            >= MIN_DISTINCT_QUOTA_FAMILIES
        )
        if len(samples) >= TOTAL_VALIDATION_SAMPLES and quotas_met and families_covered:
            break

        target_category = _next_quota_target(category_counts)
        family, profile = _source_for_attempt(target_category, attempt)
        source_index = TAIL_SOURCE_PROFILES.index((family, profile))
        sample_seed = int(base_seed) * 100000 + attempt * 997_003 + source_index
        sample_subset_id = subset_id(family, profile)
        candidate_name = f"candidate_{attempt:05d}"
        candidate_image_path = candidate_dir / "images" / sample_subset_id / f"{candidate_name}.png"
        candidate_truth_path = candidate_dir / "ground_truth" / sample_subset_id / f"{candidate_name}.json"

        config = build_subset_config(family, profile, deepcopy(base_config))
        config.setdefault("generation", {})["diversity_retry_attempts"] = QUOTA_DIVERSITY_RETRY_ATTEMPTS
        rng = random.Random(sample_seed)
        page_choice = rng.choice(PAGE_SIZE_POOL)
        config["page"] = page_choice
        exact_seen_candidate = set(exact_seen_by_subset.get(sample_subset_id, set()))
        family_counts_candidate = dict(family_signature_counts_by_subset.get(sample_subset_id, {}))
        try:
            result = generate_and_save_diverse_image(
                config,
                str(candidate_image_path),
                str(candidate_truth_path),
                sample_seed,
                sample_subset_id,
                family,
                profile,
                attempt,
                config.get("coords", {}).get("coord_mode", "manual"),
                page_choice,
                exact_seen_candidate,
                family_counts_candidate,
            )
            ground_truth = _load_json(candidate_truth_path)
            classification = classify_sample(ground_truth)
            image_hash = _sha256_file(candidate_image_path)
            truth_hash = _semantic_truth_hash(ground_truth)
            if truth_hash is None:
                raise ValueError(f"Generated ground truth is missing semantic fields: {candidate_truth_path}")
            duplicate_image = image_hash in accepted_image_hashes
            duplicate_truth = truth_hash in accepted_truth_hashes
            categories = classification["categories"]

            has_target_category = (
                target_category is not None
                and target_category in categories
                and category_counts.get(target_category, 0) < SAMPLE_CATEGORY_QUOTAS[target_category]
            )
            must_cover_family = (
                target_category is None
                and family_counts[family] < MIN_SAMPLES_PER_FAMILY
                and sum(1 for count in family_counts.values() if count >= MIN_SAMPLES_PER_FAMILY)
                < MIN_DISTINCT_QUOTA_FAMILIES
            )
            helps_quota = _categories_help_quota(categories, category_counts)
            filling_tail = quotas_met and families_covered and len(samples) < TOTAL_VALIDATION_SAMPLES
            accept = (
                not duplicate_image
                and not duplicate_truth
                and (has_target_category or must_cover_family or helps_quota or filling_tail)
            )
            if not accept:
                rejected_candidates.append(
                    {
                        "family": family,
                        "profile": profile,
                        "seed": sample_seed,
                        "attempt": attempt,
                        "duplicate_image": duplicate_image,
                        "duplicate_truth": duplicate_truth,
                        "categories": categories,
                        "target_category": target_category,
                        "reason": "duplicate_or_does_not_match_frozen_quota_target",
                    }
                )
                continue

            accepted_index = len(samples)
            sample_name = f"{sample_subset_id}_{accepted_index:04d}"
            final_image_path = output_dir / "images" / sample_subset_id / f"{sample_name}.png"
            final_truth_path = output_dir / "ground_truth" / sample_subset_id / f"{sample_name}.json"
            final_image_path.parent.mkdir(parents=True, exist_ok=True)
            final_truth_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidate_image_path), str(final_image_path))
            shutil.move(str(candidate_truth_path), str(final_truth_path))

            exact_seen_by_subset[sample_subset_id] = exact_seen_candidate
            family_signature_counts_by_subset[sample_subset_id] = family_counts_candidate
            accepted_image_hashes.add(image_hash)
            accepted_truth_hashes.add(truth_hash)
            family_counts[family] += 1
            for category in categories:
                if category in category_counts:
                    category_counts[category] += 1
            samples.append(
                {
                    "sample_id": sample_name,
                    "family": family,
                    "profile": profile,
                    "seed": sample_seed,
                    "attempt": attempt,
                    "image_path": final_image_path.relative_to(ROOT).as_posix(),
                    "ground_truth_path": final_truth_path.relative_to(ROOT).as_posix(),
                    "image_sha256": image_hash,
                    "semantic_truth_sha256": truth_hash,
                    "duplicate_image": False,
                    "duplicate_truth": False,
                    "generation": result,
                    "classification": classification,
                }
            )
        finally:
            if candidate_image_path.exists():
                candidate_image_path.unlink()
            if candidate_truth_path.exists():
                candidate_truth_path.unlink()
        if (attempt + 1) % 100 == 0:
            print(
                "quota search "
                f"attempt={attempt + 1} accepted={len(samples)} "
                f"category_counts={json.dumps(category_counts, sort_keys=True)}",
                flush=True,
            )

    shutil.rmtree(candidate_dir, ignore_errors=True)
    manifest: dict[str, Any] = {
        "name": "P0-3 extrema mechanism validation",
        "created_by": "scripts/run_p0_3_extrema_validation.py",
        "sampling_strategy": "quota_stratified_v1",
        "base_seed": base_seed,
        "rule_version": EXTREMA_FILTER_RULE_VERSION,
        "total_validation_samples": TOTAL_VALIDATION_SAMPLES,
        "min_samples_per_family": MIN_SAMPLES_PER_FAMILY,
        "min_distinct_quota_families": MIN_DISTINCT_QUOTA_FAMILIES,
        "profiles": list(QUOTA_PROFILES),
        "quota_source_profiles": {
            category: [
                {"family": family, "profile": profile}
                for family, profile in sources
            ]
            for category, sources in QUOTA_SOURCE_PROFILES.items()
        },
        "sample_category_quotas": SAMPLE_CATEGORY_QUOTAS,
        "match_tolerance": {"abs_dx": MATCH_TOLERANCE, "abs_dy": MATCH_TOLERANCE},
        "reference_seed_disjointness": list(SHIPPED_BENCHMARK_SEEDS),
        "max_total_candidate_attempts": MAX_TOTAL_CANDIDATE_ATTEMPTS,
        "quota_diversity_retry_attempts": QUOTA_DIVERSITY_RETRY_ATTEMPTS,
        "samples": samples,
        "rejected_candidates": rejected_candidates,
        "summary": {},
    }
    manifest["summary"] = summarize_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    samples = manifest.get("samples", [])
    category_counts: dict[str, int] = {}
    family_counts: dict[str, dict[str, int]] = {}
    keep_true = 0
    keep_false = 0
    drop_true = 0
    drop_false = 0
    for sample in samples:
        family = sample["family"]
        family_counts.setdefault(family, {"positive_candidate": 0, "negative_candidate": 0})
        classification = sample["classification"]
        for category in classification["categories"]:
            category_counts[category] = category_counts.get(category, 0) + 1
        keep_true += int(classification["keep_true"])
        keep_false += int(classification["keep_false"])
        drop_true += int(classification["drop_true"])
        drop_false += int(classification["drop_false"])
        if classification["true_extrema_count"] > 0:
            family_counts[family]["positive_candidate"] += 1
        if classification["keep_false"] + classification["drop_false"] > 0 or (
            classification["true_extrema_count"] == 0 and classification["candidate_count"] == 0
        ):
            family_counts[family]["negative_candidate"] += 1

    keep_total = keep_true + keep_false
    drop_total = drop_true + drop_false
    keep_precision = keep_true / keep_total if keep_total else None
    drop_precision = drop_false / drop_total if drop_total else None
    delta_precision = (
        keep_precision - (1.0 - drop_precision)
        if keep_precision is not None and drop_precision is not None
        else None
    )
    quota_status = {
        category: {
            "required": required,
            "observed": category_counts.get(category, 0),
            "passed": category_counts.get(category, 0) >= required,
        }
        for category, required in SAMPLE_CATEGORY_QUOTAS.items()
    }
    return {
        "sample_count": len(samples),
        "category_counts": category_counts,
        "sample_category_quotas": quota_status,
        "sample_category_quota_passed": all(
            status["passed"] for status in quota_status.values()
        ),
        "family_counts": family_counts,
        "distinct_family_count": sum(1 for counts in family_counts.values() if sum(counts.values()) > 0),
        "duplicate_image_count": sum(1 for sample in samples if sample["duplicate_image"]),
        "duplicate_truth_count": sum(1 for sample in samples if sample["duplicate_truth"]),
        "rejected_candidate_count": len(manifest.get("rejected_candidates", [])),
        "keep_true": keep_true,
        "keep_false": keep_false,
        "drop_true": drop_true,
        "drop_false": drop_false,
        "keep_precision": keep_precision,
        "drop_false_precision": drop_precision,
        "delta_keep_vs_drop_correctness": delta_precision,
        "h1_eligible": (
            len(samples) == TOTAL_VALIDATION_SAMPLES
            and all(not sample["duplicate_image"] and not sample["duplicate_truth"] for sample in samples)
            and all(status["passed"] for status in quota_status.values())
            and sum(
                1
                for counts in family_counts.values()
                if counts.get("positive_candidate", 0) + counts.get("negative_candidate", 0)
                >= MIN_SAMPLES_PER_FAMILY
            )
            >= MIN_DISTINCT_QUOTA_FAMILIES
        ),
    }


def _format_summary(manifest: dict[str, Any]) -> str:
    summary = manifest["summary"]
    lines = [
        f"Samples: {summary['sample_count']}",
        f"Rule: {manifest['rule_version']}",
        f"Category counts: {json.dumps(summary['category_counts'], sort_keys=True)}",
        f"Category quotas passed: {summary['sample_category_quota_passed']}",
        f"Duplicate images: {summary['duplicate_image_count']}",
        f"Duplicate semantic truth: {summary['duplicate_truth_count']}",
        f"Rejected duplicate candidates: {summary['rejected_candidate_count']}",
        (
            "KEEP true/false: "
            f"{summary['keep_true']}/{summary['keep_false']} | "
            f"DROP true/false: {summary['drop_true']}/{summary['drop_false']}"
        ),
        f"KEEP precision: {summary['keep_precision']}",
        f"DROP false precision: {summary['drop_false_precision']}",
        f"Delta correctness: {summary['delta_keep_vs_drop_correctness']}",
        f"H1 eligible: {summary['h1_eligible']}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate/evaluate the frozen P0-3 extrema validation set."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--strategy",
        choices=("quota", "balanced"),
        default="quota",
        help="Validation-set sampling strategy.",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()

    manifest_path = args.output_dir / "manifest.json"
    if args.summary_only:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        manifest = _load_json(manifest_path)
    else:
        if args.strategy == "balanced":
            manifest = generate_validation_set(
                output_dir=args.output_dir,
                base_seed=args.base_seed,
                overwrite=args.overwrite,
            )
        else:
            manifest = generate_quota_validation_set(
                output_dir=args.output_dir,
                base_seed=args.base_seed,
                overwrite=args.overwrite,
            )
    print(_format_summary(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
