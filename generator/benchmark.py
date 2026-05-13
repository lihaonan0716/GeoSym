"""GeoSym-Bench: structured benchmark generation.

Two orthogonal dimensions (function type x degradation profile):
  Function type — linear, quadratic, absolute, cubic, piecewise_linear,
                  exponential, logarithmic, rational
  Profile       — clean_render / digital_mild / digital_hard / robustness_stress

Each image contains exactly one function curve.
Each subset gets a deterministic seed range for full reproducibility.
Supports CPU-parallel generation via ``concurrent.futures`` for speed.
"""

import json
import logging
import os
import random
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from .config_loader import _deep_merge, load_config
from .core import generate_one
from .rendering.curves import (
    ALL_FUNCTION_TYPES,
    FUNCTION_TYPE_DESCRIPTIONS,
    get_richness_spec,
    summarize_generated_sample,
)

logger = logging.getLogger(__name__)


# ============================================================
# Benchmark profiles
# ============================================================

PROFILE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "clean_render": {
        "group": "main",
        "description": "Clean digital render with no synthetic degradation.",
        "noise": {
            "enabled": False,
            "ocr_safe_adaptation": False,
        },
    },
    "digital_mild": {
        "group": "main",
        "description": "Mild digital degradation using light tone jitter, compression, and resampling.",
        "noise": {
            "enabled": True,
            "ocr_safe_adaptation": False,
            "tone": {
                "enabled": True,
                "brightness_range": [0.97, 1.03],
                "contrast_range": [0.98, 1.02],
            },
            "sensor_noise": {
                "enabled": True,
                "gaussian_sigma": 1.5,
            },
            "blur": {
                "enabled": False,
                "ksize_range": [0, 1],
            },
            "compression": {
                "enabled": True,
                "jpeg_quality_range": [88, 96],
            },
            "resample": {
                "enabled": True,
                "scale_range": [0.92, 0.98],
            },
            "geometry": {
                "enabled": False,
                "rotation_angle_range": [-0.4, 0.4],
                "perspective_strength_range": [0.0, 0.0],
            },
        },
    },
    "digital_hard": {
        "group": "main",
        "description": "Stronger digital degradation while remaining task-relevant and non-camera-specific.",
        "noise": {
            "enabled": True,
            "ocr_safe_adaptation": False,
            "tone": {
                "enabled": True,
                "brightness_range": [0.94, 1.06],
                "contrast_range": [0.95, 1.05],
            },
            "sensor_noise": {
                "enabled": True,
                "gaussian_sigma": 3.0,
            },
            "blur": {
                "enabled": True,
                "ksize_range": [0, 3],
            },
            "compression": {
                "enabled": True,
                "jpeg_quality_range": [70, 88],
            },
            "resample": {
                "enabled": True,
                "scale_range": [0.78, 0.92],
            },
            "geometry": {
                "enabled": False,
                "rotation_angle_range": [-0.5, 0.5],
                "perspective_strength_range": [0.0, 0.0],
            },
        },
    },
    "robustness_stress": {
        "group": "stress",
        "description": "Aggressive synthetic stressors reserved for robustness studies, not the main benchmark.",
        "noise": {
            "enabled": True,
            "ocr_safe_adaptation": False,
            "tone": {
                "enabled": True,
                "brightness_range": [0.90, 1.10],
                "contrast_range": [0.92, 1.08],
            },
            "sensor_noise": {
                "enabled": True,
                "gaussian_sigma": 5.0,
            },
            "blur": {
                "enabled": True,
                "ksize_range": [1, 3],
            },
            "compression": {
                "enabled": True,
                "jpeg_quality_range": [45, 75],
            },
            "resample": {
                "enabled": True,
                "scale_range": [0.55, 0.80],
            },
            "geometry": {
                "enabled": True,
                "rotation_angle_range": [-1.2, 1.2],
                "perspective_strength_range": [4.0, 18.0],
            },
        },
    },
}

BENCHMARK_PROFILES = ["clean_render"]

COORD_POOL = [
    {"x_range": [-7, 7], "y_range": [-7, 7], "grid_step_x": 1, "grid_step_y": 1},
    {"x_range": [-10, 10], "y_range": [-10, 10], "grid_step_x": 2, "grid_step_y": 2},
    {"x_range": [-3, 11], "y_range": [-6, 8], "grid_step_x": 1, "grid_step_y": 1},
    {"x_range": [-3, 7], "y_range": [-4, 6], "grid_step_x": 1, "grid_step_y": 1},
    {"x_range": [-2, 8], "y_range": [-5, 9], "grid_step_x": 1, "grid_step_y": 1},
    {"x_range": [-8, 8], "y_range": [-6, 6], "grid_step_x": 2, "grid_step_y": 2},
]

PAGE_SIZE_POOL = [
    {"width": 1440, "height": 1440},
    {"width": 1920, "height": 1200},
    {"width": 1200, "height": 1200},
    {"width": 1600, "height": 1000},
]


def build_subset_config(
    func_type: str,
    profile_id: str,
    base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a complete generator config for a single (func_type, profile) subset."""
    cfg = deepcopy(base_config or load_config())
    cfg["curves"]["num_curves_range"] = [1, 1]
    cfg["curves"]["function_types"] = [func_type]
    profile_cfg = PROFILE_REGISTRY[profile_id]
    cfg = _deep_merge(cfg, {k: v for k, v in profile_cfg.items() if k not in {"description", "group"}})
    return cfg


def subset_id(func_type: str, profile_id: str) -> str:
    """e.g. 'quadratic-clean_render', 'exponential-digital_hard'."""
    return f"{func_type}-{profile_id}"


def _stable_image_seed(base_seed: int, func_type: str, profile_id: str, image_idx: int) -> int:
    """Deterministic seed without relying on Python's randomized hash()."""
    func_idx = ALL_FUNCTION_TYPES.index(func_type)
    profile_idx = BENCHMARK_PROFILES.index(profile_id)
    return int(base_seed) * 100000 + func_idx * 10000 + profile_idx * 1000 + int(image_idx)


def _diversity_cfg(img_cfg: Dict[str, Any]) -> Dict[str, int]:
    gen_cfg = img_cfg.get("generation", {})
    return {
        "retry_attempts": int(gen_cfg.get("diversity_retry_attempts", 18)),
        "family_soft_cap": int(gen_cfg.get("family_soft_cap", 8)),
    }


def generate_and_save_diverse_image(
    img_cfg: Dict[str, Any],
    img_path: str,
    gt_path: str,
    seed: Optional[int],
    sid: str,
    ftype: str,
    profile_id: str,
    idx: int,
    coord_mode: str,
    page_choice: Dict[str, Any],
    exact_seen: Optional[set[str]] = None,
    family_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Generate one image while discouraging duplicate structures within a subset."""
    exact_seen = exact_seen if exact_seen is not None else set()
    family_counts = family_counts if family_counts is not None else {}
    diversity_cfg = _diversity_cfg(img_cfg)
    richness = get_richness_spec(ftype)

    best: Optional[Tuple[np.ndarray, Dict[str, Any], Dict[str, Any], float, Optional[int]]] = None
    base_seed = seed if seed is not None else random.randint(1, 10_000_000)

    for retry in range(max(1, diversity_cfg["retry_attempts"])):
        cand_seed = None if seed is None else base_seed + retry * 9973
        img, gt = generate_one(img_cfg, seed=cand_seed)
        summary = summarize_generated_sample(gt)
        exact_sig = summary["exact_signature"]
        family_sig = summary["family_signature"]
        exact_dup = exact_sig in exact_seen
        family_count = family_counts.get(family_sig, 0)
        raw_score = float(gt.get("generation_meta", {}).get("candidate_score", len(gt.get("intersections", []))))
        score = raw_score - 9.0 * float(family_count)
        if exact_dup:
            score -= 1000.0
        if len(gt.get("intersections", [])) >= richness["preferred_min"]:
            score += 6.0

        if best is None or score > best[3]:
            best = (img, gt, summary, score, cand_seed)

        if (
            not exact_dup
            and family_count == 0
            and len(gt.get("intersections", [])) >= richness["preferred_min"]
        ):
            break

    if best is None:
        raise RuntimeError(f"Failed to generate sample for subset {sid}")

    img, gt, summary, _, final_seed = best
    gt["benchmark_meta"] = {
        "subset_id": sid,
        "func_type": ftype,
        "profile_id": profile_id,
        "profile_group": PROFILE_REGISTRY[profile_id]["group"],
        "image_index": idx,
        "seed": final_seed,
        "coord_mode": coord_mode,
        "coord_config": gt.get("x_range", []) + gt.get("y_range", []),
        "page_config": page_choice,
        "diversity_family_signature": summary["family_signature"],
        "diversity_exact_signature": summary["exact_signature"],
    }

    exact_seen.add(summary["exact_signature"])
    family_counts[summary["family_signature"]] = family_counts.get(summary["family_signature"], 0) + 1

    Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    Path(gt_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), img)
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

    return {
        "filename": f"{sid}_{idx:04d}.png",
        "num_functions": len(gt.get("functions", [])),
        "function_types": [fn["type"] for fn in gt.get("functions", [])],
        "intersection_count": len(gt.get("intersections", [])),
        "family_signature": summary["family_signature"],
    }


def generate_benchmark(
    output_dir: str | Path,
    images_per_subset: int = 50,
    base_seed: int = 42,
    base_config: Optional[Dict[str, Any]] = None,
    subsets: Optional[List[Tuple[str, str]]] = None,
    func_types: Optional[List[str]] = None,
    profiles: Optional[List[str]] = None,
    progress_callback=None,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate the full GeoSym-Bench dataset."""
    output_dir = Path(output_dir)
    base_cfg = base_config or load_config()

    if subsets is None:
        ft = func_types or ALL_FUNCTION_TYPES
        pf = profiles if profiles is not None else BENCHMARK_PROFILES
        subsets = list(product(ft, pf))

    manifest = {
        "benchmark": "GeoSym-Bench",
        "version": "5.0",
        "base_seed": base_seed,
        "images_per_subset": images_per_subset,
        "total_subsets": len(subsets),
        "total_images": len(subsets) * images_per_subset,
        "dimensions": {
            "function_types": {t: FUNCTION_TYPE_DESCRIPTIONS.get(t, t) for t in ALL_FUNCTION_TYPES},
            "profiles": {
                pid: {
                    "group": PROFILE_REGISTRY[pid]["group"],
                    "description": PROFILE_REGISTRY[pid]["description"],
                }
                for pid in BENCHMARK_PROFILES
            },
        },
        "subsets": {},
    }

    total = len(subsets) * images_per_subset
    counter = 0
    all_tasks: list[tuple] = []

    for ftype, profile_id in subsets:
        sid = subset_id(ftype, profile_id)
        cfg = build_subset_config(ftype, profile_id, base_cfg)

        img_dir = output_dir / "images" / sid
        gt_dir = output_dir / "ground_truth" / sid
        img_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        subset_stats = {
            "func_type": ftype,
            "profile_id": profile_id,
            "profile_group": PROFILE_REGISTRY[profile_id]["group"],
            "function_types": cfg["curves"]["function_types"],
            "noise_enabled": cfg.get("noise", {}).get("enabled", False),
            "count": images_per_subset,
            "images": [],
        }

        coord_mode = cfg.get("coords", {}).get("coord_mode", "manual")

        tasks: list[tuple] = []
        for i in range(images_per_subset):
            seed = _stable_image_seed(base_seed, ftype, profile_id, i)
            fname = f"{sid}_{i:04d}"
            img_path = str(img_dir / f"{fname}.png")
            gt_path = str(gt_dir / f"{fname}.json")

            img_cfg = deepcopy(cfg)
            rng = random.Random(seed)
            page_choice = rng.choice(PAGE_SIZE_POOL)
            if coord_mode == "manual":
                coord_choice = rng.choice(COORD_POOL)
                img_cfg = _deep_merge(img_cfg, {"coords": coord_choice, "page": page_choice})
            else:
                img_cfg = _deep_merge(img_cfg, {"page": page_choice})

            tasks.append((img_cfg, img_path, gt_path, seed, sid, ftype, profile_id, i, coord_mode, page_choice))

        all_tasks.append((sid, ftype, profile_id, subset_stats, tasks))

    for sid, ftype, profile_id, subset_stats, tasks in all_tasks:
        exact_seen: set[str] = set()
        family_counts: Dict[str, int] = {}
        for t in tasks:
            result = generate_and_save_diverse_image(
                img_cfg=t[0],
                img_path=t[1],
                gt_path=t[2],
                seed=t[3],
                sid=t[4],
                ftype=t[5],
                profile_id=t[6],
                idx=t[7],
                coord_mode=t[8],
                page_choice=t[9],
                exact_seen=exact_seen,
                family_counts=family_counts,
            )
            subset_stats["images"].append(result)
            counter += 1
            if progress_callback:
                progress_callback(sid, counter, total)
        subset_stats["images"].sort(key=lambda d: d["filename"])
        manifest["subsets"][sid] = subset_stats

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def list_all_subsets() -> List[Tuple[str, str]]:
    """Return all (func_type, profile_id) combinations."""
    return list(product(ALL_FUNCTION_TYPES, BENCHMARK_PROFILES))
