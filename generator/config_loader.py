"""
Load and validate generator config from YAML.
Merges with built-in defaults so partial configs are allowed.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _defaults() -> Dict[str, Any]:
    return {
        "server": {
            "app": {"host": "127.0.0.1", "port": 7870},
            "recognizer_ui": {"host": "127.0.0.1", "port": 7860},
        },
        "page": {
            "width": 1920,
            "height": 1200,
            "layout": "random",
            "layout_options": ["top_bottom", "left_right"],
            "margin_px": 40,
        },
        "chart": {
            "area_ratio_range": [0.70, 0.90],
            "aspect_ratio_range": [1.0, 1.8],
            "padding_px": 80,
            "border": {
                "enabled": True,
                "color": [0, 0, 0],
                "thickness": 3,
                "margin": 10,
            },
        },
        "coords": {
            "x_range": [-5, 5],
            "y_range": [-5, 5],
            "grid_step_x": 1,
            "grid_step_y": 1,
            "min_intersections": 6,
        },
        "ticks": {
            "font_size": 36,
            "label_color": [0, 0, 0],
        },
        "axes": {"line_width": 2, "color": [0, 0, 0]},
        "grid": {"line_width": 1, "color": [140, 140, 140], "dash_on": 10, "dash_off": 6},
        "stem": {
            "font_size": 28,
            "color": [30, 30, 30],
            "placeholder_lines": [
                "Consider the graph of the function shown below.",
                "Find the coordinates of all intersections as described.",
            ],
            "max_lines": 5,
        },
        "curves": {
            "function_types": [
                "linear", "quadratic", "absolute",
                "cubic", "piecewise_linear",
                "exponential", "logarithmic",
                "rational",
            ],
            "num_curves_range": [1, 1],
            "color": [139, 0, 0],
            "line_width": 3,
        },
        "generation": {
            "candidate_attempts": 120,
            "selection_pool_size": 10,
            "selection_score_margin": 3.0,
            "diversity_retry_attempts": 18,
            "family_soft_cap": 8,
            "intersection_targets": {
                "linear": {"hard_min": 8, "preferred_min": 10, "preferred_max": 14},
                "quadratic": {"hard_min": 4, "preferred_min": 4, "preferred_max": 5},
                "absolute": {"hard_min": 7, "preferred_min": 9, "preferred_max": 14},
                "cubic": {"hard_min": 5, "preferred_min": 5, "preferred_max": 5},
                "piecewise_linear": {"hard_min": 8, "preferred_min": 10, "preferred_max": 14},
                "exponential": {"hard_min": 3, "preferred_min": 4, "preferred_max": 5},
                "logarithmic": {"hard_min": 4, "preferred_min": 4, "preferred_max": 4},
                "rational": {"hard_min": 6, "preferred_min": 6, "preferred_max": 6},
            },
        },
        "noise": {
            "enabled": True,
            "ocr_safe_adaptation": False,
            "fragile_tone_scale": 0.75,
            "fragile_noise_scale": 0.75,
            "fragile_blur_scale": 0.80,
            "fragile_geometry_scale": 0.60,
            "tone": {
                "enabled": True,
                "brightness_range": [0.97, 1.03],
                "contrast_range": [0.98, 1.02],
            },
            "sensor_noise": {
                "enabled": False,
                "gaussian_sigma": 1.5,
            },
            "blur": {
                "enabled": False,
                "ksize_range": [0, 3],
            },
            "compression": {
                "enabled": False,
                "jpeg_quality_range": [88, 96],
            },
            "resample": {
                "enabled": False,
                "scale_range": [0.92, 0.98],
            },
            "geometry": {
                "enabled": False,
                "rotation_angle_range": [-0.5, 0.5],
                "perspective_strength_range": [0.0, 0.0],
            },
        },
        "output": {
            "image_dir": "dataset/images",
            "gt_dir": "dataset/ground_truth",
            "image_format": "png",
            "image_dpi": None,
        },
    }


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load generator config. If path is None, use default_config.yaml next to this module's config dir.
    """
    defaults = _defaults()
    if path is None:
        config_dir = Path(__file__).resolve().parent.parent / "config"
        path = config_dir / "default_config.yaml"
    path = Path(path)
    if not path.exists():
        return defaults
    with open(path, "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}
    return _deep_merge(defaults, overrides)
