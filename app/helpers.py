"""Shared constants, config utilities, dataset helpers, and LLM adapter factory.

Extracted from app.py to allow reuse across modules without circular imports.
"""

import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import yaml

# Add project root to path (one level up from app/ package)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from generator.config_loader import load_config
from evaluator.adapters.base import BaseLLMAdapter

# ==================== Global State ====================
_config = load_config()
_dataset_dir = _root / "dataset"
_images_dir = _dataset_dir / "images"
_gt_dir = _dataset_dir / "ground_truth"
_reports_dir = _root / "reports"
_api_keys_path = _root / "config" / "api_keys.yaml"


# ==================== API Key Persistence ====================

def _load_api_keys() -> Dict[str, str]:
    """Load saved API keys from config/api_keys.yaml."""
    if _api_keys_path.exists():
        try:
            with open(_api_keys_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return {k: str(v) for k, v in data.items() if v}
        except Exception:
            pass
    return {}


def _get_saved_key(method: str) -> str:
    """Get saved API key for a given display-name model.

    Resolves the provider-level key name via _MODEL_KEY_MAP (built from
    the ``models`` section in default_config.yaml).  Falls back to a
    slugified version of the display name for backward compatibility.
    """
    keys = _load_api_keys()
    key_name = _MODEL_KEY_MAP.get(method, method.lower().replace("-", "_").replace(" ", "_"))
    if key_name in keys:
        return keys[key_name]
    env_name = f"{key_name.upper()}_KEY"
    return os.environ.get(env_name, "")


def _get_saved_base_url(method: str) -> str:
    """Get the configured API base URL for a given display-name model."""
    keys = _load_api_keys()
    key_name = _MODEL_BASE_URL_MAP.get(method, "")
    if key_name:
        if key_name in keys:
            return keys[key_name]
        env_name = f"{key_name.upper()}".replace("_BASE_URL", "") + "_BASE_URL"
        if env_name in os.environ:
            return os.environ[env_name]
        direct_env_name = key_name.upper()
        if direct_env_name in os.environ:
            return os.environ[direct_env_name]
    return ""


# ==================== Helper Functions ====================

def list_dataset_images() -> List[str]:
    """List all images in dataset (supports subset sub-directories)."""
    if not _images_dir.exists():
        return []
    results = []
    for f in _images_dir.rglob("*.png"):
        rel = f.relative_to(_images_dir)
        results.append(str(rel.with_suffix("")))
    return sorted(results)


def load_image_and_gt(image_name: str) -> Tuple[np.ndarray | None, dict | None]:
    """Load image and its ground truth (supports subset sub-paths like linear-clean_render/img)."""
    img_path = _images_dir / f"{image_name}.png"
    gt_path = _gt_dir / f"{image_name}.json"

    img = None
    gt = None

    if img_path.exists():
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)

    # Backfill: if ground_truth.coordinates is missing, derive from top-level intersections
    if gt is not None:
        q = gt.get("question", {})
        gt_data = q.get("ground_truth", {})
        if gt_data and "coordinates" not in gt_data:
            intersections = gt.get("intersections", [])
            gt_data["coordinates"] = [
                [p["coordinate_x"], p["coordinate_y"]] for p in intersections
            ]

    return img, gt


def format_gt_coords(gt: dict, lang: str) -> str:
    """Format ground truth coordinates as string."""
    if not gt or "intersections" not in gt:
        return 'No intersections'
    coords = gt["intersections"]
    lines = [f"({c['coordinate_x']}, {c['coordinate_y']}) - {c['type']}" for c in coords]
    return "\n".join(lines) if lines else 'No intersections'


def format_gt_functions(gt: dict, lang: str) -> str:
    """Format ground truth function expressions and math properties."""
    if not gt or "functions" not in gt:
        return "N/A"
    functions = gt["functions"]
    if not functions:
        return "N/A"
    is_zh = False
    lines = []
    for i, fn in enumerate(functions, 1):
        ftype = fn.get("type", "unknown")
        latex = fn.get("expression_latex", "N/A")
        sympy_expr = fn.get("expression_sympy", "N/A")
        params = fn.get("parameters", {})
        lines.append(f"{'Curve'} {i} [{ftype}]")
        lines.append(f"  LaTeX:  {latex}")
        lines.append(f"  Sympy:  {sympy_expr}")
        lines.append(f"  Params: {params}")

        mp = fn.get("math_properties", {})
        if mp:
            lines.append(f"  {'Domain'}: {mp.get('domain', '')}")
            lines.append(f"  {'Range'}:  {mp.get('range', '')}")
            zeros = mp.get("zeros", [])
            zeros_str = ", ".join(str(z) for z in zeros) if zeros else ("None")
            lines.append(f"  {'Zeros'}:  {zeros_str}")
        lines.append("")

    question = gt.get("question", {})
    gt_data = question.get("ground_truth", {})
    if gt_data:
        coords = gt_data.get("coordinates", [])
        if coords:
            lines.append(f"{'Key Coordinates'}: {len(coords)} {'points'}")
            for c in coords[:20]:
                lines.append(f"  ({c[0]}, {c[1]})")
            if len(coords) > 20:
                lines.append(f"  ... +{len(coords)-20} more")

        pred_tgt = gt_data.get("prediction_targets", {})
        if pred_tgt:
            lines.append("")
            lines.append(f"{'Prediction Targets'}:")
            for x, y in pred_tgt.items():
                lines.append(f"  f({x}) = {y}")

        roc = gt_data.get("rate_of_change", {})
        if roc and "gt_value" in roc:
            lines.append(f"{'Rate of Change'}: {roc['gt_value']} "
                         f"(x∈[{roc.get('x1', '?')},{roc.get('x2', '?')}])")

    return "\n".join(lines)


def get_next_index(subset_dir: Path) -> int:
    """Get next available image index within a subset directory."""
    existing = list(subset_dir.glob("*.png"))
    if not existing:
        return 0
    indices = []
    for f in existing:
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2:
            try:
                indices.append(int(parts[1]))
            except ValueError:
                pass
    return max(indices) + 1 if indices else 0


def clear_subset(subset_id: str):
    """Clear a single subset's images and ground truth."""
    img_sub = _images_dir / subset_id
    gt_sub = _gt_dir / subset_id
    if img_sub.exists():
        shutil.rmtree(img_sub)
    if gt_sub.exists():
        shutil.rmtree(gt_sub)


def clear_dataset():
    """Clear all images and ground truth files."""
    if _images_dir.exists():
        shutil.rmtree(_images_dir)
    if _gt_dir.exists():
        shutil.rmtree(_gt_dir)
    _images_dir.mkdir(parents=True, exist_ok=True)
    _gt_dir.mkdir(parents=True, exist_ok=True)


def _resolve_coord_mode(coord_mode_label: str) -> str:
    """Map UI label to internal coord_mode key."""
    low = coord_mode_label.lower()
    if "adaptive" in low or "自适应" in low:
        return "adaptive"
    if "static" in low or "静态" in low:
        return "static"
    return "manual"


# ---------------------------------------------------------------------------
# YAML-driven model registry (loaded from config/default_config.yaml → models)
# ---------------------------------------------------------------------------

def _build_model_registry(cfg: dict):
    """Parse the ``models`` section of default_config.yaml and build:

    * MODEL_REGISTRY  – {display_name: (module_path, class_name, api_model_id)}
    * PROVIDER_MAP    – OrderedDict {provider: [display_name, ...]}
    * _MODEL_KEY_MAP  – {display_name: api_key_name}  (for key lookup)
    * _MODEL_BASE_URL_MAP – {display_name: base_url_key}  (optional gateway URL)
    """
    from collections import OrderedDict
    registry: Dict[str, Tuple[str, str, str]] = {}
    provider_map: OrderedDict = OrderedDict()
    key_map: Dict[str, str] = {}
    base_url_map: Dict[str, str] = {}

    models_cfg = cfg.get("models", {})
    for provider, pdata in models_cfg.items():
        adapter_full = pdata.get("adapter", "")
        api_key_name = pdata.get("api_key_name", provider.lower())
        base_url_key = pdata.get("base_url_key", "")
        module_path, class_name = adapter_full.rsplit(".", 1)
        display_names = []
        for display_name, api_model_id in pdata.get("models", {}).items():
            registry[display_name] = (module_path, class_name, api_model_id)
            key_map[display_name] = api_key_name
            if base_url_key:
                base_url_map[display_name] = base_url_key
            display_names.append(display_name)
        if display_names:
            provider_map[provider] = display_names
    return registry, provider_map, key_map, base_url_map


MODEL_REGISTRY, PROVIDER_MAP, _MODEL_KEY_MAP, _MODEL_BASE_URL_MAP = _build_model_registry(_config)


def _create_llm_adapter(method: str, api_key: str):
    """Create the appropriate LLM adapter based on method (display name)."""
    import importlib
    entry = MODEL_REGISTRY.get(method)
    if entry is None:
        return None
    module_path, class_name, api_model_id = entry
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    base_url = _get_saved_base_url(method)
    try:
        return cls(api_key=api_key, model=api_model_id, base_url=base_url)
    except TypeError:
        return cls(api_key=api_key, model=api_model_id)


LLM_METHODS = list(MODEL_REGISTRY.keys())
ALL_METHODS = ["GeoSym"] + LLM_METHODS
PROVIDER_NAMES = list(PROVIDER_MAP.keys())
