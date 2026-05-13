"""
Disk persistence for single-image reasoning caches.

Each reasoning run (image + reasoning model) is stored as a JSON file under
``results/single_image_reasoning/`` using a rich analysis-oriented schema.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from app.reasoning_schema import SCHEMA_VERSION, validate_reasoning_record

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / "single_image_reasoning"


def _slugify(name: str) -> str:
    """Turn a human-readable name into a safe filesystem slug."""
    return re.sub(r"[^\w\-.]", "_", name.strip())


def _cache_path(image_name: str, model_name: str) -> Path:
    img_slug = _slugify(Path(image_name).stem)
    model_slug = _slugify(model_name)
    return _CACHE_DIR / f"{img_slug}__{model_slug}.json"


def save_reasoning_cache(record: Dict[str, Any]) -> Path:
    """Persist one single-image reasoning record to disk."""
    image_name = record.get("image_name")
    model_name = record.get("reasoning_model") or record.get("model_name")
    if not image_name or not model_name:
        raise ValueError("reasoning cache record requires image_name and reasoning_model")
    validation_errors = validate_reasoning_record(record)
    if validation_errors:
        joined = "; ".join(validation_errors[:8])
        raise ValueError(f"reasoning cache record schema validation failed: {joined}")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(image_name, model_name)
    data = dict(record)
    data["schema_version"] = SCHEMA_VERSION
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Saved reasoning cache → %s", path)
    return path


def load_reasoning_cache(
    image_name: str,
    model_name: str,
) -> Optional[Dict[str, Any]]:
    """Load cached single-image reasoning data, or return *None* if not found."""
    path = _cache_path(image_name, model_name)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            logger.info("Ignoring outdated reasoning cache (schema mismatch): %s", path)
            return None
        validation_errors = validate_reasoning_record(data)
        if validation_errors:
            logger.info("Ignoring invalid reasoning cache (schema validation failed): %s", path)
            return None
        logger.info("Loaded reasoning cache ← %s", path)
        return data
    except Exception as exc:
        logger.warning("Failed to load cache %s: %s", path, exc)
        return None


def failed_conditions(cached: Dict[str, Any]) -> list[str]:
    """Return condition keys whose status is NOT 'success'."""
    conds = cached.get("conditions", {})
    return [k for k, v in conds.items() if v.get("status") != "success"]


def clear_reasoning_cache(image_name: str, model_name: str) -> bool:
    """Delete the cache file for one image+model reasoning run."""
    path = _cache_path(image_name, model_name)
    if path.is_file():
        path.unlink()
        logger.info("Cleared reasoning cache: %s", path)
        return True
    return False


def clear_image_reasoning_cache(image_name: str) -> int:
    """Delete all reasoning cache files for one image across all models.

    Returns the number of files deleted.
    """
    if not _CACHE_DIR.is_dir():
        return 0
    img_slug = _slugify(Path(image_name).stem)
    count = 0
    for f in list(_CACHE_DIR.glob(f"{img_slug}__*.json")):
        f.unlink()
        logger.info("Cleared reasoning cache: %s", f)
        count += 1
    return count


def clear_all_reasoning_cache() -> int:
    """Delete all cached single-image reasoning results."""
    if not _CACHE_DIR.is_dir():
        return 0
    count = 0
    for f in list(_CACHE_DIR.glob("*.json")):
        f.unlink()
        logger.info("Cleared reasoning cache: %s", f)
        count += 1
    return count
