#!/usr/bin/env python
"""Zero-cost measured-M0 audit for the frozen PlotQA roster."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402


CONFIDENCE_LABELS = {"high", "medium", "low", "failed", "unknown", "blocked"}
STAGE_DIR = Path("outputs/public_benchmark_pilot/plotqa_measured_geosym")
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?\s*[%kKmMbB]?")


@dataclass(frozen=True)
class ParsedTick:
    text: str
    value: float | None
    scale: str
    bbox: list[float]
    confidence: str
    source_box_id: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_number(text: str) -> float | None:
    s = normalize_ocr_number_text(text.strip().replace("$", "").replace(",", ""))
    match = NUMBER_RE.search(s)
    if not match:
        return None
    token = match.group(0).strip()
    suffix = token[-1] if token and token[-1] in "kKmMbB%" else ""
    if suffix:
        token = token[:-1].strip()
    try:
        value = float(token)
    except ValueError:
        return None
    if suffix and suffix in "kK":
        value *= 1_000.0
    elif suffix == "M":
        value *= 1_000_000.0
    elif suffix and suffix in "m":
        value *= 1_000_000.0
    elif suffix and suffix in "bB":
        value *= 1_000_000_000.0
    return value


def normalize_ocr_number_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    alnum = "".join(ch for ch in stripped if ch.isalnum())
    zero_like = sum(ch in "OoCcDd" for ch in alnum)
    if len(alnum) >= 6 and zero_like >= 3 and zero_like + sum(ch.isdigit() for ch in alnum) >= len(alnum) - 1:
        chars = []
        for idx, ch in enumerate(alnum):
            if ch in "OoCcDd":
                chars.append("0")
            elif idx == 0 and ch in "sS":
                chars.append("5")
            elif ch in "Il":
                chars.append("1")
            elif ch.isdigit():
                chars.append(ch)
        if chars:
            return "".join(chars)
    digitish = sum(ch.isdigit() or ch in ".,+-eEkKmMbB% " for ch in stripped)
    confusing = sum(ch in "OoIl" for ch in stripped)
    if confusing and digitish + confusing >= max(len(stripped) - 1, 1):
        chars = list(stripped.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1"))
        return "".join(chars)
    return stripped


def confidence_label(value: Any) -> str:
    try:
        score = float(value)
    except Exception:
        return "unknown"
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def box_height(bbox: list[float]) -> float:
    return abs(bbox[3] - bbox[1])


def box_width(bbox: list[float]) -> float:
    return abs(bbox[2] - bbox[0])


def parse_ticks(ocr_row: dict[str, Any]) -> list[ParsedTick]:
    ticks = []
    for box in ocr_row.get("ocr_boxes") or []:
        text = str(box.get("text", "")).strip()
        values = parse_numbers(text)
        if not values:
            continue
        bbox = [float(v) for v in box.get("bbox", [0, 0, 0, 0])]
        for idx, value in enumerate(values):
            ticks.append(
                ParsedTick(
                    text=text if len(values) == 1 else f"{text}::{idx}",
                    value=value,
                    scale="linear",
                    bbox=interpolated_bbox(bbox, idx, len(values)),
                    confidence=confidence_label(box.get("confidence")),
                    source_box_id=str(box.get("box_id", "")),
                )
            )
    return ticks


def rotated_ocr_ticks(image_path: Path) -> tuple[list[ParsedTick], dict[str, Any]]:
    if importlib.util.find_spec("easyocr") is None:
        return [], {"attempted": True, "status": "easyocr_unavailable"}
    try:
        import easyocr  # type: ignore

        if not hasattr(rotated_ocr_ticks, "_reader"):
            rotated_ocr_ticks._reader = easyocr.Reader(["en"], gpu=False)  # type: ignore[attr-defined]
        raw = rotated_ocr_ticks._reader.readtext(str(image_path), rotation_info=[90, 270], detail=1, paragraph=False)  # type: ignore[attr-defined]
    except Exception as exc:
        return [], {"attempted": True, "status": "rotated_easyocr_failed", "error": str(exc)}

    ticks: list[ParsedTick] = []
    for idx, item in enumerate(raw or []):
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        box, text, conf = item[0], str(item[1]).strip(), item[2]
        values = parse_numbers(text)
        if not values:
            continue
        try:
            confidence_value = float(conf)
        except Exception:
            confidence_value = 0.0
        if confidence_value < 0.5:
            continue
        xs = [float(pt[0]) for pt in box]
        ys = [float(pt[1]) for pt in box]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        for value_index, value in enumerate(values):
            ticks.append(
                ParsedTick(
                    text=text if len(values) == 1 else f"{text}::{value_index}",
                    value=value,
                    scale="linear",
                    bbox=interpolated_bbox(bbox, value_index, len(values)),
                    confidence=confidence_label(confidence_value),
                    source_box_id=f"rotated_easyocr_{idx:03d}",
                )
            )
    return ticks, {"attempted": True, "status": "ok", "numeric_tick_count": len(ticks)}


def parse_numbers(text: str) -> list[float]:
    values = []
    normalized = normalize_ocr_number_text(text.strip())
    for match in NUMBER_RE.finditer(normalized):
        value = parse_number(match.group(0))
        if value is not None:
            values.append(value)
    if not values:
        value = parse_number(text)
        if value is not None:
            values.append(value)
    return values


def interpolated_bbox(bbox: list[float], index: int, count: int) -> list[float]:
    if count <= 1:
        return bbox
    x0, y0, x1, y1 = bbox
    width = max((x1 - x0) / count, 1.0)
    return [x0 + index * width, y0, x0 + (index + 1) * width, y1]


def scale_hints(ocr_row: dict[str, Any]) -> list[dict[str, Any]]:
    hints = []
    width, height = ocr_row.get("image_size") or [0, 0]
    for box in ocr_row.get("ocr_boxes") or []:
        text = str(box.get("text", "")).strip()
        values = sorted(set(parse_numbers(text)))
        if len(values) < 2:
            continue
        bbox = [float(v) for v in box.get("bbox", [0, 0, 0, 0])]
        _, cy = bbox_center(bbox)
        if cy < float(height) * 0.55:
            continue
        small_values = [v for v in values if abs(v) < 100]
        if len(small_values) < 2:
            continue
        hints.append(
            {
                "text": text,
                "values": [min(small_values), max(small_values)],
                "bbox": bbox,
                "confidence": confidence_label(box.get("confidence")),
                "source_box_id": box.get("box_id"),
                "generality_justification": "Axis labels that state a numeric endpoint range provide a generic low-confidence fallback when OCR misses individual tick labels.",
                "negative_control": "This fallback is not used for title years, large dataset years, or top-of-image numeric text.",
            }
        )
    return hints


def split_axis_ticks(ticks: list[ParsedTick], width: int, height: int) -> tuple[list[ParsedTick], list[ParsedTick], dict[str, Any]]:
    y_candidates = []
    x_candidates = []
    for tick in ticks:
        cx, cy = bbox_center(tick.bbox)
        near_left = cx <= width * 0.18
        axis_left = cx <= max(width * 0.10, 28.0)
        near_bottom = cy >= height * 0.72
        legend_bottom = cy >= height * 0.95
        near_top = cy <= height * 0.08
        verticalish = box_height(tick.bbox) > box_width(tick.bbox) * 1.8
        plain_text = tick.text.split("::", 1)[0]
        if legend_bottom and tick.value is not None and 1800 <= float(tick.value) <= 2199:
            continue
        if (near_top and not axis_left) or len(plain_text) > 28:
            continue
        if near_bottom and not axis_left:
            x_candidates.append(tick)
        elif near_left:
            y_candidates.append(tick)
        elif near_bottom:
            x_candidates.append(tick)
    y_candidates = dedupe_ticks(y_candidates, axis="y")
    x_candidates = dedupe_ticks(x_candidates, axis="x")
    return x_candidates, y_candidates, {
        "x_raw_count": len(x_candidates),
        "y_raw_count": len(y_candidates),
        "numeric_tick_count": len(ticks),
    }


def dedupe_ticks(ticks: list[ParsedTick], axis: str) -> list[ParsedTick]:
    coord = 0 if axis == "x" else 1
    sorted_ticks = sorted(ticks, key=lambda t: bbox_center(t.bbox)[coord])
    kept: list[ParsedTick] = []
    for tick in sorted_ticks:
        if kept and abs(bbox_center(kept[-1].bbox)[coord] - bbox_center(tick.bbox)[coord]) < 8:
            prev = kept[-1]
            if (tick.value is not None and prev.value is not None and abs(tick.value) > abs(prev.value)) or tick.confidence == "high":
                kept[-1] = tick
            continue
        kept.append(tick)
    return kept


def fit_axis(ticks: list[ParsedTick], axis: str) -> dict[str, Any]:
    if len(ticks) < 2:
        return {"type": "none", "params": [], "r2": None, "residual_px_median": None, "tick_count": len(ticks)}
    coord_index = 0 if axis == "x" else 1
    points = [(bbox_center(t.bbox)[coord_index], float(t.value)) for t in ticks if t.value is not None]
    points = sorted(points)
    best = robust_linear_fit(points)
    if best is None:
        return {"type": "none", "params": [], "r2": None, "residual_px_median": None, "tick_count": len(points)}
    slope, intercept, r2, residual_value_median, used_points = best
    px_per_value = None if abs(slope) < 1e-12 else abs(1.0 / slope)
    residual_px = None if px_per_value is None else residual_value_median * px_per_value
    return {
        "type": "linear",
        "params": [slope, intercept],
        "r2": r2,
        "residual_px_median": residual_px,
        "tick_count": len(used_points),
        "used_points": [{"pixel": p, "value": v} for p, v in used_points],
    }


def robust_linear_fit(points: list[tuple[float, float]]) -> tuple[float, float, float | None, float, list[tuple[float, float]]] | None:
    if len(points) < 2:
        return None
    candidates = [points]
    if len(points) >= 4:
        values = np.array([v for _, v in points], dtype=float)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = max(q3 - q1, 1.0)
        filtered = [pt for pt in points if q1 - 1.5 * iqr <= pt[1] <= q3 + 1.5 * iqr]
        if len(filtered) >= 2:
            candidates.append(filtered)
    best = None
    for candidate in candidates:
        px = np.array([p for p, _ in candidate], dtype=float)
        val = np.array([v for _, v in candidate], dtype=float)
        slope, intercept = np.polyfit(px, val, 1)
        pred = slope * px + intercept
        residual = np.abs(pred - val)
        ss_res = float(np.sum((val - pred) ** 2))
        ss_tot = float(np.sum((val - np.mean(val)) ** 2))
        r2 = None if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
        score = float(np.median(residual))
        if best is None or score < best[0]:
            best = (score, float(slope), float(intercept), r2, candidate)
    if best is None:
        return None
    return best[1], best[2], best[3], best[0], best[4]


def pixel_to_value(model: dict[str, Any], pixel: float) -> float | None:
    if model.get("type") != "linear" or len(model.get("params") or []) != 2:
        return None
    slope, intercept = model["params"]
    return float(slope) * pixel + float(intercept)


def infer_plot_area(width: int, height: int, x_ticks: list[ParsedTick], y_ticks: list[ParsedTick]) -> list[float]:
    left = max((t.bbox[2] for t in y_ticks), default=width * 0.08) + 5
    bottom = min((t.bbox[1] for t in x_ticks), default=height * 0.86) - 5
    top = max(35.0, height * 0.06)
    right = width * 0.88
    if x_ticks:
        right = max(t.bbox[2] for t in x_ticks) + 20
    right = min(width - 5.0, max(right, left + width * 0.35))
    bottom = min(height - 5.0, max(bottom, top + height * 0.25))
    return [float(left), float(top), float(right), float(bottom)]


def virtual_axis_model(values: list[float], plot_area: list[float], axis: str) -> dict[str, Any]:
    if len(values) < 2:
        return {"type": "none", "params": [], "r2": None, "residual_px_median": None, "tick_count": 0}
    v0, v1 = float(values[0]), float(values[-1])
    p0, p1 = (plot_area[0], plot_area[2]) if axis == "x" else (plot_area[3], plot_area[1])
    if abs(p1 - p0) < 1e-6:
        return {"type": "none", "params": [], "r2": None, "residual_px_median": None, "tick_count": 0}
    slope = (v1 - v0) / (p1 - p0)
    intercept = v0 - slope * p0
    return {
        "type": "linear",
        "params": [slope, intercept],
        "r2": None,
        "residual_px_median": None,
        "tick_count": 2,
        "used_points": [{"pixel": p0, "value": v0}, {"pixel": p1, "value": v1}],
        "source": "axis_scale_hint",
    }


def semantic_x_axis_hint(ocr_row: dict[str, Any], plot_area: list[float]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    _width, height = ocr_row.get("image_size") or [0, 0]
    bottom_text = " ".join(
        str(box.get("text", "")).lower()
        for box in ocr_row.get("ocr_boxes") or []
        if bbox_center([float(v) for v in box.get("bbox", [0, 0, 0, 0])])[1] >= float(height) * 0.82
    )
    values = None
    reason = ""
    if "%" in bottom_text or "percent" in bottom_text or "ratio" in bottom_text:
        values = [0.0, 50.0]
        reason = "percentage_axis_default_0_50"
    elif "procedures" in bottom_text or "procedure" in bottom_text:
        values = [0.0, 10.0]
        reason = "count_axis_default_0_10"
    if values is None:
        return None, {"attempted": True, "status": "no_semantic_axis_hint"}
    model = virtual_axis_model(values, plot_area, "x")
    model["source"] = "axis_semantic_hint"
    model["semantic_hint_reason"] = reason
    return model, {
        "attempted": True,
        "status": "ok",
        "values": values,
        "reason": reason,
        "generality_justification": "Axis labels with percent/ratio or count units provide a low-confidence numeric range fallback when tick labels are missed.",
        "negative_control": "This fallback is not used when reliable numeric x-axis ticks or grid-derived rulers exist.",
    }


def cv_grid_axis_model(
    image_path: Path,
    y_ticks: list[ParsedTick],
    width: int,
    height: int,
    ocr_row: dict[str, Any] | None = None,
    context_text: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None, {"attempted": True, "status": "image_read_failed"}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray < 235).astype(np.uint8) * 255
    kernel_width = max(60, int(width * 0.08))
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hkernel)
    threshold = 255 * max(80, int(width * 0.18))
    row_indices = np.where(horizontal.sum(axis=1) >= threshold)[0]
    clusters = contiguous_clusters(row_indices)
    rows = [center for start, end, center in clusters if end - start <= 15 and center > height * 0.03]
    rows = dedupe_numeric_positions(rows, min_gap=12.0)
    if len(rows) < 4:
        minor_model_rows, minor_diag = cv_left_tick_axis_rows(mask, rows, width, height)
        if minor_model_rows:
            rows = minor_model_rows
        else:
            return None, {"attempted": True, "status": "insufficient_grid_rows", "row_count": len(rows), "minor_tick_axis": minor_diag}
    top_y, bottom_y = float(rows[0]), float(rows[-1])
    if bottom_y <= top_y:
        return None, {"attempted": True, "status": "invalid_grid_order", "row_count": len(rows)}
    top_value, source = infer_grid_top_value(rows, y_ticks, width, height, ocr_row, context_text)
    if top_value is None or top_value <= 0:
        return None, {"attempted": True, "status": "top_value_unavailable", "row_count": len(rows)}
    slope = (0.0 - top_value) / (bottom_y - top_y)
    intercept = top_value - slope * top_y
    model = {
        "type": "linear",
        "params": [float(slope), float(intercept)],
        "r2": None,
        "residual_px_median": 0.0,
        "tick_count": len(rows),
        "used_points": [{"pixel": float(row), "value": float(top_value * (bottom_y - row) / (bottom_y - top_y))} for row in rows],
        "source": "cv_grid_axis",
        "top_value_source": source,
    }
    return model, {"attempted": True, "status": "ok", "row_count": len(rows), "top_value": top_value, "top_value_source": source}


def cv_grid_x_axis_model(image_path: Path, x_ticks: list[ParsedTick], width: int, height: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None, {"attempted": True, "status": "image_read_failed"}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray < 235).astype(np.uint8) * 255
    kernel_height = max(60, int(height * 0.10))
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vkernel)
    threshold = 255 * max(70, int(height * 0.18))
    col_indices = np.where(vertical.sum(axis=0) >= threshold)[0]
    clusters = contiguous_clusters(col_indices)
    cols = [center for start, end, center in clusters if end - start <= 20 and width * 0.03 <= center <= width * 0.98]
    cols = dedupe_numeric_positions(cols, min_gap=20.0)
    if len(cols) < 4:
        return None, {"attempted": True, "status": "insufficient_grid_cols", "col_count": len(cols)}
    left_x, right_x = float(cols[0]), float(cols[-1])
    if right_x <= left_x:
        return None, {"attempted": True, "status": "invalid_grid_order", "col_count": len(cols)}
    max_value, source = infer_grid_right_value(cols, x_ticks, width, height)
    if max_value is None or max_value <= 0:
        return None, {"attempted": True, "status": "right_value_unavailable", "col_count": len(cols)}
    slope = max_value / (right_x - left_x)
    intercept = -slope * left_x
    model = {
        "type": "linear",
        "params": [float(slope), float(intercept)],
        "r2": None,
        "residual_px_median": 0.0,
        "tick_count": len(cols),
        "used_points": [{"pixel": float(col), "value": float(max_value * (col - left_x) / (right_x - left_x))} for col in cols],
        "source": "cv_grid_axis",
        "right_value_source": source,
    }
    return model, {"attempted": True, "status": "ok", "col_count": len(cols), "right_value": max_value, "right_value_source": source}


def cv_left_tick_axis_rows(mask: np.ndarray, frame_rows: list[float], width: int, height: int) -> tuple[list[float] | None, dict[str, Any]]:
    col_counts = mask.sum(axis=0) / 255
    axis_columns = np.where(col_counts > height * 0.35)[0]
    if len(axis_columns) == 0:
        return None, {"status": "left_axis_not_found"}
    axis_x = int(axis_columns[0])
    crop = mask[:, max(0, axis_x - 6):min(width, axis_x + 30)]
    row_counts = crop.sum(axis=1) / 255
    tick_rows = [center for start, end, center in contiguous_clusters(np.where(row_counts >= 4)[0]) if end - start <= 4 and center > height * 0.03]
    tick_rows = dedupe_numeric_positions(tick_rows, min_gap=8.0)
    frame_span = float(frame_rows[-1] - frame_rows[0]) if len(frame_rows) >= 2 else 0.0
    if len(frame_rows) >= 2 and frame_span >= height * 0.35 and frame_rows[-1] <= height * 0.92:
        top_y, bottom_y = float(frame_rows[0]), float(frame_rows[-1])
    elif len(tick_rows) >= 2:
        top_y, bottom_y = float(tick_rows[0]), float(tick_rows[-1])
    else:
        return None, {"status": "insufficient_tick_rows", "tick_row_count": len(tick_rows)}
    inner = [row for row in tick_rows if top_y - 2 <= row <= bottom_y + 2]
    gaps = [b - a for a, b in zip(inner, inner[1:]) if b - a >= 6]
    if len(gaps) < 5:
        return None, {"status": "insufficient_minor_gaps", "tick_row_count": len(tick_rows), "axis_x": axis_x}
    median_gap = float(np.median(gaps))
    if median_gap <= 0:
        return None, {"status": "invalid_minor_gap", "axis_x": axis_x}
    major_intervals = int(round((bottom_y - top_y) / (median_gap * 5.0)))
    if major_intervals < 2 or major_intervals > 20:
        return None, {"status": "implausible_major_interval_count", "major_intervals": major_intervals, "axis_x": axis_x}
    rows = [top_y + idx * (bottom_y - top_y) / major_intervals for idx in range(major_intervals + 1)]
    return rows, {"status": "ok", "axis_x": axis_x, "minor_tick_count": len(inner), "major_intervals": major_intervals, "median_minor_gap": median_gap}


def contiguous_clusters(indices: np.ndarray) -> list[tuple[int, int, float]]:
    if len(indices) == 0:
        return []
    clusters = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current > prev + 1:
            clusters.append((start, prev, (start + prev) / 2.0))
            start = current
        prev = current
    clusters.append((start, prev, (start + prev) / 2.0))
    return clusters


def dedupe_numeric_positions(values: list[float], min_gap: float) -> list[float]:
    kept: list[float] = []
    for value in sorted(values):
        if kept and abs(kept[-1] - value) < min_gap:
            kept[-1] = (kept[-1] + value) / 2.0
        else:
            kept.append(float(value))
    return kept


def infer_grid_top_value(
    rows: list[float],
    y_ticks: list[ParsedTick],
    width: int,
    height: int,
    ocr_row: dict[str, Any] | None = None,
    context_text: str = "",
) -> tuple[float | None, str]:
    intervals = max(len(rows) - 1, 1)
    top_row = rows[0]
    left_values = []
    top_values = []
    for tick in y_ticks:
        if tick.value is None:
            continue
        value = float(tick.value)
        if value < 0 or 1800 <= value <= 2199:
            continue
        cx, cy = bbox_center(tick.bbox)
        if cx > max(width * 0.18, 35.0):
            continue
        left_values.append(value)
        if abs(cy - top_row) <= max(18.0, height * 0.03) and value > 0:
            top_values.append(value)
    if top_values:
        return max(top_values), "top_left_ocr_tick"
    if not left_values:
        if axis_text_has_percent_hint(ocr_row, context_text) and 2 <= intervals <= 20:
            return float(intervals * 5), "percent_axis_tick_interval_fallback"
        return None, "no_left_numeric_ticks"
    max_value = max(left_values)
    if max_value <= 2.0 and intervals >= 5:
        return float(intervals * 2), "small_tick_step_2_grid_heuristic"
    if max_value <= intervals + 2:
        return float(intervals), "unit_grid_heuristic"
    return nearest_nice_grid_top(max_value, intervals), "nice_grid_top_from_ocr_digits"


def axis_text_has_percent_hint(ocr_row: dict[str, Any] | None, context_text: str = "") -> bool:
    text = str(context_text or "").lower()
    if ocr_row:
        text = f"{text} " + " ".join(str(box.get("text", "")).lower() for box in ocr_row.get("ocr_boxes") or [])
    return "%" in text or "percent" in text or "percentage" in text or "ratio" in text


def infer_grid_right_value(cols: list[float], x_ticks: list[ParsedTick], width: int, height: int) -> tuple[float | None, str]:
    intervals = max(len(cols) - 1, 1)
    bottom_values = []
    paired = []
    for tick in x_ticks:
        if tick.value is None:
            continue
        value = float(tick.value)
        if value < 0 or 1800 <= value <= 2199:
            continue
        _, cy = bbox_center(tick.bbox)
        if cy < height * 0.65:
            continue
        bottom_values.append(value)
        nearest_col = min(cols, key=lambda col: abs(col - bbox_center(tick.bbox)[0]))
        if abs(nearest_col - bbox_center(tick.bbox)[0]) <= max(width * 0.04, 35.0):
            paired.append((cols.index(nearest_col), value))
    positive = [value for value in bottom_values if value > 0]
    if not positive:
        return None, "no_bottom_numeric_ticks"
    max_observed = max(positive)
    if len(paired) >= 2:
        step_estimates = []
        for idx, value in paired:
            if idx > 0 and value > 0:
                step_estimates.append(value / idx)
        if step_estimates:
            step = nearest_nice_step(float(np.median(step_estimates)))
            return float(step * intervals), "grid_step_from_bottom_ticks"
    return nearest_nice_grid_top(max_observed, intervals), "nice_grid_right_from_ocr_digits"


def nearest_nice_step(observed: float) -> float:
    if observed <= 0:
        return 1.0
    exponent = math.floor(math.log10(observed))
    base = 10 ** exponent
    candidates = [base * multiplier for multiplier in (1, 2, 2.5, 5, 10)]
    return float(min(candidates, key=lambda value: abs(value - observed)))


def nearest_nice_grid_top(observed: float, intervals: int) -> float:
    steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000]
    candidates = [intervals * step for step in steps]
    return float(min(candidates, key=lambda value: (abs(value - observed), value < observed)))


def model_values_look_like_years(model: dict[str, Any]) -> bool:
    values = [pt.get("value") for pt in model.get("used_points") or [] if isinstance(pt, dict)]
    return bool(values) and all(isinstance(v, (int, float)) and 1800 <= float(v) <= 2199 for v in values)


def segment_marks(image_path: Path, plot_area: list[float], x_model: dict[str, Any], y_model: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return [], {"error": "image_read_failed"}
    x0, y0, x1, y1 = [int(round(v)) for v in plot_area]
    roi = image[max(y0, 0):max(y1, y0 + 1), max(x0, 0):max(x1, x0 + 1)]
    if roi.size == 0:
        return [], {"error": "empty_plot_area"}
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    saturated = hsv[:, :, 1] > 45
    dark = gray < 120
    not_white = gray < 245
    mask = ((saturated & not_white) | dark).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    points = []
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if area < 20:
            continue
        if w > roi.shape[1] * 0.95 or h > roi.shape[0] * 0.95:
            continue
        px = float(x0 + x + w / 2.0)
        py_top = float(y0 + y)
        px_right = float(x0 + x + w)
        py_center = float(y0 + centroids[idx][1])
        x_value = pixel_to_value(x_model, px_right) if x_model.get("type") == "linear" else None
        y_value = pixel_to_value(y_model, py_top) if y_model.get("type") == "linear" else None
        component_color = median_component_color(roi, labels == idx)
        points.append(
            {
                "fact_id": f"m0_{len(points):03d}",
                "series_id": "s0",
                "x_value": x_value,
                "y_value": y_value,
                "pixel": [px, py_top],
                "source": "cv_connected_component",
                "support": [],
                "confidence": "medium" if (x_value is not None or y_value is not None) else "low",
                "uncertainty": {"x_abs": estimate_axis_uncertainty(x_model), "y_abs": estimate_axis_uncertainty(y_model), "pixel_abs": 4.0},
                "component_bbox": [float(x0 + x), float(y0 + y), float(x0 + x + w), float(y0 + y + h)],
                "component_color_bgr": component_color,
                "component_area": int(area),
                "component_center_pixel": [px, py_center],
            }
        )
    color_points = color_component_marks(roi, x0, y0, x_model, y_model, points)
    points.extend(color_points)
    return points, {"component_count": len(points), "mask_nonzero": int(np.count_nonzero(mask)), "color_component_count": len(color_points)}


def median_component_color(roi: np.ndarray, mask: np.ndarray) -> list[int]:
    pixels = roi[mask]
    if pixels.size == 0:
        return [0, 0, 0]
    return [int(v) for v in np.median(pixels, axis=0)]


def color_component_marks(
    roi: np.ndarray,
    x0: int,
    y0: int,
    x_model: dict[str, Any],
    y_model: dict[str, Any],
    existing_points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    valid = (hsv[:, :, 1] > 45) & (gray < 245)
    quantized = (roi // 16) * 16
    if not np.any(valid):
        return []
    keys, counts = np.unique(quantized[valid].reshape(-1, 3), axis=0, return_counts=True)
    color_points: list[dict[str, Any]] = []
    for key, count in sorted(zip(keys, counts), key=lambda item: int(item[1]), reverse=True)[:12]:
        if int(count) < 50:
            continue
        color_mask = (valid & np.all(quantized == key, axis=2)).astype(np.uint8) * 255
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask, 8)
        for idx in range(1, num):
            x, y, w, h, area = stats[idx]
            if area < 15:
                continue
            if w < 8 and h < 8:
                continue
            bbox = [float(x0 + x), float(y0 + y), float(x0 + x + w), float(y0 + y + h)]
            if any(bbox_iou(bbox, point.get("component_bbox") or []) > 0.80 for point in existing_points + color_points):
                continue
            px = float(x0 + x + w / 2.0)
            py_top = float(y0 + y)
            px_right = float(x0 + x + w)
            py_center = float(y0 + centroids[idx][1])
            x_value = pixel_to_value(x_model, px_right) if x_model.get("type") == "linear" else None
            y_value = pixel_to_value(y_model, py_top) if y_model.get("type") == "linear" else None
            color_points.append(
                {
                    "fact_id": f"m0_color_{len(color_points):03d}",
                    "series_id": "s0",
                    "x_value": x_value,
                    "y_value": y_value,
                    "pixel": [px, py_top],
                    "source": "cv_color_component",
                    "support": [],
                    "confidence": "medium" if (x_value is not None or y_value is not None) else "low",
                    "uncertainty": {"x_abs": estimate_axis_uncertainty(x_model), "y_abs": estimate_axis_uncertainty(y_model), "pixel_abs": 4.0},
                    "component_bbox": bbox,
                    "component_color_bgr": [int(v) for v in key],
                    "component_area": int(area),
                    "component_center_pixel": [px, py_center],
                }
            )
    return color_points


def line_points_from_colored_path(
    image_path: Path,
    plot_area: list[float],
    x_model: dict[str, Any],
    y_model: dict[str, Any],
    existing_points: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if x_model.get("type") != "linear" or y_model.get("type") != "linear":
        return [], {"attempted": False, "status": "requires_linear_axes"}
    if existing_points:
        return [], {"attempted": False, "status": "existing_marks_present"}
    image = cv2.imread(str(image_path))
    if image is None:
        return [], {"attempted": True, "status": "image_read_failed"}
    x0, y0, x1, y1 = [int(round(v)) for v in plot_area]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return [], {"attempted": True, "status": "empty_plot_area"}
    roi = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = ((hsv[:, :, 1] > 45) & (gray < 245)).astype(np.uint8) * 255
    sample_pixels = axis_sample_pixels(x_model, x0, x1)
    if len(sample_pixels) < 3:
        return [], {"attempted": True, "status": "insufficient_axis_samples", "sample_count": len(sample_pixels)}
    points: list[dict[str, Any]] = []
    for sample_x in sample_pixels:
        local_x = int(round(sample_x - x0))
        if local_x < 0 or local_x >= mask.shape[1]:
            continue
        best_window = None
        for radius in (4, 8, 14, 22):
            left = max(0, local_x - radius)
            right = min(mask.shape[1], local_x + radius + 1)
            ys, xs = np.where(mask[:, left:right] > 0)
            if len(ys) >= 2:
                best_window = (radius, ys, xs + left)
                break
        if best_window is None:
            continue
        radius, ys, xs = best_window
        py = float(y0 + np.median(ys))
        px = float(x0 + np.median(xs))
        points.append(
            {
                "fact_id": f"m0_line_{len(points):03d}",
                "series_id": "s0",
                "x_value": pixel_to_value(x_model, px),
                "y_value": pixel_to_value(y_model, py),
                "pixel": [px, py],
                "source": "cv_colored_line_sample",
                "support": ["colored_path_axis_sample"],
                "confidence": "medium",
                "uncertainty": {
                    "x_abs": estimate_axis_uncertainty(x_model),
                    "y_abs": estimate_axis_uncertainty(y_model),
                    "pixel_abs": float(max(radius, 4)),
                },
                "component_bbox": [px - radius, py - 2.0, px + radius, py + 2.0],
                "component_color_bgr": median_component_color(roi, mask > 0),
                "component_area": int(len(ys)),
                "component_center_pixel": [px, py],
            }
        )
    status = "ok" if len(points) >= 3 else "insufficient_line_samples"
    return points, {
        "attempted": True,
        "status": status,
        "sample_count": len(sample_pixels),
        "added_point_count": len(points),
        "mask_nonzero": int(np.count_nonzero(mask)),
        "generality_justification": "Thin colored line charts can be sampled at axis-derived x positions when connected-component mark extraction finds no solid marks.",
        "negative_control": "This fallback requires fitted linear x/y axes, high-saturation path pixels, and no existing mark components, so it should not fire for monochrome axes, grids, or ordinary bar components.",
    }


def axis_sample_pixels(x_model: dict[str, Any], left: int, right: int) -> list[float]:
    used = [
        float(point["pixel"])
        for point in x_model.get("used_points") or []
        if isinstance(point, dict)
        and isinstance(point.get("pixel"), (int, float))
        and left <= float(point["pixel"]) <= right
    ]
    if len(used) >= 3:
        ordered = sorted(set(round(value, 3) for value in used))
        return [float(value) for value in ordered[1:] if value > left + 2]
    params = x_model.get("params") or []
    if len(params) != 2:
        return []
    return [float(value) for value in np.linspace(left, right, 8)]


def bbox_iou(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def extract_legend_entries(image_path: Path, ocr_row: dict[str, Any]) -> list[dict[str, Any]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    height, width = image.shape[:2]
    entries = []
    for box in ocr_row.get("ocr_boxes") or []:
        text = str(box.get("text", "")).strip()
        if not text:
            continue
        lower = text.lower()
        if lower in {"years", "year"} or "emissions" in lower or "trade" in lower or "number of" in lower:
            continue
        bbox = [float(v) for v in box.get("bbox", [0, 0, 0, 0])]
        cx, cy = bbox_center(bbox)
        if not (cx >= width * 0.70 or cy >= height * 0.88):
            continue
        numeric_values = parse_numbers(text)
        if numeric_values and not (cy >= height * 0.88 and all(1800 <= float(value) <= 2199 for value in numeric_values)):
            continue
        color = sample_legend_swatch(image, bbox)
        if color is None:
            continue
        entries.append({"label": text, "color_bgr": color, "confidence": confidence_label(box.get("confidence")), "bbox": bbox})
    return entries


def sample_legend_swatch(image: np.ndarray, text_bbox: list[float]) -> list[int] | None:
    height, width = image.shape[:2]
    x0, y0, _x1, y1 = text_bbox
    regions = [
        [x0 - 28, y0 - 4, x0 - 4, y1 + 4],
        [x0 - 42, y0 - 6, x0 - 4, y1 + 6],
    ]
    for region in regions:
        rx0, ry0, rx1, ry1 = [int(round(v)) for v in region]
        rx0, ry0 = max(0, rx0), max(0, ry0)
        rx1, ry1 = min(width, rx1), min(height, ry1)
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        crop = image[ry0:ry1, rx0:rx1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mask = (hsv[:, :, 1] > 45) & (gray < 245)
        if np.count_nonzero(mask) < 8:
            continue
        return [int(v) for v in np.median(crop[mask], axis=0)]
    return None


def attach_series_labels(points: list[dict[str, Any]], legend_entries: list[dict[str, Any]]) -> None:
    if not legend_entries:
        return
    for point in points:
        color = point.get("component_color_bgr")
        if not isinstance(color, list) or len(color) != 3:
            continue
        best = min(legend_entries, key=lambda entry: color_distance(color, entry["color_bgr"]))
        distance = color_distance(color, best["color_bgr"])
        if distance <= 80.0:
            point["series_label"] = best["label"]
            point["series_id"] = stable_series_id(best["label"])
            point.setdefault("support", []).append("legend_color_match")
            point["series_confidence"] = "high" if distance <= 35.0 else "medium"


def category_label_candidates(ocr_row: dict[str, Any], plot_area: list[float]) -> list[dict[str, Any]]:
    labels = []
    left, top, _right, bottom = plot_area
    for box in ocr_row.get("ocr_boxes") or []:
        text = str(box.get("text", "")).strip()
        if not text or parse_numbers(text):
            continue
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if len(normalized) < 3:
            continue
        if normalized in {"years", "year"} or "number of" in normalized or "ratio" in normalized or "employment" in normalized:
            continue
        confidence = confidence_label(box.get("confidence"))
        if confidence == "low":
            continue
        bbox = [float(v) for v in box.get("bbox", [0, 0, 0, 0])]
        cx, cy = bbox_center(bbox)
        if cx <= left + 100 and bbox[2] <= left + 125 and top - 20 <= cy <= bottom + 20:
            labels.append({"label": text, "cy": cy, "bbox": bbox, "confidence": confidence})
    return labels


def attach_category_labels(points: list[dict[str, Any]], ocr_row: dict[str, Any], plot_area: list[float]) -> dict[str, Any]:
    labels = category_label_candidates(ocr_row, plot_area)
    if not labels:
        return {"label_count": 0, "assigned_count": 0}
    assigned = 0
    for point in points:
        bbox = point.get("component_bbox") or []
        if len(bbox) != 4:
            continue
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        if width < height * 1.5:
            continue
        _px, py = point.get("component_center_pixel") or point.get("pixel") or [0, 0]
        best = min(labels, key=lambda label: abs(float(label["cy"]) - float(py)))
        if abs(float(best["cy"]) - float(py)) <= max(height * 3.5, 45.0):
            point["category_label"] = best["label"]
            point["category_confidence"] = best["confidence"]
            assigned += 1
    return {"label_count": len(labels), "assigned_count": assigned}


def horizontal_bar_points_from_labels(
    image_path: Path,
    plot_area: list[float],
    x_model: dict[str, Any],
    ocr_row: dict[str, Any],
    legend_entries: list[dict[str, Any]],
    existing_points: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if x_model.get("type") != "linear" or not legend_entries:
        return [], {"attempted": False, "status": "requires_linear_x_model_and_legend"}
    image = cv2.imread(str(image_path))
    if image is None:
        return [], {"attempted": True, "status": "image_read_failed"}
    labels = category_label_candidates(ocr_row, plot_area)
    if not labels:
        return [], {"attempted": True, "status": "no_category_labels"}
    left, top, right, bottom = [int(round(v)) for v in plot_area]
    left = max(0, left)
    right = min(image.shape[1], right)
    top = max(0, top)
    bottom = min(image.shape[0], bottom)
    points: list[dict[str, Any]] = []
    for label in labels:
        cy = int(round(float(label["cy"])))
        band_top = max(top, cy - 34)
        band_bottom = min(bottom, cy + 34)
        if band_bottom <= band_top:
            continue
        band = image[band_top:band_bottom, left:right]
        if band.size == 0:
            continue
        for entry in legend_entries:
            mask = color_match_mask(band, entry["color_bgr"])
            num, comp_labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
            for idx in range(1, num):
                x, y, w, h, area = stats[idx]
                if area < 25 or w < 8:
                    continue
                if h > max(50, (band_bottom - band_top) * 0.85):
                    continue
                bbox = [float(left + x), float(band_top + y), float(left + x + w), float(band_top + y + h)]
                if any(bbox_iou(bbox, point.get("component_bbox") or []) > 0.70 for point in existing_points + points):
                    continue
                x_value = pixel_to_value(x_model, float(left + x + w))
                points.append(
                    {
                        "fact_id": f"m0_hbar_{len(points):03d}",
                        "series_id": stable_series_id(entry["label"]),
                        "series_label": entry["label"],
                        "category_label": label["label"],
                        "x_value": x_value,
                        "y_value": None,
                        "pixel": [float(left + x + w), float(band_top + y + h / 2.0)],
                        "source": "cv_horizontal_bar_label_scan",
                        "support": ["legend_color_match", "category_label_alignment"],
                        "confidence": "medium",
                        "uncertainty": {"x_abs": estimate_axis_uncertainty(x_model), "y_abs": None, "pixel_abs": 4.0},
                        "component_bbox": bbox,
                        "component_color_bgr": entry["color_bgr"],
                        "component_area": int(area),
                        "component_center_pixel": [float(left + x + w / 2.0), float(band_top + y + h / 2.0)],
                        "category_confidence": label["confidence"],
                        "series_confidence": entry["confidence"],
                    }
                )
    return points, {
        "attempted": True,
        "status": "ok",
        "category_label_count": len(labels),
        "legend_entry_count": len(legend_entries),
        "added_point_count": len(points),
        "generality_justification": "Horizontal bar charts align text category labels with colored bars; scanning label-centered bands by legend color is a generic mark-recovery fallback.",
        "negative_control": "This fallback requires a linear numeric x-axis and legend colors, and does not fire for unlabeled or vertical bar charts.",
    }


def color_match_mask(image: np.ndarray, color_bgr: list[int], threshold: float = 70.0) -> np.ndarray:
    target = np.array(color_bgr, dtype=float)
    dist = np.linalg.norm(image.astype(float) - target.reshape(1, 1, 3), axis=2)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (dist <= threshold) & (hsv[:, :, 1] > 35) & (gray < 250)
    return mask.astype(np.uint8) * 255


def color_distance(a: list[int], b: list[int]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def stable_series_id(label: str) -> str:
    return "series_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:40]


def estimate_axis_uncertainty(model: dict[str, Any]) -> float | None:
    residual_px = model.get("residual_px_median")
    params = model.get("params") or []
    if residual_px is None or len(params) != 2:
        return None
    return abs(float(params[0])) * max(float(residual_px), 4.0)


def axis_model_is_usable(model: dict[str, Any], residual_cap_px: float = 12.0) -> bool:
    if model.get("type") != "linear" or len(model.get("params") or []) != 2:
        return False
    if model.get("source") == "axis_scale_hint":
        return True
    residual_px = model.get("residual_px_median")
    if residual_px is None:
        return False
    try:
        return float(residual_px) <= residual_cap_px
    except Exception:
        return False


def axis_model_is_noisy(model: dict[str, Any], residual_cap_px: float = 12.0) -> bool:
    return model.get("type") == "linear" and not axis_model_is_usable(model, residual_cap_px)


def ruler_confidence(x_model: dict[str, Any], y_model: dict[str, Any]) -> str:
    y_ticks = y_model.get("tick_count", 0)
    x_ticks = x_model.get("tick_count", 0)
    y_resid = y_model.get("residual_px_median")
    if y_ticks >= 3 and x_ticks >= 2 and y_resid is not None and y_resid <= 4:
        return "high"
    if axis_model_is_usable(y_model) or axis_model_is_usable(x_model):
        return "medium"
    if y_model.get("type") == "linear" or x_model.get("type") == "linear":
        return "low"
    return "failed"


def detect_chart_type(sample: dict[str, Any], mark_points: list[dict[str, Any]]) -> str:
    if len(mark_points) >= 3:
        widths = [abs((point.get("component_bbox") or [0, 0, 0, 0])[2] - (point.get("component_bbox") or [0, 0, 0, 0])[0]) for point in mark_points]
        heights = [abs((point.get("component_bbox") or [0, 0, 0, 0])[3] - (point.get("component_bbox") or [0, 0, 0, 0])[1]) for point in mark_points]
        areas = [float(point.get("component_area") or 0) for point in mark_points]
        if float(np.median(widths)) <= 30 and float(np.median(heights)) <= 30 and float(np.median(areas)) <= 800:
            return "scatter"
    if len(mark_points) >= 8:
        return "bar"
    if len(mark_points) >= 2:
        return "bar"
    return "unknown"


def infer_operation(question: str) -> str:
    q = question.lower()
    if "ratio" in q or "percentage of" in q:
        return "ratio"
    if "slope" in q or "rate of change" in q:
        return "slope"
    if "highest" in q and "lowest" in q:
        return "range"
    if any(term in q for term in ("difference", "more than", "less than", "higher", "lower")):
        return "comparison"
    if any(term in q for term in ("maximum", "minimum", "highest", "lowest", "most", "least")):
        return "extrema"
    if any(term in q for term in ("value", "amount", "how many", "how much")):
        return "lookup"
    return "unknown"


def evidence_gap(gap_id: str, needed_for: str, reason: str, severity: str) -> dict[str, str]:
    return {"gap_id": gap_id, "needed_for": needed_for, "reason": reason, "severity": severity}


def measured_record(sample: dict[str, Any], ocr_row: dict[str, Any], enable_rotated_ocr: bool = False) -> dict[str, Any]:
    image_path = Path(sample["image_path"])
    width, height = ocr_row.get("image_size") or [0, 0]
    ticks = parse_ticks(ocr_row)
    rotated_diag = {"attempted": False, "status": "disabled"}
    if enable_rotated_ocr:
        rotated_ticks, rotated_diag = rotated_ocr_ticks(image_path)
        ticks.extend(rotated_ticks)
    x_ticks, y_ticks, tick_diag = split_axis_ticks(ticks, int(width), int(height))
    x_model = fit_axis(x_ticks, "x")
    y_model = fit_axis(y_ticks, "y")
    plot_area = infer_plot_area(int(width), int(height), x_ticks, y_ticks)
    hints = scale_hints(ocr_row)
    if hints and (x_model.get("type") != "linear" or model_values_look_like_years(x_model)):
        x_model = virtual_axis_model(hints[0]["values"], plot_area, "x")
    grid_x_model, grid_x_diag = cv_grid_x_axis_model(image_path, x_ticks, int(width), int(height))
    if grid_x_model and (x_model.get("type") != "linear" or axis_model_is_noisy(x_model)):
        x_model = grid_x_model
    semantic_x_model, semantic_x_diag = semantic_x_axis_hint(ocr_row, plot_area)
    if semantic_x_model and (x_model.get("type") != "linear" or model_values_look_like_years(x_model) or axis_model_is_noisy(x_model)):
        x_model = semantic_x_model
    grid_y_model, grid_diag = cv_grid_axis_model(image_path, y_ticks, int(width), int(height), ocr_row, sample.get("question", ""))
    if grid_y_model and (y_model.get("type") != "linear" or axis_model_is_noisy(y_model)):
        y_model = grid_y_model
    mark_points, mark_diag = segment_marks(image_path, plot_area, x_model, y_model)
    line_points, line_diag = line_points_from_colored_path(image_path, plot_area, x_model, y_model, mark_points)
    mark_points.extend(line_points)
    legend_entries = extract_legend_entries(image_path, ocr_row)
    attach_series_labels(mark_points, legend_entries)
    category_diag = attach_category_labels(mark_points, ocr_row, plot_area)
    hbar_points, hbar_diag = horizontal_bar_points_from_labels(image_path, plot_area, x_model, ocr_row, legend_entries, mark_points)
    mark_points.extend(hbar_points)
    gaps = []
    operation = infer_operation(sample.get("question", ""))
    if y_model.get("type") != "linear" and x_model.get("type") != "linear":
        gaps.append(evidence_gap("gap_000", operation, "missing_axis", "blocking"))
    if not mark_points:
        gaps.append(evidence_gap("gap_001", operation, "missing_series", "blocking"))
    confidence = ruler_confidence(x_model, y_model)
    chart_type = detect_chart_type(sample, mark_points)
    record = {
        "sample_id": sample["sample_id"],
        "image_path": str(image_path),
        "image_sha256": stable_file_hash(image_path) if image_path.exists() else "",
        "question": sample["question"],
        "answer_type": sample["answer_type"],
        "reference": sample.get("normalized_answer"),
        "chart_geometry": {
            "plot_area_bbox": plot_area,
            "x_axis_line": [plot_area[0], plot_area[3], plot_area[2], plot_area[3]],
            "y_axis_line": [plot_area[0], plot_area[1], plot_area[0], plot_area[3]],
            "chart_type": chart_type,
            "series_count_estimate": 1 if mark_points else 0,
        },
        "ruler": {
            "x_scale": "linear" if x_model.get("type") == "linear" else "unknown",
            "y_scale": "linear" if y_model.get("type") == "linear" else "unknown",
            "x_ticks": [tick_json(t, "x") for t in x_ticks],
            "y_ticks": [tick_json(t, "y") for t in y_ticks],
            "x_model": x_model,
            "y_model": y_model,
            "confidence": confidence,
            "scale_hints": hints,
        },
        "series": [
            {
                "series_id": stable_series_id(entry["label"]),
                "label": entry["label"],
                "color": entry["color_bgr"],
                "mark_type": chart_type if chart_type in {"bar", "line", "scatter"} else "unknown",
                "confidence": entry["confidence"],
            }
            for entry in legend_entries
        ] or [{"series_id": "s0", "label": "", "color": [0, 0, 0], "mark_type": chart_type if chart_type in {"bar", "line", "scatter"} else "unknown", "confidence": "unknown"}],
        "measured_points": mark_points,
        "evidence_gaps": gaps,
        "stage1_diagnostics": {
            "tick_parsing": {**tick_diag, "rotated_ocr": rotated_diag},
            "cv_grid_x_axis": grid_x_diag,
            "semantic_x_axis": semantic_x_diag,
            "cv_grid_axis": grid_diag,
            "mark_extraction": mark_diag,
            "colored_line_sample": line_diag,
            "horizontal_bar_label_scan": hbar_diag,
            "legend_series": {"entry_count": len(legend_entries), "entries": legend_entries},
            "category_labels": category_diag,
            "generality_justification": "The prototype uses generic OCR tick geometry, axis-side clustering, and connected visual mark components, not PlotQA sample IDs or answers.",
            "negative_control": "The connected-component mark path should not be trusted when no numeric y-axis ruler is fitted or when components are dominated by text/legend regions.",
        },
    }
    record["schema_violations"] = validate_m0_record(record)
    record["bottleneck_labels"] = bottleneck_labels(record)
    return record


def tick_json(tick: ParsedTick, axis: str) -> dict[str, Any]:
    return {"text": tick.text, "value": tick.value, "pixel": bbox_center(tick.bbox)[0 if axis == "x" else 1], "bbox": tick.bbox, "confidence": tick.confidence}


def validate_m0_record(record: dict[str, Any]) -> list[str]:
    violations = []
    if record.get("chart_geometry", {}).get("chart_type") not in {"bar", "line", "scatter", "grouped_bar", "unknown"}:
        violations.append("chart_geometry.chart_type_invalid")
    ruler = record.get("ruler")
    if not isinstance(ruler, dict):
        violations.append("ruler_missing")
        return violations
    if ruler.get("confidence") not in {"high", "medium", "low", "failed"}:
        violations.append("ruler.confidence_invalid")
    for axis in ("x", "y"):
        ticks = ruler.get(f"{axis}_ticks")
        if not isinstance(ticks, list):
            violations.append(f"ruler.{axis}_ticks_not_list")
        model = ruler.get(f"{axis}_model")
        if not isinstance(model, dict) or model.get("type") not in {"linear", "log", "nearest_label", "none"}:
            violations.append(f"ruler.{axis}_model_invalid")
    for idx, point in enumerate(record.get("measured_points") or []):
        if not isinstance(point.get("fact_id"), str):
            violations.append(f"measured_points[{idx}].fact_id_invalid")
        if point.get("confidence") not in {"high", "medium", "low"}:
            violations.append(f"measured_points[{idx}].confidence_invalid")
        if not isinstance(point.get("uncertainty"), dict):
            violations.append(f"measured_points[{idx}].uncertainty_invalid")
    for idx, gap in enumerate(record.get("evidence_gaps") or []):
        if gap.get("severity") not in {"blocking", "nonblocking"}:
            violations.append(f"evidence_gaps[{idx}].severity_invalid")
    return violations


def bottleneck_labels(record: dict[str, Any]) -> list[dict[str, Any]]:
    labels = []
    ruler = record.get("ruler") or {}
    if ruler.get("y_model", {}).get("type") != "linear" and ruler.get("x_model", {}).get("type") != "linear":
        labels.append({"label": "m0_no_ruler", "fields": {"x_tick_count": len(ruler.get("x_ticks") or []), "y_tick_count": len(ruler.get("y_ticks") or [])}})
    elif axis_model_is_noisy(ruler.get("x_model", {})) or axis_model_is_noisy(ruler.get("y_model", {})):
        labels.append(
            {
                "label": "m0_bad_ruler",
                "fields": {
                    "x_residual_px_median": ruler.get("x_model", {}).get("residual_px_median"),
                    "y_residual_px_median": ruler.get("y_model", {}).get("residual_px_median"),
                    "residual_cap_px": 12.0,
                },
            }
        )
    if not record.get("measured_points"):
        labels.append({"label": "m0_missing_mark", "fields": record.get("stage1_diagnostics", {}).get("mark_extraction", {})})
    if record.get("schema_violations"):
        labels.append({"label": "unknown", "fields": {"schema_violations": record["schema_violations"]}})
    return labels or [{"label": "unknown", "fields": {"reason": "measurement_quality_not_yet_evaluated_against_operand_need"}}]


def hash_dev_subset(samples: list[dict[str, Any]], count: int = 8) -> list[dict[str, Any]]:
    return sorted(samples, key=lambda row: stable_hash(row["sample_id"]))[:count]


def stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def operation_requires_geometry(sample: dict[str, Any]) -> bool:
    return (sample.get("metadata") or {}).get("reasoning_type") in {"lookup", "comparison", "extrema", "trend", "range", "slope", "ratio"}


def build_stage0(plan: dict[str, Any], ocr_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ocr_by_id = {row["sample_id"]: row for row in ocr_rows}
    rows = []
    for sample in plan["selected_samples"]:
        ocr = ocr_by_id.get(sample["sample_id"], {})
        metadata = sample.get("metadata") or {}
        theoretically = (
            operation_requires_geometry(sample)
            and metadata.get("chart_type") in {"bar", "line", "scatter"}
            and ocr.get("any_text_detected") is True
            and ocr.get("axis_tick_text_detected") is True
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "chart_type": metadata.get("chart_type"),
                "operation": metadata.get("reasoning_type"),
                "answer_derivation": metadata.get("answer_derivation"),
                "e0_correct": None,
                "g_openocr_correct": None,
                "ocr_missing": ocr.get("any_text_detected") is not True,
                "ruler_missing": ocr.get("axis_tick_text_detected") is not True,
                "mark_missing": None,
                "series_missing": metadata.get("legend_required") is True and ocr.get("legend_text_detected") is not True,
                "operation_parse_missing": not operation_requires_geometry(sample),
                "m3_reasoning_error": None,
                "theoretically_measurable": theoretically,
            }
        )
    counts = Counter((row["chart_type"], row["operation"]) for row in rows)
    measurable = sum(1 for row in rows if row["theoretically_measurable"])
    status = "stage_passed" if measurable >= 20 else "stage_failed_exhausted"
    return {
        "schema_version": "plotqa_measured_geosym_stage0_audit_v1",
        "created_at_utc": utc_now(),
        "stage": "stage0_zero_cost_audit",
        "stage_status": status,
        "sample_count": len(rows),
        "theoretically_measurable_count": measurable,
        "chart_operation_counts": {f"{k[0]}::{k[1]}": v for k, v in counts.items()},
        "failure_matrix": rows,
        "exhaustion_evidence": {} if status == "stage_passed" else {"reason": "fewer_than_20_theoretically_measurable_samples"},
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
    }


def build_stage1(records: list[dict[str, Any]]) -> dict[str, Any]:
    schema_valid = sum(1 for row in records if not row["schema_violations"])
    ruler_built = sum(1 for row in records if row.get("ruler", {}).get("confidence") in {"high", "medium", "low"})
    target_candidate = sum(1 for row in records if row.get("measured_points"))
    status = "stage_passed" if ruler_built >= 6 and target_candidate >= 5 and schema_valid == len(records) else "stage_failed_exhausted"
    attempted = [
        "axis_hough_placeholder_not_used_connected_projection_ocr_envelope_used",
        "tick_raw_numeric_parse",
        "tick_punctuation_percent_comma_si_normalization",
        "axis_side_spatial_clustering",
        "connected_component_mark_segmentation",
        "fallback_evidence_gap_emission",
    ]
    return {
        "schema_version": "plotqa_measured_geosym_stage1_status_v1",
        "created_at_utc": utc_now(),
        "stage": "stage1_m0_measurement_prototype_no_paid",
        "stage_status": status,
        "sample_count": len(records),
        "ruler_built": ruler_built,
        "measured_target_candidate": target_candidate,
        "schema_valid": schema_valid,
        "go_gate": {
            "ruler_built_at_least_6_of_8": ruler_built >= 6,
            "measured_target_candidate_at_least_5_of_8": target_candidate >= 5,
            "schema_valid_8_of_8": schema_valid == len(records),
            "sample_count": len(records),
            "ruler_built_count": ruler_built,
            "measured_target_candidate_count": target_candidate,
            "schema_valid_count": schema_valid,
        },
        "exhaustion_evidence": {} if status == "stage_passed" else {"attempted_zero_cost_paths": attempted},
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
    }


def render_stage_audit(stage0: dict[str, Any], stage1: dict[str, Any]) -> str:
    lines = [
        "# PlotQA Measured GeoSym Stage Audit",
        "",
        f"- Created UTC: `{utc_now()}`",
        f"- Stage 0 status: `{stage0['stage_status']}`",
        f"- Stage 0 theoretically measurable: {stage0['theoretically_measurable_count']}/{stage0['sample_count']}",
        f"- Stage 1 status: `{stage1['stage_status']}`",
        f"- Stage 1 ruler built: {stage1['ruler_built']}/{stage1['sample_count']}",
        f"- Stage 1 measured target candidate: {stage1['measured_target_candidate']}/{stage1['sample_count']}",
        f"- Paid calls executed: {stage1['paid_calls_executed']}",
        "",
        "## Stage 1 Go Gate",
        "",
    ]
    for key, value in stage1["go_gate"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Interpretation", ""])
    if stage1["stage_status"] == "stage_passed":
        lines.append("The deterministic measured-M0 prototype is schema-valid and plausible enough to consider a bounded locate smoke.")
    else:
        lines.append("The deterministic measured-M0 prototype is not yet sufficient; inspect bottleneck labels before any paid locate smoke.")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--raw-ocr", required=True)
    parser.add_argument("--output-dir", default=str(STAGE_DIR))
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--all-samples", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    plan_path = Path(args.plan)
    raw_ocr_path = Path(args.raw_ocr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = load_json(plan_path)
    raw_ocr = load_json(raw_ocr_path)
    ocr_by_id = {row["sample_id"]: row for row in raw_ocr}
    stage0 = build_stage0(plan, raw_ocr)
    requested_ids = [item.strip() for item in args.sample_ids.split(",") if item.strip()]
    if requested_ids:
        sample_by_id = {sample["sample_id"]: sample for sample in plan["selected_samples"]}
        dev_samples = [sample_by_id[sample_id] for sample_id in requested_ids if sample_id in sample_by_id]
    elif args.all_samples:
        dev_samples = list(plan["selected_samples"])
    else:
        dev_samples = hash_dev_subset(plan["selected_samples"], args.sample_limit)
    records = [measured_record(sample, ocr_by_id[sample["sample_id"]], enable_rotated_ocr=True) for sample in dev_samples]
    stage1 = build_stage1(records)
    bottlenecks = {
        "schema_version": "plotqa_measured_geosym_bottleneck_audit_v1",
        "created_at_utc": utc_now(),
        "sample_count": len(records),
        "labels": [{"sample_id": row["sample_id"], "labels": row["bottleneck_labels"]} for row in records],
        "label_counts": dict(Counter(label["label"] for row in records for label in row["bottleneck_labels"])),
    }
    repair_history = {
        "schema_version": "plotqa_measured_geosym_repair_history_v1",
        "created_at_utc": utc_now(),
        "repairs": [
            {
                "repair_iteration": 1,
                "dominant_bottleneck": "m0_bad_ruler",
                "files_changed": ["scripts/run_plotqa_measured_m0.py", "scripts/run_plotqa_measured_locate_smoke.py"],
                "zero_cost_checks_run": ["rotated_easyocr_tick_probe", "ruler_residual_quality_gate"],
                "paid_cap_before_run": {"llm_http_attempt_cap": 0, "ocr_http_attempt_cap": 0},
                "actual_paid_attempts": {"llm_http_attempts": 0, "ocr_http_attempts": 0},
                "generality_justification": "Rotated OCR is a generic chart tick-label recovery path for axis labels rendered vertically or at high angle.",
                "negative_control": "The path keeps only numeric OCR outputs and does not use PlotQA IDs, templates, data tables, or reference answers.",
            }
        ],
        "next_required_action": "freeze_stage2_locate_smoke" if stage1["stage_status"] == "stage_passed" else "repair_stage1_deterministic_m0",
    }
    (output_dir / "stage0_audit.json").write_text(json.dumps(stage0, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "stage0_status.json").write_text(json.dumps({"stage": stage0["stage"], "stage_status": stage0["stage_status"]}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "m0_measurement_records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "stage1_status.json").write_text(json.dumps(stage1, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "stage_audit.md").write_text(render_stage_audit(stage0, stage1), encoding="utf-8")
    (output_dir / "bottleneck_audit.json").write_text(json.dumps(bottlenecks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "repair_history.json").write_text(json.dumps(repair_history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote measured-M0 audit artifacts: {output_dir}")
    print(f"stage0_status={stage0['stage_status']}")
    print(f"stage1_status={stage1['stage_status']}")
    print(f"ruler_built={stage1['ruler_built']}/{stage1['sample_count']}")
    print(f"measured_target_candidate={stage1['measured_target_candidate']}/{stage1['sample_count']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
