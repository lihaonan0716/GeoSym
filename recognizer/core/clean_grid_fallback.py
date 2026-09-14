"""Fallback coordinate extraction for clean Cartesian grid charts.

This path is deliberately narrow: it uses only visible image structure from a
regular Cartesian grid with black axes and a saturated colored curve. It does
not read benchmark metadata or ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CleanGridExtraction:
    success: bool
    coords: list[tuple[int, int]]
    result: dict[str, Any]


def _clusters(counts: np.ndarray, *, min_count: int, max_gap: int = 4) -> list[float]:
    indexes = np.where(counts >= min_count)[0]
    if indexes.size == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = int(indexes[0])
    for value in indexes[1:]:
        value = int(value)
        if value - prev <= max_gap:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return [(lo + hi) / 2.0 for lo, hi in groups]


def _regularize_lattice(values: list[float]) -> list[float]:
    values = sorted(values)
    if len(values) < 3:
        return values
    gaps = np.diff(values)
    typical = float(np.median(gaps))
    if typical <= 0:
        return values

    best: list[float] = []
    tolerance = max(4.0, typical * 0.22)
    for origin in values:
        candidate: list[float] = []
        for value in values:
            step = round((value - origin) / typical)
            aligned = origin + step * typical
            if abs(value - aligned) <= tolerance:
                candidate.append(value)
        if len(candidate) > len(best):
            best = candidate
    return best or values


def _nearest_axis(candidates: list[float], lattice: list[float]) -> float | None:
    if not candidates or not lattice:
        return None
    center = float(np.median(lattice))
    return min(candidates, key=lambda value: abs(value - center))


def extract_clean_cartesian_grid(image: np.ndarray) -> CleanGridExtraction:
    if image is None or image.size == 0:
        return CleanGridExtraction(False, [], {"success": False, "error": "empty_image"})

    height, width = image.shape[:2]
    blue, green, red = cv2.split(image)

    grayish = (
        (np.abs(blue.astype(int) - green.astype(int)) < 8)
        & (np.abs(green.astype(int) - red.astype(int)) < 8)
        & (blue > 110)
        & (blue < 210)
    )
    grayish[: int(height * 0.25), :] = False
    grid_x = _regularize_lattice(_clusters(grayish.sum(axis=0), min_count=max(80, height // 14), max_gap=5))
    grid_y = _regularize_lattice(_clusters(grayish.sum(axis=1), min_count=max(80, width // 14), max_gap=5))

    dark = (blue < 40) & (green < 40) & (red < 40)
    dark[: int(height * 0.25), :] = False
    axis_x = _nearest_axis(_clusters(dark.sum(axis=0), min_count=max(180, height // 5), max_gap=4), grid_x)
    axis_y = _nearest_axis(_clusters(dark.sum(axis=1), min_count=max(180, width // 5), max_gap=4), grid_y)

    if axis_x is None or axis_y is None or len(grid_x) < 3 or len(grid_y) < 3:
        return CleanGridExtraction(
            False,
            [],
            {
                "success": False,
                "error": "clean_grid_structure_not_detected",
                "diagnostics": {
                    "grid_x_count": len(grid_x),
                    "grid_y_count": len(grid_y),
                    "axis_x": axis_x,
                    "axis_y": axis_y,
                },
            },
        )

    grid_x = sorted(set(grid_x + [axis_x]))
    grid_y = sorted(set(grid_y + [axis_y]))
    x_spacing = float(np.median(np.diff(grid_x)))
    y_spacing = float(np.median(np.diff(grid_y)))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    curve = (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 40)
    x0 = max(0, int(min(grid_x) - x_spacing * 0.25))
    x1 = min(width, int(max(grid_x) + x_spacing * 0.25) + 1)
    y0 = max(0, int(min(grid_y) - y_spacing * 0.25))
    y1 = min(height, int(max(grid_y) + y_spacing * 0.25) + 1)
    roi = np.zeros(curve.shape, dtype=bool)
    roi[y0:y1, x0:x1] = True
    curve &= roi

    points: list[dict[str, Any]] = []
    for pixel_x in grid_x:
        xlo = max(0, int(round(pixel_x - x_spacing * 0.18)))
        xhi = min(width, int(round(pixel_x + x_spacing * 0.18)) + 1)
        rows = np.where(curve[:, xlo:xhi].any(axis=1))[0]
        if rows.size == 0:
            continue
        runs: list[tuple[int, int]] = []
        start = prev = int(rows[0])
        for row in rows[1:]:
            row = int(row)
            if row - prev <= 3:
                prev = row
                continue
            runs.append((start, prev))
            start = prev = row
        runs.append((start, prev))

        real_x_float = (pixel_x - axis_x) / x_spacing
        real_x = int(round(real_x_float))
        if abs(real_x_float - real_x) > 0.25:
            continue
        for top, bottom in runs:
            pixel_y = (top + bottom) / 2.0
            real_y_float = (axis_y - pixel_y) / y_spacing
            real_y = int(round(real_y_float))
            if abs(real_y_float - real_y) > 0.25:
                continue
            points.append(
                {
                    "pixel_x": float(pixel_x),
                    "pixel_y": float(pixel_y),
                    "real_x": real_x,
                    "real_y": real_y,
                    "real_x_rounded": real_x,
                    "real_y_rounded": real_y,
                    "type": "clean_grid_curve",
                    "line_types": ["v_grid", "h_grid", "curve"],
                }
            )

    unique = {(int(point["real_x_rounded"]), int(point["real_y_rounded"])): point for point in points}
    coords = sorted(unique)
    intersections = [unique[coord] for coord in coords]
    result = {
        "success": bool(coords),
        "result": {
            "coordinate_system": {"x_axis_position": axis_y, "y_axis_position": axis_x},
            "grid_mapping": {
                "vertical_lines": [{"pixel": x, "value": int(round((x - axis_x) / x_spacing))} for x in grid_x],
                "horizontal_lines": [{"pixel": y, "value": int(round((axis_y - y) / y_spacing))} for y in grid_y],
                "v_grid_x_pixels": grid_x,
                "v_grid_x_values": [int(round((x - axis_x) / x_spacing)) for x in grid_x],
                "h_grid_y_pixels": grid_y,
                "h_grid_y_values": [int(round((axis_y - y) / y_spacing)) for y in grid_y],
            },
            "intersections": intersections,
            "reasoning_intersections": intersections,
            "intersections_raw": intersections,
            "curve_info": {"has_curve": bool(coords), "method": "saturated_curve_pixels"},
            "visual_facts": None,
            "m0_fallback": "clean_cartesian_grid_fallback_v1",
        },
        "diagnostics": {
            "grid_x_count": len(grid_x),
            "grid_y_count": len(grid_y),
            "axis_x": axis_x,
            "axis_y": axis_y,
            "x_spacing": x_spacing,
            "y_spacing": y_spacing,
            "coords_count": len(coords),
        },
    }
    if not coords:
        result["error"] = "clean_grid_curve_intersections_not_detected"
    return CleanGridExtraction(bool(coords), coords, result)
