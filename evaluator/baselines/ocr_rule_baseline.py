"""
OCR + rule-based coordinate extraction baseline.

Uses Mathpix to read tick labels from chart images, reconstructs the
axis structure, then applies basic image processing to detect approximate
curve-grid intersection points.  The extracted coordinates are intentionally
*noisier* and *less complete* than GeoSym's multi-stage pipeline — that is
the point: this baseline exists to demonstrate that specialised extraction
adds measurable value over generic OCR.
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _find_largest_aligned_cluster(
    labels: List[Dict[str, Any]],
    key: str,
    tol: float,
) -> List[Dict[str, Any]]:
    """Find the largest group of labels sharing a similar coordinate value.

    Args:
        labels: candidate label dicts.
        key: ``"px"`` for vertical alignment, ``"py"`` for horizontal.
        tol: maximum deviation from the group centre.

    Returns:
        The largest cluster (≥2 members), or ``[]``.
    """
    if len(labels) < 2:
        return []

    best_cluster: List[Dict[str, Any]] = []
    sorted_labels = sorted(labels, key=lambda lb: lb[key])

    for i, anchor in enumerate(sorted_labels):
        cluster = [lb for lb in sorted_labels if abs(lb[key] - anchor[key]) <= tol]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    return best_cluster if len(best_cluster) >= 2 else []


def _img_to_data_uri(img: np.ndarray) -> str:
    """Encode an OpenCV image as a PNG data URI for the Mathpix API."""
    _, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


class OCRRuleBaseline:
    """Extract approximate integer coordinates from a chart image using
    Mathpix for tick-label reading and OpenCV for curve detection."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_coordinates(
        self,
        image_path: str | Path,
        *,
        curve_color_bgr: Tuple[int, int, int] = (0, 0, 139),
    ) -> List[Tuple[int, int]]:
        """End-to-end: image path -> list of (x, y) integer coordinate pairs.

        Args:
            image_path: path to the chart image.
            curve_color_bgr: expected BGR colour of the function curve
                (default dark-red matches the generator config).

        Returns:
            Sorted list of unique ``(x, y)`` integer tuples.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning("Failed to load image: %s", image_path)
            return []

        h, w = img.shape[:2]

        x_ticks, y_ticks = self._ocr_tick_labels(img)
        if len(x_ticks) < 2 or len(y_ticks) < 2:
            logger.warning("OCR baseline: insufficient tick labels (x=%d, y=%d)", len(x_ticks), len(y_ticks))
            return []

        axis_info = self._build_axis_mapping(x_ticks, y_ticks, w, h)
        if axis_info is None:
            return []

        curve_mask = self._detect_curve(img, curve_color_bgr)
        coords = self._match_curve_to_grid(curve_mask, axis_info)
        return coords

    def format_for_prompt(self, coords: List[Tuple[int, int]]) -> str:
        """Format OCR-extracted coordinates for prompt injection."""
        if not coords:
            return "(no points detected by OCR)"
        pts = ", ".join(f"({x}, {y})" for x, y in coords)
        return pts

    # ------------------------------------------------------------------
    # Step 1: OCR tick labels
    # ------------------------------------------------------------------

    def _ocr_tick_labels(
        self, img: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run Mathpix on the image and classify detected numbers as
        x-axis or y-axis tick labels using alignment-based clustering.

        Strategy:
          1. Run OCR on the full image, collect integer-valued detections.
          2. Classify via alignment clustering (primary) with region
             heuristics (secondary).
          3. If y-axis labels are still missing, run a focused OCR pass
             on the left half of the image near the estimated y-axis
             position.
        """
        h, w = img.shape[:2]

        all_labels = self._run_ocr(img)
        if not all_labels:
            return [], []

        logger.debug("OCR baseline: %d numeric labels detected on full image", len(all_labels))
        for lb in all_labels:
            logger.debug("  label=%d  px=%d py=%d", lb["value"], lb["px"], lb["py"])

        x_ticks, y_ticks = self._classify_axis_labels(all_labels, w, h)

        if len(x_ticks) >= 2 and len(y_ticks) < 2:
            y_ticks = self._focused_y_axis_ocr(img, x_ticks)
        elif len(y_ticks) >= 2 and len(x_ticks) < 2:
            x_ticks = self._focused_x_axis_ocr(img, y_ticks)

        return x_ticks, y_ticks

    def _run_ocr(self, img: np.ndarray, *, upscale: bool = True) -> List[Dict[str, Any]]:
        """Run Mathpix OCR on an image region and return integer-valued labels.

        When *upscale* is True the image is resized 2x before OCR,
        which improves detection of small chart labels.
        Returned pixel coordinates are scaled back to the original size.
        """
        try:
            from recognizer.mathpix.mathpix_image_ocr import MathpixImageOCRTool
        except ImportError:
            logger.warning("Mathpix OCR tool not available — OCR baseline disabled")
            return []

        factor = 2 if upscale else 1
        ocr_img = img
        if factor > 1:
            ocr_img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

        data_uri = _img_to_data_uri(ocr_img)

        try:
            tool = MathpixImageOCRTool()
            result = tool.ocr_image(data_uri)
        except Exception as exc:
            logger.warning("Mathpix OCR failed: %s", exc)
            return []

        if not isinstance(result, dict):
            logger.warning("Mathpix OCR returned unexpected type: %s", type(result))
            return []

        num_pattern = re.compile(r"^-?\d+\.?\d*$")
        labels: List[Dict[str, Any]] = []

        # Parse line_data for bounding boxes and text
        line_data = result.get("line_data", [])
        for line in line_data:
            text = (line.get("text") or "").strip()
            if not text:
                continue
            if not num_pattern.match(text):
                continue
            try:
                value = float(text)
                if value != int(value):
                    continue
                value = int(value)
            except (ValueError, OverflowError):
                continue

            # Extract bounding box: Mathpix uses {left, top, width, height}
            cnt = line.get("cnt")
            if cnt and isinstance(cnt, list) and len(cnt) >= 2:
                # cnt is a list of [x, y] pixel points (polygon)
                xs = [pt[0] for pt in cnt]
                ys = [pt[1] for pt in cnt]
                cx = int(sum(xs) / len(xs)) // factor
                cy = int(sum(ys) / len(ys)) // factor
                bw = (max(xs) - min(xs)) // factor
                bh = (max(ys) - min(ys)) // factor
            else:
                # Fallback to top-level bounding box if available
                bbox = line.get("bounding_box") or {}
                if not bbox:
                    continue
                x0 = bbox.get("x", bbox.get("left", 0))
                y0 = bbox.get("y", bbox.get("top", 0))
                bw_ = bbox.get("width", 0)
                bh_ = bbox.get("height", 0)
                cx = int(x0 + bw_ / 2) // factor
                cy = int(y0 + bh_ / 2) // factor
                bw = bw_ // factor
                bh = bh_ // factor

            labels.append({"value": value, "px": cx, "py": cy, "bw": bw, "bh": bh})

        # If line_data is empty or produced nothing, try parsing from top-level text
        if not labels:
            top_text = result.get("text", "")
            logger.debug("Mathpix line_data empty; top-level text: %r", top_text[:200])

        return labels

    def _focused_y_axis_ocr(
        self,
        img: np.ndarray,
        x_ticks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """When full-image OCR fails to find y-axis labels, estimate the
        y-axis pixel position from the x-axis ticks and run a focused OCR
        on a narrow vertical strip around it."""
        h, w = img.shape[:2]

        x_ticks_sorted = sorted(x_ticks, key=lambda t: t["value"])
        vals = [t["value"] for t in x_ticks_sorted]
        pxs = [t["px"] for t in x_ticks_sorted]

        if vals[-1] == vals[0]:
            return []

        scale = (pxs[-1] - pxs[0]) / (vals[-1] - vals[0])
        y_axis_px = int(pxs[0] + (0 - vals[0]) * scale)

        strip_half = max(int(w * 0.15), 80)
        x_lo = max(0, y_axis_px - strip_half)
        x_hi = min(w, y_axis_px + strip_half // 3)

        if x_hi <= x_lo:
            return []

        strip = img[:, x_lo:x_hi].copy()
        logger.debug("OCR baseline: focused y-axis OCR on strip x=[%d,%d] (y-axis est. px=%d)", x_lo, x_hi, y_axis_px)

        labels = self._run_ocr(strip)
        for lb in labels:
            lb["px"] += x_lo

        if not labels:
            logger.debug("OCR baseline: focused y-axis OCR found 0 labels")
            return []

        logger.debug("OCR baseline: focused y-axis OCR found %d labels", len(labels))
        y_ticks = [lb for lb in labels if abs(lb["px"] - y_axis_px) < strip_half]
        y_ticks.sort(key=lambda t: t["py"])
        return y_ticks

    def _focused_x_axis_ocr(
        self,
        img: np.ndarray,
        y_ticks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """When full-image OCR fails to find x-axis labels, estimate the
        x-axis pixel position from the y-axis ticks and run a focused OCR
        on a horizontal strip below it."""
        h, w = img.shape[:2]

        y_ticks_sorted = sorted(y_ticks, key=lambda t: t["value"])
        vals = [t["value"] for t in y_ticks_sorted]
        pys = [t["py"] for t in y_ticks_sorted]

        if vals[-1] == vals[0]:
            return []

        # y-axis: higher math values -> lower pixel y (image coords inverted)
        scale = (pys[-1] - pys[0]) / (vals[-1] - vals[0])
        x_axis_py = int(pys[0] + (0 - vals[0]) * scale)

        strip_height = max(int(h * 0.15), 80)
        y_lo = max(0, x_axis_py)
        y_hi = min(h, x_axis_py + strip_height)

        if y_hi <= y_lo:
            return []

        strip = img[y_lo:y_hi, :].copy()
        logger.debug(
            "OCR baseline: focused x-axis OCR on strip y=[%d,%d] (x-axis est. py=%d)",
            y_lo, y_hi, x_axis_py,
        )

        labels = self._run_ocr(strip)
        for lb in labels:
            lb["py"] += y_lo

        if not labels:
            logger.debug("OCR baseline: focused x-axis OCR found 0 labels")
            return []

        logger.debug("OCR baseline: focused x-axis OCR found %d labels", len(labels))
        x_ticks = [lb for lb in labels if abs(lb["py"] - x_axis_py) < strip_height]
        x_ticks.sort(key=lambda t: t["px"])
        return x_ticks

    @staticmethod
    def _clean_tick_values(ticks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove OCR misreads by keeping only values that fit into
        an arithmetic progression (the expected pattern for axis ticks)."""
        if len(ticks) < 3:
            return ticks

        values = [t["value"] for t in ticks]
        unique_vals = sorted(set(values))
        if len(unique_vals) < 2:
            return ticks

        # Try common tick spacings: 1, 2, 5
        best_inliers: List[Dict[str, Any]] = []
        for step in [1, 2, 5]:
            for base in unique_vals:
                grid = set()
                low = base - step * 20
                for i in range(41):
                    grid.add(low + step * i)
                inliers = [t for t in ticks if t["value"] in grid]
                # Remove duplicate values (keep the one closest to expected position)
                seen_vals: Dict[int, Dict[str, Any]] = {}
                for t in inliers:
                    v = t["value"]
                    if v not in seen_vals:
                        seen_vals[v] = t
                inliers = sorted(seen_vals.values(), key=lambda t: t["value"])
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers

        return best_inliers if len(best_inliers) >= 2 else ticks

    @staticmethod
    def _classify_axis_labels(
        labels: List[Dict[str, Any]], w: int, h: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Classify OCR labels into x-axis and y-axis groups using
        alignment-based clustering.

        X-axis labels share a similar y-coordinate (horizontal row).
        Y-axis labels share a similar x-coordinate (vertical column).
        """
        if len(labels) < 2:
            return [], []

        tol_y = h * 0.04
        tol_x = w * 0.04

        # --- Find x-axis labels: largest y-aligned cluster in bottom 60% ---
        bottom_labels = [lb for lb in labels if lb["py"] > h * 0.40]
        x_ticks = _find_largest_aligned_cluster(bottom_labels, key="py", tol=tol_y)

        # Validate: x-axis labels should span a reasonable horizontal range
        if len(x_ticks) >= 2:
            x_span = max(t["px"] for t in x_ticks) - min(t["px"] for t in x_ticks)
            if x_span < w * 0.15:
                x_ticks = []

        x_tick_set = {id(t) for t in x_ticks}

        # --- Find y-axis labels: largest x-aligned cluster from remaining ---
        remaining = [lb for lb in labels if id(lb) not in x_tick_set]
        y_ticks = _find_largest_aligned_cluster(remaining, key="px", tol=tol_x)

        # Validate: y-axis labels should span a reasonable vertical range
        if len(y_ticks) >= 2:
            y_span = max(t["py"] for t in y_ticks) - min(t["py"] for t in y_ticks)
            if y_span < h * 0.15:
                y_ticks = []

        x_ticks = OCRRuleBaseline._clean_tick_values(x_ticks)
        y_ticks = OCRRuleBaseline._clean_tick_values(y_ticks)

        x_ticks.sort(key=lambda t: t["px"])
        y_ticks.sort(key=lambda t: t["py"])
        return x_ticks, y_ticks

    # ------------------------------------------------------------------
    # Step 2: Build pixel <-> math coordinate mapping
    # ------------------------------------------------------------------

    def _build_axis_mapping(
        self,
        x_ticks: List[Dict[str, Any]],
        y_ticks: List[Dict[str, Any]],
        img_w: int,
        img_h: int,
    ) -> Optional[Dict[str, Any]]:
        """From tick labels, derive a linear pixel-to-math mapping for both axes."""
        if len(x_ticks) < 2 or len(y_ticks) < 2:
            return None

        x_vals = [t["value"] for t in x_ticks]
        x_pxs = [t["px"] for t in x_ticks]
        y_vals = [t["value"] for t in y_ticks]
        y_pxs = [t["py"] for t in y_ticks]

        x_scale = (x_vals[-1] - x_vals[0]) / max(x_pxs[-1] - x_pxs[0], 1)
        x_offset = x_vals[0] - x_scale * x_pxs[0]

        y_scale = (y_vals[-1] - y_vals[0]) / max(y_pxs[-1] - y_pxs[0], 1)
        y_offset = y_vals[0] - y_scale * y_pxs[0]

        x_min = min(x_vals)
        x_max = max(x_vals)
        y_min = min(y_vals)
        y_max = max(y_vals)

        grid_xs_px = {}
        for xv in range(x_min, x_max + 1):
            px = int(round((xv - x_offset) / x_scale)) if x_scale != 0 else 0
            if 0 <= px < img_w:
                grid_xs_px[xv] = px

        grid_ys_px = {}
        for yv in range(y_min, y_max + 1):
            py = int(round((yv - y_offset) / y_scale)) if y_scale != 0 else 0
            if 0 <= py < img_h:
                grid_ys_px[yv] = py

        return {
            "x_scale": x_scale, "x_offset": x_offset,
            "y_scale": y_scale, "y_offset": y_offset,
            "grid_xs_px": grid_xs_px,
            "grid_ys_px": grid_ys_px,
            "x_range": (x_min, x_max),
            "y_range": (y_min, y_max),
        }

    # ------------------------------------------------------------------
    # Step 3: Detect curve pixels
    # ------------------------------------------------------------------

    def _detect_curve(
        self,
        img: np.ndarray,
        target_bgr: Tuple[int, int, int],
        tolerance: int = 80,
    ) -> np.ndarray:
        """Create a binary mask of pixels likely belonging to the function curve."""
        target = np.array(target_bgr, dtype=np.int16)
        diff = np.abs(img.astype(np.int16) - target).sum(axis=2)
        mask = (diff < tolerance).astype(np.uint8) * 255

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat_mask = (hsv[:, :, 1] > 30).astype(np.uint8) * 255
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dark_mask = (gray < 160).astype(np.uint8) * 255

        combined = cv2.bitwise_or(mask, cv2.bitwise_and(sat_mask, dark_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        return combined

    # ------------------------------------------------------------------
    # Step 4: Match curve pixels to grid intersections
    # ------------------------------------------------------------------

    def _match_curve_to_grid(
        self,
        curve_mask: np.ndarray,
        axis_info: Dict[str, Any],
        search_radius: int = 8,
    ) -> List[Tuple[int, int]]:
        """For each integer x value, find the closest curve pixel and map it
        back to a math y coordinate."""
        grid_xs = axis_info["grid_xs_px"]
        y_scale = axis_info["y_scale"]
        y_offset = axis_info["y_offset"]
        y_min, y_max = axis_info["y_range"]

        coords: List[Tuple[int, int]] = []
        seen = set()

        for x_val, col_px in grid_xs.items():
            lo = max(0, col_px - search_radius)
            hi = min(curve_mask.shape[1], col_px + search_radius + 1)

            col_strip = curve_mask[:, lo:hi]
            if col_strip.size == 0:
                continue

            row_hits = np.where(col_strip.max(axis=1) > 0)[0]
            if len(row_hits) == 0:
                continue

            mid_row = int(np.median(row_hits))
            raw_y = y_scale * mid_row + y_offset
            y_val = int(round(raw_y))

            if y_min <= y_val <= y_max and (x_val, y_val) not in seen:
                seen.add((x_val, y_val))
                coords.append((x_val, y_val))

        coords.sort(key=lambda p: (p[0], p[1]))
        return coords
