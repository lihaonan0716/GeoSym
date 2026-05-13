"""
Visualization utilities for annotating images with detected intersections.
"""

from typing import List, Set, Tuple

import cv2
import numpy as np


def annotate_image(
    image: np.ndarray,
    gt_coords: Set[Tuple[int, int]] | List[Tuple[int, int]],
    detected_coords: Set[Tuple[int, int]] | List[Tuple[int, int]] | None = None,
    chart_bbox: List[int] | None = None,
    x_range: Tuple[float, float] = (-10, 10),
    y_range: Tuple[float, float] = (-10, 10),
    padding: int = 50,
    point_radius: int = 8,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Annotate image with ground truth and detected intersection points.

    Colors:
    - Green: Correct (in both GT and detected)
    - Red: Missed (in GT but not detected)
    - Yellow: Extra (detected but not in GT)
    - Blue circle: GT points (when detected_coords is None, just show GT)

    Args:
        image: BGR image
        gt_coords: Ground truth coordinates (math coords, integers)
        detected_coords: Detected coordinates (math coords, integers), or None
        chart_bbox: [x, y, w, h] of chart region in pixels
        x_range: Math coordinate range for x-axis
        y_range: Math coordinate range for y-axis
        padding: Padding inside chart
        point_radius: Radius of annotation circles
        line_thickness: Thickness of annotation lines

    Returns:
        Annotated image (copy)
    """
    img = image.copy()
    gt_set = set(gt_coords) if not isinstance(gt_coords, set) else gt_coords

    if chart_bbox is None:
        # Assume full image is chart
        h, w = img.shape[:2]
        chart_bbox = [0, 0, w, h]

    cx, cy, cw, ch = chart_bbox
    x_min, x_max = x_range
    y_min, y_max = y_range

    def math_to_pixel(mx: int, my: int) -> Tuple[int, int]:
        """Convert math coords to pixel coords."""
        ix = cx + padding
        iy = cy + padding
        iw = cw - 2 * padding
        ih = ch - 2 * padding
        if x_max == x_min or y_max == y_min:
            return (ix + iw // 2, iy + ih // 2)
        px = ix + (mx - x_min) / (x_max - x_min) * iw
        py = iy + (y_max - my) / (y_max - y_min) * ih
        return (int(round(px)), int(round(py)))

    if detected_coords is None:
        # Just show GT points in blue
        for coord in gt_set:
            px, py = math_to_pixel(coord[0], coord[1])
            cv2.circle(img, (px, py), point_radius, (255, 128, 0), -1)  # Blue fill
            cv2.circle(img, (px, py), point_radius, (255, 255, 255), line_thickness)  # White border
            # Label
            label = f"({coord[0]},{coord[1]})"
            cv2.putText(img, label, (px + 10, py - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 128, 0), 1, cv2.LINE_AA)
    else:
        det_set = set(detected_coords) if not isinstance(detected_coords, set) else detected_coords
        correct = gt_set & det_set
        missed = gt_set - det_set
        extra = det_set - gt_set

        # Draw correct points (green)
        for coord in correct:
            px, py = math_to_pixel(coord[0], coord[1])
            cv2.circle(img, (px, py), point_radius, (0, 200, 0), -1)  # Green fill
            cv2.circle(img, (px, py), point_radius, (255, 255, 255), line_thickness)
            label = f"({coord[0]},{coord[1]})"
            cv2.putText(img, label, (px + 10, py - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 200, 0), 1, cv2.LINE_AA)

        # Draw missed points (red)
        for coord in missed:
            px, py = math_to_pixel(coord[0], coord[1])
            cv2.circle(img, (px, py), point_radius, (0, 0, 220), -1)  # Red fill
            cv2.circle(img, (px, py), point_radius, (255, 255, 255), line_thickness)
            label = f"({coord[0]},{coord[1]})"
            cv2.putText(img, label, (px + 10, py - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 220), 1, cv2.LINE_AA)

        # Draw extra points (yellow)
        for coord in extra:
            px, py = math_to_pixel(coord[0], coord[1])
            cv2.circle(img, (px, py), point_radius, (0, 220, 255), -1)  # Yellow fill
            cv2.circle(img, (px, py), point_radius, (255, 255, 255), line_thickness)
            label = f"({coord[0]},{coord[1]})"
            cv2.putText(img, label, (px + 10, py - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 220, 255), 1, cv2.LINE_AA)

    return img


def create_legend(width: int = 300, height: int = 120) -> np.ndarray:
    """Create a legend image explaining the colors."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 40  # Dark background

    items = [
        ((0, 200, 0), "Correct (GT & Detected)"),
        ((0, 0, 220), "Missed (GT only)"),
        ((0, 220, 255), "Extra (Detected only)"),
    ]

    y = 25
    for color, text in items:
        cv2.circle(img, (20, y), 8, color, -1)
        cv2.circle(img, (20, y), 8, (255, 255, 255), 1)
        cv2.putText(img, text, (40, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)
        y += 35

    return img
