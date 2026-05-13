"""
Draw coordinate axes, grid lines, and tick labels inside a chart region.
Integer ticks only. Origin (0,0) is labeled once on the x-axis only to avoid overlap.
Math coords (x_range, y_range) map to pixel coords; Y axis math-up, image-down.
Axes, grid, and labels use black/dark gray.
Chart border can be drawn for better OCR OCR recognition.
"""

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_chart_border(
    img: np.ndarray,
    chart_rect: Tuple[int, int, int, int],
    config: dict,
) -> None:
    """
    Draw a border around the chart area for better OCR recognition.
    """
    cx, cy, cw, ch = chart_rect
    border_cfg = config.get("chart", {}).get("border", {})
    if not border_cfg.get("enabled", True):
        return
    
    color = tuple(int(x) for x in border_cfg.get("color", [0, 0, 0]))
    thickness = border_cfg.get("thickness", 2)
    margin = border_cfg.get("margin", 5)
    
    # Draw rectangle around chart area
    pt1 = (cx - margin, cy - margin)
    pt2 = (cx + cw + margin, cy + ch + margin)
    cv2.rectangle(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)


def _to_pixel(
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    cx: int,
    cy: int,
    cw: int,
    ch: int,
    padding: int,
) -> Tuple[int, int]:
    ix = cx + padding
    iy = cy + padding
    iw = cw - 2 * padding
    ih = ch - 2 * padding
    if x_max == x_min or y_max == y_min:
        return (ix + iw // 2, iy + ih // 2)
    px = ix + (x - x_min) / (x_max - x_min) * iw
    py = iy + (y_max - y) / (y_max - y_min) * ih
    return (int(round(px)), int(round(py)))


def _tick_label_text(value: float) -> str:
    """Integer ticks only."""
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _find_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/segoeui.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_text_pil(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_size: int,
    color_bgr: Tuple[int, int, int],
) -> None:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _find_font(font_size)
    text_pos = (pos[0], max(0, pos[1] - font_size))
    text_color = tuple(reversed(color_bgr))
    try:
        draw.text(
            text_pos,
            text,
            fill=text_color,
            font=font,
            stroke_width=1,
            stroke_fill=text_color,
        )
    except TypeError:
        draw.text(text_pos, text, fill=text_color, font=font)
    arr = np.array(pil)
    img[:] = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _draw_dashed_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_on: int = 10,
    dash_off: int = 6,
) -> None:
    """Draw an anti-aliased dashed line from pt1 to pt2."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy = x2 - x1, y2 - y1
    length = int(np.hypot(dx, dy))
    if length == 0:
        return

    cycle = dash_on + dash_off
    pos = 0
    while pos < length:
        seg_end = min(pos + dash_on, length)
        t0 = pos / length
        t1 = seg_end / length
        sx = int(round(x1 + dx * t0))
        sy = int(round(y1 + dy * t0))
        ex = int(round(x1 + dx * t1))
        ey = int(round(y1 + dy * t1))
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, lineType=cv2.LINE_AA)
        pos += cycle


def draw_axes_and_grid(
    img: np.ndarray,
    chart_rect: Tuple[int, int, int, int],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    x_ticks: List[float],
    y_ticks: List[float],
    config: dict,
) -> Tuple[List[dict], List[dict]]:
    """
    Draw axes and grid. Origin (0) is labeled only once: on the x-axis, not on the y-axis.
    Returns x_tick_labels, y_tick_labels for ground truth.
    """
    cx, cy, cw, ch = chart_rect
    x_min, x_max = x_range
    y_min, y_max = y_range
    padding = config.get("chart", {}).get("padding_px", 50)
    axes_cfg = config.get("axes", {})
    grid_cfg = config.get("grid", {})
    ticks_cfg = config.get("ticks", {})
    font_size = ticks_cfg.get("font_size", 24)
    ax_color = tuple(int(x) for x in axes_cfg.get("color", [0, 0, 0]))
    grid_color = tuple(int(x) for x in grid_cfg.get("color", [0, 0, 0]))
    label_color = tuple(int(x) for x in ticks_cfg.get("label_color", [0, 0, 0]))
    ax_lw = axes_cfg.get("line_width", 2)
    grid_lw = grid_cfg.get("line_width", 1)
    grid_dash_on = grid_cfg.get("dash_on", 10)
    grid_dash_off = grid_cfg.get("dash_off", 6)

    def to_px(x: float, y: float) -> Tuple[int, int]:
        return _to_pixel(x, y, x_min, x_max, y_min, y_max, cx, cy, cw, ch, padding)

    has_origin_x = 0 in x_ticks
    has_origin_y = 0 in y_ticks

    # Grid lines (dashed)
    for xv in x_ticks:
        if x_min <= xv <= x_max:
            p0, p1 = to_px(xv, y_min), to_px(xv, y_max)
            _draw_dashed_line(img, p0, p1, grid_color, grid_lw, grid_dash_on, grid_dash_off)
    for yv in y_ticks:
        if y_min <= yv <= y_max:
            p0, p1 = to_px(x_min, yv), to_px(x_max, yv)
            _draw_dashed_line(img, p0, p1, grid_color, grid_lw, grid_dash_on, grid_dash_off)

    # Axes (thicker)
    px0, py0 = to_px(x_min, 0)
    px1, py1 = to_px(x_max, 0)
    cv2.line(img, (px0, py0), (px1, py1), ax_color, ax_lw, lineType=cv2.LINE_AA)
    px0, py0 = to_px(0, y_min)
    px1, py1 = to_px(0, y_max)
    cv2.line(img, (px0, py0), (px1, py1), ax_color, ax_lw, lineType=cv2.LINE_AA)

    x_tick_labels = []
    y_tick_labels = []
    label_h = int(font_size * 1.4)
    label_w_per_char = int(font_size * 0.6)

    origin_nudge = int(label_w_per_char * 0.8)
    for xv in x_ticks:
        if x_min <= xv <= x_max:
            px, py = to_px(xv, 0)
            txt = _tick_label_text(xv)
            tw = len(txt) * label_w_per_char
            lx = px - tw // 2
            # Nudge the "0" label to the right so it doesn't hide behind the y-axis
            if abs(xv) < 1e-9:
                lx = px + origin_nudge
            ly = py + font_size + 5
            _draw_text_pil(img, txt, (lx, ly), font_size, label_color)
            x_tick_labels.append({
                "value": float(xv),
                "text": txt,
                "center_x": px,
                "center_y": py,
                "left": lx,
                "top": ly,
                "right": lx + tw,
                "bottom": ly + label_h,
            })

    for yv in y_ticks:
        if y_min <= yv <= y_max:
            # Label origin only once: skip y-axis "0" when x-axis already has "0"
            if abs(yv) < 1e-9 and has_origin_x:
                continue
            px, py = to_px(0, yv)
            txt = _tick_label_text(yv)
            tw = len(txt) * label_w_per_char
            # Compensate for _draw_text_pil's font_size offset to center label at py
            lx, ly = px - tw - 8, py + font_size // 2
            _draw_text_pil(img, txt, (lx, ly), font_size, label_color)
            y_tick_labels.append({
                "value": float(yv),
                "text": txt,
                "center_x": px,
                "center_y": py,
                "left": lx,
                "top": ly,
                "right": lx + tw,
                "bottom": ly + label_h,
            })

    return x_tick_labels, y_tick_labels
