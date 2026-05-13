"""
Compute page layout and chart region.
Layout: top_bottom (stem top, chart bottom) or left_right (stem left, chart right).
Stem area is sized to fit the actual text content; chart fills the rest.
"""

import random
from typing import List, Literal, Tuple

LayoutKind = Literal["top_bottom", "left_right"]

DEFAULT_STEM_HEIGHT = 100
DEFAULT_STEM_WIDTH = 200


def choose_layout(config: dict) -> LayoutKind:
    opts = config.get("page", {}).get("layout_options", ["top_bottom", "left_right"])
    return random.choice(opts)


def estimate_stem_height(font_size: int, num_lines: int, padding: int = 20) -> int:
    """Estimate the pixel height needed for stem text."""
    line_height = int(font_size * 1.6)
    return padding + num_lines * line_height + padding


def estimate_stem_width(font_size: int, max_chars: int, padding: int = 20) -> int:
    """Estimate the pixel width needed for stem text (rough)."""
    char_width = int(font_size * 0.6)
    return padding + max_chars * char_width + padding


def chart_bbox(
    page_w: int,
    page_h: int,
    layout: LayoutKind,
    area_ratio_min: float,
    area_ratio_max: float,
    aspect_min: float,
    aspect_max: float,
    margin_px: int,
    stem_text_lines: int = 2,
    stem_font_size: int = 28,
) -> Tuple[int, int, int, int]:
    """
    Return (x, y, w, h) of chart in page pixels.

    The chart size is determined by *area_ratio* (fraction of available area)
    and *aspect_ratio* (width / height).  The chart is centred within the
    remaining space after the stem area is allocated.
    """
    assert 0 < area_ratio_min <= area_ratio_max <= 1
    assert aspect_min <= aspect_max

    area_ratio = random.uniform(area_ratio_min, area_ratio_max)
    aspect = random.uniform(aspect_min, aspect_max)

    m = margin_px
    avail_w = page_w - 2 * m
    avail_h = page_h - 2 * m

    if layout == "top_bottom":
        stem_h = estimate_stem_height(stem_font_size, stem_text_lines)
        stem_h = max(60, min(stem_h, avail_h // 3))

        chart_y_start = m + stem_h
        remaining_h = avail_h - stem_h

        chart_h = int(area_ratio * remaining_h)
        chart_w = int(chart_h * aspect)
        chart_w = min(chart_w, avail_w)
        # Ensure chart fills at least 70% of available width for OCR reliability.
        # When the natural aspect produces a narrow chart, widen it and cap height.
        min_w = int(avail_w * 0.70)
        if chart_w < min_w:
            chart_w = min_w
        chart_h = min(int(chart_w / aspect), remaining_h)

        chart_x = m + (avail_w - chart_w) // 2
        chart_y = chart_y_start + (remaining_h - chart_h) // 2
        return (chart_x, chart_y, chart_w, chart_h)
    else:
        stem_w = estimate_stem_width(stem_font_size, 30)
        stem_w = max(120, min(stem_w, avail_w // 3))

        chart_x_start = m + stem_w
        remaining_w = avail_w - stem_w

        chart_w = int(area_ratio * remaining_w)
        chart_h = int(chart_w / aspect)
        chart_h = min(chart_h, avail_h)
        chart_w = min(chart_w, remaining_w)
        min_w = int(remaining_w * 0.70)
        if chart_w < min_w:
            chart_w = min_w
        chart_h = min(int(chart_w / aspect), avail_h)

        chart_x = chart_x_start + (remaining_w - chart_w) // 2
        chart_y = m + (avail_h - chart_h) // 2
        return (chart_x, chart_y, chart_w, chart_h)


def stem_bbox(
    page_w: int,
    page_h: int,
    layout: LayoutKind,
    chart_x: int,
    chart_y: int,
    chart_w: int,
    chart_h: int,
    margin_px: int,
) -> Tuple[int, int, int, int]:
    """Return (x, y, w, h) of stem region (question text area)."""
    m = margin_px
    if layout == "top_bottom":
        return (m, m, page_w - 2 * m, chart_y - m)
    else:
        return (m, m, chart_x - m, page_h - 2 * m)
