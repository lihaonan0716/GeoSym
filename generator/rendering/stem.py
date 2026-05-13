"""
Draw stem (question text area) on the image.
Supports automatic word-wrapping for long lines.
"""

import random
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/segoeui.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/PingFang.ttc",
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


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Word-wrap a text string to fit within max_width pixels."""
    if not text:
        return []
    try:
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
    except Exception:
        text_width = len(text) * 20
    if text_width <= max_width:
        return [text]

    words = text.split(" ")
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        try:
            tw = font.getbbox(test)[2] - font.getbbox(test)[0]
        except Exception:
            tw = len(test) * 20
        if tw <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines if lines else [text]


def estimate_wrapped_line_count(
    lines: List[str],
    font_size: int,
    max_width: int,
    max_lines: int | None = None,
) -> int:
    """Estimate how many rendered lines the stem will occupy after wrapping."""
    if max_width <= 0 or not lines:
        return 0

    font = _load_font(font_size)
    all_wrapped: List[str] = []
    for line in lines:
        all_wrapped.extend(_wrap_text(line, font, max_width))

    if max_lines is not None:
        all_wrapped = all_wrapped[:max_lines]
    return len(all_wrapped)


def draw_stem(
    img: np.ndarray,
    stem_rect: Tuple[int, int, int, int],
    lines: List[str],
    font_size: int,
    color_bgr: Tuple[int, int, int],
    max_lines: int = 5,
) -> None:
    """
    Draw text lines in stem_rect with automatic word-wrapping.
    """
    sx, sy, sw, sh = stem_rect
    if sw <= 0 or sh <= 0 or not lines:
        return

    font = _load_font(font_size)

    all_wrapped = []
    for line in lines:
        all_wrapped.extend(_wrap_text(line, font, sw - 20))

    if len(all_wrapped) > max_lines:
        all_wrapped = all_wrapped[:max_lines]

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    color_rgb = tuple(reversed(color_bgr))

    line_height = int(font_size * 1.6)
    total_text_h = len(all_wrapped) * line_height
    y_start = sy + max(10, (sh - total_text_h) // 2)

    y = y_start
    for line in all_wrapped:
        if y + line_height > sy + sh:
            break
        draw.text((sx + 10, y), line, fill=color_rgb, font=font)
        y += line_height

    arr = np.array(pil)
    img[:] = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
