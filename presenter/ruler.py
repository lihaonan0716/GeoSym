"""
Phase 1: Build Ruler — 从真实图表图像建立坐标标尺。

通过 OCR 提取坐标轴刻度标签，建立像素位置 ↔ 实际数值的映射关系。
这是 GeoSym "数值计算增强" 在真实图表场景下的核心基础设施。

设计灵感: Computer Use Agent 的 "server handles all scaling" 理念
  — LLM 只负责定位（像素坐标），系统负责精确换算（数值计算）。
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AxisMapping:
    """单轴的像素 ↔ 数值映射"""
    label: str                            # 轴标题
    tick_pixels: List[float]              # 刻度的像素位置
    tick_values: List[Any]                # 刻度的实际值 (float / str)
    scale_type: str = "linear"            # linear | log | datetime | categorical
    pixel_range: Tuple[float, float] = (0.0, 0.0)
    value_range: Tuple[Any, Any] = (0.0, 0.0)
    # 线性映射参数: value = scale * pixel + offset
    scale: float = 0.0
    offset: float = 0.0

    def pixel_to_value(self, pixel: float) -> Any:
        """核心数值计算: 像素 → 实际值"""
        if self.scale_type == "categorical":
            return self._nearest_category(pixel)
        if self.scale_type == "log":
            log_val = self.scale * pixel + self.offset
            return 10 ** log_val
        if self.scale_type == "datetime":
            epoch = self.scale * pixel + self.offset
            return epoch_to_str(epoch)
        # linear (default)
        return self.scale * pixel + self.offset

    def _nearest_category(self, pixel: float) -> Any:
        if not self.tick_pixels:
            return None
        dists = [abs(tp - pixel) for tp in self.tick_pixels]
        idx = int(np.argmin(dists))
        return self.tick_values[idx]


@dataclass
class BBox:
    """绘图区域边界框"""
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0


@dataclass
class RulerResult:
    """坐标标尺构建结果"""
    success: bool = False
    chart_type: str = "unknown"
    x_axis: Optional[AxisMapping] = None
    y_axis: Optional[AxisMapping] = None
    plot_region: BBox = field(default_factory=BBox)
    confidence: str = "low"               # high | medium | low
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tick value parsing
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Common date patterns
_DATE_PATTERNS = [
    # 2020-04, 2020/04
    (re.compile(r"^(\d{4})[-/](\d{1,2})$"), lambda m: datetime(int(m.group(1)), int(m.group(2)), 1)),
    # 2020-04-15
    (re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$"),
     lambda m: datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))),
    # Jan 2020, Apr 2020
    (re.compile(r"^([A-Za-z]{3})\s*(\d{4})$"),
     lambda m: datetime(int(m.group(2)), _MONTH_MAP.get(m.group(1).lower()[:3], 1), 1)),
    # Q1 2021
    (re.compile(r"^Q([1-4])\s*(\d{4})$"),
     lambda m: datetime(int(m.group(2)), (int(m.group(1)) - 1) * 3 + 1, 1)),
]


def _try_parse_date(text: str) -> Optional[float]:
    """尝试解析日期, 返回年份浮点数 (如 2020.25 = 2020-04)"""
    s = text.strip()
    for pat, factory in _DATE_PATTERNS:
        m = pat.match(s)
        if m:
            try:
                dt = factory(m)
                # 用年份+月份小数表示, 避免 timestamp 在 Windows 上的兼容性问题
                return dt.year + (dt.month - 1) / 12.0
            except (ValueError, KeyError, OSError):
                continue
    return None


def epoch_to_str(year_float: float) -> str:
    """将年份浮点数 (如 2020.25) 转为 'YYYY-MM' 字符串"""
    try:
        year = int(year_float)
        month_frac = year_float - year
        month = int(round(month_frac * 12)) + 1
        month = max(1, min(12, month))
        if month == 1 and abs(month_frac) < 0.02:
            return str(year)
        return f"{year}-{month:02d}"
    except (ValueError, OverflowError):
        return str(year_float)


def parse_tick_value(text: str) -> Tuple[Any, str]:
    """
    解析刻度标签文本为数值。

    Returns:
        (parsed_value, scale_type)
        scale_type: "numeric" | "datetime" | "categorical"
    """
    s = text.strip()
    if not s:
        return s, "categorical"

    # 去除千位分隔符
    cleaned = s.replace(",", "").replace(" ", "")

    # 日期格式优先检测 (在 numeric 之前, 避免 "2020" 被误判为 numeric)
    dt_val = _try_parse_date(s)
    if dt_val is not None:
        return dt_val, "datetime"

    # 纯年份 (4位数字, 1800-2199 范围)
    if re.match(r"^\d{4}$", s) and 1800 <= int(s) <= 2199:
        # 用年份数值直接作为值, 避免 Windows 上 <1970 的 timestamp 问题
        return float(int(s)), "datetime"

    # 百分比
    if cleaned.endswith("%"):
        try:
            return float(cleaned[:-1]), "numeric"
        except ValueError:
            pass

    # 带单位后缀 (e.g. "400 MPa", "$1.2T")
    m = re.match(r"^[\\$€£¥]?([-+]?\d+\.?\d*)\s*[A-Za-z%°]*$", cleaned)
    if m:
        try:
            return float(m.group(1)), "numeric"
        except ValueError:
            pass

    # 纯数字 (int or float)
    try:
        return float(cleaned), "numeric"
    except ValueError:
        pass

    # 分类标签
    return s, "categorical"


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _call_ocr(image_path: str) -> Dict[str, Any]:
    """调用 OCR, 返回原始结果"""
    from recognizer.mathpix.mathpix_image_ocr import MathpixImageOCRTool
    tool = MathpixImageOCRTool()

    # 如果是文件路径, 转为 base64 data-uri
    p = Path(image_path)
    if p.exists():
        raw = p.read_bytes()
        b64 = base64.b64encode(raw).decode()
        ext = p.suffix.lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
        src = f"data:{mime};base64,{b64}"
    else:
        src = image_path

    result = tool.ocr_image(src)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return {}
    return result if isinstance(result, dict) else {}


def _extract_line_data_boxes(ocr_result: Dict) -> List[Dict[str, Any]]:
    """从 OCR line_data 中提取文本 bbox 信息"""
    boxes = []
    for line in ocr_result.get("line_data", []):
        text = line.get("text", "").strip()
        if not text:
            continue
        cnt = line.get("cnt", [])
        if len(cnt) >= 4:
            xs = [p[0] for p in cnt]
            ys = [p[1] for p in cnt]
            boxes.append({
                "text": text,
                "left": min(xs), "right": max(xs),
                "top": min(ys), "bottom": max(ys),
                "center_x": sum(xs) / len(xs),
                "center_y": sum(ys) / len(ys),
            })
    return boxes


# ---------------------------------------------------------------------------
# Spatial clustering
# ---------------------------------------------------------------------------

def _cluster_tick_labels(
    boxes: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
) -> Tuple[List[Dict], List[Dict]]:
    """
    将 OCR 结果按空间位置聚类为 X轴刻度 和 Y轴刻度。

    策略:
      - 沿图像底部 1/4 水平排列 → X轴刻度
      - 沿图像左侧 1/4 垂直排列 → Y轴刻度
    """
    if not boxes:
        return [], []

    x_ticks: List[Dict] = []
    y_ticks: List[Dict] = []

    # 分区阈值
    bottom_threshold = image_height * 0.72
    left_threshold = image_width * 0.28

    for box in boxes:
        cy = box["center_y"]
        cx = box["center_x"]
        text = box["text"]

        # 过滤标题或过长文本 (通常不是刻度)
        if len(text) > 20:
            continue

        parsed_val, scale_type = parse_tick_value(text)

        # 底部区域 → X轴候选
        if cy > bottom_threshold and scale_type != "categorical":
            x_ticks.append({**box, "value": parsed_val, "scale_type": scale_type})
        # 左侧区域 → Y轴候选
        elif cx < left_threshold and scale_type == "numeric":
            y_ticks.append({**box, "value": parsed_val, "scale_type": scale_type})

    # 去重: 左边距的 x-tick 候选如果值域与 y-ticks 一致, 归入 y-ticks
    if y_ticks and x_ticks:
        y_vals = [float(t["value"]) for t in y_ticks]
        y_min, y_max = min(y_vals), max(y_vals)
        y_span = abs(y_max - y_min) if y_max != y_min else 1.0
        reassign = []
        keep = []
        for t in x_ticks:
            v = float(t["value"])
            in_left_margin = t["center_x"] < left_threshold
            in_y_range = (y_min - y_span * 0.5) <= v <= (y_max + y_span * 0.5)
            if in_left_margin and in_y_range:
                reassign.append(t)
            else:
                keep.append(t)
        if reassign:
            y_ticks.extend(reassign)
            x_ticks = keep

    # X轴刻度异常值过滤: 如果混入了不同量级的值, 移除离群值
    if len(x_ticks) >= 3:
        x_vals = [float(t["value"]) for t in x_ticks]
        median_val = float(np.median(x_vals))
        if median_val != 0:
            x_ticks = [
                t for t in x_ticks
                if abs(float(t["value"]) - median_val) / abs(median_val) < 0.5
                or abs(float(t["value"]) - median_val) < 1.0
            ]
        else:
            q75 = float(np.percentile([abs(v) for v in x_vals], 75))
            if q75 > 1.0:
                x_ticks = [
                    t for t in x_ticks
                    if abs(float(t["value"])) >= q75 * 0.1
                ]

    # 按位置排序
    x_ticks.sort(key=lambda t: t["center_x"])
    y_ticks.sort(key=lambda t: t["center_y"])

    return x_ticks, y_ticks


# ---------------------------------------------------------------------------
# Axis mapping builder
# ---------------------------------------------------------------------------

def _build_axis_mapping(
    tick_labels: List[Dict],
    axis: str,
) -> Optional[AxisMapping]:
    """
    建立单轴的像素 ↔ 数值映射。

    核心数值计算:
      1. 提取 (pixel, value) 对
      2. 单调性验证
      3. 最小二乘线性回归拟合 value = scale * pixel + offset
    """
    if len(tick_labels) < 2:
        return None

    # 确定 scale_type
    scale_types = [t["scale_type"] for t in tick_labels]
    dominant_type = max(set(scale_types), key=scale_types.count)

    coord_key = "center_x" if axis == "x" else "center_y"
    pixels = [float(t[coord_key]) for t in tick_labels]
    values = [float(t["value"]) for t in tick_labels]

    # 单调性检查: 像素单调递增时, 值应单调递增或递减
    # 注意: Y轴在图像中像素从上到下递增, 但数值通常从下到上递增,
    # 因此 Y轴 diffs 全为负是正常的 (单调递减也合法)
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)

    if pos > 0 and neg > 0:
        # 非单调 (既有递增又有递减): 尝试移除离群值
        logger.warning(f"[Ruler-{axis}] Non-monotonic tick values (pos={pos}, neg={neg}), attempting cleanup")
        tick_labels, pixels, values = _remove_outliers(tick_labels, pixels, values, coord_key)
        if len(pixels) < 2:
            return None

    # 线性回归: value = scale * pixel + offset
    px = np.array(pixels)
    vl = np.array(values)
    A = np.vstack([px, np.ones(len(px))]).T
    result = np.linalg.lstsq(A, vl, rcond=None)
    scale, offset = result[0]

    # 确定 scale_type
    if dominant_type == "datetime":
        final_scale_type = "datetime"
    elif dominant_type == "numeric":
        # 检查是否对数轴: 刻度值是否等比
        if len(values) >= 3 and all(v > 0 for v in values):
            ratios = [values[i + 1] / values[i] for i in range(len(values) - 1)]
            if len(set(round(r, 2) for r in ratios)) == 1 and ratios[0] > 1.5:
                final_scale_type = "log"
                # 对对数轴重新拟合: log10(value) = scale * pixel + offset
                log_vl = np.log10(vl)
                result = np.linalg.lstsq(A, log_vl, rcond=None)
                scale, offset = result[0]
            else:
                final_scale_type = "linear"
        else:
            final_scale_type = "linear"
    else:
        final_scale_type = "categorical"

    axis_label_text = ""
    return AxisMapping(
        label=axis_label_text,
        tick_pixels=pixels,
        tick_values=[t["value"] for t in tick_labels],
        scale_type=final_scale_type,
        pixel_range=(min(pixels), max(pixels)),
        value_range=(min(values), max(values)),
        scale=float(scale),
        offset=float(offset),
    )


def _remove_outliers(
    tick_labels: List[Dict],
    pixels: List[float],
    values: List[float],
    coord_key: str,
) -> Tuple[List[Dict], List[float], List[float]]:
    """移除导致非单调性的离群刻度"""
    if len(pixels) < 3:
        return tick_labels, pixels, values

    # 计算相邻间隔
    intervals = []
    for i in range(len(values) - 1):
        pixel_diff = pixels[i + 1] - pixels[i]
        value_diff = values[i + 1] - values[i]
        if abs(pixel_diff) > 1e-6:
            intervals.append(value_diff / pixel_diff)

    if not intervals:
        return tick_labels, pixels, values

    median_slope = float(np.median(intervals))

    # 保留与中位斜率一致方向的点
    keep = [True] * len(values)
    for i in range(len(values) - 1):
        pixel_diff = pixels[i + 1] - pixels[i]
        value_diff = values[i + 1] - values[i]
        if abs(pixel_diff) > 1e-6:
            local_slope = value_diff / pixel_diff
            if (median_slope > 0 and local_slope < 0) or (median_slope < 0 and local_slope > 0):
                keep[i + 1] = False

    new_ticks = [t for t, k in zip(tick_labels, keep) if k]
    new_pixels = [p for p, k in zip(pixels, keep) if k]
    new_values = [v for v, k in zip(values, keep) if k]
    return new_ticks, new_pixels, new_values


# ---------------------------------------------------------------------------
# Plot region inference
# ---------------------------------------------------------------------------

def _infer_plot_region(
    image_shape: Tuple[int, ...],
    x_ticks: List[Dict],
    y_ticks: List[Dict],
) -> BBox:
    """从刻度标签位置推断绘图区域边界"""
    h, w = image_shape[:2]
    margin = 5

    # 左边界: Y轴刻度标签的右边
    if y_ticks:
        left = max(t["right"] for t in y_ticks) + margin
    else:
        left = w * 0.1

    # 下边界: X轴刻度标签的上方
    if x_ticks:
        bottom = min(t["top"] for t in x_ticks) - margin
    else:
        bottom = h * 0.85

    # 右边界: 最后一个 X轴刻度标签附近
    if x_ticks:
        right = max(t["center_x"] for t in x_ticks) + (x_ticks[-1]["right"] - x_ticks[-1]["center_x"]) + margin
        right = min(right, w - margin)
    else:
        right = w * 0.95

    # 上边界
    top = h * 0.05

    return BBox(x0=float(left), y0=float(top), x1=float(right), y1=float(bottom))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_ruler(image_path: str, ocr_backend: str = "ocr") -> RulerResult:
    """
    从图表图像建立坐标标尺。

    Pipeline:
      1. OCR 提取文本区域
      2. 空间聚类分离 X轴刻度 vs Y轴刻度
      3. 解析刻度值
      4. 建立像素 ↔ 数值映射 (线性回归)
      5. 推断绘图区域边界

    这是 GeoSym M0 "数值计算增强" 的第一阶段:
    建立"尺子"，为后续 pixel_to_real 计算提供基础。
    """
    diagnostics: Dict[str, Any] = {"ocr_backend": ocr_backend}

    # 1. 读取图像尺寸
    img = cv2.imread(image_path)
    if img is None:
        diagnostics["error"] = f"Cannot read image: {image_path}"
        return RulerResult(diagnostics=diagnostics)

    img_h, img_w = img.shape[:2]
    diagnostics["image_size"] = {"width": img_w, "height": img_h}

    # 2. OCR
    try:
        ocr_result = _call_ocr(image_path)
        diagnostics["ocr_raw_lines"] = len(ocr_result.get("line_data", []))
    except Exception as e:
        diagnostics["error"] = f"OCR failed: {e}"
        return RulerResult(diagnostics=diagnostics)

    boxes = _extract_line_data_boxes(ocr_result)
    diagnostics["ocr_boxes"] = len(boxes)

    if not boxes:
        diagnostics["error"] = "OCR returned no text boxes"
        return RulerResult(diagnostics=diagnostics)

    # 3. 空间聚类
    x_ticks, y_ticks = _cluster_tick_labels(boxes, img_w, img_h)
    diagnostics["x_tick_count"] = len(x_ticks)
    diagnostics["y_tick_count"] = len(y_ticks)
    diagnostics["x_tick_labels"] = [{"text": t["text"], "px": round(t["center_x"], 1)} for t in x_ticks]
    diagnostics["y_tick_labels"] = [{"text": t["text"], "px": round(t["center_y"], 1)} for t in y_ticks]

    if len(x_ticks) < 2 and len(y_ticks) < 2:
        diagnostics["error"] = "Insufficient tick labels (need >=2 per axis)"
        return RulerResult(diagnostics=diagnostics)

    # 4. 建立映射
    x_axis = _build_axis_mapping(x_ticks, axis="x") if len(x_ticks) >= 2 else None
    y_axis = _build_axis_mapping(y_ticks, axis="y") if len(y_ticks) >= 2 else None

    if x_axis:
        diagnostics["x_mapping"] = {
            "scale_type": x_axis.scale_type,
            "scale": round(x_axis.scale, 8),
            "offset": round(x_axis.offset, 4),
            "pixel_range": x_axis.pixel_range,
        }
    if y_axis:
        diagnostics["y_mapping"] = {
            "scale_type": y_axis.scale_type,
            "scale": round(y_axis.scale, 8),
            "offset": round(y_axis.offset, 4),
            "pixel_range": y_axis.pixel_range,
        }

    # 5. 推断绘图区域
    plot_region = _infer_plot_region(img.shape, x_ticks, y_ticks)
    diagnostics["plot_region"] = {
        "x0": round(plot_region.x0, 1), "y0": round(plot_region.y0, 1),
        "x1": round(plot_region.x1, 1), "y1": round(plot_region.y1, 1),
    }

    # 6. 确定置信度
    has_x = x_axis is not None
    has_y = y_axis is not None
    if has_x and has_y:
        confidence = "high"
    elif has_x or has_y:
        confidence = "medium"
    else:
        confidence = "low"

    success = has_x or has_y
    diagnostics["success"] = success
    diagnostics["confidence"] = confidence

    logger.info(f"[Ruler] Built ruler: success={success}, confidence={confidence}, "
                f"x_ticks={len(x_ticks)}, y_ticks={len(y_ticks)}")

    return RulerResult(
        success=success,
        x_axis=x_axis,
        y_axis=y_axis,
        plot_region=plot_region,
        confidence=confidence,
        diagnostics=diagnostics,
    )


def format_ruler_for_prompt(ruler: RulerResult) -> str:
    """
    将标尺信息格式化为 LLM prompt 中的上下文字符串。

    这是注入给 LLM 的 "coordinate reference"，让 LLM 知道像素参考锚点，
    从而能更准确地输出像素坐标。
    """
    lines = ["## Coordinate Reference (computed by GeoSym OCR)", ""]

    if ruler.x_axis:
        ax = ruler.x_axis
        lines.append(f"X-axis: {ax.label or '(detected)'}")
        lines.append(f"  Scale type: {ax.scale_type}")
        lines.append("  Tick marks at pixel positions:")
        for px, val in zip(ax.tick_pixels, ax.tick_values):
            if ax.scale_type == "datetime":
                display_val = epoch_to_str(val) if isinstance(val, (int, float)) else str(val)
            else:
                display_val = str(val)
            lines.append(f"    pixel_x={px:.0f} → \"{display_val}\"")
        lines.append("")

    if ruler.y_axis:
        ax = ruler.y_axis
        lines.append(f"Y-axis: {ax.label or '(detected)'}")
        lines.append(f"  Scale type: {ax.scale_type}")
        lines.append("  Tick marks at pixel positions:")
        for px, val in zip(ax.tick_pixels, ax.tick_values):
            lines.append(f"    pixel_y={px:.0f} → \"{val}\"")
        lines.append("")

    pr = ruler.plot_region
    lines.append(f"Plot region: top-left ({pr.x0:.0f}, {pr.y0:.0f}) to bottom-right ({pr.x1:.0f}, {pr.y1:.0f})")
    lines.append("")

    return "\n".join(lines)
