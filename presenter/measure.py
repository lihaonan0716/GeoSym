"""
Phase 3: Measure — 像素坐标 → 精确数值转换 + 衍生量计算。

这是 GeoSym "数值计算增强" 的核心: 将 LLM 定位的像素坐标，
通过标尺（Phase 1 建立的 AxisMapping）转换为精确数值。

类比: Computer Use Agent 的坐标空间转换
  — 模型给出 normalized 坐标, 系统转为 physical pixels 执行点击。
  — GeoSym: LLM 给出像素坐标, 系统转为实际数值。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ruler import AxisMapping, RulerResult, epoch_to_str

logger = logging.getLogger(__name__)


@dataclass
class MeasuredPoint:
    """单个测量点"""
    role: str                          # peak, latest, intercept, ...
    description: str                   # LLM 的语义描述
    pixel_x: float = 0.0
    pixel_y: float = 0.0
    computed_x: Any = None             # 数值计算得到的 X 值
    computed_y: Any = None             # 数值计算得到的 Y 值
    llm_estimate: str = ""             # LLM 的原始估值
    measurement_confidence: str = "medium"
    computation_method: str = "linear_interp"


@dataclass
class DerivedFact:
    """衍生量"""
    description: str
    formula: str                       # e.g. "14.7 - 4.1 = 10.6"
    value: Any = None
    unit: str = ""


def pixel_to_real(ruler: RulerResult, pixel_x: float, pixel_y: float):
    """
    核心数值计算: 像素坐标 → 实际值。

    对应 recognizer 的 CoordinateRecognizer.pixel_to_real(),
    但扩展支持了日期轴、对数轴、分类轴。
    """
    real_x = None
    real_y = None
    method_x = "none"
    method_y = "none"

    if ruler.x_axis:
        real_x = ruler.x_axis.pixel_to_value(pixel_x)
        method_x = ruler.x_axis.scale_type

    if ruler.y_axis:
        real_y = ruler.y_axis.pixel_to_value(pixel_y)
        method_y = ruler.y_axis.scale_type

    return real_x, real_y, f"{method_x}+{method_y}"


def measure_points(
    ruler: RulerResult,
    located_points: List[Dict[str, Any]],
) -> List[MeasuredPoint]:
    """
    将 LLM 定位的像素坐标通过标尺转换为精确数值。

    对每个 located_point:
      1. pixel_to_real(pixel_x, pixel_y) → (real_x, real_y)
      2. 确定测量置信度（基于像素是否在绘图区域内、标尺置信度）
    """
    results = []
    pr = ruler.plot_region

    for pt in located_points:
        px = float(pt.get("pixel_x", 0))
        py = float(pt.get("pixel_y", 0))

        real_x, real_y, method = pixel_to_real(ruler, px, py)

        # 边界检查
        in_bounds = (pr.x0 <= px <= pr.x1) and (pr.y0 <= py <= pr.y1)

        # 置信度
        if in_bounds and ruler.confidence == "high":
            conf = "high"
        elif in_bounds:
            conf = "medium"
        else:
            conf = "low"

        # 格式化 computed 值
        display_x = _format_value(real_x, ruler.x_axis)
        display_y = _format_value(real_y, ruler.y_axis)

        results.append(MeasuredPoint(
            role=pt.get("role", "data_point"),
            description=pt.get("description", ""),
            pixel_x=px,
            pixel_y=py,
            computed_x=display_x,
            computed_y=display_y,
            llm_estimate=pt.get("estimated_value", ""),
            measurement_confidence=conf,
            computation_method=method,
        ))

    logger.info(f"[Measure] Measured {len(results)} points")
    return results


def _format_value(value: Any, axis: Optional[AxisMapping]) -> Any:
    """格式化数值用于展示"""
    if value is None:
        return None
    if axis and axis.scale_type == "datetime":
        if isinstance(value, str):
            return value
        return epoch_to_str(value)
    if isinstance(value, float):
        # 小数精度: 根据刻度间隔决定
        if axis and len(axis.tick_values) >= 2:
            try:
                tick_vals = [float(v) for v in axis.tick_values[:2]]
                interval = abs(tick_vals[1] - tick_vals[0])
                if interval >= 10:
                    return round(value, 1)
                elif interval >= 1:
                    return round(value, 2)
                else:
                    return round(value, 3)
            except (ValueError, TypeError):
                pass
        return round(value, 2)
    return value


def compute_derived_quantities(
    measured_points: List[MeasuredPoint],
    question: str,
) -> List[DerivedFact]:
    """
    基于精确测量值计算任务所需的衍生量。

    这些是确定性的数值运算，不依赖 LLM:
      - 两点差值
      - 变化率
      - 极值识别
    """
    derived: List[DerivedFact] = []

    if len(measured_points) < 2:
        return derived

    # 收集有数值 Y 的点
    numeric_points = []
    for pt in measured_points:
        try:
            y_val = float(pt.computed_y)
            numeric_points.append((pt, y_val))
        except (ValueError, TypeError):
            continue

    if len(numeric_points) < 2:
        return derived

    # 找最大/最小
    max_pt, max_val = max(numeric_points, key=lambda x: x[1])
    min_pt, min_val = min(numeric_points, key=lambda x: x[1])

    # 差值计算
    diff = round(max_val - min_val, 2)
    derived.append(DerivedFact(
        description=f"Difference between {max_pt.role} and {min_pt.role}",
        formula=f"{max_val} - {min_val} = {diff}",
        value=diff,
    ))

    # 如果问题涉及 "how much lower/higher" 或 "difference"
    q_lower = question.lower()
    if any(kw in q_lower for kw in ["lower", "higher", "difference", "change", "decline", "drop"]):
        # 找 peak 和 latest (fuzzy role matching)
        _peak_roles = {"peak", "maximum", "highest", "max", "top"}
        _latest_roles = {"latest", "most_recent", "current", "end", "last"}
        peak = next((pt for pt, _ in numeric_points if pt.role.lower() in _peak_roles), None)
        latest = next((pt for pt, _ in numeric_points if pt.role.lower() in _latest_roles), None)
        if peak and latest:
            try:
                p_val = float(peak.computed_y)
                l_val = float(latest.computed_y)
                d = round(p_val - l_val, 2)
                derived.append(DerivedFact(
                    description=f"Decline from peak to latest",
                    formula=f"{p_val} - {l_val} = {d}",
                    value=d,
                ))
            except (ValueError, TypeError):
                pass

    return derived
