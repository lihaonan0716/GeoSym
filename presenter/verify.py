"""
Phase 4: Verify & Fuse — 交叉验证测量结果，融合为增强证据。

类比: Computer Use Agent 的 pixelCompare
  — 在点击前验证目标位置像素是否发生变化。
  — GeoSym: 在输出前验证测量值是否合理，并融合为统一证据格式。

输出格式保持与现有 M1/L1/L2/L3 管线兼容
（candidate_facts / verified_facts / useful_facts）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .measure import DerivedFact, MeasuredPoint
from .ruler import RulerResult

logger = logging.getLogger(__name__)


def verify_measurements(
    ruler: RulerResult,
    measured_points: List[MeasuredPoint],
) -> List[MeasuredPoint]:
    """
    验证测量结果。

    验证手段:
      1. 边界检查: 像素坐标是否在绘图区域内
      2. 值域检查: 计算值是否在轴范围内
      3. 一致性降级: 对低置信度点标记

    返回过滤后的点列表（仅保留 verified 和 uncertain，移除 rejected）。
    """
    verified = []
    pr = ruler.plot_region

    plot_w = max(pr.x1 - pr.x0, 1)
    plot_h = max(pr.y1 - pr.y0, 1)
    margin_x = max(20, plot_w * 0.05)
    margin_y = max(20, plot_h * 0.05)

    for pt in measured_points:
        # 边界检查 (5% 比例边距，最少 20px)
        in_bounds = (pr.x0 - margin_x <= pt.pixel_x <= pr.x1 + margin_x) and \
                    (pr.y0 - margin_y <= pt.pixel_y <= pr.y1 + margin_y)

        if not in_bounds:
            logger.warning(f"[Verify] Rejected point '{pt.role}' at pixel ({pt.pixel_x}, {pt.pixel_y}): out of bounds")
            continue

        # 值域检查 (Y轴)
        if ruler.y_axis and pt.computed_y is not None:
            try:
                val = float(pt.computed_y)
                y_min = float(ruler.y_axis.value_range[0])
                y_max = float(ruler.y_axis.value_range[1])
                margin = abs(y_max - y_min) * 0.15
                if val < min(y_min, y_max) - margin or val > max(y_min, y_max) + margin:
                    pt.measurement_confidence = "low"
                    logger.warning(f"[Verify] Point '{pt.role}' y={val} outside axis range [{y_min}, {y_max}]")
            except (ValueError, TypeError):
                pass

        # 交叉验证: LLM 文字估值 vs ruler 计算值
        if pt.llm_estimate and pt.computed_y is not None and ruler.y_axis:
            try:
                import re
                nums = re.findall(r'-?[\d.]+', str(pt.llm_estimate))
                y_range = abs(float(ruler.y_axis.value_range[1]) - float(ruler.y_axis.value_range[0]))
                tick_interval = y_range / max(ruler.y_axis.n_ticks - 1, 1) if hasattr(ruler.y_axis, 'n_ticks') and ruler.y_axis.n_ticks > 1 else y_range * 0.2
                computed = float(pt.computed_y)
                for n in nums:
                    estimate = float(n)
                    if abs(estimate - computed) > tick_interval * 1.5 and abs(estimate) < y_range * 2:
                        pt.measurement_confidence = "low"
                        logger.warning(
                            f"[Verify] Cross-check mismatch for '{pt.role}': "
                            f"LLM estimate={estimate}, ruler computed={computed:.4f}, "
                            f"tick_interval={tick_interval:.4f}"
                        )
                        break
            except (ValueError, TypeError):
                pass

        verified.append(pt)

    logger.info(f"[Verify] {len(verified)}/{len(measured_points)} points passed verification")
    return verified


def fuse_evidence(
    ruler: RulerResult,
    measured_points: List[MeasuredPoint],
    derived_facts: List[DerivedFact],
    semantic_observations: Dict[str, Any],
) -> Dict[str, Any]:
    """
    将数值计算结果与语义观察融合为增强证据。

    输出格式兼容现有管线的 candidate_facts / verified_facts / useful_facts，
    但在内容中体现数值计算的贡献。
    """
    candidate_facts: List[str] = []
    verified_facts: List[str] = []
    useful_facts: List[str] = []

    # --- 从语义观察构建 verified_facts ---
    obs = semantic_observations
    chart_type = obs.get("chart_type", "unknown")
    verified_facts.append(f"Chart type: {chart_type}")

    if ruler.x_axis:
        x_info = f"X-axis: {ruler.x_axis.label or '(detected)'}, scale={ruler.x_axis.scale_type}"
        if ruler.x_axis.scale_type != "categorical":
            x_info += f", range [{ruler.x_axis.value_range[0]}, {ruler.x_axis.value_range[1]}]"
        verified_facts.append(x_info)

    if ruler.y_axis:
        y_info = f"Y-axis: {ruler.y_axis.label or '(detected)'}, scale={ruler.y_axis.scale_type}"
        if ruler.y_axis.scale_type != "categorical":
            y_info += f", range [{ruler.y_axis.value_range[0]}, {ruler.y_axis.value_range[1]}]"
        verified_facts.append(y_info)

    trend = obs.get("overall_trend", "")
    if trend:
        verified_facts.append(f"Overall trend: {trend}")

    for feature in obs.get("notable_features", []):
        verified_facts.append(f"Notable: {feature}")

    # --- 从测量点构建 candidate_facts 和 useful_facts ---
    for pt in measured_points:
        x_str = _display(pt.computed_x)
        y_str = _display(pt.computed_y)
        conf = pt.measurement_confidence

        # candidate_fact: 完整的像素→数值映射记录
        candidate_facts.append(
            f"{x_str} -> {y_str} "
            f"[pixel-calibrated: px({pt.pixel_x:.0f},{pt.pixel_y:.0f}) → ruler → {y_str}, "
            f"confidence={conf}]"
        )

        # verified_fact: 高/中置信度的人类可读描述
        if conf in ("high", "medium"):
            verified_facts.append(
                f"{pt.role.capitalize()}: {y_str} at {x_str} [GeoSym computed]"
            )

        # useful_fact: 带角色标签的摘要
        useful_facts.append(
            f"{pt.role}: {x_str} -> {y_str} [computed]"
        )

    # --- 从衍生量构建 useful_facts ---
    for df in derived_facts:
        useful_facts.append(
            f"{df.description}: {df.formula} [computed arithmetic]"
        )

    # --- chart_metadata ---
    chart_metadata = {
        "chart_type": chart_type,
        "x_axis": ruler.x_axis.label if ruler.x_axis else "",
        "y_axis": ruler.y_axis.label if ruler.y_axis else "",
        "series_count": obs.get("series_count", 1),
        "has_grid_lines": obs.get("has_grid_lines", False),
    }

    # --- computation_trace (论文关键证据) ---
    computation_trace = {
        "method": "ocr_ruler + llm_locate + pixel_to_real",
        "ruler_confidence": ruler.confidence,
        "x_axis_scale": ruler.x_axis.scale_type if ruler.x_axis else None,
        "y_axis_scale": ruler.y_axis.scale_type if ruler.y_axis else None,
        "points_measured": len(measured_points),
        "points_high_conf": sum(1 for p in measured_points if p.measurement_confidence == "high"),
        "derived_quantities": len(derived_facts),
        "fallback_used": False,
        "ruler_diagnostics": ruler.diagnostics,
    }

    return {
        "candidate_facts": candidate_facts,
        "verified_facts": verified_facts,
        "useful_facts": useful_facts,
        "chart_metadata": chart_metadata,
        "computation_trace": computation_trace,
    }


def _display(value: Any) -> str:
    """将任意值格式化为字符串"""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.2f}"
    return str(value)
