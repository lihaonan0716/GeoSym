"""
绘图工具类 - 统一处理坐标识别结果的可视化标注
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple

import sys as _sys
from pathlib import Path as _Path
_exp_root = _Path(__file__).resolve().parent.parent.parent
if str(_exp_root) not in _sys.path:
    _sys.path.insert(0, str(_exp_root))

def calculate_adaptive_font_scale(x_tick_labels: List[Dict], 
                                   y_tick_labels: List[Dict],
                                   image_shape: Tuple[int, int]) -> float:
    """
    根据刻度标签大小自适应计算字体大小
    
    通过分析刻度标签的字符尺寸，反推合适的font_scale，
    使得标注文字大小与图中原有文字大小协调一致
    
    Args:
        x_tick_labels: X轴刻度标签
        y_tick_labels: Y轴刻度标签
        image_shape: 图像尺寸 (height, width)
    
    Returns:
        字体缩放系数
    """
    # 计算平均标签大小
    all_labels = x_tick_labels + y_tick_labels
    
    if not all_labels:
        return 0.6  # 默认值
    
    # 计算每个标签的字符平均尺寸
    char_widths = []
    char_heights = []
    for label in all_labels:
        bbox = label.get('bbox', {})
        text = label.get('text', '')
        
        if not text:
            continue
        
        width = bbox.get('right', 0) - bbox.get('left', 0)
        height = bbox.get('bottom', 0) - bbox.get('top', 0)
        
        if width > 0 and height > 0:
            # 计算单个字符的平均尺寸
            char_count = len(text)
            avg_char_width = width / char_count
            char_widths.append(avg_char_width)
            char_heights.append(height)

    if not char_widths or not char_heights:
        return 0.6
    
    # 计算所有标签的平均字符尺寸
    avg_char_width = np.mean(char_widths)
    avg_char_height = np.mean(char_heights)
    
    print(f"[Visualization] Average tick-label character: width={avg_char_width:.1f}px, height={avg_char_height:.1f}px")
    
    # 估算最长的坐标文本（如 "(-10,-10)"）
    # 获取坐标范围来估算
    all_labels = x_tick_labels + y_tick_labels
    values = [abs(l.get('value', 0)) for l in all_labels if 'value' in l]
    max_value = max(values) if values else 10
    
    # 构造最长的可能文本
    if max_value >= 10:
        test_text = f"(-{int(max_value)},-{int(max_value)})"
    else:
        test_text = f"(-{int(max_value)},-{int(max_value)})"
    
    max_label_width = len(test_text) * min(avg_char_width, avg_char_height) * 0.8

    print(f"[Visualization] Maximum coordinate value={max_value:.0f}, test text='{test_text}', textwidth={max_label_width:.1f}px")

    # 反推font_scale：从小到大测试，找到能放入目标宽度的最大font_scale
    best_font_scale = 0.3
    
    for scale in np.arange(0.2, 1.0, 0.02):
        thickness = max(2, int(scale * 2))
        (text_width, text_height), baseline = cv2.getTextSize(
            test_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        
        # 加上padding
        padding = int(scale * 4)
        total_width = text_width + padding * 2
        
        if total_width <= max_label_width:
            best_font_scale = scale
        else:
            break  # 超过目标宽度，停止
    
    # 限制范围
    best_font_scale = max(0.25, min(1, best_font_scale))
    
    print(f"[Visualization] Final font_scale={best_font_scale:.2f}")
    
    return best_font_scale

def draw_text_with_background(image: np.ndarray, text: str, 
                               position: Tuple[int, int], font_scale: float,
                               text_color: Tuple[int, int, int],
                               bg_color: Tuple[int, int, int],
                               thickness: int = 2, padding: int = 5,
                               alpha: float = 0.7) -> np.ndarray:
    """
    绘制带半透明背景的文字
    
    Args:
        image: 图像
        text: 文字内容
        position: 文字位置 (x, y)
        font_scale: 字体大小
        text_color: 文字颜色 (B, G, R)
        bg_color: 背景颜色 (B, G, R)
        thickness: 文字粗细
        padding: 背景padding
        alpha: 背景透明度 (0-1)
    
    Returns:
        绘制后的图像
    """
    # 获取文字尺寸
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    x, y = position
    
    # 计算背景矩形坐标
    x1 = x - padding
    y1 = y - text_height - padding
    x2 = x + text_width + padding
    y2 = y + baseline + padding
    
    # 确保坐标在图像范围内
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # 创建半透明背景
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    
    # 混合背景
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # 绘制文字
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
               font_scale, text_color, thickness, cv2.LINE_AA)
    
    return image

def calculate_label_position_and_bbox(point: Dict, image_shape: Tuple[int, int],
                                      font_scale: float = 0.5, thickness: int = 2, 
                                      padding: int = 5) -> Dict:
    """
    统一计算标注的位置和边界框（用于碰撞检测和绘制）
    
    这个函数封装了标注位置计算的所有逻辑，确保碰撞检测和实际绘制使用相同的规则。
    
    Args:
        point: 交点数据，包含 pixel_x, pixel_y, real_x_rounded, real_y_rounded
        image_shape: 图像尺寸 (height, width)
        font_scale: 字体大小
        thickness: 文字粗细
        padding: 背景padding
    
    Returns:
        字典包含：
        {
            'text': str,  # 标注文字内容，如 "(1,2)"
            'text_pos': (x, y),  # 文字绘制位置（左下角基准点）
            'bbox': (x1, y1, x2, y2),  # 标注框边界
            'text_width': int,  # 文字宽度
            'text_height': int,  # 文字高度
            'baseline': int  # 文字基线
        }
    """
    px = point['pixel_x']
    py = point['pixel_y']
    rx = point['real_x_rounded']
    ry = point['real_y_rounded']
    
    h_img, w_img = image_shape
    
    # 生成标注文字
    label = f'({rx},{ry})'
    
    # 获取文字尺寸
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    # 默认偏移（右上方）
    text_offset_x = 12
    text_offset_y = -12
    
    # 调整文字位置避免超出边界
    text_x = px + text_offset_x
    text_y = py + text_offset_y
    
    # 边界自适应：如果超出右边界，文字放左边（根据实际文字宽度计算）
    if text_x + text_width + padding > w_img:
        text_x = px - text_width - text_offset_x
        # 如果放左边后超出左边界，则紧贴左边界
        if text_x - padding < 0:
            text_x = padding
    
    # 边界自适应：如果超出上边界，文字放下面
    if text_y - text_height - padding < 0:
        text_y = py + text_height + abs(text_offset_y)
    
    # 边界自适应：如果超出下边界，文字放上面
    if text_y + baseline + padding > h_img:
        text_y = py - abs(text_offset_y)
    
    # 计算背景矩形坐标
    x1 = text_x - padding
    y1 = text_y - text_height - padding
    x2 = text_x + text_width + padding
    y2 = text_y + baseline + padding
    
    # 最后确保坐标在图像范围内（保险）
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)
    
    return {
        'text': label,
        'text_pos': (text_x, text_y),
        'bbox': (x1, y1, x2, y2),
        'text_width': text_width,
        'text_height': text_height,
        'baseline': baseline
    }

def draw_intersections_on_image(image: np.ndarray, points: List[Dict],
                                x_tick_labels: List[Dict] = None,
                                y_tick_labels: List[Dict] = None,
                                font_scale: float = None,
                                text_color: Tuple[int, int, int] = (20, 100, 220),
                                bg_color: Tuple[int, int, int] = (255, 255, 255),
                                circle_alpha: float = 0.5,
                                text_background_alpha: float = 0.5,
                                text_alpha: float = 0.66) -> np.ndarray:
    """
    在图像上绘制交点标注（统一入口函数）
    
    Args:
        image: 要标注的图像
        points: 交点列表，每个点包含 pixel_x, pixel_y, real_x_rounded, real_y_rounded
        x_tick_labels: X轴刻度标签（用于自适应字体计算）
        y_tick_labels: Y轴刻度标签（用于自适应字体计算）
        font_scale: 字体大小（如果为None，则自动计算）
        text_color: 文字颜色 (B, G, R)，默认深橙色
        bg_color: 背景颜色 (B, G, R)，默认白色
        circle_alpha: 交点透明度 (0-1)，默认0.3
        text_background_alpha: 文字背景透明度 (0-1)，默认0.5
        text_alpha: 文字透明度 (0-1)，默认0.5
    
    Returns:
        标注后的图像
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    
    # 计算自适应字体大小（如果没有指定）
    if font_scale is None:
        if x_tick_labels is None:
            x_tick_labels = []
        if y_tick_labels is None:
            y_tick_labels = []
        font_scale = calculate_adaptive_font_scale(x_tick_labels, y_tick_labels, (h, w))
    
    thickness = max(1, int(font_scale * 2))
    padding = int(font_scale * 4)
    
    # 计算自适应交点大小（基于字体大小）
    # 基准：font_scale=0.6 时，outer_radius=7
    circle_scale = 0.6
    circle_outer_radius = max(4, int(7 * font_scale * circle_scale))
    circle_inner_radius = max(3, int(5 * font_scale * circle_scale))
    circle_border_radius = circle_outer_radius + 1
    circle_border_thickness = max(1, int(2 * font_scale * circle_scale))
    
    print(f"[DrawUtil] Adaptive parameters: font_scale={font_scale:.2f}, thickness={thickness}, padding={padding}")
    print(f"[DrawUtil] Intersection radius: outer={circle_outer_radius}px, inner={circle_inner_radius}px, border={circle_border_radius}px")
    
    # 绘制每个交点
    for pt in points:
        px = pt['pixel_x']
        py = pt['pixel_y']
        
        # 第一步：绘制圆点（从当前状态复制）
        circle_overlay = annotated.copy()
        cv2.circle(circle_overlay, (px, py), circle_outer_radius, (255, 255, 0), -1)  # 青色外圈
        cv2.circle(circle_overlay, (px, py), circle_inner_radius, (0, 255, 0), -1)    # 绿色内圈
        cv2.circle(circle_overlay, (px, py), circle_border_radius, (255, 255, 255), circle_border_thickness)  # 白色边框
        cv2.addWeighted(circle_overlay, circle_alpha, annotated, 1 - circle_alpha, 0, annotated)
        
        # 第二步：计算标注位置
        label_info = calculate_label_position_and_bbox(
            pt, (h, w), font_scale, thickness, padding
        )
        
        label = label_info['text']
        text_pos = label_info['text_pos']
        x, y = text_pos
        
        # 获取文字尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # 计算背景矩形坐标
        x1 = x - padding
        y1 = y - text_height - padding
        x2 = x + text_width + padding
        y2 = y + baseline + padding
        
        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # 第三步：绘制文字背景（从当前状态复制）
        text_background_overlay = annotated.copy()
        cv2.rectangle(text_background_overlay, (x1, y1), (x2, y2), bg_color, -1)
        cv2.addWeighted(text_background_overlay, text_background_alpha, annotated, 1 - text_background_alpha, 0, annotated)

        # 第四步：绘制文字（从当前状态复制）
        text_overlay = annotated.copy()
        cv2.putText(text_overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.addWeighted(text_overlay, text_alpha, annotated, 1 - text_alpha, 0, annotated)

    return annotated

