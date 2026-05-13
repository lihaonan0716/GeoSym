import cv2
import numpy as np
from typing import Tuple

import sys as _sys
from pathlib import Path as _Path
_exp_root = _Path(__file__).resolve().parent.parent.parent
if str(_exp_root) not in _sys.path:
    _sys.path.insert(0, str(_exp_root))
from i18n import log_tag as _lt  # noqa: E402

class CoordinateSystemPreprocessor:
    """坐标系图像预处理器"""
    
    def __init__(self):
        self.intermediate_results = {}
        # 曲线擦除参数
        self.color_saturation_threshold = 90
        self.color_value_threshold = 60
    
    def resize_to_match_original(self, enhanced_image: np.ndarray, 
                                 original_shape: Tuple[int, int]) -> Tuple[np.ndarray, dict]:
        """
        将AI增强后的图像调整回原图尺寸（中心对齐裁剪）
        
        Args:
            enhanced_image: AI增强后的图像
            original_shape: 原图尺寸 (height, width)
        
        Returns:
            tuple: (调整后的图像, 变换参数字典)
            变换参数包含:
                - scale: 缩放比例
                - crop_x: X轴裁剪偏移
                - crop_y: Y轴裁剪偏移
                - ai_output_size: AI输出图像尺寸 (width, height)
                - original_size: 原图尺寸 (width, height)
        """
        orig_h, orig_w = original_shape[:2]
        enh_h, enh_w = enhanced_image.shape[:2]
        
        print(f"[{_lt('尺寸调整')}] 原图: {orig_w}x{orig_h}, 增强后: {enh_w}x{enh_h}")
        
        # 初始化变换参数
        transform_params = {
            'scale': 1.0,
            'crop_x': 0,
            'crop_y': 0,
            'ai_output_size': (enh_w, enh_h),
            'original_size': (orig_w, orig_h)
        }
        
        if (enh_h, enh_w) == (orig_h, orig_w):
            print(f"[{_lt('尺寸调整')}] 尺寸相同，无需调整")
            return enhanced_image, transform_params
        
        # 计算缩放比例
        scale_w = orig_w / enh_w
        scale_h = orig_h / enh_h
        
        # 选择较大的比例（确保缩放后不会小于原图）
        scale = max(scale_w, scale_h)
        
        # 缩放图像
        new_w = int(enh_w * scale)
        new_h = int(enh_h * scale)
        
        print(f"[{_lt('尺寸调整')}] 缩放比例: {scale:.3f}, 缩放后: {new_w}x{new_h}")
        
        resized = cv2.resize(enhanced_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 中心裁剪到原图尺寸
        crop_x = (new_w - orig_w) // 2
        crop_y = (new_h - orig_h) // 2
        
        # 确保裁剪坐标有效
        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)
        
        result = resized[crop_y:crop_y+orig_h, crop_x:crop_x+orig_w]
        
        print(f"[{_lt('尺寸调整')}] 裁剪区域: ({crop_x}, {crop_y}) -> ({crop_x+orig_w}, {crop_y+orig_h})")
        print(f"[{_lt('尺寸调整')}] 最终尺寸: {result.shape[1]}x{result.shape[0]}")
        
        # 保存变换参数
        transform_params['scale'] = scale
        transform_params['crop_x'] = crop_x
        transform_params['crop_y'] = crop_y
        
        return result, transform_params
    
    def transform_labels_for_ai_enhanced(self, labels: list, transform_params: dict) -> list:
        """
        对刻度标签坐标进行正向变换（从原图坐标变换到AI增强调整后的坐标）
        
        变换步骤：
        1. 原图坐标 → 居中对齐到AI输出图
        2. AI输出图坐标 → 缩放
        3. 缩放后坐标 → 减去裁剪偏移
        
        Args:
            labels: 刻度标签列表，每个标签包含坐标信息（x, y, center等）
            transform_params: resize_to_match_original返回的变换参数
        
        Returns:
            变换后的刻度标签列表
        """
        if not labels or not transform_params:
            return labels
        
        scale = transform_params['scale']
        crop_x = transform_params['crop_x']
        crop_y = transform_params['crop_y']
        ai_w, ai_h = transform_params['ai_output_size']
        orig_w, orig_h = transform_params['original_size']
        
        print(f"[{_lt('刻度标签变换')}] 开始变换 {len(labels)} 个标签")
        print(f"[{_lt('刻度标签变换')}] 原图尺寸: {orig_w}x{orig_h}, AI输出: {ai_w}x{ai_h}")
        print(f"[{_lt('刻度标签变换')}] resize缩放: {scale:.3f}, 裁剪偏移: ({crop_x}, {crop_y})")
        
        # 步骤0: 计算按高度等比例缩放的比例
        height_scale = ai_h / orig_h
        scaled_width = orig_w * height_scale
        print(f"[{_lt('刻度标签变换')}] 高度缩放: {height_scale:.4f}, 缩放后宽度: {scaled_width:.1f}")
        
        # 步骤1: 计算宽度方向的居中对齐偏移
        width_offset = (scaled_width - ai_w) / 2.0
        print(f"[{_lt('刻度标签变换')}] 宽度居中偏移: {width_offset:.1f}")
        
        transformed_labels = []
        for label in labels:
            transformed_label = label.copy()
            
            # 处理 center_x 和 center_y
            if 'center_x' in label:
                orig_x = label['center_x']
                # 步骤0: 按高度等比例缩放
                step0_x = orig_x * height_scale
                # 步骤1: 宽度居中对齐
                step1_x = step0_x - width_offset
                # 步骤2: resize缩放
                step2_x = step1_x * scale
                # 步骤3: 裁剪偏移
                final_x = step2_x - crop_x
                transformed_label['center_x'] = final_x
            
            if 'center_y' in label:
                orig_y = label['center_y']
                # 步骤0: 按高度等比例缩放
                step0_y = orig_y * height_scale
                # 步骤1: 宽度居中对齐（Y坐标不变）
                step1_y = step0_y
                # 步骤2: resize缩放
                step2_y = step1_y * scale
                # 步骤3: 裁剪偏移
                final_y = step2_y - crop_y
                transformed_label['center_y'] = final_y
            
            # 处理 bbox 字典
            if 'bbox' in label and isinstance(label['bbox'], dict):
                bbox = label['bbox'].copy()
                
                if 'left' in bbox:
                    orig_coord = bbox['left']
                    step0 = orig_coord * height_scale
                    step1 = step0 - width_offset
                    step2 = step1 * scale
                    bbox['left'] = step2 - crop_x
                
                if 'right' in bbox:
                    orig_coord = bbox['right']
                    step0 = orig_coord * height_scale
                    step1 = step0 - width_offset
                    step2 = step1 * scale
                    bbox['right'] = step2 - crop_x
                
                if 'top' in bbox:
                    orig_coord = bbox['top']
                    step0 = orig_coord * height_scale
                    step1 = step0  # Y坐标在步骤1不变
                    step2 = step1 * scale
                    bbox['top'] = step2 - crop_y
                
                if 'bottom' in bbox:
                    orig_coord = bbox['bottom']
                    step0 = orig_coord * height_scale
                    step1 = step0  # Y坐标在步骤1不变
                    step2 = step1 * scale
                    bbox['bottom'] = step2 - crop_y
                
                transformed_label['bbox'] = bbox
            
            # 处理独立的坐标字段（x, y, center等）
            coord_fields = ['x', 'y', 'center', 'left', 'right', 'top', 'bottom']
            for field in coord_fields:
                if field in label and field not in ['bbox']:  # 避免重复处理
                    if field in ['x', 'left', 'right']:
                        # X坐标
                        orig_coord = label[field]
                        step0 = orig_coord * height_scale
                        step1 = step0 - width_offset
                        step2 = step1 * scale
                        transformed_label[field] = step2 - crop_x
                    elif field in ['y', 'top', 'bottom']:
                        # Y坐标
                        orig_coord = label[field]
                        step0 = orig_coord * height_scale
                        step1 = step0  # Y坐标在步骤1不变
                        step2 = step1 * scale
                        transformed_label[field] = step2 - crop_y
                    elif field == 'center' and isinstance(label[field], (list, tuple)):
                        # center是(x, y)元组
                        orig_x, orig_y = label[field]
                        # X坐标变换
                        step0_x = orig_x * height_scale
                        step1_x = step0_x - width_offset
                        step2_x = step1_x * scale
                        final_x = step2_x - crop_x
                        # Y坐标变换
                        step0_y = orig_y * height_scale
                        step1_y = step0_y  # Y坐标在步骤1不变
                        step2_y = step1_y * scale
                        final_y = step2_y - crop_y
                        transformed_label[field] = (final_x, final_y)
            
            transformed_labels.append(transformed_label)
        
        print(f"[{_lt('刻度标签变换')}] 变换完成")
        print(f"[{_lt('刻度标签变换')}] 变换后的标签: {transformed_labels}")
        return transformed_labels
    
    def detect_curve_mask(self, image, feather_size=5):
        """
        检测曲线mask（基于颜色）
        
        用于在边缘检测阶段擦除曲线边缘，但不修改原图
        
        Args:
            image: 输入图像（彩色）
            feather_size: 羽化大小（像素）
        
        Returns:
            曲线的mask（二值图像）
        """
        print(f"[{_lt('曲线检测')}] 开始检测彩色曲线（羽化{feather_size}px）...")
        
        if len(image.shape) == 2:
            print("  警告：输入是灰度图，返回空mask")
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        h, w = image.shape[:2]
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义多个颜色范围（排除黑白灰）
        s_thresh = self.color_saturation_threshold
        v_thresh = self.color_value_threshold
        
        color_masks = []
        
        # 红色
        lower_red1 = np.array([0, s_thresh, v_thresh])
        upper_red1 = np.array([10, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        lower_red2 = np.array([170, s_thresh, v_thresh])
        upper_red2 = np.array([180, 255, 255])
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        color_masks.append(('red', mask_red))
        
        # 蓝色
        lower_blue = np.array([100, s_thresh, v_thresh])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        color_masks.append(('blue', mask_blue))
        
        # 绿色
        lower_green = np.array([40, s_thresh, v_thresh])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        color_masks.append(('green', mask_green))
        
        # 橙色/黄色
        lower_orange = np.array([10, s_thresh, v_thresh])
        upper_orange = np.array([30, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        color_masks.append(('orange', mask_orange))
        
        # 紫色/品红色
        lower_purple = np.array([130, s_thresh, v_thresh])
        upper_purple = np.array([170, 255, 255])
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
        color_masks.append(('purple', mask_purple))
        
        # 合并所有彩色区域
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for color_name, mask in color_masks:
            pixel_count = np.sum(mask > 0)
            if pixel_count > 0:
                print(f"  检测到 {color_name} 颜色: {pixel_count} 像素")
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # 羽化：膨胀mask
        if feather_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (feather_size*2+1, feather_size*2+1))
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        
        total_pixels = np.sum(combined_mask > 0)
        print(f"[{_lt('曲线检测')}] 完成，检测到曲线区域: {total_pixels} 个像素")
        
        # 保存中间结果
        self.intermediate_results['curve_mask'] = combined_mask
        
        return combined_mask

    def build_text_mask(self, image_shape, x_tick_labels=None, y_tick_labels=None, padding: int = 10):
        """
        根据 OCR 刻度标签 bbox 构建文字屏蔽区域，避免文字笔画进入网格检测。
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        all_labels = (x_tick_labels or []) + (y_tick_labels or [])
        if not all_labels:
            return mask

        for label in all_labels:
            try:
                left = int(np.floor(label['left'])) - padding
                top = int(np.floor(label['top'])) - padding
                right = int(np.ceil(label['right'])) + padding
                bottom = int(np.ceil(label['bottom'])) + padding
            except (KeyError, TypeError, ValueError):
                bbox = label.get('bbox') if isinstance(label, dict) else None
                if not isinstance(bbox, dict):
                    continue
                try:
                    left = int(np.floor(bbox['left'])) - padding
                    top = int(np.floor(bbox['top'])) - padding
                    right = int(np.ceil(bbox['right'])) + padding
                    bottom = int(np.ceil(bbox['bottom'])) + padding
                except (KeyError, TypeError, ValueError):
                    continue

            left = max(0, left)
            top = max(0, top)
            right = min(w, right)
            bottom = min(h, bottom)
            if right > left and bottom > top:
                cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)

        self.intermediate_results['text_mask'] = mask
        print(f"[{_lt('文字屏蔽')}] 文本屏蔽像素数: {int(np.sum(mask > 0))}")
        return mask

    def _build_grid_candidate_map(self, gray: np.ndarray, max_line_gap: int) -> np.ndarray:
        """
        构建网格线候选图。

        不能只依赖 Canny，因为灰色虚线在轻压缩、轻模糊、抗锯齿后
        边缘响应会明显变弱。这里融合：
        1. 原灰度边缘
        2. CLAHE 增强后的边缘
        3. 面向暗细线的 black-hat 响应
        """
        # 局部对比度增强，提升浅灰虚线与白底的区分度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        gray_eq_blur = cv2.GaussianBlur(gray_eq, (3, 3), 0)

        # 常规边缘 + 增强后边缘
        edges_base = cv2.Canny(gray_blur, 40, 120, apertureSize=3)
        edges_eq = cv2.Canny(gray_eq_blur, 30, 100, apertureSize=3)

        # 用定向 black-hat 把浅灰细线直接从白底里“提”出来
        span = max(15, int(max_line_gap) * 3 + 9)
        if span % 2 == 0:
            span += 1
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (span, 3))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, span))
        blackhat_h = cv2.morphologyEx(gray_eq_blur, cv2.MORPH_BLACKHAT, h_kernel)
        blackhat_v = cv2.morphologyEx(gray_eq_blur, cv2.MORPH_BLACKHAT, v_kernel)
        blackhat = cv2.max(blackhat_h, blackhat_v)

        _, dark_line_mask = cv2.threshold(
            blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        candidate_map = cv2.bitwise_or(edges_base, edges_eq)
        candidate_map = cv2.bitwise_or(candidate_map, dark_line_mask)

        self.intermediate_results['edges_base'] = edges_base
        self.intermediate_results['edges_eq'] = edges_eq
        self.intermediate_results['dark_line_mask'] = dark_line_mask
        self.intermediate_results['grid_candidate_map'] = candidate_map
        return candidate_map
    
    def detect_grid_lines(self, image, rho=1, theta=np.pi/180, threshold=100, 
                          min_line_length=50, max_line_gap=10, curve_mask=None,
                          text_mask=None):
        """
        检测网格线
        
        Args:
            image: 输入图像
            rho: Hough变换距离精度
            theta: Hough变换角度精度
            threshold: Hough变换阈值
            min_line_length: 最小线段长度
            max_line_gap: 最大线段间隙
            curve_mask: 曲线mask（可选），用于从边缘检测中擦除曲线
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 构建网格候选图：不能只靠单一路径边缘检测
        edges = self._build_grid_candidate_map(gray, max_line_gap=max_line_gap)
        
        # 如果提供了曲线mask，从边缘检测结果中擦除曲线边缘
        if curve_mask is not None:
            print(f"  从边缘检测中擦除曲线区域...")
            edges_before = np.sum(edges > 0)
            edges[curve_mask > 0] = 0  # 擦除曲线区域的边缘
            edges_after = np.sum(edges > 0)
            print(f"  边缘像素: {edges_before} → {edges_after} (减少{edges_before - edges_after}个)")

        # 如果提供了文字mask，也从候选图中擦除标签文字区域
        if text_mask is not None:
            print(f"  从边缘检测中擦除文字区域...")
            edges_before = np.sum(edges > 0)
            edges[text_mask > 0] = 0
            edges_after = np.sum(edges > 0)
            print(f"  候选像素: {edges_before} → {edges_after} (减少{edges_before - edges_after}个)")
        
        self.intermediate_results['edges'] = edges

        # 对边缘做定向闭运算，尽量连起灰色/虚线网格被锯齿和轻微遮挡打断的缺口
        close_span = max(3, int(max_line_gap) * 2 + 1)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_span, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_span))
        edges_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, h_kernel)
        edges_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, v_kernel)
        line_edges = cv2.bitwise_or(edges_h, edges_v)
        self.intermediate_results['edges_line_enhanced'] = line_edges
        
        # Hough直线检测
        lines = cv2.HoughLinesP(line_edges, rho, theta, threshold, 
                                minLineLength=min_line_length, 
                                maxLineGap=max_line_gap)

        # 如果主检测召回太低，再做一次更宽松的补检，避免弱灰线/虚线整体漏检
        primary_count = 0 if lines is None else len(lines)
        if primary_count < 12:
            relaxed_threshold = max(35, int(round(threshold * 0.72)))
            relaxed_min_line_length = max(20, int(round(min_line_length * 0.70)))
            relaxed_max_line_gap = max(max_line_gap + 6, int(round(max_line_gap * 1.8)))
            print(
                f"  网格线主检测偏稀，启动补检: lines={primary_count}, "
                f"threshold={relaxed_threshold}, minLen={relaxed_min_line_length}, gap={relaxed_max_line_gap}"
            )
            relaxed_lines = cv2.HoughLinesP(
                line_edges, rho, theta, relaxed_threshold,
                minLineLength=relaxed_min_line_length,
                maxLineGap=relaxed_max_line_gap
            )
            if relaxed_lines is not None:
                lines = relaxed_lines if lines is None else np.concatenate([lines, relaxed_lines], axis=0)
        
        # 可视化检测到的线
        line_image = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        self.intermediate_results['detected_lines'] = line_image
        
        return lines
    
    def classify_lines(self, lines, angle_threshold=10):
        """
        将检测到的线分类为水平线和垂直线
        
        Args:
            lines: 检测到的线段
            angle_threshold: 角度阈值（度）
        """
        if lines is None:
            return [], []
        
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算角度
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = np.abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
            
            # 分类
            if angle < angle_threshold:
                horizontal_lines.append(line[0])
            elif angle > (90 - angle_threshold):
                vertical_lines.append(line[0])
        
        return horizontal_lines, vertical_lines
    
    def connect_dashed_lines(self, lines, is_horizontal=True, 
                            collinear_threshold=10, gap_threshold=50):
        """
        连接虚线段，将断续的线段连成完整的线
        
        Args:
            lines: 线段列表 [[x1, y1, x2, y2], ...]
            is_horizontal: 是否为水平线
            collinear_threshold: 共线判定阈值（像素）
            gap_threshold: 最大间隙阈值（像素），小于此值认为是虚线
        
        Returns:
            连接后的线段列表
        """
        if not lines or len(lines) == 0:
            return []
        
        lines = np.array(lines, dtype=float)
        connected_lines = []

        print(f"[{_lt('虚线连接')}] 输入线段数: {len(lines)}, 方向: {'水平' if is_horizontal else '垂直'}")

        # 先把线段规整成“正交坐标 + 主轴区间”的形式。
        segments = []
        for line in lines:
            x1, y1, x2, y2 = [float(v) for v in line]
            if is_horizontal:
                ortho = (y1 + y2) / 2.0
                start = min(x1, x2)
                end = max(x1, x2)
            else:
                ortho = (x1 + x2) / 2.0
                start = min(y1, y2)
                end = max(y1, y2)
            segments.append({
                "ortho": ortho,
                "start": start,
                "end": end,
            })

        # 第一步：按共线带分组。只允许进入与当前带中心接近的线段，
        # 防止相邻网格线被错误串连。
        segments.sort(key=lambda s: s["ortho"])
        bands = []
        current_band = [segments[0]]
        current_center = segments[0]["ortho"]

        for seg in segments[1:]:
            if abs(seg["ortho"] - current_center) <= collinear_threshold:
                current_band.append(seg)
                current_center = float(np.median([s["ortho"] for s in current_band]))
            else:
                bands.append(current_band)
                current_band = [seg]
                current_center = seg["ortho"]
        if current_band:
            bands.append(current_band)

        # 第二步：在每个共线带内，只按主轴区间的真实间隙做连接。
        for band in bands:
            band.sort(key=lambda s: (s["start"], s["end"]))
            merged_start = band[0]["start"]
            merged_end = band[0]["end"]
            merged_members = [band[0]]

            for seg in band[1:]:
                gap = max(0.0, float(seg["start"]) - float(merged_end))
                if gap <= gap_threshold:
                    merged_end = max(merged_end, seg["end"])
                    merged_members.append(seg)
                else:
                    if is_horizontal:
                        fit_x = []
                        fit_y = []
                        for m in merged_members:
                            fit_x.extend([m["start"], m["end"]])
                            fit_y.extend([m["ortho"], m["ortho"]])
                        if len(set(round(x, 3) for x in fit_x)) >= 2:
                            slope, intercept = np.polyfit(fit_x, fit_y, 1)
                        else:
                            slope, intercept = 0.0, float(np.median(fit_y))
                        y1 = float(slope * merged_start + intercept)
                        y2 = float(slope * merged_end + intercept)
                        connected_lines.append([merged_start, y1, merged_end, y2])
                    else:
                        fit_y = []
                        fit_x = []
                        for m in merged_members:
                            fit_y.extend([m["start"], m["end"]])
                            fit_x.extend([m["ortho"], m["ortho"]])
                        if len(set(round(y, 3) for y in fit_y)) >= 2:
                            slope, intercept = np.polyfit(fit_y, fit_x, 1)
                        else:
                            slope, intercept = 0.0, float(np.median(fit_x))
                        x1 = float(slope * merged_start + intercept)
                        x2 = float(slope * merged_end + intercept)
                        connected_lines.append([x1, merged_start, x2, merged_end])
                    merged_start = seg["start"]
                    merged_end = seg["end"]
                    merged_members = [seg]

            if is_horizontal:
                fit_x = []
                fit_y = []
                for m in merged_members:
                    fit_x.extend([m["start"], m["end"]])
                    fit_y.extend([m["ortho"], m["ortho"]])
                if len(set(round(x, 3) for x in fit_x)) >= 2:
                    slope, intercept = np.polyfit(fit_x, fit_y, 1)
                else:
                    slope, intercept = 0.0, float(np.median(fit_y))
                y1 = float(slope * merged_start + intercept)
                y2 = float(slope * merged_end + intercept)
                connected_lines.append([merged_start, y1, merged_end, y2])
            else:
                fit_y = []
                fit_x = []
                for m in merged_members:
                    fit_y.extend([m["start"], m["end"]])
                    fit_x.extend([m["ortho"], m["ortho"]])
                if len(set(round(y, 3) for y in fit_y)) >= 2:
                    slope, intercept = np.polyfit(fit_y, fit_x, 1)
                else:
                    slope, intercept = 0.0, float(np.median(fit_x))
                x1 = float(slope * merged_start + intercept)
                x2 = float(slope * merged_end + intercept)
                connected_lines.append([x1, merged_start, x2, merged_end])

        print(f"[{_lt('虚线连接')}] 共线带数量: {len(bands)}")
        print(f"[{_lt('虚线连接')}] 连接后线段数: {len(connected_lines)}, 减少了 {len(lines) - len(connected_lines)} 条")

        return connected_lines
    
    def extend_and_merge_lines(self, lines, is_horizontal=True, distance_threshold=20, 
                               image_shape=None, enable_adaptive_merge=True):
        """
        改进的线段合并算法：保留原始跨度与斜率，只做保守去重合并。

        这里不再把线条延伸到整张图的边界，因为那会把本来对齐的网格
        直接拉出 chart 区域，并在存在残余倾斜时造成系统性错位。
        合并阶段现在只负责：
        1. 合并明显重复的近邻线
        2. 保留虚线连接后已经稳定的真实线位置
        
        Args:
            lines: 线段列表 [[x1, y1, x2, y2], ...]
            is_horizontal: 是否为水平线
            distance_threshold: 第一次合并的距离阈值（像素）
            image_shape: 图像尺寸 (height, width)
            enable_adaptive_merge: 是否启用基于间距的二次合并
        
        Returns:
            merged_lines: 合并后的线段列表
        """
        if not lines or len(lines) == 0 or image_shape is None:
            return []
        
        height, width = image_shape[:2]
        lines = np.array(lines, dtype=float)

        # 统一线段方向，便于后续按主轴排序和拟合
        segments = []
        for line in lines:
            x1, y1, x2, y2 = [float(v) for v in line]
            if is_horizontal:
                if x2 < x1:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                ref = (y1 + y2) / 2.0
                start = x1
                end = x2
            else:
                if y2 < y1:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                ref = (x1 + x2) / 2.0
                start = y1
                end = y2
            segments.append({
                "line": [x1, y1, x2, y2],
                "ref": ref,
                "start": start,
                "end": end,
            })

        if not segments:
            return []

        segments.sort(key=lambda s: s["ref"])

        # 第一步：主坐标近邻去重，按组保留真实斜率和跨度
        grouped_segments = []
        current_group = [segments[0]]
        current_center = float(segments[0]["ref"])

        for seg in segments[1:]:
            if abs(float(seg["ref"]) - current_center) <= distance_threshold:
                current_group.append(seg)
                current_center = float(np.median([g["ref"] for g in current_group]))
            else:
                grouped_segments.append(current_group)
                current_group = [seg]
                current_center = float(seg["ref"])
        if current_group:
            grouped_segments.append(current_group)

        # 第二步：保守二次去重，只合并几乎重合的重复组
        if enable_adaptive_merge and len(grouped_segments) > 2:
            refs = np.array([np.median([g["ref"] for g in group]) for group in grouped_segments], dtype=float)
            gaps = np.diff(refs)
            positive_gaps = gaps[gaps > 1e-6]
            if len(positive_gaps) > 0:
                adaptive_threshold = min(
                    max(2.0, float(distance_threshold) * 0.45),
                    float(np.median(positive_gaps)) * 0.28,
                )
            else:
                adaptive_threshold = max(2.0, float(distance_threshold) * 0.45)

            deduped_groups = []
            current = list(grouped_segments[0])
            current_center = float(np.median([g["ref"] for g in current]))
            for group in grouped_segments[1:]:
                group_center = float(np.median([g["ref"] for g in group]))
                if abs(group_center - current_center) <= adaptive_threshold:
                    current.extend(group)
                    current_center = float(np.median([g["ref"] for g in current]))
                else:
                    deduped_groups.append(current)
                    current = list(group)
                    current_center = group_center
            if current:
                deduped_groups.append(current)
            grouped_segments = deduped_groups

        merged_lines = []
        for group in grouped_segments:
            xs = []
            ys = []
            starts = []
            ends = []
            for seg in group:
                x1, y1, x2, y2 = seg["line"]
                xs.extend([x1, x2])
                ys.extend([y1, y2])
                starts.append(seg["start"])
                ends.append(seg["end"])

            if is_horizontal:
                left = float(min(starts))
                right = float(max(ends))
                if len(set(round(x, 3) for x in xs)) >= 2:
                    slope, intercept = np.polyfit(xs, ys, 1)
                else:
                    slope, intercept = 0.0, float(np.median(ys))
                y_left = float(slope * left + intercept)
                y_right = float(slope * right + intercept)
                merged_lines.append([
                    max(0.0, min(float(width), left)),
                    max(0.0, min(float(height), y_left)),
                    max(0.0, min(float(width), right)),
                    max(0.0, min(float(height), y_right)),
                ])
            else:
                top = float(min(starts))
                bottom = float(max(ends))
                if len(set(round(y, 3) for y in ys)) >= 2:
                    slope, intercept = np.polyfit(ys, xs, 1)
                else:
                    slope, intercept = 0.0, float(np.median(xs))
                x_top = float(slope * top + intercept)
                x_bottom = float(slope * bottom + intercept)
                merged_lines.append([
                    max(0.0, min(float(width), x_top)),
                    max(0.0, min(float(height), top)),
                    max(0.0, min(float(width), x_bottom)),
                    max(0.0, min(float(height), bottom)),
                ])

        return merged_lines
    
    def calculate_rotation_angle(self, horizontal_lines, vertical_lines):
        """
        计算图像的旋转角度
        
        Args:
            horizontal_lines: 水平线列表
            vertical_lines: 垂直线列表
        """
        angles = []
        
        # 从水平线计算角度（相对于x轴）
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) > 1:  # 确保是水平线
                angle = np.arctan2(dy, dx) * 180 / np.pi
                angles.append(angle)
        
        # 从垂直线计算角度
        # 垂直线相对于x轴的角度应该是90°，如果有倾斜，减去90°就是旋转角度
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            dx = x2 - x1
            dy = y2 - y1
            if abs(dy) > 1:  # 确保是垂直线
                # 计算线段相对于x轴的角度
                angle_to_x = np.arctan2(dy, dx) * 180 / np.pi
                # 垂直线应该是90°，所以旋转角度 = angle_to_x - 90°
                rotation = angle_to_x - 90
                angles.append(rotation)
        
        if len(angles) == 0:
            return 0
        
        # 使用中位数作为旋转角度（更鲁棒）
        rotation_angle = np.median(angles)
        
        print(f"[{_lt('角度计算')}] 水平线数量: {len(horizontal_lines)}, 垂直线数量: {len(vertical_lines)}")
        print(f"[{_lt('角度计算')}] 所有角度样本: {[f'{a:.2f}' for a in sorted(angles)[:10]]}...")  # 显示排序后前10个
        print(f"[{_lt('角度计算')}] 中位数角度（原始）: {rotation_angle:.2f}°")
        
        # 角度规范化：分两步处理
        # 第一步：将角度归一化到 -180 到 180 范围
        while rotation_angle > 180:
            rotation_angle -= 360
        while rotation_angle < -180:
            rotation_angle += 360
        
        print(f"[{_lt('角度归一化步骤1')}] 归一化到±180°: {rotation_angle:.2f}°")
        
        # 第二步：处理±90°附近的角度（线段方向相反导致的±180°偏差）
        if rotation_angle > 135:
            rotation_angle = rotation_angle - 180
            print(f"[{_lt('角度归一化步骤2')}] 角度 > 135°，减去180° -> {rotation_angle:.2f}°")
        elif rotation_angle < -135:
            rotation_angle = rotation_angle + 180
            print(f"[{_lt('角度归一化步骤2')}] 角度 < -135°，加上180° -> {rotation_angle:.2f}°")
        
        # 第三步：坐标系倾斜一般不会超过±45度
        if rotation_angle > 45:
            rotation_angle = rotation_angle - 90
            print(f"[{_lt('角度归一化步骤3')}] 角度 > 45°，减去90° -> {rotation_angle:.2f}°")
        elif rotation_angle < -45:
            rotation_angle = rotation_angle + 90
            print(f"[{_lt('角度归一化步骤3')}] 角度 < -45°，加上90° -> {rotation_angle:.2f}°")
        else:
            print(f"[{_lt('角度归一化步骤3')}] 角度在±45°范围内，无需调整")
        
        print(f"[{_lt('角度计算')}] 最终角度: {rotation_angle:.2f}°")
        
        return rotation_angle
    
    def rotate_image(self, image, angle):
        """
        旋转图像并自动裁剪掉留白边缘
        
        Args:
            image: 输入图像
            angle: 旋转角度（度）
        
        Returns:
            tuple: (旋转并裁剪后的图像, 裁剪偏移量(x, y), 完整旋转矩阵)
        """
        if abs(angle) < 0.1:
            return image, (0, 0), None
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        print(f"[{_lt('图像旋转')}] 原始尺寸: {w}x{h}, 旋转角度: {angle:.2f}°")
        
        # 计算旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算旋转后的图像尺寸
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # 调整旋转矩阵（包含平移到新图像中心）
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # 执行旋转
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        
        print(f"[{_lt('图像旋转')}] 旋转后尺寸: {new_w}x{new_h}")
        
        # 使用几何计算裁剪掉白边
        cropped, crop_offset = self.crop_rotated_rectangle(rotated, angle, w, h)
        
        print(f"[{_lt('图像裁剪')}] 裁剪后尺寸: {cropped.shape[1]}x{cropped.shape[0]}, 裁剪偏移: {crop_offset}")
        
        # 返回裁剪后的图像、裁剪偏移量和完整的旋转矩阵（包含平移）
        return cropped, crop_offset, M
    
    def detect_grid_corners(self, horizontal_lines, vertical_lines, image_shape, 
                           num_lines_to_fit=3, min_line_length_ratio=0.05):
        """
        从网格线中检测四个角点（使用多条外围线加权拟合）
        
        Args:
            horizontal_lines: 水平线列表 [[x1, y1, x2, y2], ...]
            vertical_lines: 垂直线列表 [[x1, y1, x2, y2], ...]
            image_shape: 图像尺寸 (height, width)
            num_lines_to_fit: 用于拟合的外围线条数量（默认3条）
            min_line_length_ratio: 最小线段长度比例（相对于图像尺寸，默认0.3）
        
        Returns:
            corners: 四个角点 [top_left, top_right, bottom_right, bottom_left]
                    如果检测失败返回 None
        """
        if not horizontal_lines or not vertical_lines:
            print(f"[{_lt('角点检测')}] 失败：水平线或垂直线不足")
            return None
        
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            print(f"[{_lt('角点检测')}] 失败：水平线{len(horizontal_lines)}条，垂直线{len(vertical_lines)}条（至少需要各2条）")
            return None
        
        h, w = image_shape[:2]
        
        # 1. 过滤短线段
        def line_length(line):
            x1, y1, x2, y2 = line
            return np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        min_h_length = w * min_line_length_ratio
        min_v_length = h * min_line_length_ratio
        
        h_lines_filtered = [line for line in horizontal_lines if line_length(line) >= min_h_length]
        v_lines_filtered = [line for line in vertical_lines if line_length(line) >= min_v_length]
        
        print(f"[{_lt('角点检测')}] 线段过滤: 水平线 {len(horizontal_lines)}->{len(h_lines_filtered)}条, "
              f"垂直线 {len(vertical_lines)}->{len(v_lines_filtered)}条")
        
        if len(h_lines_filtered) < 2 or len(v_lines_filtered) < 2:
            print(f"[{_lt('角点检测')}] 失败：过滤后水平线{len(h_lines_filtered)}条，垂直线{len(v_lines_filtered)}条（至少需要各2条）")
            return None
        
        # 2. 拟合函数：使用多条线进行加权线性拟合
        def fit_lines(lines, is_horizontal=True):
            """
            对多条线进行加权拟合，返回拟合后的直线参数
            
            Args:
                lines: 线段列表
                is_horizontal: 是否为水平线
            
            Returns:
                (a, b): 直线参数，y = ax + b (水平线) 或 x = ay + b (垂直线)
            """
            points = []
            weights = []
            
            for line in lines:
                x1, y1, x2, y2 = line
                length = line_length(line)
                
                # 收集线段上的点和权重
                points.append([x1, y1])
                points.append([x2, y2])
                weights.extend([length, length])  # 长度作为权重
            
            points = np.array(points)
            weights = np.array(weights)
            
            if is_horizontal:
                # 水平线：拟合 y = ax + b
                X = points[:, 0]  # x坐标
                Y = points[:, 1]  # y坐标
            else:
                # 垂直线：拟合 x = ay + b
                X = points[:, 1]  # y坐标
                Y = points[:, 0]  # x坐标
            
            # 加权最小二乘拟合
            W = np.diag(weights)
            X_matrix = np.vstack([X, np.ones(len(X))]).T
            
            # (X^T W X)^-1 X^T W Y
            try:
                params = np.linalg.inv(X_matrix.T @ W @ X_matrix) @ X_matrix.T @ W @ Y
                a, b = params
                return a, b
            except:
                # 如果矩阵不可逆，使用简单平均
                if is_horizontal:
                    return 0, np.average(Y, weights=weights)
                else:
                    return 0, np.average(Y, weights=weights)
        
        # 3. 找到最外围的线并拟合
        # 水平线：按y坐标排序
        h_lines_sorted = sorted(h_lines_filtered, key=lambda line: line[1])
        top_h_lines = h_lines_sorted[:min(num_lines_to_fit, len(h_lines_sorted))]
        bottom_h_lines = h_lines_sorted[-min(num_lines_to_fit, len(h_lines_sorted)):]
        
        # 垂直线：按x坐标排序
        v_lines_sorted = sorted(v_lines_filtered, key=lambda line: line[0])
        left_v_lines = v_lines_sorted[:min(num_lines_to_fit, len(v_lines_sorted))]
        right_v_lines = v_lines_sorted[-min(num_lines_to_fit, len(v_lines_sorted)):]
        
        # 拟合四条边界线
        top_a, top_b = fit_lines(top_h_lines, is_horizontal=True)
        bottom_a, bottom_b = fit_lines(bottom_h_lines, is_horizontal=True)
        left_a, left_b = fit_lines(left_v_lines, is_horizontal=False)
        right_a, right_b = fit_lines(right_v_lines, is_horizontal=False)
        
        print(f"[{_lt('角点检测')}] 拟合边界线:")
        print(f"  上边: y = {top_a:.4f}x + {top_b:.2f}")
        print(f"  下边: y = {bottom_a:.4f}x + {bottom_b:.2f}")
        print(f"  左边: x = {left_a:.4f}y + {left_b:.2f}")
        print(f"  右边: x = {right_a:.4f}y + {right_b:.2f}")
        
        # 4. 计算拟合线的交点
        def fitted_line_intersection(h_params, v_params):
            """
            计算拟合线的交点
            h_params: (a, b) 水平线参数 y = ax + b
            v_params: (a, b) 垂直线参数 x = ay + b
            """
            h_a, h_b = h_params
            v_a, v_b = v_params
            
            # y = h_a * x + h_b
            # x = v_a * y + v_b
            # 
            # 代入得: y = h_a * (v_a * y + v_b) + h_b
            #        y = h_a * v_a * y + h_a * v_b + h_b
            #        y * (1 - h_a * v_a) = h_a * v_b + h_b
            
            denom = 1 - h_a * v_a
            if abs(denom) < 1e-6:
                # 平行或接近平行，使用简单方法
                x = v_b
                y = h_b
            else:
                y = (h_a * v_b + h_b) / denom
                x = v_a * y + v_b
            
            return (int(x), int(y))
        
        # 计算四个角点
        top_left = fitted_line_intersection((top_a, top_b), (left_a, left_b))
        top_right = fitted_line_intersection((top_a, top_b), (right_a, right_b))
        bottom_right = fitted_line_intersection((bottom_a, bottom_b), (right_a, right_b))
        bottom_left = fitted_line_intersection((bottom_a, bottom_b), (left_a, left_b))
        
        corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        
        print(f"[{_lt('角点检测')}] 成功检测到四个角点:")
        print(f"  左上: {top_left}")
        print(f"  右上: {top_right}")
        print(f"  右下: {bottom_right}")
        print(f"  左下: {bottom_left}")
        
        return corners
    
    def apply_perspective_correction(self, image, corners):
        """
        应用透视校正
        
        Args:
            image: 输入图像
            corners: 四个角点 [top_left, top_right, bottom_right, bottom_left]
        
        Returns:
            tuple: (校正后的图像, 透视变换矩阵)
        """
        h, w = image.shape[:2]
        
        # 计算目标尺寸（保持原图尺寸）
        dst_w = w
        dst_h = h
        
        # 定义目标矩形的四个角点
        dst_corners = np.array([
            [0, 0],              # 左上
            [dst_w - 1, 0],      # 右上
            [dst_w - 1, dst_h - 1],  # 右下
            [0, dst_h - 1]       # 左下
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        perspective_matrix = cv2.getPerspectiveTransform(corners, dst_corners)
        
        # 应用透视变换
        corrected = cv2.warpPerspective(image, perspective_matrix, (dst_w, dst_h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(255, 255, 255))
        
        print(f"[{_lt('透视校正')}] 原始尺寸: {w}x{h}")
        print(f"[{_lt('透视校正')}] 校正后尺寸: {dst_w}x{dst_h}")
        
        return corrected, perspective_matrix
    
    def crop_rotated_rectangle(self, image, angle, original_w, original_h):
        """
        根据旋转角度计算并裁剪出最大内接矩形
        
        使用几何公式计算旋转矩形的最大内接矩形
        
        Args:
            image: 旋转后的图像
            angle: 旋转角度（度）
            original_w: 原始图像宽度
            original_h: 原始图像高度
        
        Returns:
            tuple: (裁剪后的图像, 裁剪偏移量(x, y))
        """
        rot_h, rot_w = image.shape[:2]
        
        # 转换为弧度
        angle_rad = abs(angle) * np.pi / 180.0
        
        # 如果角度很小，直接返回（避免不必要的计算）
        if abs(angle) < 0.5:
            return image, (0, 0)
        
        # 计算sin和cos（使用绝对值）
        sin_a = abs(np.sin(angle_rad))
        cos_a = abs(np.cos(angle_rad))
        
        # 使用简单直接的几何公式
        # 对于旋转θ的矩形w×h，最大内接矩形的尺寸为：
        # crop_w = w * cos(θ) - h * sin(θ)
        # crop_h = h * cos(θ) - w * sin(θ)
        
        crop_w = int(original_w * cos_a - original_h * sin_a)
        crop_h = int(original_h * cos_a - original_w * sin_a)
        
        print(f"[{_lt('裁剪计算')}] sin={sin_a:.4f}, cos={cos_a:.4f}")
        print(f"[{_lt('裁剪计算')}] 公式计算: crop_w={crop_w}, crop_h={crop_h}")
        
        # 确保尺寸为正数且不超过旋转后的图像
        if crop_w <= 0 or crop_h <= 0:
            print(f"[{_lt('裁剪计算')}] 警告：计算结果为负，使用保守裁剪")
            # 使用保守的估计
            crop_w = int(min(original_w, original_h) * cos_a)
            crop_h = int(min(original_w, original_h) * cos_a)
        
        # 限制在合理范围内
        crop_w = min(crop_w, rot_w - 20)
        crop_h = min(crop_h, rot_h - 20)
        
        # 确保至少保留原尺寸的70%
        crop_w = max(crop_w, int(original_w * 0.7))
        crop_h = max(crop_h, int(original_h * 0.7))
        
        # 从中心裁剪
        center_x = rot_w // 2
        center_y = rot_h // 2
        
        x1 = center_x - crop_w // 2
        y1 = center_y - crop_h // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(rot_w, x2)
        y2 = min(rot_h, y2)
        
        # 执行裁剪
        cropped = image[y1:y2, x1:x2]
        
        print(f"[{_lt('裁剪计算')}] 原始: {original_w}x{original_h}, 角度: {angle:.2f}°")
        print(f"[{_lt('裁剪计算')}] 旋转后: {rot_w}x{rot_h}")
        print(f"[{_lt('裁剪计算')}] 内接矩形: {crop_w}x{crop_h}")
        print(f"[{_lt('裁剪计算')}] 裁剪区域: ({x1},{y1}) -> ({x2},{y2})")
        
        # 返回裁剪后的图像和裁剪偏移量
        return cropped, (x1, y1)
    
    
    def process(self, image, 
                enable_curve_erase=True,
                enable_dashed_line_connection=True,
                enable_perspective_correction=True,
                enable_rotation_correction=True,
                x_tick_labels=None,
                y_tick_labels=None,
                # 倾斜校正参数（第一次检测）
                rotation_hough_threshold=65,
                rotation_min_line_length=60,
                rotation_max_line_gap=10,
                rotation_angle_threshold=5,
                # 网格检测参数（第二次检测）
                grid_hough_threshold=65,
                grid_min_line_length=45,
                grid_max_line_gap=10,
                grid_angle_threshold=5,
                # 合并参数
                merge_distance_threshold=20,
                # 虚线连接参数
                dashed_gap_threshold=10,
                dashed_collinear_threshold=5,
                # 曲线检测参数
                curve_saturation_threshold=90,
                curve_value_threshold=60,
                curve_feather_size=3):
        """
        完整的预处理流程
        
        Args:
            image: 输入图像
            enable_curve_erase: 是否先擦除彩色曲线
            enable_perspective_correction: 是否进行透视校正
            enable_rotation_correction: 是否进行倾斜校正
            
            倾斜校正参数（用于计算旋转角度）:
            rotation_hough_threshold: Hough变换阈值
            rotation_min_line_length: 最小线段长度
            rotation_max_line_gap: 最大线段间隙
            rotation_angle_threshold: 角度分类阈值
            
            网格检测参数（用于精确提取网格）:
            grid_hough_threshold: Hough变换阈值
            grid_min_line_length: 最小线段长度
            grid_max_line_gap: 最大线段间隙
            grid_angle_threshold: 角度分类阈值
            
            合并参数:
            merge_distance_threshold: 相邻线段合并距离阈值
            
            曲线检测参数:
            curve_saturation_threshold: 颜色饱和度阈值
            curve_value_threshold: 颜色亮度阈值
            curve_feather_size: 羽化大小（膨胀像素数）
        """
        self.intermediate_results = {}
        result = image.copy()
        
        # 保存原图
        self.intermediate_results['original'] = image.copy()
        
        # 0. 曲线检测（可选）- 生成mask用于后续擦除边缘
        curve_mask = None
        text_mask = self.build_text_mask(
            result.shape, x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels, padding=10
        )
        if enable_curve_erase:
            print("\n====== 步骤0: 曲线检测 ======")
            self.color_saturation_threshold = curve_saturation_threshold
            self.color_value_threshold = curve_value_threshold
            curve_mask = self.detect_curve_mask(result, feather_size=curve_feather_size)
        
        # 1. 第一次网格检测：提取断续的线段（使用曲线mask擦除边缘）
        print("\n====== 步骤1: 第一次网格检测 ======")
        lines = self.detect_grid_lines(result, 
                                       threshold=grid_hough_threshold,
                                       min_line_length=grid_min_line_length,
                                       max_line_gap=grid_max_line_gap,
                                       curve_mask=curve_mask,
                                       text_mask=text_mask)
        
        horizontal_lines, vertical_lines = self.classify_lines(lines, grid_angle_threshold)
        print(f"  检测到: 水平线{len(horizontal_lines)}条, 垂直线{len(vertical_lines)}条")
        
        # 1.5. 虚线连接：将断续的线段连接成实线网格
        if enable_dashed_line_connection and (horizontal_lines or vertical_lines):
            print("\n====== 步骤1.5: 虚线连接 ======")
            
            # 保存连接前的线条（用于可视化对比）
            original_h_lines = horizontal_lines.copy() if horizontal_lines else []
            original_v_lines = vertical_lines.copy() if vertical_lines else []
            
            if horizontal_lines:
                horizontal_lines = self.connect_dashed_lines(
                    horizontal_lines, 
                    is_horizontal=True,
                    collinear_threshold=dashed_collinear_threshold,
                    gap_threshold=dashed_gap_threshold
                )
            if vertical_lines:
                vertical_lines = self.connect_dashed_lines(
                    vertical_lines, 
                    is_horizontal=False,
                    collinear_threshold=dashed_collinear_threshold,
                    gap_threshold=dashed_gap_threshold
                )
            
            # 可视化虚线连接结果
            dashed_connection_image = result.copy()
            
            # 绘制连接前的线条（灰色，细线）
            for line in original_h_lines:
                x1, y1, x2, y2 = [int(coord) for coord in line]
                cv2.line(dashed_connection_image, (x1, y1), (x2, y2), (128, 128, 128), 1)
            for line in original_v_lines:
                x1, y1, x2, y2 = [int(coord) for coord in line]
                cv2.line(dashed_connection_image, (x1, y1), (x2, y2), (128, 128, 128), 1)
            
            # 绘制连接后的线条（彩色，粗线）
            for line in horizontal_lines:
                x1, y1, x2, y2 = [int(coord) for coord in line]
                cv2.line(dashed_connection_image, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 红色
            for line in vertical_lines:
                x1, y1, x2, y2 = [int(coord) for coord in line]
                cv2.line(dashed_connection_image, (x1, y1), (x2, y2), (255, 0, 0), 3)  # 蓝色
            
            self.intermediate_results['dashed_connection'] = dashed_connection_image
            print(f"  可视化: 灰色细线=原始线段, 红色粗线=连接后的水平线, 蓝色粗线=连接后的垂直线")
        
        # 2. 透视校正：基于实线网格校正原图
        perspective_matrix = None
        grid_corners = None
        
        if enable_perspective_correction and horizontal_lines and vertical_lines:
            print("\n====== 步骤2: 透视校正 ======")
            
            # 检测网格四个角点（使用连接后的线条）
            grid_corners = self.detect_grid_corners(horizontal_lines, vertical_lines, result.shape)
            
            if grid_corners is not None:
                # 应用透视校正
                result, perspective_matrix = self.apply_perspective_correction(result, grid_corners)
                self.intermediate_results['after_perspective'] = result.copy()
                self.intermediate_results['perspective_matrix'] = perspective_matrix
                self.intermediate_results['grid_corners'] = grid_corners
                
                # 同步透视变换曲线mask
                if curve_mask is not None:
                    print(f"  同步透视变换曲线mask...")
                    h, w = result.shape[:2]
                    curve_mask = cv2.warpPerspective(curve_mask, perspective_matrix, (w, h))
                if text_mask is not None:
                    print(f"  同步透视变换文字mask...")
                    h, w = result.shape[:2]
                    text_mask = cv2.warpPerspective(text_mask, perspective_matrix, (w, h))
            else:
                print(f"[{_lt('透视校正')}] 跳过：角点检测失败")

        # 2.5 透视后的网格重检测：后续倾斜校正必须基于当前图像，而不是
        # 透视前的旧线段；否则角度会明显失真。
        angle_h_lines = horizontal_lines
        angle_v_lines = vertical_lines
        if perspective_matrix is not None:
            print("\n====== 步骤2.5: 透视后角度重检测 ======")
            angle_lines = self.detect_grid_lines(
                result,
                threshold=rotation_hough_threshold,
                min_line_length=rotation_min_line_length,
                max_line_gap=rotation_max_line_gap,
                curve_mask=curve_mask,
                text_mask=text_mask,
            )
            angle_h_lines, angle_v_lines = self.classify_lines(angle_lines, rotation_angle_threshold)
            print(f"  角度重检测到: 水平线{len(angle_h_lines)}条, 垂直线{len(angle_v_lines)}条")
            if enable_dashed_line_connection and (angle_h_lines or angle_v_lines):
                if angle_h_lines:
                    angle_h_lines = self.connect_dashed_lines(
                        angle_h_lines,
                        is_horizontal=True,
                        collinear_threshold=dashed_collinear_threshold,
                        gap_threshold=dashed_gap_threshold
                    )
                if angle_v_lines:
                    angle_v_lines = self.connect_dashed_lines(
                        angle_v_lines,
                        is_horizontal=False,
                        collinear_threshold=dashed_collinear_threshold,
                        gap_threshold=dashed_gap_threshold
                    )
                print(f"  角度重检测连接后: 水平线{len(angle_h_lines)}条, 垂直线{len(angle_v_lines)}条")
        
        # 3. 倾斜校正：基于实线网格校正原图
        rotation_angle = 0
        rotation_matrix = None
        rotation_center = None
        original_size = None
        
        if enable_rotation_correction and (angle_h_lines or angle_v_lines):
            print("\n====== 步骤3: 倾斜校正 ======")
            rotation_angle = self.calculate_rotation_angle(angle_h_lines, angle_v_lines)
            
            # 保存原始尺寸和旋转中心
            h, w = result.shape[:2]
            original_size = (w, h)
            rotation_center = (w // 2, h // 2)
            
            # 执行旋转并获取完整的旋转矩阵（包含平移到新中心的调整）
            result, crop_offset, rotation_matrix = self.rotate_image(result, rotation_angle)
            
            self.intermediate_results['after_rotation'] = result.copy()
            self.intermediate_results['rotation_angle'] = rotation_angle
            self.intermediate_results['rotation_matrix'] = rotation_matrix  # 保存完整矩阵
            self.intermediate_results['rotation_center'] = rotation_center
            self.intermediate_results['original_size'] = original_size
            self.intermediate_results['crop_offset'] = crop_offset  # 保存裁剪偏移量
            
            # 同步旋转曲线mask
            if curve_mask is not None:
                print(f"  同步旋转曲线mask...")
                curve_mask, _, _ = self.rotate_image(curve_mask, rotation_angle)
            if text_mask is not None:
                print(f"  同步旋转文字mask...")
                text_mask, _, _ = self.rotate_image(text_mask, rotation_angle)
        
        # 4. 第二次网格检测：校正后重新检测网格线
        print("\n====== 步骤4: 第二次网格检测 ======")
        lines = self.detect_grid_lines(result, 
                                       threshold=grid_hough_threshold,
                                       min_line_length=grid_min_line_length,
                                       max_line_gap=grid_max_line_gap,
                                       curve_mask=curve_mask,
                                       text_mask=text_mask)
        
        horizontal_lines, vertical_lines = self.classify_lines(lines, grid_angle_threshold)
        print(f"  检测到: 水平线{len(horizontal_lines)}条, 垂直线{len(vertical_lines)}条")

        # 最终检测后再做一次虚线连接，避免校正后的灰色虚线在最终映射前仍是碎片
        if enable_dashed_line_connection and (horizontal_lines or vertical_lines):
            print("\n====== 步骤4.5: 最终虚线连接 ======")
            if horizontal_lines:
                horizontal_lines = self.connect_dashed_lines(
                    horizontal_lines,
                    is_horizontal=True,
                    collinear_threshold=dashed_collinear_threshold,
                    gap_threshold=dashed_gap_threshold
                )
            if vertical_lines:
                vertical_lines = self.connect_dashed_lines(
                    vertical_lines,
                    is_horizontal=False,
                    collinear_threshold=dashed_collinear_threshold,
                    gap_threshold=dashed_gap_threshold
                )
            print(f"  最终连接后: 水平线{len(horizontal_lines)}条, 垂直线{len(vertical_lines)}条")
        
        # 5. 延伸并合并网格线：得到最终的网格
        print("\n====== 步骤5: 延伸并合并网格线 ======")
        if horizontal_lines:
            merged_h_lines = self.extend_and_merge_lines(
                horizontal_lines, 
                is_horizontal=True,
                distance_threshold=merge_distance_threshold,
                image_shape=result.shape
            )
        else:
            merged_h_lines = []
        
        if vertical_lines:
            merged_v_lines = self.extend_and_merge_lines(
                vertical_lines, 
                is_horizontal=False,
                distance_threshold=merge_distance_threshold,
                image_shape=result.shape
            )
        else:
            merged_v_lines = []
        
        # 可视化合并后的网格线
        merged_grid_image = result.copy()
        for line in merged_h_lines:
            x1, y1, x2, y2 = [int(coord) for coord in line]
            cv2.line(merged_grid_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        for line in merged_v_lines:
            x1, y1, x2, y2 = [int(coord) for coord in line]
            cv2.line(merged_grid_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        print(f"  合并后: 水平线{len(merged_h_lines)}条, 垂直线{len(merged_v_lines)}条")
        
        # 可视化透视校正检测到的角点（如果有）
        if grid_corners is not None and perspective_matrix is not None:
            # 需要将原始检测的角点变换到当前图像坐标系
            # 如果进行了透视校正，角点需要经过透视变换
            # 如果还进行了旋转，角点需要再经过旋转变换
            
            # 创建一个变换后的角点副本用于显示
            display_corners = grid_corners.copy()
            
            # 应用透视变换
            h_temp = np.ones((4, 3))
            h_temp[:, :2] = grid_corners
            transformed = perspective_matrix @ h_temp.T
            transformed = transformed[:2, :] / transformed[2, :]
            display_corners = transformed.T
            
            # 如果有旋转，应用旋转变换
            if rotation_matrix is not None:
                h_temp = np.ones((4, 3))
                h_temp[:, :2] = display_corners
                transformed = rotation_matrix @ h_temp.T
                display_corners = transformed.T
            
            # 绘制角点（红色大圆点）
            for i, corner in enumerate(display_corners):
                x, y = int(corner[0]), int(corner[1])
                # 确保坐标在图像范围内
                if 0 <= x < merged_grid_image.shape[1] and 0 <= y < merged_grid_image.shape[0]:
                    cv2.circle(merged_grid_image, (x, y), 10, (0, 0, 255), -1)  # 红色实心圆
                    cv2.circle(merged_grid_image, (x, y), 12, (255, 255, 255), 2)  # 白色边框
        
        self.intermediate_results['merged_grid'] = merged_grid_image
        self.intermediate_results['merged_h_lines'] = merged_h_lines
        self.intermediate_results['merged_v_lines'] = merged_v_lines
        
        # 计算网格间距统计信息
        if len(merged_h_lines) > 1:
            h_coords = np.array([line[1] for line in merged_h_lines])
            h_gaps = np.diff(h_coords)
            self.intermediate_results['h_gap_median'] = np.median(h_gaps)
            self.intermediate_results['h_gap_min'] = np.min(h_gaps)
            self.intermediate_results['h_gap_max'] = np.max(h_gaps)
        
        if len(merged_v_lines) > 1:
            v_coords = np.array([line[0] for line in merged_v_lines])
            v_gaps = np.diff(v_coords)
            self.intermediate_results['v_gap_median'] = np.median(v_gaps)
            self.intermediate_results['v_gap_min'] = np.min(v_gaps)
            self.intermediate_results['v_gap_max'] = np.max(v_gaps)
        
        # 最终结果就是旋转校正后的图
        self.intermediate_results['final'] = result.copy()
        
        return result, self.intermediate_results
