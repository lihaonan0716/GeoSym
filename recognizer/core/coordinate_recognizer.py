import cv2
import numpy as np
import base64
from typing import Any, List, Dict, Tuple, Optional
from ..mathpix.mathpix_image_ocr import MathpixImageOCRTool
import json
from collections import defaultdict
from .drawing_utils import (
    calculate_adaptive_font_scale,
    calculate_label_position_and_bbox,
    draw_text_with_background
)

import sys as _sys
from pathlib import Path as _Path
_exp_root = _Path(__file__).resolve().parent.parent.parent
if str(_exp_root) not in _sys.path:
    _sys.path.insert(0, str(_exp_root))

class CoordinateRecognizer:
    """坐标系识别器 - 第二阶段"""
    
    def __init__(self):
        self.ocr_tool = MathpixImageOCRTool()
        self.axis_info = {}
        self.coordinate_mapping = {}
        self._last_reasoning_debug = {}
        # 颜色提取参数
        self.color_saturation_threshold = 90
        self.color_value_threshold = 60
        self.morph_connect_enabled = True
        self.morph_kernel_size = 7
        self.morph_iterations = 2
        # 整数近似阈值
        self.integer_approximation_threshold = 0.1
    
    def approximate_to_integer(self, value: float, threshold: float = None) -> Tuple[float, bool]:
        """
        判断是否应该将值近似为整数
        
        Args:
            value: 原始值
            threshold: 阈值，如果值距离最近的整数小于等于此阈值，则近似为整数
        
        Returns:
            (近似后的值, 是否为整数)
        """
        if threshold is None:
            threshold = self.integer_approximation_threshold
        
        nearest_int = round(value)
        distance = abs(value - nearest_int)
        
        if distance <= threshold:
            return nearest_int, True
        else:
            return value, False

    def _line_y_at_x(self, line: List[float], x_ref: float) -> float:
        """Sample a line's y-position at the given x reference."""
        x1, y1, x2, y2 = [float(v) for v in line]
        if abs(x2 - x1) < 1e-6:
            return float((y1 + y2) / 2.0)
        t = (float(x_ref) - x1) / (x2 - x1)
        return float(y1 + t * (y2 - y1))

    def _line_x_at_y(self, line: List[float], y_ref: float) -> float:
        """Sample a line's x-position at the given y reference."""
        x1, y1, x2, y2 = [float(v) for v in line]
        if abs(y2 - y1) < 1e-6:
            return float((x1 + x2) / 2.0)
        t = (float(y_ref) - y1) / (y2 - y1)
        return float(x1 + t * (x2 - x1))

    def _sample_horizontal_positions(self, grid_h_lines: List, x_ref: float) -> List[Tuple[List[float], float]]:
        sampled = []
        for line in grid_h_lines:
            sampled.append((line, self._line_y_at_x(line, x_ref)))
        return sampled

    def _sample_vertical_positions(self, grid_v_lines: List, y_ref: float) -> List[Tuple[List[float], float]]:
        sampled = []
        for line in grid_v_lines:
            sampled.append((line, self._line_x_at_y(line, y_ref)))
        return sampled
        
    def call_ocr(self, image: np.ndarray) -> Dict:
        """
        调用OCR识别坐标系信息

        Args:
            image: OpenCV图像（BGR格式）

        Returns:
            OCR API返回的结果
        """
        _, buffer = cv2.imencode('.png', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        base64_src = f"data:image/png;base64,{base64_image}"
        return self.ocr_tool.ocr_image(base64_src)
    
    def extract_tick_labels(self, ocr_result: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        从OCR结果中提取X轴和Y轴的刻度标签
        
        Args:
            ocr_result: OCR OCR结果
        
        Returns:
            (x_tick_labels, y_tick_labels)
        """
        x_tick_labels = []
        y_tick_labels = []
        
        if 'line_data' not in ocr_result:
            return x_tick_labels, y_tick_labels
        
        print("\n[OCRlabeltext] textline_data...")
        x_label_count = 0
        y_label_count = 0
        
        for item in ocr_result['line_data']:
            if not isinstance(item, dict):
                continue
            
            item_type = item.get('type', '')
            
            if item_type == 'x_axis_tick_label' and 'text' in item and 'cnt' in item:
                # 提取x轴刻度标签
                x_label_count += 1
                try:
                    value = float(item['text'])
                    cnt = item['cnt']
                    if len(cnt) >= 4:
                        x_coords = [point[0] for point in cnt]
                        y_coords = [point[1] for point in cnt]
                        
                        label_info = {
                            'value': value,
                            'left': min(x_coords),
                            'top': min(y_coords),
                            'right': max(x_coords),
                            'bottom': max(y_coords),
                            'center_x': (min(x_coords) + max(x_coords)) / 2,
                            'center_y': (min(y_coords) + max(y_coords)) / 2
                        }
                        x_tick_labels.append(label_info)
                        print(f"  ✓ X-axislabel: text='{item['text']}', value={value}, center_x={label_info['center_x']:.1f}")
                except ValueError as e:
                    print(f"  ✗ X-axislabeltextFailed: text='{item['text']}', error={e}")
                    continue
            
            elif item_type == 'y_axis_tick_label' and 'text' in item and 'cnt' in item:
                # 提取y轴刻度标签
                y_label_count += 1
                try:
                    value = float(item['text'])
                    cnt = item['cnt']
                    if len(cnt) >= 4:
                        x_coords = [point[0] for point in cnt]
                        y_coords = [point[1] for point in cnt]
                        
                        label_info = {
                            'value': value,
                            'left': min(x_coords),
                            'top': min(y_coords),
                            'right': max(x_coords),
                            'bottom': max(y_coords),
                            'center_x': (min(x_coords) + max(x_coords)) / 2,
                            'center_y': (min(y_coords) + max(y_coords)) / 2
                        }
                        y_tick_labels.append(label_info)
                        print(f"  ✓ Y-axis labels: text='{item['text']}', value={value}, center_y={label_info['center_y']:.1f}")
                except ValueError as e:
                    print(f"  ✗ Y-axis labelstextFailed: text='{item['text']}', error={e}")
                    continue
        
        print(f"\n[OCRlabeltext] text:")
        print(f"  line_datatextX-axis label counttext: {x_label_count}")
        print(f"  line_datatextY-axis label counttext: {y_label_count}")
        print(f"  textX-axislabel: {len(x_tick_labels)}text")
        print(f"  textY-axis labels: {len(y_tick_labels)}text")
        
        # 按位置排序
        x_tick_labels.sort(key=lambda l: l['center_x'])
        y_tick_labels.sort(key=lambda l: l['center_y'])
        
        print(f"  X-axislabeltext (left to right): {[l['value'] for l in x_tick_labels]}")
        print(f"  Y-axis labelstext (top to bottom): {[l['value'] for l in y_tick_labels]}")
        
        return x_tick_labels, y_tick_labels
    
    def _filter_ocr_labels(self, labels: List[Dict], axis: str) -> List[Dict]:
        """
        过滤OCR识别错误的刻度标签
        
        策略：
        0. 位置聚类：删除不在主流位置线上的标签
           - X轴：删除center_y偏离主流的标签（应该在同一水平线）
           - Y轴：删除center_x偏离主流的标签（应该在同一垂直线）
        1. 单调性检查：按位置排序后，值应该是单调的（X轴递增，Y轴递减），允许重复值
        2. 重复值处理：基于像素位置的均匀性选择保留哪个重复值
           - 计算所有非重复标签的平均像素间距
           - 选择像素间距最接近平均间距的候选（位置最合理）
        3. 间隔过滤：计算P75分位的单位间隔，删除异常间隔的两端
        
        Args:
            labels: 刻度标签列表
            axis: 'x' 或 'y'
        
        Returns:
            过滤后的标签列表
        """
        if len(labels) < 2:
            return labels
        
        print(f"\n[OCRlabelfilter-{axis.upper()}text] Starting filtering...")
        print(f"  Original label count: {len(labels)}")
        
        # 0. 位置聚类过滤：删除不在主流位置线上的标签
        if axis == 'x':
            # X轴标签应该在同一水平线上（center_y接近）
            position_key = 'center_y'
            position_name = 'Y坐标'
        else:
            # Y轴标签应该在同一垂直线上（center_x接近）
            position_key = 'center_x'
            position_name = 'X坐标'
        
        # 计算位置的中位数
        positions = [l[position_key] for l in labels]
        position_median = np.median(positions)
        
        # 计算标签尺寸（用于设置阈值）
        if axis == 'x':
            sizes = [l['bottom'] - l['top'] for l in labels]  # 高度
        else:
            sizes = [l['right'] - l['left'] for l in labels]  # 宽度
        
        size_median = np.median(sizes)
        position_threshold = 1.5 * size_median  # 允许偏离1.5倍标签尺寸
        
        print(f"  Position-cluster filtering ({position_name}): ")
        print(f"    median: {position_median:.1f}, threshold: ±{position_threshold:.1f}")
        
        # 过滤偏离主流位置的标签
        clustered_labels = []
        for l in labels:
            pos = l[position_key]
            deviation = abs(pos - position_median)
            if deviation <= position_threshold:
                clustered_labels.append(l)
                print(f"    ✓ Keep: text={l['value']}, {position_name}={pos:.1f}, deviation={deviation:.1f}")
            else:
                print(f"    ✗ text: text={l['value']}, {position_name}={pos:.1f}, deviation={deviation:.1f} (texthreshold)")
        
        print(f"  positionclusterafter: {len(clustered_labels)}textlabel")
        
        if len(clustered_labels) < 2:
            print(f"  ⚠️ filterafterlabeltext2text, textClustering result")
            return clustered_labels
        
        # 1. 按位置排序
        if axis == 'x':
            # X轴：从左到右排序
            sorted_labels = sorted(clustered_labels, key=lambda l: l['center_x'])
            print(f"  Sort direction: left to right (textcenter_x)")
        else:
            # Y轴：从上到下排序
            sorted_labels = sorted(clustered_labels, key=lambda l: l['center_y'])
            print(f"  Sort direction: top to bottom (textcenter_y)")
        
        print(f"  textaftertext: {[l['value'] for l in sorted_labels]}")
        
        # 2. 少数服从多数：删除破坏主流趋势的标签
        if len(sorted_labels) < 3:
            print(f"  labelcounttext3text, textMonotonicity check")
        else:
            # 2.1 计算相邻差值，判断主流趋势
            diffs = []
            for i in range(len(sorted_labels) - 1):
                diff = sorted_labels[i + 1]['value'] - sorted_labels[i]['value']
                diffs.append(diff)
            
            print(f"  Adjacent differences: {[f'{d:.1f}' for d in diffs]}")
            
            # 统计正负趋势
            positive_count = sum(1 for d in diffs if d > 0)
            negative_count = sum(1 for d in diffs if d < 0)
            
            # 判断主流趋势
            if axis == 'x':
                # X轴期望递增（允许相等，即重复值）
                expected_trend = "递增"
                is_good_trend = lambda d: d >= 0  # 允许0（重复值）
                is_bad_trend = lambda d: d < 0    # 只有负值才是坏趋势
            else:
                # Y轴期望递减（允许相等，即重复值）
                expected_trend = "递减"
                is_good_trend = lambda d: d <= 0  # 允许0（重复值）
                is_bad_trend = lambda d: d > 0    # 只有正值才是坏趋势
            
            print(f"  Expected trend: {expected_trend} (textduplicate value)")
            print(f"  positive difference count: {positive_count}, negative difference count: {negative_count}")
            
            # 2.2 反复删除不符合主流趋势的点
            max_iterations = 10
            iteration = 0
            
            while iteration < max_iterations and len(sorted_labels) >= 3:
                iteration += 1
                removed_in_this_iteration = False
                
                # 重新计算差值
                diffs = []
                for i in range(len(sorted_labels) - 1):
                    diff = sorted_labels[i + 1]['value'] - sorted_labels[i]['value']
                    diffs.append(diff)
                
                # 找到第一个破坏趋势的差值（不包括重复值）
                for i, diff in enumerate(diffs):
                    if is_bad_trend(diff):
                        # 差值破坏趋势，需要删除 sorted_labels[i] 或 sorted_labels[i+1]
                        # 策略：看哪个点更离群
                        
                        # 计算删除每个点后的"改善度"
                        # 方案1：删除 i
                        score_remove_i = 0
                        if i > 0:
                            # 看 i-1 到 i+1 的差值是否符合趋势
                            new_diff = sorted_labels[i + 1]['value'] - sorted_labels[i - 1]['value']
                            if is_good_trend(new_diff):
                                score_remove_i += 1
                        if i < len(diffs) - 1:
                            # 看删除i后，i+1到i+2的关系
                            score_remove_i += 0.5  # 不影响后面
                        
                        # 方案2：删除 i+1
                        score_remove_i_plus_1 = 0
                        if i < len(diffs) - 1:
                            # 看 i 到 i+2 的差值是否符合趋势
                            new_diff = sorted_labels[i + 2]['value'] - sorted_labels[i]['value']
                            if is_good_trend(new_diff):
                                score_remove_i_plus_1 += 1
                        if i > 0:
                            # 看删除i+1后，i-1到i的关系
                            score_remove_i_plus_1 += 0.5  # 不影响前面
                        
                        # 选择改善度更高的方案
                        if score_remove_i > score_remove_i_plus_1:
                            removed_label = sorted_labels.pop(i)
                            print(f"  iteration{iteration}: Removed label {removed_label['value']} (position{i}, breaks{expected_trend}trend)")
                        else:
                            removed_label = sorted_labels.pop(i + 1)
                            print(f"  iteration{iteration}: Removed label {removed_label['value']} (position{i+1}, breaks{expected_trend}trend)")
                        
                        removed_in_this_iteration = True
                        break  # 删除一个后重新开始
                
                if not removed_in_this_iteration:
                    print(f"  Monotonicity checkComplete (iteration{iteration}text)")
                    break
            
            if len(sorted_labels) < 2:
                print(f"  ⚠️ filterafterlabeltext2text, textoriginallabel")
                return labels
            
            # 检查是否至少有2个不同的值
            unique_values_after_monotone = set(l['value'] for l in sorted_labels)
            if len(unique_values_after_monotone) < 2:
                print(f"  ❌ textfilteraftertext{len(unique_values_after_monotone)}text, OCRtextFailed")
                print(f"  Currently retained values: {list(unique_values_after_monotone)}")
                return []  # 返回空列表表示失败
            
            print(f"  textfilterafter: {len(sorted_labels)}textlabel, {len(unique_values_after_monotone)}text")
            print(f"  filteraftertext: {[l['value'] for l in sorted_labels]}")
        
        # 2.5. 处理重复值（基于间隔一致性）
        if len(sorted_labels) >= 3:
            print(f"\n  textduplicate value...")
            deduplicated = []
            i = 0
            removed_duplicates = 0
            
            while i < len(sorted_labels):
                current_label = sorted_labels[i]
                
                # 查找连续的重复值
                duplicates = [current_label]
                j = i + 1
                while j < len(sorted_labels) and sorted_labels[j]['value'] == current_label['value']:
                    duplicates.append(sorted_labels[j])
                    j += 1
                
                if len(duplicates) > 1:
                    # 有重复值，需要选择保留哪个
                    print(f"    Foundduplicate value: {current_label['value']} ({len(duplicates)}text)")
                    
                    # 获取位置坐标key
                    if axis == 'x':
                        pos_key = 'center_x'
                    else:
                        pos_key = 'center_y'
                    
                    # 计算所有非重复标签之间的像素间距（用于参考）
                    non_duplicate_labels = []
                    for k, label in enumerate(sorted_labels):
                        # 跳过当前这组重复值
                        if k < i or k >= j:
                            non_duplicate_labels.append(label)
                    
                    # 计算平均像素间距
                    avg_pixel_spacing = None
                    if len(non_duplicate_labels) >= 2:
                        pixel_spacings = []
                        for k in range(len(non_duplicate_labels) - 1):
                            spacing = abs(non_duplicate_labels[k + 1][pos_key] - non_duplicate_labels[k][pos_key])
                            pixel_spacings.append(spacing)
                        avg_pixel_spacing = np.mean(pixel_spacings)
                        print(f"      Average pixel spacing (reference): {avg_pixel_spacing:.1f}")
                    
                    # 对每个候选计算得分（基于像素位置的均匀性）
                    best_candidate = None
                    best_score = float('-inf')
                    
                    for candidate in duplicates:
                        candidate_idx = sorted_labels.index(candidate)
                        candidate_pos = candidate[pos_key]
                        
                        # 计算该候选和前后非重复标签的像素间距
                        pixel_intervals = []
                        
                        # 前一个非重复标签的像素间距
                        if candidate_idx > 0:
                            # 找前面第一个不是重复值的标签
                            prev_idx = candidate_idx - 1
                            while prev_idx >= i:  # 跳过重复值组
                                prev_idx -= 1
                            if prev_idx >= 0:
                                prev_pos = sorted_labels[prev_idx][pos_key]
                                pixel_intervals.append(abs(candidate_pos - prev_pos))
                        
                        # 后一个非重复标签的像素间距
                        if j < len(sorted_labels):
                            next_pos = sorted_labels[j][pos_key]
                            pixel_intervals.append(abs(next_pos - candidate_pos))
                        
                        # 计算得分：像素间距越接近平均间距，得分越高
                        if len(pixel_intervals) > 0 and avg_pixel_spacing:
                            # 计算偏离度的负值（偏离越小，得分越高）
                            deviations = [abs(interval - avg_pixel_spacing) for interval in pixel_intervals]
                            score = -np.mean(deviations)
                        elif len(pixel_intervals) > 0:
                            # 没有参考间距，使用间距均匀性
                            if len(pixel_intervals) == 2:
                                score = -abs(pixel_intervals[0] - pixel_intervals[1])
                            else:
                                score = pixel_intervals[0]
                        else:
                            score = 0
                        
                        print(f"      candidate text={candidate['value']}, position={candidate_pos:.1f}, pixel intervals={[f'{x:.1f}' for x in pixel_intervals]}, score={score:.2f}")
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = candidate
                    
                    # 保留最佳候选
                    deduplicated.append(best_candidate)
                    pos = best_candidate[pos_key]
                    print(f"    ✓ Keep: text={best_candidate['value']}, position={pos:.1f}")
                    removed_duplicates += len(duplicates) - 1
                else:
                    # 没有重复，直接添加
                    deduplicated.append(current_label)
                
                i = j  # 跳到下一组
            
            if removed_duplicates > 0:
                print(f"  duplicate valuetext: removed{removed_duplicates}duplicate labels")
                sorted_labels = deduplicated
                print(f"  After deduplicationtext: {[l['value'] for l in sorted_labels]}")
        
        # 3. 计算相邻标签的间隔
        intervals = []
        for i in range(len(sorted_labels) - 1):
            val_diff = abs(sorted_labels[i + 1]['value'] - sorted_labels[i]['value'])
            intervals.append(val_diff)
        
        print(f"  text: {[f'{v:.1f}' for v in intervals]}")
        
        # 4. 计算P75分位的单位间隔
        p75_interval = np.percentile(intervals, 75)
        print(f"  P75text: {p75_interval:.2f}")
        
        # 5. 成对间隔检查：识别异常间隔，成对删除
        min_valid_interval = p75_interval * 0.5
        max_valid_interval = p75_interval * 1.5
        
        print(f"  text: [{min_valid_interval:.2f}, {max_valid_interval:.2f}]")
        print(f"  text (Exceptiontext)...")
        
        # 第一步：识别所有异常间隔
        abnormal_intervals = []  # [(i, i+1, interval), ...]
        for i in range(len(sorted_labels) - 1):
            interval = abs(sorted_labels[i + 1]['value'] - sorted_labels[i]['value'])
            if interval < min_valid_interval or interval > max_valid_interval:
                abnormal_intervals.append((i, i + 1, interval))
                print(f"  Exceptiontext: [{sorted_labels[i]['value']}] ↔ [{sorted_labels[i+1]['value']}] = {interval:.2f}")
        
        # 第二步：删除异常间隔的两端标签
        keep_flags = [True] * len(sorted_labels)
        
        for i, j, interval in abnormal_intervals:
            # 异常间隔表示中间缺失了刻度，两端标签都可能有误，全部删除
            if keep_flags[i]:
                keep_flags[i] = False
                print(f"  ✗ Removed label {sorted_labels[i]['value']} (position{i}, Exceptiontext{interval:.2f}text)")
            if keep_flags[j]:
                keep_flags[j] = False
                print(f"  ✗ Removed label {sorted_labels[j]['value']} (position{j}, Exceptiontext{interval:.2f}text)")
        
        # 输出保留的标签
        for i, label in enumerate(sorted_labels):
            if keep_flags[i]:
                left_interval = abs(sorted_labels[i]['value'] - sorted_labels[i-1]['value']) if i > 0 and keep_flags[i-1] else None
                right_interval = abs(sorted_labels[i+1]['value'] - sorted_labels[i]['value']) if i < len(sorted_labels)-1 and keep_flags[i+1] else None
                left_interval_debug = left_interval if left_interval else 'N/A'
                right_interval_debug = right_interval if right_interval else 'N/A'
                print(f"  ✓ Keeplabel {sorted_labels[i]['value']} (position{i}, text={left_interval_debug}, text={right_interval_debug})")
        
        # 应用过滤
        filtered_labels = [label for i, label in enumerate(sorted_labels) if keep_flags[i]]
        
        # 最后检查：必须至少有2个不同的值
        unique_values = set(l['value'] for l in filtered_labels)
        if len(unique_values) < 2:
            print(f"\n[OCRlabelfilter-{axis.upper()}text] ❌ filteraftertext{len(unique_values)}text, OCRtextFailed")
            print(f"  Currently retained values: {list(unique_values)}")
            print(f"  textrequired2textcoordinatetext")
            return []  # 返回空列表表示失败
        
        print(f"\n[OCRlabelfilter-{axis.upper()}text] Complete")
        print(f"  finalKeep: {len(filtered_labels)}textlabel")
        print(f"  Distinct value count: {len(unique_values)}text")
        print(f"  Final value sequence: {[l['value'] for l in filtered_labels]}")
        
        return filtered_labels
    
    def infer_axes_position(self, x_tick_labels: List[Dict], y_tick_labels: List[Dict],
                           grid_h_lines: List, grid_v_lines: List) -> Dict:
        """
        根据刻度标签从网格线中选择X轴和Y轴
        
        Args:
            x_tick_labels: X轴刻度标签列表
            y_tick_labels: Y轴刻度标签列表
            grid_h_lines: 水平网格线
            grid_v_lines: 垂直网格线
        
        Returns:
            坐标轴信息字典
        """
        axes_info = {}
        
        # 验证刻度标签数量
        if len(x_tick_labels) < 2 or len(y_tick_labels) < 2:
            return axes_info
        
        # 1. 找X轴：基于刻度标签尺寸的偏移策略
        # 计算X轴刻度标签的平均高度
        x_label_heights = [label['bbox']['bottom'] - label['bbox']['top'] for label in x_tick_labels]
        x_label_avg_height = np.mean(x_label_heights)
        
        # 计算X轴刻度标签中心Y坐标的中位数
        x_label_centers_y = [label['center_y'] for label in x_tick_labels]
        x_label_center_y_median = np.median(x_label_centers_y)
        x_label_centers_x = [label['center_x'] for label in x_tick_labels]
        x_probe_x = float(np.median(x_label_centers_x))
        
        # 参考点：标签中心向上偏移平均高度的1/4
        # 注意：图像坐标系中y向下递增，所以"向上"是减去
        x_reference_y = x_label_center_y_median - x_label_avg_height / 4
        
        print(f"[AxisInfer] X-axislabel: textY={x_label_center_y_median:.1f}, textheight={x_label_avg_height:.1f}")
        print(f"[AxisInfer] X-axisreferencetext: y={x_reference_y:.1f} (text - height/4)")
        
        # 在水平网格线中找参考点上方最近的那条
        # "上方"意味着 line_y < x_reference_y
        min_dist = float('inf')
        x_axis_line = None
        for line, line_y in self._sample_horizontal_positions(grid_h_lines, x_probe_x):
            
            # 只考虑在参考点上方的线
            if line_y < x_reference_y:
                dist = abs(line_y - x_reference_y)
                if dist < min_dist:
                    min_dist = dist
                    x_axis_line = line
                    axes_info['x_axis_y'] = line_y
        
        if x_axis_line is not None:
            print(f"[AxisInfer] ✓ X-axisposition: y={axes_info['x_axis_y']:.1f} (textreferencetext{min_dist:.1f}px)")
        else:
            print(f"[AxisInfer] ⚠️ textFoundX-axis (textreferencetexthorizontal grid lines)")
        
        axes_info['x_axis_line'] = x_axis_line
        
        # 2. 找Y轴：基于刻度标签尺寸的偏移策略
        # 计算Y轴刻度标签的平均宽度
        y_label_widths = [label['bbox']['right'] - label['bbox']['left'] for label in y_tick_labels]
        y_label_avg_width = np.mean(y_label_widths)
        
        # 计算Y轴刻度标签中心X坐标的中位数
        y_label_centers_x = [label['center_x'] for label in y_tick_labels]
        y_label_center_x_median = np.median(y_label_centers_x)
        y_label_centers_y = [label['center_y'] for label in y_tick_labels]
        y_probe_y = float(np.median(y_label_centers_y))
        
        # 参考点：标签中心向右偏移平均宽度的1/4
        # x向右递增，所以"向右"是加上
        y_reference_x = y_label_center_x_median + y_label_avg_width / 4
        
        print(f"[AxisInfer] Y-axis labels: textX={y_label_center_x_median:.1f}, textwidth={y_label_avg_width:.1f}")
        print(f"[AxisInfer] Y-axisreferencetext: x={y_reference_x:.1f} (text + width/4)")
        
        # 在垂直网格线中找参考点右侧最近的那条
        # "右侧"意味着 line_x > y_reference_x
        min_dist = float('inf')
        y_axis_line = None
        for line, line_x in self._sample_vertical_positions(grid_v_lines, y_probe_y):
            
            # 只考虑在参考点右侧的线
            if line_x > y_reference_x:
                dist = abs(line_x - y_reference_x)
                if dist < min_dist:
                    min_dist = dist
                    y_axis_line = line
                    axes_info['y_axis_x'] = line_x
        
        if y_axis_line is not None:
            print(f"[AxisInfer] ✓ Y-axisposition: x={axes_info['y_axis_x']:.1f} (textreferencetext{min_dist:.1f}px)")
        else:
            print(f"[AxisInfer] ⚠️ textFoundY-axis (textreferencetextvertical grid lines)")
        
        axes_info['y_axis_line'] = y_axis_line
        
        # 3. 原点：X轴和Y轴的交点
        if 'y_axis_x' in axes_info and 'x_axis_y' in axes_info:
            axes_info['origin_x'] = axes_info['y_axis_x']
            axes_info['origin_y'] = axes_info['x_axis_y']
        
        return axes_info
    
    def build_coordinate_mapping(self, x_tick_labels: List[Dict], y_tick_labels: List[Dict], 
                                 axes_info: Dict, grid_h_lines: List, grid_v_lines: List) -> Dict:
        """
        建立像素坐标到真实坐标的映射关系
        
        优化逻辑（v2）：
        1. 第一阶段：直接匹配 - 网格线在标签bbox内，直接赋值标签值
        2. 第二阶段：推算未匹配的网格线
           - 使用已知标签对计算单位间隔（中位数）
           - 基于最近邻推算未知网格线的值
        3. 回退机制：如果匹配失败，使用旧算法
        
        Args:
            x_tick_labels: X轴刻度标签列表
            y_tick_labels: Y轴刻度标签列表
            axes_info: 坐标轴信息（包含原点）
            grid_h_lines: 水平网格线列表
            grid_v_lines: 垂直网格线列表
        
        Returns:
            映射信息字典
        """
        mapping = {}
        mapping['v_grid_lines'] = [list(map(float, line)) for line in grid_v_lines]
        mapping['h_grid_lines'] = [list(map(float, line)) for line in grid_h_lines]

        x_label_centers_y = [float(label['center_y']) for label in x_tick_labels] if x_tick_labels else []
        y_label_centers_x = [float(label['center_x']) for label in y_tick_labels] if y_tick_labels else []
        v_sample_y = float(np.median(x_label_centers_y)) if x_label_centers_y else float(axes_info.get('origin_y', 0.0))
        h_sample_x = float(np.median(y_label_centers_x)) if y_label_centers_x else float(axes_info.get('origin_x', 0.0))
        mapping['v_grid_reference_y'] = v_sample_y
        mapping['h_grid_reference_x'] = h_sample_x

        # 提取垂直网格线在标签参考行上的x坐标（排序）
        v_grid_sampled = sorted(
            self._sample_vertical_positions(grid_v_lines, v_sample_y),
            key=lambda item: item[1]
        )
        v_grid_x = [int(round(item[1])) for item in v_grid_sampled]
        
        # 提取水平网格线在标签参考列上的y坐标（排序）
        h_grid_sampled = sorted(
            self._sample_horizontal_positions(grid_h_lines, h_sample_x),
            key=lambda item: item[1]
        )
        h_grid_y = [int(round(item[1])) for item in h_grid_sampled]
        
        print(f"[CoordMapping] vertical grid linescount: {len(v_grid_x)}")
        print(f"[CoordMapping] horizontal grid linescount: {len(h_grid_y)}")
        
        # ===== X方向映射（优化版）=====
        if len(x_tick_labels) >= 2 and len(v_grid_x) >= 2:
            v_grid_values = self._map_grid_to_values_v2(
                v_grid_x, x_tick_labels, 
                coord_key='center_x', 
                bbox_keys=('left', 'right'),
                axis_name='X'
            )
            
            if v_grid_values is not None:
                # 成功使用优化算法
                mapping['v_grid_x_pixels'] = v_grid_x
                mapping['v_grid_x_values'] = v_grid_values
                
                # 计算插值参数 (最小二乘回归, 对 OCR 噪声更鲁棒)
                if len(v_grid_x) >= 2:
                    reg_px, reg_vl = self._remove_regression_outliers(
                        [float(p) for p in v_grid_x],
                        [float(v) for v in v_grid_values],
                    )
                    px_arr = np.array(reg_px, dtype=float)
                    vl_arr = np.array(reg_vl, dtype=float)
                    A = np.vstack([px_arr, np.ones(len(px_arr))]).T
                    lstsq_result = np.linalg.lstsq(A, vl_arr, rcond=None)
                    mapping['x_scale'] = float(lstsq_result[0][0])
                    mapping['x_offset'] = float(lstsq_result[0][1])
            else:
                # 回退到旧算法
                print(f"[CoordMapping] Xtext")
                self._map_grid_fallback(mapping, x_tick_labels, v_grid_x, 'x', 'center_x')
        
        # ===== Y方向映射（优化版）=====
        if len(y_tick_labels) >= 2 and len(h_grid_y) >= 2:
            h_grid_values = self._map_grid_to_values_v2(
                h_grid_y, y_tick_labels,
                coord_key='center_y',
                bbox_keys=('top', 'bottom'),
                axis_name='Y'
            )
            
            if h_grid_values is not None:
                # 成功使用优化算法
                mapping['h_grid_y_pixels'] = h_grid_y
                mapping['h_grid_y_values'] = h_grid_values
                
                # 计算插值参数 (最小二乘回归, 对 OCR 噪声更鲁棒)
                if len(h_grid_y) >= 2:
                    reg_px, reg_vl = self._remove_regression_outliers(
                        [float(p) for p in h_grid_y],
                        [float(v) for v in h_grid_values],
                    )
                    px_arr = np.array(reg_px, dtype=float)
                    vl_arr = np.array(reg_vl, dtype=float)
                    A = np.vstack([px_arr, np.ones(len(px_arr))]).T
                    lstsq_result = np.linalg.lstsq(A, vl_arr, rcond=None)
                    mapping['y_scale'] = float(lstsq_result[0][0])
                    mapping['y_offset'] = float(lstsq_result[0][1])
            else:
                # 回退到旧算法
                print(f"[CoordMapping] Ytext")
                self._map_grid_fallback(mapping, y_tick_labels, h_grid_y, 'y', 'center_y')
        
        # ===== 基于坐标轴的校准（重要！）=====
        # 原理：Y轴应该是x=0，X轴应该是y=0
        # 如果映射结果显示Y轴不是x=0，说明整体有偏移，需要统一调整
        
        # print(f"\n[CoordMapping] === textaxistext ===")

        # # 1. 校准X方向（基于Y轴）
        # if 'y_axis_x' in axes_info and 'v_grid_x_pixels' in mapping and 'v_grid_x_values' in mapping:
        #     y_axis_pixel_x = axes_info['y_axis_x']
            
        #     # 找到最接近Y轴的垂直网格线
        #     v_grid_x = mapping['v_grid_x_pixels']
        #     v_grid_values = mapping['v_grid_x_values']
            
        #     min_dist = float('inf')
        #     closest_grid_idx = None
        #     for i, grid_x in enumerate(v_grid_x):
        #         dist = abs(grid_x - y_axis_pixel_x)
        #         if dist < min_dist:
        #             min_dist = dist
        #             closest_grid_idx = i
            
        #     if closest_grid_idx is not None:
        #         y_axis_mapped_value = v_grid_values[closest_grid_idx]
                
        #         print(f"[AxisCalibrate] Y-axisposition: x={y_axis_pixel_x:.1f}px")
        #         print(f"[AxisCalibrate] Y-axistextgrid line: text{closest_grid_idx}, x={v_grid_x[closest_grid_idx]}px, text={y_axis_mapped_value:.2f}")
                
        #         # 如果Y轴对应的值不是0，需要统一调整
        #         if abs(y_axis_mapped_value) > 0.01:  # 容忍0.01的误差
        #             x_offset_correction = -y_axis_mapped_value
        #             print(f"[AxisCalibrate] ⚠️ Y-axistext0, requiredtext")
        #             print(f"[AxisCalibrate] Xtext: {x_offset_correction:+.2f}")
                    
        #             # 调整所有垂直网格线的值
        #             mapping['v_grid_x_values'] = [v + x_offset_correction for v in v_grid_values]
                    
        #             # 重新计算插值参数
        #             if len(v_grid_x) >= 2:
        #                 new_v_grid_values = mapping['v_grid_x_values']
        #                 mapping['x_scale'] = (new_v_grid_values[1] - new_v_grid_values[0]) / (v_grid_x[1] - v_grid_x[0])
        #                 mapping['x_offset'] = new_v_grid_values[0] - mapping['x_scale'] * v_grid_x[0]
                    
        #             print(f"[AxisCalibrate] ✓ Xtext, Y-axistext x=0")
        #             print(f"[AxisCalibrate] textafterverticaltext: [{mapping['v_grid_x_values'][0]:.2f}, {mapping['v_grid_x_values'][-1]:.2f}]")
        #         else:
        #             print(f"[AxisCalibrate] ✓ Y-axistext (x≈0), textXtext")
        # else:
        #     print(f"[AxisCalibrate] ⚠️ textY-axistextXtext, textXtext")
        
        # # 2. 校准Y方向（基于X轴）
        # if 'x_axis_y' in axes_info and 'h_grid_y_pixels' in mapping and 'h_grid_y_values' in mapping:
        #     x_axis_pixel_y = axes_info['x_axis_y']
            
        #     # 找到最接近X轴的水平网格线
        #     h_grid_y = mapping['h_grid_y_pixels']
        #     h_grid_values = mapping['h_grid_y_values']
            
        #     min_dist = float('inf')
        #     closest_grid_idx = None
        #     for i, grid_y in enumerate(h_grid_y):
        #         dist = abs(grid_y - x_axis_pixel_y)
        #         if dist < min_dist:
        #             min_dist = dist
        #             closest_grid_idx = i
            
        #     if closest_grid_idx is not None:
        #         x_axis_mapped_value = h_grid_values[closest_grid_idx]
                
        #         print(f"\n[AxisCalibrate] X-axisposition: y={x_axis_pixel_y:.1f}px")
        #         print(f"[AxisCalibrate] X-axistextgrid line: text{closest_grid_idx}, y={h_grid_y[closest_grid_idx]}px, text={x_axis_mapped_value:.2f}")
                
        #         # 如果X轴对应的值不是0，需要统一调整
        #         if abs(x_axis_mapped_value) > 0.01:  # 容忍0.01的误差
        #             y_offset_correction = -x_axis_mapped_value
        #             print(f"[AxisCalibrate] ⚠️ X-axistext0, requiredtext")
        #             print(f"[AxisCalibrate] Ytext: {y_offset_correction:+.2f}")
                    
        #             # 调整所有水平网格线的值
        #             mapping['h_grid_y_values'] = [v + y_offset_correction for v in h_grid_values]
                    
        #             # 重新计算插值参数
        #             if len(h_grid_y) >= 2:
        #                 new_h_grid_values = mapping['h_grid_y_values']
        #                 mapping['y_scale'] = (new_h_grid_values[1] - new_h_grid_values[0]) / (h_grid_y[1] - h_grid_y[0])
        #                 mapping['y_offset'] = new_h_grid_values[0] - mapping['y_scale'] * h_grid_y[0]
                    
        #             print(f"[AxisCalibrate] ✓ Ytext, X-axistext y=0")
        #             print(f"[AxisCalibrate] textafterhorizontaltext: [{mapping['h_grid_y_values'][0]:.2f}, {mapping['h_grid_y_values'][-1]:.2f}]")
        #         else:
        #             print(f"[AxisCalibrate] ✓ X-axistext (y≈0), textYtext")
        # else:
        #     print(f"[AxisCalibrate] ⚠️ textX-axistextYtext, textYtext")
        
        # print(f"[CoordMapping] === axistextComplete ===\n")
        
        # 保存刻度信息
        mapping['x_tick_labels'] = x_tick_labels
        mapping['y_tick_labels'] = y_tick_labels
        
        return mapping
    
    @staticmethod
    def _remove_regression_outliers(pixels: List[float], values: List[float]) -> Tuple[List[float], List[float]]:
        """
        移除会干扰线性回归的离群刻度匹配。

        策略: 计算相邻对的局部斜率, 移除与中位斜率方向不一致的点。
        仅在 ≥3 个点时激活, 避免对 2 点回归产生影响。
        复用自 presenter/ruler.py 的 _remove_outliers() 逻辑。
        """
        if len(pixels) < 3:
            return pixels, values

        intervals = []
        for i in range(len(values) - 1):
            px_diff = pixels[i + 1] - pixels[i]
            val_diff = values[i + 1] - values[i]
            if abs(px_diff) > 1e-6:
                intervals.append(val_diff / px_diff)

        if not intervals:
            return pixels, values

        median_slope = float(np.median(intervals))

        keep = [True] * len(values)
        for i in range(len(values) - 1):
            px_diff = pixels[i + 1] - pixels[i]
            val_diff = values[i + 1] - values[i]
            if abs(px_diff) > 1e-6:
                local_slope = val_diff / px_diff
                if (median_slope > 0 and local_slope < 0) or (median_slope < 0 and local_slope > 0):
                    keep[i + 1] = False

        new_px = [p for p, k in zip(pixels, keep) if k]
        new_vl = [v for v, k in zip(values, keep) if k]

        if len(new_px) < 2:
            return pixels, values  # 不能过度过滤

        return new_px, new_vl

    def _map_grid_to_values_v2(self, grid_positions: List[float], tick_labels: List[Dict],
                                coord_key: str, bbox_keys: Tuple[str, str], axis_name: str) -> Optional[List[float]]:
        """
        优化的网格线到值的映射算法（v2）
        
        Args:
            grid_positions: 网格线位置列表（已排序）
            tick_labels: 刻度标签列表
            coord_key: 标签中心坐标的key（'center_x' 或 'center_y'）
            bbox_keys: bbox范围的key元组（('left','right') 或 ('top','bottom')）
            axis_name: 轴名称（用于日志）
        
        Returns:
            网格线对应的值列表，失败返回None
        """
        print(f"[coordinatetext-{axis_name}] textv2")

        print(f"[coordinatetext-{axis_name}] tick_labels: {tick_labels}")
        print(f"[coordinatetext-{axis_name}] grid_positions: {grid_positions}")
        print(f"[coordinatetext-{axis_name}] bbox_keys: {bbox_keys}")
        print(f"[coordinatetext-{axis_name}] coord_key: {coord_key}")
        print(f"[coordinatetext-{axis_name}] axis_name: {axis_name}")
        
        grid_values = [None] * len(grid_positions)
        bbox_min_key, bbox_max_key = bbox_keys
        
        # ===== 阶段1：基于中心点距离的最近匹配 =====
        # 策略：每个标签匹配距离最近的网格线，使用贪心算法确保一对一匹配
        
        print(f"[coordinatetext-{axis_name}] text1: text")
        
        # 根据grid_positions计算有效边界
        grid_min = min(grid_positions)
        grid_max = max(grid_positions)
        print(f"  grid linetext: [{grid_min:.1f}, {grid_max:.1f}]")
        
        # 计算所有标签到所有网格线的距离
        potential_matches = []  # [(grid_idx, label, distance), ...]
        
        for i, grid_pos in enumerate(grid_positions):
            for label in tick_labels:
                label_center = label[coord_key]
                
                # 检查标签中心是否在网格线范围内（防止出界的标签匹配到网格线）
                if label_center < grid_min or label_center > grid_max:
                    continue

                distance = abs(label_center - grid_pos)
                potential_matches.append((i, label, distance))
        
        # 按距离排序，优先匹配距离最近的
        potential_matches.sort(key=lambda x: x[2])
        
        print(f"  text {len(tick_labels)} textlabel text {len(grid_positions)} textgrid line")
        print(f"  text {len(potential_matches)} text")
        
        # 使用贪心算法进行唯一匹配（每个网格线只匹配一个标签）
        matched_grids = set()
        matched_labels = set()
        
        for grid_idx, label, distance in potential_matches:
            label_id = id(label)  # 使用对象ID作为唯一标识
            
            # 如果该网格线和标签都还没被匹配过
            if grid_idx not in matched_grids and label_id not in matched_labels:
                grid_values[grid_idx] = label['value']
                matched_grids.add(grid_idx)
                matched_labels.add(label_id)
                print(f"  ✓ grid line#{grid_idx} @ {grid_positions[grid_idx]:.0f}px ↔ label {label['value']} "
                      f"(中心={label[coord_key]:.0f}px, 距离={distance:.1f}px)")
        
        # 统计匹配情况
        matched_count = sum(1 for v in grid_values if v is not None)
        print(f"[coordinatetext-{axis_name}] text: {matched_count}/{len(grid_positions)} textgrid line")
        
        if matched_count < 2:
            print(f"[coordinatetext-{axis_name}] ⚠️ textgrid linetext2text, text")
            return None
        
        # ===== 阶段2：计算单位间隔（中位数）=====
        known_pairs = [(i, val) for i, val in enumerate(grid_values) if val is not None]
        known_pairs.sort(key=lambda x: x[0])  # 按索引排序
        
        intervals = []
        for i in range(len(known_pairs) - 1):
            idx1, val1 = known_pairs[i]
            idx2, val2 = known_pairs[i + 1]
            
            value_diff = val2 - val1
            grid_diff = idx2 - idx1
            
            if grid_diff > 0:
                interval = value_diff / grid_diff
                intervals.append(interval)
                print(f"  text: label{val1}text{val2}, text{grid_diff}text, text={interval:.3f}")
        
        if not intervals:
            print(f"[coordinatetext-{axis_name}] ⚠️ text")
            return None
        
        # 使用中位数作为单位间隔
        unit_interval = np.median(intervals)
        print(f"[coordinatetext-{axis_name}] text (median): {unit_interval:.3f}")
        
        # ===== 阶段3：推算未匹配的网格线（基于最近邻）=====
        known_indices = [idx for idx, _ in known_pairs]
        
        for i in range(len(grid_positions)):
            if grid_values[i] is None:
                # 找到最近的已知网格线
                nearest_known_idx = min(known_indices, key=lambda idx: abs(idx - i))
                nearest_value = grid_values[nearest_known_idx]
                
                # 根据距离和单位间隔推算
                distance = i - nearest_known_idx
                grid_values[i] = nearest_value + distance * unit_interval
                
                print(f"  -> grid line#{i} @ {grid_positions[i]:.0f}px text {grid_values[i]:.3f} "
                      f"(基于网格#{nearest_known_idx}={nearest_value:.3f}, 距离{distance}格)")
        
        print(f"[coordinatetext-{axis_name}] ✅ textComplete")
        return grid_values
    
    def _map_grid_fallback(self, mapping: Dict, tick_labels: List[Dict], 
                           grid_positions: List[float], axis: str, coord_key: str):
        """
        回退算法：使用旧的最近邻匹配方法
        
        Args:
            mapping: 映射字典（将结果写入）
            tick_labels: 刻度标签列表
            grid_positions: 网格线位置列表
            axis: 轴标识 ('x' 或 'y')
            coord_key: 标签坐标key
        """
        labels_sorted = sorted(tick_labels, key=lambda l: l[coord_key])
        
        # 对每个刻度标签，找到最近的网格线
        label_to_grid = []
        for label in labels_sorted:
            label_pos = label[coord_key]
            label_value = label['value']
            
            distances = [abs(grid_pos - label_pos) for grid_pos in grid_positions]
            min_dist_idx = np.argmin(distances)
            
            label_to_grid.append({
                'label_value': label_value,
                'grid_index': min_dist_idx
            })
        
        if len(label_to_grid) >= 2:
            # 使用前两个标签计算间隔
            first = label_to_grid[0]
            second = label_to_grid[1]
            
            value_diff = second['label_value'] - first['label_value']
            grid_diff = second['grid_index'] - first['grid_index']
            
            if grid_diff > 0:
                value_per_grid = value_diff / grid_diff
                base_grid_index = first['grid_index']
                base_value = first['label_value']
                
                grid_values = [
                    base_value + (i - base_grid_index) * value_per_grid
                    for i in range(len(grid_positions))
                ]
                
                if axis == 'x':
                    mapping['v_grid_x_pixels'] = grid_positions
                    mapping['v_grid_x_values'] = grid_values
                    if len(grid_positions) >= 2:
                        # 最小二乘回归 (与 build_coordinate_mapping 一致)
                        px_arr = np.array(grid_positions, dtype=float)
                        vl_arr = np.array(grid_values, dtype=float)
                        A_fb = np.vstack([px_arr, np.ones(len(px_arr))]).T
                        res = np.linalg.lstsq(A_fb, vl_arr, rcond=None)
                        mapping['x_scale'] = float(res[0][0])
                        mapping['x_offset'] = float(res[0][1])
                else:  # y
                    mapping['h_grid_y_pixels'] = grid_positions
                    mapping['h_grid_y_values'] = grid_values
                    if len(grid_positions) >= 2:
                        px_arr = np.array(grid_positions, dtype=float)
                        vl_arr = np.array(grid_values, dtype=float)
                        A_fb = np.vstack([px_arr, np.ones(len(px_arr))]).T
                        res = np.linalg.lstsq(A_fb, vl_arr, rcond=None)
                        mapping['y_scale'] = float(res[0][0])
                        mapping['y_offset'] = float(res[0][1])
    
    def pixel_to_real(self, pixel_x: float, pixel_y: float,
                      use_continuous: bool = False) -> Tuple[float, float]:
        """
        将像素坐标转换为真实坐标

        两种模式:
        - use_continuous=False (默认): 找最近网格线吸附，确保交点在网格值上
        - use_continuous=True: 使用 scale*pixel+offset 连续插值，可测量任意位置

        Args:
            pixel_x: 像素x坐标
            pixel_y: 像素y坐标
            use_continuous: 是否使用连续插值模式

        Returns:
            (real_x, real_y)
        """
        if not self.coordinate_mapping:
            return None, None

        # 连续插值模式: 直接使用回归参数
        if use_continuous:
            real_x = self.coordinate_mapping.get('x_scale', 1.0) * pixel_x + self.coordinate_mapping.get('x_offset', 0.0)
            real_y = self.coordinate_mapping.get('y_scale', 1.0) * pixel_y + self.coordinate_mapping.get('y_offset', 0.0)
            return real_x, real_y

        # ===== 网格吸附模式 (默认, 与原逻辑相同) =====
        # 计算真实X坐标：找最近的垂直网格线
        real_x = None
        if 'v_grid_lines' in self.coordinate_mapping and 'v_grid_x_values' in self.coordinate_mapping:
            v_lines = self.coordinate_mapping['v_grid_lines']
            v_values = np.array(self.coordinate_mapping['v_grid_x_values'])
            distances = np.array([
                abs(self._line_x_at_y(line, pixel_y) - pixel_x) for line in v_lines
            ], dtype=float)
            min_idx = np.argmin(distances)
            real_x = v_values[min_idx]
        elif 'v_grid_x_pixels' in self.coordinate_mapping and 'v_grid_x_values' in self.coordinate_mapping:
            v_pixels = np.array(self.coordinate_mapping['v_grid_x_pixels'])
            v_values = np.array(self.coordinate_mapping['v_grid_x_values'])
            distances = np.abs(v_pixels - pixel_x)
            min_idx = np.argmin(distances)
            real_x = v_values[min_idx]
        else:
            # 回退到线性插值
            real_x = self.coordinate_mapping.get('x_scale', 1.0) * pixel_x + self.coordinate_mapping.get('x_offset', 0.0)
        
        # ===== 计算真实Y坐标：找最近的水平网格线 =====
        real_y = None
        if 'h_grid_lines' in self.coordinate_mapping and 'h_grid_y_values' in self.coordinate_mapping:
            h_lines = self.coordinate_mapping['h_grid_lines']
            h_values = np.array(self.coordinate_mapping['h_grid_y_values'])
            distances = np.array([
                abs(self._line_y_at_x(line, pixel_x) - pixel_y) for line in h_lines
            ], dtype=float)
            min_idx = np.argmin(distances)
            real_y = h_values[min_idx]
        elif 'h_grid_y_pixels' in self.coordinate_mapping and 'h_grid_y_values' in self.coordinate_mapping:
            h_pixels = np.array(self.coordinate_mapping['h_grid_y_pixels'])
            h_values = np.array(self.coordinate_mapping['h_grid_y_values'])
            distances = np.abs(h_pixels - pixel_y)
            min_idx = np.argmin(distances)
            real_y = h_values[min_idx]
        else:
            # 回退到线性插值
            real_y = self.coordinate_mapping.get('y_scale', 1.0) * pixel_y + self.coordinate_mapping.get('y_offset', 0.0)
        
        return real_x, real_y
    
    def extract_curve(self, image: np.ndarray, grid_h_lines: List, grid_v_lines: List,
                     ocr_result: Dict, line_thickness: int = 5) -> Tuple[np.ndarray, Dict]:
        """
        提取曲线：基于颜色的新方案
        
        核心思路：曲线通常是彩色的（红、蓝、绿等），网格和坐标轴是黑灰色
        
        Args:
            image: 原始图像（彩色）
            grid_h_lines: 水平网格线列表（备用）
            grid_v_lines: 垂直网格线列表（备用）
            ocr_result: OCR OCR结果（备用）
            line_thickness: 擦除线条的粗细（备用）
        
        Returns:
            (提取出的曲线图像, 中间结果字典)
        """
        intermediates = {}
        h, w = image.shape[:2]
        
        print(f"[CurveExtract] text: textcolortextcurve")
        
        # 确保是彩色图像
        if len(image.shape) == 2:
            print("  Warning: text, textcolortext")
            # 回退到旧方案
            return self._extract_curve_grayscale(image, grid_h_lines, grid_v_lines, 
                                                ocr_result, line_thickness)
        
        color_image = image.copy()
        
        # ===== 步骤1: 转换到HSV颜色空间 =====
        print(f"[CurveExtract] text1: textHSVcolortext")
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # ===== 步骤2: 提取彩色区域（曲线） =====
        print(f"[CurveExtract] text2: text")
        
        # Adaptive saturation threshold: darker backgrounds reduce edge
        # pixel saturation due to anti-aliasing blending, so we lower the
        # threshold proportionally to keep detection recall stable.
        _h_ch, _s_ch, _v_ch = cv2.split(hsv)
        _bg_mask = (_s_ch < 30) & (_v_ch > 100)
        if np.any(_bg_mask):
            bg_v_mean = float(_v_ch[_bg_mask].mean())
        else:
            bg_v_mean = 255.0
        
        _REF_BG_V = 250.0
        _MIN_SCALE = 0.60
        ratio = bg_v_mean / _REF_BG_V
        scale = max(_MIN_SCALE, ratio ** 1.5)
        s_thresh = int(round(self.color_saturation_threshold * scale))
        v_thresh = int(round(self.color_value_threshold * scale))
        
        print(f"[CurveExtract] textV={bg_v_mean:.0f}, "
              f"自适应阈值 S>{s_thresh} V>{v_thresh} "
              f"(基准 S>{self.color_saturation_threshold} V>{self.color_value_threshold})")
        
        # HSV: H(色调0-180), S(饱和度0-255), V(明度0-255)
        color_masks = []
        
        # 红色 (有两个范围，因为红色在HSV色环的两端)
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
                print(f"  text {color_name} color: {pixel_count} pixel")
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        intermediates['binary'] = combined_mask.copy()
        
        # ===== 步骤3: 形态学操作 - 去噪和连接 =====
        print(f"[CurveExtract] text3: text (text+text)")
        
        # 先去除小噪点
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # 连接断裂部分
        if self.morph_connect_enabled:
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                     (self.morph_kernel_size, self.morph_kernel_size))
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_close, 
                                     iterations=self.morph_iterations)
        
        intermediates['skeleton'] = result.copy()  # 简化，不做骨架提取
        intermediates['filtered'] = result.copy()
        intermediates['connected'] = result.copy()
        
        print(f"[CurveExtract] textComplete, curvepixeltext: {np.sum(result > 0)}")
        
        return result, intermediates
    
    def _skeleton_direction_filter(self, binary_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        骨架提取 + 方向分析过滤
        
        通过分析骨架上每个点的局部方向变化来区分直线和曲线
        
        Returns:
            (filtered_mask, skeleton)
        """
        h, w = binary_image.shape[:2]
        
        # 1. 提取骨架（Zhang-Suen细化算法）
        skeleton = cv2.ximgproc.thinning(binary_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        print(f"   textpixeltext: {np.sum(skeleton > 0)}")
        
        # 2. 分析方向变化，过滤直线
        # 对骨架进行轮廓提取
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        filtered_mask = np.zeros((h, w), dtype=np.uint8)
        kept_count = 0
        
        for contour in contours:
            if len(contour) < 20:  # 太短的忽略
                continue
            
            # 计算方向变化率
            direction_change = self._calculate_direction_variance(contour)
            
            print(f"   text: {len(contour)}, text: {direction_change:.4f}")
            
            # 方向变化大的是曲线，保留
            # 方向变化小的是直线（网格线残留），过滤
            if direction_change > self.direction_variance_threshold:
                cv2.drawContours(filtered_mask, [contour], -1, 255, 1)
                kept_count += 1
        
        print(f"   Keeptext: {kept_count}/{len(contours)}")
        
        return filtered_mask, skeleton
    
    def _calculate_direction_variance(self, contour: np.ndarray, window_size: int = 10) -> float:
        """
        计算轮廓的方向变化率
        
        Args:
            contour: 轮廓点
            window_size: 滑动窗口大小
        
        Returns:
            方向变化率（0-1之间，越大表示方向变化越剧烈）
        """
        if len(contour) < window_size * 2:
            return 0.0
        
        # 简化轮廓点
        points = contour[:, 0, :]  # [N, 2]
        
        # 计算每个点的局部方向
        angles = []
        for i in range(window_size, len(points) - window_size):
            # 前后各取window_size个点
            prev_pt = points[i - window_size]
            next_pt = points[i + window_size]
            
            # 计算方向角
            dx = next_pt[0] - prev_pt[0]
            dy = next_pt[1] - prev_pt[1]
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        if len(angles) == 0:
            return 0.0
        
        # 计算角度的标准差（归一化）
        angles = np.array(angles)
        # 处理角度的周期性（-π到π）
        angles_diff = np.diff(angles)
        angles_diff = np.arctan2(np.sin(angles_diff), np.cos(angles_diff))  # 归一化到[-π, π]
        
        # 计算标准差
        variance = np.std(np.abs(angles_diff))
        
        # 归一化到[0, 1]
        normalized_variance = min(variance / np.pi, 1.0)
        
        return normalized_variance
    
    def _smart_endpoint_connection(self, skeleton: np.ndarray) -> np.ndarray:
        """
        智能端点连接：连接距离近且方向匹配的端点
        
        使用self中的参数：
        - self.endpoint_max_distance
        - self.endpoint_angle_threshold
        
        Returns:
            连接后的图像
        """
        h, w = skeleton.shape[:2]
        result = skeleton.copy()
        
        # 1. 找出所有端点
        endpoints = self._find_endpoints(skeleton)
        print(f"   text: {len(endpoints)}")
        print(f"   textparameter: text={self.endpoint_max_distance}, anglethreshold={self.endpoint_angle_threshold}°")
        
        if len(endpoints) < 2:
            return result
        
        # 2. 对每个端点，计算其切线方向
        endpoint_info = []
        for ep in endpoints:
            direction = self._estimate_endpoint_direction(skeleton, ep)
            endpoint_info.append({
                'pos': ep,
                'direction': direction
            })
        
        # 3. 匹配端点对
        connected_count = 0
        matched = set()
        
        for i, ep1 in enumerate(endpoint_info):
            if i in matched:
                continue
            
            for j, ep2 in enumerate(endpoint_info[i+1:], start=i+1):
                if j in matched:
                    continue
                
                # 计算距离
                dist = np.linalg.norm(np.array(ep1['pos']) - np.array(ep2['pos']))
                
                if dist > self.endpoint_max_distance:
                    continue
                
                # 检查方向是否匹配
                if ep1['direction'] is not None and ep2['direction'] is not None:
                    # 计算两个端点指向彼此的理想角度
                    ideal_angle = np.arctan2(ep2['pos'][1] - ep1['pos'][1], 
                                            ep2['pos'][0] - ep1['pos'][0])
                    
                    # ep1的方向应该指向ep2
                    angle_diff1 = abs(self._angle_difference(ep1['direction'], ideal_angle))
                    # ep2的方向应该指向ep1
                    angle_diff2 = abs(self._angle_difference(ep2['direction'], ideal_angle + np.pi))
                    
                    # 角度差都应该小于阈值
                    if (angle_diff1 < np.radians(self.endpoint_angle_threshold) and 
                        angle_diff2 < np.radians(self.endpoint_angle_threshold)):
                        # 连接这两个端点
                        cv2.line(result, ep1['pos'], ep2['pos'], 255, 1)
                        matched.add(i)
                        matched.add(j)
                        connected_count += 1
                        break
        
        print(f"   text{connected_count}text")
        
        return result
    
    def _find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """
        找出骨架图像中的所有端点
        
        端点定义：8邻域内只有1个前景像素的点
        """
        endpoints = []
        h, w = skeleton.shape[:2]
        
        # 遍历所有前景像素
        y_coords, x_coords = np.where(skeleton > 0)
        
        for y, x in zip(y_coords, x_coords):
            # 检查8邻域
            neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if skeleton[ny, nx] > 0:
                            neighbors += 1
            
            # 只有1个邻居的是端点
            if neighbors == 1:
                endpoints.append((x, y))
        
        return endpoints
    
    def _estimate_endpoint_direction(self, skeleton: np.ndarray, 
                                     endpoint: Tuple[int, int], 
                                     search_len: int = 10) -> Optional[float]:
        """
        估计端点的切线方向
        
        Args:
            skeleton: 骨架图像
            endpoint: 端点坐标 (x, y)
            search_len: 搜索长度
        
        Returns:
            方向角（弧度），如果无法估计则返回None
        """
        x, y = endpoint
        h, w = skeleton.shape[:2]
        
        # 沿着骨架追踪一小段距离
        path = [endpoint]
        visited = {endpoint}
        
        current = endpoint
        for _ in range(search_len):
            # 找下一个相邻的骨架点
            cx, cy = current
            found = False
            
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= ny < h and 0 <= nx < w:
                        if skeleton[ny, nx] > 0 and (nx, ny) not in visited:
                            path.append((nx, ny))
                            visited.add((nx, ny))
                            current = (nx, ny)
                            found = True
                            break
                if found:
                    break
            
            if not found:
                break
        
        # 需要至少2个点才能计算方向
        if len(path) < 2:
            return None
        
        # 计算从端点到路径末端的方向
        start = np.array(path[0])
        end = np.array(path[-1])
        direction = end - start
        
        angle = np.arctan2(direction[1], direction[0])
        return angle
    
    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """
        计算两个角度之间的差异（考虑周期性）
        
        Returns:
            角度差（弧度），范围[-π, π]
        """
        diff = angle1 - angle2
        # 归一化到[-π, π]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
    
    def _filter_curve_contours(self, binary_image: np.ndarray, 
                               min_area: int = 15, 
                               aspect_threshold: float = 8,
                               angle_tol: float = 12,
                               close_size: int = 7,
                               close_iter: int = 2) -> np.ndarray:
        """
        通过轮廓分析过滤曲线：
        1. 先按照接近垂直或水平的线段长度去消除
        2. 把曲线按照轮廓连起来，变成连续的曲线
        3. 消除剩余的曲线以外的噪点
        
        Args:
            binary_image: 二值图（白色为前景）
            min_area: 最小轮廓面积
            aspect_threshold: 长宽比阈值
            angle_tol: 角度容差（度）
            close_size: 形态学闭运算核大小
            close_iter: 形态学闭运算迭代次数
        
        Returns:
            过滤后的曲线图像
        """
        # 找出所有轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return binary_image
        
        # 创建空白画布
        h, w = binary_image.shape[:2]
        
        # 第一步：过滤掉接近水平或垂直的线段
        curve_contours = []
        
        print(f"[CurveFilter] text: {len(contours)}")
        print(f"[CurveFilter] parameter - text: {min_area}, texthreshold: {aspect_threshold}, angletext: {angle_tol}°")
        
        for contour in contours:
            # 计算轮廓的基本属性
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # 过滤太小的轮廓（噪点）
            if area < min_area or perimeter < min_area:
                continue
            
            # 判断是否为接近水平或垂直的线段
            should_keep = True
            
            if len(contour) >= 5:
                # 拟合最小外接矩形
                rect = cv2.minAreaRect(contour)
                rect_w, rect_h = rect[1]
                if rect_w == 0 or rect_h == 0:
                    continue
                    
                angle = rect[2]
                
                # 计算长宽比
                ratio = max(rect_w, rect_h) / min(rect_w, rect_h)
                
                # 只对细长结构进行角度检查
                if ratio > aspect_threshold:
                    # 标准化角度到[0, 90]
                    normalized_angle = abs(angle) if abs(angle) <= 45 else 90 - abs(angle)
                    if angle < -45:
                        normalized_angle = abs(angle + 90)
                    
                    # 检查是否接近水平（0度）或垂直（90度）
                    # 水平：normalized_angle接近0
                    # 垂直：normalized_angle接近90
                    angle_from_horizontal = min(abs(normalized_angle), abs(normalized_angle - 90))
                    
                    if angle_from_horizontal < angle_tol:
                        should_keep = False
            
            if should_keep:
                curve_contours.append(contour)
        
        print(f"[CurveFilter] filteraftertext: {len(curve_contours)}")
        
        if len(curve_contours) == 0:
            return np.zeros((h, w), dtype=np.uint8)
        
        # 第二步：绘制曲线轮廓
        curve_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(curve_mask, curve_contours, -1, 255, -1)
        
        # 使用形态学闭运算连接断裂的曲线段
        print(f"[CurveConnect] text: {close_size}, iterationtext: {close_iter}")
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        curve_mask = cv2.morphologyEx(curve_mask, cv2.MORPH_CLOSE, kernel_close, iterations=close_iter)
        
        # 第三步：再次提取轮廓，消除小噪点
        contours_final, _ = cv2.findContours(curve_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 只保留较大的轮廓
        result = np.zeros((h, w), dtype=np.uint8)
        kept_count = 0
        for contour in contours_final:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # 保留面积和周长都足够大的轮廓
            if area > min_area * 3 and perimeter > min_area * 3:
                cv2.drawContours(result, [contour], -1, 255, -1)
                kept_count += 1
        
        print(f"[CurveFilter] finalKeeptext: {kept_count}")
        
        # 可选：对曲线进行轻微膨胀，使其更连续
        if kept_count > 0:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            result = cv2.dilate(result, kernel_dilate, iterations=1)
        
        return result
    
    def find_line_intersections(self, curve_binary: np.ndarray, 
                               grid_h_lines: List, grid_v_lines: List,
                               axes_info: Dict, tolerance: int = 5) -> List[Dict]:
        """
        查找曲线与网格线/坐标轴的交点（至少3条线交汇）
        
        使用提取的曲线和合并后的网格线进行交点检测
        
        Args:
            curve_binary: 曲线二值图（白色为曲线）
            grid_h_lines: 水平网格线
            grid_v_lines: 垂直网格线
            axes_info: 坐标轴信息
            tolerance: 交点容差（像素）
        
        Returns:
            交点列表
        """
        intersections = []
        h, w = curve_binary.shape[:2]
        
        # 提取曲线上的所有像素点
        curve_points = np.argwhere(curve_binary > 128)  # [row, col] = [y, x]
        
        if len(curve_points) == 0:
            return intersections
        
        # 转换为 [x, y] 格式
        curve_points_xy = curve_points[:, [1, 0]]  # [x, y]
        
        # 准备所有线条信息（包含类型标记）
        all_lines = []
        
        # 添加水平网格线
        for line in grid_h_lines:
            x1, y1, x2, y2 = line
            all_lines.append({
                'coords': (x1, y1, x2, y2),
                'type': 'h_grid',
                'pos': (y1 + y2) / 2  # y坐标
            })
        
        # 添加X轴（如果存在且不与网格线重复）
        x_axis_added = False
        if 'x_axis_line' in axes_info and axes_info['x_axis_line'] is not None:
            x1, y1, x2, y2 = axes_info['x_axis_line']
            x_axis_y = (y1 + y2) / 2
            
            # 检查是否与现有的水平网格线重复（容差10像素）
            is_duplicate = False
            for line in all_lines:
                if line['type'] == 'h_grid':
                    if abs(line['pos'] - x_axis_y) < 10:
                        # 将该网格线标记为X轴
                        line['type'] = 'x_axis'
                        is_duplicate = True
                        x_axis_added = True
                        break
            
            # 如果不重复，添加X轴
            if not is_duplicate:
                all_lines.append({
                    'coords': (x1, y1, x2, y2),
                    'type': 'x_axis',
                    'pos': x_axis_y
                })
                x_axis_added = True
        
        # 添加垂直网格线
        for line in grid_v_lines:
            x1, y1, x2, y2 = line
            all_lines.append({
                'coords': (x1, y1, x2, y2),
                'type': 'v_grid',
                'pos': (x1 + x2) / 2  # x坐标
            })
        
        # 添加Y轴（如果存在且不与网格线重复）
        y_axis_added = False
        if 'y_axis_line' in axes_info and axes_info['y_axis_line'] is not None:
            x1, y1, x2, y2 = axes_info['y_axis_line']
            y_axis_x = (x1 + x2) / 2
            
            # 检查是否与现有的垂直网格线重复（容差10像素）
            is_duplicate = False
            for line in all_lines:
                if line['type'] == 'v_grid':
                    if abs(line['pos'] - y_axis_x) < 10:
                        # 将该网格线标记为Y轴
                        line['type'] = 'y_axis'
                        is_duplicate = True
                        y_axis_added = True
                        break
            
            # 如果不重复，添加Y轴
            if not is_duplicate:
                all_lines.append({
                    'coords': (x1, y1, x2, y2),
                    'type': 'y_axis',
                    'pos': y_axis_x
                })
                y_axis_added = True
        
        # 候选交点：曲线与每条网格线的交点
        # 新策略：直接记录网格线的像素位置和真实坐标值
        candidate_points = []
        
        # 先从mapping中获取网格线的真实坐标映射
        v_grid_pixels = []
        v_grid_values = []
        h_grid_pixels = []
        h_grid_values = []
        
        if 'v_grid_x_pixels' in self.coordinate_mapping and 'v_grid_x_values' in self.coordinate_mapping:
            v_grid_pixels = np.array(self.coordinate_mapping['v_grid_x_pixels'])
            v_grid_values = np.array(self.coordinate_mapping['v_grid_x_values'])
        
        if 'h_grid_y_pixels' in self.coordinate_mapping and 'h_grid_y_values' in self.coordinate_mapping:
            h_grid_pixels = np.array(self.coordinate_mapping['h_grid_y_pixels'])
            h_grid_values = np.array(self.coordinate_mapping['h_grid_y_values'])

        # Tight local validation window around a grid crossing. This is stricter
        # than the coarse line-level tolerance and prevents "near-miss" curve
        # pixels from being treated as true intersections.
        crossing_probe_tol = max(1.5, float(tolerance) * 0.55)
        
        # 辅助函数：根据像素坐标找到最接近的网格线的真实坐标
        def find_nearest_grid_value(pixel_pos, grid_pixels, grid_values):
            if len(grid_pixels) == 0:
                return None
            distances = np.abs(grid_pixels - pixel_pos)
            min_idx = np.argmin(distances)
            # 只有距离足够近才返回（容差5像素）
            if distances[min_idx] < 5:
                return grid_values[min_idx]
            return None
        
        for line_info in all_lines:
            x1, y1, x2, y2 = line_info['coords']
            line_type = line_info['type']
            
            if line_type in ['h_grid', 'x_axis']:
                # 水平线：检查曲线在该y坐标附近的点
                line_y = int((y1 + y2) / 2)
                
                # 获取该水平线的真实y坐标
                real_y_value = find_nearest_grid_value(line_y, h_grid_pixels, h_grid_values)
                
                # 找到曲线上y坐标接近line_y的点
                y_distances = np.abs(curve_points_xy[:, 1] - line_y)
                nearby_indices = np.where(y_distances <= tolerance)[0]
                
                for idx in nearby_indices:
                    px, py = curve_points_xy[idx]
                    candidate_points.append({
                        'pixel_x': px,
                        'pixel_y': line_y,
                        'curve_px': px,
                        'curve_py': py,
                        'line_type': line_type,
                        'real_y_line': real_y_value  # grid-line value only as evidence
                    })
            
            else:  # v_grid or y_axis
                # 垂直线：检查曲线在该x坐标附近的点
                line_x = int((x1 + x2) / 2)
                
                # 获取该垂直线的真实x坐标
                real_x_value = find_nearest_grid_value(line_x, v_grid_pixels, v_grid_values)
                
                # 找到曲线上x坐标接近line_x的点
                x_distances = np.abs(curve_points_xy[:, 0] - line_x)
                nearby_indices = np.where(x_distances <= tolerance)[0]
                
                for idx in nearby_indices:
                    px, py = curve_points_xy[idx]
                    candidate_points.append({
                        'pixel_x': line_x,
                        'pixel_y': py,
                        'curve_px': px,
                        'curve_py': py,
                        'line_type': line_type,
                        'real_x_line': real_x_value  # grid-line value only as evidence
                    })
        
        # 对候选点进行聚类，找出至少3条线交汇的点
        if len(candidate_points) == 0:
            return intersections
        
        # 使用字典按位置聚类
        point_groups = {}
        for pt in candidate_points:
            # 将坐标量化到tolerance范围内
            key_x = int(round(pt['pixel_x'] / tolerance)) * tolerance
            key_y = int(round(pt['pixel_y'] / tolerance)) * tolerance
            key = (key_x, key_y)
            
            if key not in point_groups:
                point_groups[key] = []
            point_groups[key].append(pt)
        
        # 统计每个位置的交汇线数
        for key, group in point_groups.items():
            # 计算平均像素位置
            avg_pixel_x = int(np.mean([pt['pixel_x'] for pt in group]))
            avg_pixel_y = int(np.mean([pt['pixel_y'] for pt in group]))
            
            # 统计有多少条不同类型的线
            line_types_set = set([pt['line_type'] for pt in group])
            line_types_list = list(line_types_set)

            has_h = any(t in ('h_grid', 'x_axis') for t in line_types_set)
            has_v = any(t in ('v_grid', 'y_axis') for t in line_types_set)

            # Require both horizontal and vertical evidence to form a grid
            # intersection candidate; this is more robust than pure count.
            if has_h and has_v:
                v_snaps = [
                    float(pt['pixel_x']) for pt in group
                    if pt.get('line_type') in ('v_grid', 'y_axis')
                ]
                h_snaps = [
                    float(pt['pixel_y']) for pt in group
                    if pt.get('line_type') in ('h_grid', 'x_axis')
                ]
                if not v_snaps or not h_snaps:
                    continue

                cross_x = float(np.median(v_snaps))
                cross_y = float(np.median(h_snaps))
                local_mask = (
                    (np.abs(curve_points_xy[:, 0] - cross_x) <= crossing_probe_tol)
                    & (np.abs(curve_points_xy[:, 1] - cross_y) <= crossing_probe_tol)
                )
                local_points = curve_points_xy[local_mask]
                if len(local_points) == 0:
                    continue

                local_offsets = np.max(np.abs(local_points - np.array([cross_x, cross_y])), axis=1)
                best_idx = int(np.argmin(local_offsets))
                if float(local_offsets[best_idx]) > crossing_probe_tol:
                    continue

                # 加上曲线本身
                line_types_list.append('curve')
                lines_count = len(line_types_list)
                # Derive real coordinates from actual curve location (not the
                # snapped line coordinates used for clustering).
                curve_x = int(round(float(local_points[best_idx][0])))
                curve_y = int(round(float(local_points[best_idx][1])))
                real_x, real_y = self.pixel_to_real(curve_x, curve_y)
                if real_x is None or real_y is None:
                    # Fallback to line-derived values only when mapping fails.
                    real_x_list = [
                        pt['real_x_line'] for pt in group
                        if 'real_x_line' in pt and pt['real_x_line'] is not None
                    ]
                    real_y_list = [
                        pt['real_y_line'] for pt in group
                        if 'real_y_line' in pt and pt['real_y_line'] is not None
                    ]
                    if real_x is None and len(real_x_list) > 0:
                        real_x = float(np.mean(real_x_list))
                    if real_y is None and len(real_y_list) > 0:
                        real_y = float(np.mean(real_y_list))
                if real_x is None or real_y is None:
                    continue
                
                # 使用阈值判断是否应该近似为整数
                real_x_approx, is_x_integer = self.approximate_to_integer(real_x)
                real_y_approx, is_y_integer = self.approximate_to_integer(real_y)
                
                intersections.append({
                    'pixel_x': int(curve_x),
                    'pixel_y': int(curve_y),
                    'real_x': float(real_x),
                    'real_y': float(real_y),
                    'real_x_rounded': real_x_approx,
                    'real_y_rounded': real_y_approx,
                    'is_x_integer': is_x_integer,
                    'is_y_integer': is_y_integer,
                    'lines_count': lines_count,
                    'line_types': line_types_list
                })
        
        # 最终去重
        intersections = self._merge_nearby_points(intersections, tolerance * 2)

        # Fallback path: if grouping-based intersections collapse to empty,
        # recover points from vertical-grid probing with crossing validation.
        if len(intersections) == 0 and len(v_grid_pixels) > 0 and len(h_grid_pixels) > 0:
            h_val_to_pixel: Dict[int, float] = {}
            for hp, hv in zip(h_grid_pixels, h_grid_values):
                iv = int(round(float(hv)))
                if abs(float(hv) - iv) > 0.25:
                    continue
                if iv not in h_val_to_pixel:
                    h_val_to_pixel[iv] = float(hp)

            fallback: List[Dict[str, Any]] = []
            x_probe_tol = max(1.5, float(tolerance) * 1.0)
            y_probe_tol = max(1.5, float(tolerance) * 0.7)
            for vx_px, vx_val in zip(v_grid_pixels, v_grid_values):
                # Prefer exact integer x-grid lines for coordinate points.
                xi = int(round(float(vx_val)))
                if abs(float(vx_val) - xi) > 0.25:
                    continue

                x_dist = np.abs(curve_points_xy[:, 0] - float(vx_px))
                idxs = np.where(x_dist <= x_probe_tol)[0]
                if len(idxs) < 2:
                    continue

                ys = curve_points_xy[idxs, 1].astype(float)
                y_px = float(np.median(ys))
                real_x, real_y = self.pixel_to_real(float(vx_px), y_px)
                if real_x is None or real_y is None:
                    continue

                x_round, is_x_integer = self.approximate_to_integer(float(real_x))
                y_round, is_y_integer = self.approximate_to_integer(float(real_y))
                if not (is_x_integer and is_y_integer):
                    continue

                yi = int(round(float(y_round)))
                if yi not in h_val_to_pixel:
                    continue
                target_y_px = float(h_val_to_pixel[yi])
                if abs(y_px - target_y_px) > y_probe_tol:
                    continue

                fallback.append({
                    'pixel_x': int(round(float(vx_px))),
                    'pixel_y': int(round(y_px)),
                    'real_x': float(real_x),
                    'real_y': float(real_y),
                    'real_x_rounded': int(round(float(x_round))),
                    'real_y_rounded': int(round(float(y_round))),
                    'is_x_integer': True,
                    'is_y_integer': True,
                    'lines_count': 3,
                    'line_types': ['v_grid', 'h_grid', 'curve'],
                })

            intersections = self._merge_nearby_points(fallback, tolerance * 2)
        
        return intersections
    
    def _prepare_reasoning_intersections(self, intersections: List[Dict]) -> List[Dict]:
        """
        Prepare intersections for downstream reasoning.

        This path keeps the point set as complete as possible while still doing
        minimal cleanup:
        1. keep only integer-aligned points
        2. recover high-confidence near-integer points (mapping jitter only)
        3. dedupe repeated integer coordinates

        Unlike visualization filtering, this method never sparsifies points for
        label collision avoidance.
        """
        if not intersections:
            self._last_reasoning_debug = {"empty_input": True}
            return intersections

        raw_points: List[Tuple[float, float]] = []
        for p in intersections:
            try:
                rx = float(p.get('real_x'))
                ry = float(p.get('real_y'))
            except (TypeError, ValueError):
                continue
            if np.isfinite(rx) and np.isfinite(ry):
                raw_points.append((rx, ry))

        def _supports_horizontal_level(xi: float, yi: float) -> bool:
            """Check whether y=yi has local above/below support near xi."""
            if not raw_points:
                return True
            # Use a slightly wider x-window so run endpoints are still judged
            # with enough neighboring context.
            window = [(x, y) for x, y in raw_points if abs(x - xi) <= 2.2]
            if len(window) < 3:
                # Not enough evidence to claim a true crossing/touch.
                return False
            eps = 0.2
            has_above = any((y - yi) > eps for _, y in window)
            has_below = any((yi - y) > eps for _, y in window)
            return has_above and has_below

        unique_points = {}
        for p in intersections:
            if p['is_x_integer'] and p['is_y_integer']:
                key = (p['real_x_rounded'], p['real_y_rounded'])
                if key not in unique_points:
                    unique_points[key] = p

        # Conservative recall boost with continuity validation:
        # 1) candidate is near integer and has strong grid-crossing evidence
        # 2) candidate must agree with the strict integer backbone trend
        #    (local interpolation/extrapolation from already-trusted points)
        rescue_threshold = max(float(self.integer_approximation_threshold), 0.45)
        backbone: Dict[int, List[int]] = defaultdict(list)
        for p in unique_points.values():
            try:
                bx = int(round(float(p['real_x_rounded'])))
                by = int(round(float(p['real_y_rounded'])))
            except (TypeError, ValueError):
                continue
            backbone[bx].append(by)

        def _median_y(x_key: int) -> Optional[float]:
            vals = backbone.get(x_key, [])
            if not vals:
                return None
            return float(np.median(vals))

        def _predict_y_from_backbone(x_val: int) -> Optional[float]:
            xs = sorted(backbone.keys())
            if len(xs) < 2:
                return None

            left_keys = [k for k in xs if k < x_val]
            right_keys = [k for k in xs if k > x_val]

            # Primary: bounded local interpolation (most stable).
            if left_keys and right_keys:
                xl = left_keys[-1]
                xr = right_keys[0]
                if (xr - xl) <= 5:
                    yl = _median_y(xl)
                    yr = _median_y(xr)
                    if yl is not None and yr is not None and abs(xr - xl) > 1e-6:
                        slope = (yr - yl) / float(xr - xl)
                        if abs(slope) <= 5.0:
                            return yl + slope * float(x_val - xl)

            # Secondary: short-range one-sided extrapolation.
            if len(left_keys) >= 2:
                x0, x1 = left_keys[-2], left_keys[-1]
                if (x1 - x0) <= 3 and (x_val - x1) <= 5:
                    y0 = _median_y(x0)
                    y1 = _median_y(x1)
                    if y0 is not None and y1 is not None and abs(x1 - x0) > 1e-6:
                        slope = (y1 - y0) / float(x1 - x0)
                        if abs(slope) <= 5.0:
                            return y1 + slope * float(x_val - x1)
            if len(right_keys) >= 2:
                x0, x1 = right_keys[0], right_keys[1]
                if (x1 - x0) <= 3 and (x0 - x_val) <= 5:
                    y0 = _median_y(x0)
                    y1 = _median_y(x1)
                    if y0 is not None and y1 is not None and abs(x1 - x0) > 1e-6:
                        slope = (y1 - y0) / float(x1 - x0)
                        if abs(slope) <= 5.0:
                            return y0 - slope * float(x0 - x_val)
            return None

        rescued = 0
        rescued_keys: List[Tuple[int, int]] = []
        for p in intersections:
            if p.get('is_x_integer') and p.get('is_y_integer'):
                continue
            if int(p.get('lines_count', 0)) < 2:
                continue

            line_types = set(p.get('line_types', []) or [])
            has_vertical = ('v_grid' in line_types) or ('y_axis' in line_types)
            has_horizontal = ('h_grid' in line_types) or ('x_axis' in line_types)
            if not (has_vertical and has_horizontal):
                continue

            try:
                rx = float(p.get('real_x'))
                ry = float(p.get('real_y'))
            except (TypeError, ValueError):
                continue

            xr = round(rx)
            yr = round(ry)
            if abs(rx - xr) > rescue_threshold or abs(ry - yr) > rescue_threshold:
                continue

            y_pred = _predict_y_from_backbone(int(xr))
            if y_pred is None:
                continue
            if abs(float(yr) - float(y_pred)) > 1.5:
                continue

            key = (int(xr), int(yr))
            if key in unique_points:
                continue

            rescued_point = dict(p)
            rescued_point['real_x_rounded'] = int(xr)
            rescued_point['real_y_rounded'] = int(yr)
            rescued_point['is_x_integer'] = True
            rescued_point['is_y_integer'] = True
            unique_points[key] = rescued_point
            rescued += 1
            rescued_keys.append(key)

        # Rebuild backbone after rescue for the gap-fill pass.
        backbone.clear()
        for p in unique_points.values():
            try:
                bx = int(round(float(p['real_x_rounded'])))
                by = int(round(float(p['real_y_rounded'])))
            except (TypeError, ValueError):
                continue
            backbone[bx].append(by)

        # Gap-fill pass: column-sweep approach.
        # For each vertical grid line within/near the backbone x-range, scan
        # curve_binary to find where the curve actually is, then snap to the
        # nearest integer-y grid line.  This is robust for non-linear curves
        # (e.g. cubic) where backbone extrapolation fails.
        gap_filled = 0
        gap_filled_keys: List[Tuple[int, int]] = []
        curve_bin = getattr(self, '_last_curve_binary', None)
        mapping = self.coordinate_mapping or {}
        v_pix = np.array(mapping.get('v_grid_x_pixels', []))
        v_val = np.array(mapping.get('v_grid_x_values', []))
        h_pix = np.array(mapping.get('h_grid_y_pixels', []))
        h_val = np.array(mapping.get('h_grid_y_values', []))
        bb_xs = sorted(backbone.keys())
        if curve_bin is not None and len(v_pix) > 0 and len(h_pix) > 0 and len(bb_xs) >= 2:
            x_min, x_max = bb_xs[0], bb_xs[-1]
            col_tol = 4
            extend = 2
            h_int_val = np.round(h_val).astype(int)

            for xi in range(x_min - extend, x_max + extend + 1):
                if xi in backbone:
                    continue

                # Locate the pixel column for this integer x.
                vd = np.abs(v_val - xi)
                if vd.min() > 0.6:
                    continue
                px_x = float(v_pix[np.argmin(vd)])

                # Scan curve_binary along this column to find curve pixels.
                x_lo = max(0, int(px_x) - col_tol)
                x_hi = min(curve_bin.shape[1], int(px_x) + col_tol + 1)
                col_strip = curve_bin[:, x_lo:x_hi]
                row_hits = np.where(col_strip.max(axis=1) > 128)[0]
                if len(row_hits) < 2:
                    continue

                # Cluster consecutive row hits into distinct crossings
                # (a single curve may have multiple y-crossings at one x).
                clusters: List[List[int]] = []
                cur_cluster = [int(row_hits[0])]
                for ri in range(1, len(row_hits)):
                    if row_hits[ri] - row_hits[ri - 1] <= 3:
                        cur_cluster.append(int(row_hits[ri]))
                    else:
                        clusters.append(cur_cluster)
                        cur_cluster = [int(row_hits[ri])]
                clusters.append(cur_cluster)

                for cluster in clusters:
                    median_py = float(np.median(cluster))

                    # Snap to nearest integer-y grid line.
                    hd = np.abs(h_pix - median_py)
                    best_idx = int(np.argmin(hd))
                    if hd[best_idx] > col_tol + 2:
                        continue
                    yi = int(h_int_val[best_idx])
                    if abs(float(h_val[best_idx]) - yi) > 0.5:
                        continue

                    key = (xi, yi)
                    if key in unique_points:
                        continue

                    unique_points[key] = {
                        'pixel_x': int(round(px_x)),
                        'pixel_y': int(round(float(h_pix[best_idx]))),
                        'real_x': float(xi),
                        'real_y': float(yi),
                        'real_x_rounded': xi,
                        'real_y_rounded': yi,
                        'is_x_integer': True,
                        'is_y_integer': True,
                        'lines_count': 3,
                        'line_types': ['v_grid', 'h_grid', 'curve'],
                    }
                    gap_filled += 1
                    gap_filled_keys.append(key)

        prepared = list(unique_points.values())
        prepared.sort(key=lambda p: (p['real_x_rounded'], p['real_y_rounded']))

        # Horizontal asymptote cleanup:
        # remove long flat integer-y runs that have no local crossing support.
        # This targets false points like (..., y=2) on exponential tails.
        by_y: Dict[int, List[Dict]] = defaultdict(list)
        for p in prepared:
            try:
                yy = int(round(float(p['real_y_rounded'])))
            except (TypeError, ValueError):
                continue
            by_y[yy].append(p)

        to_drop = set()
        cleanup_runs_debug: List[Dict[str, Any]] = []
        for yy, group in by_y.items():
            # Need at least 1 point to evaluate singleton/short tail cleanup.
            if len(group) < 1:
                continue
            group_sorted = sorted(group, key=lambda q: float(q.get('real_x_rounded', 0.0)))
            # find contiguous x runs (step <= 1.1 on integer grid)
            runs: List[List[Dict]] = []
            cur = [group_sorted[0]]
            for q in group_sorted[1:]:
                if abs(float(q['real_x_rounded']) - float(cur[-1]['real_x_rounded'])) <= 1.1:
                    cur.append(q)
                else:
                    runs.append(cur)
                    cur = [q]
            runs.append(cur)

            for run_idx, run in enumerate(runs):
                unsupported = [
                    r for r in run
                    if not _supports_horizontal_level(float(r['real_x_rounded']), float(yy))
                ]
                # For long runs (>=3), drop when almost all points unsupported.
                should_drop_long = (
                    len(run) >= 3 and
                    len(unsupported) >= max(2, len(run) - 1)
                )

                # For short runs (len==2), also drop when both unsupported and
                # the run is isolated on one side of the x-axis support (tail-like).
                should_drop_short = False
                short_isolated_tail = False
                if len(run) == 2:
                    left_neighbor = runs[run_idx - 1][-1] if run_idx > 0 else None
                    right_neighbor = runs[run_idx + 1][0] if run_idx + 1 < len(runs) else None
                    run_left = float(run[0]['real_x_rounded'])
                    run_right = float(run[-1]['real_x_rounded'])
                    gap_left = (run_left - float(left_neighbor['real_x_rounded'])) if left_neighbor is not None else None
                    gap_right = (float(right_neighbor['real_x_rounded']) - run_right) if right_neighbor is not None else None

                    # Isolated tail when one side has no neighbor and the other
                    # side has a large jump, or both sides are separated by jumps.
                    isolated_tail = (
                        (left_neighbor is None and right_neighbor is None) or
                        (left_neighbor is None and gap_right is not None and gap_right >= 2.0) or
                        (right_neighbor is None and gap_left is not None and gap_left >= 2.0) or
                        (gap_left is not None and gap_left >= 2.0 and gap_right is not None and gap_right >= 2.0)
                    )

                    # Single-run fallback:
                    # compare this y-level run against points on other y-levels.
                    if isolated_tail and left_neighbor is None and right_neighbor is None:
                        other_x = []
                        for q in prepared:
                            try:
                                qx = float(q.get('real_x_rounded', 0.0))
                                qy = int(round(float(q.get('real_y_rounded', 0.0))))
                            except (TypeError, ValueError):
                                continue
                            if qy != yy:
                                other_x.append(qx)
                        if other_x:
                            min_other = min(other_x)
                            max_other = max(other_x)
                            # Tail-like if run is clearly detached from main x support.
                            detached_left = run_right <= (min_other - 2.0)
                            detached_right = run_left >= (max_other + 2.0)
                            short_isolated_tail = detached_left or detached_right
                        else:
                            short_isolated_tail = True
                    else:
                        short_isolated_tail = isolated_tail

                    # For 2-point detached tails we drop aggressively.
                    # Local support is optional here because exponential/
                    # rational tails can have noisy pseudo-support.
                    should_drop_short = short_isolated_tail

                # For singleton runs (len==1), drop when unsupported and clearly
                # detached from the main x-support on this y-level and others.
                should_drop_single = False
                if len(run) == 1 and len(unsupported) == 1:
                    point_x = float(run[0]['real_x_rounded'])
                    left_neighbor = runs[run_idx - 1][-1] if run_idx > 0 else None
                    right_neighbor = runs[run_idx + 1][0] if run_idx + 1 < len(runs) else None
                    gap_left = (
                        point_x - float(left_neighbor['real_x_rounded'])
                        if left_neighbor is not None else None
                    )
                    gap_right = (
                        float(right_neighbor['real_x_rounded']) - point_x
                        if right_neighbor is not None else None
                    )
                    isolated_same_y = (
                        (left_neighbor is None and right_neighbor is None) or
                        (left_neighbor is None and gap_right is not None and gap_right >= 2.0) or
                        (right_neighbor is None and gap_left is not None and gap_left >= 2.0) or
                        (gap_left is not None and gap_left >= 2.0 and gap_right is not None and gap_right >= 2.0)
                    )
                    if isolated_same_y:
                        other_x = []
                        for q in prepared:
                            try:
                                qx = float(q.get('real_x_rounded', 0.0))
                                qy = int(round(float(q.get('real_y_rounded', 0.0))))
                            except (TypeError, ValueError):
                                continue
                            if qy != yy:
                                other_x.append(qx)
                        if other_x:
                            min_other = min(other_x)
                            max_other = max(other_x)
                            detached_left = point_x <= (min_other - 2.0)
                            detached_right = point_x >= (max_other + 2.0)
                            should_drop_single = detached_left or detached_right
                        else:
                            should_drop_single = True

                if should_drop_long or should_drop_short or should_drop_single:
                    for r in run:
                        to_drop.add((int(r['real_x_rounded']), int(r['real_y_rounded'])))

                cleanup_runs_debug.append({
                    "y": int(yy),
                    "run_x": [int(run[0]['real_x_rounded']), int(run[-1]['real_x_rounded'])],
                    "run_len": len(run),
                    "unsupported": len(unsupported),
                    "drop_long": bool(should_drop_long),
                    "drop_short": bool(should_drop_short),
                    "drop_single": bool(should_drop_single),
                })

        if to_drop:
            prepared = [
                p for p in prepared
                if (int(p['real_x_rounded']), int(p['real_y_rounded'])) not in to_drop
            ]

        # Vertical asymptote cleanup (edge spike removal):
        # Detect edge points whose local slope jump is far larger than the
        # neighboring trend, typical for logarithmic/rational asymptote side.
        vertical_drop: set[Tuple[int, int]] = set()
        vertical_debug: List[Dict[str, Any]] = []
        by_x: Dict[int, List[Dict]] = defaultdict(list)
        for p in prepared:
            try:
                xx = int(round(float(p['real_x_rounded'])))
            except (TypeError, ValueError):
                continue
            by_x[xx].append(p)

        xs_sorted = sorted(by_x.keys())
        if len(xs_sorted) >= 3:
            median_y = {xk: float(np.median([float(q['real_y_rounded']) for q in by_x[xk]])) for xk in xs_sorted}

            def _slope(xa: int, xb: int) -> Optional[float]:
                if xa == xb:
                    return None
                return (median_y[xb] - median_y[xa]) / float(xb - xa)

            # left-edge check
            x0, x1, x2 = xs_sorted[0], xs_sorted[1], xs_sorted[2]
            s01 = _slope(x0, x1)
            s12 = _slope(x1, x2)
            drop_left = False
            if s01 is not None and s12 is not None:
                if abs(s01) >= max(3.0, 2.5 * abs(s12) + 1.0) and len(by_x[x0]) <= 2:
                    drop_left = True
                    for q in by_x[x0]:
                        vertical_drop.add((int(q['real_x_rounded']), int(q['real_y_rounded'])))
            vertical_debug.append({
                "side": "left",
                "x": int(x0),
                "s_edge": None if s01 is None else round(float(s01), 4),
                "s_next": None if s12 is None else round(float(s12), 4),
                "drop": drop_left,
            })

            # right-edge check
            xr0, xr1, xr2 = xs_sorted[-1], xs_sorted[-2], xs_sorted[-3]
            sr01 = _slope(xr0, xr1)
            sr12 = _slope(xr1, xr2)
            drop_right = False
            if sr01 is not None and sr12 is not None:
                if abs(sr01) >= max(3.0, 2.5 * abs(sr12) + 1.0) and len(by_x[xr0]) <= 2:
                    drop_right = True
                    for q in by_x[xr0]:
                        vertical_drop.add((int(q['real_x_rounded']), int(q['real_y_rounded'])))
            vertical_debug.append({
                "side": "right",
                "x": int(xr0),
                "s_edge": None if sr01 is None else round(float(sr01), 4),
                "s_next": None if sr12 is None else round(float(sr12), 4),
                "drop": drop_right,
            })

        if vertical_drop:
            prepared = [
                p for p in prepared
                if (int(p['real_x_rounded']), int(p['real_y_rounded'])) not in vertical_drop
            ]
            to_drop = set(to_drop).union(vertical_drop)

        if cleanup_runs_debug:
            print(f"[{'ReasoningIntersections'}] run-cleanup diagnostics:")
            for item in cleanup_runs_debug:
                print(
                    f"  y={item['y']} run={item['run_x']} len={item['run_len']} "
                    f"unsupported={item['unsupported']} "
                    f"drop_long={item['drop_long']} drop_short={item['drop_short']}"
                )
        if vertical_debug:
            print(f"[{'ReasoningIntersections'}] vertical-cleanup diagnostics:")
            for item in vertical_debug:
                print(
                    f"  side={item['side']} x={item['x']} "
                    f"s_edge={item['s_edge']} s_next={item['s_next']} drop={item['drop']}"
                )
        if to_drop:
            print(f"[{'ReasoningIntersections'}] dropped keys: {sorted(list(to_drop))}")

        print(
            f"[{'ReasoningIntersections'}] 保留整数去重点: {len(intersections)} -> {len(prepared)} "
            f"(rescued={rescued})"
        )
        self._last_reasoning_debug = {
            "input_count": len(intersections),
            "integer_seed_count": len(unique_points) - gap_filled,
            "rescued_count": rescued,
            "rescued_keys": rescued_keys,
            "gap_filled_count": gap_filled,
            "gap_filled_keys": gap_filled_keys,
            "dropped_keys": sorted(list(to_drop)),
            "cleanup_runs": cleanup_runs_debug,
            "vertical_cleanup": vertical_debug,
            "final_count": len(prepared),
        }
        return prepared

    def _filter_key_intersections(self, intersections: List[Dict], 
                                  coordinate_mapping: Dict, 
                                  image_shape: Tuple[int, int],
                                  collision_margin: int = 3) -> List[Dict]:
        """
        过滤交点，只保留关键点
        
        优先级：
        1. 保留X轴、Y轴交点，找极值点，删除中间冗余点
        2. 保留边界端点（不在最外侧网格上的）
        3. 边缘过滤（在最外侧网格且靠边的点）
        4. 稀疏化其余点
        
        Args:
            intersections: 所有交点列表
            coordinate_mapping: 坐标映射信息
            image_shape: 图像尺寸 (height, width)
            collision_margin: 碰撞检测时的安全边距（像素），标注框之间至少保持此距离
        
        Returns:
            过滤后的关键点列表
        """
        if not intersections:
            return intersections
        
        # 可视化路径也从轻量推理点集开始，避免与推理输入出现额外偏差
        intersections = self._prepare_reasoning_intersections(intersections)
        
        print(f"[IntersFilter] originalintersectiontext: {len(intersections)}")
        
        # 分类点：只保留 x、y 都是整数的点作为轴交点
        x_axis_points = [p for p in intersections 
                        if p['real_y_rounded'] == 0 and p['is_x_integer'] and p['is_y_integer']]
        y_axis_points = [p for p in intersections 
                        if p['real_x_rounded'] == 0 and p['is_x_integer'] and p['is_y_integer']]
        
        print(f"[IntersFilter] X-axisintersection: {len(x_axis_points)}, Y-axisintersection: {len(y_axis_points)}")
        
        # ==== 步骤1: 保留关键点 + 找极值点 + 删除中间冗余 ====
        key_points = set()
        
        # 保留所有坐标轴交点（只保留整数坐标）
        print(f"\n[Step1] KeepX-axistextY-axisintersection (textcoordinate)")
        for p in x_axis_points + y_axis_points:
            key_points.add((p['pixel_x'], p['pixel_y']))
            if p['is_x_integer'] and p['is_y_integer']:
                print(f"  ✓ textintersection: ({int(p['real_x_rounded'])}, {int(p['real_y_rounded'])})")
            else:
                print(f"  ○ textintersection: ({p['real_x_rounded']:.2f}, {p['real_y_rounded']:.2f})")
        
        # 查找极值点
        extrema_points = []
        if len(x_axis_points) >= 2:
            x_axis_sorted = sorted(x_axis_points, key=lambda p: p['real_x_rounded'])
            print(f"\n[Step1] textextrema point ({len(x_axis_sorted)-1}text)")
            
            for i in range(len(x_axis_sorted) - 1):
                x_min = x_axis_sorted[i]['real_x_rounded']
                x_max = x_axis_sorted[i+1]['real_x_rounded']
                
                # 找在这两个X轴交点之间的所有点
                between = [p for p in intersections 
                          if x_min < p['real_x_rounded'] < x_max]
                
                if between:
                    # 找y值最大和最小的点
                    y_values = [p['real_y_rounded'] for p in between]
                    max_y = max(y_values)
                    min_y = min(y_values)
                    
                    print(f"  text[{x_min}, {x_max}]: {len(between)}text")
                    
                    # 保留所有最大值点（可能有多个）
                    max_points = [p for p in between if p['real_y_rounded'] == max_y]
                    for p in max_points:
                        key_points.add((p['pixel_x'], p['pixel_y']))
                        extrema_points.append(p)
                        print(f"    ✓ text: ({p['real_x_rounded']}, {p['real_y_rounded']})")
                    
                    # 保留所有最小值点（可能有多个，且不同于最大值）
                    if min_y != max_y:
                        min_points = [p for p in between if p['real_y_rounded'] == min_y]
                        for p in min_points:
                            key_points.add((p['pixel_x'], p['pixel_y']))
                            extrema_points.append(p)
                            print(f"    ✓ text: ({p['real_x_rounded']}, {p['real_y_rounded']})")
                    
                    # 输出被删除的点
                    for p in between:
                        if (p['pixel_x'], p['pixel_y']) not in key_points:
                            print(f"    ✗ text: ({p['real_x_rounded']}, {p['real_y_rounded']})")
        
        # 查找间断函数点（X坐标相同的点对）
        discontinuity_points = []
        print(f"\n[Step1.5] text (Xcoordinatetext)")
        
        # 按X坐标分组
        x_groups = defaultdict(list)
        for p in intersections:
            x_groups[p['real_x_rounded']].append(p)
        
        # 找出有多个点的X坐标（表示间断/跳跃）
        # discontinuity_groups = {}  # {x_val: [points]}
        # for x_val, points in x_groups.items():
        #     if len(points) >= 2:
        #         print(f"  Founddiscontinuity pointtext: x={x_val}, {len(points)}text")
        #         discontinuity_groups[x_val] = points
        #         for p in points:
        #             discontinuity_points.append(p)
        #             print(f"    ✓ discontinuity point: ({p['real_x_rounded']}, {p['real_y_rounded']})")
        
        # # 如果间断点太多（≥6个），进行稀疏化
        # if len(discontinuity_points) >= 6:
        #     print(f"\n[Step1.6] discontinuity pointtext (text{len(discontinuity_points)}text ≥ 6)")
            
        #     # 获取所有间断点组的X坐标，排序
        #     x_vals = sorted(discontinuity_groups.keys())
            
        #     if len(x_vals) >= 3:
        #         x_min_disc = x_vals[0]
        #         x_max_disc = x_vals[-1]
        #         x_second_min = x_vals[1] if len(x_vals) >= 2 else None
        #         x_second_max = x_vals[-2] if len(x_vals) >= 2 else None
                
        #         print(f"  textXcoordinatetext: [{x_min_disc}, {x_max_disc}]")
        #         print(f"  textboundary, Keeptextboundary: [{x_second_min}, {x_second_max}]")
                
        #         # 保留的间断点
        #         kept_discontinuity = []
                
        #         # 删除边界上的间断点
        #         for x_val in [x_min_disc, x_max_disc]:
        #             if x_val in discontinuity_groups:
        #                 for p in discontinuity_groups[x_val]:
        #                     print(f"  ✗ textboundarydiscontinuity point: ({p['real_x_rounded']}, {p['real_y_rounded']})")
                
        #         # 保留次边界的间断点
        #         for x_val in [x_second_min, x_second_max]:
        #             if x_val and x_val in discontinuity_groups:
        #                 for p in discontinuity_groups[x_val]:
        #                     kept_discontinuity.append(p)
        #                     print(f"  ✓ Keeptextboundarydiscontinuity point: ({p['real_x_rounded']}, {p['real_y_rounded']})")
                
        #         # 对剩余的间断点组进行稀疏化（每隔一个保留一个）
        #         remaining_x_vals = [x for x in x_vals if x not in [x_min_disc, x_max_disc, x_second_min, x_second_max]]
                
        #         if remaining_x_vals:
        #             print(f"  text{len(remaining_x_vals)}textdiscontinuity pointtext, text (text2textKeep1text)")
        #             for i, x_val in enumerate(remaining_x_vals):
        #                 if i % 2 == 0:  # 每2个保留1个
        #                     for p in discontinuity_groups[x_val]:
        #                         kept_discontinuity.append(p)
        #                         print(f"  ✓ textKeepdiscontinuity point: ({p['real_x_rounded']}, {p['real_y_rounded']})")
        #                 else:
        #                     for p in discontinuity_groups[x_val]:
        #                         print(f"  ✗ textdiscontinuity point: ({p['real_x_rounded']}, {p['real_y_rounded']})")
                
        #         # 更新间断点列表
        #         discontinuity_points = kept_discontinuity
        #         print(f"  textafterKeep: {len(discontinuity_points)}textdiscontinuity point")
        
        # # 将保留的间断点加入关键点集合
        # for p in discontinuity_points:
        #     key_points.add((p['pixel_x'], p['pixel_y']))
        
        print(f"\n[Step1Summary] X-axisintersection: {len(x_axis_points)}, Y-axisintersection: {len(y_axis_points)}, extrema point: {len(extrema_points)}, discontinuity point: {len(discontinuity_points)}")
        
        # ==== 步骤2: 保留边界端点（严格模式：两个维度都在边界）====
        boundary_points = set()
        
        print(f"\n[Step2] Keepboundarytext (text: textboundarytext)")
        
        if intersections:
            x_values = sorted(set([p['real_x_rounded'] for p in intersections]))
            y_values = sorted(set([p['real_y_rounded'] for p in intersections]))
            
            x_min_val = x_values[0]
            x_max_val = x_values[-1]
            y_min_val = y_values[0]
            y_max_val = y_values[-1]
            
            print(f"  boundarytext: X[{x_min_val}, {x_max_val}], Y[{y_min_val}, {y_max_val}]")
            
            # 检查网格信息
            v_grids = coordinate_mapping.get('v_grid_x_pixels', [])
            h_grids = coordinate_mapping.get('h_grid_y_pixels', [])
            
            if v_grids and h_grids:
                leftmost_grid = min(v_grids)
                rightmost_grid = max(v_grids)
                topmost_grid = min(h_grids)
                bottommost_grid = max(h_grids)
                
                print(f"  text: X[{leftmost_grid:.0f}, {rightmost_grid:.0f}], Y[{topmost_grid:.0f}, {bottommost_grid:.0f}]")
                
                # 严格模式：只保留同时在x和y边界上的点（四个角）
                for p in intersections:
                    x = p['real_x_rounded']
                    y = p['real_y_rounded']
                    px = p['pixel_x']
                    py = p['pixel_y']
                    
                    # 检查是否同时在x边界和y边界上
                    x_on_boundary = (x == x_min_val or x == x_max_val)
                    y_on_boundary = (y == y_min_val or y == y_max_val)
                    
                    if x_on_boundary and y_on_boundary:
                        # 还需要检查是否在最外侧网格上（如果在，则不保留）
                        on_outermost_x = (abs(px - leftmost_grid) < 3 or abs(px - rightmost_grid) < 3)
                        on_outermost_y = (abs(py - topmost_grid) < 3 or abs(py - bottommost_grid) < 3)
                        
                        if not (on_outermost_x and on_outermost_y):
                            boundary_points.add((px, py))
                            boundary_type = []
                            if x == x_min_val:
                                boundary_type.append('x_min')
                            if x == x_max_val:
                                boundary_type.append('x_max')
                            if y == y_min_val:
                                boundary_type.append('y_min')
                            if y == y_max_val:
                                boundary_type.append('y_max')
                            print(f"  ✓ Keepboundarytext: ({x}, {y}) - {'+'.join(boundary_type)}")
                        else:
                            print(f"  ✗ text: ({x}, {y})")
        
        # ==== 步骤3: 边缘过滤 ====
        print(f"\n[Step3] edgefilter (text)")
        
        filtered_out = 0
        if len(intersections) >= 2 and v_grids and h_grids:
            leftmost_grid = min(v_grids)
            rightmost_grid = max(v_grids)
            topmost_grid = min(h_grids)
            bottommost_grid = max(h_grids)
            
            # 计算网格间距
            v_spacing = np.median(np.diff(sorted(v_grids))) if len(v_grids) > 1 else 50
            h_spacing = np.median(np.diff(sorted(h_grids))) if len(h_grids) > 1 else 50
            grid_spacing = (v_spacing + h_spacing) / 2
            
            threshold = 1.5 * grid_spacing
            h_img, w_img = image_shape
            
            print(f"  texthreshold: {threshold:.0f}px (1.5×text)")
            
            for p in intersections:
                px, py = p['pixel_x'], p['pixel_y']
                
                # 检查是否在最外侧网格上
                on_outermost = (abs(px - leftmost_grid) < 3 or 
                               abs(px - rightmost_grid) < 3 or
                               abs(py - topmost_grid) < 3 or
                               abs(py - bottommost_grid) < 3)
                
                if on_outermost:
                    # 检查距离边界
                    dist_to_edge = min(px, w_img - px, py, h_img - py)
                    
                    if dist_to_edge < threshold:
                        # 如果是关键点，不过滤
                        if (px, py) in key_points or (px, py) in boundary_points:
                            print(f"  ○ text: ({p['real_x_rounded']}, {p['real_y_rounded']}) text{dist_to_edge:.0f}px")
                        else:
                            filtered_out += 1
                            print(f"  ✗ edgefilter: ({p['real_x_rounded']}, {p['real_y_rounded']}) text{dist_to_edge:.0f}px < {threshold:.0f}px")
        
        print(f"[Step3Summary] edgefiltertext: {filtered_out}text")
        
        # ==== 应用过滤，收集通过边缘过滤的点 ====
        protected_points = key_points.union(boundary_points)
        
        # 先应用边缘过滤，收集所有通过过滤的点
        candidates = []
        for p in intersections:
            px, py = p['pixel_x'], p['pixel_y']
            
            # 检查是否需要边缘过滤
            # if v_grids and h_grids and len(intersections) >= 2:
            #     on_outermost = (abs(px - min(v_grids)) < 3 or 
            #                    abs(px - max(v_grids)) < 3 or
            #                    abs(py - min(h_grids)) < 3 or
            #                    abs(py - max(h_grids)) < 3)
                
            #     if on_outermost:
            #         dist_to_edge = min(px, image_shape[1] - px, py, image_shape[0] - py)
            #         v_spacing = np.median(np.diff(sorted(v_grids))) if len(v_grids) > 1 else 50
            #         h_spacing = np.median(np.diff(sorted(h_grids))) if len(h_grids) > 1 else 50
            #         threshold = 1.5 * (v_spacing + h_spacing) / 2
                    
            #         if dist_to_edge < threshold and (px, py) not in protected_points:
            #             continue  # 过滤掉
            
            candidates.append(p)
        
        # ==== 步骤4: 基于标注框碰撞检测的智能稀疏化 ====
        print(f"\n[Step4] text")
        
        print(f"  candidatetext: {len(candidates)}")
        print(f"  text: {len([p for p in candidates if (p['pixel_x'], p['pixel_y']) in protected_points])}text")
        print(f"  text: text (text{collision_margin}pxtext), textKeeptext")
        
        # 获取字体参数（与visualize_result中一致）
        # 使用自适应字体大小计算
        x_tick_labels = coordinate_mapping.get('x_tick_labels', [])
        y_tick_labels = coordinate_mapping.get('y_tick_labels', [])
        font_scale = self._calculate_adaptive_font_scale(x_tick_labels, y_tick_labels, image_shape)
        thickness = max(1, int(font_scale * 2))
        padding = int(font_scale * 8)
        
        print(f"  fontparameter: font_scale={font_scale:.2f}, thickness={thickness}, padding={padding}")
        
        # === 步骤4.1: 为每个点计算标注框 ===
        print(f"\n  [Step4.1] text (textpositiontext)")
        point_bboxes = {}  # {(px, py): bbox}
        point_dict = {}  # {(px, py): point_data}
        point_label_info = {}  # {(px, py): label_info} 保存标注信息供后续使用
        
        for p in candidates:
            px, py = p['pixel_x'], p['pixel_y']
            
            # 使用统一的函数计算标注位置和边界框
            label_info = self._calculate_label_position_and_bbox(
                p, image_shape, font_scale, thickness, padding
            )
            
            point_bboxes[(px, py)] = label_info['bbox']
            point_dict[(px, py)] = p
            point_label_info[(px, py)] = label_info
        
        print(f"  textComplete: {len(point_bboxes)}text")
        
        # === 步骤4.2: 构建碰撞图 ===
        print(f"\n  [Step4.2] text (text{collision_margin}pxtext)")
        collision_graph = defaultdict(set)  # {(px, py): set of colliding (px, py)}
        
        point_keys = list(point_bboxes.keys())
        collision_count = 0
        
        for i, p1 in enumerate(point_keys):
            for p2 in point_keys[i+1:]:
                if self._check_bbox_collision(point_bboxes[p1], point_bboxes[p2], margin=collision_margin):
                    collision_graph[p1].add(p2)
                    collision_graph[p2].add(p1)
                    collision_count += 1
        
        print(f"  text: {collision_count}text")
        print(f"  text: {len([p for p in point_keys if p in collision_graph])}text")
        
        # === 步骤4.3: 迭代删除碰撞点 ===
        print(f"\n  [Step4.3] iterationtext")
        
        remaining_points = set(point_keys)  # 当前保留的点
        iteration = 0
        
        while True:
            # 检查是否还有碰撞
            has_collision = False
            for p in remaining_points:
                if p in collision_graph:
                    # 检查碰撞点是否还在remaining中
                    colliding = collision_graph[p].intersection(remaining_points)
                    if colliding:
                        has_collision = True
                        break
            
            if not has_collision:
                print(f"  iteration{iteration}: text, Complete")
                break
            
            iteration += 1
            print(f"\n  === iteration{iteration} ===")
            
            # 计算每个有碰撞的点的"冲突分数"
            conflict_scores = {}
            
            for p in remaining_points:
                if p in collision_graph:
                    colliding = collision_graph[p].intersection(remaining_points)
                    if not colliding:
                        continue
                    
                    # 如果是保护点，跳过（永不删除）
                    if p in protected_points:
                        continue
                    
                    # 计算冲突分数 = 碰撞数量 / 局部密度
                    # 碰撞越多、密度越大（距离越小），分数越高，越应该删除
                    num_collisions = len(colliding)
                    
                    # 计算局部密度（使用所有当前保留的点）
                    remaining_point_list = [point_dict[rp] for rp in remaining_points]
                    local_density = self._calculate_local_density(
                        point_dict[p], remaining_point_list, k=5
                    )
                    
                    # 避免除零
                    if local_density < 1.0:
                        local_density = 1.0
                    
                    conflict_score = num_collisions / local_density
                    conflict_scores[p] = conflict_score
            
            # 如果没有可删除的点（都是保护点），强制删除碰撞最多的非保护点
            if not conflict_scores:
                print(f"  Warning: text, text")
                break
            
            # 找出冲突分数最高的点（最应该删除的）
            point_to_remove = max(conflict_scores.items(), key=lambda x: x[1])
            p_remove, score = point_to_remove
            
            # 删除这个点
            remaining_points.remove(p_remove)
            
            # 确定点类型
            p_data = point_dict[p_remove]
            colliding = collision_graph[p_remove].intersection(remaining_points)
            
            remaining_point_list = [point_dict[rp] for rp in remaining_points]
            local_density = self._calculate_local_density(
                p_data, remaining_point_list + [p_data], k=5
            )
            
            print(f"  ✗ text: ({p_data['real_x_rounded']}, {p_data['real_y_rounded']}) "
                  f"| 碰撞数={len(colliding)} | 局部密度={local_density:.1f}px | 冲突分数={score:.4f}")
        
        print(f"\n  [Step4.3Sum] iteration{iteration}text, text{len(candidates) - len(remaining_points)}text")
        
        # === 步骤4.4: 输出保留的点 ===
        print(f"\n  [Step4.4] Keeptext:")
        filtered_intersections = []
        
        for p_key in sorted(remaining_points, key=lambda k: point_dict[k]['real_x_rounded']):
            p = point_dict[p_key]
            filtered_intersections.append(p)
        
            # 确定点类型
            point_type = "普通点"
            if p_key in protected_points:
                if p_key in key_points:
                    if p['real_y_rounded'] == 0:
                        point_type = "X轴交点（保护）"
                    elif p['real_x_rounded'] == 0:
                        point_type = "Y轴交点（保护）"
                    elif p in discontinuity_points:
                        point_type = "间断点（保护）"
                    else:
                        point_type = "极值点（保护）"
                elif p_key in boundary_points:
                    point_type = "边界点（保护）"
            
            print(f"    ✓ {point_type}: ({p['real_x_rounded']}, {p['real_y_rounded']})")
        
        print(f"[Step4Summary] text: {len(candidates)}text -> Keep{len(filtered_intersections)}text")
        
        print(f"\n[FinalResult] original{len(intersections)}text -> Keep{len(filtered_intersections)}text")
        
        return filtered_intersections
    
    def _merge_nearby_points(self, points: List[Dict], tolerance: int) -> List[Dict]:
        """
        合并距离很近的交点
        
        Args:
            points: 交点列表
            tolerance: 容差
        
        Returns:
            合并后的交点列表
        """
        if len(points) == 0:
            return points
        
        merged = []
        used = set()
        
        for i, pt1 in enumerate(points):
            if i in used:
                continue
            
            # 找到所有与pt1接近的点
            group = [pt1]
            for j, pt2 in enumerate(points):
                if j <= i or j in used:
                    continue
                
                dist = np.sqrt(
                    (pt1['pixel_x'] - pt2['pixel_x']) ** 2 + 
                    (pt1['pixel_y'] - pt2['pixel_y']) ** 2
                )
                
                if dist <= tolerance:
                    group.append(pt2)
                    used.add(j)
            
            # 合并这组点（取平均）
            avg_real_x = np.mean([pt['real_x'] for pt in group])
            avg_real_y = np.mean([pt['real_y'] for pt in group])
            
            # 重新使用阈值判断是否应该近似为整数
            real_x_approx, is_x_integer = self.approximate_to_integer(avg_real_x)
            real_y_approx, is_y_integer = self.approximate_to_integer(avg_real_y)
            
            avg_point = {
                'pixel_x': int(np.mean([pt['pixel_x'] for pt in group])),
                'pixel_y': int(np.mean([pt['pixel_y'] for pt in group])),
                'real_x': avg_real_x,
                'real_y': avg_real_y,
                'real_x_rounded': real_x_approx,
                'real_y_rounded': real_y_approx,
                'is_x_integer': is_x_integer,
                'is_y_integer': is_y_integer,
                'lines_count': max([pt['lines_count'] for pt in group]),
                'line_types': list(set(sum([pt['line_types'] for pt in group], [])))
            }
            
            merged.append(avg_point)
            used.add(i)
        
        return merged
    
    def visualize_grid_coordinates(self, image: np.ndarray, 
                                   grid_h_lines: List, grid_v_lines: List,
                                   x_tick_labels: List[Dict], y_tick_labels: List[Dict],
                                   axes_info: Dict) -> np.ndarray:
        """
        可视化网格线坐标映射
        
        在图上显示：
        1. 每条垂直网格线及其对应的x坐标值
        2. 每条水平网格线及其对应的y坐标值
        3. OCR识别到的刻度标签位置
        
        Args:
            image: 原始图像
            grid_h_lines: 水平网格线
            grid_v_lines: 垂直网格线
            x_tick_labels: X轴刻度标签
            y_tick_labels: Y轴刻度标签
            axes_info: 坐标轴信息
        
        Returns:
            标注后的图像
        """
        result = image.copy()
        h, w = image.shape[:2]
        
        # 从映射中获取网格线坐标值
        if not self.coordinate_mapping:
            return result
        
        # ===== 绘制垂直网格线和坐标值 =====
        if 'v_grid_x_pixels' in self.coordinate_mapping and 'v_grid_x_values' in self.coordinate_mapping:
            v_pixels = self.coordinate_mapping['v_grid_x_pixels']
            v_values = self.coordinate_mapping['v_grid_x_values']
            v_lines = self.coordinate_mapping.get('v_grid_lines')
            
            for i, (px, val) in enumerate(zip(v_pixels, v_values)):
                # 检查是否是Y轴
                is_y_axis = False
                if 'y_axis_line' in axes_info and axes_info['y_axis_line'] is not None:
                    ref_y = float(self.coordinate_mapping.get('v_grid_reference_y', h / 2))
                    y_axis_x = self._line_x_at_y(axes_info['y_axis_line'], ref_y)
                    if abs(px - y_axis_x) < 10:
                        is_y_axis = True
                
                # 绘制网格线
                color = (255, 0, 0) if is_y_axis else (150, 150, 150)  # Y轴蓝色，其他灰色
                thickness = 3 if is_y_axis else 1
                if isinstance(v_lines, list) and i < len(v_lines):
                    x1, y1, x2, y2 = [int(round(v)) for v in v_lines[i]]
                    cv2.line(result, (x1, y1), (x2, y2), color, thickness)
                else:
                    cv2.line(result, (int(px), 0), (int(px), h), color, thickness)
                
                # 标注坐标值（整数格式）
                label = f'x={int(round(val))}'
                if is_y_axis:
                    label = f'Y-axis (x={int(round(val))})'
                
                # 文字位置：顶部
                text_y = 30 if i % 2 == 0 else 60
                cv2.putText(result, label, (int(px) - 30, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # ===== 绘制水平网格线和坐标值 =====
        if 'h_grid_y_pixels' in self.coordinate_mapping and 'h_grid_y_values' in self.coordinate_mapping:
            h_pixels = self.coordinate_mapping['h_grid_y_pixels']
            h_values = self.coordinate_mapping['h_grid_y_values']
            h_lines = self.coordinate_mapping.get('h_grid_lines')
            
            for i, (py, val) in enumerate(zip(h_pixels, h_values)):
                # 检查是否是X轴
                is_x_axis = False
                if 'x_axis_line' in axes_info and axes_info['x_axis_line'] is not None:
                    ref_x = float(self.coordinate_mapping.get('h_grid_reference_x', w / 2))
                    x_axis_y = self._line_y_at_x(axes_info['x_axis_line'], ref_x)
                    if abs(py - x_axis_y) < 10:
                        is_x_axis = True
                
                # 绘制网格线
                color = (0, 0, 255) if is_x_axis else (150, 150, 150)  # X轴红色，其他灰色
                thickness = 3 if is_x_axis else 1
                if isinstance(h_lines, list) and i < len(h_lines):
                    x1, y1, x2, y2 = [int(round(v)) for v in h_lines[i]]
                    cv2.line(result, (x1, y1), (x2, y2), color, thickness)
                else:
                    cv2.line(result, (0, int(py)), (w, int(py)), color, thickness)
                
                # 标注坐标值（整数格式）
                label = f'y={int(round(val))}'
                if is_x_axis:
                    label = f'X-axis (y={int(round(val))})'
                
                # 文字位置：左侧
                text_x = 10 if i % 2 == 0 else 80
                cv2.putText(result, label, (text_x, int(py) + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # ===== 标注OCR识别到的刻度标签 =====
        # X轴刻度标签（绿色框）
        for label in x_tick_labels:
            print(f"x_tick_labels: {label}")
            x = int(label['center_x'])
            y = int(label['center_y'])
            value = label['value']
            
            # 绘制中心点
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)
            
            # 绘制边界框（仅当标签未经过旋转变换时）
            # 如果标签包含 original_center_x，说明经过了旋转变换，bbox已不准确，跳过绘制
            if 'original_center_x' not in label:
                if 'left' in label:
                    # 直接包含坐标字段
                    x1, y1 = int(label['left']), int(label['top'])
                    x2, y2 = int(label['right']), int(label['bottom'])
                    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif 'bbox' in label:
                    # bbox嵌套在字典里
                    bbox = label['bbox']
                    x1, y1 = int(bbox['left']), int(bbox['top'])
                    x2, y2 = int(bbox['right']), int(bbox['bottom'])
                    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 标注值（显示OCR识别的原始值）
            cv2.putText(result, f'OCR:{value}', (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Y轴刻度标签（黄色框）
        for label in y_tick_labels:
            print(f"y_tick_labels: {label}")
            x = int(label['center_x'])
            y = int(label['center_y'])
            value = label['value']
            
            # 绘制中心点
            cv2.circle(result, (x, y), 5, (0, 255, 255), -1)
            
            # 绘制边界框（仅当标签未经过旋转变换时）
            # 如果标签包含 original_center_x，说明经过了旋转变换，bbox已不准确，跳过绘制
            if 'original_center_x' not in label:
                if 'left' in label:
                    # 直接包含坐标字段
                    x1, y1 = int(label['left']), int(label['top'])
                    x2, y2 = int(label['right']), int(label['bottom'])
                    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
                elif 'bbox' in label:
                    # bbox嵌套在字典里
                    bbox = label['bbox']
                    x1, y1 = int(bbox['left']), int(bbox['top'])
                    x2, y2 = int(bbox['right']), int(bbox['bottom'])
                    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # 标注值（显示OCR识别的原始值）
            cv2.putText(result, f'OCR:{value}', (x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # ===== 标注原点 =====
        if 'origin_x' in axes_info and 'origin_y' in axes_info:
            ox = int(axes_info['origin_x'])
            oy = int(axes_info['origin_y'])
            cv2.circle(result, (ox, oy), 10, (255, 0, 255), -1)
            cv2.circle(result, (ox, oy), 12, (255, 255, 255), 2)
            cv2.putText(result, 'Origin(0,0)', (ox + 15, oy - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # 添加图例
        legend_y = h - 100
        cv2.putText(result, 'Legend:', (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, '- Red line: X-axis', (10, legend_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(result, '- Blue line: Y-axis', (10, legend_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(result, '- Green box: X tick labels', (10, legend_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(result, '- Yellow box: Y tick labels', (250, legend_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return result
    
    def _draw_text_with_background(self, image: np.ndarray, text: str, 
                                   position: Tuple[int, int], font_scale: float,
                                   text_color: Tuple[int, int, int],
                                   bg_color: Tuple[int, int, int],
                                   thickness: int = 2, padding: int = 5,
                                   alpha: float = 0.7) -> np.ndarray:
        """
        绘制带半透明背景的文字（调用统一工具函数）
        
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
        return draw_text_with_background(
            image, text, position, font_scale, text_color, bg_color,
            thickness, padding, alpha
        )
    
    def _calculate_label_position_and_bbox(self, point: Dict, image_shape: Tuple[int, int],
                                          font_scale: float = 0.5, thickness: int = 2, 
                                          padding: int = 5) -> Dict:
        """
        统一计算标注的位置和边界框（调用统一工具函数）
        
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
        return calculate_label_position_and_bbox(
            point, image_shape, font_scale, thickness, padding
        )
    
    def _check_bbox_collision(self, bbox1: Tuple[int, int, int, int],
                             bbox2: Tuple[int, int, int, int],
                             margin: int = 0) -> bool:
        """
        检测两个边界框是否碰撞（部分重叠或距离小于margin）
        
        Args:
            bbox1: 第一个边界框 (x1, y1, x2, y2)
            bbox2: 第二个边界框 (x1, y1, x2, y2)
            margin: 安全边距（像素），两个框之间至少要保持此距离
        
        Returns:
            True表示碰撞（重叠或距离<margin），False表示不碰撞
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 如果需要安全边距，将每个bbox向外扩展margin/2
        if margin > 0:
            half_margin = margin / 2
            x1_1 -= half_margin
            y1_1 -= half_margin
            x2_1 += half_margin
            y2_1 += half_margin
            
            x1_2 -= half_margin
            y1_2 -= half_margin
            x2_2 += half_margin
            y2_2 += half_margin
        
        # 检测是否有重叠
        # 如果两个矩形不重叠，则满足以下条件之一：
        # - bbox1完全在bbox2的左边：x2_1 < x1_2
        # - bbox1完全在bbox2的右边：x1_1 > x2_2
        # - bbox1完全在bbox2的上边：y2_1 < y1_2
        # - bbox1完全在bbox2的下边：y1_1 > y2_2
        
        # 因此，有重叠的条件是这些条件都不满足
        return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)
    
    def _calculate_local_density(self, point: Dict, all_points: List[Dict], k: int = 5) -> float:
        """
        计算点的局部密度（到最近K个邻居的平均距离）
        
        Args:
            point: 当前点
            all_points: 所有点的列表
            k: 邻居数量
        
        Returns:
            局部密度值（距离越大表示越稀疏）
        """
        px, py = point['pixel_x'], point['pixel_y']
        
        # 计算到所有其他点的距离
        distances = []
        for other in all_points:
            if other == point:
                continue
            ox, oy = other['pixel_x'], other['pixel_y']
            dist = np.sqrt((px - ox)**2 + (py - oy)**2)
            distances.append(dist)
        
        if not distances:
            return float('inf')  # 如果只有一个点，认为是无限稀疏
        
        # 取最近的K个邻居的平均距离
        k_actual = min(k, len(distances))
        nearest_k = sorted(distances)[:k_actual]
        avg_distance = np.mean(nearest_k)
        
        return avg_distance
    
    def _calculate_adaptive_font_scale(self, x_tick_labels: List[Dict], 
                                       y_tick_labels: List[Dict],
                                       image_shape: Tuple[int, int]) -> float:
        """
        根据刻度标签大小自适应计算字体大小（调用统一工具函数）
        
        Args:
            x_tick_labels: X轴刻度标签
            y_tick_labels: Y轴刻度标签
            image_shape: 图像尺寸 (height, width)
        
        Returns:
            字体缩放系数
        """
        return calculate_adaptive_font_scale(
            x_tick_labels, y_tick_labels, image_shape
        )
    
    def visualize_result(self, image: np.ndarray, axes_info: Dict, 
                        intersections: List[Dict]) -> np.ndarray:
        """
        可视化标注结果（优化版）
        
        Args:
            image: 原始图像
            axes_info: 坐标轴信息
            intersections: 交点列表
        
        Returns:
            标注后的图像
        """
        result = image.copy()
        h, w = image.shape[:2]
        
        # 计算自适应字体大小
        x_tick_labels = self.coordinate_mapping.get('x_tick_labels', [])
        y_tick_labels = self.coordinate_mapping.get('y_tick_labels', [])
        font_scale = self._calculate_adaptive_font_scale(x_tick_labels, y_tick_labels, (h, w))
        
        print(f"[Visualization] Adaptive fonttext: {font_scale:.2f}")
        
        # 标注X轴（红色粗线）- 使用网格线绘制
        if 'x_axis_line' in axes_info and axes_info['x_axis_line'] is not None:
            x1, y1, x2, y2 = axes_info['x_axis_line']
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
            # X-axis标签放在右侧
            cv2.putText(result, 'X-axis', (w - 100, int((y1+y2)/2) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif 'x_axis_y' in axes_info:
            # 后备方案：使用y坐标绘制
            y = int(axes_info['x_axis_y'])
            cv2.line(result, (0, y), (w, y), (0, 0, 255), 3)
            # X-axis标签放在右侧
            cv2.putText(result, 'X-axis', (w - 100, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 标注Y轴（蓝色粗线）- 使用网格线绘制
        if 'y_axis_line' in axes_info and axes_info['y_axis_line'] is not None:
            x1, y1, x2, y2 = axes_info['y_axis_line']
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
            cv2.putText(result, 'Y-axis', (int((x1+x2)/2) + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        elif 'y_axis_x' in axes_info:
            # 后备方案：使用x坐标绘制
            x = int(axes_info['y_axis_x'])
            cv2.line(result, (x, 0), (x, h), (255, 0, 0), 3)
            cv2.putText(result, 'Y-axis', (x + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # 标注原点（绿色圆点）
        if 'origin_x' in axes_info and 'origin_y' in axes_info:
            ox = int(axes_info['origin_x'])
            oy = int(axes_info['origin_y'])
            cv2.circle(result, (ox, oy), 10, (0, 255, 0), -1)
            cv2.circle(result, (ox, oy), 12, (255, 255, 255), 2)
            cv2.putText(result, 'Origin(0,0)', (ox + 15, oy - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 标注交点（优化版：带背景、自适应字体、新配色）
        # 使用统一的位置计算函数，确保与碰撞检测一致
        thickness = max(1, int(font_scale * 2))
        padding = int(font_scale * 8)
        
        for i, pt in enumerate(intersections):
            px = pt['pixel_x']
            py = pt['pixel_y']
            
            # 画点（青色外圈 + 绿色内圈）
            cv2.circle(result, (px, py), 7, (255, 255, 0), -1)  # 青色外圈
            cv2.circle(result, (px, py), 5, (0, 255, 0), -1)    # 绿色内圈
            cv2.circle(result, (px, py), 8, (255, 255, 255), 2)  # 白色边框
            
            # 使用统一函数计算标注位置（确保与碰撞检测时的逻辑完全一致）
            label_info = self._calculate_label_position_and_bbox(
                pt, (h, w), font_scale, thickness, padding
            )
            
            label = label_info['text']
            text_pos = label_info['text_pos']
            
            # 绘制带背景的文字
            # 使用深橙色文字 + 白色半透明背景（高对比度，容易阅读）
            self._draw_text_with_background(
                result, label, text_pos,
                font_scale=font_scale,
                text_color=(20, 100, 220),      # 深橙色文字（BGR: B=20, G=100, R=220）
                bg_color=(255, 255, 255),       # 白色背景
                thickness=thickness,
                padding=padding,
                alpha=0.75                       # 75%透明度
            )
        
        return result
    
    def transform_tick_labels_with_rotation(self, tick_labels: List[Dict], 
                                            rotation_matrix: np.ndarray,
                                            original_size: Tuple[int, int],
                                            rotated_image_shape: Tuple[int, int],
                                            crop_offset: Tuple[int, int] = (0, 0),
                                            perspective_matrix: np.ndarray = None) -> List[Dict]:
        """
        对刻度标签的坐标进行变换（透视变换 + 旋转变换）
        
        Args:
            tick_labels: 原始刻度标签列表（包含像素坐标）
            rotation_matrix: 旋转矩阵 (2x3)
            original_size: 原始图像尺寸 (width, height)
            rotated_image_shape: 旋转后图像尺寸 (height, width)
            crop_offset: 裁剪偏移量 (x, y)，默认为 (0, 0)
            perspective_matrix: 透视变换矩阵 (3x3)，如果为None则跳过透视变换
        
        Returns:
            变换后的刻度标签列表
        """
        if not tick_labels:
            return tick_labels
        
        print(f"[CoordTransform] originaltextcount: {len(tick_labels)}, crop offset: {crop_offset}")
        print(f"[CoordTransform] perspective transform: {'enabled' if perspective_matrix is not None else 'disabled'}")
        print(f"[CoordTransform] rotation transform: {'enabled' if rotation_matrix is not None else 'disabled'}")
        
        transformed_labels = []
        for label in tick_labels:
            # 获取中心点坐标
            center_x = label.get('center_x', 0)
            center_y = label.get('center_y', 0)
            
            new_center_x = center_x
            new_center_y = center_y
            
            # 步骤1: 应用透视变换（如果有）
            if perspective_matrix is not None:
                point_homo = np.array([center_x, center_y, 1])
                transformed_homo = perspective_matrix @ point_homo
                new_center_x = transformed_homo[0] / transformed_homo[2]
                new_center_y = transformed_homo[1] / transformed_homo[2]
            
            # 步骤2: 应用旋转变换（如果有）
            if rotation_matrix is not None:
                point = np.array([new_center_x, new_center_y, 1])
                transformed_point = rotation_matrix @ point
                new_center_x = transformed_point[0]
                new_center_y = transformed_point[1]
            
            # 步骤3: 减去裁剪偏移量（旋转后进行了裁剪，坐标需要调整）
            new_center_x = new_center_x - crop_offset[0]
            new_center_y = new_center_y - crop_offset[1]
            
            # 创建新的标签
            transformed_label = {
                'text': label.get('text', ''),
                'value': label.get('value', 0.0),
                'center_x': new_center_x,
                'center_y': new_center_y,
                # 保留原始bbox信息（虽然可能不准确，但保留以备用）
                'bbox': label.get('bbox', {}),
                'original_center_x': center_x,  # 保存原始坐标
                'original_center_y': center_y
            }
            
            transformed_labels.append(transformed_label)
            
            print(f"  label'{label['text']}': ({center_x:.1f}, {center_y:.1f}) -> ({new_center_x:.1f}, {new_center_y:.1f})")
        
        print(f"[CoordTransform] textComplete")
        return transformed_labels
    
    def inverse_transform_points(self, points: List[Dict], 
                                 rotation_matrix: np.ndarray,
                                 original_size: Tuple[int, int],
                                 rotated_image_shape: Tuple[int, int],
                                 crop_offset: Tuple[int, int] = (0, 0),
                                 perspective_matrix: np.ndarray = None,
                                 ai_transform_params: dict = None) -> List[Dict]:
        """
        对交点坐标进行逆变换（逆旋转 + 逆透视 + AI增强逆变换，从校正图映射回原图）
        
        Args:
            points: 校正图上的交点列表
            rotation_matrix: 旋转矩阵 (2x3)，包含平移到新中心的调整
            original_size: 原始图像尺寸 (width, height)
            rotated_image_shape: 旋转后图像尺寸 (height, width)
            crop_offset: 裁剪偏移量 (x, y)，默认为 (0, 0)
            perspective_matrix: 透视变换矩阵 (3x3)，如果为None则跳过透视逆变换
            ai_transform_params: AI增强变换参数，如果为None则跳过AI逆变换
        
        Returns:
            原图坐标系下的交点列表
        """
        if not points:
            return points
        
        print(f"[InvTransform] intersectioncount: {len(points)}, crop offset: {crop_offset}")
        print(f"[InvTransform] text: {'enabled' if rotation_matrix is not None else 'disabled'}")
        print(f"[InvTransform] text: {'enabled' if perspective_matrix is not None else 'disabled'}")
        print(f"[InvTransform] AIEnhancetext: {'enabled' if ai_transform_params is not None else 'disabled'}")
        
        # 准备逆变换矩阵
        M_inv = None
        if rotation_matrix is not None:
            # 计算旋转逆变换矩阵
            # 对于仿射变换矩阵，我们需要构造完整的3x3矩阵然后求逆
            M_full = np.vstack([rotation_matrix, [0, 0, 1]])
            M_inv_full = np.linalg.inv(M_full)
            M_inv = M_inv_full[:2, :]  # 取前两行作为逆仿射矩阵
        
        P_inv = None
        if perspective_matrix is not None:
            # 计算透视逆变换矩阵
            P_inv = np.linalg.inv(perspective_matrix)
        
        transformed_points = []
        for pt in points:
            # 获取校正图上的像素坐标
            px = pt['pixel_x']
            py = pt['pixel_y']
            
            # 步骤1: 加上裁剪偏移量，得到旋转后大图像上的坐标
            px_current = px + crop_offset[0]
            py_current = py + crop_offset[1]
            
            # 步骤2: 应用旋转逆变换（如果有）
            if M_inv is not None:
                point_homo = np.array([px_current, py_current, 1])
                inv_transformed = M_inv @ point_homo
                px_current = inv_transformed[0]
                py_current = inv_transformed[1]
            
            # 步骤3: 应用透视逆变换（如果有）
            if P_inv is not None:
                point_homo = np.array([px_current, py_current, 1])
                inv_transformed_homo = P_inv @ point_homo
                px_current = inv_transformed_homo[0] / inv_transformed_homo[2]
                py_current = inv_transformed_homo[1] / inv_transformed_homo[2]
            
            # 步骤4: 应用AI增强逆变换（如果有）
            if ai_transform_params is not None:
                scale = ai_transform_params['scale']
                crop_x = ai_transform_params['crop_x']
                crop_y = ai_transform_params['crop_y']
                ai_w, ai_h = ai_transform_params['ai_output_size']
                orig_w, orig_h = ai_transform_params['original_size']
                
                # 4.1: 加上裁剪偏移，得到resize缩放后图像的坐标
                px_step3_inv = px_current + crop_x
                py_step3_inv = py_current + crop_y
                
                # 4.2: 除以resize缩放比例，得到AI输出图像的坐标
                px_step2_inv = px_step3_inv / scale
                py_step2_inv = py_step3_inv / scale
                
                # 4.3: 加回宽度居中偏移
                height_scale = ai_h / orig_h
                scaled_width = orig_w * height_scale
                width_offset = (scaled_width - ai_w) / 2.0
                px_step1_inv = px_step2_inv + width_offset
                py_step1_inv = py_step2_inv  # Y坐标不变
                
                # 4.4: 除以高度缩放比例，得到原图坐标
                px_current = px_step1_inv / height_scale
                py_current = py_step1_inv / height_scale
            
            # 最终坐标就是原图上的坐标
            original_px = int(round(px_current))
            original_py = int(round(py_current))
            
            # 创建新的交点
            transformed_pt = pt.copy()
            transformed_pt['original_pixel_x'] = pt['pixel_x']  # 保存校正图坐标
            transformed_pt['original_pixel_y'] = pt['pixel_y']
            transformed_pt['pixel_x'] = original_px  # 更新为原图坐标
            transformed_pt['pixel_y'] = original_py
            
            transformed_points.append(transformed_pt)
            
            print(f"  intersection({pt['real_x_rounded']}, {pt['real_y_rounded']}): "
                  f"校正图({px}, {py}) -> 原图({original_px}, {original_py})")
        
        print(f"[InvTransform] textComplete")
        return transformed_points

    # ------------------------------------------------------------------
    # Visual Fact Extraction (Module M0)
    # ------------------------------------------------------------------
    # These methods extract mathematical facts directly from the curve
    # pixel data using the same CV pipeline that detects coordinates.
    # Unlike the integer-grid inference in intervention.py, these produce
    # sub-pixel-accurate results grounded in visual evidence.
    # ------------------------------------------------------------------

    def _pixel_to_math_linear(self, pixel_x: float, pixel_y: float) -> Tuple[Optional[float], Optional[float]]:
        """Convert pixel coordinates to math coordinates via linear interpolation.

        Unlike pixel_to_real_coordinate() which snaps to the nearest grid
        line, this gives continuous (sub-grid) precision suitable for
        intercepts and extrema that don't fall on grid intersections.
        """
        if not self.coordinate_mapping:
            return None, None
        x_scale = self.coordinate_mapping.get('x_scale')
        x_offset = self.coordinate_mapping.get('x_offset')
        y_scale = self.coordinate_mapping.get('y_scale')
        y_offset = self.coordinate_mapping.get('y_offset')
        if x_scale is None or y_scale is None:
            return None, None
        math_x = x_scale * pixel_x + x_offset
        math_y = y_scale * pixel_y + y_offset
        return round(math_x, 4), round(math_y, 4)

    # ------------------------------------------------------------------
    # NOTE: Visual-fact computation methods (compute_visual_facts and all
    # supporting helpers) have been removed.  Visual facts are now inferred
    # by the LLM in the reasoning pipeline (see
    # app.reasoning_pipeline.infer_visual_facts_with_llm).
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray, grid_h_lines: List, grid_v_lines: List,
                tolerance: int = 3, erase_thickness: int = 5,
                color_saturation_threshold: int = 90,
                color_value_threshold: int = 60,
                morph_connect_enabled: bool = True,
                morph_kernel_size: int = 7,
                morph_iterations: int = 2,
                integer_approximation_threshold: float = 0.3,
                # 新增参数：支持使用已有的刻度信息和变换矩阵
                x_tick_labels: List[Dict] = None,
                y_tick_labels: List[Dict] = None,
                perspective_matrix: np.ndarray = None,
                rotation_matrix: np.ndarray = None,
                original_size: Tuple[int, int] = None,
                crop_offset: Tuple[int, int] = (0, 0)) -> Dict:
        """
        完整的坐标识别流程
        
        Args:
            image: 校正后的图像（彩色）
            grid_h_lines: 水平网格线
            grid_v_lines: 垂直网格线
            tolerance: 交点容差
            erase_thickness: 擦除网格线的线条粗细（备用）
            color_saturation_threshold: 颜色饱和度阈值
            color_value_threshold: 颜色亮度阈值
            morph_connect_enabled: 是否启用形态学连接
            morph_kernel_size: 形态学核大小
            morph_iterations: 形态学迭代次数
            x_tick_labels: X轴刻度标签（可选，如果提供则不调用OCR）
            y_tick_labels: Y轴刻度标签（可选，如果提供则不调用OCR）
            rotation_matrix: 旋转矩阵（可选，用于变换刻度坐标）
            original_size: 原始图像尺寸（可选）
            crop_offset: 裁剪偏移量（可选，默认为(0, 0)）
        
        Returns:
            处理结果字典
        """
        # 设置颜色提取参数
        self.color_saturation_threshold = color_saturation_threshold
        self.color_value_threshold = color_value_threshold
        self.morph_connect_enabled = morph_connect_enabled
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        # 设置整数近似阈值
        self.integer_approximation_threshold = integer_approximation_threshold
        result = {
            'success': False,
            'error': None,
            'axes_info': {},
            'intersections': [],
            'annotated_image': None,
            'curve_binary': None,
            'intermediate_binary': None,
            'intermediate_skeleton': None,
            'intermediate_filtered': None,
            'intermediate_connected': None
        }
        
        try:
            # 1. 获取刻度标签（优先使用传入的，否则调用OCR OCR）
            if x_tick_labels is not None and y_tick_labels is not None:
                print("textlabeltext (textOCR OCR)...")
                
                # 如果提供了旋转矩阵，需要变换刻度坐标
                if perspective_matrix is not None or rotation_matrix is not None:
                    print("textcoordinate...")
                    h, w = image.shape[:2]
                    x_tick_labels = self.transform_tick_labels_with_rotation(
                        x_tick_labels, rotation_matrix, original_size, (h, w), crop_offset, perspective_matrix
                    )
                    y_tick_labels = self.transform_tick_labels_with_rotation(
                        y_tick_labels, rotation_matrix, original_size, (h, w), crop_offset, perspective_matrix
                    )
                
                # 已经有刻度标签，标记为使用提供的标签
                use_provided_labels = True
                ocr_result = None  # 不需要ocr_result
            else:
                # 调用OCR
                print("textOCR...")
                ocr_result = self.call_ocr(image)
                
                if isinstance(ocr_result, str):
                    result['error'] = f"OCR错误: {ocr_result}"
                    return result
                
                use_provided_labels = False
            
            # 2. 提取刻度标签（仅当未提供时）
            if not use_provided_labels:
                print("textlabel...")
                x_tick_labels, y_tick_labels = self.extract_tick_labels(ocr_result)
                
                # 过滤OCR识别错误的标签
                print("\nfilterOCRtextlabel...")
                x_tick_labels = self._filter_ocr_labels(x_tick_labels, 'x')
                y_tick_labels = self._filter_ocr_labels(y_tick_labels, 'y')
                print(f"\nfilterafter: X-axis{len(x_tick_labels)}textlabel, Y-axis{len(y_tick_labels)}textlabel")
            else:
                print("textlabel (text0Completefiltertextcoordinatetext)...")
            
            print(f"Found {len(x_tick_labels)} textX-axistextlabel")
            print(f"Found {len(y_tick_labels)} textY-axistextlabel")
            
            if len(x_tick_labels) < 2 or len(y_tick_labels) < 2:
                result['error'] = f"刻度标签不足：X轴{len(x_tick_labels)}个，Y轴{len(y_tick_labels)}个（至少需要各2个）"
                return result
            
            # 3. 推测坐标轴位置（从网格线中选择）
            print("textaxisposition...")
            axes_info = self.infer_axes_position(x_tick_labels, y_tick_labels, 
                                                grid_h_lines, grid_v_lines)
            self.axis_info = axes_info
            result['axes_info'] = axes_info
            
            # 4. 建立坐标映射
            print("textcoordinatetext...")
            coordinate_mapping = self.build_coordinate_mapping(
                x_tick_labels, y_tick_labels, axes_info,
                grid_h_lines, grid_v_lines
            )
            self.coordinate_mapping = coordinate_mapping
            
            # 5. 提取曲线
            print("textcurve...")
            curve_binary, intermediates = self.extract_curve(
                image, grid_h_lines, grid_v_lines, ocr_result, erase_thickness
            )
            result['curve_binary'] = curve_binary
            result['intermediate_binary'] = intermediates.get('binary')
            result['intermediate_skeleton'] = intermediates.get('skeleton')
            result['intermediate_filtered'] = intermediates.get('filtered')
            result['intermediate_connected'] = intermediates.get('connected')
            
            # 6. 可视化网格线坐标映射
            print("textgrid linecoordinatetext...")
            grid_coord_vis = self.visualize_grid_coordinates(
                image, grid_h_lines, grid_v_lines, 
                x_tick_labels, y_tick_labels, axes_info
            )
            result['grid_coord_visualization'] = grid_coord_vis
            
            # 7. 查找交点
            print("textintersection...")
            intersections_raw = self.find_line_intersections(
                curve_binary, grid_h_lines, grid_v_lines, axes_info, tolerance
            )
            
            print(f"Found {len(intersections_raw)} textintersection")
            
            # 保存原始交点（用于对比）
            result['intersections_raw'] = intersections_raw
            
            # 7.5. 准备推理交点（只做轻量清洗，不做可视化稀疏化）
            print("textintersection...")
            self._last_curve_binary = curve_binary
            intersections_reasoning = self._prepare_reasoning_intersections(
                intersections_raw
            )
            result['reasoning_intersections'] = intersections_reasoning
            result['reasoning_debug'] = self._last_reasoning_debug
            dbg = self._last_reasoning_debug
            print(f"textintersection: {dbg.get('final_count', '?')} text "
                  f"(种子={dbg.get('integer_seed_count', '?')}, "
                  f"rescue={dbg.get('rescued_count', '?')}, "
                  f"gap_fill={dbg.get('gap_filled_count', '?')}, "
                  f"dropped={len(dbg.get('dropped_keys', []))})")

            # 7.6. 跳过关键交点过滤
            # 之前这里会为了可视化稀疏化/筛选点集，导致一些真实交点在
            # Stage 2 最终输出和标注图里被移除。当前改为直接保留原始交点，
            # 只使用 find_line_intersections() 的检测结果，不再做展示层删点。
            print("textintersectionfilter, textKeeporiginalintersection...")
            intersections_filtered = intersections_raw
            result['intersections'] = intersections_filtered

            # 7.7. Visual facts are now inferred by LLM (see reasoning pipeline)
            result['visual_facts'] = None

            # 8. 可视化最终结果
            print("text...")
            annotated_image = self.visualize_result(image, axes_info, intersections_filtered)
            result['annotated_image'] = annotated_image
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = f"处理过程出错: {str(e)}"
            import traceback

            traceback.print_exc()
        
        return result

