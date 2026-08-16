"""
题目图片处理模块
从题目图片中提取坐标系chart
"""

import ast
import io
import sys
import re
import logging
import warnings

# Ensure stdout supports full Unicode on Windows (GBK can't encode ⚠/✅/etc.)
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass

warnings.filterwarnings("ignore", message=".*non-text parts.*thought_signature.*")

import cv2
import numpy as np
import yaml
from typing import Dict, Tuple, Optional, List
from ..mathpix.mathpix_image_ocr import MathpixImageOCRTool
import json
import base64
from pydantic import BaseModel, Field


def _get_genai():
    """Lazy-load google.genai to avoid pulling ~112 MB of google libs at import time."""
    from google import genai
    logging.getLogger("google.genai.types").setLevel(logging.ERROR)
    return genai


def _get_genai_config():
    """Lazy-load GenerateContentConfig."""
    from google.genai.types import GenerateContentConfig
    return GenerateContentConfig

import sys as _sys
from pathlib import Path as _Path
_exp_root = _Path(__file__).resolve().parent.parent.parent
if str(_exp_root) not in _sys.path:
    _sys.path.insert(0, str(_exp_root))
from i18n import log_tag as _lt  # noqa: E402

_API_KEYS_PATH = _exp_root / "config" / "api_keys.yaml"


def _load_gemini_api_key() -> str:
    """Load Gemini API key from config/api_keys.yaml.

    Returns empty string if not configured (Gemini correction is optional).
    """
    if not _API_KEYS_PATH.exists():
        return ""
    with open(_API_KEYS_PATH, "r", encoding="utf-8") as f:
        keys = yaml.safe_load(f) or {}
    return keys.get("gemini_3_pro", "")


# ==================== Pydantic模型定义（用于Gemini结构化输出） ====================

class CoordinateLabel(BaseModel):
    """单个坐标标签"""
    key: str = Field(description="坐标刻度标签编号，如 'X1', 'X2', 'Y1', 'Y2' 等")
    text: str = Field(description="识别到的X轴、Y轴坐标刻度标签数值，如 '1/2', 'π', '-3', '0.5', '-1.5'")

class CoordinateRecognition(BaseModel):
    """坐标识别结果"""
    x_labels: List[CoordinateLabel] = Field(description="X轴坐标刻度标签列表")
    y_labels: List[CoordinateLabel] = Field(description="Y轴坐标刻度标签列表")


# ==================== ChartExtractor类 ====================

class ChartExtractor:
    """从题目图片中提取坐标系图表"""
    
    def __init__(self, label_parse_config: Optional[Dict] = None):
        """
        初始化ChartExtractor
        
        Args:
            label_parse_config: 标签解析配置字典，支持以下选项：
                - allow_integer: 允许整数（如：1, -2, 0），默认True
                - allow_float: 允许小数（如：0.5, -1.5），默认False
                - allow_fraction: 允许分数（如：1/2, -3/4），默认False
                - allow_pi: 允许π相关（如：π, 2π, π/2），默认False
                - allow_sqrt: 允许平方根（如：√2），默认False
                注意：LaTeX命令始终会被处理（因为整数也可能用LaTeX格式）
        """
        self.ocr_tool = MathpixImageOCRTool()
        # 默认容差参数
        self.horizontal_tolerance = 1.5  # 水平线容差（倍数）
        self.vertical_tolerance = 1.5    # 垂直线容差（倍数）
        self.interval_percentile = 50    # 间隔分位数（默认P50，即中位数）
        
        # 标签解析配置（默认只支持整数）
        default_config = {
            'allow_integer': True,
            'allow_float': True,
            'allow_fraction': False,
            'allow_pi': False,
            'allow_sqrt': False,
        }
        
        if label_parse_config is None:
            self.label_parse_config = default_config
        else:
            # 合并用户配置和默认配置
            self.label_parse_config = {**default_config, **label_parse_config}
    
    def extract_chart_from_image(self, image: np.ndarray, 
                                horizontal_tolerance: float = None,
                                vertical_tolerance: float = None,
                                interval_percentile: float = None,
                                gemini_enabled: bool = False) -> Dict:
        """
        从题目图片中提取坐标系chart
        
        流程：
        1. 调用OCR OCR识别整张图片
        2. 从line_data中找到chart并收集所有可能的label（5种类型）
        3. 对label进行数值过滤和位置聚类
        4. 裁剪出chart区域
        5. 转换坐标标签位置（从相对chart到相对原图）
        6. (可选) 使用Gemini识别坐标
        
        Args:
            image: 输入的题目图片
            horizontal_tolerance: 水平线容差（倍数），默认使用self.horizontal_tolerance
            vertical_tolerance: 垂直线容差（倍数），默认使用self.vertical_tolerance
            interval_percentile: 间隔分位数（0-100），默认使用self.interval_percentile
            gemini_enabled: 是否启用Gemini坐标识别，默认False
            gemini_credentials: Gemini API凭据字典
        
        Returns:
            结果字典，包含：
            - success: 是否成功
            - error: 错误信息（如果失败）
            - chart_image: 裁剪出的chart图片
            - chart_bbox: chart在原图中的位置 (x, y, w, h)
            - x_tick_labels: X轴刻度标签列表（坐标已转换为相对chart）
            - y_tick_labels: Y轴刻度标签列表（坐标已转换为相对chart）
            - ocr_result: 完整的OCR结果
            - debug_info: 调试信息（所有候选label的详细信息）
            - debug_image: 调试可视化图片（标注所有label位置）
            - gemini_debug_info: Gemini识别调试信息
        """
        # 使用传入的容差参数，如果没有则使用默认值
        if horizontal_tolerance is None:
            horizontal_tolerance = self.horizontal_tolerance
        if vertical_tolerance is None:
            vertical_tolerance = self.vertical_tolerance
        if interval_percentile is None:
            interval_percentile = self.interval_percentile
        
        result = {
            'success': False,
            'error': None,
            'chart_image': None,
            'chart_bbox': None,
            'x_tick_labels': [],
            'y_tick_labels': [],
            'ocr_result': None,
            'debug_info': {},
            'debug_image': None
        }
        
        try:
            # 1. 保存图片为临时文件用于OCR
            import tempfile
            import os
            
            # 打印原始图片信息
            h, w = image.shape[:2]
            print(f"[Chart提取] 原始图片尺寸: {w}x{h}, 通道数: {image.shape[2] if len(image.shape) > 2 else 1}")
            
            # 图片预处理：如果超过1920x1080则缩放
            MAX_WIDTH = 1920
            MAX_HEIGHT = 1080
            
            processed_image = image
            # if w > MAX_WIDTH or h > MAX_HEIGHT:
            #     # 计算缩放比例（保持宽高比）
            #     scale_w = MAX_WIDTH / w
            #     scale_h = MAX_HEIGHT / h
            #     scale = min(scale_w, scale_h)  # 使用较小的缩放比例，确保两个维度都不超标
                
            #     new_w = int(w * scale)
            #     new_h = int(h * scale)
                
            #     print(f"[Chart提取] 图片过大，缩放到: {new_w}x{new_h} (缩放比例: {scale:.2f})")
            #     processed_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # else:
            #     print(f"[Chart提取] 图片尺寸符合要求，无需缩放")
            
            # 使用JPG格式保存（有损压缩）
            _tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = _tmp.name
            _tmp.close()
            print(f"[Chart提取] 临时文件路径: {temp_path}")
            
            # 保存为JPG格式，质量参数100
            save_success = cv2.imwrite(temp_path, processed_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if not save_success:
                result['error'] = f"图片保存失败: {temp_path}"
                return result
            
            # 检查文件是否存在及大小
            if not os.path.exists(temp_path):
                result['error'] = f"临时文件不存在: {temp_path}"
                return result
            
            file_size = os.path.getsize(temp_path)
            print(f"[Chart提取] 临时文件大小: {file_size} 字节")
            
            if file_size == 0:
                result['error'] = f"临时文件为空: {temp_path}"
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return result
            
            print("[Chart提取] 调用OCR识别题目图片...")

            # 2. 调用OCR
            try:
                base64_data = self._image_to_base64(temp_path)
                print(f"[Chart提取] Base64编码长度: {len(base64_data)} 字符")
                ocr_result = self.ocr_tool.ocr_image(f"data:image/jpeg;base64,{base64_data}")

            except Exception as e:
                result['error'] = f"OCR调用失败: {str(e)}"
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return result
            
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # 打印OCR结果基本信息
            if ocr_result:
                print(f"[Chart提取] OCR返回的键: {list(ocr_result.keys())}")
                if 'line_data' in ocr_result:
                    print(f"[Chart提取] line_data 项数: {len(ocr_result['line_data'])}")
                else:
                    print("[Chart提取] WARNING: OCR结果中没有 line_data 字段")
            else:
                print("[Chart提取] ⚠️ OCR返回结果为空")
            
            if not ocr_result or 'line_data' not in ocr_result:
                result['error'] = "OCR识别失败或未返回line_data"
                return result
            
            # 调用主流程方法处理
            return self._process_chart_with_ocr_result(
                image, ocr_result, horizontal_tolerance, vertical_tolerance, interval_percentile,
                gemini_enabled=gemini_enabled)
            
        except Exception as e:
            import traceback
            result['error'] = f"处理失败: {str(e)}\n{traceback.format_exc()}"
            return result
    
    def extract_chart_from_ocr_result(self, 
                                      image: np.ndarray,
                                      ocr_result: Dict,
                                      horizontal_tolerance: float = None,
                                      vertical_tolerance: float = None,
                                      interval_percentile: float = None,
                                      gemini_enabled: bool = False) -> Dict:
        """
        从已有的OCR结果中提取chart（不调用OCR API）
        
        用于API场景：用户已经调用过OCR，直接传入OCR结果
        
        Args:
            image: 输入的题目图片
            ocr_result: OCR OCR结果（JSON）
            horizontal_tolerance: 水平线容差（倍数），默认使用self.horizontal_tolerance
            vertical_tolerance: 垂直线容差（倍数），默认使用self.vertical_tolerance
            interval_percentile: 间隔分位数（0-100），默认使用self.interval_percentile
            gemini_enabled: 是否启用Gemini坐标识别，默认False
        
        Returns:
            结果字典，同 extract_chart_from_image
        """
        # 使用传入的容差参数，如果没有则使用默认值
        if horizontal_tolerance is None:
            horizontal_tolerance = self.horizontal_tolerance
        if vertical_tolerance is None:
            vertical_tolerance = self.vertical_tolerance
        if interval_percentile is None:
            interval_percentile = self.interval_percentile
        
        # 打印原始图片信息
        h, w = image.shape[:2]
        print(f"[Chart提取] 原始图片尺寸: {w}x{h}, 通道数: {image.shape[2] if len(image.shape) > 2 else 1}")
        
        # 验证OCR结果
        if not ocr_result or 'line_data' not in ocr_result:
            return {
            'success': False,
                'error': "OCR结果无效或未包含line_data",
            'chart_image': None,
            'chart_bbox': None,
            'x_tick_labels': [],
            'y_tick_labels': [],
            'ocr_result': ocr_result,
            'debug_info': {},
            'debug_image': None
        }
        
        print(f"[Chart提取] OCR结果包含 {len(ocr_result['line_data'])} 个line_data项")
        
        # 调用主流程方法处理
        return self._process_chart_with_ocr_result(
            image, ocr_result, horizontal_tolerance, vertical_tolerance, interval_percentile,
            gemini_enabled=gemini_enabled)
    
    # ==================== 核心处理方法（拆分后的私有方法） ====================
    
    def _preprocess_ocr_and_image(self, image: np.ndarray, ocr_result: Dict) -> np.ndarray:
        """
        方法1: 处理OCR自动旋转，确保图片和OCR结果对齐
        
        Args:
            image: 原始图片
            ocr_result: OCR识别结果
        
        Returns:
            旋转后的图片（如果需要旋转）
        """
        auto_rotate_degrees = ocr_result.get('auto_rotate_degrees', 0)
        print(f"[Chart提取] OCR自动旋转角度: {auto_rotate_degrees}度")
        
        if auto_rotate_degrees == 0:
            return image
        
        print(f"[Chart提取] 检测到图片被OCR自动旋转了{auto_rotate_degrees}度，正在旋转原始图片...")
        original_shape = image.shape[:2]
        print(f"[Chart提取] 旋转前图片尺寸: {original_shape[1]}x{original_shape[0]} (宽x高)")
        
        # 根据旋转角度选择合适的旋转方法
        if auto_rotate_degrees == 90:
            # 逆时针90度 = OpenCV的顺时针270度
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif auto_rotate_degrees == 180:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        elif auto_rotate_degrees == 270:
            # 逆时针270度 = OpenCV的顺时针90度
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            print(f"[Chart提取] ⚠️ 不支持的旋转角度: {auto_rotate_degrees}度，跳过旋转")
            return image
        
        rotated_shape = rotated_image.shape[:2]
        print(f"[Chart提取] 旋转后图片尺寸: {rotated_shape[1]}x{rotated_shape[0]} (宽x高)")
        
        # 验证旋转后的尺寸是否与OCR返回的尺寸一致
        ocr_width = ocr_result.get('image_width', 0)
        ocr_height = ocr_result.get('image_height', 0)
        if ocr_width > 0 and ocr_height > 0:
            if rotated_shape[1] != ocr_width or rotated_shape[0] != ocr_height:
                print(f"[Chart提取] ⚠️ 旋转后尺寸({rotated_shape[1]}x{rotated_shape[0]}) 与OCR尺寸({ocr_width}x{ocr_height})不匹配")
            else:
                print(f"[Chart提取] ✅ 旋转后尺寸与OCR尺寸匹配")
        
        return rotated_image
    
    def _extract_and_cluster_labels(self, 
                                   ocr_result: Dict,
                                   horizontal_tolerance: float,
                                   vertical_tolerance: float) -> Tuple[Dict, Dict, List, Dict]:
        """
        方法2: 从OCR结果中提取chart并聚类标签
        
        Args:
            ocr_result: OCR识别结果
            horizontal_tolerance: 水平线容差
            vertical_tolerance: 垂直线容差
        
        Returns:
            (chart_info, cluster_result, all_candidate_labels, label_type_counts)
        
        Raises:
            ValueError: 如果未找到chart或标签数量不足
        """
        print("[Chart提取] 从line_data中查找chart并收集候选标签...")
        
        # 从line_data中找到chart和所有候选标签
        chart_info = self._find_chart_in_line_data(ocr_result['line_data'])
        
        if not chart_info:
            raise ValueError("未在图片中找到坐标系chart")
        
        print(f"[Chart提取] 找到chart: {chart_info['bbox']}")
        
        all_candidate_labels = chart_info.get('all_candidate_labels', [])
        label_type_counts = chart_info.get('label_type_counts', {})
        
        print(f"[Chart提取] 候选标签类型统计: {label_type_counts}")
        
        if len(all_candidate_labels) == 0:
            raise ValueError("未找到任何候选刻度标签")
            
        # 位置聚类：分离X轴和Y轴标签
        print("[Chart提取] 对候选标签进行位置聚类...")
        cluster_result = self._cluster_labels_by_position(
            all_candidate_labels,
            horizontal_tolerance,
            vertical_tolerance
        )
        
        x_labels_raw = cluster_result['x_labels']
        y_labels_raw = cluster_result['y_labels']
        
        print(f"[Chart提取] 聚类结果: X轴{len(x_labels_raw)}个, Y轴{len(y_labels_raw)}个")
            
        # 检查聚类结果
        if len(x_labels_raw) < 2:
            raise ValueError(f"X轴刻度标签数量不足（需要≥2个，实际{len(x_labels_raw)}个）")
            
        if len(y_labels_raw) < 2:
            raise ValueError(f"Y轴刻度标签数量不足（需要≥2个，实际{len(y_labels_raw)}个）")
        
        return chart_info, cluster_result, all_candidate_labels, label_type_counts
    
    def _crop_and_convert_coordinates(self,
                                     image: np.ndarray,
                                     chart_bbox: Dict,
                                     x_labels_raw: List[Dict],
                                     y_labels_raw: List[Dict]) -> Tuple[np.ndarray, Tuple, List[Dict], List[Dict]]:
        """
        方法3: 裁剪chart并转换标签坐标
        
        Args:
            image: 原始图片（已旋转）
            chart_bbox: chart边界框 {left, top, right, bottom}
            x_labels_raw: X轴原始标签列表
            y_labels_raw: Y轴原始标签列表
        
        Returns:
            (chart_image, chart_bbox_tuple, x_labels_converted, y_labels_converted)
        """
        print("[Chart提取] 裁剪chart区域...")
        
        # 裁剪chart区域
        x1 = int(chart_bbox['left'])
        y1 = int(chart_bbox['top'])
        x2 = int(chart_bbox['right'])
        y2 = int(chart_bbox['bottom'])
        
        # 确保坐标在图像范围内
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        chart_image = image[y1:y2, x1:x2].copy()
        
        print(f"[Chart提取] Chart尺寸: {x2-x1}x{y2-y1}")
            
        # 转换刻度标签坐标（从相对原图转换为相对chart）
        x_labels_converted = []
        for label in x_labels_raw:
            converted_label = {
                'text': label['text'],
                'value': label['value'],
                'bbox': {
                    'left': label['bbox']['left'] - x1,
                    'top': label['bbox']['top'] - y1,
                    'right': label['bbox']['right'] - x1,
                    'bottom': label['bbox']['bottom'] - y1
                },
                'center_x': label['center_x'] - x1,
                'center_y': label['center_y'] - y1,
                'raw_center_x': label['center_x'],
                'raw_center_y': label['center_y'],
                'original_type': label.get('original_type', 'unknown')
            }
            x_labels_converted.append(converted_label)

        # NOTE: this loop and the print below used to sit INSIDE the x-label
        # loop above.  The y list was therefore rebuilt from scratch once per
        # x label -- correct output, wasted work -- and with no x labels at
        # all, `y_labels_converted` was never bound and the return raised
        # UnboundLocalError instead of returning two empty lists.  Upstream
        # gating currently makes that unreachable, but the binding must not
        # depend on an unrelated caller's precondition.
        y_labels_converted = []
        for label in y_labels_raw:
            converted_label = {
                'text': label['text'],
                'value': label['value'],
                'bbox': {
                    'left': label['bbox']['left'] - x1,
                    'top': label['bbox']['top'] - y1,
                    'right': label['bbox']['right'] - x1,
                    'bottom': label['bbox']['bottom'] - y1
                },
                'center_x': label['center_x'] - x1,
                'center_y': label['center_y'] - y1,
                'raw_center_x': label['center_x'],
                'raw_center_y': label['center_y'],
                'original_type': label.get('original_type', 'unknown')
            }
            y_labels_converted.append(converted_label)

        print(f"[Chart提取] 转换后X轴标签: {len(x_labels_converted)}个, Y轴标签: {len(y_labels_converted)}个")

        chart_bbox_tuple = (x1, y1, x2-x1, y2-y1)
        return chart_image, chart_bbox_tuple, x_labels_converted, y_labels_converted
    
    def _filter_and_validate_labels(self,
                                    x_labels: List[Dict],
                                    y_labels: List[Dict],
                                    interval_percentile: float) -> Tuple[List[Dict], List[Dict]]:
        """
        方法4: 过滤OCR错误标签并验证
        
        Args:
            x_labels: X轴标签列表
            y_labels: Y轴标签列表
            interval_percentile: 间隔分位数
        
        Returns:
            (x_labels_filtered, y_labels_filtered)
        
        Raises:
            ValueError: 如果过滤后标签数量不足
        """
        print("[Chart提取] 开始过滤OCR标签...")
        
        x_labels_filtered = self._filter_ocr_labels(x_labels, 'x', interval_percentile)
        y_labels_filtered = self._filter_ocr_labels(y_labels, 'y', interval_percentile)
            
        # 检查过滤后的标签是否有效
        if len(x_labels_filtered) < 2 or len(y_labels_filtered) < 2:
            raise ValueError(f"OCR标签过滤失败：X轴{len(x_labels_filtered)}个，Y轴{len(y_labels_filtered)}个（至少需要各2个且有不同值）")
        
        return x_labels_filtered, y_labels_filtered
    
    def _generate_debug_outputs(self,
                               image: np.ndarray,
                               chart_bbox: Dict,
                               all_candidate_labels: List[Dict],
                               cluster_result: Dict,
                               label_type_counts: Dict,
                               x_labels_raw: List[Dict],
                               y_labels_raw: List[Dict],
                               x_labels_filtered: List[Dict],
                               y_labels_filtered: List[Dict],
                               horizontal_tolerance: float,
                               vertical_tolerance: float,
                               interval_percentile: float) -> Tuple[Dict, np.ndarray]:
        """
        方法5: 生成调试信息和可视化图片
        
        Args:
            image: 原始图片（已旋转）
            chart_bbox: chart边界框
            all_candidate_labels: 所有候选标签
            cluster_result: 聚类结果
            label_type_counts: 标签类型统计
            x_labels_raw: X轴原始标签
            y_labels_raw: Y轴原始标签
            x_labels_filtered: X轴过滤后标签
            y_labels_filtered: Y轴过滤后标签
            horizontal_tolerance: 水平线容差
            vertical_tolerance: 垂直线容差
            interval_percentile: 间隔分位数
        
        Returns:
            (debug_info, debug_image)
        """
        print("[Chart提取] 生成调试信息...")
        
        debug_info = self._generate_debug_info(
                all_candidate_labels, cluster_result, label_type_counts,
                x_labels_raw, y_labels_raw, x_labels_filtered, y_labels_filtered,
                horizontal_tolerance, vertical_tolerance, interval_percentile
            )
            
        print("[Chart提取] 生成可视化调试图片...")
        debug_image = self._generate_debug_image(
                image, chart_bbox, all_candidate_labels, x_labels_raw, y_labels_raw, x_labels_filtered, y_labels_filtered
            )
            
        return debug_info, debug_image
    
    def _generate_gemini_input_image(self,
                                    chart_image: np.ndarray,
                                    x_labels: List[Dict],
                                    y_labels: List[Dict]) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
        """
        生成Gemini输入图片：在chart上标注坐标框和序号
        
        Args:
            chart_image: 裁剪后的chart图片
            x_labels: X轴标签列表（坐标相对于chart）
            y_labels: Y轴标签列表（坐标相对于chart）
        
        Returns:
            (标注后的图片, 排序后的X轴标签, 排序后的Y轴标签)
        """
        # 导入字体自适应函数
        from .drawing_utils import calculate_adaptive_font_scale
        
        # 复制图片用于标注
        x_labels_annotated_image = chart_image.copy()
        y_labels_annotated_image = chart_image.copy()
        
        # 对标签排序并添加key
        x_labels_sorted = sorted(x_labels, key=lambda l: l['center_x'])  # 从左到右
        for i, label in enumerate(x_labels_sorted, 1):
            label['key'] = f'X{i}'  # 添加唯一标识
        
        y_labels_sorted = sorted(y_labels, key=lambda l: l['center_y'])  # 从上到下
        for i, label in enumerate(y_labels_sorted, 1):
            label['key'] = f'Y{i}'  # 添加唯一标识
        
        print(f"[Gemini输入] X轴标签数量: {len(x_labels_sorted)}, Y轴标签数量: {len(y_labels_sorted)}")
        
        # 获取图片尺寸用于边界检测
        img_height, img_width = x_labels_annotated_image.shape[:2]
        alpha = 0.8  # 提高不透明度，让标注更清晰
        
        # 计算自适应字体大小
        base_font_scale = calculate_adaptive_font_scale(x_labels, y_labels, (img_height, img_width))
        # 适当放大字体，让序号更清晰
        font_scale = base_font_scale * 0.8
        # 增加字体粗细，让文字更清晰
        font_thickness = max(1, round(font_scale * 2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        padding = max(2, round(font_scale * 4))  # 增大padding
        
        print(f"[Gemini输入] 自适应字体: base={base_font_scale:.2f}, final={font_scale:.2f}, thickness={font_thickness}")
        
        # 绘制X轴标签（绿色半透明）
        for i, label in enumerate(x_labels_sorted, 1):
            bbox = label['bbox']
            width = int(bbox['right'] - bbox['left'])
            center_x = int((bbox['right'] + bbox['left']) / 2)
            left = int(center_x - 0.6 * width)
            top = int(bbox['top'])
            right = int(center_x + 0.6 * width)
            bottom = int(bbox['bottom'])
            
            # 创建半透明覆盖层
            overlay = x_labels_annotated_image.copy()
            
            # 绘制绿色矩形框（更细）
            cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 0), 1)
            
            # 准备标注文本
            label_text = f"{label['key']}"
            
            # 获取文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            # 计算文本绘制位置（水平居中在 center_x）
            text_x = center_x - text_width // 2  # 水平居中
            
            # 检测边界并决定标注位置（上方或下方）
            if top - text_height - baseline - padding * 2 < 0:
                # 上方超出边界，标注到下方
                bg_top = bottom + 2
                bg_bottom = bottom + text_height + baseline + padding * 2 + 2
                text_y = bottom + text_height + padding
            else:
                # 标注到上方（默认位置）
                bg_top = top - text_height - baseline - padding * 2
                bg_bottom = top - 2
                text_y = top - padding - baseline
            
            # 计算背景框位置（水平居中，确保文字不超出）
            bg_left = text_x - padding
            bg_right = text_x + text_width + padding
            
            # 在overlay上绘制白色背景
            cv2.rectangle(overlay, (bg_left, bg_top), (bg_right, bg_bottom), (255, 255, 255), -1)
            # 应用半透明效果（alpha=0.5）
            cv2.addWeighted(overlay, alpha, x_labels_annotated_image, 1-alpha, 0, x_labels_annotated_image)

            # 在overlay上绘制序号文本（绿色）
            cv2.putText(x_labels_annotated_image, label_text, (text_x, text_y),
                       font, font_scale, (0, 200, 0), font_thickness)
        
        # 绘制Y轴标签（红色半透明）
        for i, label in enumerate(y_labels_sorted, 1):
            bbox = label['bbox']
            width = int(bbox['right'] - bbox['left'])
            center_x = int((bbox['right'] + bbox['left']) / 2)
            center_y = int((bbox['top'] + bbox['bottom']) / 2)
            left = int(center_x - 0.6 * width)
            top = int(bbox['top'])
            right = int(center_x + 0.6 * width)
            bottom = int(bbox['bottom'])
            
            # 创建半透明覆盖层
            overlay = y_labels_annotated_image.copy()
            
            # 绘制红色矩形框（更细）
            cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 255), 1)
            
            # 准备标注文本
            label_text = f"{label['key']}"
            
            # 获取文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            # 计算文本绘制位置（垂直居中在 center_y）
            text_y = center_y + text_height // 2  # 垂直居中
            
            # 检测边界并决定标注位置（左侧或右侧）
            if left - text_width - padding * 2 - 4 < 0:
                # 左侧超出边界，标注到右侧
                bg_left = right + 2
                text_x = right + padding
            else:
                # 标注到左侧（默认位置）
                bg_left = left - text_width - padding * 2 - 4
                text_x = left - text_width - padding - 4
            
            # 计算背景框位置（垂直居中，确保文字不超出）
            bg_top = text_y - text_height - padding
            bg_bottom = text_y + baseline + padding
            
            # 确保背景框宽度足够
            if left - text_width - padding * 2 - 4 < 0:
                # 右侧
                bg_right = bg_left + text_width + padding * 2
            else:
                # 左侧
                bg_right = left - 2
            
            # 在overlay上绘制白色背景
            cv2.rectangle(overlay, (bg_left, bg_top), (bg_right, bg_bottom), (255, 255, 255), -1)
            
            # 应用半透明效果（alpha=0.5）
            cv2.addWeighted(overlay, alpha, y_labels_annotated_image, 1-alpha, 0, y_labels_annotated_image)

            # 在overlay上绘制序号文本（红色）
            cv2.putText(y_labels_annotated_image, label_text, (text_x, text_y),
                       font, font_scale, (0, 0, 200), font_thickness)
        
        return x_labels_annotated_image, y_labels_annotated_image, x_labels_sorted, y_labels_sorted
    
    def _recognize_coordinates_with_gemini(self,
                                          chart_image: np.ndarray,
                                          x_labels: List[Dict],
                                          y_labels: List[Dict]) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        使用Gemini识别坐标系的真实坐标值（通过API Key方式）
        
        Args:
            chart_image: 裁剪后的chart图片
            x_labels: X轴标签列表（OCR结果）
            y_labels: Y轴标签列表（OCR结果）
        
        Returns:
            (x_labels_corrected, y_labels_corrected, gemini_debug_info)
        """
        gemini_debug_info = {
            'enabled': True,
            'success': False,
            'error': None,
            'annotated_image': None,
            'gemini_response': None,
            'ocr_vs_gemini': []
        }
        
        try:
            print("[Gemini识别] 开始使用Gemini识别坐标...")
            
            # 1. 生成标注图片
            x_labels_annotated_image, y_labels_annotated_image, x_labels_sorted, y_labels_sorted = self._generate_gemini_input_image(
                chart_image, x_labels, y_labels
            )
            h1, w1 = x_labels_annotated_image.shape[:2]
            h2, w2 = y_labels_annotated_image.shape[:2]
            max_h = max(h1, h2)

            def pad_to_height(img, target_h):
                h, w = img.shape[:2]
                if h == target_h:
                    return img
                return cv2.copyMakeBorder(img, 0, target_h - h, 0, 0,
                                          borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

            x_img_pad = pad_to_height(x_labels_annotated_image, max_h)
            y_img_pad = pad_to_height(y_labels_annotated_image, max_h)
            annotated_image = np.concatenate([x_img_pad, y_img_pad], axis=1)
            gemini_debug_info['annotated_image'] = annotated_image
            
            # 2. 将图片编码为 bytes (Gemini SDK Part.from_bytes)
            _, x_buffer = cv2.imencode('.jpg', x_labels_annotated_image)
            _, y_buffer = cv2.imencode('.jpg', y_labels_annotated_image)
            x_image_bytes = x_buffer.tobytes()
            y_image_bytes = y_buffer.tobytes()
            
            # 3. 创建 Gemini client（API Key 方式，与表达式识别一致）
            api_key = _load_gemini_api_key()
            if not api_key:
                raise ValueError(
                    "Gemini API key (gemini_3_pro) not configured in api_keys.yaml. "
                    "Gemini correction is optional but improves accuracy. "
                    "Get a key at https://aistudio.google.com/apikey"
                )
            client_gemini = _get_genai().Client(api_key=api_key)
            
            # 4. 构建 prompt
            ocr_x_labels = [
                {"key": lbl['key'], "text": lbl['text']}
                for lbl in x_labels_sorted
            ]
            ocr_y_labels = [
                {"key": lbl['key'], "text": lbl['text']}
                for lbl in y_labels_sorted
            ]
            ocr_labels_json = json.dumps({
                "x_labels": ocr_x_labels,
                "y_labels": ocr_y_labels
            }, ensure_ascii=False, indent=2)
            
            prompt = (
                "图片中有两个坐标系，其中：\n"
                "- 第一个坐标系中的绿色框标注了X轴刻度标签（编号为X1, X2, X3...）\n"
                "- 第二个坐标系中的红色框标注了Y轴刻度标签（编号为Y1, Y2, Y3...）\n\n"
                "这是原始OCR识别的X轴、Y轴刻度标签：\n"
                f"```json\n{ocr_labels_json}\n```\n\n"
                "注意不是所有的刻度标签都有编号和框，"
                "检查每个有编号和框的刻度标签值的原始OCR识别结果是否正确。"
            )
            
            print("[Gemini识别] 发送请求到Gemini API...")
            
            # 5. 调用 Gemini API（API Key + google-genai SDK）
            from google.genai import types as _gtypes
            import io as _io, contextlib as _ctxlib

            response = client_gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    _gtypes.Content(
                        role="user",
                        parts=[
                            _gtypes.Part.from_text(text=prompt),
                            _gtypes.Part.from_bytes(data=x_image_bytes, mime_type="image/jpeg"),
                            _gtypes.Part.from_bytes(data=y_image_bytes, mime_type="image/jpeg"),
                        ],
                    )
                ],
                config=_get_genai_config()(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=CoordinateRecognition.model_json_schema(),
                ),
            )
            
            print("[Gemini识别] 收到响应")
            
            # 6. 解析响应（抑制 SDK 的直接 print 输出）
            with _ctxlib.redirect_stderr(_io.StringIO()):
                resp_text = response.text
            recognition_result = CoordinateRecognition.model_validate_json(resp_text)
            gemini_debug_info['gemini_response'] = recognition_result.model_dump()
            
            print(f"[Gemini识别] 解析结果: X轴{len(recognition_result.x_labels)}个, Y轴{len(recognition_result.y_labels)}个")
            
            # 7. 创建OCR标签的key映射
            ocr_x_dict = {label['key'].upper(): label for label in x_labels_sorted}
            ocr_y_dict = {label['key'].upper(): label for label in y_labels_sorted}
            
            # 创建Gemini标签的key映射
            gemini_x_dict = {label.key.upper(): label for label in recognition_result.x_labels}
            gemini_y_dict = {label.key.upper(): label for label in recognition_result.y_labels}
            
            print(f"[Gemini识别] OCR X轴keys: {list(ocr_x_dict.keys())}")
            print(f"[Gemini识别] Gemini X轴keys: {list(gemini_x_dict.keys())}")
            print(f"[Gemini识别] OCR Y轴keys: {list(ocr_y_dict.keys())}")
            print(f"[Gemini识别] Gemini Y轴keys: {list(gemini_y_dict.keys())}")
            
            # 8. 基于key匹配更新标签值
            x_labels_corrected = []
            
            # 遍历所有OCR标签，生成对比信息
            for ocr_label in x_labels_sorted:
                key = ocr_label['key']
                ocr_text = ocr_label['text']
                ocr_value = ocr_label['value']
                
                if key in gemini_x_dict:
                    # Gemini识别到了这个标签
                    gemini_label = gemini_x_dict[key]
                    gemini_value = self._parse_label_value(gemini_label.text)
                    is_modified = (gemini_label.text != ocr_text) or (gemini_value != ocr_value)
                    
                    gemini_debug_info['ocr_vs_gemini'].append({
                        'axis': 'X',
                        'key': key,
                        'ocr_text': ocr_text,
                        'ocr_value': ocr_value,
                        'gemini_text': gemini_label.text,
                        'gemini_value': gemini_value,
                        'is_modified': is_modified,
                        'recognized': True
                    })
                    
                    # 只保留value非None的标签
                    if gemini_value is not None:
                        corrected_label = ocr_label.copy()
                        corrected_label['text'] = gemini_label.text
                        corrected_label['value'] = gemini_value
                        x_labels_corrected.append(corrected_label)
                        
                        if is_modified:
                            print(f"[Gemini识别] {key} 修正: '{ocr_text}'({ocr_value}) -> '{gemini_label.text}'({gemini_value})")
                    else:
                        print(f"[Gemini识别] {key} 识别失败：value为None，丢弃")
                else:
                    # Gemini未识别到这个标签
                    gemini_debug_info['ocr_vs_gemini'].append({
                        'axis': 'X',
                        'key': key,
                        'ocr_text': ocr_text,
                        'ocr_value': ocr_value,
                        'gemini_text': '-',
                        'gemini_value': None,
                        'is_modified': False,
                        'recognized': False
                    })
                    print(f"[Gemini识别] {key} 未被Gemini识别，丢弃")
            
            # 处理Y轴标签
            y_labels_corrected = []
            
            for ocr_label in y_labels_sorted:
                key = ocr_label['key']
                ocr_text = ocr_label['text']
                ocr_value = ocr_label['value']
                
                if key in gemini_y_dict:
                    # Gemini识别到了这个标签
                    gemini_label = gemini_y_dict[key]
                    gemini_value = self._parse_label_value(gemini_label.text)
                    is_modified = (gemini_label.text != ocr_text) or (gemini_value != ocr_value)
                    
                    gemini_debug_info['ocr_vs_gemini'].append({
                        'axis': 'Y',
                        'key': key,
                        'ocr_text': ocr_text,
                        'ocr_value': ocr_value,
                        'gemini_text': gemini_label.text,
                        'gemini_value': gemini_value,
                        'is_modified': is_modified,
                        'recognized': True
                    })
                    
                    # 只保留value非None的标签
                    if gemini_value is not None:
                        corrected_label = ocr_label.copy()
                        corrected_label['text'] = gemini_label.text
                        corrected_label['value'] = gemini_value
                        y_labels_corrected.append(corrected_label)
                        
                        if is_modified:
                            print(f"[Gemini识别] {key} 修正: '{ocr_text}'({ocr_value}) -> '{gemini_label.text}'({gemini_value})")
                    else:
                        print(f"[Gemini识别] {key} 识别失败：value为None，丢弃")
                else:
                    # Gemini未识别到这个标签
                    gemini_debug_info['ocr_vs_gemini'].append({
                        'axis': 'Y',
                        'key': key,
                        'ocr_text': ocr_text,
                        'ocr_value': ocr_value,
                        'gemini_text': '-',
                        'gemini_value': None,
                        'is_modified': False,
                        'recognized': False
                    })
                    print(f"[Gemini识别] {key} 未被Gemini识别，丢弃")
            
            gemini_debug_info['success'] = True
            print(f"[Gemini识别] ✅ 识别成功: X轴{len(x_labels_corrected)}个, Y轴{len(y_labels_corrected)}个")
            
            return x_labels_corrected, y_labels_corrected, gemini_debug_info
            
        except Exception as e:
            import traceback
            error_msg = f"Gemini识别失败: {str(e)}\n{traceback.format_exc()}"
            print(f"[Gemini识别] ⚠️ {error_msg}")
            gemini_debug_info['error'] = error_msg
            gemini_debug_info['success'] = False
            
            # Fallback到OCR值，但需要过滤掉value为None的标签
            print(f"[Gemini识别] Fallback到OCR值，过滤无效标签...")
            
            x_labels_filtered = []
            for label in x_labels:
                # 重新解析value确保有效
                value = self._parse_label_value(label['text'])
                if value is not None:
                    label_copy = label.copy()
                    label_copy['value'] = value
                    x_labels_filtered.append(label_copy)
                else:
                    print(f"[Gemini识别] Fallback时过滤X轴标签: {label['text']} (value=None)")
            
            y_labels_filtered = []
            for label in y_labels:
                # 重新解析value确保有效
                value = self._parse_label_value(label['text'])
                if value is not None:
                    label_copy = label.copy()
                    label_copy['value'] = value
                    y_labels_filtered.append(label_copy)
                else:
                    print(f"[Gemini识别] Fallback时过滤Y轴标签: {label['text']} (value=None)")
            
            print(f"[Gemini识别] Fallback结果: X轴{len(x_labels_filtered)}个, Y轴{len(y_labels_filtered)}个")
            
            return x_labels_filtered, y_labels_filtered, gemini_debug_info
    
    def _process_chart_with_ocr_result(self,
                                      image: np.ndarray,
                                      ocr_result: Dict,
                                      horizontal_tolerance: float,
                                      vertical_tolerance: float,
                                      interval_percentile: float,
                                      gemini_enabled: bool = False) -> Dict:
        """
        主流程方法: 处理已有OCR结果的chart提取
        
        包含核心处理流程：旋转处理、查找chart、聚类、裁剪、Gemini识别、过滤、生成调试信息
        
        Args:
            image: 输入图片
            ocr_result: OCR识别结果
            horizontal_tolerance: 水平线容差
            vertical_tolerance: 垂直线容差
            interval_percentile: 间隔分位数
            gemini_enabled: 是否启用Gemini坐标识别
            gemini_credentials: Gemini API凭据
        
        Returns:
            结果字典，包含所有提取的数据
        """
        result = {
            'success': False,
            'error': None,
            'chart_image': None,
            'chart_bbox': None,
            'x_tick_labels': [],
            'y_tick_labels': [],
            'ocr_result': ocr_result,
            'debug_info': {},
            'debug_image': None,
            'gemini_debug_info': {'enabled': False}  # 新增
        }
        
        try:
            # 步骤1: 处理OCR自动旋转
            image = self._preprocess_ocr_and_image(image, ocr_result)
            
            # 步骤2: 提取chart并聚类标签
            try:
                chart_info, cluster_result, all_candidate_labels, label_type_counts = \
                    self._extract_and_cluster_labels(ocr_result, horizontal_tolerance, vertical_tolerance)
            except ValueError as e:
                result['error'] = str(e)
                print(f"[Chart提取] 标签提取/聚类失败: {result['error']}")
                # 如果失败，尝试生成部分调试信息
                if 'chart_info' in locals():
                    result['debug_info'] = {
                        'all_candidates': all_candidate_labels if 'all_candidate_labels' in locals() else [],
                        'cluster_result': cluster_result if 'cluster_result' in locals() else {},
                        'label_type_counts': label_type_counts if 'label_type_counts' in locals() else {}
                    }
                return result
            
            x_labels_raw = cluster_result['x_labels']
            y_labels_raw = cluster_result['y_labels']
            chart_bbox = chart_info['bbox']
            
            # 步骤3: 裁剪chart并转换坐标
            chart_image, chart_bbox_tuple, x_labels_converted, y_labels_converted = \
                self._crop_and_convert_coordinates(image, chart_bbox, x_labels_raw, y_labels_raw)
            
            # 步骤3.5: Gemini坐标识别（可选）
            if gemini_enabled:
                try:
                    print("[Chart提取] 步骤3.5: 使用Gemini识别坐标...")
                    x_labels_to_filter, y_labels_to_filter, gemini_debug_info = \
                        self._recognize_coordinates_with_gemini(
                            chart_image, x_labels_converted, y_labels_converted
                        )
                    result['gemini_debug_info'] = gemini_debug_info
                except (FileNotFoundError, ValueError) as e:
                    print(f"[Chart提取] ⚠️ Gemini凭据未配置，跳过: {e}")
                    gemini_enabled = False
            
            if not gemini_enabled:
                print("[Chart提取] 跳过Gemini识别，使用OCR原始值")
                x_labels_to_filter = []
                for label in x_labels_converted:
                    value = self._parse_label_value(label['text'])
                    if value is not None:
                        label_copy = label.copy()
                        label_copy['value'] = value
                        x_labels_to_filter.append(label_copy)
                
                y_labels_to_filter = []
                for label in y_labels_converted:
                    value = self._parse_label_value(label['text'])
                    if value is not None:
                        label_copy = label.copy()
                        label_copy['value'] = value
                        y_labels_to_filter.append(label_copy)
                
                print(f"[Chart提取] OCR标签: X轴{len(x_labels_to_filter)}个, Y轴{len(y_labels_to_filter)}个")
            
            # 步骤4: 过滤并验证标签
            try:
                x_labels_filtered, y_labels_filtered = \
                    self._filter_and_validate_labels(x_labels_to_filter, y_labels_to_filter, interval_percentile)
            except ValueError as e:
                x_labels_filtered = x_labels_raw
                y_labels_filtered = y_labels_raw
                result['error'] = str(e)
                print(f"[Chart提取] 标签过滤失败: {result['error']}")
                try:
                    debug_info, debug_image = self._generate_debug_outputs(
                        image, chart_bbox, all_candidate_labels, cluster_result, label_type_counts,
                        x_labels_raw, y_labels_raw, x_labels_converted, y_labels_converted,
                        horizontal_tolerance, vertical_tolerance, interval_percentile
                    )
                    result['debug_info'] = debug_info
                    result['debug_image'] = debug_image
                except Exception:
                    pass
                return result
            
            # 组装成功结果 (before debug outputs so encoding errors don't lose labels)
            result['success'] = True
            result['chart_image'] = chart_image
            result['chart_bbox'] = chart_bbox_tuple
            result['x_tick_labels'] = x_labels_filtered
            result['y_tick_labels'] = y_labels_filtered

            # 步骤5: 生成调试信息 (non-critical)
            try:
                debug_info, debug_image = self._generate_debug_outputs(
                    image, chart_bbox, all_candidate_labels, cluster_result, label_type_counts,
                    x_labels_raw, y_labels_raw, x_labels_filtered, y_labels_filtered,
                    horizontal_tolerance, vertical_tolerance, interval_percentile
                )
                result['debug_info'] = debug_info
                result['debug_image'] = debug_image
            except Exception as dbg_err:
                print(f"[Chart] Debug output generation failed (non-critical): {dbg_err}")

            print(f"[Chart] Extraction OK: X={len(x_labels_filtered)}, Y={len(y_labels_filtered)}")
            
            return result
            
        except Exception as e:
            import traceback
            result['error'] = f"处理失败: {str(e)}\n{traceback.format_exc()}"
            print(f"[Chart提取] 处理异常: {result['error']}")
            return result
    
    # ==================== 辅助方法 ====================
    
    def _generate_debug_info(self, all_candidates, cluster_result, label_type_counts,
                           x_labels_raw, y_labels_raw, x_labels_filtered, y_labels_filtered,
                           horizontal_tolerance, vertical_tolerance, interval_percentile) -> Dict:
        """
        生成详细的调试信息
        
        Returns:
            包含各阶段处理信息的字典
        """
        debug_info = {
            'parameters': {
                'horizontal_tolerance': horizontal_tolerance,
                'vertical_tolerance': vertical_tolerance,
                'interval_percentile': interval_percentile
            },
            'stage1_collection': {
                'total_candidates': len(all_candidates),
                'label_type_counts': label_type_counts,
                'candidate_details': [
                    {
                        'text': label.get('text', ''),
                        'type': label.get('original_type', 'unknown'),
                        'cnt': label.get('cnt', [])
                    }
                    for label in all_candidates
                ]
            },
            'stage2_clustering': {
                'numeric_count': len(cluster_result.get('all_numeric', [])),
                'horizontal_clusters_count': len(cluster_result.get('horizontal_clusters', [])),
                'vertical_clusters_count': len(cluster_result.get('vertical_clusters', [])),
                'x_axis_raw_count': len(x_labels_raw),
                'y_axis_raw_count': len(y_labels_raw),
                'x_axis_raw': [
                    {
                        'text': label['text'],
                        'value': label['value'],
                        'center_x': label['center_x'],
                        'center_y': label['center_y'],
                        'original_type': label.get('original_type', 'unknown')
                    }
                    for label in x_labels_raw
                ],
                'y_axis_raw': [
                    {
                        'text': label['text'],
                        'value': label['value'],
                        'center_x': label['center_x'],
                        'center_y': label['center_y'],
                        'original_type': label.get('original_type', 'unknown')
                    }
                    for label in y_labels_raw
                ]
            },
            'stage3_filtering': {
                'x_axis_filtered_count': len(x_labels_filtered),
                'y_axis_filtered_count': len(y_labels_filtered),
                'x_axis_filtered': [
                    {
                        'text': label['text'],
                        'value': label['value'],
                        'center_x': label['center_x'],
                        'center_y': label['center_y']
                    }
                    for label in x_labels_filtered
                ],
                'y_axis_filtered': [
                    {
                        'text': label['text'],
                        'value': label['value'],
                        'center_x': label['center_x'],
                        'center_y': label['center_y']
                    }
                    for label in y_labels_filtered
                ]
            }
        }
        
        return debug_info
    
    def _generate_debug_image(self, image, chart_bbox, all_candidates, 
                            x_labels_raw, y_labels_raw, x_labels_filtered, y_labels_filtered) -> np.ndarray:
        """
        生成可视化调试图片，标注所有label的位置
        
        Args:
            image: 原始图片
            chart_bbox: chart的bbox
            all_candidates: 所有候选label
            x_labels_raw: X轴聚类结果
            y_labels_raw: Y轴聚类结果
        
        Returns:
            标注后的图片
        """
        # 复制图片用于绘制
        debug_img = image.copy()
        
        # 绘制chart边界框（蓝色）
        x1, y1 = int(chart_bbox['left']), int(chart_bbox['top'])
        x2, y2 = int(chart_bbox['right']), int(chart_bbox['bottom'])
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(debug_img, 'Chart', (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # 创建已标注label的唯一标识集合（text只用于展示，中心坐标用于判断）
        def get_center(label):
            """只返回整数化的center坐标"""
            if 'raw_center_x' in label and 'raw_center_y' in label:
                center_x = int(round(label['raw_center_x']))
                center_y = int(round(label['raw_center_y']))
                return (center_x, center_y)
            if 'center_x' in label and 'center_y' in label:
                center_x = int(round(label['center_x']))
                center_y = int(round(label['center_y']))
                return (center_x, center_y)
            cnt = label.get('cnt', [])
            if not cnt or len(cnt) < 4:
                return None
            xs = [point[0] for point in cnt]
            ys = [point[1] for point in cnt]
            center_x = int(round((min(xs) + max(xs)) / 2))
            center_y = int(round((min(ys) + max(ys)) / 2))
            return (center_x, center_y)

        # center到text的映射：以x轴、y轴分别建立映射，优先x_labels_filtered/y_labels_filtered里的text
        x_label_center2text = {
            get_center(label): label.get('text', '')
            for label in x_labels_filtered if get_center(label) is not None
        }
        y_label_center2text = {
            get_center(label): label.get('text', '')
            for label in y_labels_filtered if get_center(label) is not None
        }
        x_label_centers = set(x_label_center2text.keys())
        y_label_centers = set(y_label_center2text.keys())
        
        # 绘制所有候选label
        for label in all_candidates:
            orig_text = label.get('text', '')
            cnt = label.get('cnt', [])
            if not cnt or len(cnt) < 4:
                continue
            
            # 计算bbox
            xs = [point[0] for point in cnt]
            ys = [point[1] for point in cnt]
            left, top = int(min(xs)), int(min(ys))
            right, bottom = int(max(xs)), int(max(ys))
            
            # 根据center判断类别，并用filtered出来的text覆盖
            center = get_center(label)
            if center is not None:
                if center in x_label_centers:
                    color = (0, 255, 0)
                    label_type = 'X'
                    text = x_label_center2text[center]
                elif center in y_label_centers:
                    color = (0, 0, 255)
                    label_type = 'Y'
                    text = y_label_center2text[center]
                else:
                    color = (128, 128, 128)
                    label_type = '?'
                    text = orig_text
            else:
                color = (128, 128, 128)
                label_type = '?'
                text = orig_text
            
            # 绘制边框
            cv2.rectangle(debug_img, (left, top), (right, bottom), color, 2)
            
            # 绘制文本（加粗，带半透明白色背景）
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2  # 加粗
            
            # 获取文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # 计算背景矩形位置（留一些padding）
            padding = 2
            bg_left = left
            bg_top = top - text_height - baseline - padding * 2
            bg_right = left + text_width + padding * 2
            bg_bottom = top - padding
            
            # 创建半透明白色背景
            overlay = debug_img.copy()
            cv2.rectangle(overlay, (bg_left, bg_top), (bg_right, bg_bottom), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.6, debug_img, 0.4, 0, debug_img)
            
            # 绘制文本（使用标签颜色）
            cv2.putText(debug_img, text, 
                       (left + padding, top - padding - baseline), 
                       font, font_scale, color, font_thickness)
        
        # 添加图例
        legend_y = 30
        cv2.putText(debug_img, 'Legend:', (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(debug_img, 'Green=X-axis', (10, legend_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(debug_img, 'Red=Y-axis', (10, legend_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(debug_img, 'Gray=Rejected', (10, legend_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        cv2.putText(debug_img, 'Blue=Chart', (10, legend_y + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return debug_img
    
    def _image_to_base64(self, image_path: str) -> str:
        """将图片转换为base64编码"""
        import base64
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _find_chart_in_line_data(self, line_data: List[Dict]) -> Optional[Dict]:
        """
        从line_data中找到chart，并收集所有可能的刻度标签（5种类型）
        
        Args:
            line_data: OCR返回的line_data
        
        Returns:
            chart信息字典，包含bbox和所有候选标签
        """
        print(f"[Chart提取] 开始扫描 line_data 中的 chart，候选项数: {len(line_data)}")
        # 1. 优先找到 type="chart" 且 subtype="analytical" 的元素。
        #    如果没有，再保守地回退到 subtype="line" 且带有 children/included 的 chart。
        chart_item = None
        line_fallback_item = None
        for item in line_data:
            if item.get('type') != 'chart':
                continue
            subtype = item.get("subtype")
            if subtype == "analytical":
                chart_item = item
                break
            if (
                subtype == "line"
                and line_fallback_item is None
                and (
                    bool(item.get('children_ids'))
                    or bool(item.get('included'))
                )
            ):
                line_fallback_item = item

        if chart_item is None and line_fallback_item is not None:
            chart_item = line_fallback_item
            print(
                f"[Chart提取] 未找到 analytical chart，回退使用 line chart: "
                f"id={chart_item.get('id')}"
            )
        
        if not chart_item:
            chart_like = [
                {
                    'type': item.get('type'),
                    'subtype': item.get('subtype'),
                    'id': item.get('id'),
                    'has_children_ids': bool(item.get('children_ids')),
                    'has_included': bool(item.get('included')),
                }
                for item in line_data
                if item.get('type') == 'chart'
            ]
            print(f"[Chart提取] 未找到 analytical chart。chart 类元素: {chart_like}")
            return None
        
        # 2. 获取chart的bbox和children_ids
        bbox = chart_item.get('cnt', [])  # 注意：坐标在cnt字段，不是rect
        children_ids = chart_item.get('children_ids', [])
        print(
            f"[Chart提取] chart subtype={chart_item.get('subtype')}, id={chart_item.get('id')}, "
            f"children_ids={len(children_ids) if isinstance(children_ids, list) else 0}, "
            f"included={len(chart_item.get('included', [])) if isinstance(chart_item.get('included'), list) else 0}"
        )
        
        # 3. 根据children_ids去line_data中查找对应的标签
        # 先建立id到元素的映射
        id_to_item = {item.get('id'): item for item in line_data if 'id' in item}
        
        # 收集5种类型的label
        all_candidate_labels = []
        label_type_counts = {
            'x_axis_tick_label': 0,
            'y_axis_tick_label': 0,
            'model_label': 0,
            'legend_label': 0,
            'chart_info': 0,
            'other': 0
        }
        
        # 4. 遍历children_ids，收集所有类型的标签
        # 需要处理嵌套情况：如果label没有text但有children_ids，则递归查找
        target_types = ['x_axis_tick_label', 'y_axis_tick_label', 'model_label', 
                       'legend_label', 'chart_info']
        
        def collect_label_recursively(item_id, original_type, parent_cnt=None, visited=None):
            """
            递归收集label，处理嵌套情况
            
            Args:
                item_id: 当前item的id
                original_type: 原始类型（从顶层label继承）
                parent_cnt: 父级的cnt（如果子级没有cnt则使用父级的）
            
            Returns:
                收集到的label列表（可能为空或包含多个）
            """
            if visited is None:
                visited = set()
            if item_id in visited:
                print(f"[Chart提取] ⚠️ 检测到 children_ids 循环引用，跳过 item_id={item_id}, type={original_type}")
                return []
            visited = set(visited)
            visited.add(item_id)

            item = id_to_item.get(item_id)
            if not item:
                print(f"[Chart提取] ⚠️ children item 缺失: id={item_id}, type={original_type}")
                return []
            
            # 获取cnt，如果没有则使用父级的
            item_cnt = item.get('cnt', parent_cnt)
            
            # 如果有text，说明是leaf节点，直接返回
            if 'text' in item and item.get('text'):
                # 确保有cnt
                if not item.get('cnt'):
                    item['cnt'] = item_cnt
                # 记录原始类型
                item['original_type'] = original_type
                return [item]
            
            # 如果没有text但有children_ids，递归查找
            if 'children_ids' in item and item.get('children_ids'):
                results = []
                for child_id in item['children_ids']:
                    results.extend(
                        collect_label_recursively(child_id, original_type, item_cnt, visited)
                    )
                return results
            
            # 既没有text也没有children_ids，可能是错误数据
            return []
        
        if children_ids:
            print(f"[Chart提取] 找到chart，children_ids数量: {len(children_ids)}")
            for child_id in children_ids:
                child = id_to_item.get(child_id)
                if not child:
                    continue
                
                child_type = child.get('type', '')
                
                if child_type in target_types:
                    # 递归收集所有有text的label
                    labels = collect_label_recursively(child_id, child_type)
                    all_candidate_labels.extend(labels)
                    label_type_counts[child_type] += len(labels)
                else:
                    label_type_counts['other'] += 1
        else:
            # Fallback: chart没有children_ids，尝试从新API字段中提取标签
            print("[Chart提取] ⚠️ Chart没有children_ids，尝试从新API字段提取标签...")
            print(f"[Chart提取] chart_item keys: {list(chart_item.keys())}")
            
            # --- 策略A: 从 'included' 字段提取（新版OCR可能把子元素放在这里） ---
            included = chart_item.get('included', [])
            if included:
                print(f"[Chart提取] 发现 'included' 字段，包含 {len(included)} 个子元素")
                # included 可能是 list[dict]，每个dict类似于 line_data item
                for inc_item in included:
                    if isinstance(inc_item, dict):
                        inc_type = inc_item.get('type', '')
                        if inc_type in target_types:
                            if inc_item.get('text') and inc_item.get('cnt'):
                                inc_item['original_type'] = inc_type
                                all_candidate_labels.append(inc_item)
                                label_type_counts[inc_type] += len([inc_item])
                            elif inc_item.get('children_ids'):
                                # 把included的子元素也注册到id_to_item中
                                if 'id' in inc_item:
                                    id_to_item[inc_item['id']] = inc_item
                                for ci in inc_item.get('children_ids', []):
                                    labels = collect_label_recursively(
                                        ci, inc_type, inc_item.get('cnt')
                                    )
                                    all_candidate_labels.extend(labels)
                                    label_type_counts[inc_type] += len(labels)
            
            # --- 策略B: 从 'conversion_output' / 'data' 字段提取 ---
            if not all_candidate_labels:
                conv_output = chart_item.get('conversion_output', '')
                chart_data = chart_item.get('data', [])
                error_id = chart_item.get('error_id', '')
                if conv_output:
                    print(f"[Chart提取] conversion_output (前500字符): {str(conv_output)[:500]}")
                if chart_data:
                    print(f"[Chart提取] data字段: {str(chart_data)[:500]}")
                if error_id:
                    print(f"[Chart提取] error_id: {error_id}")
            
            # --- 策略C: 扫描line_data中所有同级标签元素 ---
            if not all_candidate_labels:
                print("[Chart提取] 尝试从line_data同级元素收集标签...")
                for item in line_data:
                    if item is chart_item:
                        continue
                    item_type = item.get('type', '')
                    if item_type in target_types:
                        labels = collect_label_recursively(item.get('id'), item_type, item.get('cnt'))
                        if not labels and item.get('text'):
                            item['original_type'] = item_type
                            labels = [item]
                        all_candidate_labels.extend(labels)
                        label_type_counts[item_type] += len(labels)
                    elif item_type == 'chart':
                        sub_children = item.get('children_ids', [])
                        for sc_id in sub_children:
                            sc = id_to_item.get(sc_id)
                            if not sc:
                                continue
                            sc_type = sc.get('type', '')
                            if sc_type in target_types:
                                labels = collect_label_recursively(sc_id, sc_type)
                                all_candidate_labels.extend(labels)
                                label_type_counts[sc_type] += len(labels)
                            else:
                                label_type_counts['other'] += 1
            
            # --- 策略D: 宽松搜索所有有text+cnt的非chart元素 ---
            if not all_candidate_labels:
                print("[Chart提取] 仍未找到标签，尝试宽松搜索所有有text+cnt的元素...")
                for item in line_data:
                    if item is chart_item:
                        continue
                    if item.get('text') and item.get('cnt'):
                        item['original_type'] = item.get('type', 'other')
                        all_candidate_labels.append(item)
                        t = item.get('type', 'other')
                        if t in label_type_counts:
                            label_type_counts[t] += 1
                        else:
                            label_type_counts['other'] += 1
            
            if not all_candidate_labels:
                print("[Chart提取] ⚠️ 所有策略均未找到任何标签")
                # Dump full line_data structure for debugging
                import json
                for i, item in enumerate(line_data):
                    print(f"[Chart提取] line_data[{i}]: keys={list(item.keys())}, type={item.get('type')}, "
                          f"subtype={item.get('subtype')}, has_text={bool(item.get('text'))}, "
                          f"has_cnt={bool(item.get('cnt'))}, has_included={bool(item.get('included'))}, "
                          f"has_children_ids={bool(item.get('children_ids'))}")
                return None
        
        print(f"[Chart提取] 收集到的label类型统计: {label_type_counts}")
        print(f"[Chart提取] 总候选label数量: {len(all_candidate_labels)}")
        
        # 5. 将cnt格式转换为bbox格式
        # cnt是[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]格式
        # 需要转换为 {left, top, right, bottom}
        if bbox and len(bbox) >= 4:
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            bbox_dict = {
                'left': min(xs),
                'top': min(ys),
                'right': max(xs),
                'bottom': max(ys)
            }
        else:
            bbox_dict = {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
        
        return {
            'bbox': bbox_dict,
            'all_candidate_labels': all_candidate_labels,
            'label_type_counts': label_type_counts
        }
    
    def _cluster_labels_by_position(self, labels: List[Dict], 
                                   horizontal_tolerance: float,
                                   vertical_tolerance: float) -> Dict:
        """
        根据位置对label进行聚类，分离出X轴和Y轴的刻度标签
        
        策略：
        1. 数值过滤：只保留能解析为数字的label
        2. 水平聚类：按center_y聚类，找最大聚类作为X轴候选
        3. 垂直聚类：按center_x聚类，找最大聚类作为Y轴候选
        
        Args:
            labels: 所有候选label列表
            horizontal_tolerance: 水平线容差（倍数）
            vertical_tolerance: 垂直线容差（倍数）
        
        Returns:
            包含x_labels和y_labels的字典
        """
        print(f"\n[{_lt('位置聚类')}] 开始对{len(labels)}个候选label进行聚类...")
        print(f"[{_lt('位置聚类')}] 容差参数: 水平={horizontal_tolerance}, 垂直={vertical_tolerance}")
        
        # 1. 数值过滤 + 计算中心点
        numeric_labels = []
        for label in labels:
            text = label.get('text', '')
            value = self._parse_label_value(text)
            
            if value is not None:
                # 计算bbox和中心点
                cnt = label.get('cnt', [])
                if not cnt or len(cnt) < 4:
                    continue
                
                xs = [point[0] for point in cnt]
                ys = [point[1] for point in cnt]
                bbox = {
                    'left': min(xs),
                    'top': min(ys),
                    'right': max(xs),
                    'bottom': max(ys)
                }
                
                center_x = (bbox['left'] + bbox['right']) / 2
                center_y = (bbox['top'] + bbox['bottom']) / 2
                
                numeric_labels.append({
                    'text': text,
                    'value': value,
                    'bbox': bbox,
                    'center_x': center_x,
                    'center_y': center_y,
                    'original_type': label.get('original_type', 'unknown')
                })
            else:
                print(f"  [{_lt('数值过滤')}] 排除非数值label: '{text}' (repr={repr(label.get('text',''))})")
        
        print(f"[{_lt('位置聚类')}] 数值过滤后: {len(numeric_labels)}个label")
        
        if len(numeric_labels) < 4:  # 至少需要X轴2个+Y轴2个
            print(f"[{_lt('位置聚类')}] ⚠️ 数值label不足4个，无法聚类")
            return {'x_labels': [], 'y_labels': [], 'all_numeric': numeric_labels}
        
        # 2. 水平聚类（X轴候选）：按center_y聚类
        print(f"\n[{_lt('位置聚类')}] 开始水平聚类（X轴候选）...")
        
        # 计算label高度中位数（用于设置阈值）
        heights = [l['bbox']['bottom'] - l['bbox']['top'] for l in numeric_labels]
        height_median = np.median(heights)
        horizontal_threshold = horizontal_tolerance * height_median
        
        print(f"  标签高度中位数: {height_median:.1f}, 阈值: {horizontal_threshold:.1f}")
        
        # 按center_y排序
        sorted_by_y = sorted(numeric_labels, key=lambda l: l['center_y'])
        
        # 贪心聚类：找到最大的水平聚类
        horizontal_clusters = []
        i = 0
        while i < len(sorted_by_y):
            cluster = [sorted_by_y[i]]
            base_y = sorted_by_y[i]['center_y']
            
            # 找到所有center_y接近的label
            j = i + 1
            while j < len(sorted_by_y):
                if abs(sorted_by_y[j]['center_y'] - base_y) <= horizontal_threshold:
                    cluster.append(sorted_by_y[j])
                    j += 1
                else:
                    break
            
            horizontal_clusters.append(cluster)
            i = j
        
        print(f"  找到{len(horizontal_clusters)}个水平聚类")
        for idx, cluster in enumerate(horizontal_clusters):
            y_values = [l['center_y'] for l in cluster]
            print(f"    聚类{idx}: {len(cluster)}个label, Y范围={min(y_values):.1f}-{max(y_values):.1f}")
        
        # 选择最大的聚类作为X轴
        x_axis_cluster = max(horizontal_clusters, key=len) if horizontal_clusters else []
        print(f"  选择最大聚类作为X轴: {len(x_axis_cluster)}个label")
        
        # 3. 垂直聚类（Y轴候选）：按center_x聚类
        print(f"\n[{_lt('位置聚类')}] 开始垂直聚类（Y轴候选）...")
        
        # 计算label宽度中位数（用于设置阈值）
        widths = [l['bbox']['right'] - l['bbox']['left'] for l in numeric_labels]
        width_median = np.median(widths)
        vertical_threshold = vertical_tolerance * width_median
        
        print(f"  标签宽度中位数: {width_median:.1f}, 阈值: {vertical_threshold:.1f}")
        
        # 按right边界排序（用于Y轴聚类，避免负号导致的中心偏移）
        sorted_by_x = sorted(numeric_labels, key=lambda l: l['bbox']['right'])
        
        # 贪心聚类：找到最大的垂直聚类
        vertical_clusters = []
        i = 0
        while i < len(sorted_by_x):
            cluster = [sorted_by_x[i]]
            base_x = sorted_by_x[i]['bbox']['right']
            
            # 找到所有right边界接近的label
            j = i + 1
            while j < len(sorted_by_x):
                if abs(sorted_by_x[j]['bbox']['right'] - base_x) <= vertical_threshold:
                    cluster.append(sorted_by_x[j])
                    j += 1
                else:
                    break
            
            vertical_clusters.append(cluster)
            i = j
        
        print(f"  找到{len(vertical_clusters)}个垂直聚类")
        for idx, cluster in enumerate(vertical_clusters):
            x_values = [l['bbox']['right'] for l in cluster]
            print(f"    聚类{idx}: {len(cluster)}个label, Right范围={min(x_values):.1f}-{max(x_values):.1f}")
        
        # 选择最大的聚类作为Y轴
        y_axis_cluster = max(vertical_clusters, key=len) if vertical_clusters else []
        print(f"  选择最大聚类作为Y轴: {len(y_axis_cluster)}个label")
        
        print(f"\n[{_lt('位置聚类')}] 聚类完成: X轴{len(x_axis_cluster)}个, Y轴{len(y_axis_cluster)}个")
        
        # 4. 处理重复label（同时出现在X轴和Y轴）
        print(f"\n[{_lt('位置聚类')}] 检查并处理重复label...")
        
        # 创建label的唯一标识（使用center坐标）
        def make_label_key(label):
            return (label['text'], round(label['center_x'], 1), round(label['center_y'], 1))
        
        x_keys = {make_label_key(l): l for l in x_axis_cluster}
        y_keys = {make_label_key(l): l for l in y_axis_cluster}
        
        # 找到重复的label
        duplicate_keys = set(x_keys.keys()) & set(y_keys.keys())
        
        if duplicate_keys:
            print(f"  发现{len(duplicate_keys)}个重复label，需要分配到正确的轴")
            
            # 计算X轴和Y轴的平均中心位置
            x_axis_center_y = np.mean([l['center_y'] for l in x_axis_cluster])
            y_axis_center_x = np.mean([l['center_x'] for l in y_axis_cluster])
            
            print(f"  X轴平均center_y: {x_axis_center_y:.1f}")
            print(f"  Y轴平均center_x: {y_axis_center_x:.1f}")
            
            for key in duplicate_keys:
                label = x_keys[key]  # 从任一个取都一样
                text = label['text']
                original_type = label.get('original_type', 'unknown')
                
                # 策略1: 优先根据原始类型判断
                if original_type == 'x_axis_tick_label':
                    # 保留在X轴，从Y轴移除
                    y_keys.pop(key)
                    print(f"    label '{text}' 根据类型({original_type})分配到X轴")
                elif original_type == 'y_axis_tick_label':
                    # 保留在Y轴，从X轴移除
                    x_keys.pop(key)
                    print(f"    label '{text}' 根据类型({original_type})分配到Y轴")
                else:
                    # 策略2: 对于其他类型，根据距离判断
                    dist_to_x = abs(label['center_y'] - x_axis_center_y)
                    dist_to_y = abs(label['center_x'] - y_axis_center_x)
                    
                    if dist_to_x < dist_to_y:
                        # 更接近X轴，从Y轴移除
                        y_keys.pop(key)
                        print(f"    label '{text}' (类型={original_type}) 根据距离分配到X轴 (距离={dist_to_x:.1f} < {dist_to_y:.1f})")
                    else:
                        # 更接近Y轴，从X轴移除
                        x_keys.pop(key)
                        print(f"    label '{text}' (类型={original_type}) 根据距离分配到Y轴 (距离={dist_to_y:.1f} <= {dist_to_x:.1f})")
            
            # 更新聚类结果
            x_axis_cluster = list(x_keys.values())
            y_axis_cluster = list(y_keys.values())
            
            print(f"  去重后: X轴{len(x_axis_cluster)}个, Y轴{len(y_axis_cluster)}个")
        else:
            print(f"  没有发现重复label")
        
        return {
            'x_labels': x_axis_cluster,
            'y_labels': y_axis_cluster,
            'all_numeric': numeric_labels,
            'horizontal_clusters': horizontal_clusters,
            'vertical_clusters': vertical_clusters
        }
    
    def _convert_labels_to_chart_coords(self, labels: List[Dict], 
                                       chart_x: int, chart_y: int) -> List[Dict]:
        """
        将刻度标签的坐标从相对原图转换为相对chart
        
        Args:
            labels: 刻度标签列表
            chart_x: chart在原图中的x偏移
            chart_y: chart在原图中的y偏移
        
        Returns:
            转换后的标签列表
        """
        converted = []
        
        for label in labels:
            # 从cnt字段获取坐标（格式：[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]）
            cnt = label.get('cnt', [])
            
            if not cnt or len(cnt) < 4:
                continue
            
            # 将cnt转换为bbox格式
            xs = [point[0] for point in cnt]
            ys = [point[1] for point in cnt]
            bbox = {
                'left': min(xs),
                'top': min(ys),
                'right': max(xs),
                'bottom': max(ys)
            }
            
            # 转换坐标（从相对原图转为相对chart）
            converted_label = {
                'text': label.get('text', ''),
                'value': self._parse_label_value(label.get('text', '')),
                'bbox': {
                    'left': bbox['left'] - chart_x,
                    'top': bbox['top'] - chart_y,
                    'right': bbox['right'] - chart_x,
                    'bottom': bbox['bottom'] - chart_y
                }
            }
            
            # 计算中心点
            converted_label['center_x'] = (converted_label['bbox']['left'] + 
                                          converted_label['bbox']['right']) / 2
            converted_label['center_y'] = (converted_label['bbox']['top'] + 
                                          converted_label['bbox']['bottom']) / 2
            
            converted.append(converted_label)
        
        return converted
    
    def _parse_label_value(self, text: str) -> Optional[float]:
        """
        解析标签文本为数值，根据配置支持不同类型：
        - 整数：1, -2, 0（根据 allow_integer）
        - 小数：0.5, -1.5（根据 allow_float）
        - 分数：1/2, -3/4（根据 allow_fraction）
        - 带π的：π, 2π, π/2（根据 allow_pi）
        - 平方根：√2（根据 allow_sqrt）
        - 负号变体：−, –, -（始终支持）
        - LaTeX命令：始终处理
        
        Args:
            text: 标签文本
        
        Returns:
            数值，如果无法解析或不符合配置则返回None
        """
        try:
            # 移除空格
            text = text.strip()

            # 先替换 unicode 负号变体为 ASCII（在清理不可见字符之前）
            text = text.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '-')

            # 移除不可见 Unicode 字符 (zero-width spaces, soft hyphens, BOM, etc.)
            text = re.sub(r'[\u200b\u200c\u200d\u00ad\ufeff\u2060\u200e\u200f\u00a0]', '', text)

            # 移除 \( ... \)
            text = re.sub(r'\\?\\\((.+?)\\?\\\)', r'\1', text)
            # 移除 \[ ... \]
            text = re.sub(r'\\?\\\[(.+?)\\?\\\]', r'\1', text)
            # 移除 $$ ... $$
            text = re.sub(r'\$\$(.+?)\$\$', r'\1', text)
            # 移除 $ ... $
            text = re.sub(r'\$(.+?)\$', r'\1', text)
            
            # 再次去除空格
            text = text.strip()
            
            if not text:
                return None
            
            # 处理 LaTeX 命令
            # 1. 处理 \frac{分子}{分母} - 转换为 (分子)/(分母)
            frac_pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
            text = re.sub(frac_pattern, r'(\1)/(\2)', text)
            
            # 2. 处理 \sqrt{数值}
            if '\\sqrt' in text:
                if not self.label_parse_config['allow_sqrt']:
                    # 不支持平方根，直接返回None
                    return None
                
                # 支持平方根，计算
                def replace_sqrt(match):
                    try:
                        inner = match.group(1)
                        # 递归处理内部表达式（可能包含π等）
                        inner = inner.replace('\\pi', str(math.pi))
                        value = ast.literal_eval(inner)
                        return str(math.sqrt(value))
                    except:
                        return match.group(0)  # 解析失败，保持原样
                
                text = re.sub(r'\\sqrt\{([^}]+)\}', replace_sqrt, text)
            
            # 3. 替换希腊字母 LaTeX 命令为对应符号
            text = text.replace('\\pi', 'π')
            text = text.replace('\\Pi', 'π')
            # 其他常见希腊字母（如果坐标轴用到）
            text = text.replace('\\theta', 'θ')
            text = text.replace('\\Theta', 'Θ')
            text = text.replace('\\alpha', 'α')
            text = text.replace('\\beta', 'β')
            text = text.replace('\\gamma', 'γ')
            text = text.replace('\\delta', 'δ')
            text = text.replace('\\lambda', 'λ')
            text = text.replace('\\mu', 'μ')
            text = text.replace('\\sigma', 'σ')
            text = text.replace('\\omega', 'ω')
            
            # 4. 移除可能的空格
            text = text.replace(' ', '')
            
            # 替换负号变体为标准负号
            text = text.replace('−', '-').replace('–', '-')
            
            # 检查是否包含π
            if 'π' in text or 'pi' in text.lower():
                if not self.label_parse_config['allow_pi']:
                    # 不支持π，返回None
                    return None
                
                # 支持π，计算数值
                # 统一替换为 pi
                text = text.replace('π', '*pi').replace('Pi', '*pi').replace('PI', '*pi')
                # 处理开头的*pi（如：π -> *pi，应该是 pi）
                if text.startswith('*pi'):
                    text = 'pi' + text[3:]
                # 替换pi为数值
                text = text.replace('pi', str(math.pi))
                value = ast.literal_eval(text)
                return float(value)
            
            # 检查是否包含分数
            if '/' in text:
                if not self.label_parse_config['allow_fraction']:
                    # 不支持分数，返回None
                    return None
                
                # 支持分数，计算
                parts = text.split('/')
                if len(parts) == 2:
                    numerator = ast.literal_eval(parts[0].strip())
                    denominator = ast.literal_eval(parts[1].strip())
                    if denominator != 0:
                        value = float(numerator / denominator)
                        # 检查结果是否符合配置
                        if value.is_integer():
                            # 是整数
                            if self.label_parse_config['allow_integer']:
                                return value
                            else:
                                return None
                        else:
                            # 是小数
                            if self.label_parse_config['allow_float']:
                                return value
                            else:
                                return None
                    else:
                        return None
                else:
                    # 多个/，不是合法分数
                    return None
            
            # 其他情况：必须是单个数字（正数、负数、小数）
            # 不能包含运算符（除了开头的负号）
            # 检查是否包含多个运算符（说明不是单个数字）
            if text.count('-') > 1 or '+' in text or '*' in text:
                # 包含多个减号或其他运算符，不是单个数字
                return None
            
            # 尝试直接转换为float
            value = float(text)
            
            # 检查结果是否符合配置
            if value.is_integer():
                # 是整数
                if self.label_parse_config['allow_integer']:
                    return value
                else:
                    return None
            else:
                # 是小数
                if self.label_parse_config['allow_float']:
                    return value
                else:
                    return None
            
        except Exception:
            return None
    
    def _filter_ocr_labels(self, labels: List[Dict], axis: str, interval_percentile: float = 50) -> List[Dict]:
        """
        过滤OCR识别错误的刻度标签
        
        策略（按执行顺序）：
        0. 位置聚类：删除不在主流位置线上的标签
           - X轴：删除center_y偏离主流的标签（应该在同一水平线）
           - Y轴：删除center_x偏离主流的标签（应该在同一垂直线）
        1. 按位置排序
        2. 间隔过滤：基于比率一致性删除异常间隔的两端（增强版，包括重复值异常检测）
           - 计算像素间隔/数值间隔的比率
           - 使用RANSAC找到主流比率
           - 删除比率异常的间隔两端
           - 删除重复值（数值间隔为0）的间隔两端
        3. 单调性检查：值应该是单调的（X轴递增，Y轴递减），允许重复值
           - 反复删除破坏单调性的标签
        4. 重复值处理：基于像素位置的均匀性选择保留哪个重复值
           - 计算所有非重复标签的平均像素间距
           - 选择像素间距最接近平均间距的候选（位置最合理）
        
        Args:
            labels: 刻度标签列表
            axis: 'x' 或 'y'
            interval_percentile: 间隔分位数（0-100），默认50（中位数）
        
        Returns:
            过滤后的标签列表
        """
        if len(labels) < 2:
            return labels
        
        print(f"\n[OCR标签过滤-{axis.upper()}轴] 开始过滤...")
        print(f"  原始标签数: {len(labels)}")
        
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
            sizes = [l['bbox']['bottom'] - l['bbox']['top'] for l in labels]  # 高度
        else:
            sizes = [l['bbox']['right'] - l['bbox']['left'] for l in labels]  # 宽度
        
        size_median = np.median(sizes)
        position_threshold = 1.5 * size_median  # 允许偏离1.5倍标签尺寸
        
        print(f"  位置聚类过滤（{position_name}）：")
        print(f"    中位数: {position_median:.1f}, 阈值: ±{position_threshold:.1f}")
        
        # 过滤偏离主流位置的标签
        clustered_labels = []
        for l in labels:
            pos = l[position_key]
            deviation = abs(pos - position_median)
            if deviation <= position_threshold:
                clustered_labels.append(l)
                print(f"    [OK] 保留: 值={l['value']}, {position_name}={pos:.1f}, 偏离={deviation:.1f}")
            else:
                print(f"    [RM] 删除: 值={l['value']}, {position_name}={pos:.1f}, 偏离={deviation:.1f} (超出阈值)")
        
        print(f"  位置聚类后: {len(clustered_labels)}个标签")
        
        if len(clustered_labels) < 2:
            print(f"  ⚠️ 过滤后标签不足2个，返回聚类结果")
            return clustered_labels
        
        # 1. 按位置排序
        if axis == 'x':
            # X轴：从左到右排序
            sorted_labels = sorted(clustered_labels, key=lambda l: l['center_x'])
            print(f"  排序方向: 从左到右（按center_x）")
        else:
            # Y轴：从上到下排序
            sorted_labels = sorted(clustered_labels, key=lambda l: l['center_y'])
            print(f"  排序方向: 从上到下（按center_y）")
        
        print(f"  排序后的值序列: {[l['value'] for l in sorted_labels]}")
        
        # 2. 间隔过滤：基于比率一致性删除异常间隔的两端（包括重复值异常）
        # 确定使用哪个位置坐标
        if axis == 'x':
            pos_key = 'center_x'  # X轴标签按水平位置排序
        else:
            pos_key = 'center_y'  # Y轴标签按垂直位置排序
        
        if len(sorted_labels) < 3:
            print(f"  标签数量不足3个，跳过间隔过滤")
        else:
            # 2.1 计算相邻标签的数值间隔和像素间隔
            value_intervals = []  # 数值间隔
            pixel_intervals = []  # 像素间隔
            
            for i in range(len(sorted_labels) - 1):
                # 数值间隔
                val_diff = abs(sorted_labels[i + 1]['value'] - sorted_labels[i]['value'])
                value_intervals.append(val_diff)
                
                # 像素间隔
                pixel_diff = abs(sorted_labels[i + 1][pos_key] - sorted_labels[i][pos_key])
                pixel_intervals.append(pixel_diff)
            
            print(f"  数值间隔序列: {[f'{v:.1f}' for v in value_intervals]}")
            print(f"  像素间隔序列: {[f'{p:.1f}' for p in pixel_intervals]}")
            
            # 2.2 基于比率一致性的检查：找到大多数比率一致的间隔
            print(f"\n  === 比率一致性检查 ===")
            
            # 计算每个间隔的像素/数值比率（px per unit）
            ratios = []
            for i in range(len(sorted_labels) - 1):
                value_interval = value_intervals[i]
                pixel_interval = pixel_intervals[i]
                
                if value_interval > 0:
                    ratio = pixel_interval / value_interval
                    ratios.append(ratio)
                    print(f"  间隔{i}: [{sorted_labels[i]['value']}]→[{sorted_labels[i+1]['value']}] "
                          f"像素={pixel_interval:.1f}px, 数值={value_interval:.1f}, 比率={ratio:.2f} px/unit")
                else:
                    # 数值间隔为0（重复值），视为潜在异常
                    ratios.append(None)
                    print(f"  间隔{i}: [{sorted_labels[i]['value']}]→[{sorted_labels[i+1]['value']}] "
                          f"数值间隔为0（重复值），标记为潜在异常")
            
            # 过滤掉None值，得到有效比率
            valid_ratios = [r for r in ratios if r is not None]
            
            # 初始化异常间隔列表
            abnormal_intervals = []
            
            if len(valid_ratios) < 2:
                print(f"  ⚠️ 有效比率不足2个，跳过比率检查")
            else:
                # 多轮聚类：找到大多数一致的比率
                print(f"\n  === 多轮聚类找主流比率 ===")
                
                # 计算比率的中位数和标准差
                ratio_median = np.median(valid_ratios)
                ratio_std = np.std(valid_ratios)
                
                print(f"  初始比率中位数: {ratio_median:.2f} px/unit")
                print(f"  初始比率标准差: {ratio_std:.2f}")
                
                # 使用RANSAC思想：反复采样，找最大一致集
                max_inlier_count = 0
                best_inlier_indices = []
                best_ratio_median = ratio_median
                
                # 多轮检查，每次以不同的比率为基准
                for trial in range(len(valid_ratios)):
                    base_ratio = valid_ratios[trial]
                    
                    # 找到与当前基准比率接近的所有间隔（±20%容差）
                    inlier_indices = []
                    for idx, ratio in enumerate(ratios):
                        if ratio is not None:
                            deviation = abs(ratio - base_ratio) / base_ratio
                            if deviation <= 0.2:  # 20%容差
                                inlier_indices.append(idx)
                    
                    # 更新最大一致集
                    if len(inlier_indices) > max_inlier_count:
                        max_inlier_count = len(inlier_indices)
                        best_inlier_indices = inlier_indices
                        # 计算一致集的中位数比率
                        inlier_ratios = [ratios[idx] for idx in inlier_indices]
                        best_ratio_median = np.median(inlier_ratios)
                
                print(f"\n  找到最大一致集: {max_inlier_count}/{len(valid_ratios)}个间隔")
                print(f"  主流比率: {best_ratio_median:.2f} px/unit")
                print(f"  一致集间隔索引: {best_inlier_indices}")
                
                # 标记异常间隔（不在一致集中的，包括重复值）
                for idx in range(len(ratios)):
                    ratio = ratios[idx]
                    
                    if ratio is None:
                        # 重复值（数值间隔为0），标记为异常
                        abnormal_intervals.append((
                            idx, idx + 1, 
                            value_intervals[idx], 
                            pixel_intervals[idx],
                            f"重复值异常: 数值间隔为0"
                        ))
                        print(f"  ⚠️ 异常间隔{idx}: [{sorted_labels[idx]['value']}]→[{sorted_labels[idx+1]['value']}]")
                        print(f"      原因: 重复值（数值间隔为0）")
                    elif idx not in best_inlier_indices:
                        # 比率异常
                        deviation = abs(ratio - best_ratio_median) / best_ratio_median
                        
                        abnormal_intervals.append((
                            idx, idx + 1, 
                            value_intervals[idx], 
                            pixel_intervals[idx],
                            f"比率异常: {ratio:.2f} px/unit (偏离主流{deviation*100:.1f}%)"
                        ))
                        
                        print(f"  ⚠️ 异常间隔{idx}: [{sorted_labels[idx]['value']}]→[{sorted_labels[idx+1]['value']}]")
                        print(f"      比率={ratio:.2f} px/unit, 主流={best_ratio_median:.2f}, 偏离={deviation*100:.1f}%")
            
            # 2.3 删除异常间隔的两端标签
            keep_flags = [True] * len(sorted_labels)
            
            for i, j, value_interval, pixel_interval, reason in abnormal_intervals:
                # 异常间隔表示这段有问题，两端标签都可能有误，全部删除
                if keep_flags[i]:
                    keep_flags[i] = False
                    print(f"  [RM] 删除标签 {sorted_labels[i]['value']} (位置{i}, 异常间隔的左端)")
                    print(f"      原因: {reason}")
                if keep_flags[j]:
                    keep_flags[j] = False
                    print(f"  [RM] 删除标签 {sorted_labels[j]['value']} (位置{j}, 异常间隔的右端)")
                    print(f"      原因: {reason}")
            
            # 应用过滤
            sorted_labels = [label for i, label in enumerate(sorted_labels) if keep_flags[i]]
            
            # 检查过滤后的标签数量
            if len(sorted_labels) < 2:
                print(f"  ⚠️ 间隔过滤后标签不足2个，返回原始标签")
                return labels
            
            # 检查是否至少有2个不同的值
            unique_values_after_interval = set(l['value'] for l in sorted_labels)
            if len(unique_values_after_interval) < 2:
                print(f"  ❌ 间隔过滤后只剩{len(unique_values_after_interval)}个不同的值，OCR识别失败")
                print(f"  当前保留的值: {list(unique_values_after_interval)}")
                return []  # 返回空列表表示失败
            
            print(f"  间隔过滤后: {len(sorted_labels)}个标签，{len(unique_values_after_interval)}个不同值")
            print(f"  过滤后的值序列: {[l['value'] for l in sorted_labels]}")
        
        # 3. 单调性检查：按位置排序后，值应该是单调的（X轴递增，Y轴递减），允许重复值
        if len(sorted_labels) < 3:
            print(f"\n  标签数量不足3个，跳过单调性检查")
        else:
            print(f"\n  === 单调性检查 ===")
            
            # 3.1 计算相邻差值，判断主流趋势
            diffs = []
            for i in range(len(sorted_labels) - 1):
                diff = sorted_labels[i + 1]['value'] - sorted_labels[i]['value']
                diffs.append(diff)
            
            print(f"  相邻差值: {[f'{d:.1f}' for d in diffs]}")
            
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
            
            print(f"  期望趋势: {expected_trend}（允许重复值）")
            print(f"  正向差值数: {positive_count}, 负向差值数: {negative_count}")
            
            # 3.2 反复删除不符合主流趋势的点
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
                            print(f"  迭代{iteration}: 删除标签 {removed_label['value']} (位置{i}, 破坏{expected_trend}趋势)")
                        else:
                            removed_label = sorted_labels.pop(i + 1)
                            print(f"  迭代{iteration}: 删除标签 {removed_label['value']} (位置{i+1}, 破坏{expected_trend}趋势)")
                        
                        removed_in_this_iteration = True
                        break  # 删除一个后重新开始
                
                if not removed_in_this_iteration:
                    print(f"  单调性检查完成（迭代{iteration}次）")
                    break
            
            if len(sorted_labels) < 2:
                print(f"  ⚠️ 单调性过滤后标签不足2个，返回原始标签")
                return labels
            
            # 检查是否至少有2个不同的值
            unique_values_after_monotone = set(l['value'] for l in sorted_labels)
            if len(unique_values_after_monotone) < 2:
                print(f"  ❌ 单调性过滤后只剩{len(unique_values_after_monotone)}个不同的值，OCR识别失败")
                print(f"  当前保留的值: {list(unique_values_after_monotone)}")
                return []  # 返回空列表表示失败
            
            print(f"  单调性过滤后: {len(sorted_labels)}个标签，{len(unique_values_after_monotone)}个不同值")
            print(f"  过滤后的值序列: {[l['value'] for l in sorted_labels]}")
        
        # 4. 处理重复值（基于间隔一致性）
        if len(sorted_labels) >= 3:
            print(f"\n  检查重复值...")
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
                    print(f"    发现重复值: {current_label['value']} ({len(duplicates)}个)")
                    
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
                        print(f"      平均像素间距（参考）: {avg_pixel_spacing:.1f}")
                    
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
                        
                        print(f"      候选 值={candidate['value']}, 位置={candidate_pos:.1f}, 像素间距={[f'{x:.1f}' for x in pixel_intervals]}, 得分={score:.2f}")
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = candidate
                    
                    # 保留最佳候选
                    deduplicated.append(best_candidate)
                    pos = best_candidate[pos_key]
                    print(f"    [OK] 保留: 值={best_candidate['value']}, 位置={pos:.1f}")
                    removed_duplicates += len(duplicates) - 1
                else:
                    # 没有重复，直接添加
                    deduplicated.append(current_label)
                
                i = j  # 跳到下一组
            
            if removed_duplicates > 0:
                print(f"  重复值处理: 删除了{removed_duplicates}个重复标签")
                sorted_labels = deduplicated
                print(f"  去重后的值序列: {[l['value'] for l in sorted_labels]}")
        
        # 最后检查：必须至少有2个不同的值
        unique_values = set(l['value'] for l in sorted_labels)
        if len(unique_values) < 2:
            print(f"\n[OCR标签过滤-{axis.upper()}轴] ❌ 过滤后只有{len(unique_values)}个不同的值，OCR识别失败")
            print(f"  当前保留的值: {list(unique_values)}")
            print(f"  至少需要2个不同的值才能建立坐标映射")
            return []  # 返回空列表表示失败
        
        print(f"\n[OCR标签过滤-{axis.upper()}轴] 完成")
        print(f"  最终保留: {len(sorted_labels)}个标签")
        print(f"  不同值数量: {len(unique_values)}个")
        print(f"  最终值序列: {[l['value'] for l in sorted_labels]}")
        print(f"  最终bbox序列: {[l['bbox'] for l in sorted_labels]}")
        
        return sorted_labels

