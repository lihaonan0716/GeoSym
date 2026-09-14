"""
坐标系识别核心处理器

提供统一的处理接口，供API和Gradio共用
"""

import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
import base64, time
import requests
from io import BytesIO
from PIL import Image

from .chart_extractor import ChartExtractor
from .image_enhancer import ImageEnhancer
from .coordinate_recognizer import CoordinateRecognizer
from .coordinate_system_preprocessor import CoordinateSystemPreprocessor

import sys as _sys
from pathlib import Path as _Path
_exp_root = _Path(__file__).resolve().parent.parent.parent
if str(_exp_root) not in _sys.path:
    _sys.path.insert(0, str(_exp_root))

class CoordinateProcessor:
    """
    坐标系识别核心处理器
    
    统一的处理接口，封装完整的处理流程
    """
    
    def __init__(self):
        """初始化处理器"""
        self.chart_extractor = ChartExtractor()
        self.image_enhancer = ImageEnhancer()
        self.preprocessor = CoordinateSystemPreprocessor()
        self.coordinate_recognizer = CoordinateRecognizer()
    
    def load_image(self, image_input: str) -> Optional[np.ndarray]:
        """
        从base64或URL加载图像
        
        Args:
            image_input: base64编码的图像字符串（可带前缀）或图像URL
        
        Returns:
            OpenCV格式的图像(BGR)，失败返回None
        """
        try:
            # 判断是URL还是base64
            if image_input.startswith('http://') or image_input.startswith('https://'):
                # 从URL加载
                print(f"[ImageLoad] Loading image from URL: {image_input[:50]}...")
                response = requests.get(image_input, timeout=30)
                response.raise_for_status()
                image_data = response.content
                pil_image = Image.open(BytesIO(image_data))
            else:
                # 从base64加载
                print(f"[ImageLoad] Loading image from base64")
                # 移除可能的前缀
                if ',' in image_input:
                    image_input = image_input.split(',', 1)[1]
                
                image_data = base64.b64decode(image_input)
                pil_image = Image.open(BytesIO(image_data))
            
            # 转换为OpenCV格式
            image_array = np.array(pil_image)
            
            # 处理不同的颜色格式
            if len(image_array.shape) == 2:
                # 灰度图
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            elif image_array.shape[2] == 3:
                # RGB转BGR
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            elif image_array.shape[2] == 4:
                # RGBA转BGR
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(f"textsupportstextimagetext: {image_array.shape}")
            
            print(f"[ImageLoad] Succeeded, size: {image_bgr.shape[1]}x{image_bgr.shape[0]}")
            return image_bgr
            
        except Exception as e:
            print(f"[ImageLoad] Failed: {str(e)}")
            return None

    def _chart_image_to_base64(self, image: Optional[np.ndarray], quality: int = 90) -> Optional[str]:
        """
        将 numpy 图像编码为 Base64 字符串（JPEG）。

        Args:
            image: OpenCV BGR 图像
            quality: JPEG 压缩质量

        Returns:
            str: data URI 格式的 Base64 字符串，或 None（图像为空或编码失败）
        """
        if image is None:
            return None

        success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            print(f"[ImageEncode] chart_image textFailed")
            return None

        base64_str = base64.b64encode(buffer.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_str}"
    
    def extract_chart_from_ocr(self, 
                               image: np.ndarray,
                               ocr_result: Dict,
                               horizontal_tolerance: float = 1.5,
                               vertical_tolerance: float = 1.5,
                               interval_percentile: float = 50) -> Dict:
        """
        从OCR结果中提取chart（不调用OCR API）
        
        Args:
            image: 原始题目图片
            ocr_result: OCR返回的OCR结果（JSON）
            horizontal_tolerance: 水平线容差倍数
            vertical_tolerance: 垂直线容差倍数
            interval_percentile: 间隔分位数
        
        Returns:
            结果字典，包含：
            - success: 是否成功
            - chart_image: 裁剪的chart图像
            - chart_bbox: chart边界框 (x, y, w, h)
            - x_tick_labels: X轴刻度标签
            - y_tick_labels: Y轴刻度标签
            - ocr_result: 完整OCR结果
            - debug_info: 调试信息
        """
        print("\n" + "="*50)
        print("Stage 0: chart extraction from OCR data")
        print("="*50)
        print(f"[Params] horizontal_tolerance = {horizontal_tolerance}")
        print(f"[Params] vertical_tolerance = {vertical_tolerance}")
        print(f"[Params] interval_percentile = {interval_percentile}")
        
        try:
            # 使用ChartExtractor处理OCR结果（不调用API）
            result = self.chart_extractor.extract_chart_from_ocr_result(
                image=image,
                ocr_result=ocr_result,
                horizontal_tolerance=horizontal_tolerance,
                vertical_tolerance=vertical_tolerance,
                interval_percentile=interval_percentile
            )
            
            if result['success']:
                print(f"[Charttext] text")
                print(f"  - Chart region: {result['chart_bbox']}")
                print(f"  - X-axis label count: {len(result['x_tick_labels'])}")
                print(f"  - Y-axis label count: {len(result['y_tick_labels'])}")
            else:
                print(f"[Charttext] Failed: {result.get('error', 'unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"[Charttext] Exception: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Chart提取异常: {str(e)}'
            }
    
    def enhance_image(self, 
                     image: np.ndarray,
                     model_id: str = "gemini",
                     prompt: Optional[str] = None) -> Dict:
        """
        阶段0.5: AI图像增强（可选）
        
        Args:
            image: 输入图像
            model_id: 模型ID
            prompt: 提示词，None时使用默认
        
        Returns:
            结果字典，包含：
            - success: 是否成功
            - processed_image: 增强后的图像
            - model_used: 使用的模型
            - error: 错误信息（如果失败）
        """
        print("\n" + "="*50)
        print(f"Stage 0.5: AI image enhancement (model: {model_id})")
        print("="*50)
        
        try:
            result = self.image_enhancer.process_image(
                image=image,
                model_name=model_id,
                prompt=prompt
            )
            
            if result['success']:
                print(f"[AIEnhance] text")
                # 尺寸调整：增强后的图像调整回原图尺寸
                enhanced_image = result['processed_image']
                transform_params = None
                if enhanced_image.shape[:2] != image.shape[:2]:
                    print(f"[AIEnhance] Resized: {enhanced_image.shape[1]}x{enhanced_image.shape[0]} -> {image.shape[1]}x{image.shape[0]}")
                    enhanced_image, transform_params = self.preprocessor.resize_to_match_original(
                        enhanced_image, 
                        image.shape[:2]
                    )
                    result['processed_image'] = enhanced_image
                    result['ai_transform_params'] = transform_params
                else:
                    # 尺寸相同，也记录变换参数（虽然是单位变换）
                    result['ai_transform_params'] = {
                        'scale': 1.0,
                        'crop_x': 0,
                        'crop_y': 0,
                        'ai_output_size': (enhanced_image.shape[1], enhanced_image.shape[0]),
                        'original_size': (image.shape[1], image.shape[0])
                    }
            else:
                print(f"[AIEnhance] Failed: {result.get('error', 'unknown error')}")
            
            result['model_used'] = model_id
            return result
            
        except Exception as e:
            print(f"[AIEnhance] Exception: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'AI增强异常: {str(e)}',
                'model_used': model_id
            }
    
    def preprocess(self,
                  image: np.ndarray,
                  enable_curve_erase: bool = True,
                  enable_dashed_line_connection: bool = True,
                  enable_perspective_correction: bool = True,
                  enable_rotation_correction: bool = True,
                  x_tick_labels: Optional[List[Dict]] = None,
                  y_tick_labels: Optional[List[Dict]] = None,
                  rotation_hough_threshold: int = 65,
                  rotation_min_line_length: int = 60,
                  rotation_max_line_gap: int = 10,
                  rotation_angle_threshold: int = 5,
                  grid_hough_threshold: int = 65,
                  grid_min_line_length: int = 45,
                  grid_max_line_gap: int = 10,
                  grid_angle_threshold: int = 5,
                  merge_distance_threshold: int = 20,
                  dashed_gap_threshold: int = 10,
                  dashed_collinear_threshold: int = 3,
                  curve_saturation_threshold: int = 90,
                  curve_value_threshold: int = 60,
                  curve_feather_size: int = 3) -> Dict:
        """
        阶段1: 图像预处理
        
        包括倾斜校正、网格线检测、线段合并等
        
        Args:
            image: 输入图像
            enable_curve_erase: 是否擦除彩色曲线
            enable_dashed_line_connection: 是否连接虚线
            enable_perspective_correction: 是否透视校正
            enable_rotation_correction: 是否倾斜校正
            ...: 其他预处理参数
        
        Returns:
            结果字典，包含：
            - success: 是否成功
            - processed_image: 预处理后的图像
            - grid_h_lines: 水平网格线位置列表
            - grid_v_lines: 垂直网格线位置列表
            - intermediate_results: 中间结果字典
            - error: 错误信息（如果失败）
        """
        print("\n" + "="*50)
        print("Stage 1: image preprocessing")
        print("="*50)
        print(f"[Toggle] enable_curve_erase = {enable_curve_erase}")
        print(f"[Toggle] enable_dashed_line_connection = {enable_dashed_line_connection}")
        print(f"[Toggle] enable_perspective_correction = {enable_perspective_correction}")
        print(f"[Toggle] enable_rotation_correction = {enable_rotation_correction}")
        print(f"[RotCorrection] rotation_hough_threshold = {rotation_hough_threshold}")
        print(f"[RotCorrection] rotation_min_line_length = {rotation_min_line_length}")
        print(f"[RotCorrection] rotation_max_line_gap = {rotation_max_line_gap}")
        print(f"[RotCorrection] rotation_angle_threshold = {rotation_angle_threshold}")
        print(f"[GridDetect] grid_hough_threshold = {grid_hough_threshold}")
        print(f"[GridDetect] grid_min_line_length = {grid_min_line_length}")
        print(f"[GridDetect] grid_max_line_gap = {grid_max_line_gap}")
        print(f"[GridDetect] grid_angle_threshold = {grid_angle_threshold}")
        print(f"[LineMerge] merge_distance_threshold = {merge_distance_threshold}")
        print(f"[DashedConn] dashed_gap_threshold = {dashed_gap_threshold}")
        print(f"[DashedConn] dashed_collinear_threshold = {dashed_collinear_threshold}")
        print(f"[CurveErase] curve_saturation_threshold = {curve_saturation_threshold}")
        print(f"[CurveErase] curve_value_threshold = {curve_value_threshold}")
        print(f"[CurveErase] curve_feather_size = {curve_feather_size}")
        
        try:
            # process() 返回 (processed_image, intermediate_results)
            processed_image, intermediate_results = self.preprocessor.process(
                image=image,
                enable_curve_erase=enable_curve_erase,
                enable_dashed_line_connection=enable_dashed_line_connection,
                enable_perspective_correction=enable_perspective_correction,
                enable_rotation_correction=enable_rotation_correction,
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels,
                rotation_hough_threshold=rotation_hough_threshold,
                rotation_min_line_length=rotation_min_line_length,
                rotation_max_line_gap=rotation_max_line_gap,
                rotation_angle_threshold=rotation_angle_threshold,
                grid_hough_threshold=grid_hough_threshold,
                grid_min_line_length=grid_min_line_length,
                grid_max_line_gap=grid_max_line_gap,
                grid_angle_threshold=grid_angle_threshold,
                merge_distance_threshold=merge_distance_threshold,
                dashed_gap_threshold=dashed_gap_threshold,
                dashed_collinear_threshold=dashed_collinear_threshold,
                curve_saturation_threshold=curve_saturation_threshold,
                curve_value_threshold=curve_value_threshold,
                curve_feather_size=curve_feather_size
            )
            
            # 从 intermediate_results 中获取网格线
            h_lines = intermediate_results.get('merged_h_lines', [])
            v_lines = intermediate_results.get('merged_v_lines', [])
            
            print(f"[Preprocess] text")
            print(f"  - horizontal grid lines: {len(h_lines)}text")
            print(f"  - vertical grid lines: {len(v_lines)}text")
            
            # 保存调试图片
            # try:
            #     import os
            #     import time
            #     debug_dir = "debug_stage1"
            #     os.makedirs(debug_dir, exist_ok=True)
            #     timestamp = int(time.time())
                
            #     # 保存预处理后的图像
            #     cv2.imwrite(f"{debug_dir}/1_processed_image_{timestamp}.jpg", processed_image)
                
            #     # 保存网格线可视化图像
            #     merged_grid = intermediate_results.get('merged_grid')
            #     if merged_grid is not None:
            #         cv2.imwrite(f"{debug_dir}/2_merged_grid_{timestamp}.jpg", merged_grid)
                
            #     # 绘制最终网格线
            #     vis_image = processed_image.copy()
            #     for y in h_lines:
            #         cv2.line(vis_image, (0, int(y)), (vis_image.shape[1], int(y)), (0, 0, 255), 2)
            #     for x in v_lines:
            #         cv2.line(vis_image, (int(x), 0), (int(x), vis_image.shape[0]), (255, 0, 0), 2)
            #     cv2.imwrite(f"{debug_dir}/3_final_grid_lines_{timestamp}.jpg", vis_image)
                
            #     print(f"[Debug] text1imagetext: {debug_dir}/")
            # except Exception as e:
            #     print(f"[Debug] textimageFailed: {str(e)}")
            
            return {
                'success': True,
                'processed_image': processed_image,
                'grid_h_lines': h_lines,
                'grid_v_lines': v_lines,
                'intermediate_results': intermediate_results
            }
            
        except Exception as e:
            print(f"[Preprocess] Exception: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'预处理异常: {str(e)}'
            }
    
    def recognize_coordinates(self,
                            image: np.ndarray,
                            grid_h_lines: List[float],
                            grid_v_lines: List[float],
                            x_tick_labels: List[Dict],
                            y_tick_labels: List[Dict],
                            tolerance: int = 3,
                            enable_morph_connect: bool = True,
                            morph_kernel_size: int = 7,
                            morph_iterations: int = 2,
                            color_saturation_threshold: int = 90,
                            color_value_threshold: int = 60,
                            integer_approximation_threshold: float = 0.2,
                            perspective_matrix: np.ndarray = None,
                            rotation_matrix: np.ndarray = None,
                            original_size: tuple = None,
                            crop_offset: tuple = (0, 0)) -> Dict:
        """
        阶段2: 坐标识别
        
        识别坐标轴、建立坐标映射、提取曲线、检测交点
        
        Args:
            image: 预处理后的图像
            grid_h_lines: 水平网格线位置
            grid_v_lines: 垂直网格线位置
            x_tick_labels: X轴刻度标签
            y_tick_labels: Y轴刻度标签
            tolerance: 交点容差（像素）
            ...: 其他识别参数
        
        Returns:
            结果字典，包含：
            - success: 是否成功
            - coordinate_system: 坐标系信息
            - grid_mapping: 网格映射
            - intersections: 交点列表
            - curve_info: 曲线信息
            - error: 错误信息（如果失败）
        """
        print("\n" + "="*50)
        print("Stage 2: coordinate recognition")
        print("="*50)
        print(f"[Params] tolerance = {tolerance}")
        print(f"[Params] enable_morph_connect = {enable_morph_connect}")
        print(f"[Params] morph_kernel_size = {morph_kernel_size}")
        print(f"[Params] morph_iterations = {morph_iterations}")
        print(f"[Params] color_saturation_threshold = {color_saturation_threshold}")
        print(f"[Params] color_value_threshold = {color_value_threshold}")
        print(f"[Params] integer_approximation_threshold = {integer_approximation_threshold}")
        
        try:
            # 设置参数
            self.coordinate_recognizer.morph_connect_enabled = enable_morph_connect
            self.coordinate_recognizer.morph_kernel_size = morph_kernel_size
            self.coordinate_recognizer.morph_iterations = morph_iterations
            self.coordinate_recognizer.color_saturation_threshold = color_saturation_threshold
            self.coordinate_recognizer.color_value_threshold = color_value_threshold
            self.coordinate_recognizer.integer_approximation_threshold = integer_approximation_threshold
            
            # 执行识别
            result = self.coordinate_recognizer.process(
                image=image,
                grid_h_lines=grid_h_lines,
                grid_v_lines=grid_v_lines,
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels,
                tolerance=tolerance,
                perspective_matrix=perspective_matrix,
                rotation_matrix=rotation_matrix,
                original_size=original_size,
                crop_offset=crop_offset
            )
            
            # 检查是否成功（不能只检查 'error' in result，因为 error 键总是存在）
            if not result.get('success', False) or result.get('error'):
                error_msg = result.get('error', 'unknown error')
                print(f"[CoordRecog] Failed: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg
                }
            
            print(f"[CoordRecog] text")
            if 'intersections' in result:
                print(f"  - Detected intersections: {len(result['intersections'])}text")
            
            # 从 axes_info 提取坐标系信息
            axes_info = result.get('axes_info', {})
            origin_pixel = None
            if 'origin_x' in axes_info and 'origin_y' in axes_info:
                origin_pixel = {'x': axes_info['origin_x'], 'y': axes_info['origin_y']}
            
            # 从 coordinate_mapping 提取网格映射
            coord_mapping = result.get('coordinate_mapping', {})
            
            # 构建水平网格线映射 (pixel_y -> value)
            h_grid_pixels = coord_mapping.get('h_grid_y_pixels', [])
            h_grid_values = coord_mapping.get('h_grid_y_values', [])
            horizontal_lines = [
                {'pixel': float(p), 'value': float(v)} 
                for p, v in zip(h_grid_pixels, h_grid_values)
            ]
            
            # 构建垂直网格线映射 (pixel_x -> value)
            v_grid_pixels = coord_mapping.get('v_grid_x_pixels', [])
            v_grid_values = coord_mapping.get('v_grid_x_values', [])
            vertical_lines = [
                {'pixel': float(p), 'value': float(v)} 
                for p, v in zip(v_grid_pixels, v_grid_values)
            ]
            
            # 保存调试图片
            # try:
            #     import os
            #     import time
            #     debug_dir = "debug_stage2"
            #     os.makedirs(debug_dir, exist_ok=True)
            #     timestamp = int(time.time())
                
            #     # 1. 保存输入图像
            #     cv2.imwrite(f"{debug_dir}/1_input_image_{timestamp}.jpg", image)
                
            #     # 2. 保存曲线二值化图像
            #     curve_binary = result.get('curve_binary')
            #     if curve_binary is not None:
            #         cv2.imwrite(f"{debug_dir}/2_curve_binary_{timestamp}.jpg", curve_binary)
                
            #     # 3. 绘制坐标轴
            #     vis_axes = image.copy()
            #     if axes_info.get('x_axis_y') is not None:
            #         y = int(axes_info['x_axis_y'])
            #         cv2.line(vis_axes, (0, y), (vis_axes.shape[1], y), (255, 0, 0), 2)
            #     if axes_info.get('y_axis_x') is not None:
            #         x = int(axes_info['y_axis_x'])
            #         cv2.line(vis_axes, (x, 0), (x, vis_axes.shape[0]), (0, 255, 0), 2)
            #     cv2.imwrite(f"{debug_dir}/3_coordinate_axes_{timestamp}.jpg", vis_axes)
                
            #     # 4. 绘制网格线
            #     vis_grid = image.copy()
            #     for line in horizontal_lines:
            #         y = int(line['pixel'])
            #         cv2.line(vis_grid, (0, y), (vis_grid.shape[1], y), (0, 0, 255), 1)
            #     for line in vertical_lines:
            #         x = int(line['pixel'])
            #         cv2.line(vis_grid, (x, 0), (x, vis_grid.shape[0]), (255, 0, 0), 1)
            #     cv2.imwrite(f"{debug_dir}/4_grid_mapping_{timestamp}.jpg", vis_grid)
                
            #     # 5. 绘制交点
            #     vis_intersections = image.copy()
            #     intersections = result.get('intersections', [])
            #     for pt in intersections:
            #         px = int(pt.get('pixel_x', 0))
            #         py = int(pt.get('pixel_y', 0))
            #         cv2.circle(vis_intersections, (px, py), 5, (0, 255, 0), -1)
            #         cv2.circle(vis_intersections, (px, py), 7, (255, 255, 255), 1)
            #     cv2.imwrite(f"{debug_dir}/5_intersections_{timestamp}.jpg", vis_intersections)
                
            #     # 6. 综合可视化
            #     vis_all = image.copy()
            #     # 绘制网格线（浅色）
            #     for line in horizontal_lines:
            #         y = int(line['pixel'])
            #         cv2.line(vis_all, (0, y), (vis_all.shape[1], y), (200, 200, 255), 1)
            #     for line in vertical_lines:
            #         x = int(line['pixel'])
            #         cv2.line(vis_all, (x, 0), (x, vis_all.shape[0]), (255, 200, 200), 1)
            #     # 绘制坐标轴（粗线）
            #     if axes_info.get('x_axis_y') is not None:
            #         y = int(axes_info['x_axis_y'])
            #         cv2.line(vis_all, (0, y), (vis_all.shape[1], y), (255, 0, 0), 2)
            #     if axes_info.get('y_axis_x') is not None:
            #         x = int(axes_info['y_axis_x'])
            #         cv2.line(vis_all, (x, 0), (x, vis_all.shape[0]), (0, 255, 0), 2)
            #     # 绘制交点
            #     for pt in intersections:
            #         px = int(pt.get('pixel_x', 0))
            #         py = int(pt.get('pixel_y', 0))
            #         cv2.circle(vis_all, (px, py), 5, (0, 255, 0), -1)
            #         cv2.circle(vis_all, (px, py), 7, (255, 255, 255), 1)
            #     cv2.imwrite(f"{debug_dir}/6_all_combined_{timestamp}.jpg", vis_all)
                
            #     print(f"[Debug] text2imagetext: {debug_dir}/")
            #     print(f"  - textimage, curvetext, axis, text, intersection, text")
            # except Exception as e:
            #     print(f"[Debug] textimageFailed: {str(e)}")
            
            # 构建返回结果
            return {
                'success': True,
                'coordinate_system': {
                    'x_axis_position': axes_info.get('x_axis_y'),
                    'y_axis_position': axes_info.get('y_axis_x'),
                    'origin_pixel': origin_pixel,
                    'origin_value': {'x': 0.0, 'y': 0.0}  # 原点值始终是(0,0)
                },
                'grid_mapping': {
                    'horizontal_lines': horizontal_lines,
                    'vertical_lines': vertical_lines
                },
                'intersections': result.get('intersections', []),
                'reasoning_intersections': result.get('reasoning_intersections', []),
                'curve_info': {
                    'has_curve': result.get('curve_binary') is not None
                },
                'visual_facts': None,  # Now inferred by LLM in reasoning pipeline
                'reasoning_debug': result.get('reasoning_debug', {}),
            }
            
        except Exception as e:
            print(f"[CoordRecog] Exception: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'坐标识别异常: {str(e)}'
            }
    
    def _enhance_with_ruler(
        self,
        x_tick_labels: List[Dict],
        y_tick_labels: List[Dict],
        recognize_result: Dict,
    ) -> Dict:
        """
        阶段2.5: 用 Presenter 风格的回归标尺对 CV 检测结果做增强。

        1. 从刻度标签构建回归标尺 (lstsq)
        2. 对每个 CV 检测的交点，计算连续插值值
        3. 交叉验证: 比较 grid-snapped vs ruler-interpolated
        4. 将 ruler 值附加到每个交点的 dict 中

        这不修改原有的 real_x/real_y/real_x_rounded/real_y_rounded，
        只新增 ruler_x/ruler_y 字段用于验证和展示。
        """
        import numpy as np

        enhancement = {
            'ruler_confidence': 'low',
            'x_ruler': None,
            'y_ruler': None,
            'points_enhanced': 0,
            'cross_validation': [],
        }

        # 构建 X 轴标尺
        x_ruler = self._build_axis_ruler(x_tick_labels, coord_key='center_x')
        y_ruler = self._build_axis_ruler(y_tick_labels, coord_key='center_y')

        if x_ruler:
            enhancement['x_ruler'] = {
                'scale': round(x_ruler['scale'], 8),
                'offset': round(x_ruler['offset'], 4),
                'num_ticks': x_ruler['num_ticks'],
                'residual': round(x_ruler.get('residual', 0), 6),
            }
        if y_ruler:
            enhancement['y_ruler'] = {
                'scale': round(y_ruler['scale'], 8),
                'offset': round(y_ruler['offset'], 4),
                'num_ticks': y_ruler['num_ticks'],
                'residual': round(y_ruler.get('residual', 0), 6),
            }

        has_x = x_ruler is not None
        has_y = y_ruler is not None
        enhancement['ruler_confidence'] = 'high' if (has_x and has_y) else ('medium' if (has_x or has_y) else 'low')

        if not has_x and not has_y:
            return enhancement

        # 对每个交点做连续插值增强
        for key in ('intersections', 'reasoning_intersections'):
            points = recognize_result.get(key, [])
            for pt in points:
                px = pt.get('pixel_x', 0)
                py = pt.get('pixel_y', 0)

                if has_x:
                    ruler_x = x_ruler['scale'] * float(px) + x_ruler['offset']
                    pt['ruler_x'] = round(ruler_x, 4)
                if has_y:
                    ruler_y = y_ruler['scale'] * float(py) + y_ruler['offset']
                    pt['ruler_y'] = round(ruler_y, 4)

                enhancement['points_enhanced'] += 1

                # 交叉验证
                if has_x and has_y:
                    grid_x = pt.get('real_x')
                    grid_y = pt.get('real_y')
                    if grid_x is not None and grid_y is not None:
                        dev_x = abs(pt['ruler_x'] - float(grid_x))
                        dev_y = abs(pt['ruler_y'] - float(grid_y))
                        enhancement['cross_validation'].append({
                            'pixel': (px, py),
                            'grid': (grid_x, grid_y),
                            'ruler': (pt['ruler_x'], pt['ruler_y']),
                            'deviation': (round(dev_x, 4), round(dev_y, 4)),
                        })

        return enhancement

    @staticmethod
    def _build_axis_ruler(tick_labels: List[Dict], coord_key: str) -> Optional[Dict]:
        """从刻度标签构建单轴回归标尺"""
        import numpy as np

        if not tick_labels or len(tick_labels) < 2:
            return None

        pixels = []
        values = []
        for t in tick_labels:
            try:
                pixels.append(float(t[coord_key]))
                values.append(float(t['value']))
            except (ValueError, TypeError, KeyError):
                continue

        if len(pixels) < 2:
            return None

        px_arr = np.array(pixels, dtype=float)
        vl_arr = np.array(values, dtype=float)
        A = np.vstack([px_arr, np.ones(len(px_arr))]).T
        result = np.linalg.lstsq(A, vl_arr, rcond=None)
        scale, offset = float(result[0][0]), float(result[0][1])

        # 计算残差
        predicted = scale * px_arr + offset
        residual = float(np.sqrt(np.mean((predicted - vl_arr) ** 2)))

        return {
            'scale': scale,
            'offset': offset,
            'num_ticks': len(pixels),
            'residual': residual,
        }

    def process_full(self,
                    image_input: str,
                    ocr_result: Dict,
                    config: Optional[Dict] = None) -> Dict:
        """
        完整处理流程
        
        执行所有阶段的处理
        
        Args:
            image_input: base64编码的图像或URL
            ocr_result: OCR OCR结果
            config: 配置字典，包含各阶段参数
        
        Returns:
            完整结果字典
        """
        print("\n" + "="*70)
        print(" Starting full processing pipeline ")
        print("="*70)
        
        # 默认配置
        if config is None:
            config = {}
        
        stage0_config = config.get('stage0', {})
        stage05_config = config.get('stage05', {})
        stage1_config = config.get('stage1', {})
        stage2_config = config.get('stage2', {})
        
        result = {
            'success': False,
            'stages': {},
            'result': {}
        }
        
        try:
            # 加载图像
            image = self.load_image(image_input)
            if image is None:
                return {
                    'success': False,
                    'error': '图像加载失败'
                }
            
            # 阶段0: Chart提取
            stage0_start_time = int(time.time() * 1000)
            stage0_result = self.extract_chart_from_ocr(
                image=image,
                ocr_result=ocr_result,
                **stage0_config
            )
            
            chart_image_base64 = self._chart_image_to_base64(stage0_result.get('chart_image'))
            result['stages']['stage0'] = {
                'status': 'success' if stage0_result['success'] else 'failed',
                'error': stage0_result.get('error'),
                'chart_image': chart_image_base64,
                'duration': int(time.time() * 1000) - stage0_start_time
            }
            
            if not stage0_result['success']:
                return result
            
            chart_image = stage0_result['chart_image']
            x_tick_labels = stage0_result['x_tick_labels']
            y_tick_labels = stage0_result['y_tick_labels']
            
            # 阶段0.5: AI增强（可选）
            stage05_start_time = int(time.time() * 1000)
            if stage05_config.get('enabled', False):
                model_id = stage05_config.get('model_id', 'gemini')
                prompt = stage05_config.get('prompt')
                
                enhance_result = self.enhance_image(
                    image=chart_image,
                    model_id=model_id,
                    prompt=prompt
                )
                
                result['stages']['stage05'] = {
                    'status': 'success' if enhance_result['success'] else 'failed',
                    'model_used': model_id,
                    'error': enhance_result.get('error'),
                    'duration': int(time.time() * 1000) - stage05_start_time
                }
                
                if enhance_result['success']:
                    chart_image = enhance_result['processed_image']
                    
                    # 保存AI增强变换参数
                    ai_transform_params = enhance_result.get('ai_transform_params')
                    if ai_transform_params:
                        result['ai_transform_params'] = ai_transform_params
                        
                        # 对刻度标签进行正向变换
                        print(f"[AIEnhance] Applying forward transform to tick labels...")
                        if x_tick_labels:
                            x_tick_labels = self.preprocessor.transform_labels_for_ai_enhanced(
                                x_tick_labels, ai_transform_params
                            )
                        if y_tick_labels:
                            y_tick_labels = self.preprocessor.transform_labels_for_ai_enhanced(
                                y_tick_labels, ai_transform_params
                            )
            
            # 阶段1: 预处理
            stage1_start_time = int(time.time() * 1000)
            preprocess_result = self.preprocess(
                image=chart_image,
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels,
                **stage1_config
            )
            
            result['stages']['stage1'] = {
                'status': 'success' if preprocess_result['success'] else 'failed',
                'grid_lines_detected': {
                    'horizontal': len(preprocess_result.get('grid_h_lines', [])),
                    'vertical': len(preprocess_result.get('grid_v_lines', []))
                } if preprocess_result['success'] else None,
                'error': preprocess_result.get('error'),
                'duration': int(time.time() * 1000) - stage1_start_time
            }
            
            if not preprocess_result['success']:
                return result
            
            processed_image = preprocess_result['processed_image']
            grid_h_lines = preprocess_result['grid_h_lines']
            grid_v_lines = preprocess_result['grid_v_lines']
            intermediate_results = preprocess_result.get('intermediate_results', {})
            
            # 获取变换矩阵参数（用于阶段2和阶段3）
            perspective_matrix = intermediate_results.get('perspective_matrix')
            rotation_matrix = intermediate_results.get('rotation_matrix')
            original_size = intermediate_results.get('original_size')
            crop_offset = intermediate_results.get('crop_offset', (0, 0))
            
            # 阶段2: 坐标识别
            stage2_start_time = int(time.time() * 1000)
            recognize_result = self.recognize_coordinates(
                image=processed_image,
                grid_h_lines=grid_h_lines,
                grid_v_lines=grid_v_lines,
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels,
                perspective_matrix=perspective_matrix,
                rotation_matrix=rotation_matrix,
                original_size=original_size,
                crop_offset=crop_offset,
                **stage2_config
            )
            
            result['stages']['stage2'] = {
                'status': 'success' if recognize_result['success'] else 'failed',
                'intersections_count': len(recognize_result.get('intersections', [])) if recognize_result['success'] else 0,
                'error': recognize_result.get('error'),
                'duration': int(time.time() * 1000) - stage2_start_time
            }
            
            if not recognize_result['success']:
                return result

            # 阶段2.5: Ruler验证 + 连续插值增强
            # 用 Presenter 风格的 lstsq 回归标尺对 CV 检测结果做交叉验证，
            # 并为每个交点补充连续插值值（不受网格吸附限制）。
            stage25_start_time = int(time.time() * 1000)
            try:
                ruler_enhancement = self._enhance_with_ruler(
                    x_tick_labels, y_tick_labels,
                    recognize_result,
                )
                result['stages']['stage25'] = {
                    'status': 'success',
                    'ruler_confidence': ruler_enhancement.get('ruler_confidence', 'N/A'),
                    'points_enhanced': ruler_enhancement.get('points_enhanced', 0),
                    'duration': int(time.time() * 1000) - stage25_start_time,
                }
                result['ruler_enhancement'] = ruler_enhancement
            except Exception as e:
                print(f"[Stage 2.5] Ruler enhancement skipped: {e}")
                result['stages']['stage25'] = {
                    'status': 'skipped',
                    'error': str(e),
                    'duration': int(time.time() * 1000) - stage25_start_time,
                }

            # 阶段3: 坐标逆变换（将交点从处理后的图像坐标转换回原图Chart坐标）
            stage3_start_time = int(time.time() * 1000)
            intersections = recognize_result.get('intersections', [])
            reasoning_intersections = recognize_result.get('reasoning_intersections', [])
            
            if intersections or reasoning_intersections:
                print("\n" + "="*50)
                print("Stage 3: inverse coordinate transform")
                print("="*50)
                
                # 获取变换信息
                intermediate_results = preprocess_result.get('intermediate_results', {})
                rotation_matrix = intermediate_results.get('rotation_matrix')
                perspective_matrix = intermediate_results.get('perspective_matrix')
                original_size = intermediate_results.get('original_size')  # (width, height)
                crop_offset = intermediate_results.get('crop_offset', (0, 0))
                rotated_image_shape = processed_image.shape[:2]  # (height, width)
                
                # 检查是否需要逆变换
                needs_transform = rotation_matrix is not None or perspective_matrix is not None
                
                if needs_transform:
                    print(f"[InvTransform] Detected transforms:")
                    if rotation_matrix is not None:
                        print(f"  - rotation transform: enabled")
                    if perspective_matrix is not None:
                        print(f"  - perspective transform: enabled")
                    print(f"  - crop offset: {crop_offset}")
                    
                    # 获取AI增强变换参数
                    ai_transform_params = result.get('ai_transform_params')
                    if ai_transform_params:
                        print(f"  - AIEnhancetext: enabled")
                    
                    print(f"  - visualization intersection count: {len(intersections)}")
                    print(f"  - reasoning intersection count: {len(reasoning_intersections)}")
                    
                    # 执行逆变换
                    try:
                        def _inverse_transform(points):
                            if not points:
                                return []
                            return self.coordinate_recognizer.inverse_transform_points(
                                points=points,
                                rotation_matrix=rotation_matrix,
                                original_size=original_size,
                                rotated_image_shape=rotated_image_shape,
                                crop_offset=crop_offset,
                                perspective_matrix=perspective_matrix,
                                ai_transform_params=ai_transform_params
                            )
                        
                        # 更新交点列表（保留原始坐标信息）
                        intersections = _inverse_transform(intersections)
                        reasoning_intersections = _inverse_transform(reasoning_intersections)
                        print(
                            f"[InvTransform] 成功完成，"
                            f"{len(intersections)}个可视化交点和"
                            f"{len(reasoning_intersections)}个推理交点已转换到原图坐标"
                        )
                        
                    except Exception as e:
                        print(f"[InvTransform] Warning: textFailed, textoriginalcoordinate: {str(e)}")
                else:
                    print(f"[InvTransform] No transform needed (no rotation or perspective correction was applied)")
                
                result['stages']['stage3'] = {
                    'status': 'success',
                    'transformed': needs_transform,
                    'intersections': intersections,
                    'reasoning_intersections': reasoning_intersections,
                    'duration': int(time.time() * 1000) - stage3_start_time
                }
            else:
                print(f"\n[Stage3] Skipped (no intersections)")
                result['stages']['stage3'] = {
                    'status': 'skipped',
                    'reason': 'no_intersections',
                    'intersections': [],
                    'reasoning_intersections': [],
                    'duration': int(time.time() * 1000) - stage3_start_time
                }
            
            # 生成阶段3的原图标注（在原始chart图像上绘制逆变换后的交点）
            annotated_image = None
            if intersections and stage0_result.get('chart_image') is not None:
                try:
                    from .drawing_utils import draw_intersections_on_image
                    chart_image = stage0_result['chart_image']
                    annotated_image = draw_intersections_on_image(
                        chart_image,
                        intersections,
                        x_tick_labels=x_tick_labels,
                        y_tick_labels=y_tick_labels,
                    )
                    print(f"[Stage3] Annotations drawn on original chart image")
                except Exception as e:
                    print(f"[Stage3] textFailed: {str(e)}")
            
            # 构建最终结果
            result['success'] = True
            result['result'] = {
                'chart_bbox': stage0_result['chart_bbox'],
                'x_tick_labels': x_tick_labels,
                'y_tick_labels': y_tick_labels,
                'coordinate_system': recognize_result['coordinate_system'],
                'grid_mapping': recognize_result['grid_mapping'],
                'intersections': intersections,  # 使用逆变换后的交点
                'reasoning_intersections': reasoning_intersections,
                'intersections_raw': recognize_result.get('intersections_raw', []),
                'curve_info': recognize_result['curve_info'],
                'annotated_image': annotated_image,  # 阶段3的原图标注
                'visual_facts': None,  # Now inferred by LLM in reasoning pipeline
                'reasoning_debug': recognize_result.get('reasoning_debug', {}),
                'ruler_enhancement': result.get('ruler_enhancement'),
            }
            
            print("\n" + "="*70)
            print(" Full processing pipeline completed successfully ")
            print("="*70)
            
            return result
            
        except Exception as e:
            print(f"\n[FullPipeline] Exception: {str(e)}")
            import traceback

            traceback.print_exc()
            result['error'] = f'处理异常: {str(e)}'
            return result

