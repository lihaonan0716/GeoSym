"""
API 路由定义
"""

import logging
import base64
import json, os
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import time

import numpy as np
import serverconfig
import urllib.parse

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse

from core.processor import CoordinateProcessor
from core.coordinate_recognizer import CoordinateRecognizer

import sys as _sys
from pathlib import Path as _Path
_exp_root = _Path(__file__).resolve().parent.parent
if str(_exp_root) not in _sys.path:
    _sys.path.insert(0, str(_exp_root))
from i18n import log_tag as _lt  # noqa: E402

# 获取日志记录器
logger = logging.getLogger(__name__)

# 创建路由实例
router = APIRouter()

# 初始化处理器
processor = CoordinateProcessor()

recognizer = CoordinateRecognizer()

def _decode_chart_image(chart_image_base64: Optional[str]) -> Optional[np.ndarray]:
    """将 stage0 中的 Base64 chart_image 解码为 numpy 数组"""
    if not chart_image_base64:
        return None

    if "," in chart_image_base64:
        chart_image_base64 = chart_image_base64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(chart_image_base64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV 解码失败")
        return image
    except Exception as exc:
        logger.error(f"chart_image 解码失败: {exc}", exc_info=True)
        return None

@router.get("/")
async def root():
    return {
        "service": "Coordinate Recognizer",
        "version": "1.0.0",
        "status": "running",
    }

@router.get("/health")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "Coordinate Recognizer",
            "version": "1.0.0",
        },
    )

@router.post("/api/v1/recognize", response_class=JSONResponse)
async def recognize_coordinates(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    ocr_result: str = Form(...),
):
    """
    坐标识别入口

    接收文件或base64编码的图像，以及line_data参数
    """

    try:
        start = int(time.time() * 1000)
        # 获取图像数据
        if file:
            image_data = await file.read()
            image_base64_str = base64.b64encode(image_data).decode("utf-8")
        elif image_base64:
            # 移除可能的前缀
            if "," in image_base64:
                image_base64_str = image_base64.split(",", 1)[1]
            else:
                image_base64_str = image_base64
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "必须提供文件或image_base64参数",
                    "timestamp": datetime.now().isoformat(),
                    "duration": int(time.time() * 1000) - start,
                },
            )

        try: 
            # 判断ocr_result 是否经过了urlencode
            ocr_result_obj = json.loads(urllib.parse.unquote(ocr_result))
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"ocr_result 不是有效的JSON字符串: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "duration": int(time.time() * 1000) - start,
                },
            )
        
        # 调整旋转角度
        auto_rotate_degrees = ocr_result_obj.get('auto_rotate_degrees', 0)
        if auto_rotate_degrees == 90:
            ocr_result_obj['auto_rotate_degrees'] = 270
        elif auto_rotate_degrees == -90:
            ocr_result_obj['auto_rotate_degrees'] = 90
            
        # 调用处理器
        result = processor.process_full(
            image_input=image_base64_str,
            ocr_result=ocr_result_obj,
            config={
                "stage05": {
                    "enabled": True,
                    "model_id": "flux-kontext-pro-azure",
                    "prompt": None,  # 使用默认prompt
                }
            },
        )

        if not result.get("success"):
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "message": result.get("error", "处理失败"),
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                    "duration": int(time.time() * 1000) - start,
                },
            )

        stage1_data = result.get('result')

        # 将坐标信息绘制回原图
        transformed_points = result["stages"]["stage3"]["intersections"]

        # transformed_points 为空，不绘制图
        stage_code_start_time = int(time.time() * 1000)
        if transformed_points:

            timestamp = int(time.time() * 1000)
            chart_image_base64 = result["stages"]["stage0"]["chart_image"]
            processed_image = _decode_chart_image(chart_image_base64)
            if processed_image is None:
                raise ValueError("chart_image 无法解码")

            code = draw_image_to_base64(
                processed_image, transformed_points, stage1_data
            )
            result["base64"] = code
            
        del result["stages"]["stage0"]["chart_image"]

        result["stages"]["base64"] = {
            "duration": int(time.time() * 1000) - stage_code_start_time
        }

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": result,
                "timestamp": datetime.now().isoformat(),
                "duration": int(time.time() * 1000) - start,
            },
        )

    except Exception as e:
        logger.error(f"坐标识别失败: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"服务器内部错误: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "duration": int(time.time() * 1000) - start,
            },
        )

def draw_image_to_base64(processed_image, transformed_points, stage1_data):
    from core.drawing_utils import draw_intersections_on_image

    timestamp = int(time.time() * 1000)
    
    # 定义绘制标注的辅助函数
    def draw_annotations_on_image(image, points):
        """在给定图像上绘制标注"""
        x_tick_labels = stage1_data.get("x_tick_labels", [])
        y_tick_labels = stage1_data.get("y_tick_labels", [])
        
        return draw_intersections_on_image(
            image, points, x_tick_labels, y_tick_labels
        )

    # 1. 在AI增强后的图上绘制标注
    print(f"[{_lt('阶段3')}] 在AI增强图上绘制标注...")

    # 2. 在真正的原始图上绘制标注（如果有的话）
    annotated_raw_rgb = None
    if processed_image is not None:
        print(f"[{_lt('阶段3')}] 在原始Chart图上绘制标注...")
        annotated_raw = draw_annotations_on_image(processed_image, transformed_points)
        annotated_raw_rgb = cv2.cvtColor(annotated_raw, cv2.COLOR_BGR2RGB)

    h, w = processed_image.shape[:2]
    info_text = f"""✅ 阶段3完成！原图尺寸: {w}x{h} 交点数量: {len(transformed_points)} 已将校正图上的交点坐标映射回原图，并在原图上绘制标注。"""
    
    # 保存文件
    output_dir = getattr(serverconfig, "OUTPUT_DIR", None) or Path("debug_stage05")
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    chart_path = output_dir / f"annotated_raw_{timestamp}.jpg"
    cv2.imwrite(str(chart_path), annotated_raw)

    # 读取本地文件 并转为 base64
    with open(str(chart_path), "rb") as f:
        chart_data = f.read()
    code = base64.b64encode(chart_data).decode("utf-8")

    # 如果文件存在，删除本地文件
    if chart_path.exists():
        chart_path.unlink()

    return code
