"""
坐标系识别系统 - FastAPI 服务

提供RESTful API接口
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 模块搜索路径
# 这样无论从哪里运行此脚本，都能正确导入 core 等模块
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
from datetime import datetime
from typing import Dict, Any

from core.processor import CoordinateProcessor
from core.image_enhancer import ImageEnhancer
from api.api_models import (
    RecognizeCoordinatesRequest,
    RecognizeCoordinatesResponse,
    ApiResponse,
    ErrorDetail,
    ModelInfo,
    RecognizeResult,
    ProcessingStages,
    Stage0Result,
    Stage05Result,
    Stage1Result,
    Stage2Result,
    CoordinateSystem,
    GridMapping,
    GridLine,
    Point,
    Intersection,
    TickLabel,
    BBox,
    CurveInfo
)


# 创建FastAPI应用
app = FastAPI(
    title="坐标系识别API",
    description="Extract coordinate systems, recognize curves, and compute intersection coordinates from problem images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
_ALLOWED_ORIGINS = os.environ.get(
    "GEOSYM_CORS_ORIGINS", "http://localhost:7860,http://localhost:8870"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

# 初始化核心处理器
processor = CoordinateProcessor()


# ===== 工具函数 =====

def generate_request_id() -> str:
    """生成请求ID"""
    return f"req_{uuid.uuid4().hex[:12]}"


def create_success_response(data: Any, message: str = "操作成功") -> Dict:
    """创建成功响应"""
    return {
        "status": "success",
        "data": data,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "request_id": generate_request_id()
    }


def create_error_response(code: str, message: str, details: Any = None) -> Dict:
    """创建错误响应"""
    return {
        "status": "error",
        "error": {
            "code": code,
            "message": message,
            "details": details
        },
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "request_id": generate_request_id()
    }


# ===== 异常处理 =====

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            code=f"HTTP_{exc.status_code}",
            message=exc.detail
        )
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    print(f"[APIException] {str(exc)}")
    import traceback
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            code="INTERNAL_ERROR",
            message="服务器内部错误",
            details=None
        )
    )


# ===== 路由 =====

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "坐标系识别API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/models")
async def get_models():
    """
    获取支持的AI模型列表
    
    Returns:
        模型列表
    """
    try:
        # 获取可用模型
        available_models = processor.image_enhancer.model_manager.get_available_models()
        
        models = []
        for model_id, model_name in available_models.items():
            # 获取默认prompt
            try:
                default_prompt = processor.image_enhancer.model_manager.get_default_prompt(model_id)
            except:
                default_prompt = None
            
            # 获取配置信息
            model_config = None
            for config in processor.image_enhancer.model_manager.models_config:
                if config['id'] == model_id:
                    model_config = config
                    break
            
            if model_config:
                models.append({
                    "id": model_id,
                    "name": model_name,
                    "provider": model_config.get('provider', 'unknown'),
                    "enabled": model_config.get('enabled', False),
                    "default_prompt": default_prompt
                })
        
        return create_success_response(
            data={"models": models},
            message="获取模型列表成功"
        )
        
    except Exception as e:
        return create_error_response(
            code="MODEL_LIST_ERROR",
            message=f"获取模型列表失败: {str(e)}"
        )


@app.post("/api/v1/recognize-coordinates")
async def recognize_coordinates(request: RecognizeCoordinatesRequest):
    """
    识别坐标系和交点
    
    完整流程：
    - 阶段0: 从OCR结果提取Chart
    - 阶段0.5: AI增强（可选）
    - 阶段1: 图像预处理
    - 阶段2: 坐标识别
    
    Args:
        request: 识别请求
    
    Returns:
        识别结果
    """
    request_id = generate_request_id()
    
    try:
        print(f"\n[API] text: {request_id}")
        
        # 提取配置
        config_dict = {}
        if request.config:
            if request.config.stage0:
                config_dict['stage0'] = request.config.stage0.model_dump()
            if request.config.stage05:
                config_dict['stage05'] = request.config.stage05.model_dump()
            if request.config.stage1:
                config_dict['stage1'] = request.config.stage1.model_dump()
            if request.config.stage2:
                config_dict['stage2'] = request.config.stage2.model_dump()
        
        # 执行完整处理
        result = processor.process_full(
            image_input=request.image,
            ocr_result=request.ocr_result,
            config=config_dict
        )
        
        if not result['success']:
            return JSONResponse(
                status_code=422,
                content=create_error_response(
                    code="PROCESSING_FAILED",
                    message=result.get('error', '处理失败'),
                    details=result.get('stages', {})
                )
            )
        
        # 构建响应数据
        response_data = {
            "stages": {},
            "result": {}
        }
        
        # 处理阶段信息
        stages = result.get('stages', {})
        for stage_name, stage_data in stages.items():
            response_data["stages"][stage_name] = stage_data
        
        # 处理结果数据
        result_data = result.get('result', {})
        
        # 转换TickLabel
        x_tick_labels = []
        for label in result_data.get('x_tick_labels', []):
            x_tick_labels.append({
                "text": label['text'],
                "value": label['value'],
                "bbox": label['bbox'],
                "center_x": label['center_x'],
                "center_y": label['center_y'],
                "original_type": label.get('original_type')
            })
        
        y_tick_labels = []
        for label in result_data.get('y_tick_labels', []):
            y_tick_labels.append({
                "text": label['text'],
                "value": label['value'],
                "bbox": label['bbox'],
                "center_x": label['center_x'],
                "center_y": label['center_y'],
                "original_type": label.get('original_type')
            })
        
        # 转换GridMapping
        grid_mapping = result_data.get('grid_mapping', {})
        horizontal_lines = []
        for line in grid_mapping.get('horizontal_lines', []):
            if isinstance(line, dict):
                horizontal_lines.append({
                    "pixel": line.get('pixel'),
                    "value": line.get('value')
                })
            else:
                # 如果是元组(pixel, value)
                horizontal_lines.append({
                    "pixel": line[0] if len(line) > 0 else None,
                    "value": line[1] if len(line) > 1 else None
                })
        
        vertical_lines = []
        for line in grid_mapping.get('vertical_lines', []):
            if isinstance(line, dict):
                vertical_lines.append({
                    "pixel": line.get('pixel'),
                    "value": line.get('value')
                })
            else:
                vertical_lines.append({
                    "pixel": line[0] if len(line) > 0 else None,
                    "value": line[1] if len(line) > 1 else None
                })
        
        # 转换Intersection
        intersections = []
        for intersection in result_data.get('intersections', []):
            intersections.append({
                "pixel_x": intersection.get('pixel_x'),
                "pixel_y": intersection.get('pixel_y'),
                "corrected_pixel_x": intersection.get('original_pixel_x'),  # 校正后坐标
                "corrected_pixel_y": intersection.get('original_pixel_y'),  # 校正后坐标
                "coordinate_x": intersection.get('real_x'),  # 真实坐标X（映射自real_x）
                "coordinate_y": intersection.get('real_y'),  # 真实坐标Y（映射自real_y）
                "real_x_rounded": intersection.get('real_x_rounded'),
                "real_y_rounded": intersection.get('real_y_rounded'),
                "is_axis_point": intersection.get('is_axis_point', False),
                "type": intersection.get('type', 'curve_grid')
            })
        
        # 坐标系信息
        coord_system = result_data.get('coordinate_system', {})
        origin_pixel = coord_system.get('origin_pixel')
        origin_value = coord_system.get('origin_value')
        
        response_data["result"] = {
            "chart_bbox": result_data.get('chart_bbox'),
            "x_tick_labels": x_tick_labels,
            "y_tick_labels": y_tick_labels,
            "coordinate_system": {
                "x_axis_position": coord_system.get('x_axis_position'),
                "y_axis_position": coord_system.get('y_axis_position'),
                "origin_pixel": {"x": origin_pixel['x'], "y": origin_pixel['y']} if origin_pixel else None,
                "origin_value": {"x": origin_value['x'], "y": origin_value['y']} if origin_value else None
            },
            "grid_mapping": {
                "horizontal_lines": horizontal_lines,
                "vertical_lines": vertical_lines
            },
            "intersections": intersections,
            "curve_info": result_data.get('curve_info', {"has_curve": False})
        }
        
        return create_success_response(
            data=response_data,
            message="坐标识别成功"
        )
        
    except Exception as e:
        print(f"[API] Processing exception: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content=create_error_response(
                code="INTERNAL_ERROR",
                message="服务器内部错误"
            )
        )


# ===== 启动服务 =====

def start_server(host: str = "127.0.0.1", port: int = 8080):
    """启动API服务"""
    print(f"""
╔══════════════════════════════════════════════════════════╗
║          坐标系识别系统 - API 服务                         ║
╚══════════════════════════════════════════════════════════╝

服务地址: http://{host}:{port}
API文档: http://{host}:{port}/docs
健康检查: http://{host}:{port}/health

""")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    host = "127.0.0.1"
    port = 8080
    
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    start_server(host=host, port=port)

