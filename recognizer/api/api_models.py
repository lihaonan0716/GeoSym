"""
API数据模型定义

使用Pydantic模型定义请求和响应格式
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime


# ===== 基础数据类型 =====

class BBox(BaseModel):
    """边界框"""
    left: float
    top: float
    right: float
    bottom: float


class Point(BaseModel):
    """二维点"""
    x: float
    y: float


class TickLabel(BaseModel):
    """刻度标签"""
    text: str
    value: float
    bbox: BBox
    center_x: float
    center_y: float
    original_type: Optional[str] = None


class GridLine(BaseModel):
    """网格线映射"""
    pixel: float = Field(..., description="Pixel position")
    value: Optional[float] = Field(None, description="Coordinate value")


class CoordinateSystem(BaseModel):
    """坐标系信息"""
    x_axis_position: Optional[float] = Field(None, description="X-axis position in pixels")
    y_axis_position: Optional[float] = Field(None, description="Y-axis position in pixels")
    origin_pixel: Optional[Point] = Field(None, description="Origin pixel coordinate")
    origin_value: Optional[Point] = Field(None, description="Origin value coordinate")


class Intersection(BaseModel):
    """交点"""
    pixel_x: float = Field(..., description="X coordinate on the original chart after Stage 3 inverse transform")
    pixel_y: float = Field(..., description="Y coordinate on the original chart after Stage 3 inverse transform")
    corrected_pixel_x: Optional[float] = Field(None, description="X coordinate on the corrected image if rotation or perspective correction was applied")
    corrected_pixel_y: Optional[float] = Field(None, description="Y coordinate on the corrected image if rotation or perspective correction was applied")
    coordinate_x: Union[float, int, str] = Field(..., description="Real coordinate-system X value")
    coordinate_y: Union[float, int, str] = Field(..., description="Real coordinate-system Y value")
    real_x_rounded: Optional[int] = Field(None, description="Rounded real X coordinate")
    real_y_rounded: Optional[int] = Field(None, description="Rounded real Y coordinate")
    is_axis_point: bool = Field(False, description="Whether the point lies on an axis")
    type: str = Field(..., description="Intersection type: origin/x_axis/y_axis/curve_grid")


class GridMapping(BaseModel):
    """网格映射"""
    horizontal_lines: List[GridLine] = Field(default_factory=list, description="Horizontal grid lines")
    vertical_lines: List[GridLine] = Field(default_factory=list, description="Vertical grid lines")


class CurveInfo(BaseModel):
    """曲线信息"""
    has_curve: bool = Field(False, description="Whether a curve was detected")


# ===== 请求模型 =====

class Stage0Config(BaseModel):
    """阶段0配置"""
    horizontal_tolerance: float = Field(1.5, description="Horizontal clustering tolerance multiplier")
    vertical_tolerance: float = Field(1.5, description="Vertical clustering tolerance multiplier")
    interval_percentile: float = Field(50, description="Interval percentile(0-100)")


class Stage05Config(BaseModel):
    """阶段0.5配置"""
    enabled: bool = Field(False, description="Whether AI enhancement is enabled")
    model_id: str = Field("gemini", description="Model ID")
    prompt: Optional[str] = Field(None, description="Prompt; uses the default when None")


class Stage1Config(BaseModel):
    """阶段1配置"""
    enable_curve_erase: bool = Field(True, description="Whether to erase curve edges")
    enable_dashed_line_connection: bool = Field(True, description="Whether to connect dashed lines")
    enable_perspective_correction: bool = Field(True, description="Whether to apply perspective correction")
    enable_rotation_correction: bool = Field(True, description="Whether to apply tilt correction")
    rotation_hough_threshold: int = Field(65, description="Tilt-detection Hough threshold")
    rotation_min_line_length: int = Field(60, description="Tilt-detection minimum line length")
    rotation_max_line_gap: int = Field(10, description="Tilt-detection maximum line gap")
    rotation_angle_threshold: int = Field(5, description="Tilt-detection angle threshold")
    grid_hough_threshold: int = Field(65, description="Grid-detection Hough threshold")
    grid_min_line_length: int = Field(45, description="Grid-detection minimum line length")
    grid_max_line_gap: int = Field(10, description="Grid-detection maximum line gap")
    grid_angle_threshold: int = Field(5, description="Grid-detection angle threshold")
    merge_distance_threshold: int = Field(20, description="Line-merge distance threshold")
    dashed_gap_threshold: int = Field(10, description="Dashed-line connection gap threshold")
    dashed_collinear_threshold: int = Field(3, description="Dashed-line connection collinearity threshold")
    curve_saturation_threshold: int = Field(90, description="Curve saturation threshold")
    curve_value_threshold: int = Field(60, description="Curve value threshold")
    curve_feather_size: int = Field(3, description="Curve feather size")


class Stage2Config(BaseModel):
    """阶段2配置"""
    tolerance: int = Field(3, description="Intersection tolerance in pixels")
    enable_morph_connect: bool = Field(True, description="Whether morphological connection is enabled")
    morph_kernel_size: int = Field(7, description="Morphological kernel size")
    morph_iterations: int = Field(2, description="Morphological iteration count")
    color_saturation_threshold: int = Field(90, description="Color saturation threshold")
    color_value_threshold: int = Field(60, description="Color value threshold")
    integer_approximation_threshold: float = Field(0.1, description="Integer approximation threshold")


class ProcessConfig(BaseModel):
    """完整处理配置"""
    stage0: Optional[Stage0Config] = Field(default_factory=Stage0Config)
    stage05: Optional[Stage05Config] = Field(default_factory=Stage05Config)
    stage1: Optional[Stage1Config] = Field(default_factory=Stage1Config)
    stage2: Optional[Stage2Config] = Field(default_factory=Stage2Config)


class RecognizeCoordinatesRequest(BaseModel):
    """识别坐标请求"""
    image: str = Field(..., description="Image as base64 data or URL")
    ocr_result: Dict[str, Any] = Field(..., description="OCR result")
    config: Optional[ProcessConfig] = Field(None, description="Processing configuration")


# ===== 响应模型 =====

class StageResult(BaseModel):
    """阶段处理结果"""
    status: str = Field(..., description="success/failed")
    error: Optional[str] = Field(None, description="Error message")


class Stage0Result(StageResult):
    """阶段0结果"""
    pass


class Stage05Result(StageResult):
    """阶段0.5结果"""
    model_used: Optional[str] = None


class Stage1Result(StageResult):
    """阶段1结果"""
    grid_lines_detected: Optional[Dict[str, int]] = None


class Stage2Result(StageResult):
    """阶段2结果"""
    intersections_count: Optional[int] = None


class ProcessingStages(BaseModel):
    """处理阶段结果"""
    stage0: Stage0Result
    stage05: Optional[Stage05Result] = None
    stage1: Stage1Result
    stage2: Stage2Result


class RecognizeResult(BaseModel):
    """识别结果"""
    chart_bbox: Optional[tuple] = Field(None, description="Chart bounding box(x, y, w, h)")
    x_tick_labels: List[TickLabel] = Field(default_factory=list)
    y_tick_labels: List[TickLabel] = Field(default_factory=list)
    coordinate_system: CoordinateSystem
    grid_mapping: GridMapping
    intersections: List[Intersection] = Field(default_factory=list)
    curve_info: CurveInfo


class ApiResponse(BaseModel):
    """API统一响应格式"""
    status: str = Field(..., description="success/error")
    data: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: Optional[str] = None


class RecognizeCoordinatesResponse(ApiResponse):
    """识别坐标响应"""
    data: Optional[Dict[str, Any]] = None
    

class ErrorDetail(BaseModel):
    """错误详情"""
    code: str
    message: str
    details: Optional[Any] = None


class ModelInfo(BaseModel):
    """模型信息"""
    id: str
    name: str
    provider: str
    enabled: bool
    default_prompt: Optional[str] = None

