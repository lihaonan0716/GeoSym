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
    pixel: float = Field(..., description="像素位置")
    value: Optional[float] = Field(None, description="坐标值")


class CoordinateSystem(BaseModel):
    """坐标系信息"""
    x_axis_position: Optional[float] = Field(None, description="X轴位置（像素）")
    y_axis_position: Optional[float] = Field(None, description="Y轴位置（像素）")
    origin_pixel: Optional[Point] = Field(None, description="原点像素坐标")
    origin_value: Optional[Point] = Field(None, description="原点数值坐标")


class Intersection(BaseModel):
    """交点"""
    pixel_x: float = Field(..., description="原图Chart上的X坐标（经过阶段3逆变换后）")
    pixel_y: float = Field(..., description="原图Chart上的Y坐标（经过阶段3逆变换后）")
    corrected_pixel_x: Optional[float] = Field(None, description="校正后图像的X坐标（如果进行了旋转/透视校正）")
    corrected_pixel_y: Optional[float] = Field(None, description="校正后图像的Y坐标（如果进行了旋转/透视校正）")
    coordinate_x: Union[float, int, str] = Field(..., description="真实坐标系X值")
    coordinate_y: Union[float, int, str] = Field(..., description="真实坐标系Y值")
    real_x_rounded: Optional[int] = Field(None, description="取整后的真实X坐标")
    real_y_rounded: Optional[int] = Field(None, description="取整后的真实Y坐标")
    is_axis_point: bool = Field(False, description="是否为坐标轴上的点")
    type: str = Field(..., description="交点类型: origin/x_axis/y_axis/curve_grid")


class GridMapping(BaseModel):
    """网格映射"""
    horizontal_lines: List[GridLine] = Field(default_factory=list, description="水平网格线")
    vertical_lines: List[GridLine] = Field(default_factory=list, description="垂直网格线")


class CurveInfo(BaseModel):
    """曲线信息"""
    has_curve: bool = Field(False, description="是否检测到曲线")


# ===== 请求模型 =====

class Stage0Config(BaseModel):
    """阶段0配置"""
    horizontal_tolerance: float = Field(1.5, description="水平聚类容差倍数")
    vertical_tolerance: float = Field(1.5, description="垂直聚类容差倍数")
    interval_percentile: float = Field(50, description="间隔分位数(0-100)")


class Stage05Config(BaseModel):
    """阶段0.5配置"""
    enabled: bool = Field(False, description="是否启用AI增强")
    model_id: str = Field("gemini", description="模型ID")
    prompt: Optional[str] = Field(None, description="提示词，None时使用默认")


class Stage1Config(BaseModel):
    """阶段1配置"""
    enable_curve_erase: bool = Field(True, description="是否擦除曲线边缘")
    enable_dashed_line_connection: bool = Field(True, description="是否连接虚线")
    enable_perspective_correction: bool = Field(True, description="是否透视校正")
    enable_rotation_correction: bool = Field(True, description="是否倾斜校正")
    rotation_hough_threshold: int = Field(65, description="倾斜检测Hough阈值")
    rotation_min_line_length: int = Field(60, description="倾斜检测最小线段长度")
    rotation_max_line_gap: int = Field(10, description="倾斜检测最大线段间隙")
    rotation_angle_threshold: int = Field(5, description="倾斜检测角度阈值")
    grid_hough_threshold: int = Field(65, description="网格检测Hough阈值")
    grid_min_line_length: int = Field(45, description="网格检测最小线段长度")
    grid_max_line_gap: int = Field(10, description="网格检测最大线段间隙")
    grid_angle_threshold: int = Field(5, description="网格检测角度阈值")
    merge_distance_threshold: int = Field(20, description="线段合并距离阈值")
    dashed_gap_threshold: int = Field(10, description="虚线连接间隙阈值")
    dashed_collinear_threshold: int = Field(3, description="虚线连接共线阈值")
    curve_saturation_threshold: int = Field(90, description="曲线饱和度阈值")
    curve_value_threshold: int = Field(60, description="曲线亮度阈值")
    curve_feather_size: int = Field(3, description="曲线羽化大小")


class Stage2Config(BaseModel):
    """阶段2配置"""
    tolerance: int = Field(3, description="交点容差（像素）")
    enable_morph_connect: bool = Field(True, description="是否形态学连接")
    morph_kernel_size: int = Field(7, description="形态学核大小")
    morph_iterations: int = Field(2, description="形态学迭代次数")
    color_saturation_threshold: int = Field(90, description="颜色饱和度阈值")
    color_value_threshold: int = Field(60, description="颜色亮度阈值")
    integer_approximation_threshold: float = Field(0.1, description="整数近似阈值")


class ProcessConfig(BaseModel):
    """完整处理配置"""
    stage0: Optional[Stage0Config] = Field(default_factory=Stage0Config)
    stage05: Optional[Stage05Config] = Field(default_factory=Stage05Config)
    stage1: Optional[Stage1Config] = Field(default_factory=Stage1Config)
    stage2: Optional[Stage2Config] = Field(default_factory=Stage2Config)


class RecognizeCoordinatesRequest(BaseModel):
    """识别坐标请求"""
    image: str = Field(..., description="图像：base64编码或URL")
    ocr_result: Dict[str, Any] = Field(..., description="OCR OCR结果")
    config: Optional[ProcessConfig] = Field(None, description="处理配置")


# ===== 响应模型 =====

class StageResult(BaseModel):
    """阶段处理结果"""
    status: str = Field(..., description="success/failed")
    error: Optional[str] = Field(None, description="错误信息")


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
    chart_bbox: Optional[tuple] = Field(None, description="Chart边界框(x, y, w, h)")
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

