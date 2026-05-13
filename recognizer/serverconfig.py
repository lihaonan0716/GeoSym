"""
Coordinate Recognizer 配置文件
配置文件路径和其他设置
"""

from pathlib import Path
import sys

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================
# 文件路径配置
# ============================================

# 临时文件目录（用于存储上传的图片等临时文件）
# 使用项目根目录下的 temp 目录，而不是系统 /temp（macOS 上只读）
# 根据系统判断，是否是macOS，如果不是，则使用/tmp目录
if sys.platform.lower() == "darwin":
    TEMP_DIR = PROJECT_ROOT / "temp"
else:
    TEMP_DIR = Path("/tmp")


# 日志文件目录
LOG_DIR = TEMP_DIR / "logs"

# 日志文件名模板（支持日期格式化）
# 可用格式: %Y-%m-%d, %Y%m%d, %Y%m%d_%H%M%S 等
LOG_FILE_TEMPLATE = "server_%Y%m%d.log"
# Web 请求/响应日志文件模板，采用与 server log 相同的分割策略
WEB_LOG_FILE_TEMPLATE = "web_%Y%m%d.log"

# 输出文件目录（用于存储处理后的图片）
OUTPUT_DIR = TEMP_DIR / "output"

# 调试图片目录（用于存储调试阶段的图片）
DEBUG_DIR = TEMP_DIR / "debug_stage"

# ============================================
# 服务器配置
# ============================================

# 默认主机地址
DEFAULT_HOST = "127.0.0.1"

# 默认端口
DEFAULT_PORT = 8080

# 默认日志级别
DEFAULT_LOG_LEVEL = "info"

# 是否默认启用日志文件
DEFAULT_ENABLE_LOG_FILE = True

# ============================================
# 辅助函数
# ============================================


def ensure_dirs():
    """确保所有必要的目录存在"""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def get_log_file_path(template: str = None) -> Path:
    """
    根据模板生成日志文件路径
    
    Args:
        template: 日志文件名模板，如果为 None 则使用默认模板
        
    Returns:
        日志文件路径
    """
    from datetime import datetime
    
    if template is None:
        template = LOG_FILE_TEMPLATE
    
    log_filename = datetime.now().strftime(template)
    return LOG_DIR / log_filename

