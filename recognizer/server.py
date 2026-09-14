"""
API 服务入口文件
提供高并发 API 服务，包含日志和健康检查功能
"""

import sys
import os
import json
import logging
import multiprocessing
import uuid
from pathlib import Path
from datetime import datetime, date

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from router import router
import serverconfig

# 添加项目根目录到 Python 模块搜索路径
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 确保所有必要的目录存在
serverconfig.ensure_dirs()

WEB_LOG_TEMPLATE = getattr(serverconfig, "WEB_LOG_FILE_TEMPLATE", "web_%Y%m%d.log")


class DailyRotatingFileHandler(logging.FileHandler):
    """按天轮转的文件处理器，日志跨天时自动切换目标文件。"""

    def __init__(self, template: str, log_dir: Path, encoding: str = "utf-8"):
        self.template = template
        self.log_dir = log_dir
        self._current_date = datetime.now().date()
        initial_path = self._format_path(self._current_date)
        super().__init__(initial_path, encoding=encoding)

    def _format_path(self, current_date: date) -> Path:
        return self.log_dir / current_date.strftime(self.template)

    def emit(self, record: logging.LogRecord) -> None:
        current_date = datetime.now().date()
        if current_date != self._current_date:
            self._current_date = current_date
            if self.stream:
                self.stream.close()
            self.baseFilename = str(self._format_path(current_date))
            self.stream = self._open()
        super().emit(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    log_file_template: str = None,
):
    """
    配置日志系统

    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件名（仅文件名或路径），如果提供则固定写入该文件
        log_file_template: 日志文件名模板（支持日期格式化），用于按天轮转
    """
    # 使用配置文件中的日志目录
    log_dir = serverconfig.LOG_DIR
    log_dir.mkdir(exist_ok=True)

    # 配置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 配置根日志记录器
    handlers = []

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    handlers.append(console_handler)

    log_path_for_message = None
    if log_file_template and not log_file:
        log_path_for_message = serverconfig.get_log_file_path(
            template=log_file_template
        )
        file_handler = DailyRotatingFileHandler(
            template=log_file_template,
            log_dir=log_dir,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)
    elif log_file:
        log_path = log_dir / log_file
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)
        log_path_for_message = log_path

    # 配置日志级别
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,  # 强制重新配置
    )

    # 设置第三方库的日志级别
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized, level: {log_level}")
    if log_path_for_message:
        logger.info(f"Log file: {log_path_for_message}")
    else:
        logger.info("Log file: disabled (console only)")

    return logger


_web_logger = None


def get_web_logger():
    global _web_logger
    if _web_logger:
        return _web_logger

    handler = DailyRotatingFileHandler(
        template=WEB_LOG_TEMPLATE,
        log_dir=serverconfig.LOG_DIR,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    web_logger = logging.getLogger("recognizer.web")
    web_logger.setLevel(logging.INFO)
    web_logger.handlers = []
    web_logger.propagate = False
    web_logger.addHandler(handler)

    _web_logger = web_logger
    return _web_logger


# ============================================
# FastAPI 应用
# ============================================

# 初始化日志
logger = setup_logging()
web_logger = get_web_logger()

# 创建 FastAPI 应用
app = FastAPI(
    title="Coordinate Recognizer API 服务",
    description="High-concurrency Coordinate Recognizer API service",
    version="1.0.0",
    redoc_url="/redoc",
)

# 添加 CORS 中间件
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

# 注册路由
app.include_router(router)

# ============================================
# 中间件
# ============================================


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    start_time = datetime.now()

    # 记录请求信息
    logger.info(f"Received request: {request.method} {request.url.path}")

    # 读取请求体，但 Starlette 会缓存结果，后续 handler 仍可访问
    request_body_bytes = await request.body()
    request_body = request_body_bytes.decode("utf-8", errors="replace")
    request_content_type = request.headers.get("content-type", "")
    
    strs = request_body.split("Content-Disposition:")
    record_parts = []
    for s in strs:
        # 检查当前片段中是否包含图片类型的Content-Type
        if "Content-Type: image/" in s:
            continue
        record_parts.append(s)

    request_body_for_log = "\n".join(record_parts)
    request_id = str(uuid.uuid4())
    request_log_entry = {
        "type": "request",
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "request": {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "path_params": request.path_params,
            "content_type": request_content_type,
            "body": request_body_for_log,
            "body_omitted": request_body_for_log is None,
        },
    }
    web_logger.info(json.dumps(request_log_entry, ensure_ascii=False))

    # 处理请求
    response = await call_next(request)

    # 确保 JSON 响应有正确的 Content-Type（包含 charset）
    response_content_type = response.headers.get("Content-Type", "")
    if isinstance(response, JSONResponse):
        if "charset" not in response_content_type.lower():
            response.headers["Content-Type"] = "application/json; charset=utf-8"
    elif not response_content_type:
        response.headers["Content-Type"] = "application/json; charset=utf-8"

    # 读取响应体内容
    response_body_bytes = b""
    async for chunk in response.body_iterator:
        response_body_bytes += chunk

    # 重新构造 Response，确保 downstream 还能读取 body
    response_headers = dict(response.headers)
    wrapped_response = Response(
        content=response_body_bytes,
        status_code=response.status_code,
        headers=response_headers,
        media_type=None,
        background=response.background,
    )

    process_time = (datetime.now() - start_time).total_seconds()

    # 记录响应信息
    logger.info(
        f"请求完成: {request.method} {request.url.path} - "
        f"状态码: {response.status_code} - 耗时: {process_time:.3f}s"
    )

    response_log_entry = {
        "type": "response",
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "duration": process_time,
        "response": {
            "status_code": response.status_code,
            "headers": response_headers,
            "body": response_body_bytes.decode("utf-8", errors="replace"),
        },
        "request": {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "path_params": request.path_params,
            "content_type": request_content_type,
            "body": request_body_for_log,
            "body_omitted": request_body_for_log is None,
        },
    }
    web_logger.info(json.dumps(response_log_entry, ensure_ascii=False))

    return wrapped_response


# ============================================
# 异常处理
# ============================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat(),
        },
    )


# ============================================
# 启动服务
# ============================================


def get_default_workers() -> int:
    """
    根据 CPU 核心数计算合理的 worker 数量

    对于 I/O 密集型应用（如 API 服务），使用公式: (CPU核心数 * 2) + 1
    确保至少返回 1 个 worker

    Returns:
        int: 推荐的 worker 数量
    """
    cpu_count = multiprocessing.cpu_count()
    # 对于 I/O 密集型应用，可以使用更多 worker
    # 公式: (CPU核心数 * 2) + 1
    workers = (cpu_count * 2) + 1
    # 确保至少为 1
    return max(1, workers)


def start_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    workers: int = None,
    log_level: str = "info",
    reload: bool = False,
    log_file: str = None,
    enable_log_file: bool = True,
    timeout_graceful_shutdown: int = 30,
):
    """
    启动 API 服务

    Args:
        host: 服务监听地址
        port: 服务端口
        workers: Worker 进程数（用于高并发），如果为 None 则根据 CPU 核心数自动计算
        log_level: 日志级别
        reload: 是否启用自动重载（开发模式）
        log_file: 日志文件路径（可选），如果为 None 且 enable_log_file=True，则使用 serverconfig.LOG_FILE_TEMPLATE 并按天轮转
        enable_log_file: 是否启用日志文件，如果为 False 则只输出到控制台
        timeout_graceful_shutdown: 优雅关闭超时时间（秒），超过此时间将强制关闭
    """

    log_file_template = None
    log_file_description = None
    if enable_log_file:
        if log_file:
            log_file_description = log_file
        else:
            log_file_template = serverconfig.LOG_FILE_TEMPLATE
            log_file_description = serverconfig.get_log_file_path(
                template=log_file_template
            )
    else:
        log_file = None

    # 重新配置日志，添加文件处理器（如果需要）
    setup_logging(
        log_level=log_level.upper(),
        log_file=log_file,
        log_file_template=log_file_template,
    )
    logger = logging.getLogger(__name__)

    # 如果未指定 workers，则根据 CPU 核心数自动计算
    if workers is None:
        workers = get_default_workers()

    cpu_count = multiprocessing.cpu_count()
    process_id = os.getpid()
    logger.info("=" * 60)
    logger.info("Starting Coordinate Recognizer API service")
    logger.info("=" * 60)
    logger.info(f"Process ID: {process_id}")
    logger.info(f"CPU cores: {cpu_count}")
    logger.info(f"Service URL: http://{host}:{port}")
    logger.info(f"Health check: http://{host}:{port}/health")
    logger.info(f"Worker count: {workers}")
    logger.info(f"text: {log_level}")
    if log_file_description:
        logger.info(f"Log file: {log_file_description}")
    else:
        logger.info("Log file: disabled (console only)")
    logger.info(f"Auto reload: {reload}")
    logger.info(f"Graceful shutdown timeout: {timeout_graceful_shutdown}s")
    logger.info("=" * 60)

    # 启动 uvicorn 服务器
    # workers > 1 时使用多进程模式实现高并发
    uvicorn.run(
        "server:app",  # 使用字符串形式以便多进程加载
        host=host,
        port=port,
        workers=workers if not reload else 1,  # 开发模式下只能使用单进程
        log_level=log_level,
        reload=reload,
        access_log=True,
        timeout_graceful_shutdown=timeout_graceful_shutdown,
    )


if __name__ == "__main__":
    import argparse

    # 使用示例
    examples = """
使用示例:

  # 使用默认配置启动（自动根据 CPU 核心数计算 worker 数量）
  python server.py

  # 指定端口和日志级别
  python server.py --port 9000 --log-level debug

  # 开发模式（启用自动重载，单进程，不写日志文件）
  python server.py --reload --log-level debug --no-log-file

  # 生产环境（指定 worker 数量，日志自动写入文件）
  python server.py --host 0.0.0.0 --port 8080 --workers 8 --log-level info

  # 指定日志文件路径
  python server.py --log-file custom.log

  # 查看帮助信息
  python server.py --help
"""

    parser = argparse.ArgumentParser(
        description="Start the Coordinate Recognizer API service",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("API_SERVER_HOST", serverconfig.DEFAULT_HOST),
        help="Service bind address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_SERVER_PORT", str(serverconfig.DEFAULT_PORT))),
        help="Service port",
    )
    # 计算默认 worker 数量
    default_workers = get_default_workers()
    env_workers = os.getenv("API_SERVER_WORKERS")
    if env_workers:
        default_workers = int(env_workers)

    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Worker process count for high concurrency, default: {default_workers}, computed from CPU cores)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", serverconfig.DEFAULT_LOG_LEVEL),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto reload (development mode, single process only)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=os.getenv("LOG_FILE", None),
        help="Log file path (default: auto-generated as server_YYYYMMDD_HHMMSS.log)",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable log file and write only to console",
    )
    parser.add_argument(
        "--timeout-graceful-shutdown",
        type=int,
        default=30,
        help="Graceful shutdown timeout in seconds, default: 30",
    )

    args = parser.parse_args()
    
    # 确定是否启用日志文件（命令行参数优先，否则使用配置文件默认值）
    enable_log_file = serverconfig.DEFAULT_ENABLE_LOG_FILE if not args.no_log_file else False

    start_server(
        host=args.host,
        port=args.port,
        workers=args.workers, 
        log_level=args.log_level,
        reload=args.reload,
        log_file=args.log_file, 
        enable_log_file=enable_log_file,
        timeout_graceful_shutdown=args.timeout_graceful_shutdown,
    )
