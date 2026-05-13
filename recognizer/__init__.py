"""
Coordinate Recognizer - A system for recognizing coordinates from mathematical function graphs.

Usage:
    from recognizer import CoordinateProcessor
    
    processor = CoordinateProcessor()
    result = processor.process_full(image_input, ocr_result, config)
"""

__version__ = "1.0.0"

from .core.processor import CoordinateProcessor
from .core.coordinate_recognizer import CoordinateRecognizer
from .core.chart_extractor import ChartExtractor
from .core.coordinate_system_preprocessor import CoordinateSystemPreprocessor
from .core.drawing_utils import draw_intersections_on_image

__all__ = [
    "CoordinateProcessor",
    "CoordinateRecognizer", 
    "ChartExtractor",
    "CoordinateSystemPreprocessor",
    "draw_intersections_on_image",
]
