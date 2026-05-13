"""
Adapter for calling recognizer via local import.
The recognizer/ directory resides inside Experiment/ and is imported
directly as a Python package (no pip install needed).
Uses the complete pipeline including OCR OCR - identical to recognizer's own flow.
"""

import base64
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from recognizer import CoordinateProcessor
from recognizer.core.chart_extractor import ChartExtractor
from i18n import t, get_lang


class RecognizerAdapter:
    """
    Adapter to call recognizer for intersection detection.
    Uses the complete pipeline including OCR OCR.
    """

    def __init__(self):
        self._processor = CoordinateProcessor()
        self._chart_extractor = ChartExtractor()

    def recognize_intersections(
        self,
        image: np.ndarray | str | Path,
        gt: Dict[str, Any] = None,
        skip_enhancement: bool = True,
        gemini_enabled: bool = False,
    ) -> Tuple[List[Tuple[int, int]], Dict[str, Any], np.ndarray | None]:
        """
        Run recognizer complete pipeline on image.
        This is identical to how recognizer processes images.

        Args:
            image: Image as numpy array (BGR), file path, or Path
            gt: Ground truth dict (optional, not used - kept for API compatibility)
            skip_enhancement: If True, disable stage 0.5 (AI image enhancement)
            gemini_enabled: If True, use Gemini API for OCR label correction

        Returns:
            (list of (x, y) integer tuples, full result dict, annotated result image or None)
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(t("eval_recognizer_load_failed", get_lang()).format(path=image))

        # Stage 0: Extract chart using OCR OCR (same as recognizer)
        extract_result = self._chart_extractor.extract_chart_from_image(
            image, gemini_enabled=gemini_enabled)
        
        if not extract_result.get('success'):
            error_msg = extract_result.get('error', t("eval_recognizer_extract_failed", get_lang()))
            return [], {'success': False, 'error': error_msg}, None

        # Get OCR result from chart extractor
        ocr_result = extract_result.get('ocr_result', {})

        # Encode image to base64 for processor
        _, buffer = cv2.imencode(".png", image)
        image_b64 = f"data:image/png;base64,{base64.b64encode(buffer.tobytes()).decode()}"

        # Config: mirror the interactive function-specialized M0 demo defaults
        # so the M0 Test tab and the step-by-step recognizer tab stay aligned.
        config = {
            "stage0": {
                "horizontal_tolerance": 1.5,
                "vertical_tolerance": 1.5,
                "interval_percentile": 50,
            },
            "stage05": {"enabled": not skip_enhancement},
            "stage1": {
                "enable_curve_erase": True,
                "enable_dashed_line_connection": True,
                "enable_perspective_correction": True,
                "enable_rotation_correction": True,
                "rotation_hough_threshold": 60,
                "rotation_min_line_length": 60,
                "rotation_max_line_gap": 10,
                "rotation_angle_threshold": 5,
                "grid_hough_threshold": 65,
                "grid_min_line_length": 45,
                "grid_max_line_gap": 10,
                "grid_angle_threshold": 5,
                "merge_distance_threshold": 20,
                "dashed_gap_threshold": 10,
                "dashed_collinear_threshold": 3,
                "curve_saturation_threshold": 90,
                "curve_value_threshold": 60,
                "curve_feather_size": 3,
            },
            "stage2": {
                "tolerance": 3,
                "enable_morph_connect": True,
                "morph_kernel_size": 7,
                "morph_iterations": 2,
                "color_saturation_threshold": 90,
                "color_value_threshold": 60,
                "integer_approximation_threshold": 0.1,
            },
        }

        # Run recognizer's complete pipeline with real OCR OCR result
        result = self._processor.process_full(
            image_input=image_b64,
            ocr_result=ocr_result,
            config=config,
        )

        # Extract intersections
        coords = []
        if result.get("success") and result.get("result"):
            payload = result["result"]
            intersections = (
                payload.get("intersections")
                or payload.get("reasoning_intersections")
                or payload.get("intersections_raw", [])
            )
            for pt in intersections:
                x = pt.get("real_x_rounded")
                y = pt.get("real_y_rounded")
                if x is None:
                    x = pt.get("real_x")
                if y is None:
                    y = pt.get("real_y")
                if x is not None and y is not None:
                    coords.append((int(round(x)), int(round(y))))

        # Get annotated image directly from recognizer result
        annotated_image = None
        if result.get("success") and result.get("result"):
            annotated_image = result["result"].get("annotated_image")

        return coords, result, annotated_image

    def __repr__(self) -> str:
        return "RecognizerAdapter(recognizer)"
