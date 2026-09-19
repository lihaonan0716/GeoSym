"""PaddleOCR-backed OCR rule baseline for locked GLM-4.6V replication."""

from __future__ import annotations

import logging
import re
import sys
import tempfile
import types
import uuid
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

from evaluator.baselines.ocr_rule_baseline import OCRRuleBaseline

logger = logging.getLogger(__name__)


class PaddleOCRBaselineError(RuntimeError):
    """Fail-closed PaddleOCR baseline error."""


def _is_canonical_integer_token(text: str) -> bool:
    token = text.strip()
    if not re.fullmatch(r"-?\d+(?:\.0+)?", token):
        return False
    try:
        value = int(float(token))
    except (TypeError, ValueError, OverflowError):
        return False
    return token == str(value)


def _polygon_center(poly: Any, factor: int) -> tuple[int, int, int, int] | None:
    if not isinstance(poly, (list, tuple)) or not poly:
        return None
    try:
        xs = [float(point[0]) for point in poly]
        ys = [float(point[1]) for point in poly]
    except Exception:
        return None
    cx = int(sum(xs) / len(xs)) // factor
    cy = int(sum(ys) / len(ys)) // factor
    bw = int(max(xs) - min(xs)) // factor
    bh = int(max(ys) - min(ys)) // factor
    return cx, cy, bw, bh


def _payload_from_paddle_result(result: Any) -> dict[str, Any] | None:
    payload = result.json if hasattr(result, "json") else result
    if isinstance(payload, dict) and "res" in payload:
        payload = payload["res"]
    return payload if isinstance(payload, dict) else None


def labels_from_paddle_payload(payload: dict[str, Any], *, factor: int) -> List[dict[str, Any]]:
    texts = payload.get("rec_texts") or payload.get("texts") or []
    polys = payload.get("rec_polys") or payload.get("dt_polys") or payload.get("polys") or []
    labels: List[dict[str, Any]] = []
    for idx, text_obj in enumerate(texts):
        text = str(text_obj or "").strip()
        if not _is_canonical_integer_token(text):
            continue
        center = _polygon_center(polys[idx] if idx < len(polys) else None, factor)
        if center is None:
            continue
        cx, cy, bw, bh = center
        labels.append({"value": int(float(text)), "px": cx, "py": cy, "bw": bw, "bh": bh})
    return labels


class PaddleOCRRuleBaseline(OCRRuleBaseline):
    """OCRRuleBaseline variant that reads labels with local PaddleOCR only."""

    def __init__(self, *, device: str = "gpu:0") -> None:
        self.device = device
        self.local_inference_jobs = 0
        self.last_error_reason: str | None = None
        self._reader = None

    def extract_coordinates(self, image_path: str | Path, *, curve_color_bgr=(0, 0, 139)) -> List[tuple[int, int]]:
        self.last_error_reason = None
        try:
            coords = super().extract_coordinates(image_path, curve_color_bgr=curve_color_bgr)
        except PaddleOCRBaselineError:
            raise
        except Exception as exc:
            self.last_error_reason = f"paddleocr_rule_baseline_error:{type(exc).__name__}:{exc}"
            raise PaddleOCRBaselineError(self.last_error_reason) from exc
        if not coords:
            self.last_error_reason = self.last_error_reason or "paddleocr_rule_baseline_no_coordinates"
            raise PaddleOCRBaselineError(self.last_error_reason)
        return coords

    def _get_reader(self):
        if self._reader is None:
            self._install_modelscope_import_shim()
            from paddleocr import PaddleOCR  # type: ignore

            self._reader = PaddleOCR(
                lang="en",
                device=self.device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._reader

    @staticmethod
    def _install_modelscope_import_shim() -> None:
        if "modelscope" in sys.modules:
            return
        modelscope = types.ModuleType("modelscope")

        def disabled_snapshot_download(*args, **kwargs):
            raise RuntimeError("modelscope_download_disabled_for_locked_cached_paddleocr")

        modelscope.snapshot_download = disabled_snapshot_download  # type: ignore[attr-defined]
        sys.modules["modelscope"] = modelscope
        sys.modules.setdefault("modelscope.hub", types.ModuleType("modelscope.hub"))
        sys.modules.setdefault("modelscope.hub.errors", types.ModuleType("modelscope.hub.errors"))

    def _run_ocr(self, img: np.ndarray, *, upscale: bool = True) -> List[dict[str, Any]]:
        factor = 2 if upscale else 1
        ocr_img = img
        if factor > 1:
            ocr_img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        tmp_path = None
        try:
            tmp_path = Path(tempfile.gettempdir()) / f"geosym_paddleocr_{uuid.uuid4().hex}.png"
            cv2.imwrite(str(tmp_path), ocr_img)
            reader = self._get_reader()
            labels: List[dict[str, Any]] = []
            self.local_inference_jobs += 1
            if hasattr(reader, "predict"):
                for result in reader.predict(str(tmp_path)) or []:
                    payload = _payload_from_paddle_result(result)
                    if payload:
                        labels.extend(labels_from_paddle_payload(payload, factor=factor))
            elif hasattr(reader, "ocr"):
                raw = reader.ocr(str(tmp_path), cls=False)
                rows = raw[0] if raw and isinstance(raw[0], list) else raw
                payload = {"rec_texts": [], "rec_polys": []}
                for item in rows or []:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        text_conf = item[1]
                        text = text_conf[0] if isinstance(text_conf, (list, tuple)) and text_conf else text_conf
                        payload["rec_texts"].append(text)
                        payload["rec_polys"].append(item[0])
                labels.extend(labels_from_paddle_payload(payload, factor=factor))
            return labels
        except Exception as exc:
            self.last_error_reason = f"paddleocr_backend_error:{type(exc).__name__}:{exc}"
            raise PaddleOCRBaselineError(self.last_error_reason) from exc
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    logger.debug("Failed to remove temporary PaddleOCR image: %s", tmp_path)

    def _ocr_tick_labels(self, img: np.ndarray):
        x_ticks, y_ticks = super()._ocr_tick_labels(img)
        if len(x_ticks) < 2 or len(y_ticks) < 2:
            self.last_error_reason = f"paddleocr_insufficient_tick_labels:x={len(x_ticks)}:y={len(y_ticks)}"
            raise PaddleOCRBaselineError(self.last_error_reason)
        return x_ticks, y_ticks
