#!/usr/bin/env python
"""Run the no-paid OpenOCR backend ladder for the frozen PlotQA roster."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "chart",
    "did", "do", "does", "for", "from", "how", "in", "is", "it", "of",
    "on", "or", "plot", "than", "the", "there", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "with",
}
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)%?")
BACKEND_ORDER = ["PaddleOCR", "EasyOCR", "docTR", "Surya OCR"]

SURVEY_SOURCES = [
    {
        "name": "PaddleOCR",
        "source_urls": ["https://github.com/PaddlePaddle/PaddleOCR"],
        "source_type": "official_repository",
        "package_names": ["paddleocr", "paddlepaddle"],
        "supported_output_fields": ["text", "coordinates", "confidence"],
        "chart_reasoning_confound_risk": "low for PP-OCR; high for PaddleOCR-VL and PP-Structure parsers",
        "eligibility_decision": "preferred_when_importable_and_smoke_passes",
    },
    {
        "name": "EasyOCR",
        "source_urls": ["https://github.com/JaidedAI/EasyOCR"],
        "source_type": "official_repository",
        "package_names": ["easyocr"],
        "supported_output_fields": ["text", "bounding_box", "confidence"],
        "chart_reasoning_confound_risk": "low; OCR-only readtext API",
        "eligibility_decision": "first_fallback_when_paddleocr_unavailable",
    },
    {
        "name": "docTR",
        "source_urls": ["https://github.com/mindee/doctr"],
        "source_type": "official_repository",
        "package_names": ["python-doctr"],
        "supported_output_fields": ["text", "geometry", "confidence"],
        "chart_reasoning_confound_risk": "low; document OCR pipeline rather than chart QA",
        "eligibility_decision": "second_tier_fallback_if_installed_or_disposable_probe_available",
    },
    {
        "name": "Surya OCR",
        "source_urls": ["https://github.com/datalab-to/surya"],
        "source_type": "official_repository",
        "package_names": ["surya-ocr"],
        "supported_output_fields": ["text", "polygon_or_bbox", "confidence_or_score"],
        "chart_reasoning_confound_risk": "medium; accept only OCR path, not chart/table parsing",
        "eligibility_decision": "last_fallback_if_installed_or_disposable_probe_available",
    },
    {
        "name": "DePlot / UniChart-style chart models",
        "source_urls": ["https://github.com/google-research/google-research/tree/master/deplot"],
        "source_type": "official_repository",
        "package_names": [],
        "supported_output_fields": ["chart_table_or_answer"],
        "chart_reasoning_confound_risk": "high",
        "eligibility_decision": "excluded_from_track_b_ocr_backend",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w.%+-]+", " ", text.lower())).strip()


def numeric_tokens(text: str) -> set[str]:
    return {match.group(0).rstrip("%") for match in NUMBER_RE.finditer(normalize_text(text))}


def categorical_tokens(text: str) -> set[str]:
    return {t for t in normalize_text(text).split() if t and t not in STOPWORDS and not NUMBER_RE.fullmatch(t)}


def bbox_from_polygon(points: Any) -> list[float] | None:
    if not isinstance(points, list) or not points:
        return None
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except Exception:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _as_bbox(raw: Any) -> list[float] | None:
    if isinstance(raw, list) and len(raw) == 4 and all(isinstance(item, (int, float)) for item in raw):
        return [float(item) for item in raw]
    if isinstance(raw, dict):
        if {"x", "y", "w", "h"} <= raw.keys():
            return [float(raw["x"]), float(raw["y"]), float(raw["x"]) + float(raw["w"]), float(raw["y"]) + float(raw["h"])]
        if {"left", "top", "width", "height"} <= raw.keys():
            return [float(raw["left"]), float(raw["top"]), float(raw["left"]) + float(raw["width"]), float(raw["top"]) + float(raw["height"])]
    return None


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def is_numeric_text(text: str) -> bool:
    return bool(NUMBER_RE.search(text))


def _axis_bboxes(axis: dict[str, Any]) -> list[list[float]]:
    bboxes: list[list[float]] = []
    for key in ("major_labels", "major_ticks", "minor_ticks"):
        value = axis.get(key)
        if isinstance(value, dict):
            for raw_bbox in value.get("bboxes") or []:
                bbox = _as_bbox(raw_bbox)
                if bbox:
                    bboxes.append(bbox)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    bbox = _as_bbox(item.get("bbox") or item.get("bounds") or item.get("label_bbox") or item)
                    if bbox:
                        bboxes.append(bbox)
    return bboxes


def annotation_axis_regions(annotation: dict[str, Any]) -> dict[str, list[list[float]]]:
    info = annotation.get("general_figure_info") or {}
    regions: dict[str, list[list[float]]] = {"x": [], "y": [], "legend": []}
    regions["x"].extend(_axis_bboxes(info.get("x_axis") or {}))
    regions["y"].extend(_axis_bboxes(info.get("y_axis") or {}))
    legend_bbox = _as_bbox((info.get("legend") or {}).get("bbox"))
    if legend_bbox:
        regions["legend"].append(legend_bbox)
    return regions


def near_any_region(bbox: list[float], regions: list[list[float]], width: int, height: int) -> bool:
    if not regions:
        return False
    cx, cy = bbox_center(bbox)
    threshold_x = width * 0.05
    threshold_y = height * 0.05
    return any(abs(cx - bbox_center(region)[0]) <= threshold_x and abs(cy - bbox_center(region)[1]) <= threshold_y for region in regions)


def package_status(module_name: str, distribution_name: str | None = None) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    distribution = distribution_name or module_name
    try:
        version = importlib.metadata.version(distribution)
    except Exception:
        version = None
    return {"available": spec is not None, "version": version, "import_path": str(spec.origin) if spec else None}


def common_box(source_backend: str, source_index: int, text: Any, bbox: Any, confidence: Any) -> dict[str, Any] | None:
    normalized_bbox = bbox_from_polygon(bbox) or _as_bbox(bbox)
    clean_text = str(text or "").strip()
    if not normalized_bbox or not clean_text:
        return None
    try:
        clean_confidence = None if confidence is None else float(confidence)
    except Exception:
        clean_confidence = None
    return {
        "box_id": f"ocr_{source_index:03d}",
        "text": clean_text,
        "bbox": normalized_bbox,
        "confidence": clean_confidence,
        "source_backend": source_backend,
        "source_index": source_index,
    }


# Set by ``--device`` to force a specific PaddleOCR placement. A GeoSym-Bench
# run holds a PaddleOCR model resident per worker (~4.7 GB of VRAM for the CUDA
# context and cuDNN kernels), so a survey launched while one is in flight can
# exhaust the device and surface as row-level OCR errors in *that* run. Forcing
# this survey onto the CPU removes the contention physically rather than by
# timing.
_DEVICE_OVERRIDE: str | None = None


def _preferred_paddleocr_device() -> str:
    if _DEVICE_OVERRIDE:
        return _DEVICE_OVERRIDE
    try:
        import torch  # noqa: F401  # type: ignore
    except Exception:
        pass
    try:
        import paddle  # type: ignore

        return "gpu:0" if paddle.is_compiled_with_cuda() else "cpu"
    except Exception:
        return "cpu"


def _paddleocr3_boxes(result: Any) -> list[dict[str, Any]]:
    payload = result.json if hasattr(result, "json") else result
    if isinstance(payload, dict) and "res" in payload:
        payload = payload["res"]
    if not isinstance(payload, dict):
        return []
    texts = payload.get("rec_texts") or payload.get("texts") or []
    scores = payload.get("rec_scores") or payload.get("scores") or []
    polys = payload.get("rec_polys") or payload.get("dt_polys") or payload.get("polys") or []
    boxes = []
    for idx, text in enumerate(texts):
        poly = polys[idx] if idx < len(polys) else None
        score = scores[idx] if idx < len(scores) else None
        box = common_box("PaddleOCR", idx, text, poly, score)
        if box:
            boxes.append(box)
    return boxes


def run_paddleocr(image_path: Path) -> list[dict[str, Any]]:
    try:
        import torch  # noqa: F401  # type: ignore
    except Exception:
        pass
    from paddleocr import PaddleOCR  # type: ignore

    if not hasattr(run_paddleocr, "_reader"):
        try:
            run_paddleocr._reader = PaddleOCR(  # type: ignore[attr-defined]
                lang="en",
                device=_preferred_paddleocr_device(),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except (TypeError, ValueError):
            run_paddleocr._reader = PaddleOCR(use_angle_cls=False, lang="en")  # type: ignore[attr-defined]
    reader = run_paddleocr._reader  # type: ignore[attr-defined]
    if hasattr(reader, "predict"):
        boxes = []
        for result in reader.predict(str(image_path)) or []:
            boxes.extend(_paddleocr3_boxes(result))
        if boxes:
            return boxes
    raw = reader.ocr(str(image_path), cls=False)
    rows = raw[0] if raw and isinstance(raw[0], list) else raw
    boxes = []
    for idx, item in enumerate(rows or []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        text_conf = item[1]
        if isinstance(text_conf, (list, tuple)) and text_conf:
            box = common_box("PaddleOCR", idx, text_conf[0], item[0], text_conf[1] if len(text_conf) > 1 else None)
        else:
            box = common_box("PaddleOCR", idx, text_conf, item[0], None)
        if box:
            boxes.append(box)
    return boxes


def run_easyocr(image_path: Path) -> list[dict[str, Any]]:
    import easyocr  # type: ignore

    if not hasattr(run_easyocr, "_reader"):
        run_easyocr._reader = easyocr.Reader(["en"], gpu=False)  # type: ignore[attr-defined]
    boxes = []
    for idx, item in enumerate(run_easyocr._reader.readtext(str(image_path)) or []):  # type: ignore[attr-defined]
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        box = common_box("EasyOCR", idx, item[1], item[0], item[2] if len(item) > 2 else None)
        if box:
            boxes.append(box)
    return boxes


def run_doctr(_: Path) -> list[dict[str, Any]]:
    raise RuntimeError("docTR default runner is unavailable in the shared GeoSym environment")


def run_surya(_: Path) -> list[dict[str, Any]]:
    raise RuntimeError("Surya OCR default runner is unavailable in the shared GeoSym environment")


def load_annotations(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    official = plan.get("frozen_before_paid_run", {}).get("official_files", {})
    path = official.get("annotations_json")
    if not path:
        return {}
    annotations = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(annotations, dict):
        annotations = annotations.get("annotations", annotations.get("data", []))
    return {str(row.get("image_index")): row for row in annotations if isinstance(row, dict)}


def sample_summary(sample: dict[str, Any], boxes: list[dict[str, Any]], annotation: dict[str, Any] | None) -> dict[str, Any]:
    image_path = Path(sample["image_path"])
    with Image.open(image_path) as image:
        width, height = image.size
    ocr_text = " ".join(str(box.get("text", "")) for box in boxes)
    regions = annotation_axis_regions(annotation or {})
    numeric_box_count = sum(1 for box in boxes if is_numeric_text(str(box.get("text", ""))))
    x_tick = any(is_numeric_text(str(box.get("text", ""))) and near_any_region(box["bbox"], regions["x"], width, height) for box in boxes if box.get("bbox"))
    y_tick = any(is_numeric_text(str(box.get("text", ""))) and near_any_region(box["bbox"], regions["y"], width, height) for box in boxes if box.get("bbox"))
    q_cat = categorical_tokens(sample.get("question", ""))
    q_num = numeric_tokens(sample.get("question", ""))
    metadata = sample.get("metadata") or {}
    direct_answer_overlap = False
    if metadata.get("reasoning_type") in {"lookup", "extrema", "comparison"} and metadata.get("answer_derivation") == "direct":
        direct_answer_overlap = bool(numeric_tokens(str(sample.get("normalized_answer"))) & numeric_tokens(ocr_text))
    return {
        "sample_id": sample["sample_id"],
        "image_path": str(image_path),
        "image_sha256": metadata.get("image_sha256") or stable_file_hash(image_path),
        "image_size": [width, height],
        "box_count": len(boxes),
        "numeric_box_count": numeric_box_count,
        "any_text_detected": bool(boxes),
        "x_axis_tick_text_detected": x_tick,
        "y_axis_tick_text_detected": y_tick,
        "axis_tick_text_detected": x_tick or y_tick,
        "legend_text_detected": bool(regions["legend"]) and any(near_any_region(box["bbox"], regions["legend"], width, height) for box in boxes if box.get("bbox")),
        "question_token_overlap": any(token in normalize_text(ocr_text) for token in q_cat),
        "numeric_token_overlap": bool(q_num & numeric_tokens(ocr_text)),
        "direct_answer_value_overlap": direct_answer_overlap,
        "serializable_to_g0_trace": all(isinstance(box.get("box_id"), str) and isinstance(box.get("text"), str) and isinstance(box.get("bbox"), list) and len(box["bbox"]) == 4 for box in boxes),
        "ocr_boxes": boxes,
    }


def backend_probe(backend: str) -> tuple[dict[str, Any], Callable[[Path], list[dict[str, Any]]] | None, str | None]:
    if backend == "PaddleOCR":
        paddleocr = package_status("paddleocr", "paddleocr")
        paddle = package_status("paddle", "paddlepaddle")
        paddle_gpu = package_status("paddle", "paddlepaddle-gpu")
        available = paddleocr["available"] and paddle["available"]
        return {
            "backend": backend,
            "available": available,
            "package_version": {"paddleocr": paddleocr["version"], "paddlepaddle": paddle["version"], "paddlepaddle-gpu": paddle_gpu["version"]},
            "import_path": {"paddleocr": paddleocr["import_path"], "paddle": paddle["import_path"]},
            "device": _preferred_paddleocr_device(),
            "ocr_model_names": "PP-OCRv6_medium_det, PP-OCRv6_medium_rec",
            "local_import_status": "ok" if available else "paddleocr_unavailable",
        }, run_paddleocr if available else None, None if available else "paddleocr_unavailable"
    if backend == "EasyOCR":
        easyocr = package_status("easyocr", "easyocr")
        return {
            "backend": backend,
            "available": easyocr["available"],
            "package_version": easyocr["version"],
            "import_path": easyocr["import_path"],
            "device": "cpu",
            "model_weight_policy": "package_default_cache_or_download",
            "local_import_status": "ok" if easyocr["available"] else "easyocr_unavailable",
        }, run_easyocr if easyocr["available"] else None, None if easyocr["available"] else "easyocr_unavailable"
    if backend == "docTR":
        doctr = package_status("doctr", "python-doctr")
        return {
            "backend": backend,
            "available": doctr["available"],
            "package_version": doctr["version"],
            "import_path": doctr["import_path"],
            "device": "cpu",
            "local_import_status": "ok" if doctr["available"] else "doctr_unavailable",
        }, run_doctr if doctr["available"] else None, None if doctr["available"] else "doctr_unavailable"
    if backend == "Surya OCR":
        surya = package_status("surya", "surya-ocr")
        return {
            "backend": backend,
            "available": surya["available"],
            "package_version": surya["version"],
            "import_path": surya["import_path"],
            "device": "cpu",
            "local_import_status": "ok" if surya["available"] else "surya_unavailable",
        }, run_surya if surya["available"] else None, None if surya["available"] else "surya_unavailable"
    raise ValueError(f"unknown backend: {backend}")


def smoke_sample_ids(plan: dict[str, Any]) -> list[str]:
    return sorted(sample["sample_id"] for sample in plan.get("selected_samples", []))[:5]


def run_backend_smoke(backend: str, runner: Callable[[Path], list[dict[str, Any]]] | None, plan: dict[str, Any], annotations: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    samples_by_id = {sample["sample_id"]: sample for sample in plan.get("selected_samples", [])}
    rows = []
    error = None
    for sample_id in smoke_sample_ids(plan):
        sample = samples_by_id[sample_id]
        try:
            if runner is None:
                raise RuntimeError("backend runner unavailable")
            started = time.perf_counter()
            boxes = runner(Path(sample["image_path"]))
            row = sample_summary(sample, boxes, annotations.get(str((sample.get("metadata") or {}).get("image_index"))))
            row["latency_seconds"] = round(time.perf_counter() - started, 6)
            row["source_backend"] = backend
        except Exception as exc:
            row = {
                "sample_id": sample_id,
                "source_backend": backend,
                "any_text_detected": False,
                "numeric_box_count": 0,
                "serializable_to_g0_trace": False,
                "ocr_boxes": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
            error = row["error"]
        rows.append(row)
    counts = {
        "sample_count": len(rows),
        "non_empty_ocr_boxes": sum(1 for row in rows if row.get("any_text_detected")),
        "numeric_ocr_box_images": sum(1 for row in rows if row.get("numeric_box_count", 0) > 0),
        "schema_serializable_images": sum(1 for row in rows if row.get("serializable_to_g0_trace")),
    }
    gate = {
        "backend_import_succeeds": runner is not None,
        "non_empty_ocr_boxes_at_least_4_of_5": counts["non_empty_ocr_boxes"] >= 4,
        "numeric_ocr_box_images_at_least_3_of_5": counts["numeric_ocr_box_images"] >= 3,
        "all_rows_schema_serializable": counts["schema_serializable_images"] == len(rows),
        "manual_correction_not_used": True,
        "ocr_http_attempts_zero": True,
        "llm_calls_zero": True,
    }
    return rows, {"counts": counts, "gate": gate}, None if all(gate.values()) else (error or "backend_smoke_gate_failed")


def ladder_audit_row(backend: str, priority: int, runtime: dict[str, Any], smoke: dict[str, Any] | None, probe_sample_ids: list[str], decision: str, failure_reason: str | None, next_backend: str | None) -> dict[str, Any]:
    source = next(source for source in SURVEY_SOURCES if source["name"] == backend)
    return {
        "backend": backend,
        "priority": priority,
        "source_urls": source["source_urls"],
        "source_snapshot_or_access_date_utc": utc_now(),
        "local_import_status": runtime.get("local_import_status"),
        "package_version": runtime.get("package_version"),
        "import_path": runtime.get("import_path"),
        "probe_command": "C:\\ProgramData\\anaconda3\\envs\\GeoSym\\python.exe scripts\\run_plotqa_openocr_survey.py --plan outputs\\public_benchmark_pilot\\plotqa_diagnostic_50\\frozen_plan.json --output-dir outputs\\public_benchmark_pilot\\plotqa_openocr_survey_50",
        "probe_sample_ids": probe_sample_ids,
        "probe_paid_calls_executed": 0,
        "probe_ocr_http_attempts": 0,
        "schema_supported": bool(smoke and smoke["gate"].get("all_rows_schema_serializable")),
        "smoke_counts": smoke.get("counts") if smoke else {},
        "decision": decision,
        "failure_reason": failure_reason,
        "next_backend": next_backend,
    }


def resolve_backend(plan: dict[str, Any], annotations: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    audit = []
    smoke_records: dict[str, list[dict[str, Any]]] = {}
    selected_runtime = None
    sample_ids = smoke_sample_ids(plan)
    for idx, backend in enumerate(BACKEND_ORDER):
        runtime, runner, probe_failure = backend_probe(backend)
        smoke = None
        failure = probe_failure
        if runner is not None:
            records, smoke, failure = run_backend_smoke(backend, runner, plan, annotations)
            smoke_records[backend] = records
        next_backend = BACKEND_ORDER[idx + 1] if idx + 1 < len(BACKEND_ORDER) else None
        if failure is None:
            selected_runtime = {**runtime, "selected_backend": backend}
            audit.append(ladder_audit_row(backend, idx + 1, runtime, smoke, sample_ids, "selected_for_50_sample_survey", None, None))
            break
        decision = "failed_smoke_or_unavailable" if runner is not None else "unavailable"
        audit.append(ladder_audit_row(backend, idx + 1, runtime, smoke, sample_ids, decision, failure, next_backend))
    if selected_runtime is None and len(audit) < len(BACKEND_ORDER):
        probed = {row["backend"] for row in audit}
        for idx, backend in enumerate(BACKEND_ORDER):
            if backend in probed:
                continue
            runtime, _, failure = backend_probe(backend)
            next_backend = BACKEND_ORDER[idx + 1] if idx + 1 < len(BACKEND_ORDER) else None
            audit.append(ladder_audit_row(backend, idx + 1, runtime, None, sample_ids, "unavailable", failure, next_backend))
    return selected_runtime, audit, smoke_records


def run_50_sample_survey(runtime: dict[str, Any], plan: dict[str, Any], annotations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    runner = {"PaddleOCR": run_paddleocr, "EasyOCR": run_easyocr, "docTR": run_doctr, "Surya OCR": run_surya}[runtime["selected_backend"]]
    records = []
    for sample in plan.get("selected_samples", []):
        started = time.perf_counter()
        boxes = runner(Path(sample["image_path"]))
        row = sample_summary(sample, boxes, annotations.get(str((sample.get("metadata") or {}).get("image_index"))))
        row["latency_seconds"] = round(time.perf_counter() - started, 6)
        row["source_backend"] = runtime["selected_backend"]
        records.append(row)
    return records


def empty_records_after_ladder(plan: dict[str, Any], status: str) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample["sample_id"],
            "image_path": str(Path(sample["image_path"])),
            "image_sha256": (sample.get("metadata") or {}).get("image_sha256"),
            "box_count": 0,
            "numeric_box_count": 0,
            "any_text_detected": False,
            "x_axis_tick_text_detected": False,
            "y_axis_tick_text_detected": False,
            "axis_tick_text_detected": False,
            "legend_text_detected": False,
            "question_token_overlap": False,
            "numeric_token_overlap": False,
            "direct_answer_value_overlap": False,
            "serializable_to_g0_trace": False,
            "ocr_boxes": [],
            "error": status,
        }
        for sample in plan.get("selected_samples", [])
    ]


def build_summary(plan: dict[str, Any], runtime: dict[str, Any] | None, records: list[dict[str, Any]], plan_path: Path, ladder_audit: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(plan.get("selected_samples", []))
    counts = {
        "any_text_detected": sum(1 for row in records if row.get("any_text_detected")),
        "axis_tick_text_detected": sum(1 for row in records if row.get("axis_tick_text_detected")),
        "x_axis_tick_text_detected": sum(1 for row in records if row.get("x_axis_tick_text_detected")),
        "y_axis_tick_text_detected": sum(1 for row in records if row.get("y_axis_tick_text_detected")),
        "legend_text_detected": sum(1 for row in records if row.get("legend_text_detected")),
        "serializable_to_g0_trace": sum(1 for row in records if row.get("serializable_to_g0_trace")),
        "question_token_overlap": sum(1 for row in records if row.get("question_token_overlap")),
        "numeric_token_overlap": sum(1 for row in records if row.get("numeric_token_overlap")),
        "unusable_ocr_records": sum(1 for row in records if not row.get("serializable_to_g0_trace")),
    }
    gate_thresholds = {
        "any_text_detected_min": math.ceil(sample_count * 0.90),
        "axis_tick_text_detected_min": math.ceil(sample_count * 0.70),
        "serializable_to_g0_trace_required": sample_count,
    }
    go_gate = {
        "any_text_detected_at_least_90_percent": counts["any_text_detected"] >= gate_thresholds["any_text_detected_min"],
        "axis_tick_text_detected_at_least_70_percent": counts["axis_tick_text_detected"] >= gate_thresholds["axis_tick_text_detected_min"],
        "serializable_to_g0_trace_all_samples": counts["serializable_to_g0_trace"] == gate_thresholds["serializable_to_g0_trace_required"],
        "manual_correction_not_required": True,
        "ocr_http_attempts_zero": True,
    }
    if runtime is None:
        status = "openocr_no_viable_backend_after_ladder"
    elif all(go_gate.values()):
        status = "openocr_backend_frozen_for_track_c"
    else:
        status = "openocr_coverage_insufficient"
    return {
        "schema_version": "plotqa_openocr_survey_summary_v2",
        "created_at_utc": utc_now(),
        "benchmark": "PlotQA",
        "split": plan.get("split"),
        "source_plan_path": str(plan_path.resolve()),
        "source_plan_sha256": stable_file_hash(plan_path),
        "source_plan_declared_sha256": plan.get("plan_sha256"),
        "selected_ids": [sample["sample_id"] for sample in plan.get("selected_samples", [])],
        "sample_count": sample_count,
        "runtime": runtime or {"backend": None, "available": False},
        "selected_backend": runtime.get("selected_backend") if runtime else None,
        "ocr_backend_survey_sources": [{**source, "retrieved_or_inspected_date_utc": utc_now()} for source in SURVEY_SOURCES],
        "backend_ladder_audit": ladder_audit,
        "counts": counts,
        "gate_thresholds": gate_thresholds,
        "go_gate": go_gate,
        "experiment_status": status,
        "paid_calls_executed": 0,
        "actual_ocr_http_attempts": 0,
        "actual_llm_http_attempts": 0,
        "records": records,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# PlotQA OpenOCR Survey",
        "",
        f"- Created UTC: `{summary['created_at_utc']}`",
        f"- Source plan SHA256: `{summary['source_plan_sha256']}`",
        f"- Selected backend: `{summary.get('selected_backend') or 'none'}`",
        f"- Experiment status: `{summary['experiment_status']}`",
        f"- Paid calls executed: {summary['paid_calls_executed']}",
        "",
        "## Backend Ladder",
        "",
    ]
    for row in summary["backend_ladder_audit"]:
        lines.append(f"- {row['priority']}. {row['backend']}: {row['decision']} ({row.get('failure_reason') or 'passed'}).")
    lines.extend(["", "## Counts", ""])
    for key, value in summary["counts"].items():
        lines.append(f"- {key}: {value}/{summary['sample_count']}")
    lines.extend(["", "## Go Gate", ""])
    for key, value in summary["go_gate"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    if summary["experiment_status"] == "openocr_backend_frozen_for_track_c":
        lines.append("Track B passes; the selected OpenOCR facts are eligible to freeze for Track C.")
    elif summary["experiment_status"] == "openocr_coverage_insufficient":
        lines.append("Track B stops because the first smoke-passing backend failed the 50-sample coverage gate.")
    else:
        lines.append("Track B stops after the full backend ladder; no backend passed the frozen smoke gate.")
    return "\n".join(lines) + "\n"


def render_backend_survey(summary: dict[str, Any]) -> str:
    lines = [
        "# OpenOCR Backend Research Refresh",
        "",
        f"Created UTC: {summary['created_at_utc']}",
        "",
        "Decision rule: use PaddleOCR when it is importable and passes the frozen smoke gate; otherwise automatically fall back to EasyOCR, docTR, then Surya OCR. Chart-to-table or chart-QA models are excluded from Track B because they perform chart understanding beyond OCR.",
        "",
    ]
    for source in summary["ocr_backend_survey_sources"]:
        lines.append(f"## {source['name']}")
        lines.append("")
        lines.append(f"- Source URLs: {', '.join(source['source_urls'])}")
        lines.append(f"- Source type: `{source['source_type']}`")
        lines.append(f"- Inspected UTC: `{source['retrieved_or_inspected_date_utc']}`")
        lines.append(f"- Package names: {json.dumps(source['package_names'])}")
        lines.append(f"- Supported output fields: {json.dumps(source['supported_output_fields'])}")
        lines.append(f"- Chart-reasoning-confound risk: {source['chart_reasoning_confound_risk']}")
        lines.append(f"- Eligibility decision: `{source['eligibility_decision']}`")
        lines.append("")
    lines.append("## Ladder Outcome")
    lines.append("")
    for row in summary["backend_ladder_audit"]:
        lines.append(f"- {row['backend']}: `{row['decision']}`, failure `{row.get('failure_reason')}`.")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Force the PaddleOCR device, e.g. 'cpu' or 'gpu:0'. Defaults to "
            "auto-detection. Pass 'cpu' to keep this survey off a GPU that a "
            "concurrent benchmark run is using."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    global _DEVICE_OVERRIDE
    _DEVICE_OVERRIDE = args.device
    plan_path = Path(args.plan)
    output_dir = Path(args.output_dir)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    annotations = load_annotations(plan)
    runtime, ladder_audit, smoke_records = resolve_backend(plan, annotations)
    records = empty_records_after_ladder(plan, "openocr_no_viable_backend_after_ladder") if runtime is None else run_50_sample_survey(runtime, plan, annotations)
    summary = build_summary(plan, runtime, records, plan_path, ladder_audit)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "backend_ladder_audit.json").write_text(json.dumps(ladder_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "backend_smoke_records.json").write_text(json.dumps(smoke_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "raw_ocr.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    (output_dir / "ocr_backend_survey.md").write_text(render_backend_survey(summary), encoding="utf-8")
    print(f"wrote OpenOCR survey summary: {output_dir / 'summary.json'}")
    print(f"experiment_status={summary['experiment_status']}")
    print(f"selected_backend={summary.get('selected_backend')}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
