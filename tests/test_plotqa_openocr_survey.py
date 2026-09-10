import json
from pathlib import Path

from PIL import Image

from scripts.run_plotqa_openocr_survey import (
    build_summary,
    categorical_tokens,
    empty_records_after_ladder,
    ladder_audit_row,
    numeric_tokens,
    sample_summary,
)


def test_openocr_unavailable_fails_closed(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan = {
        "split": "test",
        "plan_sha256": "declared",
        "selected_samples": [{"sample_id": f"s{i}", "image_path": str(tmp_path / f"{i}.png"), "metadata": {}} for i in range(50)],
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    records = empty_records_after_ladder(plan, "openocr_no_viable_backend_after_ladder")
    audit = [
        ladder_audit_row(
            "PaddleOCR",
            1,
            {"local_import_status": "paddleocr_unavailable"},
            None,
            [f"s{i}" for i in range(5)],
            "unavailable",
            "paddleocr_unavailable",
            "EasyOCR",
        ),
        ladder_audit_row(
            "EasyOCR",
            2,
            {"local_import_status": "easyocr_unavailable"},
            None,
            [f"s{i}" for i in range(5)],
            "unavailable",
            "easyocr_unavailable",
            "docTR",
        ),
    ]
    summary = build_summary(plan, None, records, plan_path, audit)

    assert summary["experiment_status"] == "openocr_no_viable_backend_after_ladder"
    assert summary["counts"]["unusable_ocr_records"] == 50
    assert summary["paid_calls_executed"] == 0
    assert summary["backend_ladder_audit"][0]["failure_reason"] == "paddleocr_unavailable"


def test_overlap_tokenization_uses_frozen_stopwords() -> None:
    assert "chart" not in categorical_tokens("What does this chart show?")
    assert "ukraine" in categorical_tokens("Net enrolment rate in Ukraine")
    assert numeric_tokens("greater than 12% in 2000") == {"12", "2000"}


def test_sample_summary_detects_axis_tick_near_annotation(tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    sample = {
        "sample_id": "s0",
        "image_path": str(image_path),
        "question": "What is the value in Ukraine in 2000?",
        "normalized_answer": 12,
        "metadata": {"reasoning_type": "lookup", "answer_derivation": "direct"},
    }
    boxes = [
        {"box_id": "ocr_000", "text": "2000", "bbox": [45, 85, 55, 95], "confidence": 0.9},
        {"box_id": "ocr_001", "text": "Ukraine", "bbox": [30, 10, 70, 20], "confidence": 0.9},
        {"box_id": "ocr_002", "text": "12", "bbox": [2, 45, 12, 55], "confidence": 0.9},
    ]
    annotation = {
        "general_figure_info": {
            "x_axis": {"major_labels": {"bboxes": [{"x": 45, "y": 90, "w": 10, "h": 9}]}},
            "y_axis": {"major_ticks": {"bboxes": [{"x": 0, "y": 45, "w": 10, "h": 10}]}},
        }
    }

    row = sample_summary(sample, boxes, annotation)

    assert row["x_axis_tick_text_detected"] is True
    assert row["y_axis_tick_text_detected"] is True
    assert row["question_token_overlap"] is True
    assert row["numeric_token_overlap"] is True
    assert row["direct_answer_value_overlap"] is True
