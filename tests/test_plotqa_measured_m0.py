from pathlib import Path

import cv2
import numpy as np

from scripts.run_plotqa_measured_m0 import (
    attach_category_labels,
    build_stage1,
    bottleneck_labels,
    extract_legend_entries,
    horizontal_bar_points_from_labels,
    infer_grid_top_value,
    line_points_from_colored_path,
    measured_record,
    parse_number,
    ruler_confidence,
    validate_m0_record,
)


def test_parse_number_supports_common_tick_formats() -> None:
    assert parse_number("1,200") == 1200.0
    assert parse_number("10OOOO") == 100000.0
    assert parse_number("socoooooooo") == 50_000_000_000.0
    assert parse_number("150000ooooo0") == 150_000_000_000.0
    assert parse_number("2ocdoooooooo") == 200_000_000_000.0
    assert parse_number("45%") == 45.0
    assert parse_number("$2.5M") == 2_500_000.0
    assert parse_number("-1.2e3") == -1200.0
    assert parse_number("Ukraine") is None


def test_validate_m0_record_accepts_minimal_gap_record() -> None:
    record = {
        "chart_geometry": {"chart_type": "unknown"},
        "ruler": {
            "confidence": "failed",
            "x_ticks": [],
            "y_ticks": [],
            "x_model": {"type": "none"},
            "y_model": {"type": "none"},
        },
        "measured_points": [],
        "evidence_gaps": [{"gap_id": "gap_000", "needed_for": "lookup", "reason": "missing_axis", "severity": "blocking"}],
    }

    assert validate_m0_record(record) == []


def test_measured_record_emits_schema_valid_points_or_gaps(tmp_path: Path) -> None:
    image_path = tmp_path / "bar.png"
    image = np.ones((120, 160, 3), dtype=np.uint8) * 255
    cv2.rectangle(image, (55, 40), (75, 100), (0, 0, 180), -1)
    cv2.imwrite(str(image_path), image)
    sample = {
        "sample_id": "s0",
        "image_path": str(image_path),
        "question": "What is the value?",
        "answer_type": "numeric",
        "normalized_answer": 10.0,
        "metadata": {"chart_type": "bar", "reasoning_type": "lookup"},
    }
    ocr = {
        "sample_id": "s0",
        "image_size": [160, 120],
        "ocr_boxes": [
            {"box_id": "ocr_000", "text": "0", "bbox": [4, 96, 20, 106], "confidence": 0.9},
            {"box_id": "ocr_001", "text": "10", "bbox": [4, 36, 24, 46], "confidence": 0.9},
            {"box_id": "ocr_002", "text": "2010", "bbox": [50, 108, 78, 118], "confidence": 0.9},
            {"box_id": "ocr_003", "text": "2011", "bbox": [90, 108, 118, 118], "confidence": 0.9},
        ],
    }

    record = measured_record(sample, ocr)

    assert record["schema_violations"] == []
    assert record["ruler"]["y_model"]["type"] == "linear"
    assert record["measured_points"] or record["evidence_gaps"]


def test_stage1_gate_counts_schema_valid_records() -> None:
    records = [
        {
            "schema_violations": [],
            "ruler": {"confidence": "medium"},
            "measured_points": [{"fact_id": "m0_000"}],
        }
        for _ in range(8)
    ]

    stage = build_stage1(records)

    assert stage["stage_status"] == "stage_passed"
    assert stage["go_gate"]["schema_valid_8_of_8"] is True


def test_noisy_ruler_is_low_confidence_and_labeled_bad() -> None:
    x_model = {"type": "linear", "params": [1.0, 0.0], "tick_count": 3, "residual_px_median": 80.0}
    y_model = {"type": "linear", "params": [1.0, 0.0], "tick_count": 3, "residual_px_median": 20.0}
    record = {
        "ruler": {"confidence": ruler_confidence(x_model, y_model), "x_model": x_model, "y_model": y_model},
        "measured_points": [{"fact_id": "m0_000"}],
        "schema_violations": [],
    }

    assert record["ruler"]["confidence"] == "low"
    assert bottleneck_labels(record)[0]["label"] == "m0_bad_ruler"


def test_bottom_year_legend_entries_are_not_filtered_as_axis_ticks(tmp_path: Path) -> None:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    image[86:96, 118:130] = [0, 0, 255]
    image_path = tmp_path / "legend.png"
    cv2.imwrite(str(image_path), image)
    ocr_row = {"ocr_boxes": [{"text": "1991", "bbox": [134, 84, 164, 98], "confidence": 0.99}]}

    entries = extract_legend_entries(image_path, ocr_row)

    assert entries[0]["label"] == "1991"


def test_category_labels_use_left_text_but_ignore_low_confidence_noise() -> None:
    points = [
        {
            "component_bbox": [130, 80, 300, 110],
            "component_center_pixel": [215, 95],
            "support": [],
        }
    ]
    ocr_row = {
        "ocr_boxes": [
            {"text": "Bulgaria", "bbox": [92, 87, 147, 101], "confidence": 0.99},
            {"text": "L", "bbox": [7, 251, 23, 345], "confidence": 0.19},
        ]
    }

    diag = attach_category_labels(points, ocr_row, [34, 40, 1200, 620])

    assert diag["assigned_count"] == 1
    assert points[0]["category_label"] == "Bulgaria"


def test_horizontal_bar_label_scan_recovers_legend_colored_bars(tmp_path: Path) -> None:
    image = np.full((160, 260, 3), 255, dtype=np.uint8)
    image[48:58, 80:180] = [0, 220, 220]
    image[68:78, 80:140] = [0, 120, 0]
    image_path = tmp_path / "hbar.png"
    cv2.imwrite(str(image_path), image)
    ocr_row = {
        "ocr_boxes": [
            {"text": "Italy", "bbox": [20, 55, 55, 69], "confidence": 0.99},
        ]
    }
    legend = [
        {"label": "Cabo Verde", "color_bgr": [0, 220, 220], "confidence": "high"},
        {"label": "Cambodia", "color_bgr": [0, 120, 0], "confidence": "high"},
    ]
    x_model = {"type": "linear", "params": [1.0, -80.0], "residual_px_median": 0.0}

    points, diag = horizontal_bar_points_from_labels(image_path, [70, 20, 230, 130], x_model, ocr_row, legend, [])

    assert diag["status"] == "ok"
    assert {point["series_label"] for point in points} == {"Cabo Verde", "Cambodia"}
    assert {point["category_label"] for point in points} == {"Italy"}
    assert max(point["x_value"] for point in points) == 100.0


def test_horizontal_bar_label_scan_requires_legend_and_linear_x(tmp_path: Path) -> None:
    image_path = tmp_path / "empty.png"
    cv2.imwrite(str(image_path), np.full((40, 40, 3), 255, dtype=np.uint8))

    points, diag = horizontal_bar_points_from_labels(image_path, [0, 0, 40, 40], {"type": "none"}, {}, [], [])

    assert points == []
    assert diag["status"] == "requires_linear_x_model_and_legend"


def test_colored_line_sample_recovers_dashed_line_points(tmp_path: Path) -> None:
    image = np.full((120, 220, 3), 255, dtype=np.uint8)
    for start, end in [((40, 92), (80, 84)), ((80, 84), (120, 60)), ((120, 60), (160, 35))]:
        cv2.line(image, start, end, (180, 180, 0), 2, lineType=cv2.LINE_AA)
    image_path = tmp_path / "line.png"
    cv2.imwrite(str(image_path), image)
    x_model = {
        "type": "linear",
        "params": [1.0, 0.0],
        "used_points": [{"pixel": 20.0, "value": 0.0}, {"pixel": 40.0, "value": 1.0}, {"pixel": 80.0, "value": 2.0}, {"pixel": 120.0, "value": 3.0}, {"pixel": 160.0, "value": 4.0}],
        "residual_px_median": 0.0,
    }
    y_model = {"type": "linear", "params": [-0.1, 10.0], "residual_px_median": 0.0}

    points, diag = line_points_from_colored_path(image_path, [20, 20, 190, 100], x_model, y_model, [])

    assert diag["status"] == "ok"
    assert len(points) >= 3
    assert all(point["source"] == "cv_colored_line_sample" for point in points)
    assert all(point["y_value"] is not None for point in points)


def test_colored_line_sample_does_not_fire_when_marks_exist(tmp_path: Path) -> None:
    image_path = tmp_path / "empty.png"
    cv2.imwrite(str(image_path), np.full((40, 40, 3), 255, dtype=np.uint8))

    points, diag = line_points_from_colored_path(
        image_path,
        [0, 0, 40, 40],
        {"type": "linear", "params": [1.0, 0.0]},
        {"type": "linear", "params": [1.0, 0.0]},
        [{"fact_id": "existing"}],
    )

    assert points == []
    assert diag["status"] == "existing_marks_present"


def test_percent_axis_interval_fallback_requires_percent_text() -> None:
    rows = [40.0, 80.0, 120.0, 160.0, 200.0]
    ocr_row = {"ocr_boxes": [{"text": "Earning(%)"}]}

    assert infer_grid_top_value(rows, [], 240, 240, ocr_row) == (20.0, "percent_axis_tick_interval_fallback")
    assert infer_grid_top_value(rows, [], 240, 240, {"ocr_boxes": [{"text": "Earnings"}]}) == (None, "no_left_numeric_ticks")
