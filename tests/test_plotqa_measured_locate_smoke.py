import json
from pathlib import Path

from scripts.run_plotqa_measured_locate_smoke import (
    build_plan,
    extract_points,
    operation_posthoc_candidates,
    posthoc_direct_tolerance,
    rescore_existing_result,
    selected_record_points_for_question,
)


def test_extract_points_flags_forbidden_value_fields() -> None:
    text = '{"located_points":[{"role":"target","series_label":"s","pixel_x":10,"pixel_y":20,"chart_value":5}]}'

    points, strict_json = extract_points(text)

    assert strict_json is True
    assert points[0]["pixel_x"] == 10.0
    assert points[0]["value_leak"] is True


def test_stage2_plan_freezes_eight_attempt_cap(tmp_path: Path) -> None:
    diagnostic_plan = tmp_path / "plan.json"
    measured_records = tmp_path / "records.json"
    samples = [
        {
            "sample_id": f"s{i}",
            "image_path": str(tmp_path / f"{i}.png"),
            "question": "What is the value?",
            "answer_type": "numeric",
            "normalized_answer": 1,
        }
        for i in range(8)
    ]
    diagnostic_plan.write_text(
        json.dumps(
            {
                "selected_samples": samples,
                "frozen_before_paid_run": {
                    "reasoning_model": "gpt-5.5",
                    "model_provenance": {"display_name": "GPT-5.5"},
                },
            }
        ),
        encoding="utf-8",
    )
    measured_records.write_text(
        json.dumps([{"sample_id": sample["sample_id"], "ruler": {}, "measured_points": []} for sample in samples]),
        encoding="utf-8",
    )

    plan = build_plan(diagnostic_plan, measured_records, tmp_path)

    assert plan["paid_run_approved"] is True
    assert plan["llm_http_attempt_cap"] == 8
    assert plan["ocr_http_attempt_cap"] == 0
    assert plan["dry_run"]["paid_calls_executed"] == 0


def test_rescore_reuses_locate_pixels_with_new_measured_records(tmp_path: Path) -> None:
    sample = {
        "sample_id": "s0",
        "image_path": str(tmp_path / "0.png"),
        "question": "What is the value?",
        "answer_type": "numeric",
        "normalized_answer": 10.0,
    }
    measured_records = tmp_path / "records.json"
    measured_records.write_text(
        json.dumps(
            [
                {
                    "sample_id": "s0",
                    "ruler": {
                        "x_model": {"type": "none", "params": []},
                        "y_model": {"type": "linear", "params": [-1.0, 20.0], "residual_px_median": 0.0},
                    },
                    "measured_points": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    plan = {
        "stage": "stage2_mllm_pixel_locate_smoke",
        "plan_sha256": "abc",
        "condition": "M0_measured_facts_only_locate",
        "llm_http_attempt_cap": 8,
        "interpretation_limits": [],
        "selected_samples": [sample],
    }
    existing = {
        "actual_llm_http_attempts": 1,
        "results": [
            {
                "sample_id": "s0",
                "condition": "M0_measured_facts_only_locate",
                "completed": True,
                "strict_json": True,
                "located_points": [{"role": "target", "series_label": "", "pixel_x": 5.0, "pixel_y": 10.0, "confidence": "high"}],
                "measured_points": [{"y_value": 999.0}],
                "returned_value_leak": False,
                "raw_text": "{}",
                "llm_http_attempts": 1,
                "ocr_http_attempts": 0,
                "posthoc_direct_tolerance": {"eligible": True, "within_tolerance": False},
            }
        ],
    }

    result = rescore_existing_result(plan, tmp_path / "frozen_plan.json", existing, measured_records)

    assert result["summary"]["posthoc_direct_within_tolerance"] == 1
    assert result["results"][0]["measured_points"][0]["y_value"] == 10.0
    assert result["results"][0]["rescore_source"]["new_paid_calls_executed"] == 0


def test_operation_posthoc_candidates_include_derived_values() -> None:
    assert operation_posthoc_candidates("What is the ratio of A to B?", [80.0, 40.0], [])[:1] == [2.0]
    candidates = operation_posthoc_candidates("What is the difference between the highest and the second highest value?", [7.0, 5.0, 1.0], [])
    assert 2.0 in candidates
    pairwise = operation_posthoc_candidates("What is the difference between A in 2008 and A in 2009?", [], [372194.0, 357027.0, 375834.0])
    assert 15167.0 in pairwise


def test_posthoc_categorical_tolerance_scores_series_label_after_freeze() -> None:
    sample = {"answer_type": "text", "normalized_answer": "Indonesia"}
    measured = [{"series_label": "indonesia", "role": "minimum"}]

    result = posthoc_direct_tolerance(sample, measured)

    assert result["eligible"] is True
    assert result["within_tolerance"] is True


def test_posthoc_categorical_tolerance_scores_signed_difference_comparison() -> None:
    sample = {
        "answer_type": "text",
        "normalized_answer": "yes",
        "question": "Is the difference between A in 2006 and 2009 greater than the difference between B in 2006 and 2009?",
    }
    measured = [
        {"series_label": "A", "x_value": 50.0, "y_value": 2006.1},
        {"series_label": "A", "x_value": 45.0, "y_value": 2008.8},
        {"series_label": "B", "x_value": 22.0, "y_value": 2006.0},
        {"series_label": "B", "x_value": 28.0, "y_value": 2008.9},
    ]

    result = posthoc_direct_tolerance(sample, measured)

    assert {"value": "yes", "correct": True} in result["candidate_scores"]
    assert result["within_tolerance"] is True


def test_selected_record_points_prefers_series_named_in_question() -> None:
    record = {
        "measured_points": [
            {"series_label": "Norway", "x_value": 100.0, "source": "cv_color_component"},
            {"category_label": "Uzbekistan", "x_value": 5.0, "source": "cv_color_component"},
        ]
    }
    sample = {"question": "What is the highest value in Uzbekistan?"}

    selected = selected_record_points_for_question(record, sample)

    assert len(selected) == 1
    assert selected[0]["category_label"] == "Uzbekistan"


def test_selected_record_points_prefers_joint_category_and_series_matches() -> None:
    record = {
        "measured_points": [
            {"category_label": "Bulgaria", "series_label": "1991", "x_value": 47.0, "source": "cv_color_component"},
            {"category_label": "Bulgaria", "series_label": "1992", "x_value": 49.0, "source": "cv_color_component"},
            {"category_label": "Brazil", "series_label": "1991", "x_value": 50.0, "source": "cv_color_component"},
        ]
    }
    sample = {"question": "What is the difference between the value in 1991 and 1992 in Bulgaria?"}

    selected = selected_record_points_for_question(record, sample)

    assert [point["category_label"] for point in selected] == ["Bulgaria", "Bulgaria"]


def test_selected_record_points_tolerates_short_ocr_label_confusion() -> None:
    record = {
        "measured_points": [
            {"category_label": "Finland", "series_label": "2004", "x_value": 3.0, "source": "cv_color_component"},
            {"category_label": "Ethiopia", "series_label": "2006", "x_value": 9.0, "source": "cv_color_component"},
        ]
    }
    sample = {"question": "What is the difference between the value in 2004 in Fiji and 2006 in Ethiopia?"}

    selected = selected_record_points_for_question(record, sample)

    assert {point["category_label"] for point in selected} == {"Finland", "Ethiopia"}
