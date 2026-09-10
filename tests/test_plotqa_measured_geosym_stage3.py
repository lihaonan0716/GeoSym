from scripts.run_plotqa_measured_geosym_stage3 import (
    corrected_numeric_prediction_from_response,
    deterministic_prediction_from_measured,
    measured_prompt_facts,
    targeted_extrema_prediction,
    threshold_count_prediction,
)


def test_deterministic_prediction_uses_signed_grouped_bar_difference() -> None:
    sample = {
        "answer_type": "numeric",
        "question": "What is the difference between the value in 2004 in Fiji and the value in 2006 in Ethiopia?",
    }
    record = {
        "measured_points": [
            {"category_label": "Finland", "series_label": "2004", "x_value": 3.4, "source": "cv_color_component"},
            {"category_label": "Ethiopia", "series_label": "2006", "x_value": 9.0, "source": "cv_color_component"},
            {"category_label": "Ethiopia", "series_label": "2004", "x_value": 9.0, "source": "cv_color_component"},
        ]
    }

    prediction = deterministic_prediction_from_measured(sample, record, None)

    assert prediction == -5.6


def test_deterministic_prediction_answers_axis_tick_spacing() -> None:
    sample = {
        "answer_type": "numeric",
        "question": "What is the difference between two consecutive major ticks on the Y-axis ?",
    }
    record = {
        "measured_points": [],
        "ruler": {
            "y_ticks": [
                {"text": "120000000", "value": 120000000.0},
                {"text": "100000000", "value": 100000000.0},
                {"text": "80000000", "value": 80000000.0},
            ]
        },
    }

    prediction = deterministic_prediction_from_measured(sample, record, None)

    assert prediction == 20_000_000.0


def test_deterministic_prediction_answers_axis_e_notation() -> None:
    sample = {
        "answer_type": "text",
        "question": "Are the values on the major ticks of X-axis written in scientific E-notation ?",
    }
    record = {"measured_points": [], "ruler": {"x_ticks": [{"text": "40000000", "value": 40000000.0}]}}

    prediction = deterministic_prediction_from_measured(sample, record, None)

    assert prediction == "no"


def test_deterministic_extrema_ignores_locate_smoke_points() -> None:
    sample = {
        "answer_type": "numeric",
        "question": "What is the difference between the highest and the lowest exports in Uzbekistan?",
    }
    record = {
        "chart_geometry": {"chart_type": "bar"},
        "measured_points": [
            {"series_label": "Uzbekistan", "x_value": 11.8, "source": "cv_color_component"},
            {"series_label": "Uzbekistan", "x_value": 3.5, "source": "cv_color_component"},
        ],
    }
    locate_row = {
        "measured_points": [
            {"series_label": "Uzbekistan", "x_value": 1.8, "source": "mllm_locate"},
        ]
    }

    prediction = deterministic_prediction_from_measured(sample, record, locate_row)

    assert round(prediction, 1) == 8.3


def test_deterministic_extrema_fails_closed_on_low_confidence_ruler() -> None:
    sample = {
        "answer_type": "numeric",
        "question": "What is the difference between the highest and the lowest exports in Uzbekistan?",
    }
    record = {
        "chart_geometry": {"chart_type": "bar"},
        "ruler": {"confidence": "low"},
        "measured_points": [
            {"series_label": "Uzbekistan", "x_value": 11.8, "source": "cv_color_component"},
            {"series_label": "Uzbekistan", "x_value": 3.5, "source": "cv_color_component"},
        ],
    }

    prediction = deterministic_prediction_from_measured(sample, record, None)

    assert prediction is None


def test_deterministic_year_series_difference_fails_closed_for_vertical_grouped_offsets() -> None:
    sample = {
        "answer_type": "numeric",
        "question": "What is the difference between emissions in Zambia in 2007 and European Union in 2005?",
    }
    record = {
        "ruler": {"x_ticks": [{"value": 2005.0}, {"value": 2006.0}, {"value": 2007.0}]},
        "measured_points": [
            {"series_label": "Zambia", "x_value": 2007.2, "y_value": 7.2, "source": "cv_color_component"},
            {"series_label": "European Union", "x_value": 2004.8, "y_value": 3.2, "source": "cv_color_component"},
        ],
    }

    prediction = deterministic_prediction_from_measured(sample, record, None)

    assert prediction is None


def test_deterministic_year_series_difference_keeps_same_sign_grouped_offsets_fail_closed() -> None:
    sample = {
        "answer_type": "numeric",
        "question": "What is the difference between emissions in Zambia in 2007 and European Union in 2005?",
    }
    record = {
        "ruler": {"x_ticks": [{"value": 2005.0}, {"value": 2006.0}, {"value": 2007.0}]},
        "measured_points": [
            {"series_label": "Zambia", "x_value": 2007.2, "y_value": 7.2, "source": "cv_color_component"},
            {"series_label": "European Union", "x_value": 2005.2, "y_value": 4.4, "source": "cv_color_component"},
        ],
    }

    prediction = deterministic_prediction_from_measured(sample, record, None)

    assert prediction is None


def test_curated_bar_facts_filter_sliver_artifacts_but_raw_keeps_them() -> None:
    record = {
        "chart_geometry": {"chart_type": "bar"},
        "measured_points": [
            {
                "fact_id": "m0_color_000",
                "series_label": "Benin",
                "x_value": 1.2,
                "y_value": 0.26,
                "source": "cv_color_component",
                "confidence": "medium",
                "component_bbox": [133.0, 511.0, 227.0, 526.0],
                "component_area": 1410,
            },
            {
                "fact_id": "m0_color_001",
                "series_label": "Benin",
                "x_value": 1.2,
                "y_value": 0.016,
                "source": "cv_color_component",
                "confidence": "medium",
                "component_bbox": [133.0, 526.0, 227.0, 527.0],
                "component_area": 94,
            },
            {
                "fact_id": "m0_color_002",
                "series_label": None,
                "x_value": 2.0,
                "y_value": 0.016,
                "source": "cv_color_component",
                "confidence": "medium",
                "component_bbox": [229.0, 526.0, 322.0, 527.0],
                "component_area": 93,
            },
        ],
    }

    raw_ids = [fact["fact_id"] for fact in measured_prompt_facts(record, None, curated=False)["measured_facts"]]
    curated_ids = [fact["fact_id"] for fact in measured_prompt_facts(record, None, curated=True)["measured_facts"]]

    assert raw_ids == ["m0_color_000", "m0_color_001", "m0_color_002"]
    assert curated_ids == ["m0_color_000"]


def test_response_normalizer_restores_signed_difference_from_label_year_operands() -> None:
    sample = {
        "answer_type": "numeric",
        "question": "What is the difference between the net bilateral aid flow in Chile in 2003 and the net bilateral aid flow in Rwanda in 2001?",
    }
    record = {
        "measured_points": [
            {"series_label": "Chile"},
            {"series_label": "Rwanda"},
        ]
    }
    raw_text = (
        '{"raw_text":"Rwanda in 2001 is about 19.1 million and Chile in 2003 is about 1.2 million, '
        'so the difference is about 17.9 million.","normalized_prediction":"17900000"}'
    )

    prediction = corrected_numeric_prediction_from_response(sample, record, raw_text, 17_900_000.0)

    assert prediction == -17_900_000.0


def test_response_normalizer_keeps_positive_difference_when_directed_operands_positive() -> None:
    sample = {
        "answer_type": "numeric",
        "question": "What is the difference between emissions in Zambia in 2007 and European Union in 2005?",
    }
    record = {
        "measured_points": [
            {"series_label": "Zambia"},
            {"series_label": "European Union"},
        ]
    }
    raw_text = '{"raw_text":"Zambia 2007 is 7.19 and European Union 2005 is 4.44; difference is 2.75.","normalized_prediction":"2.75"}'

    prediction = corrected_numeric_prediction_from_response(sample, record, raw_text, 2.75)

    assert prediction == 2.75


def test_threshold_count_prediction_uses_measured_y_values_only() -> None:
    measured = [{"y_value": 0.1}, {"y_value": 0.25}, {"y_value": 0.7}, {"y_value": None}]

    assert threshold_count_prediction("in how many years is the value greater than 0.2", measured) == 2.0
    assert threshold_count_prediction("in how many years is the value less than 0.2", measured) == 1.0
    assert threshold_count_prediction("in how many years is the value greater than 0.2", measured[:2]) is None


def test_targeted_extrema_prediction_requires_question_label() -> None:
    record = {"ruler": {"confidence": "medium"}}
    measured = [
        {"series_label": "Sweden", "x_value": 1982.0, "y_value": 0.7},
        {"series_label": "Sweden", "x_value": 1983.0, "y_value": 0.4},
    ]

    assert targeted_extrema_prediction("across all years what is the minimum percentage in sweden", measured, record) == 0.4
    assert targeted_extrema_prediction("across all countries what is the minimum percentage", measured, record) is None
