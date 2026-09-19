from scripts.run_plotqa_locate import (
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
                {"text": "120000000", "value": 120000000.0, "pixel": 100.0},
                {"text": "100000000", "value": 100000000.0, "pixel": 200.0},
                {"text": "80000000", "value": 80000000.0, "pixel": 300.0},
                {"text": "60000000", "value": 60000000.0, "pixel": 400.0},
            ]
        },
    }

    prediction = deterministic_prediction_from_measured(sample, record, None)

    assert prediction == 20_000_000.0


def test_axis_tick_spacing_fails_closed_when_ocr_mixes_year_labels_into_the_axis() -> None:
    """OCR routinely merges category labels into the numeric tick list.

    A mixed list yields a meaningless spacing such as 2005, so the solver must
    abstain and let the answering model read the rendered axis instead.
    """
    sample = {
        "answer_type": "numeric",
        "question": "What is the difference between two consecutive major ticks on the Y-axis ?",
    }
    record = {
        "measured_points": [],
        "ruler": {
            "y_ticks": [
                {"text": "1999", "value": 1999.0, "pixel": 96.0},
                {"text": "1998", "value": 1998.0, "pixel": 252.0},
                {"text": "2", "value": 2.0, "pixel": 340.0},
                {"text": "1997", "value": 1997.0, "pixel": 409.0},
            ]
        },
    }

    assert deterministic_prediction_from_measured(sample, record, None) is None


def test_axis_tick_spacing_fails_closed_when_ticks_are_unevenly_spaced() -> None:
    sample = {
        "answer_type": "numeric",
        "question": "What is the difference between two consecutive major ticks on the X-axis ?",
    }
    record = {
        "measured_points": [],
        "ruler": {
            "x_ticks": [
                {"text": "50000000000", "value": 50000000000.0, "pixel": 100.0},
                {"text": "100000000000", "value": 100000000000.0, "pixel": 200.0},
                {"text": "150000000000", "value": 150000000000.0, "pixel": 300.0},
                {"text": "300000000000", "value": 300000000000.0, "pixel": 400.0},
            ]
        },
    }

    assert deterministic_prediction_from_measured(sample, record, None) is None


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


def test_sliver_artifacts_are_rejected_by_the_m0_prime_chain() -> None:
    """Artifact rejection is an M0' property, not a curation privilege.

    It used to run only for the curated arm, which meant the arms differed in
    evidence *quality* as well as in selection -- the control arm was handed
    known-bad points. Admission now runs unconditionally; the curated arm it
    was once contrasted against retired with M1'/M2' on 2026-09-19.
    """
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

    evidence = measured_prompt_facts(record, None)
    fact_ids = [fact["fact_id"] for fact in evidence["measured_facts"]]

    # The two 1px-tall components are artifacts and are dropped.
    assert fact_ids == ["m0_color_000"]

    # With no OCR row the axis cannot be fitted, so the chain declares it
    # inadmissible rather than guessing. Declaring the failure is the designed
    # behaviour: a guessed axis produces numbers that look measured but are not.
    # (The M1'/M2' assertions that used to follow -- tier `low`, withheld
    # injection -- retired with those modules on 2026-09-19.)
    assert evidence["ruler_refit"]["admissible"] is False


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
    measured = [{"y_value": 0.1}, {"y_value": 0.25}, {"y_value": 0.7}]

    assert threshold_count_prediction("in how many years is the value greater than 0.2", measured) == 2.0
    assert threshold_count_prediction("in how many years is the value less than 0.2", measured) == 1.0
    assert threshold_count_prediction("in how many years is the value greater than 0.2", measured[:2]) is None


def test_threshold_count_fails_closed_when_a_mark_is_measured_twice() -> None:
    """M0 emits one point per CV component, not per plotted mark.

    A bar split into several components inflates the count, so a duplicated
    measured value must suppress the branch.
    """
    measured = [{"y_value": 0.1}, {"y_value": 0.7}, {"y_value": 0.7}, {"y_value": 0.9}]

    assert threshold_count_prediction("in how many years is the value greater than 0.2", measured) is None


def test_targeted_extrema_prediction_requires_question_label() -> None:
    record = {
        "ruler": {"confidence": "medium"},
        "measured_points": [
            {"series_label": "Sweden", "x_value": 1982.0, "y_value": 0.7},
            {"series_label": "Sweden", "x_value": 1983.0, "y_value": 0.4},
        ],
    }
    measured = record["measured_points"]

    assert targeted_extrema_prediction("across all years what is the minimum percentage in sweden", measured, record) == 0.4
    assert targeted_extrema_prediction("across all countries what is the minimum percentage", measured, record) is None


def test_targeted_extrema_fails_closed_when_question_asks_for_a_category() -> None:
    """"In which year was X maximum" wants the argmax label, not the value.

    Returning a measured magnitude here is a category/value type error.
    """
    record = {
        "ruler": {"confidence": "high"},
        "measured_points": [
            {"series_label": "Iran", "x_value": 1999.0, "y_value": 7.8},
            {"series_label": "Iran", "x_value": 2000.0, "y_value": 3.3},
        ],
    }
    sample = {"answer_type": "numeric", "question": "In which year was the total resource balance in Iran maximum?"}

    assert deterministic_prediction_from_measured(sample, record, None) is None


def test_targeted_extrema_fails_closed_when_label_filter_drops_part_of_the_series() -> None:
    """A min/max over a fragment of the series is not the chart's min/max."""
    record = {
        "ruler": {"confidence": "high"},
        "measured_points": [
            {"series_label": "Romania", "x_value": 2004.0, "y_value": 841.2},
            {"series_label": "Romania", "x_value": 2005.0, "y_value": 692.9},
            {"series_label": "Romania", "x_value": 2006.0, "y_value": 631.1},
            {"series_label": "Poland", "x_value": 2004.0, "y_value": 120.0},
        ],
    }
    measured = [point for point in record["measured_points"] if point["series_label"] == "Romania"]

    assert targeted_extrema_prediction("across all years what is the minimum percentage in romania", measured, record) is None


def test_ratio_question_does_not_fall_through_to_a_subtraction() -> None:
    record = {
        "ruler": {"confidence": "high"},
        "measured_points": [
            {"series_label": "Tonga", "x_value": 2013.0, "y_value": 4.0},
            {"series_label": "Tonga", "x_value": 2014.0, "y_value": 11.3},
        ],
    }
    sample = {
        "answer_type": "numeric",
        "question": "What is the ratio of the growth rate in Tonga in 2013 to that in 2014?",
    }

    assert deterministic_prediction_from_measured(sample, record, None) is None


def test_metric_named_ratio_is_not_treated_as_a_ratio_operation() -> None:
    """"employment to population ratio" is a metric name, not a quotient request."""
    record = {
        "measured_points": [
            {"series_label": "1991", "category_label": "Chad", "x_value": 3.4, "source": "cv_color_component"},
            {"series_label": "1992", "category_label": "Chad", "x_value": 5.1, "source": "cv_color_component"},
        ]
    }
    sample = {
        "answer_type": "numeric",
        "question": "What is the difference between the employment to population ratio in 1991 and employment to population ratio in 1992 in Chad?",
    }

    assert deterministic_prediction_from_measured(sample, record, None) is not None
