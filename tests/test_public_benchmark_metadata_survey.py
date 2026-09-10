from argparse import Namespace

from scripts.survey_public_benchmark_metadata import (
    BENCHMARK_SPECS,
    infer_schema_from_row,
    is_geosym_compatible,
    summarize_plotqa_official,
    summarize_rows,
)


def test_plotqa_tag_serialization_has_axes_data_bboxes_but_not_qa() -> None:
    row = {
        "image": object(),
        "text": (
            "<s_y>10<sep/>20</s_y><s_x>2010<sep/>2011</s_x>"
            "<s_name>Series</s_name><s_label>Series</s_label><s_bboxes><s_x>1</s_x><s_w>2</s_w></s_bboxes>"
        ),
    }

    schema = infer_schema_from_row(row)

    assert schema["has_image"] is True
    assert schema["has_x_axis"] is True
    assert schema["has_y_axis"] is True
    assert schema["has_plot_bbox"] is True
    assert schema["has_data_table"] is True
    assert schema["has_question"] is False
    assert is_geosym_compatible(schema, require_qa=True) is False


def test_plotqa_vega_text_schema_has_mark_axes_and_data() -> None:
    row = {
        "image": object(),
        "text": "{'mark': 'line', 'encoding': {'x': {'field': 'Year'}, 'y': {'field': 'Value'}}, 'data': {'values': []}}",
    }

    schema = infer_schema_from_row(row)

    assert schema["chart_type"] == "line"
    assert schema["has_x_axis"] is True
    assert schema["has_y_axis"] is True
    assert schema["has_data_table"] is True


def test_summarize_rows_counts_qa_ready_eligibility() -> None:
    rows = [
        {
            "image": object(),
            "question": "What is the value?",
            "answer": "10",
            "answer_type": "numeric",
            "mark": "line",
            "x_axis": {"label": "x"},
            "y_axis": {"label": "y"},
            "ticks": [1, 2],
            "data": [{"x": 1, "y": 2}],
        }
    ]

    summary = summarize_rows(rows, require_qa=True)

    assert summary["eligible_count"] == 1
    assert summary["coverage"]["has_question"] == 1
    assert summary["coverage"]["has_answer"] == 1


def test_candidate_specs_keep_plotqa_first() -> None:
    assert BENCHMARK_SPECS[0].name == "PlotQA"
    assert BENCHMARK_SPECS[0].official_schema_summary["x_axis"] is True


def test_official_plotqa_json_probe_counts_eligible_rows(tmp_path) -> None:
    annotations = [
        {
            "image_index": 1,
            "type": "line",
            "models": [{"x": [1, 2], "y": [3, 4]}],
            "general_figure_info": {
                "x_axis": {"major_ticks": [{"label": "1"}]},
                "y_axis": {"major_ticks": [{"label": "3"}]},
                "legend": {"bbox": [0, 0, 1, 1]},
                "plot_info": {"bbox": [0, 0, 10, 10]},
                "figure_info": {"bbox": [0, 0, 20, 20]},
            },
        }
    ]
    qa_pairs = [
        {
            "image_index": 1,
            "question_string": "What is the value?",
            "answer": "3",
            "type": "line",
        },
        {
            "image_index": 1,
            "question_string": "What is the second value?",
            "answer": "4",
            "type": "line",
        },
    ]
    annotations_path = tmp_path / "annotations.json"
    qa_path = tmp_path / "qa_pairs.json"
    annotations_path.write_text(__import__("json").dumps(annotations), encoding="utf-8")
    qa_path.write_text(__import__("json").dumps(qa_pairs), encoding="utf-8")

    summary = summarize_plotqa_official(annotations_path, qa_path, None, limit=8, require_qa=True)

    assert summary["eligible_count"] == 2
    assert summary["coverage"]["has_question"] == 2
    assert summary["chart_type_counts"] == {"line": 2}
