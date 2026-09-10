from scripts.export_chartqa_manifest import _answer_type, _first_scalar, _is_full_split_export, _local_sample_id, _metadata_value


def test_prefix_limit_metadata_marks_non_full_split() -> None:
    assert _is_full_split_export(limit=0, stopped_by_limit=False) is True
    assert _is_full_split_export(limit=200, stopped_by_limit=True) is False


def test_answer_type_detection_is_zero_cost_and_general() -> None:
    assert _answer_type("1,234.5") == "numeric"
    assert _answer_type(["1,234.5"]) == "numeric"
    assert _answer_type("No") == "text"
    assert _first_scalar([["No"]]) == "No"


def test_metadata_value_fails_closed_to_unknown() -> None:
    assert _metadata_value({}, ("chart_type",)) == "unknown"
    assert _metadata_value({"chart_type": "Simple Bar"}, ("chart_type",)) == "simple_bar"


def test_local_sample_id_is_stable() -> None:
    assert _local_sample_id(3, "q", "i") == _local_sample_id(3, "q", "i")
    assert _local_sample_id(3, "q", "i").startswith("chartqa_test_3_")
