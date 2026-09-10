import json

from scripts.selective_propagation import parse_selective_answer, score_selective_answer


def response(**kwargs) -> str:
    payload = {
        "answer_status": "answerable",
        "normalized_prediction": "42",
        "supporting_fact_ids": ["f0"],
        "missing_fact_keys": [],
        "contradictory_fact_ids": [],
        "risk_reason": "",
    }
    payload.update(kwargs)
    return json.dumps(payload)


def test_non_answerable_numeric_response_does_not_extract_incidental_years() -> None:
    raw = response(
        answer_status="insufficient_evidence",
        normalized_prediction=None,
        risk_reason="missing 2008 and 2009 values",
    )

    record = parse_selective_answer(raw, "numeric")

    assert record.schema_valid is True
    assert record.answer_status == "insufficient_evidence"
    assert record.prediction_parseable is False
    assert record.normalized_prediction is None


def test_non_answerable_with_prediction_fails_closed() -> None:
    raw = response(answer_status="insufficient_evidence", normalized_prediction="2008")

    record = parse_selective_answer(raw, "numeric")

    assert record.schema_valid is False
    assert record.answer_status == "schema_invalid"
    assert record.normalized_prediction is None


def test_malformed_json_fails_closed() -> None:
    record = parse_selective_answer("Final answer: 2008", "numeric")

    assert record.schema_valid is False
    assert record.answer_status == "schema_invalid"
    assert record.prediction_parseable is False


def test_score_separates_correct_answer_from_correct_abstention() -> None:
    answer = parse_selective_answer(response(normalized_prediction="10"), "numeric")
    abstain = parse_selective_answer(response(answer_status="contradiction", normalized_prediction=None), "numeric")

    answer_score = score_selective_answer(answer, 10, "numeric", "answerable")
    abstain_score = score_selective_answer(abstain, 10, "numeric", "contradiction")

    assert answer_score["correct_answer"] is True
    assert answer_score["correct_abstention"] is False
    assert abstain_score["correct_answer"] is False
    assert abstain_score["correct_abstention"] is True
    assert abstain_score["task_success"] is True
