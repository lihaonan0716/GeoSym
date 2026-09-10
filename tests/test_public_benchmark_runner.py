import json
from pathlib import Path

import pytest

from scripts.run_public_benchmark_pilot import (
    assign_attribution,
    extract_prediction,
    first_json_object,
    load_plan,
    normalize_fact_list,
    normalize_trace_record,
    rescore_existing,
    run_paid,
    score_prediction,
    validate_trace_record,
)


def write_plan(tmp_path: Path, conditions: list[str] | None = None, sample_count: int = 8) -> Path:
    conditions = conditions or ["E0_public_direct", "E1_public_cot", "G0_public_evidence_trace"]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}", encoding="utf-8")
    samples = []
    for idx in range(sample_count):
        image = tmp_path / f"chart_{idx}.png"
        image.write_bytes(b"png")
        samples.append(
            {
                "sample_id": f"sample_{idx}",
                "image_path": str(image),
                "question": "What is the highest value?",
                "answer": "10",
                "answer_type": "numeric",
                "normalized_answer": 10.0,
                "metadata": {
                    "reasoning_type": "extrema",
                    "answer_derivation": "direct",
                    "quantitative_reasoning_required": True,
                },
            }
        )
    plan = {
        "schema_version": "chartqa_g0_smoke_plan_v1",
        "benchmark": "ChartQA",
        "manifest_path": str(manifest),
        "plan_sha256": "abc",
        "paid_run_approved": True,
        "frozen_before_paid_run": {
            "conditions": conditions,
            "reasoning_model": "gpt-5.5",
            "sample_condition_job_cap": len(samples) * len(conditions),
            "llm_http_attempt_cap": len(samples) * sum(3 if c == "G0_public_evidence_trace" else 1 for c in conditions),
            "ocr_http_attempt_cap": 0,
            "prompt_hashes": {"p": "h"},
            "interpretation_limits": ["test"],
        },
        "selected_samples": samples,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def write_plotqa_plan(tmp_path: Path, sample_count: int = 50) -> Path:
    samples = []
    conditions = ["E0_plotqa_direct", "E1_plotqa_cot", "G0_plotqa_evidence_trace"]
    for idx in range(sample_count):
        image = tmp_path / f"plotqa_{idx}.png"
        image.write_bytes(b"png")
        samples.append(
            {
                "sample_id": f"plotqa_{idx}",
                "image_path": str(image),
                "question": "What is the value?",
                "answer": "10",
                "answer_type": "numeric",
                "normalized_answer": 10.0,
                "metadata": {"benchmark": "PlotQA", "image_index": str(idx)},
            }
        )
    plan = {
        "schema_version": "plotqa_g0_smoke_plan_v1",
        "benchmark": "PlotQA",
        "plan_sha256": "abc",
        "paid_run_approved": True,
        "frozen_before_paid_run": {
            "conditions": conditions,
            "reasoning_model": "api-model",
            "model_provenance": {
                "display_name": "Model",
                "api_model_id": "api-model",
            },
            "sample_selection": {"sampling_frame": "official_plotqa_full_split_hash_pool"},
            "sample_condition_job_cap": sample_count * len(conditions),
            "llm_http_attempt_cap": sample_count * 5,
            "ocr_http_attempt_cap": 0,
            "prompt_hashes": {"p": "h"},
            "interpretation_limits": ["test"],
        },
        "selected_samples": samples,
    }
    path = tmp_path / "plotqa_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_first_json_object_extracts_fenced_object() -> None:
    assert first_json_object("note\n```json\n{\"a\": 1}\n```") == {"a": 1}


def test_extract_prediction_handles_json_and_final_answer_text() -> None:
    assert extract_prediction('{"raw_text": "because", "normalized_prediction": "No."}', "text") == "no"
    assert extract_prediction("Final answer: 3.50 units", "numeric") == 3.5


def test_phase1_numeric_scorer_uses_frozen_tolerance() -> None:
    score = score_prediction(104.9, 100.0, "numeric")
    assert score["correct"] is True
    assert score["tolerance"] == 5.0
    assert score_prediction(101.0, 0.0, "numeric")["correct"] is False


def test_runner_accepts_frozen_phase1_plan(tmp_path: Path) -> None:
    path = write_plan(tmp_path)
    plan = load_plan(path)
    assert plan["frozen_before_paid_run"]["llm_http_attempt_cap"] == 40


def test_runner_rejects_legacy_e5_condition_for_public_pilot(tmp_path: Path) -> None:
    path = write_plan(tmp_path, ["E5"])
    with pytest.raises(ValueError, match="unsupported Phase 1 conditions"):
        load_plan(path)


def test_runner_rejects_non_8_sample_phase1_plan(tmp_path: Path) -> None:
    path = write_plan(tmp_path, ["E0_public_direct"], sample_count=1)
    with pytest.raises(ValueError, match="exactly 8"):
        load_plan(path)


def test_runner_accepts_plotqa_50_sample_diagnostic_plan(tmp_path: Path) -> None:
    path = write_plotqa_plan(tmp_path, sample_count=50)
    plan = load_plan(path)
    assert plan["frozen_before_paid_run"]["llm_http_attempt_cap"] == 250


def test_paid_runner_rejects_unapproved_source_prefix_plan(tmp_path: Path) -> None:
    path = write_plan(tmp_path)
    plan = load_plan(path)
    plan["paid_run_approved"] = False

    with pytest.raises(ValueError, match="not approved for paid execution"):
        run_paid(plan, path)


def test_attribution_unknown_for_derived_reasoning_without_operand_schema() -> None:
    sample = {
        "metadata": {"reasoning_type": "range", "answer_derivation": "derived"},
        "answer_type": "numeric",
    }
    record = {
        "reference": 10.0,
        "answer_type": "numeric",
        "completed": True,
        "parseable": True,
        "correct": False,
        "raw_observations": [{"fact_id": "raw_001", "text": "10", "numeric_entities": [10.0]}],
        "retained_facts": [{"fact_id": "raw_001", "text": "10", "numeric_entities": [10.0]}],
    }
    label, proxy, rationale = assign_attribution(record, sample)
    assert label == "attribution_unknown"
    assert proxy["prediction_value_parseable"] is True
    assert "operand schema" in rationale


def test_attribution_distinguishes_extraction_curation_reasoning_failures() -> None:
    sample = {
        "metadata": {"reasoning_type": "lookup", "answer_derivation": "direct"},
        "answer_type": "numeric",
    }
    base = {
        "reference": 10.0,
        "answer_type": "numeric",
        "completed": True,
        "parseable": True,
        "correct": False,
        "raw_observations": [],
        "retained_facts": [],
    }
    assert assign_attribution(base, sample)[0] == "extraction_failure"
    with_raw = {**base, "raw_observations": [{"fact_id": "raw_001", "text": "10", "numeric_entities": [10.0]}]}
    assert assign_attribution(with_raw, sample)[0] == "curation_failure"
    with_retained = {**with_raw, "retained_facts": [{"fact_id": "raw_001", "text": "10", "numeric_entities": [10.0]}]}
    assert assign_attribution(with_retained, sample)[0] == "reasoning_failure"


def test_plotqa_attribution_uses_proxy_failure_labels() -> None:
    sample = {
        "metadata": {"benchmark": "PlotQA", "reasoning_type": "lookup", "answer_derivation": "direct"},
        "answer_type": "numeric",
    }
    record = {
        "reference": 10.0,
        "answer_type": "numeric",
        "completed": True,
        "parseable": True,
        "correct": False,
        "raw_observations": [],
        "retained_facts": [],
    }

    assert assign_attribution(record, sample)[0] == "proxy_extraction_failure"


def test_rescore_existing_uses_cached_raw_text_without_paid_calls() -> None:
    result = {
        "results": [
            {
                "completed": True,
                "raw_text": "Final answer: No.",
                "answer_type": "text",
                "reference": "no",
                "parseable": True,
                "correct": False,
            }
        ]
    }
    rescored = rescore_existing(result)
    assert rescored["summary"]["correct_jobs"] == 1
    assert rescored["results"][0]["normalized_prediction"] == "no"


def test_trace_validator_rejects_non_schema_numeric_entities_and_confidence() -> None:
    record = {
        "sample_id": "p0",
        "raw_observations": [
            {
                "fact_id": "raw_001",
                "source_step": "extract",
                "text": "value is 10",
                "numeric_entities": [{"value": 10}],
                "categorical_entities": [],
                "confidence": "0.95",
            }
        ],
        "retained_facts": [],
        "rejected_or_low_confidence_facts": [],
        "answer": {"raw_text": "10", "normalized_prediction": 10.0},
        "attribution_label": "correct",
    }

    violations = validate_trace_record(record)

    assert "raw_observations[0]:numeric_entities_not_number_array" in violations
    assert "raw_observations[0]:confidence_not_ordinal" in violations


def test_fact_normalization_outputs_trace_schema_values() -> None:
    facts = normalize_fact_list(
        [
            {
                "fact_id": "raw_001",
                "source_step": "extract",
                "text": "value is 10",
                "numeric_entities": ["10"],
                "categorical_entities": ["A"],
                "confidence": "0.95",
            }
        ],
        "extract",
    )

    assert facts == [
        {
            "fact_id": "raw_001",
            "source_step": "extract",
            "text": "value is 10",
            "numeric_entities": [10.0],
            "categorical_entities": ["A"],
            "confidence": "high",
        }
    ]


def test_fact_normalization_flattens_categorical_entity_objects() -> None:
    facts = normalize_fact_list(
        [
            {
                "fact_id": "raw_001",
                "source_step": "extract",
                "text": "Ukraine males",
                "numeric_entities": [],
                "categorical_entities": [
                    {"attribute": "country", "value": "Ukraine"},
                    {"attribute": "gender", "value": "males"},
                ],
                "confidence": "high",
            }
        ],
        "extract",
    )

    assert facts[0]["categorical_entities"] == ["Ukraine", "males"]
    assert validate_trace_record(
        {
            "raw_observations": facts,
            "retained_facts": facts,
            "rejected_or_low_confidence_facts": [],
            "answer": {"raw_text": "Ukraine", "normalized_prediction": "ukraine"},
            "attribution_label": "correct",
        }
    ) == []


def test_normalize_trace_record_repairs_cached_trace_shape() -> None:
    record = {
        "raw_observations": [
            {
                "fact_id": "raw_001",
                "source_step": "extract",
                "text": "value is 10 for Ukraine",
                "numeric_entities": ["10"],
                "categorical_entities": [{"attribute": "country", "value": "Ukraine"}],
                "confidence": "0.95",
            }
        ],
        "retained_facts": [
            {
                "fact_id": "raw_001",
                "source_step": "curate",
                "text": "value is 10 for Ukraine",
                "numeric_entities": ["10"],
                "categorical_entities": [{"attribute": "country", "value": "Ukraine"}],
                "confidence": "0.95",
            }
        ],
        "rejected_or_low_confidence_facts": [],
        "answer": {"raw_text": "10", "normalized_prediction": 10.0},
        "attribution_label": "correct",
    }

    repaired = normalize_trace_record(record)

    assert repaired["raw_observations"][0]["numeric_entities"] == [10.0]
    assert repaired["raw_observations"][0]["categorical_entities"] == ["Ukraine"]
    assert repaired["raw_observations"][0]["confidence"] == "high"
    assert validate_trace_record(repaired) == []
