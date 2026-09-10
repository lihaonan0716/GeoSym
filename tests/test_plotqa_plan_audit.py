from scripts.audit_plotqa_public_benchmark_plan import audit_plan
from scripts.prepare_plotqa_public_benchmark_pilot import DEFAULT_CONDITIONS


def valid_plan(sample_count: int = 8):
    return {
        "benchmark": "PlotQA",
        "paid_run_approved": True,
        "paid_calls_executed": 0,
        "frozen_before_paid_run": {
            "official_files": {
                "annotations_sha256": "a",
                "qa_pairs_sha256": "q",
                "image_manifest_sha256": "i",
            },
            "reasoning_model": "api-model",
            "model_provenance": {
                "display_name": "Model",
                "provider": "Provider",
                "adapter": "module.Adapter",
                "api_model_id": "api-model",
                "api_key_name": "key",
                "base_url_key": "base_url",
                "gateway_policy": "base_url_key",
                "config_path": "config/default_config.yaml",
                "config_sha256": "c",
            },
            "sample_selection": {"sampling_frame": "official_plotqa_full_split_hash_pool"},
            "conditions": list(DEFAULT_CONDITIONS),
            "sample_condition_job_cap": sample_count * 3,
            "llm_http_attempt_cap": sample_count * 5,
            "ocr_http_attempt_cap": 0,
        },
        "selected_samples": [
            {
                "sample_id": f"p{i}",
                "image_path": f"{i}.png",
                "question": "What is the value?",
                "answer": "1",
                "answer_type": "numeric",
                "normalized_answer": 1.0,
                "metadata": {
                    "image_index": str(i),
                    "qa_index": str(i),
                    "chart_type": "line",
                    "image_readable": True,
                    "axis_visible": True,
                    "has_x_axis": True,
                    "has_y_axis": True,
                    "has_tick_labels": True,
                    "has_data_table": True,
                    "legend_required": False,
                    "legend_available": False,
                    "reasoning_type": "lookup",
                    "answer_derivation": "direct",
                    "quantitative_reasoning_required": True,
                    "geosym_compatibility": "presenter_candidate",
                },
            }
            for i in range(sample_count)
        ],
    }


def test_plotqa_plan_audit_accepts_valid_plan() -> None:
    report = audit_plan(valid_plan())

    assert report["valid"] is True
    assert report["violations"] == []


def test_plotqa_plan_audit_accepts_valid_50_sample_plan() -> None:
    report = audit_plan(valid_plan(sample_count=50))

    assert report["valid"] is True
    assert report["violations"] == []


def test_plotqa_plan_audit_rejects_missing_hash_and_duplicate_image() -> None:
    plan = valid_plan()
    plan["frozen_before_paid_run"]["official_files"]["qa_pairs_sha256"] = ""
    plan["selected_samples"][1]["metadata"]["image_index"] = "0"

    report = audit_plan(plan)

    assert report["valid"] is False
    assert "missing_qa_pairs_sha256" in report["violations"]
    assert "selected_image_index_not_unique" in report["violations"]


def test_plotqa_plan_audit_rejects_zero_cost_flag_on_paid_plan() -> None:
    plan = valid_plan()
    plan["zero_cost_only"] = True

    report = audit_plan(plan)

    assert report["valid"] is False
    assert "paid_approved_plan_marked_zero_cost_only" in report["violations"]


def test_plotqa_plan_audit_rejects_structural_template_mismatch() -> None:
    plan = valid_plan()
    sample = plan["selected_samples"][0]
    sample["question"] = "How many legend labels are there ?"
    sample["metadata"]["qa_template"] = "structural"
    sample["metadata"]["qa_type"] = "line"
    sample["metadata"]["reasoning_type"] = "lookup"
    sample["metadata"]["answer_derivation"] = "direct"
    sample["metadata"]["quantitative_reasoning_required"] = True

    report = audit_plan(plan)

    assert report["valid"] is False
    assert any("structural_qa_template_excluded" in item for item in report["violations"])
    assert any("reasoning_type_mismatch:lookup!=text" in item for item in report["violations"])


def test_plotqa_plan_audit_rejects_missing_model_provenance_and_bad_exclusions() -> None:
    plan = valid_plan()
    del plan["frozen_before_paid_run"]["model_provenance"]
    plan["excluded_summary"] = {"eligible": 1}

    report = audit_plan(plan)

    assert report["valid"] is False
    assert "missing_model_provenance" in report["violations"]
    assert "excluded_summary_contains_eligible" in report["violations"]
