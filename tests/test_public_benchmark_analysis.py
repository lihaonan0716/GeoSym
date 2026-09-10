from scripts.analyze_public_benchmark_pilot import build_report, render_markdown


def test_public_benchmark_analysis_separates_accuracy_and_attribution() -> None:
    plan = {
        "benchmark": "ChartQA",
        "split": "test",
        "manifest_sha256": "m",
        "candidate_pool_sha256": "c",
        "plan_sha256": "p",
        "paid_run_approved": True,
        "frozen_before_paid_run": {
            "sample_selection": {
                "selected_ids": [f"s{i}" for i in range(8)],
                "sampling_frame": "full_split_hash_pool",
                "full_split_export": True,
            }
        },
        "selected_samples": [
            {
                "sample_id": f"s{i}",
                "question": "What is the highest value?",
                "metadata": {
                    "chart_type": "line",
                    "axis_visible": True,
                    "image_readable": True,
                    "geosym_compatibility": "presenter_candidate",
                    "reasoning_type": "extrema",
                    "quantitative_reasoning_required": True,
                },
            }
            for i in range(8)
        ],
    }
    rows = []
    for idx in range(8):
        sample_id = f"s{idx}"
        direct_correct = idx != 0
        rows.extend(
            [
                {
                    "condition": "E0_public_direct",
                    "sample_id": sample_id,
                    "completed": True,
                    "parseable": True,
                    "correct": direct_correct,
                },
                {
                    "condition": "E1_public_cot",
                    "sample_id": sample_id,
                    "completed": True,
                    "parseable": True,
                    "correct": direct_correct,
                },
                {
                    "condition": "G0_public_evidence_trace",
                    "sample_id": sample_id,
                    "completed": True,
                    "parseable": True,
                    "correct": True,
                    "raw_observations": [],
                    "retained_facts": [],
                    "answer": {"raw_text": "10", "normalized_prediction": 10.0},
                    "attribution_label": "correct",
                },
            ]
        )
    result = {
        "conditions": ["E0_public_direct", "E1_public_cot", "G0_public_evidence_trace"],
        "reasoning_model": "gpt-5.5",
        "prompt_hashes": {},
        "llm_http_attempt_cap": 40,
        "actual_llm_http_attempts": 40,
        "ocr_http_attempt_cap": 0,
        "actual_ocr_http_attempts": 0,
        "interpretation_limits": ["smoke only"],
        "results": rows,
    }

    report = build_report(plan, result)

    assert report["condition_summary"]["G0_public_evidence_trace"]["correct"] == 8
    assert report["delta_proxy"]["rescues"] == 1
    assert report["delta_proxy"]["degradations"] == 0
    assert report["g0_trace"]["attribution_counts"] == {"correct": 8}
    assert report["selection_validity"]["valid"] is True
    assert report["experiment_status"] == "go"


def test_public_benchmark_analysis_invalidates_pie_or_prefix_selection() -> None:
    plan = {
        "benchmark": "ChartQA",
        "split": "test",
        "paid_run_approved": False,
        "frozen_before_paid_run": {
            "sample_selection": {
                "selected_ids": [f"s{i}" for i in range(8)],
                "sampling_frame": "source_prefix_diagnostic",
                "full_split_export": False,
            }
        },
        "selected_samples": [
            {
                "sample_id": f"s{i}",
                "question": "Identify the percentage in the pie chart.",
                "metadata": {
                    "chart_type": "simple_bar",
                    "axis_visible": True,
                    "image_readable": True,
                    "geosym_compatibility": "presenter_candidate",
                    "reasoning_type": "lookup",
                    "quantitative_reasoning_required": True,
                },
            }
            for i in range(8)
        ],
    }
    result = {
        "conditions": ["G0_public_evidence_trace"],
        "reasoning_model": "gpt-5.5",
        "llm_http_attempt_cap": 40,
        "actual_llm_http_attempts": 24,
        "ocr_http_attempt_cap": 0,
        "actual_ocr_http_attempts": 0,
        "interpretation_limits": [],
        "results": [],
    }

    report = build_report(plan, result)

    assert report["selection_validity"]["valid"] is False
    assert "paid_run_not_approved" in report["selection_validity"]["violations"]
    assert "not_full_split_hash_pool" in report["selection_validity"]["violations"]
    assert any("question_cues_excluded_chart_type:pie" in item for item in report["selection_validity"]["violations"])
    assert report["experiment_status"] == "invalidated_selector_diagnostic"


def test_public_benchmark_analysis_accepts_valid_plotqa_gate() -> None:
    plan = {
        "benchmark": "PlotQA",
        "split": "test",
        "plan_sha256": "p",
        "paid_run_approved": True,
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
            "sample_selection": {
                "selected_ids": [f"p{i}" for i in range(8)],
                "sampling_frame": "official_plotqa_full_split_hash_pool",
            },
        },
        "selected_samples": [
            {
                "sample_id": f"p{i}",
                "question": "What is the value?",
                "metadata": {
                    "benchmark": "PlotQA",
                    "image_index": str(i),
                    "chart_type": "line",
                    "axis_visible": True,
                    "image_readable": True,
                    "has_x_axis": True,
                    "has_y_axis": True,
                    "has_tick_labels": True,
                    "has_data_table": True,
                    "legend_required": False,
                    "legend_available": False,
                    "qa_template": "lookup",
                    "qa_type": "line",
                    "geosym_compatibility": "presenter_candidate",
                    "reasoning_type": "lookup",
                    "answer_derivation": "direct",
                    "quantitative_reasoning_required": True,
                },
            }
            for i in range(8)
        ],
    }
    rows = []
    for idx in range(8):
        for condition in ("E0_plotqa_direct", "E1_plotqa_cot", "G0_plotqa_evidence_trace"):
            rows.append(
                {
                    "condition": condition,
                    "sample_id": f"p{idx}",
                    "completed": True,
                    "parseable": True,
                    "correct": True,
                    "raw_observations": [],
                    "retained_facts": [],
                    "answer": {"raw_text": "1", "normalized_prediction": 1.0},
                    "attribution_label": "correct",
                }
            )
    result = {
        "conditions": ["E0_plotqa_direct", "E1_plotqa_cot", "G0_plotqa_evidence_trace"],
        "reasoning_model": "gpt-5.5",
        "prompt_hashes": {},
        "llm_http_attempt_cap": 40,
        "actual_llm_http_attempts": 40,
        "ocr_http_attempt_cap": 0,
        "actual_ocr_http_attempts": 0,
        "interpretation_limits": [],
        "results": rows,
    }

    report = build_report(plan, result)

    assert report["schema_version"] == "plotqa_g0_smoke_analysis_v1"
    assert report["official_files"] == {
        "annotations_sha256": "a",
        "qa_pairs_sha256": "q",
        "image_manifest_sha256": "i",
    }
    assert report["experiment_status"] == "plotqa_smoke_go"
    assert report["go_gate"]["selection_validity_gate"] is True

    markdown = render_markdown(report)
    assert "annotations.json SHA256: `a`" in markdown
    assert "qa_pairs.json SHA256: `q`" in markdown
    assert "image manifest SHA256: `i`" in markdown
    assert "Manifest SHA256: `None`" not in markdown


def test_public_benchmark_analysis_downgrades_invalid_plotqa_trace_schema() -> None:
    plan = {
        "benchmark": "PlotQA",
        "split": "test",
        "plan_sha256": "p",
        "paid_run_approved": True,
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
            "sample_selection": {
                "selected_ids": [f"p{i}" for i in range(8)],
                "sampling_frame": "official_plotqa_full_split_hash_pool",
            },
        },
        "selected_samples": [
            {
                "sample_id": f"p{i}",
                "question": "What is the value?",
                "metadata": {
                    "benchmark": "PlotQA",
                    "image_index": str(i),
                    "chart_type": "line",
                    "axis_visible": True,
                    "image_readable": True,
                    "has_x_axis": True,
                    "has_y_axis": True,
                    "has_tick_labels": True,
                    "has_data_table": True,
                    "legend_required": False,
                    "legend_available": False,
                    "qa_template": "lookup",
                    "qa_type": "line",
                    "geosym_compatibility": "presenter_candidate",
                    "reasoning_type": "lookup",
                    "answer_derivation": "direct",
                    "quantitative_reasoning_required": True,
                },
            }
            for i in range(8)
        ],
    }
    rows = []
    for idx in range(8):
        rows.extend(
            [
                {
                    "condition": "E0_plotqa_direct",
                    "sample_id": f"p{idx}",
                    "completed": True,
                    "parseable": True,
                    "correct": True,
                },
                {
                    "condition": "E1_plotqa_cot",
                    "sample_id": f"p{idx}",
                    "completed": True,
                    "parseable": True,
                    "correct": True,
                },
                {
                    "condition": "G0_plotqa_evidence_trace",
                    "sample_id": f"p{idx}",
                    "completed": True,
                    "parseable": True,
                    "correct": True,
                    "raw_observations": [
                        {
                            "fact_id": "raw_001",
                            "source_step": "extract",
                            "text": "value is 1",
                            "numeric_entities": [{"value": 1}],
                            "categorical_entities": [],
                            "confidence": "0.95",
                        }
                    ],
                    "retained_facts": [],
                    "answer": {"raw_text": "1", "normalized_prediction": 1.0},
                    "attribution_label": "correct",
                },
            ]
        )
    result = {
        "conditions": ["E0_plotqa_direct", "E1_plotqa_cot", "G0_plotqa_evidence_trace"],
        "reasoning_model": "gpt-5.5",
        "prompt_hashes": {},
        "llm_http_attempt_cap": 40,
        "actual_llm_http_attempts": 40,
        "ocr_http_attempt_cap": 0,
        "actual_ocr_http_attempts": 0,
        "interpretation_limits": [],
        "results": rows,
    }

    report = build_report(plan, result)

    assert report["g0_trace"]["parseable_trace_records"] == 0
    assert report["go_gate"]["g0_parseable_trace_at_least_7"] is False
    assert report["experiment_status"] == "plotqa_access_or_selector_diagnostic"


def test_public_benchmark_analysis_labels_valid_50_sample_plotqa_no_superiority() -> None:
    plan = {
        "benchmark": "PlotQA",
        "split": "test",
        "plan_sha256": "p",
        "paid_run_approved": True,
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
            "sample_selection": {
                "selected_ids": [f"p{i}" for i in range(50)],
                "sampling_frame": "official_plotqa_full_split_hash_pool",
            },
        },
        "selected_samples": [
            {
                "sample_id": f"p{i}",
                "question": "What is the value?",
                "metadata": {
                    "benchmark": "PlotQA",
                    "image_index": str(i),
                    "chart_type": "line",
                    "axis_visible": True,
                    "image_readable": True,
                    "has_x_axis": True,
                    "has_y_axis": True,
                    "has_tick_labels": True,
                    "has_data_table": True,
                    "legend_required": False,
                    "legend_available": False,
                    "qa_template": "lookup",
                    "qa_type": "line",
                    "geosym_compatibility": "presenter_candidate",
                    "reasoning_type": "lookup",
                    "answer_derivation": "direct",
                    "quantitative_reasoning_required": True,
                },
            }
            for i in range(50)
        ],
    }
    rows = []
    for idx in range(50):
        rows.extend(
            [
                {
                    "condition": "E0_plotqa_direct",
                    "sample_id": f"p{idx}",
                    "completed": True,
                    "parseable": True,
                    "correct": idx < 43,
                },
                {
                    "condition": "E1_plotqa_cot",
                    "sample_id": f"p{idx}",
                    "completed": True,
                    "parseable": True,
                    "correct": idx < 42,
                },
                {
                    "condition": "G0_plotqa_evidence_trace",
                    "sample_id": f"p{idx}",
                    "completed": True,
                    "parseable": True,
                    "correct": idx < 41,
                    "raw_observations": [],
                    "retained_facts": [],
                    "answer": {"raw_text": "1", "normalized_prediction": 1.0},
                    "attribution_label": "correct" if idx < 41 else "proxy_extraction_failure",
                },
            ]
        )
    result = {
        "conditions": ["E0_plotqa_direct", "E1_plotqa_cot", "G0_plotqa_evidence_trace"],
        "reasoning_model": "gpt-5.5",
        "prompt_hashes": {},
        "llm_http_attempt_cap": 250,
        "actual_llm_http_attempts": 250,
        "ocr_http_attempt_cap": 0,
        "actual_ocr_http_attempts": 0,
        "interpretation_limits": [],
        "results": rows,
    }

    report = build_report(plan, result)

    assert report["selection_validity"]["valid"] is True
    assert report["experiment_status"] == "plotqa_diagnostic_no_superiority"
    assert "valid access and selection" in render_markdown(report)
