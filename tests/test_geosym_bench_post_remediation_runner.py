import json
from pathlib import Path

from scripts.run_geosym_bench_post_remediation_subset import (
    AttemptCounter,
    M0_EXECUTION_MODE_FALLBACK_FIRST,
    OcrAttemptCounter,
    build_dry_run,
    checkpoint_requires_reauthorization,
    condition_row,
    count_mathpix_http_attempts,
    count_llm_http_attempts,
    parse_args,
    sample_image_name,
)
from scripts.prepare_public_benchmark_pilot import canonical_json_bytes, sha256_hex_bytes
from evaluator.adapters.base import BaseLLMAdapter


def test_sample_image_name_maps_subset_sample_to_dataset_path() -> None:
    sample = {"sample_id": "absolute-clean_render_0000", "family": "absolute"}

    assert sample_image_name(sample) == "absolute-clean_render/absolute-clean_render_0000"


def test_track_b_dry_run_freezes_caps(tmp_path: Path) -> None:
    audit_path = tmp_path / "metadata_dedup_audit.json"
    samples = [
        {
            "sample_id": f"absolute-clean_render_{i:04d}",
            "family": "absolute",
            "dedup_key": f"k{i}",
        }
        for i in range(3)
    ]
    audit_path.write_text(json.dumps({"selected_subset": samples}), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "stage_status": "geosym_bench_rerun_manifest_frozen_pending_paid_execution",
        "source_artifacts": {"metadata_dedup_audit": {"path": str(audit_path)}},
        "N_paper_subset": 3,
        "conditions": ["E0", "E1", "E2", "E3", "E4", "E5", "E5_raw"],
        "reasoning_llm_call_cap": 42,
        "judge_llm_call_cap": 21,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dry_run = build_dry_run(manifest, manifest_path)

    assert dry_run["stage_status"] == "ready_for_paid_execution"
    assert dry_run["paid_calls_executed"] == 0
    assert dry_run["sample_condition_job_cap"] == 21
    assert dry_run["reasoning_llm_call_cap"] == 42
    assert dry_run["judge_llm_call_cap"] == 21


def test_track_b_dry_run_blocks_zero_ocr_cap_for_current_ocr_conditions(tmp_path: Path) -> None:
    audit_path = tmp_path / "metadata_dedup_audit.json"
    samples = [
        {
            "sample_id": "absolute-clean_render_0000",
            "family": "absolute",
            "dedup_key": "k0",
        }
    ]
    audit_path.write_text(json.dumps({"selected_subset": samples}), encoding="utf-8")
    manifest = {
        "stage_status": "geosym_bench_rerun_manifest_frozen_pending_paid_execution",
        "source_artifacts": {"metadata_dedup_audit": {"path": str(audit_path)}},
        "N_paper_subset": 1,
        "conditions": ["E0", "E1", "E2", "E3", "E4", "E5", "E5_raw"],
        "reasoning_llm_call_cap": 14,
        "judge_llm_call_cap": 7,
        "ocr_http_attempt_cap": 0,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dry_run = build_dry_run(manifest, manifest_path)

    assert dry_run["stage_status"] == "blocking_findings_present"
    assert dry_run["paid_calls_executed"] == 0
    assert any(finding.startswith("ocr_zero_cap_conflicts") for finding in dry_run["blocking_findings"])


def test_track_b_dry_run_scopes_caps_to_selected_conditions(tmp_path: Path) -> None:
    audit_path = tmp_path / "metadata_dedup_audit.json"
    samples = [
        {"sample_id": f"absolute-clean_render_{i:04d}", "family": "absolute", "dedup_key": f"k{i}"}
        for i in range(3)
    ]
    audit_path.write_text(json.dumps({"selected_subset": samples}), encoding="utf-8")
    manifest = {
        "stage_status": "geosym_bench_rerun_manifest_frozen_pending_paid_execution",
        "source_artifacts": {"metadata_dedup_audit": {"path": str(audit_path)}},
        "N_paper_subset": 3,
        "conditions": ["E5", "E5_raw"],
        "sample_condition_job_cap": 6,
        "reasoning_llm_call_cap": 21,
        "judge_llm_call_cap": 6,
        "ocr_http_attempt_cap": 6,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dry_run = build_dry_run(manifest, manifest_path)

    assert dry_run["stage_status"] == "ready_for_paid_execution"
    assert dry_run["conditions"] == ["E5", "E5_raw"]
    assert dry_run["output_conditions"] == ["E0", "E1", "E2", "E3", "E4", "E5", "E5_raw"]
    assert dry_run["sample_condition_job_cap"] == 6
    assert dry_run["reasoning_llm_call_cap"] == 21
    assert dry_run["judge_llm_call_cap"] == 6
    assert dry_run["ocr_http_attempt_cap"] == 6
    assert dry_run["ocr_dependent_conditions"] == ["E5", "E5_raw"]


def test_track_b_dry_run_allows_zero_ocr_cap_for_fallback_first_mode(tmp_path: Path) -> None:
    audit_path = tmp_path / "metadata_dedup_audit.json"
    samples = [
        {"sample_id": f"absolute-clean_render_{i:04d}", "family": "absolute", "dedup_key": f"k{i}"}
        for i in range(2)
    ]
    audit_path.write_text(json.dumps({"selected_subset": samples}), encoding="utf-8")
    manifest = {
        "stage_status": "geosym_bench_rerun_manifest_frozen_pending_paid_execution",
        "source_artifacts": {"metadata_dedup_audit": {"path": str(audit_path)}},
        "N_paper_subset": 2,
        "conditions": ["E5", "E5_raw"],
        "m0_execution_mode": M0_EXECUTION_MODE_FALLBACK_FIRST,
        "sample_condition_job_cap": 4,
        "reasoning_llm_call_cap": 14,
        "judge_llm_call_cap": 4,
        "ocr_http_attempt_cap": 0,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dry_run = build_dry_run(manifest, manifest_path)

    assert dry_run["stage_status"] == "ready_for_paid_execution"
    assert dry_run["m0_execution_mode"] == M0_EXECUTION_MODE_FALLBACK_FIRST
    assert dry_run["ocr_http_attempt_cap"] == 0
    assert dry_run["ocr_dependent_conditions"] == []


def test_track_b_parse_args_rejects_unknown_condition() -> None:
    import pytest

    with pytest.raises(SystemExit):
        parse_args(["--conditions", "E9"])


def test_attempt_counter_counts_transport_attempts_and_restores_wrapper() -> None:
    class FlakyAdapter(BaseLLMAdapter):
        def __init__(self) -> None:
            super().__init__("flaky")
            self.calls = 0

        def _call_api(self, image_base64: str, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary timeout")
            return "{}"

    original = BaseLLMAdapter._call_with_retry
    counter = AttemptCounter()

    with count_llm_http_attempts(counter):
        result = FlakyAdapter().call_text_only("prompt")

    assert result == "{}"
    assert counter.http_attempts == 2
    assert BaseLLMAdapter._call_with_retry is original


def test_mathpix_counter_counts_posts_and_forces_single_retry() -> None:
    import recognizer.mathpix.mathpix_image_ocr as mathpix_ocr

    original_retries = mathpix_ocr._MAX_RETRIES
    original_post = mathpix_ocr.requests.post
    counter = OcrAttemptCounter(cap=1)

    def fake_post(*args, **kwargs):
        class Response:
            status_code = 200

            def json(self):
                return {}

        return Response()

    try:
        mathpix_ocr.requests.post = fake_post
        with count_mathpix_http_attempts(counter):
            assert mathpix_ocr._MAX_RETRIES == 1
            assert mathpix_ocr.requests.post("url").json() == {}

        assert counter.http_attempts == 1
        assert mathpix_ocr._MAX_RETRIES == original_retries
        assert mathpix_ocr.requests.post is fake_post
    finally:
        mathpix_ocr.requests.post = original_post
        mathpix_ocr._MAX_RETRIES = original_retries


def test_track_b_checkpoint_requires_reauthorization_for_unknown_attempts() -> None:
    checkpoint = {
        "stage_status": "interrupted_paid_run_with_unaccounted_in_progress_sample",
        "blocking_findings": ["in_progress_sample_attempt_count_unknown:absolute-clean_render_0013"],
        "in_progress_sample": {"sample_id": "absolute-clean_render_0013"},
    }

    assert checkpoint_requires_reauthorization(checkpoint) is True
    assert checkpoint_requires_reauthorization({"stage_status": "partial_paid_run", "blocking_findings": []}) is False


def test_condition_row_hashes_staged_pipeline_chain_text() -> None:
    sample = {"sample_id": "absolute-clean_render_0000", "family": "absolute", "dedup_key": "k0"}
    cache = {
        "conditions": {
            "E5": {
                "status": "success",
                "cached": False,
                "eval": {"total_score": 90.0, "dimension_scores": {}, "sub_scores": {}},
                "judge": {"raw_judge_response": "{}"},
                "pipeline": {
                    "steps": {"L3": {"raw": "{\"expression\":\"x+1\"}", "parsed": {"expression": "x+1"}}},
                    "final_parsed": {
                        "expression": "x+1",
                        "domain": "all real numbers",
                        "range": "all real numbers",
                        "inference_explanation": "The plotted line has slope 1.",
                    },
                },
            },
        },
    }

    row = condition_row(sample, cache, "E5")

    # Until 2026-09-19 this expected the four ``final_parsed`` answer lines,
    # because layered pipelines were rebuilt from their conclusions while
    # single-step ones sent their raw output verbatim. That asymmetry was the
    # bug -- the Judge scored ``evidence_support`` on layered arms after their
    # derivation had been removed -- so the expectation now follows the raw
    # step transcript both styles share.
    empty_hash = sha256_hex_bytes(canonical_json_bytes(""))
    expected_hash = sha256_hex_bytes(
        canonical_json_bytes('=== L3 ===\n{"expression":"x+1"}')
    )
    assert row["chain_text_sha256"] == expected_hash
    assert row["chain_text_sha256"] != empty_hash
