import json
from pathlib import Path

from scripts.prepare_public_benchmark_pilot import stable_file_hash
from scripts.run_robustness_confirmation import existing_paid_result_requires_reauthorization, load_resume_checkpoint


def test_stress_128_paid_result_requires_reauthorization_for_unknown_attempts() -> None:
    result = {
        "stage_status": "m1m2_stress_128_paid_run_interrupted_uncheckpointed",
        "actual_llm_http_attempts": None,
        "actual_ocr_http_attempts": 0,
        "blocking_findings": ["actual_llm_http_attempts_unknown"],
    }

    assert existing_paid_result_requires_reauthorization(result) is True
    assert (
        existing_paid_result_requires_reauthorization(
            {
                "stage_status": "m1m2_stress_128_paid_run_interrupted",
                "actual_llm_http_attempts": 12,
                "actual_ocr_http_attempts": 0,
                "blocking_findings": [],
            }
        )
        is False
    )


def test_stress_128_resume_materializes_dangling_timeout_attempt(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest = {"llm_http_attempt_cap": 896}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    completed = [
        {
            "stress_id": f"s{i}",
            "condition": "G_selective_propagation",
            "task_success": True,
        }
        for i in range(227)
    ]
    attempted = [
        {
            "stress_id": f"s{i}",
            "condition": "G_selective_propagation",
            "expected_answerability": "answerable",
            "answer_type": "numeric",
            "reference": 1.0,
        }
        for i in range(228)
    ]
    checkpoint = {
        "source_artifacts": {"manifest": {"sha256": stable_file_hash(manifest_path)}},
        "results": completed,
        "attempted_jobs": attempted,
        "actual_llm_http_attempts": 228,
        "error": "Request timed out.",
    }
    (output_dir / "stress_128_paid_result.json").write_text(json.dumps(checkpoint), encoding="utf-8")

    results, attempted_jobs, attempts = load_resume_checkpoint(output_dir, manifest_path, manifest)

    assert attempts == 228
    assert len(attempted_jobs) == 228
    assert len(results) == 228
    assert results[-1]["stress_id"] == "s227"
    assert results[-1]["answer_status"] == "transport_failure"
    assert results[-1]["task_success"] is False
