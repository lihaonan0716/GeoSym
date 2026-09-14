import json
from pathlib import Path

import pytest

_DEV_LOCAL_ARTIFACTS = [
    Path("docs/PAPER_LEVEL_EVIDENCE_PACKAGE_PLAN.md"),
    Path("outputs/paper_level_evidence_package"),
]
REQUIRES_PAPER_PACKAGE_ARTIFACTS = pytest.mark.skipif(
    any(not path.exists() for path in _DEV_LOCAL_ARTIFACTS),
    reason="requires dev-local outputs/ and docs/ artifacts; see README",
)

from scripts.finalize_full_paper_evidence_package import (
    build_submission_audit,
    build_track_c_artifacts,
    build_track_b_rerun_manifest,
    build_track_d_paper_stress_audit,
    build_manuscript_integration_audit,
    build_track_d_summary,
    open_model_candidates,
    track_b_paid_result_is_complete,
)


def test_track_c_stops_when_openocr_not_frozen(tmp_path: Path) -> None:
    diagnostic_plan = tmp_path / "diagnostic_plan.json"
    diagnostic_result = tmp_path / "diagnostic_result.json"
    openocr_summary = tmp_path / "openocr_summary.json"
    diagnostic_plan.write_text(
        json.dumps({"split": "test", "plan_sha256": "p", "selected_samples": [{"sample_id": f"s{i}"} for i in range(50)]}),
        encoding="utf-8",
    )
    diagnostic_result.write_text(json.dumps({"results": []}), encoding="utf-8")
    openocr_summary.write_text(json.dumps({"experiment_status": "openocr_runtime_unavailable"}), encoding="utf-8")

    frozen, paid_result, analysis = build_track_c_artifacts(diagnostic_plan, diagnostic_result, openocr_summary, tmp_path)

    assert frozen["paid_run_approved"] is False
    assert paid_result["paid_calls_executed"] == 0
    assert analysis["experiment_status"] == "track_c_stopped_openocr_gate"
    assert analysis["stop_reason"] == "openocr_runtime_unavailable"


def test_open_model_candidates_excludes_proprietary_and_accepts_local_marker() -> None:
    config = {
        "models": {
            "OpenAI": {"adapter": "x", "models": {"GPT": "gpt"}},
            "LocalOllama": {"adapter": "adapters.ollama.Adapter", "models": {"LLaVA": "llava"}},
        }
    }

    assert open_model_candidates(config) == [
        {
            "provider": "LocalOllama",
            "display_name": "LLaVA",
            "api_model_id": "llava",
            "adapter": "adapters.ollama.Adapter",
        }
    ]


def test_track_d_stops_when_track_c_incomplete(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("models: {}\n", encoding="utf-8")

    summary = build_track_d_summary(config_path, {"experiment_status": "track_c_stopped_openocr_gate", "stop_reason": "x"}, tmp_path)

    assert summary["experiment_status"] == "open_mllm_not_configured"
    assert summary["paid_calls_executed"] == 0


def write_stage(path: Path, stage_status: str, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "test", "stage_status": stage_status, "paid_calls_executed": 0}
    payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_submission_audit_distinguishes_clean_stops_from_completed_experiments(tmp_path: Path) -> None:
    output_root = tmp_path / "paper_level_evidence_package"
    write_stage(output_root / "track_a_governance_cleanup" / "track_a_governance_audit.json", "track_a_governance_clean")
    write_stage(output_root / "geosym_bench_controlled" / "metadata_dedup_audit.json", "geosym_bench_dedup_audit_passed_subset_ready")
    write_stage(
        output_root / "geosym_bench_controlled" / "post_remediation_rerun_manifest.json",
        "geosym_bench_post_remediation_rerun_failed_uncounted_interruption_clean_stop",
        blocking_findings=["post_remediation_paid_result_not_complete:interrupted_paid_run_with_unaccounted_in_progress_sample"],
    )
    write_stage(output_root / "plotqa_public_500" / "pre_execution_audit.json", "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim")
    write_stage(
        output_root / "m1m2_stress_128" / "pre_execution_audit.json",
        "m1m2_stress_128_failed_uncounted_interruption_clean_stop",
        blocking_findings=["m1m2_stress_128_paid_result_incomplete:m1m2_stress_128_paid_run_interrupted_uncheckpointed"],
    )
    write_stage(output_root / "open_repro_sensitivity" / "summary.json", "open_mllm_not_configured")
    write_stage(output_root / "specialized_chart_model_anchor" / "summary.json", "specialized_chart_model_runtime_unavailable")
    write_stage(output_root / "fusion_diagnostics" / "confidence_tier_accuracy.json", "confidence_tier_accuracy_completed")
    write_stage(output_root / "fusion_diagnostics" / "parameter_sensitivity.json", "parameter_sensitivity_completed")
    write_stage(output_root / "fusion_diagnostics" / "cost_latency.json", "schema_initialized_from_track_a_ledger")
    write_stage(output_root / "fusion_diagnostics" / "real_chart_error_decomposition.json", "real_chart_reference_audit_complete_measurement_not_run")
    write_stage(output_root / "fusion_diagnostics" / "fusion_positioning_audit.json", "fusion_positioning_claim_boundaries_recorded")
    write_stage(output_root / "fusion_diagnostics" / "weighting_sensitivity.json", "weighting_sensitivity_completed")
    write_stage(output_root / "submission_audit" / "manuscript_integration_audit.json", "manuscript_integration_claim_safety_complete", blocking_findings=[])
    write_stage(output_root / "submission_audit" / "final_review_loop.json", "final_review_loop_claim_safety_no_blocking_findings", blocking_findings=[])

    audit = build_submission_audit(output_root)

    assert audit["stage_status"] == "paper_level_package_ready_for_manuscript_rewrite_with_clean_stops"
    assert audit["paper_level_experiments_complete"] is False
    assert audit["paper_level_experiment_paths_resolved"] is True
    assert "track_b_post_remediation_rerun_completed" in audit["incomplete_experiment_items"]
    assert audit["unresolved_experiment_items"] == []


def test_final_review_residual_risks_do_not_report_completed_track_c_as_missing(tmp_path: Path) -> None:
    from scripts.finalize_full_paper_evidence_package import build_final_review_loop_audit

    output_root = tmp_path / "paper_level_evidence_package"
    write_stage(output_root / "evidence_ledger.json", "ledger_clean")
    write_stage(output_root / "track_a_governance_cleanup" / "track_a_governance_audit.json", "track_a_governance_clean")
    write_stage(output_root / "geosym_bench_controlled" / "metadata_dedup_audit.json", "geosym_bench_dedup_audit_passed_subset_ready")
    write_stage(
        output_root / "geosym_bench_controlled" / "post_remediation_rerun_manifest.json",
        "geosym_bench_post_remediation_rerun_failed_uncounted_interruption_clean_stop",
        blocking_findings=["post_remediation_paid_result_not_complete"],
    )
    write_stage(output_root / "plotqa_public_500" / "pre_execution_audit.json", "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim")
    write_stage(
        output_root / "m1m2_stress_128" / "pre_execution_audit.json",
        "m1m2_stress_128_failed_uncounted_interruption_clean_stop",
        blocking_findings=["m1m2_stress_128_paid_result_incomplete"],
    )
    write_stage(output_root / "open_repro_sensitivity" / "summary.json", "open_mllm_not_configured")
    write_stage(output_root / "specialized_chart_model_anchor" / "summary.json", "specialized_chart_model_runtime_unavailable")
    write_stage(output_root / "fusion_diagnostics" / "fusion_positioning_audit.json", "fusion_positioning_claim_boundaries_recorded")
    write_stage(output_root / "fusion_diagnostics" / "weighting_sensitivity.json", "weighting_sensitivity_completed")
    write_stage(
        output_root / "submission_audit" / "manuscript_integration_audit.json",
        "manuscript_integration_claim_safety_complete",
        forbidden_positive_language_hits=[],
        blocking_findings=[],
    )

    audit = build_final_review_loop_audit(output_root)
    risks = "\n".join(audit["residual_risks"])

    assert audit["stage_status"] == "final_review_loop_claim_safety_no_blocking_findings"
    assert audit["next_required_action"] == "start_manuscript_rewrite_from_evidence_ledger_with_clean_stop_limits"
    assert "No Track C 500 paid confirmation exists" not in risks
    assert "Track C 500 completed, but the result is negative" in risks


def test_final_review_blocks_completed_track_b_with_empty_chain_hash_audit(tmp_path: Path) -> None:
    from scripts.finalize_full_paper_evidence_package import build_final_review_loop_audit

    output_root = tmp_path / "paper_level_evidence_package"
    write_stage(output_root / "evidence_ledger.json", "ledger_clean")
    write_stage(output_root / "track_a_governance_cleanup" / "track_a_governance_audit.json", "track_a_governance_clean")
    write_stage(output_root / "geosym_bench_controlled" / "metadata_dedup_audit.json", "geosym_bench_dedup_audit_passed_subset_ready")
    write_stage(
        output_root / "geosym_bench_controlled" / "post_remediation_rerun_manifest.json",
        "geosym_bench_post_remediation_rerun_completed",
        blocking_findings=[],
    )
    write_stage(
        output_root / "geosym_bench_controlled" / "post_remediation_e5_backfill_full_240_final_analysis.json",
        "full_240_all_conditions_with_repaired_e5_e5raw_complete",
        summary={"overall_by_condition": {condition: {"rows": 240, "success": 240} for condition in ["E0", "E1", "E2", "E3", "E4", "E5", "E5_raw"]}},
        cached_conditions_reused=["E0", "E1", "E2", "E3", "E4"],
        chain_text_reuse_audit={"rows_checked": 480, "empty_chain_text_hash_rows": 480, "judge_reuse_violations": []},
        governance={"within_global_llm_cap": True, "within_global_ocr_cap": True, "sample_replacement": False, "conditions_frozen": True},
    )
    write_stage(output_root / "geosym_bench_controlled" / "clean_grid_fallback_m0_quality_8x3.json", "completed_zero_cost_local_cv_diagnostic")
    write_stage(output_root / "plotqa_public_500" / "pre_execution_audit.json", "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim")
    write_stage(output_root / "m1m2_stress_128" / "pre_execution_audit.json", "m1m2_stress_128_confirmation_completed")
    write_stage(output_root / "open_repro_sensitivity" / "summary.json", "open_mllm_not_configured")
    write_stage(output_root / "specialized_chart_model_anchor" / "summary.json", "specialized_chart_model_runtime_unavailable")
    write_stage(output_root / "fusion_diagnostics" / "fusion_positioning_audit.json", "fusion_positioning_claim_boundaries_recorded")
    write_stage(output_root / "fusion_diagnostics" / "weighting_sensitivity.json", "weighting_sensitivity_completed")
    write_stage(
        output_root / "submission_audit" / "manuscript_integration_audit.json",
        "manuscript_integration_claim_safety_complete",
        forbidden_positive_language_hits=[],
        blocking_findings=[],
    )

    audit = build_final_review_loop_audit(output_root)

    assert audit["stage_status"] == "final_review_loop_blocked"
    assert "track_b_empty_chain_text_hash_rows:480" in audit["blocking_findings"]


def test_final_review_records_track_b_m1m2_not_identifiable_as_claim_limit(tmp_path: Path) -> None:
    from scripts.finalize_full_paper_evidence_package import build_final_review_loop_audit

    output_root = tmp_path / "paper_level_evidence_package"
    write_stage(output_root / "evidence_ledger.json", "ledger_clean")
    write_stage(output_root / "track_a_governance_cleanup" / "track_a_governance_audit.json", "track_a_governance_clean")
    write_stage(output_root / "geosym_bench_controlled" / "metadata_dedup_audit.json", "geosym_bench_dedup_audit_passed_subset_ready")
    write_stage(
        output_root / "geosym_bench_controlled" / "post_remediation_rerun_manifest.json",
        "geosym_bench_post_remediation_rerun_completed",
        blocking_findings=[],
    )
    write_stage(
        output_root / "geosym_bench_controlled" / "post_remediation_e5_backfill_full_240_final_analysis.json",
        "full_240_all_conditions_with_repaired_e5_e5raw_complete",
        summary={"overall_by_condition": {condition: {"rows": 240, "success": 240} for condition in ["E0", "E1", "E2", "E3", "E4", "E5", "E5_raw"]}},
        cached_conditions_reused=["E0", "E1", "E2", "E3", "E4"],
        chain_text_reuse_audit={"rows_checked": 480, "empty_chain_text_hash_rows": 0, "judge_reuse_violations": []},
        governance={"within_global_llm_cap": True, "within_global_ocr_cap": True, "sample_replacement": False, "conditions_frozen": True},
    )
    write_stage(output_root / "geosym_bench_controlled" / "clean_grid_fallback_m0_quality_8x3.json", "completed_zero_cost_local_cv_diagnostic")
    write_stage(
        output_root / "geosym_bench_controlled" / "track_b_m1m2_identifiability_analysis.json",
        "track_b_m1m2_not_identifiable_from_repaired_rerun",
        identifiability_blockers=[
            "track_b_no_judge_reuse_pairs",
            "track_b_min_detectable_effect_exceeds_historical_m1m2_effect",
        ],
        power_analysis={"required_pairs_for_historical_effect_normal_approx": 6700},
    )
    write_stage(output_root / "plotqa_public_500" / "pre_execution_audit.json", "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim")
    write_stage(output_root / "m1m2_stress_128" / "pre_execution_audit.json", "m1m2_stress_128_confirmation_completed")
    write_stage(output_root / "open_repro_sensitivity" / "summary.json", "open_mllm_not_configured")
    write_stage(output_root / "specialized_chart_model_anchor" / "summary.json", "specialized_chart_model_runtime_unavailable")
    write_stage(output_root / "fusion_diagnostics" / "fusion_positioning_audit.json", "fusion_positioning_claim_boundaries_recorded")
    write_stage(output_root / "fusion_diagnostics" / "weighting_sensitivity.json", "weighting_sensitivity_completed")
    write_stage(
        output_root / "submission_audit" / "manuscript_integration_audit.json",
        "manuscript_integration_claim_safety_complete",
        forbidden_positive_language_hits=[],
        blocking_findings=[],
    )

    audit = build_final_review_loop_audit(output_root)

    assert audit["stage_status"] == "final_review_loop_claim_safety_no_blocking_findings"
    assert not audit["blocking_findings"]
    assert any("does not make M1/M2 identifiable" in risk for risk in audit["residual_risks"])
    assert audit["next_required_action"] == "start_manuscript_rewrite_from_evidence_ledger_with_clean_stop_limits"


def test_track_b_reauthorized_result_supersedes_default_failed_smoke(tmp_path: Path, monkeypatch) -> None:
    import scripts.finalize_full_paper_evidence_package as finalizer

    output_root = tmp_path / "paper_level_evidence_package"
    track_b_dir = output_root / "geosym_bench_controlled"
    samples = [
        {"sample_id": f"absolute-clean_render_{index:04d}", "family": "absolute", "dedup_key": f"k{index}"}
        for index in range(240)
    ]
    write_stage(
        track_b_dir / "metadata_dedup_audit.json",
        "geosym_bench_dedup_audit_passed_subset_ready",
        selected_subset=samples,
        selected_family_counts={"absolute": 240},
        pool_summary={"N_generated_pool": 480},
    )
    write_stage(
        track_b_dir / "post_remediation_paid_result.json",
        "interrupted_paid_run_with_unaccounted_in_progress_sample",
        actual_llm_http_attempts=None,
        actual_ocr_http_attempts=12,
        blocking_findings=["in_progress_sample_attempt_count_unknown:s1"],
    )
    write_stage(
        track_b_dir / "post_remediation_reauthorized_paid_result.json",
        "geosym_bench_post_remediation_rerun_completed",
        completed_sample_count=240,
        actual_llm_http_attempts=5040,
        actual_ocr_http_attempts=0,
        conditions=["E0", "E1", "E2", "E3", "E4", "E5", "E5_raw"],
        summary={
            "overall_by_condition": {
                "E0": {"rows": 240, "success": 240},
                "E1": {"rows": 240, "success": 240},
                "E2": {"rows": 240, "success": 240},
                "E3": {"rows": 240, "success": 240},
                "E4": {"rows": 240, "success": 240},
                "E5": {"rows": 240, "success": 240},
                "E5_raw": {"rows": 240, "success": 240},
            }
        },
    )
    monkeypatch.setattr(
        finalizer,
        "select_track_b_model_provenance",
        lambda config_path: (
            {"provider": "OpenAI", "api_model_id": "gpt-test", "saved_key_available_at_manifest_time": True},
            {"provider": "Qwen", "api_model_id": "qwen-test", "saved_key_available_at_manifest_time": True},
        ),
    )
    monkeypatch.setattr(finalizer, "run_prompt_fairness_audit", lambda: [{"passed": True}])

    manifest = build_track_b_rerun_manifest(output_root)

    assert manifest["stage_status"] == "geosym_bench_post_remediation_rerun_completed"
    assert manifest["completed_paid_result_available"] is True
    assert not manifest["blocking_findings"]
    sources = manifest["source_artifacts"]
    assert sources["completed_paid_result"]["path"].endswith("post_remediation_reauthorized_paid_result.json")
    assert sources["default_interrupted_paid_result"]["status_label"] == "diagnostic_failed_smoke"


def test_track_b_condition_gate_rejects_completed_stage_with_failed_e5_rows() -> None:
    payload = {
        "stage_status": "geosym_bench_post_remediation_rerun_completed",
        "actual_llm_http_attempts": 2792,
        "actual_ocr_http_attempts": 720,
        "conditions": ["E0", "E1", "E2", "E3", "E4", "E5", "E5_raw"],
        "summary": {
            "overall_by_condition": {
                "E5": {"rows": 240, "success": 0},
                "E5_raw": {"rows": 240, "success": 0},
            }
        },
    }

    complete, findings = track_b_paid_result_is_complete(payload, expected_samples=240)

    assert complete is False
    assert "condition_success_not_240:E5:0" in findings
    assert "condition_success_not_240:E5_raw:0" in findings


def test_track_b_final_backfill_result_supersedes_mathpix_first_failure(tmp_path: Path, monkeypatch) -> None:
    import scripts.finalize_full_paper_evidence_package as finalizer

    output_root = tmp_path / "paper_level_evidence_package"
    track_b_dir = output_root / "geosym_bench_controlled"
    samples = [
        {"sample_id": f"absolute-clean_render_{index:04d}", "family": "absolute", "dedup_key": f"k{index}"}
        for index in range(240)
    ]
    write_stage(
        track_b_dir / "metadata_dedup_audit.json",
        "geosym_bench_dedup_audit_passed_subset_ready",
        selected_subset=samples,
        selected_family_counts={"absolute": 240},
        pool_summary={"N_generated_pool": 480},
    )
    write_stage(
        track_b_dir / "post_remediation_reauthorized_paid_result.json",
        "geosym_bench_post_remediation_rerun_completed",
        actual_llm_http_attempts=2792,
        actual_ocr_http_attempts=720,
        conditions=["E0", "E1", "E2", "E3", "E4", "E5", "E5_raw"],
        summary={"overall_by_condition": {"E5": {"rows": 240, "success": 0}, "E5_raw": {"rows": 240, "success": 0}}},
    )
    write_stage(
        track_b_dir / "post_remediation_e5_backfill_full_240_final_analysis.json",
        "full_240_e5_e5raw_backfill_complete",
        N_paper_subset=240,
        m0_execution_mode="clean_grid_fallback_first",
        actual_llm_http_attempts_completed_runs=2221,
        actual_ocr_http_attempts_completed_runs=0,
        conditions=["E5", "E5_raw"],
        summary={"overall_by_condition": {"E5": {"rows": 240, "success": 240}, "E5_raw": {"rows": 240, "success": 240}}},
        unresolved_rows=[],
    )
    monkeypatch.setattr(
        finalizer,
        "select_track_b_model_provenance",
        lambda config_path: (
            {"provider": "OpenAI", "api_model_id": "gpt-test", "saved_key_available_at_manifest_time": True},
            {"provider": "Qwen", "api_model_id": "qwen-test", "saved_key_available_at_manifest_time": True},
        ),
    )
    monkeypatch.setattr(finalizer, "run_prompt_fairness_audit", lambda: [{"passed": True}])

    manifest = build_track_b_rerun_manifest(output_root)

    assert manifest["stage_status"] == "geosym_bench_post_remediation_rerun_completed"
    assert manifest["completed_paid_result_available"] is True
    assert manifest["conditions"] == ["E5", "E5_raw"]
    assert manifest["m0_execution_mode"] == "clean_grid_fallback_first"
    assert manifest["actual_llm_http_attempts_from_completed_paid_result"] == 2221
    assert manifest["actual_ocr_http_attempts_from_completed_paid_result"] == 0
    sources = manifest["source_artifacts"]
    assert sources["completed_paid_result"]["path"].endswith("post_remediation_e5_backfill_full_240_final_analysis.json")
    assert sources["reauthorized_paid_result"]["status_label"] == "diagnostic_failed_infrastructure"


def test_track_b_full_seven_condition_final_result_is_complete(tmp_path: Path, monkeypatch) -> None:
    import scripts.finalize_full_paper_evidence_package as finalizer

    output_root = tmp_path / "paper_level_evidence_package"
    track_b_dir = output_root / "geosym_bench_controlled"
    samples = [
        {"sample_id": f"absolute-clean_render_{index:04d}", "family": "absolute", "dedup_key": f"k{index}"}
        for index in range(240)
    ]
    write_stage(
        track_b_dir / "metadata_dedup_audit.json",
        "geosym_bench_dedup_audit_passed_subset_ready",
        selected_subset=samples,
        selected_family_counts={"absolute": 240},
        pool_summary={"N_generated_pool": 480},
    )
    write_stage(
        track_b_dir / "post_remediation_e5_backfill_full_240_final_analysis.json",
        "full_240_all_conditions_with_repaired_e5_e5raw_complete",
        N_paper_subset=240,
        m0_execution_mode="clean_grid_fallback_first",
        actual_llm_http_attempts_completed_runs=2221,
        actual_ocr_http_attempts_completed_runs=0,
        conditions=["E0", "E1", "E2", "E3", "E4", "E5", "E5_raw"],
        executed_conditions=["E5", "E5_raw"],
        cached_conditions_reused=["E0", "E1", "E2", "E3", "E4"],
        summary={
            "overall_by_condition": {
                "E0": {"rows": 240, "success": 229},
                "E1": {"rows": 240, "success": 226},
                "E2": {"rows": 240, "success": 224},
                "E3": {"rows": 240, "success": 227},
                "E4": {"rows": 240, "success": 232},
                "E5": {"rows": 240, "success": 240},
                "E5_raw": {"rows": 240, "success": 240},
            }
        },
        unresolved_rows=[],
        governance={"within_global_llm_cap": True, "within_global_ocr_cap": True},
        chain_text_reuse_audit={
            "distinct_chain_text_sha256": 480,
            "empty_chain_text_hash_rows": 0,
            "judge_reuse_required_pairs": 0,
            "judge_reuse_violations": [],
        },
    )
    monkeypatch.setattr(
        finalizer,
        "select_track_b_model_provenance",
        lambda config_path: (
            {"provider": "OpenAI", "api_model_id": "gpt-test", "saved_key_available_at_manifest_time": True},
            {"provider": "Qwen", "api_model_id": "qwen-test", "saved_key_available_at_manifest_time": True},
        ),
    )
    monkeypatch.setattr(finalizer, "run_prompt_fairness_audit", lambda: [{"passed": True}])

    manifest = build_track_b_rerun_manifest(output_root)

    assert manifest["stage_status"] == "geosym_bench_post_remediation_rerun_completed"
    assert manifest["conditions"] == ["E0", "E1", "E2", "E3", "E4", "E5", "E5_raw"]
    assert manifest["completed_paid_result_condition_gate_findings"] == []


@REQUIRES_PAPER_PACKAGE_ARTIFACTS
def test_track_d_reauthorized_result_supersedes_default_failed_smoke(tmp_path: Path, monkeypatch) -> None:
    import scripts.finalize_full_paper_evidence_package as finalizer

    output_root = tmp_path / "paper_level_evidence_package"
    track_d_dir = output_root / "m1m2_stress_128"
    write_stage(track_d_dir / "source_pool_feasibility.json", "m1m2_stress_128_source_pool_feasible")
    write_stage(
        track_d_dir / "stress_128_manifest.json",
        "m1m2_stress_128_manifest_frozen_pending_paid_cap_approval",
        sample_level_job_cap=128,
        llm_http_attempt_cap=896,
        ocr_http_attempt_cap=0,
        condition_list=["a", "b", "c", "d", "e", "f", "g"],
        blocking_findings=[],
    )
    write_stage(
        track_d_dir / "stress_128_dry_run.json",
        "m1m2_stress_128_dry_run_passed",
        job_count=896,
        actual_llm_http_attempts=0,
        actual_ocr_http_attempts=0,
        blocking_findings=[],
    )
    write_stage(
        track_d_dir / "stress_128_paid_result.json",
        "m1m2_stress_128_paid_run_interrupted_uncheckpointed",
        actual_llm_http_attempts=None,
        actual_ocr_http_attempts=0,
        blocking_findings=["actual_llm_http_attempts_unknown"],
    )
    write_stage(
        track_d_dir / "stress_128_reauthorized" / "stress_128_paid_result.json",
        "m1m2_stress_128_confirmation_completed",
        paid_calls_executed=896,
        actual_llm_http_attempts=896,
        actual_ocr_http_attempts=0,
    )
    monkeypatch.setattr(finalizer, "governance_chain_is_clean", lambda payload: (True, []))

    audit = build_track_d_paper_stress_audit(output_root)

    assert audit["stage_status"] == "m1m2_stress_128_confirmation_completed"
    assert not audit["blocking_findings"]
    sources = audit["source_artifacts"]
    assert sources["stress_128_paid_result"]["path"].endswith("stress_128_reauthorized\\stress_128_paid_result.json")
    assert sources["default_interrupted_paid_result"]["status_label"] == "diagnostic_failed_smoke"


def test_manuscript_integration_scopes_completed_track_d_as_neutral_evidence(tmp_path: Path) -> None:
    output_root = tmp_path / "paper_level_evidence_package"
    track_d_result = output_root / "m1m2_stress_128" / "stress_128_reauthorized" / "stress_128_paid_result.json"
    track_d_result.parent.mkdir(parents=True)
    track_d_result.write_text(
        json.dumps(
            {
                "stage_status": "m1m2_stress_128_confirmation_completed",
                "condition_summary": {
                    "G_selective_propagation": {
                        "rows": 128,
                        "task_success": 82,
                        "harmful_propagation": 45,
                        "correct_answer": 18,
                        "correct_abstention": 64,
                    },
                    "G_all_evidence": {"rows": 128, "task_success": 85, "correct_answer": 25},
                    "G_candidate_unrouted": {"rows": 128, "task_success": 82},
                    "G_random_k": {"rows": 128, "task_success": 82, "harmful_propagation": 24, "correct_answer": 18},
                },
            }
        ),
        encoding="utf-8",
    )
    write_stage(
        output_root / "geosym_bench_controlled" / "post_remediation_rerun_manifest.json",
        "geosym_bench_post_remediation_rerun_failed_uncounted_interruption_clean_stop",
    )
    write_stage(output_root / "plotqa_public_500" / "pre_execution_audit.json", "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim")
    write_stage(
        output_root / "m1m2_stress_128" / "pre_execution_audit.json",
        "m1m2_stress_128_confirmation_completed",
        source_artifacts={"stress_128_paid_result": {"path": str(track_d_result)}},
    )
    write_stage(output_root / "open_repro_sensitivity" / "summary.json", "open_mllm_not_configured")
    write_stage(output_root / "specialized_chart_model_anchor" / "summary.json", "specialized_chart_model_runtime_unavailable")
    write_stage(output_root / "fusion_diagnostics" / "fusion_positioning_audit.json", "fusion_positioning_claim_boundaries_recorded")

    audit = build_manuscript_integration_audit(output_root)
    m1m2_claim = next(row for row in audit["claim_matrix"] if row["evidence"] == "M1/M2 candidate-guided stress evidence")

    assert audit["stage_status"] == "manuscript_integration_claim_safety_complete"
    assert not audit["blocking_findings"]
    assert audit["claim_safety_gates"]["expanded_track_d_paid_confirmation_scoped"] is True
    assert "82/128" in m1m2_claim["allowed_claim"]
    assert "G_random_k" in m1m2_claim["allowed_claim"]
    assert "18/64" in m1m2_claim["allowed_claim"]
    assert "25/64" in m1m2_claim["allowed_claim"]
    assert "neutral or negative mechanism evidence" in m1m2_claim["allowed_claim"]


def test_manuscript_integration_does_not_call_track_c_cached_e0e1_matched(tmp_path: Path) -> None:
    output_root = tmp_path / "paper_level_evidence_package"
    write_stage(
        output_root / "geosym_bench_controlled" / "post_remediation_rerun_manifest.json",
        "geosym_bench_post_remediation_rerun_completed",
    )
    write_stage(output_root / "plotqa_public_500" / "pre_execution_audit.json", "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim")
    write_stage(output_root / "m1m2_stress_128" / "pre_execution_audit.json", "m1m2_stress_128_failed_uncounted_interruption_clean_stop")
    write_stage(output_root / "open_repro_sensitivity" / "summary.json", "open_mllm_not_configured")
    write_stage(output_root / "specialized_chart_model_anchor" / "summary.json", "specialized_chart_model_runtime_unavailable")
    write_stage(output_root / "fusion_diagnostics" / "fusion_positioning_audit.json", "fusion_positioning_claim_boundaries_recorded")

    audit = build_manuscript_integration_audit(output_root)
    track_c_claim = next(row for row in audit["claim_matrix"] if row["evidence"] == "PlotQA public 500 package")

    assert "not a same-roster matched comparison" in track_c_claim["allowed_claim"]
    assert "matched cached E0/E1" not in track_c_claim["allowed_claim"]


def test_manuscript_integration_scopes_completed_track_b_without_clean_stop_text(tmp_path: Path) -> None:
    output_root = tmp_path / "paper_level_evidence_package"
    write_stage(
        output_root / "geosym_bench_controlled" / "post_remediation_rerun_manifest.json",
        "geosym_bench_post_remediation_rerun_completed",
    )
    write_stage(
        output_root / "geosym_bench_controlled" / "track_b_m1m2_identifiability_analysis.json",
        "track_b_m1m2_not_identifiable_from_repaired_rerun",
    )
    write_stage(output_root / "plotqa_public_500" / "pre_execution_audit.json", "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim")
    write_stage(
        output_root / "m1m2_stress_128" / "pre_execution_audit.json",
        "m1m2_stress_128_failed_uncounted_interruption_clean_stop",
    )
    write_stage(output_root / "open_repro_sensitivity" / "summary.json", "open_mllm_not_configured")
    write_stage(output_root / "specialized_chart_model_anchor" / "summary.json", "specialized_chart_model_runtime_unavailable")
    write_stage(output_root / "fusion_diagnostics" / "fusion_positioning_audit.json", "fusion_positioning_claim_boundaries_recorded")

    audit = build_manuscript_integration_audit(output_root)
    track_b_claim = next(row for row in audit["claim_matrix"] if row["evidence"] == "GeoSym-Bench controlled package")

    assert audit["stage_status"] == "manuscript_integration_claim_safety_complete"
    assert not audit["blocking_findings"]
    assert audit["claim_safety_gates"]["expanded_track_b_paid_confirmation_scoped"] is True
    assert "post-remediation rerun completed" in track_b_claim["allowed_claim"]
    assert "end-to-end execution evidence only" in track_b_claim["allowed_claim"]
    assert "stopped at an uncounted interruption" not in track_b_claim["allowed_claim"]
