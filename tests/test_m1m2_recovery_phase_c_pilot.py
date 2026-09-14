import json
from pathlib import Path

from scripts.run_m1m2_recovery_phase_c_pilot import apply_checkpoint_guard, build_dry_run, replace_job_rows


def test_phase_c_pilot_dry_run_enforces_true_call_graph_cap():
    prereg = {
        "created_at_utc": "ignored",
        "conditions": ["E5", "E5_raw"],
        "cap": {"pilot_llm_http_attempt_cap": 216, "pilot_ocr_http_attempt_cap": 0},
    }
    dry_rows = []
    for family in ["absolute", "linear", "quadratic", "cubic", "rational", "logarithmic", "exponential", "piecewise_linear"]:
        for level in ["q0_native", "q1_moderate_degradation", "q2_strong_degradation"]:
            dry_rows.append({"sample_id": f"{family}-clean_render_0000", "family": family, "quality_level": level})
    dry = {"stage_status": "phase_c_dry_run_passed", "rows": dry_rows}
    ledger = {"llm_cap": 4000, "cumulative_llm": 2403, "ocr_cap": 1200, "cumulative_ocr": 23}

    result = build_dry_run(prereg, dry, ledger)

    assert result["stage_status"] == "ready_for_paid_pilot"
    assert result["pilot_llm_http_attempt_cap"] == 216
    assert result["pilot_ocr_http_attempt_cap"] == 0
    assert result["condition_call_caps"] == {"E5": 5, "E5_raw": 4}
    assert result["decoding_overrides"]["openai_sdk_max_retries"] == 0
    assert result["decoding_overrides"]["qwen_sdk_max_retries"] == 0


def test_phase_c_pilot_dry_run_rejects_stale_192_cap():
    prereg = {
        "created_at_utc": "ignored",
        "conditions": ["E5", "E5_raw"],
        "cap": {"pilot_llm_http_attempt_cap": 192, "pilot_ocr_http_attempt_cap": 0},
    }
    dry_rows = [
        {"sample_id": f"s{i}", "family": "linear", "quality_level": "q0_native"}
        for i in range(8)
    ]
    ledger = {"llm_cap": 4000, "cumulative_llm": 2403, "ocr_cap": 1200, "cumulative_ocr": 23}

    result = build_dry_run(prereg, {"stage_status": "phase_c_dry_run_passed", "rows": dry_rows}, ledger)

    assert result["stage_status"] == "blocking_findings_present"
    assert any("pilot_llm_cap_mismatch" in finding for finding in result["blocking_findings"])


def test_phase_c_pilot_dry_run_uses_requested_checkpoint_path(tmp_path):
    prereg = {
        "created_at_utc": "ignored",
        "conditions": ["E5", "E5_raw"],
        "cap": {"pilot_llm_http_attempt_cap": 216, "pilot_ocr_http_attempt_cap": 0},
    }
    dry_rows = []
    for family in ["absolute", "linear", "quadratic", "cubic", "rational", "logarithmic", "exponential", "piecewise_linear"]:
        for level in ["q0_native", "q1_moderate_degradation", "q2_strong_degradation"]:
            dry_rows.append({"sample_id": f"{family}-clean_render_0000", "family": family, "quality_level": level})
    ledger = {"llm_cap": 4000, "cumulative_llm": 2403, "ocr_cap": 1200, "cumulative_ocr": 23}
    checkpoint = tmp_path / "repair_round_checkpoint.json"
    checkpoint.write_text(json.dumps({"actual_llm_http_attempts": 216}), encoding="utf-8")

    result = build_dry_run(
        prereg,
        {"stage_status": "phase_c_dry_run_passed", "rows": dry_rows},
        ledger,
        Path(checkpoint),
    )

    assert result["stage_status"] == "blocking_findings_present"
    assert "pilot_llm_cap_exhausted:used=216:cap=216" in result["blocking_findings"]


def test_phase_c_pilot_dry_run_uses_manifest_round_cap(tmp_path, monkeypatch):
    manifest = tmp_path / "round3_manifest.json"
    manifest.write_text(
        json.dumps({"cap": {"llm_http_attempt_cap": 87, "ocr_http_attempt_cap": 0}, "round_label": "round3"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("GEOSYM_PHASE_C_MANIFEST", str(manifest))
    prereg = {
        "created_at_utc": "ignored",
        "conditions": ["E5", "E5_raw"],
        "cap": {"pilot_llm_http_attempt_cap": 216, "pilot_ocr_http_attempt_cap": 0},
    }
    dry_rows = []
    for family in ["absolute", "linear", "quadratic", "cubic", "rational", "logarithmic", "exponential", "piecewise_linear"]:
        for level in ["q0_native", "q1_moderate_degradation", "q2_strong_degradation"]:
            dry_rows.append({"sample_id": f"{family}-clean_render_0000", "family": family, "quality_level": level})
    ledger = {"llm_cap": 4000, "cumulative_llm": 2808, "ocr_cap": 1200, "cumulative_ocr": 23}

    result = build_dry_run(prereg, {"stage_status": "phase_c_dry_run_passed", "rows": dry_rows}, ledger, tmp_path / "round3_result.json")

    assert result["stage_status"] == "ready_for_paid_pilot"
    assert result["pilot_llm_http_attempt_cap"] == 87
    assert result["original_preregistered_pilot_llm_http_attempt_cap"] == 216


def test_phase_c_pilot_checkpoint_guard_blocks_in_progress_checkpoint(tmp_path):
    dry_run = {"stage_status": "ready_for_paid_pilot", "blocking_findings": []}
    checkpoint = tmp_path / "phase_c_pilot_paid_result.json"
    checkpoint.write_text(
        json.dumps({
            "stage_status": "running_paid_pilot",
            "actual_llm_http_attempts": 1,
            "actual_ocr_http_attempts": 0,
            "completed_sample_quality_jobs": 0,
            "in_progress_job": {"sample_id": "s0", "quality_level": "q0_native"},
        }),
        encoding="utf-8",
    )

    result = apply_checkpoint_guard(dry_run, checkpoint)

    assert result["stage_status"] == "blocking_findings_present"
    assert "existing_paid_checkpoint_has_in_progress_job_reauthorization_required" in result["blocking_findings"]
    assert result["checkpoint_summary"]["actual_llm_http_attempts"] == 1


def test_phase_c_replaces_failed_job_rows_before_retry():
    old_rows = [
        {"sample_id": "s0", "quality_level": "q0_native", "condition": "E5", "status": "runner_error"},
        {"sample_id": "s0", "quality_level": "q0_native", "condition": "E5_raw", "status": "runner_error"},
        {"sample_id": "s1", "quality_level": "q0_native", "condition": "E5", "status": "success"},
    ]
    new_rows = [
        {"sample_id": "s0", "quality_level": "q0_native", "condition": "E5", "status": "success"},
        {"sample_id": "s0", "quality_level": "q0_native", "condition": "E5_raw", "status": "success"},
    ]

    result = replace_job_rows(old_rows, new_rows)

    assert len(result) == 3
    assert all(row["status"] == "success" for row in result if row["sample_id"] == "s0")
