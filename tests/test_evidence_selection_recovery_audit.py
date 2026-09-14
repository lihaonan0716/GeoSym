import json
from pathlib import Path

from scripts.audit_evidence_selection_recovery import build_audit, ledger_attempts_for_artifact, repair_ledger


def test_ledger_attempts_for_artifact_sums_matching_entries(tmp_path: Path):
    artifact = tmp_path / "result.json"
    artifact.write_text("{}", encoding="utf-8")
    other = tmp_path / "other.json"
    ledger = {
        "entries": [
            {"artifact_path": str(artifact), "llm_attempts": 9},
            {"artifact_path": str(other), "llm_attempts": 100},
            {"artifact_path": str(artifact.resolve()), "llm_attempts": 18},
        ]
    }

    assert ledger_attempts_for_artifact(ledger, artifact) == 27


def test_repair_ledger_adds_only_missing_checkpoint_attempts(tmp_path: Path):
    artifact = tmp_path / "phase_c_result.json"
    artifact.write_text("{}", encoding="utf-8")
    ledger_path = tmp_path / "budget_ledger.json"
    ledger_path.write_text(
        json.dumps({
            "entries": [{"artifact_path": str(artifact), "llm_attempts": 27, "ocr_attempts": 0}],
            "cumulative_llm": 2644,
            "cumulative_ocr": 23,
            "interpretation_limits": [],
        }),
        encoding="utf-8",
    )
    audit = {
        "phase_c_summary": {
            "audited_checkpoints": [
                {
                    "path": str(artifact.resolve()),
                    "sha256": "abc",
                    "unledgered_llm_attempts_for_checkpoint": 143,
                }
            ]
        },
        "source_artifacts": {
            "phase_c_pilot_result": {
                "path": str(artifact.resolve()),
                "sha256": "abc",
            }
        },
    }

    result = repair_ledger(audit, ledger_path)
    repaired = json.loads(ledger_path.read_text(encoding="utf-8"))

    assert result["repaired"] is True
    assert repaired["entries"][-1]["llm_attempts"] == 143
    assert repaired["cumulative_llm"] == 2787


def test_build_audit_checks_all_phase_c_checkpoints(tmp_path: Path):
    recovery = tmp_path / "outputs" / "m1m2_recovery"
    paper = tmp_path / "outputs" / "paper_level_evidence_package" / "geosym_bench_controlled"
    log = tmp_path / "outputs" / "autonomous_run_log"
    recovery.mkdir(parents=True)
    paper.mkdir(parents=True)
    log.mkdir(parents=True)

    (recovery / "m1_tier_discriminative_validity.json").write_text(
        json.dumps({
            "stage_status": "m1_tier_discriminative_validity_completed",
            "primary_240": {"prereg_verdict": "discriminative_validity_supported"},
        }),
        encoding="utf-8",
    )
    (recovery / "secondary_metric_variance_study.json").write_text(
        json.dumps({"stage_status": "phase_a_zero_cost_variance_study_completed"}),
        encoding="utf-8",
    )
    (recovery / "preregistration.json").write_text(
        json.dumps({"stage_status": "phase_b_preregistered_pending_phase_c_cap", "cap": {"pilot_llm_http_attempt_cap": 216}}),
        encoding="utf-8",
    )
    (recovery / "phase_c_dry_run.json").write_text(
        json.dumps({"stage_status": "phase_c_dry_run_passed"}),
        encoding="utf-8",
    )
    round1 = recovery / "phase_c_pilot_paid_result.json"
    round2 = recovery / "phase_c_pilot_repair_round_2_paid_result.json"
    round1.write_text(json.dumps({"stage_status": "partial_paid_pilot", "actual_llm_http_attempts": 213}), encoding="utf-8")
    round2.write_text(
        json.dumps({
            "stage_status": "running_paid_pilot",
            "actual_llm_http_attempts": 176,
            "completed_sample_quality_jobs": 16,
            "results": [{"status": "success", "chain_text_sha256": "a"}],
            "governance": {"chain_hash_distinct_gate": {"passes_nonempty_every_row": True, "passes_distinct_gt_0_9n": True}},
        }),
        encoding="utf-8",
    )
    ledger = {
        "entries": [
            {"artifact_path": str(round1.resolve()), "llm_attempts": 198},
            {"artifact_path": str(round2.resolve()), "llm_attempts": 176},
        ],
        "cumulative_llm": 2778,
        "cumulative_ocr": 23,
    }
    ledger_path = log / "budget_ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    import scripts.audit_evidence_selection_recovery as progress

    original_recovery = progress.RECOVERY_ROOT
    try:
        progress.RECOVERY_ROOT = recovery
        audit = build_audit(
            phase_c_result_path=round2,
            phase_c_result_paths=[round1, round2],
            ledger_path=ledger_path,
        )
    finally:
        progress.RECOVERY_ROOT = original_recovery

    assert "budget_ledger_missing_phase_c_attempts:phase_c_pilot_paid_result.json:15" in audit["blocking_findings"]
    assert audit["phase_c_summary"]["total_unledgered_llm_attempts_for_checkpoints"] == 15
