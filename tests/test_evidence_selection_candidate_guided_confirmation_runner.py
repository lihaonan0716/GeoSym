import json
from pathlib import Path

from scripts.freeze_evidence_selection_candidate_guided_confirmation_manifest import load_default_model_provenance
from scripts.run_evidence_selection_candidate_guided_confirmation import build_dry_run, review_confirmation_manifest
from scripts.selective_propagation import SELECTIVE_CONDITIONS


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def manifest() -> dict:
    selected = [
        {
            "stress_id": f"candidate_guided_confirmation_{idx:03d}",
            "source_sample_id": f"s{idx:02d}",
            "perturbation_family": "spurious_low_confidence_fact" if idx < 8 else "missing_required_fact",
            "expected_answerability": "answerable" if idx < 8 else "insufficient_evidence",
        }
        for idx in range(16)
    ]
    return {
        "stage_status": "candidate_guided_confirmation_manifest_frozen_pending_paid_cap_approval",
        "redesign_name": "candidate_guided_selective_propagation_v3_1",
        "paid_calls_executed": 0,
        "input_mode": "evidence_only",
        "sample_level_job_cap": 16,
        "condition_list": list(SELECTIVE_CONDITIONS),
        "llm_http_attempt_cap": 16 * len(SELECTIVE_CONDITIONS),
        "ocr_http_attempt_cap": 0,
        "retry_cap_per_sample_condition": 1,
        "max_output_tokens": 256,
        "safe_default": "do_not_execute_without_explicit_paid_cap_approval",
        "expected_answerability_counts": {"answerable": 8, "insufficient_evidence": 8},
        "random_seed": 20260830,
        "selected_roster": selected,
    }


def stress_job(idx: int) -> dict:
    return {
        "stress_id": f"candidate_guided_confirmation_{idx:03d}",
        "source_sample_id": f"s{idx:02d}",
        "perturbation_family": "spurious_low_confidence_fact" if idx < 8 else "missing_required_fact",
        "expected_answerability": "answerable" if idx < 8 else "insufficient_evidence",
        "stress_schema": {
            "condition": "G_full_guarded_schema_only",
            "expected_answerability": "answerable" if idx < 8 else "insufficient_evidence",
            "stress_hash": f"h{idx}",
            "evidence_candidates": [{"fact_id": f"f{idx}", "reliability_tier": "high"}],
            "evidence_gaps": [{"gap_id": f"g{idx}", "key": "measured_value", "needed_for_question": True, "severity": "blocking"}],
            "selected_evidence_set": {
                "selected_fact_ids": [f"f{idx}"],
                "suppressed_fact_ids": [f"suppressed{idx}"],
                "surfaced_gap_ids": [] if idx < 8 else [f"g{idx}"],
                "blocking_gap_keys": [] if idx < 8 else ["measured_value"],
                "propagation_policy": "answer_with_candidate_and_selected_facts" if idx < 8 else "abstain_insufficient_evidence",
                "answer_candidate": {"normalized_prediction": 1, "supporting_fact_ids": [f"f{idx}"]} if idx < 8 else None,
            },
        },
    }


def test_review_confirmation_manifest_requires_explicit_paid_cap() -> None:
    payload = manifest()

    assert review_confirmation_manifest(payload) == []
    assert "approved_llm_cap_missing_or_mismatch" in review_confirmation_manifest(payload, for_paid=True)
    assert review_confirmation_manifest(payload, approved_llm_cap=16 * len(SELECTIVE_CONDITIONS), for_paid=True) == []


def test_confirmation_dry_run_builds_expected_jobs_without_paid_calls(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.json"
    roster_path = tmp_path / "roster.json"
    write_json(manifest_path, manifest())
    write_json(records_path, [{"sample_id": f"s{i:02d}", "question": "What is the value?", "answer_type": "numeric", "reference": 1} for i in range(16)])
    write_json(roster_path, {"jobs": [stress_job(i) for i in range(16)]})

    result = build_dry_run(manifest_path, records_path, roster_path, tmp_path / "out")

    assert result["stage_status"] == "candidate_guided_confirmation_dry_run_passed"
    assert result["paid_calls_executed"] == 0
    assert result["actual_llm_http_attempts"] == 0
    assert result["job_count"] == 16 * len(SELECTIVE_CONDITIONS)


def test_default_model_provenance_is_read_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "default_config.yaml"
    config_path.write_text(
        """
models:
  Example:
    adapter: "evaluator.adapters.openai.OpenAIAdapter"
    api_key_name: "example"
    models:
      "Config Model": "config-model-id"
""".strip(),
        encoding="utf-8",
    )

    provenance = load_default_model_provenance(config_path)

    assert provenance["provider"] == "Example"
    assert provenance["model_display_name"] == "Config Model"
    assert provenance["model_registry_id_or_config_value"] == "config-model-id"
    assert provenance["config_sha256"]
