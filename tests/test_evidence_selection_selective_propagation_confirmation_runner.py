import json
from pathlib import Path

from scripts.run_evidence_selection_selective_propagation_confirmation import build_confirmation_dry_run, review_confirmation_manifest
from scripts.selective_propagation import SELECTIVE_CONDITIONS


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def families() -> list[str]:
    return [
        "axis_notation_or_unit_ambiguity",
        "contradictory_duplicate_fact",
        "missing_required_fact",
        "ruler_or_axis_scale_perturbation",
        "series_or_legend_ambiguity",
        "spurious_low_confidence_fact",
    ]


def expected_for_family(family: str) -> str:
    if family == "spurious_low_confidence_fact":
        return "answerable"
    if family == "contradictory_duplicate_fact":
        return "contradiction"
    return "insufficient_evidence"


def manifest_payload() -> dict:
    selected = []
    idx = 0
    for family in families():
        for _ in range(8):
            selected.append(
                {
                    "stress_id": f"selective_stress_{idx:03d}",
                    "source_sample_id": f"s{idx:02d}",
                    "perturbation_family": family,
                    "expected_answerability": expected_for_family(family),
                }
            )
            idx += 1
    return {
        "stage_status": "selective_confirmation_manifest_frozen_pending_approval",
        "paid_calls_executed": 0,
        "requires_explicit_paid_cap_approval_before_execution": True,
        "approval_required_for_llm_http_attempt_cap": 48 * len(SELECTIVE_CONDITIONS),
        "input_mode": "evidence_only",
        "sample_level_job_cap": 48,
        "condition_list": list(SELECTIVE_CONDITIONS),
        "llm_http_attempt_cap": 48 * len(SELECTIVE_CONDITIONS),
        "ocr_http_attempt_cap": 0,
        "retry_cap_per_sample_condition": 1,
        "max_output_tokens": 256,
        "estimated_cost_usd_upper_bound": 1.7,
        "perturbation_family_counts": {family: 8 for family in families()},
        "expected_answerability_counts": {"answerable": 8, "contradiction": 8, "insufficient_evidence": 32},
        "random_seed": 20260830,
        "selected_roster": selected,
    }


def stress_job(idx: int, family: str) -> dict:
    return {
        "stress_id": f"selective_stress_{idx:03d}",
        "source_sample_id": f"s{idx:02d}",
        "perturbation_family": family,
        "expected_answerability": expected_for_family(family),
        "stress_schema": {
            "condition": "G_selective_schema",
            "expected_answerability": expected_for_family(family),
            "stress_hash": f"h{idx}",
            "evidence_candidates": [{"fact_id": f"f{idx}", "reliability_tier": "high"}],
            "evidence_gaps": [{"gap_id": f"g{idx}", "key": "measured_value"}],
            "selected_evidence_set": {
                "selected_fact_ids": [f"f{idx}"],
                "surfaced_gap_ids": [f"g{idx}"],
                "propagation_policy": "answer_with_selected_facts" if family == "spurious_low_confidence_fact" else "abstain_insufficient_evidence",
            },
        },
    }


def test_confirmation_manifest_requires_explicit_paid_cap() -> None:
    assert review_confirmation_manifest(manifest_payload(), paid_cap_approved=True) == []
    assert f"explicit_{48 * len(SELECTIVE_CONDITIONS)}_paid_cap_approval_missing" in review_confirmation_manifest(manifest_payload(), paid_cap_approved=False)


def test_confirmation_dry_run_builds_expected_evidence_only_jobs(tmp_path: Path) -> None:
    jobs = []
    idx = 0
    for family in families():
        for _ in range(8):
            jobs.append(stress_job(idx, family))
            idx += 1
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.json"
    stress_path = tmp_path / "stress.json"
    write_json(manifest_path, manifest_payload())
    write_json(records_path, [{"sample_id": f"s{i:02d}", "question": "What is the value?", "answer_type": "numeric", "reference": 1} for i in range(48)])
    write_json(stress_path, {"jobs": jobs})

    result = build_confirmation_dry_run(manifest_path, records_path, stress_path, tmp_path / "out")

    assert result["stage_status"] == "selective_confirmation_dry_run_ready_pending_approval"
    assert result["paid_calls_executed"] == 0
    assert result["requires_explicit_paid_cap_approval_before_execution"] is True
    assert result["approved_paid_cap_for_this_dry_run"] is False
    assert result["job_count"] == 48 * len(SELECTIVE_CONDITIONS)
    assert all(job["input_mode"] if "input_mode" in job else True for job in result["jobs"])
