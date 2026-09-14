import json
from pathlib import Path

from scripts.run_evidence_selection_selective_propagation_pilot import build_dry_run, review_manifest
from scripts.selective_propagation import SELECTIVE_CONDITIONS


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def manifest() -> dict:
    families = [
        "axis_notation_or_unit_ambiguity",
        "contradictory_duplicate_fact",
        "missing_required_fact",
        "ruler_or_axis_scale_perturbation",
        "series_or_legend_ambiguity",
        "spurious_low_confidence_fact",
    ]
    selected = []
    idx = 0
    for family in families:
        for _ in range(2):
            selected.append(
                {
                    "stress_id": f"selective_stress_{idx:03d}",
                    "source_sample_id": f"s{idx:02d}",
                    "perturbation_family": family,
                    "expected_answerability": "answerable" if family == "spurious_low_confidence_fact" else "insufficient_evidence",
                }
            )
            idx += 1
    return {
        "stage_status": "selective_paid_pilot_manifest_frozen",
        "paid_calls_executed": 0,
        "input_mode": "evidence_only",
        "sample_level_job_cap": 12,
        "condition_list": list(SELECTIVE_CONDITIONS),
        "llm_http_attempt_cap": 12 * len(SELECTIVE_CONDITIONS),
        "ocr_http_attempt_cap": 0,
        "retry_cap_per_sample_condition": 1,
        "max_output_tokens": 256,
        "estimated_cost_usd_upper_bound": 0.1,
        "perturbation_family_counts": {family: 2 for family in families},
        "random_seed": 20260830,
        "selected_roster": selected,
    }


def stress_job(idx: int, family: str) -> dict:
    return {
        "stress_id": f"selective_stress_{idx:03d}",
        "source_sample_id": f"s{idx:02d}",
        "perturbation_family": family,
        "expected_answerability": "answerable" if family == "spurious_low_confidence_fact" else "insufficient_evidence",
        "stress_schema": {
            "condition": "G_selective_schema",
            "expected_answerability": "answerable" if family == "spurious_low_confidence_fact" else "insufficient_evidence",
            "stress_hash": f"h{idx}",
            "evidence_candidates": [{"fact_id": f"f{idx}", "reliability_tier": "high"}],
            "evidence_gaps": [{"gap_id": f"g{idx}", "key": "measured_value"}],
            "selected_evidence_set": {
                "selected_fact_ids": [f"f{idx}"],
                "surfaced_gap_ids": [f"g{idx}"],
                "propagation_policy": "abstain_insufficient_evidence",
            },
        },
    }


def test_review_manifest_enforces_selective_caps() -> None:
    assert review_manifest(manifest()) == []
    bad = manifest()
    bad["max_output_tokens"] = 4096
    assert "max_output_tokens_not_256" in review_manifest(bad)


def test_review_manifest_accepts_answerability_preserving_v2_sampling() -> None:
    payload = manifest()
    payload["redesign_name"] = "answerability_preserving_selective_propagation_v2"
    payload["perturbation_family_counts"] = {
        "axis_notation_or_unit_ambiguity": 2,
        "contradictory_duplicate_fact": 1,
        "missing_required_fact": 1,
        "ruler_or_axis_scale_perturbation": 1,
        "series_or_legend_ambiguity": 1,
        "spurious_low_confidence_fact": 6,
    }
    payload["expected_answerability_counts"] = {"answerable": 6, "contradiction": 1, "insufficient_evidence": 5}

    assert review_manifest(payload) == []


def test_review_manifest_accepts_candidate_guided_v31_sampling() -> None:
    payload = manifest()
    payload["redesign_name"] = "candidate_guided_selective_propagation_v3_1"
    payload["perturbation_family_counts"] = {
        "axis_notation_or_unit_ambiguity": 2,
        "contradictory_duplicate_fact": 1,
        "missing_required_fact": 1,
        "ruler_or_axis_scale_perturbation": 1,
        "series_or_legend_ambiguity": 1,
        "spurious_low_confidence_fact": 6,
    }
    payload["expected_answerability_counts"] = {"answerable": 6, "contradiction": 1, "insufficient_evidence": 5}

    assert review_manifest(payload) == []


def test_selective_dry_run_builds_expected_evidence_only_jobs(tmp_path: Path) -> None:
    families = list(manifest()["perturbation_family_counts"].keys())
    jobs = []
    idx = 0
    for family in families:
        for _ in range(2):
            jobs.append(stress_job(idx, family))
            idx += 1
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.json"
    stress_path = tmp_path / "stress.json"
    write_json(manifest_path, manifest())
    write_json(records_path, [{"sample_id": f"s{i:02d}", "question": "What is the value?", "answer_type": "numeric", "reference": 1} for i in range(12)])
    write_json(stress_path, {"jobs": jobs})

    result = build_dry_run(manifest_path, records_path, stress_path, tmp_path / "out")

    assert result["stage_status"] == "selective_paid_pilot_dry_run_passed"
    assert result["paid_calls_executed"] == 0
    assert result["job_count"] == 12 * len(SELECTIVE_CONDITIONS)
    assert all(job["condition"].startswith("G_") for job in result["jobs"])
