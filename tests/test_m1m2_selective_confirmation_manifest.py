import json
from pathlib import Path

from scripts.freeze_m1m2_selective_confirmation_manifest import build_manifest
from scripts.selective_propagation import SELECTIVE_CONDITIONS


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def roster_payload() -> dict:
    families = [
        "axis_notation_or_unit_ambiguity",
        "contradictory_duplicate_fact",
        "missing_required_fact",
        "ruler_or_axis_scale_perturbation",
        "series_or_legend_ambiguity",
        "spurious_low_confidence_fact",
    ]
    jobs = []
    idx = 0
    for family in families:
        for _ in range(8):
            expected = "answerable" if family == "spurious_low_confidence_fact" else "insufficient_evidence"
            if family == "contradictory_duplicate_fact":
                expected = "contradiction"
            jobs.append(
                {
                    "stress_id": f"s{idx:02d}",
                    "source_sample_id": f"sample{idx:02d}",
                    "perturbation_family": family,
                    "expected_answerability": expected,
                    "stress_schema": {
                        "condition": "G_selective_schema",
                        "expected_answerability": expected,
                        "stress_hash": f"h{idx}",
                        "evidence_candidates": [{"fact_id": f"f{idx}", "reliability_tier": "high"}],
                        "evidence_gaps": [],
                        "selected_evidence_set": {"selected_fact_ids": [f"f{idx}"], "surfaced_gap_ids": [], "propagation_policy": "answer_with_selected_facts"},
                    },
                }
            )
            idx += 1
    return {"stage_status": "selective_stress_roster_frozen", "jobs": jobs}


def test_confirmation_manifest_freezes_288_cap_pending_approval(tmp_path: Path) -> None:
    roster_path = tmp_path / "roster.json"
    pilot_path = tmp_path / "pilot.json"
    zero_path = tmp_path / "zero.json"
    records_path = tmp_path / "records.json"
    write_json(roster_path, roster_payload())
    write_json(pilot_path, {"stage_status": "m1m2_selective_propagation_supported"})
    write_json(zero_path, {"stage_status": "ready_for_paid_manifest", "plotqa_guardrail": {"passed": True}})
    write_json(records_path, [{"sample_id": f"sample{i:02d}", "question": "What is the value?"} for i in range(48)])

    result = build_manifest(roster_path, pilot_path, zero_path, records_path, tmp_path / "out", "GPT-5.5")

    assert result["stage_status"] == "selective_confirmation_manifest_frozen_pending_approval"
    assert result["paid_calls_executed"] == 0
    assert result["requires_explicit_paid_cap_approval_before_execution"] is True
    assert result["sample_level_job_cap"] == 48
    assert result["llm_http_attempt_cap"] == 48 * len(SELECTIVE_CONDITIONS)
    assert result["ocr_http_attempt_cap"] == 0
    assert all(count == 8 for count in result["perturbation_family_counts"].values())
    assert result["estimated_cost_usd_upper_bound"] > 0
