import json
from pathlib import Path

from scripts.freeze_evidence_selection_selective_propagation_pilot_manifest import build_manifest
from scripts.selective_propagation import SELECTIVE_CONDITIONS


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


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
            "evidence_gaps": [],
            "selected_evidence_set": {"selected_fact_ids": [f"f{idx}"], "surfaced_gap_ids": [], "propagation_policy": "answer_with_selected_facts"},
        },
    }


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
            jobs.append(stress_job(idx, family))
            idx += 1
    return {"stage_status": "selective_stress_roster_frozen", "jobs": jobs}


# NOTE (2026-09-19): `test_zero_cost_projection_preserves_plotqa_guardrail` was
# removed with its module, `project_evidence_selection_selective_propagation_
# zero_cost.py` -- a pre-paid projection for the M1/M2 rescue that retired with
# them. The manifest test below is kept: it covers the roster freeze that still
# reproduces the 896 paid stress_128 rows.


def test_paid_manifest_selects_two_per_family_and_records_cost(tmp_path: Path) -> None:
    roster_path = tmp_path / "roster.json"
    zero_path = tmp_path / "zero.json"
    records_path = tmp_path / "records.json"
    write_json(roster_path, roster_payload())
    write_json(zero_path, {"stage_status": "ready_for_paid_manifest", "plotqa_guardrail": {"passed": True}})
    write_json(records_path, [{"sample_id": f"s{i:02d}", "question": "What is the value?"} for i in range(48)])

    result = build_manifest(roster_path, zero_path, records_path, tmp_path / "out", "GPT-5.5")

    assert result["stage_status"] == "selective_paid_pilot_manifest_frozen"
    assert result["paid_calls_executed"] == 0
    assert result["sample_level_job_cap"] == 12
    assert result["llm_http_attempt_cap"] == 12 * len(SELECTIVE_CONDITIONS)
    assert result["ocr_http_attempt_cap"] == 0
    assert all(count == 2 for count in result["perturbation_family_counts"].values())
    assert result["estimated_cost_usd_upper_bound"] > 0
