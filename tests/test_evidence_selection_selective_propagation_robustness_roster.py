import json
from pathlib import Path

from scripts.build_evidence_selection_selective_propagation_robustness_roster import build_roster


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def schema_row(idx: int) -> dict:
    return {
        "sample_id": f"s{idx:02d}",
        "condition": "G_full_guarded_schema_only",
        "question_hash": f"q{idx}",
        "evidence_candidates": [
            {
                "fact_id": f"f{idx}",
                "key": "measured_value",
                "value": {"x_value": idx, "y_value": idx + 1},
                "reliability_tier": "medium",
                "status": "present",
            }
        ],
        "evidence_gaps": [],
        "selected_evidence_set": {
            "selected_fact_ids": [f"f{idx}"],
            "suppressed_fact_ids": [],
            "surfaced_gap_ids": [],
            "propagation_policy": "answer_with_selected_facts",
        },
    }


def test_selective_stress_roster_freezes_48_jobs_and_oracle(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    write_json(schema_path, {"stage_status": "ready_for_zero_cost_projection", "outputs": [schema_row(i) for i in range(50)]})

    result = build_roster(schema_path, tmp_path / "out")

    assert result["stage_status"] == "selective_stress_roster_frozen"
    assert result["paid_calls_executed"] == 0
    assert result["sample_level_job_count"] == 48
    assert result["perturbation_family_counts"] == {
        "axis_notation_or_unit_ambiguity": 8,
        "contradictory_duplicate_fact": 8,
        "missing_required_fact": 8,
        "ruler_or_axis_scale_perturbation": 8,
        "series_or_legend_ambiguity": 8,
        "spurious_low_confidence_fact": 8,
    }
    assert result["expected_answerability_counts"] == {
        "answerable": 8,
        "contradiction": 8,
        "insufficient_evidence": 32,
    }
    assert all(job["answer_independent_selection"] for job in result["jobs"])


def test_selective_missing_fact_removes_selected_fact_and_sets_abstention(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    write_json(schema_path, {"outputs": [schema_row(i) for i in range(50)]})

    result = build_roster(schema_path, tmp_path / "out")
    first = result["jobs"][0]["stress_schema"]

    assert first["expected_answerability"] == "insufficient_evidence"
    assert first["selected_evidence_set"]["selected_fact_ids"] == []
    assert first["selected_evidence_set"]["propagation_policy"] == "abstain_insufficient_evidence"
