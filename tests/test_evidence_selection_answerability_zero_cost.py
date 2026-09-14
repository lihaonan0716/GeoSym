import json
from pathlib import Path

from scripts.build_evidence_selection_answerability_roster import build_roster
from scripts.project_evidence_selection_answerability_zero_cost import build_projection
from scripts.selective_propagation import facts_for_selective_condition


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def source_job(idx: int, family: str, expected: str) -> dict:
    return {
        "stress_id": f"selective_stress_{idx:03d}",
        "source_sample_id": f"s{idx:02d}",
        "perturbation_family": family,
        "expected_answerability": expected,
        "stress_schema": {
            "condition": "G_selective_schema",
            "expected_answerability": expected,
            "stress_hash": f"h{idx}",
            "evidence_candidates": [{"fact_id": f"f{idx}", "key": "measured_value", "reliability_tier": "high"}],
            "evidence_gaps": [
                {
                    "gap_id": f"g{idx}",
                    "key": "measured_value",
                    "needed_for_question": True,
                    "severity": "blocking",
                    "reason": "source_risk_gap",
                }
            ],
            "selected_evidence_set": {
                "selected_fact_ids": [f"f{idx}"],
                "suppressed_fact_ids": [f"low{idx}"],
                "surfaced_gap_ids": [f"g{idx}"],
                "propagation_policy": "answer_with_selected_facts" if expected == "answerable" else "abstain_insufficient_evidence",
            },
        },
    }


def source_roster() -> dict:
    families = [
        ("axis_notation_or_unit_ambiguity", "insufficient_evidence"),
        ("contradictory_duplicate_fact", "contradiction"),
        ("missing_required_fact", "insufficient_evidence"),
        ("ruler_or_axis_scale_perturbation", "insufficient_evidence"),
        ("series_or_legend_ambiguity", "insufficient_evidence"),
        ("spurious_low_confidence_fact", "answerable"),
    ]
    jobs = []
    idx = 0
    for family, expected in families:
        for _ in range(8):
            jobs.append(source_job(idx, family, expected))
            idx += 1
    return {"stage_status": "selective_stress_roster_frozen", "jobs": jobs}


def guarded_projection() -> dict:
    return {
        "stage_status": "plotqa_guardrail_preserved",
        "summary": {"G_full_guarded": {"correct": 48, "completed": 50, "parseable": 50}},
        "guardrail": {"passed": True, "protected_output_regressions": 0},
    }


def test_answerability_preserving_roster_hides_blocking_gaps_for_answerable_distractors(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    write_json(source_path, source_roster())

    result = build_roster(source_path, tmp_path / "roster")

    assert result["stage_status"] == "answerability_preserving_roster_frozen"
    answerable = [job for job in result["jobs"] if job["expected_answerability"] == "answerable"]
    assert len(answerable) == 8
    for job in answerable:
        facts = facts_for_selective_condition(job["stress_schema"], "G_selective_propagation", 20260830, job["stress_id"])
        assert facts["routing_policy"] == "answer_with_selected_facts"
        assert facts["required_fact_coverage_complete"] is True
        assert facts["blocking_gap_keys"] == []
        assert facts["evidence_gaps"] == []


def test_answerability_preserving_zero_cost_passes_plotqa_and_schema_gates(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    guarded_path = tmp_path / "guarded.json"
    write_json(source_path, source_roster())
    write_json(guarded_path, guarded_projection())
    roster = build_roster(source_path, tmp_path / "roster")
    roster_path = tmp_path / "roster" / "answerability_preserving_roster.json"
    assert roster["stage_status"] == "answerability_preserving_roster_frozen"

    result = build_projection(roster_path, guarded_path, tmp_path / "projection")

    assert result["stage_status"] == "ready_for_answerability_preserving_paid_manifest"
    assert result["paid_calls_executed"] == 0
    assert result["plotqa_guardrail"]["passed"] is True
    assert result["answerable_visible_blocking_gaps"] == 0
    assert result["gates"]["answerable_route_answer_with_selected_facts"] is True
