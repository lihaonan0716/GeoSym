import json
from pathlib import Path

from scripts.build_robustness_benchmark import build_stress_benchmark


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
            "propagation_policy": "use_directly",
        },
    }


def test_stress_benchmark_freezes_24_answer_independent_jobs(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    write_json(schema_path, {"stage_status": "ready_for_zero_cost_projection", "outputs": [schema_row(i) for i in range(50)]})

    result = build_stress_benchmark(schema_path, tmp_path / "out")

    assert result["stage_status"] == "stress_benchmark_frozen"
    assert result["paid_calls_executed"] == 0
    assert result["sample_level_job_count"] == 24
    assert result["perturbation_counts"] == {
        "axis_notation_ambiguity": 3,
        "missing_required_fact": 6,
        "ruler_perturbation": 4,
        "series_ambiguity": 4,
        "spurious_fact": 4,
        "visual_artifact_or_extrema_ambiguity": 3,
    }
    assert all(job["answer_independent_selection"] for job in result["jobs"])


def test_stress_benchmark_missing_fact_surfaces_blocking_gap(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    write_json(schema_path, {"outputs": [schema_row(i) for i in range(10)]})

    result = build_stress_benchmark(schema_path, tmp_path / "out")
    first = result["jobs"][0]["stress_schema"]

    assert first["selected_evidence_set"]["selected_fact_ids"] == []
    assert first["evidence_gaps"][-1]["reason"] == "stress_removed_required_fact"
    assert first["evidence_gaps"][-1]["severity"] == "blocking"
