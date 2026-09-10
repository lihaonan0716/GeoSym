import json
from pathlib import Path

from scripts.run_m1m2_paid_pilot import build_dry_run, facts_for_condition, review_manifest


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def manifest() -> dict:
    return {
        "stage_status": "paid_pilot_manifest_frozen",
        "paid_calls_executed": 0,
        "sample_level_job_cap": 8,
        "conditions": ["G_full_guarded", "G_all_evidence", "G_top_confidence", "G_random_k"],
        "llm_http_attempt_cap": 32,
        "ocr_http_attempt_cap": 0,
        "random_seed": 20260830,
        "selected_roster": [{"sample_id": f"s{i:02d}"} for i in range(8)],
    }


def schema_row(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "condition": "G_full_guarded_schema_only",
        "evidence_candidates": [
            {"fact_id": "low", "reliability_tier": "low"},
            {"fact_id": "high", "reliability_tier": "high"},
            {"fact_id": "medium", "reliability_tier": "medium"},
        ],
        "evidence_gaps": [{"gap_id": "gap0", "reason": "not_detected"}],
        "selected_evidence_set": {
            "selected_fact_ids": ["medium"],
            "suppressed_fact_ids": ["high", "low"],
            "surfaced_gap_ids": ["gap0"],
            "propagation_policy": "use_with_caution",
        },
    }


def test_review_manifest_enforces_frozen_caps() -> None:
    assert review_manifest(manifest()) == []
    bad = manifest()
    bad["llm_http_attempt_cap"] = 31
    assert "llm_cap_not_32" in review_manifest(bad)


def test_dry_run_writes_32_jobs_and_no_paid_calls(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.json"
    schema_path = tmp_path / "schema.json"
    write_json(manifest_path, manifest())
    write_json(records_path, [{"sample_id": f"s{i:02d}"} for i in range(8)])
    write_json(schema_path, {"outputs": [schema_row(f"s{i:02d}") for i in range(8)]})

    result = build_dry_run(manifest_path, records_path, schema_path, tmp_path / "out")

    assert result["stage_status"] == "paid_pilot_dry_run_passed"
    assert result["paid_calls_executed"] == 0
    assert len(result["jobs"]) == 32
    assert all(job["would_execute_paid_call"] for job in result["jobs"])


def test_facts_for_condition_uses_matched_budget() -> None:
    row = schema_row("s0")

    guarded = facts_for_condition(row, "G_full_guarded", 20260830)
    all_evidence = facts_for_condition(row, "G_all_evidence", 20260830)
    top = facts_for_condition(row, "G_top_confidence", 20260830)
    random_k = facts_for_condition(row, "G_random_k", 20260830)

    assert [fact["fact_id"] for fact in guarded["measured_facts"]] == ["medium"]
    assert [fact["fact_id"] for fact in all_evidence["measured_facts"]] == ["low", "high", "medium"]
    assert [fact["fact_id"] for fact in top["measured_facts"]] == ["high"]
    assert len(random_k["measured_facts"]) == 1
    assert guarded["evidence_gaps"] == [{"gap_id": "gap0", "reason": "not_detected"}]
    assert all_evidence["evidence_gaps"] == []
