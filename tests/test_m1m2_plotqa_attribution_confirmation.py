import json

import pytest

# Dev-local scripts (`scripts/*` is gitignored): skip rather than abort the
# suite's collection in a clean clone.
_analysis = pytest.importorskip(
    "scripts.analyze_m1m2_plotqa_attribution_confirmation")
_runner = pytest.importorskip(
    "scripts.run_m1m2_plotqa_attribution_confirmation")

CURATED_ARM = _analysis.CURATED_ARM
assemble = _analysis.assemble
exact_mcnemar_two_sided = _analysis.exact_mcnemar_two_sided
paired_stats = _analysis.paired_stats

EXPECTED_NEW_JOBS = _runner.EXPECTED_NEW_JOBS
LLM_ATTEMPT_CAP = _runner.LLM_ATTEMPT_CAP
build_plan = _runner.build_plan
partition_samples = _runner.partition_samples
review_plan = _runner.review_plan


def _cached_rows(solver_ids, llm_ids):
    rows = []
    for sample_id in solver_ids:
        rows.append({"sample_id": sample_id, "condition": "G_full_measured_geosym", "deterministic_m3": True, "correct": False})
    for sample_id in llm_ids:
        rows.append({"sample_id": sample_id, "condition": "G_full_measured_geosym", "deterministic_m3": False, "correct": True})
    for sample_id in list(solver_ids) + list(llm_ids):
        rows.append({"sample_id": sample_id, "condition": "G_measured_raw", "deterministic_m3": False, "correct": True})
    return rows


def test_partition_sends_only_solver_answered_samples_to_the_paid_arm() -> None:
    """Samples that already have a curated LLM answer must not be paid for again."""
    samples = [{"sample_id": "a"}, {"sample_id": "b"}, {"sample_id": "c"}]
    cached = {}
    for row in _cached_rows(["a"], ["b", "c"]):
        cached.setdefault(row["sample_id"], {})[row["condition"]] = row

    needs_paid, reusable = partition_samples(samples, cached)

    assert [s["sample_id"] for s in needs_paid] == ["a"]
    assert reusable == ["b", "c"]


def _write_fixture(tmp_path, *, sample_count=500, solver_count=EXPECTED_NEW_JOBS, prompt_drift=False, plan_mismatch=False):
    ids = [f"s{i:04d}" for i in range(sample_count)]
    solver_ids, llm_ids = ids[:solver_count], ids[solver_count:]
    source_plan = tmp_path / "source_plan.json"
    source_plan.write_text(
        json.dumps(
            {
                "benchmark": "PlotQA",
                "selected_samples": [
                    {"sample_id": i, "question": "q", "answer_type": "numeric", "image_path": "img.png", "normalized_answer": 1.0}
                    for i in ids
                ],
            }
        ),
        encoding="utf-8",
    )
    records = tmp_path / "records.json"
    records.write_text(
        json.dumps([{"sample_id": i, "question": "q", "answer_type": "numeric", "measured_points": [], "ruler": {"confidence": "high"}} for i in ids]),
        encoding="utf-8",
    )

    from scripts.run_plotqa_locate import prompt_hashes

    hashes = dict(prompt_hashes())
    if prompt_drift:
        hashes["full_measured"] = "drifted"
    cached_plan = tmp_path / "cached_plan.json"
    cached_plan.write_text(
        json.dumps(
            {
                "plan_sha256": "cachedplanhash",
                "model_provenance": {"display_name": "GPT-5.5", "api_model_id": "gpt-5.5"},
                "prompt_hashes": hashes,
            }
        ),
        encoding="utf-8",
    )
    cached_result = tmp_path / "cached_result.json"
    cached_result.write_text(
        json.dumps(
            {
                "plan_sha256": "other" if plan_mismatch else "cachedplanhash",
                "results": _cached_rows(solver_ids, llm_ids),
            }
        ),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"results": []}), encoding="utf-8")
    return source_plan, records, cached_result, baseline, cached_plan


def test_plan_is_approved_and_caps_only_the_missing_curated_rows(tmp_path) -> None:
    source_plan, records, cached_result, baseline, cached_plan = _write_fixture(tmp_path)

    plan = build_plan(source_plan, records, cached_result, baseline, tmp_path / "out", cached_plan)
    review = review_plan(plan)

    assert plan["paid_run_approved"] is True
    assert plan["dry_run"]["stop_reason"] is None
    assert len(plan["paid_sample_ids"]) == EXPECTED_NEW_JOBS
    assert len(plan["reused_curated_sample_ids"]) == 500 - EXPECTED_NEW_JOBS
    assert plan["llm_http_attempt_cap"] == LLM_ATTEMPT_CAP
    assert plan["ocr_http_attempt_cap"] == 0
    assert review["blocking_findings"] == []
    assert review["ready_for_paid_run"] is True


def test_plan_fails_closed_on_prompt_hash_drift(tmp_path) -> None:
    """A curated prompt change would make new rows incomparable to cached ones."""
    source_plan, records, cached_result, baseline, cached_plan = _write_fixture(tmp_path, prompt_drift=True)

    plan = build_plan(source_plan, records, cached_result, baseline, tmp_path / "out", cached_plan)

    assert plan["paid_run_approved"] is False
    assert plan["dry_run"]["stop_reason"] == "curated_prompt_hash_drift_vs_cached_run"


def test_plan_fails_closed_when_cached_plan_does_not_match_cached_result(tmp_path) -> None:
    source_plan, records, cached_result, baseline, cached_plan = _write_fixture(tmp_path, plan_mismatch=True)

    plan = build_plan(source_plan, records, cached_result, baseline, tmp_path / "out", cached_plan)

    assert plan["paid_run_approved"] is False
    assert plan["dry_run"]["stop_reason"] == "cached_plan_does_not_match_cached_result"


def test_plan_fails_closed_when_new_job_count_departs_from_protocol(tmp_path) -> None:
    source_plan, records, cached_result, baseline, cached_plan = _write_fixture(tmp_path, solver_count=EXPECTED_NEW_JOBS + 5)

    plan = build_plan(source_plan, records, cached_result, baseline, tmp_path / "out", cached_plan)

    assert plan["paid_run_approved"] is False
    assert plan["dry_run"]["stop_reason"].startswith("unexpected_new_job_count")


def test_assemble_rejects_a_solver_row_that_was_never_reanswered() -> None:
    plan = {"selected_ids": ["a"]}
    cached = {
        "a": {
            "G_full_measured_geosym": {"correct": True, "deterministic_m3": True},
            "G_measured_raw": {"correct": True},
        }
    }

    assembled, problems = assemble(plan, cached, {})

    assert assembled == []
    assert problems == ["solver_row_not_reanswered:a"]


def test_assemble_prefers_the_newly_paid_row_over_the_cached_solver_row() -> None:
    plan = {"selected_ids": ["a"]}
    cached = {
        "a": {
            "G_full_measured_geosym": {"correct": False, "deterministic_m3": True},
            "G_measured_raw": {"correct": False, "completed": True},
        }
    }
    new_rows = {"a": {"condition": CURATED_ARM, "correct": True, "completed": True}}

    assembled, problems = assemble(plan, cached, new_rows)

    assert problems == []
    assert assembled[0]["curated_correct"] is True
    assert assembled[0]["curated_source"] == "paid_new"


def test_exact_mcnemar_matches_known_values() -> None:
    # Symmetric discordance gives p = 1; a 39/18 split reproduces the observed subset p.
    assert exact_mcnemar_two_sided(5, 5) == 1.0
    assert round(exact_mcnemar_two_sided(39, 18), 5) == 0.00751
    assert exact_mcnemar_two_sided(0, 0) == 1.0


def test_paired_stats_reports_a_positive_ci_for_a_consistent_advantage() -> None:
    pairs = [(True, False)] * 30 + [(True, True)] * 60 + [(False, True)] * 10
    stats = paired_stats(pairs)

    assert stats["curated_correct"] == 90
    assert stats["raw_correct"] == 70
    assert stats["bootstrap_ci_95"][0] > 0
    assert stats["mcnemar_exact_two_sided_p"] < 0.05
