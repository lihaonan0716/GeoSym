"""Tests for the Stage D analysis module.

These functions produce numbers that go into reports, so a silent error here is
an integrity problem rather than a crash. The statistical helpers are checked
against closed-form values, and the weighting recomputation is checked against
the authoritative three-group scheme in config/default_config.yaml.
"""

from __future__ import annotations

import pytest

# The analysis script is dev-local (`scripts/*` is gitignored). Skip rather
# than fail collection: a clean clone has no such module, and an ImportError
# here aborts the whole suite instead of the one module that needs it.
_module = pytest.importorskip("scripts.analyze_glm46v_geosym_bench_240x7")

analysis_2_completion = _module.analysis_2_completion
analysis_8_layer_health = _module.analysis_8_layer_health
analysis_9_parse_failure_audit = _module.analysis_9_parse_failure_audit
bootstrap_ci = _module.bootstrap_ci
equal_weight_score = _module.equal_weight_score
governance_audit = _module.governance_audit
grouped_weight_score = _module.grouped_weight_score
wilson_interval = _module.wilson_interval


def _row(condition, **kwargs):
    row = {
        "sample_id": kwargs.pop("sample_id", "s0"),
        "condition": condition,
        "family": kwargs.pop("family", "linear"),
        "status": kwargs.pop("status", "success"),
        "total_score": kwargs.pop("total_score", 80.0),
        "reasoning_finish_reasons": kwargs.pop("reasoning_finish_reasons", {}),
        "empty_output_count": kwargs.pop("empty_output_count", 0),
        "ocr_http_attempts": 0,
        "parse_gates": kwargs.pop("parse_gates", {"eval_present": True}),
    }
    row.update(kwargs)
    return row


def test_wilson_interval_matches_known_values() -> None:
    # 0/10 successes: lower bound pinned at 0, upper bound the standard 0.2775.
    zero = wilson_interval(0, 10)
    assert zero["rate"] == 0.0
    assert zero["ci_low"] == 0.0
    assert zero["ci_high"] == pytest.approx(0.2775, abs=1e-3)

    half = wilson_interval(50, 100)
    assert half["rate"] == 0.5
    assert half["ci_low"] == pytest.approx(0.4038, abs=1e-3)
    assert half["ci_high"] == pytest.approx(0.5962, abs=1e-3)


def test_wilson_interval_is_bounded_and_safe_on_empty_input() -> None:
    full = wilson_interval(10, 10)
    assert full["ci_high"] == 1.0
    assert wilson_interval(0, 0)["rate"] is None


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic() -> None:
    values = [float(v) for v in range(1, 21)]

    first = bootstrap_ci(values)
    second = bootstrap_ci(values)

    assert first == second  # fixed seed: reruns must reproduce the report
    assert first["mean"] == pytest.approx(10.5)
    assert first["ci_low"] < 10.5 < first["ci_high"]


def test_bootstrap_ci_flags_whether_zero_is_excluded() -> None:
    assert bootstrap_ci([5.0] * 30)["ci_excludes_zero"] is True
    assert bootstrap_ci([-5.0] * 30)["ci_excludes_zero"] is True
    # A contrast straddling zero must not be reported as a real effect.
    assert bootstrap_ci([-10.0, 10.0] * 20)["ci_excludes_zero"] is False


def test_bootstrap_ci_handles_empty_input() -> None:
    assert bootstrap_ci([])["n"] == 0
    assert bootstrap_ci([])["mean"] is None


def test_cluster_bootstrap_widens_the_interval_versus_row_resampling() -> None:
    """Samples in a family share a difficulty profile; ignoring that understates variance."""
    values = [0.0] * 20 + [40.0] * 20
    families = ["a"] * 20 + ["b"] * 20

    naive = bootstrap_ci(values)
    clustered = bootstrap_ci(values, clusters=families)

    assert clustered["cluster_bootstrap"] is True
    naive_width = naive["ci_high"] - naive["ci_low"]
    clustered_width = clustered["ci_high"] - clustered["ci_low"]
    assert clustered_width > naive_width


def test_grouped_weight_score_matches_the_authoritative_yaml_scheme() -> None:
    """G1 0.30 / G2 0.30 / G3 0.40, with within-group weights. Not equal weights."""
    dims = {
        "intercepts": 100.0,
        "extrema": 100.0,
        "predictions": 0.0,
        "rate_of_change": 0.0,
        "monotonicity": 0.0,
        "symmetry": 0.0,
        "expression": 0.0,
        "domain": 0.0,
        "range": 0.0,
        "judge_inference_score": 0.0,
    }

    # Only G1 scores, and G1 carries 0.30 of the total.
    assert grouped_weight_score({"dimension_scores": dims}) == pytest.approx(30.0)
    # Equal weighting would give 2 of 10 dimensions at 100 = 20.0, which differs.
    assert equal_weight_score({"dimension_scores": dims}) == pytest.approx(20.0)


def test_grouped_and_equal_weighting_agree_only_when_all_dimensions_are_equal() -> None:
    dims = {k: 75.0 for k in
            ["intercepts", "extrema", "predictions", "rate_of_change", "monotonicity",
             "symmetry", "expression", "domain", "range", "judge_inference_score"]}

    assert grouped_weight_score({"dimension_scores": dims}) == pytest.approx(75.0)
    assert equal_weight_score({"dimension_scores": dims}) == pytest.approx(75.0)


def test_weight_helpers_return_none_without_dimension_scores() -> None:
    assert grouped_weight_score({"dimension_scores": None}) is None
    assert equal_weight_score({}) is None


def test_completion_reports_attempted_alongside_success() -> None:
    rows = [
        _row("E5", sample_id="s0"),
        _row("E5", sample_id="s1", status="pipeline_error", total_score=None,
             parse_gates={"eval_present": False}, error_reason="Request timed out."),
    ]

    stats = analysis_2_completion(rows, roster_size=4)["E5"]

    assert stats["attempted"] == 2
    assert stats["roster_size"] == 4
    assert stats["attempted_rate"] == 0.5  # never hide the roster denominator
    assert stats["success"] == 1
    assert stats["parseable"] == 1
    assert stats["errors"] == 1


def test_layer_health_separates_per_call_and_per_row_truncation() -> None:
    rows = [
        _row("E5", sample_id=f"s{i}",
             reasoning_finish_reasons={"L1": ["stop"], "L2": ["stop"], "L3": ["length"]},
             empty_output_steps=["L3"], empty_output_count=1)
        for i in range(4)
    ] + [
        _row("E5", sample_id=f"s{i}",
             reasoning_finish_reasons={"L1": ["stop"], "L2": ["stop"], "L3": ["stop"]})
        for i in range(4, 10)
    ]

    health = analysis_8_layer_health(rows)["E5"]

    assert health["semantic_calls"] == 30
    assert health["truncated_calls"] == 4
    # Per-call dilutes 40% down to 13.3% by averaging in two healthy layers.
    # Rates are rounded to 6dp by the module, so compare at that precision.
    assert health["truncation_rate_calls"] == pytest.approx(4 / 30, abs=1e-6)
    assert health["truncation_rate_rows"] == pytest.approx(0.4, abs=1e-6)
    assert health["by_step"]["L3"]["length_rate"] == pytest.approx(0.4, abs=1e-6)
    assert health["empty_output_calls"] == 4


def test_parse_failure_audit_assigns_root_cause_categories() -> None:
    rows = [
        _row("E1", error_reason="Request timed out."),
        _row("E3", error_reason="paddleocr_insufficient_tick_labels:x=1:y=10"),
        _row("E4", error_reason="paddleocr_rule_baseline_no_coordinates"),
        _row("E5", error_reason="clean_grid_unavailable"),
        _row("E0", error_reason="something else entirely"),
    ]

    audit = analysis_9_parse_failure_audit(rows)

    assert audit["total_error_rows"] == 5
    assert set(audit["by_category"]) == {
        "transport_timeout",
        "ocr_insufficient_labels",
        "ocr_backend_failure",
        "m0_clean_grid_unavailable",
        "other",
    }


def test_governance_audit_flags_protocol_violations() -> None:
    manifest = {
        "plan_sha256": "abc",
        "caps": {"llm_http_attempt_cap": 100},
        "reasoning_model_provenance": {"api_model_id": "glm-4.6v", "max_tokens": 16384},
        "judge_model_provenance": {"api_model_id": "glm-4.6v"},  # same as reasoning
        "selected_ids": ["s0"],
    }
    result = {
        "plan_sha256": "different",
        "actual_llm_http_attempts": 150,
        "results": [_row("E5", sample_id="s0", ocr_http_attempts=2)],
    }

    audit = governance_audit(result, manifest)

    assert audit["status"] == "blocking_findings_present"
    assert "judge_not_independent" in audit["blocking_findings"]
    assert "plan_sha256_mismatch" in audit["blocking_findings"]
    assert any(f.startswith("ocr_http_attempts_nonzero") for f in audit["blocking_findings"])
    assert any(f.startswith("llm_attempts_exceed_cap") for f in audit["blocking_findings"])


def test_governance_audit_passes_a_clean_run() -> None:
    manifest = {
        "plan_sha256": "abc",
        "caps": {"llm_http_attempt_cap": 5040},
        "reasoning_model_provenance": {"api_model_id": "glm-4.6v", "max_tokens": 16384},
        "judge_model_provenance": {"api_model_id": "qwen3-vl-plus"},
        "selected_ids": ["s0"],
    }
    result = {
        "plan_sha256": "abc",
        "actual_llm_http_attempts": 21,
        "results": [_row("E5", sample_id="s0")],
    }

    assert governance_audit(result, manifest)["status"] == "no_blocking_findings"


def test_governance_audit_rejects_a_stale_4096_token_manifest() -> None:
    manifest = {
        "plan_sha256": "abc",
        "caps": {"llm_http_attempt_cap": 5040},
        "reasoning_model_provenance": {"api_model_id": "glm-4.6v", "max_tokens": 4096},
        "judge_model_provenance": {"api_model_id": "qwen3-vl-plus"},
        "selected_ids": ["s0"],
    }
    result = {"plan_sha256": "abc", "actual_llm_http_attempts": 21, "results": []}

    assert "unexpected_max_tokens:4096" in governance_audit(result, manifest)["blocking_findings"]


def test_governance_audit_detects_rows_outside_the_frozen_roster() -> None:
    manifest = {
        "plan_sha256": "abc",
        "caps": {"llm_http_attempt_cap": 5040},
        "reasoning_model_provenance": {"api_model_id": "glm-4.6v", "max_tokens": 16384},
        "judge_model_provenance": {"api_model_id": "qwen3-vl-plus"},
        "selected_ids": ["s0"],
    }
    result = {
        "plan_sha256": "abc",
        "actual_llm_http_attempts": 21,
        "results": [_row("E5", sample_id="not-in-roster")],
    }

    assert "rows_outside_frozen_roster" in governance_audit(result, manifest)["blocking_findings"]
