from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# The replication under test is the completed seven-arm run (E5 included),
# so its fixtures must carry seven conditions to match the gate arithmetic.
from scripts.batch_experiment import HISTORICAL_CONDITIONS as ALL_CONDITIONS_FOR_TEST

#: The paid runner is dev-local (`scripts/*` is gitignored) and is imported
#: inside individual test bodies below, so a module-level `importorskip` would
#: not cover them -- and skipping the whole module would also hide the six
#: tests here that exercise the adapters and the OCR baseline, which are public
#: code a clean clone must still check. Mark only the tests that need it.
requires_runner = pytest.mark.skipif(
    importlib.util.find_spec("scripts.run_glm46v_geosym_bench_240x7") is None,
    reason="paid runner is dev-local; absent in a clean clone")

from evaluator.adapters.glm import GLMAdapter
from evaluator.adapters.qwen import QwenVLAdapter
from evaluator.baselines.paddleocr_rule_baseline import (
    PaddleOCRBaselineError,
    PaddleOCRRuleBaseline,
    labels_from_paddle_payload,
)


class _Message:
    content = '{"ok": true}'


class _Choice:
    finish_reason = "length"
    message = _Message()


class _Response:
    id = "resp-test"
    model = "stub-model"
    choices = [_Choice()]


class _Completions:
    def create(self, **kwargs):
        return _Response()


class _Chat:
    completions = _Completions()


class _Client:
    chat = _Chat()


def test_paddleocr_canonical_integer_tokens_reject_observed_malformed_values() -> None:
    payload = {
        "rec_texts": ["-2", "0", "3", "12.0", "01", "012", "0123", "000000000000000", "4.5"],
        "rec_polys": [[[0, 0], [20, 0], [20, 10], [0, 10]]] * 9,
    }

    labels = labels_from_paddle_payload(payload, factor=1)

    assert [label["value"] for label in labels] == [-2, 0, 3]


def test_paddleocr_polygon_to_center_rescales_to_original_pixels() -> None:
    payload = {
        "rec_texts": ["5"],
        "rec_polys": [[[20, 40], [60, 40], [60, 80], [20, 80]]],
    }

    labels = labels_from_paddle_payload(payload, factor=2)

    assert labels == [{"value": 5, "px": 20, "py": 30, "bw": 20, "bh": 20}]


def test_paddleocr_fail_closed_on_insufficient_axis_labels() -> None:
    class SparsePaddleOCR(PaddleOCRRuleBaseline):
        def _run_ocr(self, img, *, upscale=True):
            return [{"value": 0, "px": 10, "py": 10, "bw": 5, "bh": 5}]

    baseline = SparsePaddleOCR()
    img = np.ones((64, 64, 3), dtype=np.uint8) * 255

    with pytest.raises(PaddleOCRBaselineError, match="paddleocr_insufficient_tick_labels"):
        baseline._ocr_tick_labels(img)


def test_glm_adapter_captures_finish_reason_and_returns_string() -> None:
    adapter = GLMAdapter(api_key="test", model="glm-4.6v")
    adapter._client = _Client()

    raw = adapter.call_with_prompt(np.ones((1, 1, 3), dtype=np.uint8), "prompt")

    assert isinstance(raw, str)
    assert adapter.last_finish_reason == "length"
    assert adapter.last_response_metadata["finish_reason"] == "length"


def test_qwen_adapter_captures_finish_reason_and_returns_string() -> None:
    adapter = QwenVLAdapter(api_key="test", model="qwen3-vl-plus")
    adapter._client = _Client()

    raw = adapter.call_with_prompt(np.ones((1, 1, 3), dtype=np.uint8), "prompt")

    assert isinstance(raw, str)
    assert adapter.last_finish_reason == "length"
    assert adapter.last_response_metadata["finish_reason"] == "length"


@requires_runner
def test_clean_grid_only_m0_fails_closed_without_mathpix(monkeypatch) -> None:
    import evaluator.adapters.recognizer as recognizer_module
    from scripts.run_glm46v_geosym_bench_240x7 import clean_grid_only_m0

    def failed_clean_grid(image):
        class Result:
            success = False
            coords = []
            result = {"success": False, "error": "clean_grid_structure_not_detected"}

        return Result()

    def forbidden_mathpix_path(self, image, gt=None, skip_enhancement=True, gemini_enabled=False):
        raise AssertionError("Mathpix path should not be reached")

    monkeypatch.setattr(
        "recognizer.core.clean_grid_fallback.extract_clean_cartesian_grid",
        failed_clean_grid,
    )
    monkeypatch.setattr(
        recognizer_module.RecognizerAdapter,
        "recognize_intersections",
        forbidden_mathpix_path,
    )

    with clean_grid_only_m0():
        adapter = recognizer_module.RecognizerAdapter()
        coords, result, annotated = adapter.recognize_intersections(np.ones((16, 16, 3), dtype=np.uint8))

    assert coords == []
    assert annotated is None
    assert result["error"] == "clean_grid_unavailable"


@requires_runner
def test_run_reasoning_all_conditions_default_signature_preserves_mathpix_path() -> None:
    import inspect

    from app.reasoning import run_reasoning_all_conditions

    parameter = inspect.signature(run_reasoning_all_conditions).parameters["ocr_baseline"]
    assert parameter.default is None


# ── Re-run fixes F1-F4 ──────


def _manifest_stub(sample_ids):
    from scripts.run_glm46v_geosym_bench_240x7 import GLM_MAX_TOKENS

    return {
        "selected_samples": [
            {"sample_id": sid, "family": sid.split("-clean_render")[0]} for sid in sample_ids
        ],
        "reasoning_model_provenance": {"max_tokens": GLM_MAX_TOKENS},
    }


@requires_runner
def test_pilot_roster_is_pre_registered_and_spans_eight_families() -> None:
    """F1: the 2026-09-15 run executed 16 consecutive `absolute` samples."""
    from collections import Counter

    from scripts.run_glm46v_geosym_bench_240x7 import PILOT_16_ROSTER

    families = Counter(sid.split("-clean_render")[0] for sid in PILOT_16_ROSTER)

    assert len(PILOT_16_ROSTER) == 16
    assert len(set(PILOT_16_ROSTER)) == 16
    assert len(families) == 8
    assert set(families.values()) == {2}


@requires_runner
def test_select_roster_returns_requested_ids_in_manifest_order() -> None:
    from scripts.run_glm46v_geosym_bench_240x7 import select_roster

    manifest = _manifest_stub(
        ["absolute-clean_render_0000", "cubic-clean_render_0028", "linear-clean_render_0016"]
    )

    selected = select_roster(manifest, ["linear-clean_render_0016", "absolute-clean_render_0000"])

    assert [s["sample_id"] for s in selected] == [
        "absolute-clean_render_0000",
        "linear-clean_render_0016",
    ]


@requires_runner
def test_select_roster_rejects_ids_outside_the_frozen_manifest() -> None:
    from scripts.run_glm46v_geosym_bench_240x7 import select_roster

    manifest = _manifest_stub(["absolute-clean_render_0000"])

    with pytest.raises(ValueError, match="roster_ids_not_in_manifest"):
        select_roster(manifest, ["not-a-real-sample"])


@requires_runner
def test_select_roster_without_ids_runs_the_full_manifest() -> None:
    from scripts.run_glm46v_geosym_bench_240x7 import select_roster

    manifest = _manifest_stub(["absolute-clean_render_0000", "cubic-clean_render_0028"])

    assert len(select_roster(manifest, None)) == 2


@requires_runner
def test_manifest_timeout_is_600s_and_review_rejects_divergence() -> None:
    """At 180s, 26.7% of pilot rows died with APITimeoutError on hard families."""
    from scripts.run_glm46v_geosym_bench_240x7 import GLM_TIMEOUT_SECONDS, review_manifest

    assert GLM_TIMEOUT_SECONDS == 600

    manifest = {"reasoning_model_provenance": {"timeout_seconds": 180, "max_tokens": 16384}}
    assert "glm_timeout_seconds_mismatch" in review_manifest(manifest)["blocking_findings"]


def test_glm_adapter_honours_the_exported_timeout_and_retry_settings(monkeypatch) -> None:
    """A manifest value that never reaches the SDK client would waste the run."""
    from evaluator.adapters.glm import GLMAdapter

    monkeypatch.setenv("GEOSYM_GLM_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("GEOSYM_GLM_MAX_RETRIES", "0")

    client = GLMAdapter(api_key="test", model="glm-4.6v")._get_client()

    assert client.timeout == 600.0
    assert client.max_retries == 0


@requires_runner
def test_plotqa_runner_shares_the_same_glm_settings() -> None:
    """Both GLM tracks hit the same two walls; the constants must not drift."""
    from scripts.run_glm46v_geosym_bench_240x7 import GLM_MAX_TOKENS, GLM_TIMEOUT_SECONDS
    from scripts.run_plotqa_measured_public_confirmation import (
        GLM_MAX_TOKENS as PLOTQA_MAX_TOKENS,
        GLM_TIMEOUT_SECONDS as PLOTQA_TIMEOUT,
    )

    assert PLOTQA_MAX_TOKENS == GLM_MAX_TOKENS
    assert float(PLOTQA_TIMEOUT) == float(GLM_TIMEOUT_SECONDS)


@requires_runner
def test_manifest_max_tokens_is_16384_and_review_rejects_divergence() -> None:
    """F2: 4096 was consumed by thinking tokens before any visible output."""
    from scripts.run_glm46v_geosym_bench_240x7 import GLM_MAX_TOKENS, review_manifest

    assert GLM_MAX_TOKENS == 16384

    manifest = {"reasoning_model_provenance": {"max_tokens": 4096}}
    assert "glm_max_tokens_mismatch" in review_manifest(manifest)["blocking_findings"]


@requires_runner
def test_self_consistency_records_every_call_not_just_the_winning_vote() -> None:
    """F3: keying a scalar by step name dropped 2 of E2's 3 finish reasons."""
    from scripts.run_glm46v_geosym_bench_240x7 import row_finish_reasons, step_finish_reasons

    pipeline = {
        "steps": {"SelfConsistency": {"raw": "kept", "finish_reason": "stop"}},
        "samples": [
            {"raw": "a", "finish_reason": "stop"},
            {"raw": "", "finish_reason": "length"},
            {"raw": "c", "finish_reason": "stop"},
        ],
    }

    reasons = step_finish_reasons(pipeline)

    assert reasons["SelfConsistency"] == ["stop", "length", "stop"]
    assert row_finish_reasons({"reasoning_finish_reasons": reasons}).count("length") == 1


@requires_runner
def test_step_finish_reasons_keeps_each_layer_of_a_multi_step_pipeline() -> None:
    from scripts.run_glm46v_geosym_bench_240x7 import step_finish_reasons

    pipeline = {
        "steps": {
            "L1": {"raw": "a", "finish_reason": "stop"},
            "L2": {"raw": "b", "finish_reason": "stop"},
            "L3": {"raw": "", "finish_reason": "length"},
        }
    }

    assert step_finish_reasons(pipeline) == {"L1": ["stop"], "L2": ["stop"], "L3": ["length"]}


@requires_runner
def test_empty_output_counts_exhausted_thinking_but_not_partial_truncation() -> None:
    """F4: an empty body with finish_reason='length' is the GLM failure mode."""
    from scripts.run_glm46v_geosym_bench_240x7 import empty_output_steps

    pipeline = {
        "steps": {
            "L1": {"raw": "fine", "finish_reason": "stop"},
            "L2": {"raw": "cut off mid-sen", "finish_reason": "length"},
            "L3": {"raw": "   ", "finish_reason": "length"},
        }
    }

    assert empty_output_steps(pipeline) == ["L3"]


@requires_runner
def test_smoke_gate_fails_closed_on_any_empty_output() -> None:
    from scripts.run_glm46v_geosym_bench_240x7 import gate_report

    rows = [
        {
            "sample_id": "absolute-clean_render_0029",
            "condition": "E5",
            "status": "success",
            "total_score": 90.0,
            "reasoning_finish_reasons": {"L3": ["length"]},
            "empty_output_count": 1,
            "ocr_http_attempts": 0,
        }
    ]

    report = gate_report(rows, stage="smoke_2")

    assert report["status"] == "failed"
    assert any(f.startswith("empty_outputs_present") for f in report["failures"])


@requires_runner
def test_pilot_gate_enforces_the_15pct_truncation_rule_that_was_breached() -> None:
    """The 2026-09-15 pilot hit 18.8% on E5 L3 and expanded to Stage C anyway."""
    from scripts.run_glm46v_geosym_bench_240x7 import gate_report

    rows = []
    for index in range(16):
        # 4/16 E5 rows truncated = 25%, above the pre-registered 15% ceiling.
        # L1/L2 always succeed, so the per-call rate would be a diluted 8.3% —
        # the gate must read this per row.
        reason = "length" if index < 4 else "stop"
        for condition in ("E5", "E5_raw"):
            rows.append(
                {
                    "sample_id": f"s{index}",
                    "condition": condition,
                    "status": "success",
                    "total_score": 80.0,
                    "reasoning_finish_reasons": {
                        "L1": ["stop"],
                        "L2": ["stop"],
                        "L3": [reason if condition == "E5" else "stop"],
                    },
                    "empty_output_count": 0,
                    "ocr_http_attempts": 0,
                }
            )

    report = gate_report(rows, stage="pilot_16")

    assert report["status"] == "failed"
    assert any(f.startswith("E5_truncation_ge_15pct") for f in report["failures"])
    assert not any(f.startswith("E5_raw_truncation") for f in report["failures"])


def _smoke_rows(overrides=None):
    from scripts.run_glm46v_geosym_bench_240x7 import SMOKE_2_ROSTER

    overrides = overrides or {}
    rows = []
    for sample_id in SMOKE_2_ROSTER:
        for condition in ALL_CONDITIONS_FOR_TEST:
            row = {
                "sample_id": sample_id,
                "condition": condition,
                "status": "success",
                "total_score": 90.0,
                "reasoning_finish_reasons": {"L1": ["stop"]},
                "empty_output_count": 0,
                "ocr_http_attempts": 0,
            }
            row.update(overrides.get((sample_id, condition), {}))
            rows.append(row)
    return rows


@requires_runner
def test_smoke_gate_accepts_a_failure_that_names_its_failing_component() -> None:
    """The frozen gate allows a failure carrying a row-level error_reason."""
    from scripts.run_glm46v_geosym_bench_240x7 import SMOKE_2_ROSTER, gate_report

    rows = _smoke_rows({
        (SMOKE_2_ROSTER[1], "E2"): {
            "status": "pipeline_error",
            "total_score": None,
            "error_reason": "Request timed out.",
        }
    })

    report = gate_report(rows, stage="smoke_2")

    assert report["status"] == "green"
    # Diagnosed, so non-blocking — but recorded, never silent.
    assert report["diagnosed_failure_rows"] == 1
    assert any("Request timed out." in w for w in report["warnings"])


@requires_runner
def test_smoke_gate_blocks_a_failure_with_no_diagnosable_cause() -> None:
    from scripts.run_glm46v_geosym_bench_240x7 import SMOKE_2_ROSTER, gate_report

    rows = _smoke_rows({
        (SMOKE_2_ROSTER[1], "E2"): {
            "status": "pipeline_error",
            "total_score": None,
            "error_reason": None,
        }
    })

    report = gate_report(rows, stage="smoke_2")

    assert report["status"] == "failed"
    assert any(f.startswith("E2_undiagnosed_failure_rows") for f in report["failures"])


@requires_runner
def test_gate_fails_a_half_finished_run_instead_of_reporting_green() -> None:
    """A crashed run must not pass by deriving its denominator from its own rows."""
    from scripts.run_glm46v_geosym_bench_240x7 import gate_report

    # Only 1 of the 2 smoke_2 roster samples completed.
    rows = [
        {
            "sample_id": "absolute-clean_render_0029",
            "condition": condition,
            "status": "success",
            "total_score": 90.0,
            "reasoning_finish_reasons": {"L1": ["stop"]},
            "empty_output_count": 0,
            "ocr_http_attempts": 0,
        }
        for condition in ALL_CONDITIONS_FOR_TEST
    ]

    report = gate_report(rows, stage="smoke_2")

    assert report["status"] == "failed"
    assert "incomplete_roster:1/2" in report["failures"]
    assert "incomplete_rows:7/14" in report["failures"]


@requires_runner
def test_gate_passes_a_complete_smoke_run() -> None:
    from scripts.run_glm46v_geosym_bench_240x7 import SMOKE_2_ROSTER, gate_report

    rows = [
        {
            "sample_id": sample_id,
            "condition": condition,
            "status": "success",
            "total_score": 90.0,
            "reasoning_finish_reasons": {"L1": ["stop"]},
            "empty_output_count": 0,
            "ocr_http_attempts": 0,
        }
        for sample_id in SMOKE_2_ROSTER
        for condition in ALL_CONDITIONS_FOR_TEST
    ]

    assert gate_report(rows, stage="smoke_2")["status"] == "green"


@requires_runner
def test_gate_flags_an_incomplete_full_run_against_the_manifest_roster() -> None:
    from scripts.run_glm46v_geosym_bench_240x7 import gate_report

    rows = [
        {
            "sample_id": f"s{i}",
            "condition": condition,
            "status": "success",
            "total_score": 90.0,
            "reasoning_finish_reasons": {"L1": ["stop"]},
            "empty_output_count": 0,
            "ocr_http_attempts": 0,
        }
        for i in range(173)  # the 2026-09-15 run died at 173 of 240 samples
        for condition in ALL_CONDITIONS_FOR_TEST
    ]

    report = gate_report(rows, stage="full_240", expected_samples=240)

    assert report["status"] == "failed"
    assert "incomplete_roster:173/240" in report["failures"]


@requires_runner
def test_gate_flags_nonzero_ocr_http_attempts_as_protocol_violation() -> None:
    from scripts.run_glm46v_geosym_bench_240x7 import gate_report

    rows = [
        {
            "sample_id": "s0",
            "condition": "E3",
            "status": "success",
            "total_score": 50.0,
            "reasoning_finish_reasons": {"OCRRulesQA": ["stop"]},
            "empty_output_count": 0,
            "ocr_http_attempts": 1,
        }
    ]

    assert any(
        f.startswith("ocr_http_attempts_nonzero")
        for f in gate_report(rows, stage="pilot_16")["failures"]
    )


@requires_runner
def test_shards_partition_the_roster_without_overlap_or_loss() -> None:
    from scripts.run_glm46v_geosym_bench_240x7 import shard_roster

    samples = [{"sample_id": f"s{i}"} for i in range(240)]

    shards = [shard_roster(samples, i, 4) for i in range(4)]
    recovered = sorted((s["sample_id"] for shard in shards for s in shard), key=lambda x: int(x[1:]))

    assert [len(s) for s in shards] == [60, 60, 60, 60]
    assert recovered == [s["sample_id"] for s in samples]
    # No sample may appear in two shards: concurrent workers would race on its
    # reasoning cache and double-count attempts.
    flat = [s["sample_id"] for shard in shards for s in shard]
    assert len(flat) == len(set(flat))


@requires_runner
def test_shard_roster_rejects_an_invalid_index() -> None:
    from scripts.run_glm46v_geosym_bench_240x7 import shard_roster

    with pytest.raises(ValueError, match="invalid_shard"):
        shard_roster([{"sample_id": "s0"}], 4, 4)


@requires_runner
def test_merge_shards_refuses_overlapping_rows() -> None:
    import json as _json

    from scripts.run_glm46v_geosym_bench_240x7 import merge_shards

    manifest = {"plan_sha256": "abc", "caps": {"sample_condition_job_cap": 4, "llm_http_attempt_cap": 10}, "conditions": ["E0"]}

    def write(tmp, name, sample_id):
        path = tmp / name
        path.write_text(
            _json.dumps(
                {
                    "plan_sha256": "abc",
                    "actual_llm_http_attempts": 2,
                    "results": [
                        {"sample_id": sample_id, "condition": "E0", "status": "success", "total_score": 1.0}
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        a = write(tmp, "paid_result.shard00of02.json", "s0")
        b = write(tmp, "paid_result.shard01of02.json", "s0")  # same sample: overlap

        with pytest.raises(ValueError, match="duplicate_row_across_shards"):
            merge_shards([a, b], manifest)


@requires_runner
def test_merge_shards_combines_attempts_and_rows() -> None:
    import json as _json
    import tempfile

    from scripts.run_glm46v_geosym_bench_240x7 import merge_shards

    manifest = {
        "plan_sha256": "abc",
        "caps": {"sample_condition_job_cap": 2, "llm_http_attempt_cap": 10},
        "conditions": ["E0"],
    }

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        paths = []
        for index, sample_id in enumerate(["s0", "s1"]):
            path = tmp / f"paid_result.shard{index:02d}of02.json"
            path.write_text(
                _json.dumps(
                    {
                        "plan_sha256": "abc",
                        "actual_llm_http_attempts": 3,
                        "results": [
                            {"sample_id": sample_id, "condition": "E0", "status": "success", "total_score": 50.0}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            paths.append(path)

        merged = merge_shards(paths, manifest)

    assert merged["completed_row_count"] == 2
    assert merged["actual_llm_http_attempts"] == 6
    assert merged["stage_status"] == "glm46v_geosym_bench_controlled_240x7_completed"


@requires_runner
def test_exclusive_writer_lock_blocks_a_second_concurrent_runner() -> None:
    """Two runners on one checkpoint recompute and re-pay for the same rows."""
    import tempfile

    from scripts.run_glm46v_geosym_bench_240x7 import exclusive_writer_lock

    with tempfile.TemporaryDirectory() as raw:
        result_path = Path(raw) / "paid_result.json"

        with exclusive_writer_lock(result_path):
            with pytest.raises(RuntimeError, match="result_file_locked_by_another_run"):
                with exclusive_writer_lock(result_path):
                    pass

        # Lock is released on exit, so a later run proceeds normally.
        with exclusive_writer_lock(result_path):
            pass


@requires_runner
def test_exclusive_writer_lock_allows_distinct_shard_files() -> None:
    import tempfile

    from scripts.run_glm46v_geosym_bench_240x7 import exclusive_writer_lock

    with tempfile.TemporaryDirectory() as raw:
        first = Path(raw) / "paid_result.shard00of02.json"
        second = Path(raw) / "paid_result.shard01of02.json"

        with exclusive_writer_lock(first), exclusive_writer_lock(second):
            pass  # different shards must run concurrently


@requires_runner
def test_summary_reports_score_with_its_denominator() -> None:
    from scripts.run_glm46v_geosym_bench_240x7 import summarize

    rows = [
        {
            "condition": "E5",
            "status": "success",
            "total_score": 80.0,
            "reasoning_finish_reasons": {"L3": ["stop"]},
            "empty_output_count": 0,
        },
        {
            "condition": "E5",
            "status": "pipeline_error",
            "total_score": None,
            "reasoning_finish_reasons": {},
            "empty_output_count": 0,
            "error_reason": "Request timed out.",
        },
    ]

    summary = summarize(rows)["E5"]

    assert summary["rows"] == 2
    assert summary["success"] == 1
    assert summary["scored_rows"] == 1
    assert summary["mean_total_score"] == 80.0
