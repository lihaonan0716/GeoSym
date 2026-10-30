#!/usr/bin/env python
"""Run the locked GLM-4.6V GeoSym-Bench controlled 240x7 replication.

Safe default: build the frozen manifest, dry-run, and prepaid review only.
Use ``--run-paid`` after those gates pass. Paid mode checkpoints after every
``(sample_id, condition)`` row and resumes completed rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluator.adapters.base as adapter_base  # noqa: E402
from app.helpers import _get_saved_key  # noqa: E402
from app.reasoning import run_reasoning_all_conditions  # noqa: E402
from app.reasoning_cache import load_reasoning_cache  # noqa: E402
from app.reasoning_pipeline import build_chain_text  # noqa: E402
from evaluator.baselines.paddleocr_rule_baseline import PaddleOCRRuleBaseline  # noqa: E402
# The 240x7 replication completed on 2026-09-18 with all seven arms, E5
# included. Its gates verify that finished artifact, so they must keep
# counting seven conditions; switching to the live list would silently
# relax every row-count gate from 1680 to 1440.
from scripts.batch_experiment import HISTORICAL_CONDITIONS as ALL_CONDITIONS  # noqa: E402
from scripts.prepare_public_benchmark_pilot import canonical_json_bytes, sha256_hex_bytes, stable_file_hash  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "paper_level_evidence_package" / "geosym_bench_controlled_glm46v_240x7"
METADATA_AUDIT = ROOT / "outputs" / "paper_level_evidence_package" / "geosym_bench_controlled" / "metadata_dedup_audit.json"
ROSTER_SHA256 = "4116f04be2207c072a6b91f6d93bbf7470a2f9b40bf4c86606ef941c7d40e63b"
METADATA_FILE_SHA256 = "ae31246aa8fde899211daff57838a6ad3246494bbaf8103ab2aefb622eed3423"
REASONING_MODEL = "GLM-4.6V"
JUDGE_MODEL = "Qwen3-VL-Plus"
REASONING_API_MODEL = "glm-4.6v"
JUDGE_API_MODEL = "qwen3-vl-plus"
REASONING_CALLS_BY_CONDITION = {"E0": 1, "E1": 1, "E2": 3, "E3": 1, "E4": 1, "E5": 4, "E5_raw": 3}
JUDGE_CALLS_BY_CONDITION = {condition: 1 for condition in ALL_CONDITIONS}
TRANSPORT_RETRY_CAP = 1
PILOT_ROW_CAP = 16 * len(ALL_CONDITIONS)

# GLM-4.6V is a thinking model: with max_tokens=4096 the 2026-09-15 run spent the
# whole budget on internal thinking tokens and returned finish_reason='length'
# with an EMPTY body on 90.9% of truncated calls. Completed outputs never exceeded
# ~2100 tokens, so the cap was never the visible-text bottleneck. Raising it is a
# symmetric, prompt-free configuration change applied identically to all seven
# conditions.
GLM_MAX_TOKENS = 16384

# Raising max_tokens removed the token wall and exposed a time wall. At 180s the
# Stage B pilot lost ~17% of calls (27% of rows) to APITimeoutError, and the
# failures tracked family difficulty exactly as truncation had: linear 0%,
# cubic/logarithmic 50%. Harder problems make this backbone think longer, so a
# fixed 180s budget censors precisely the hard cases. Fitting the tail from the
# censoring point (P(T>180s)=17.3%, scale 58.4s) puts 600s at ~0% per-call
# timeout. Symmetric across all seven conditions, no prompt change.
# The change is symmetric across all seven conditions.
GLM_TIMEOUT_SECONDS = 600

SUPERSEDED_PLAN_SHA256 = "48d1f14727ec937f070c8f2a777e3cbd2001163995f68f01c07a0b4414f73f33"

# Stage B0 / Stage B rosters, frozen before execution. The 2026-09-15 run never
# executed Stage B0 and ran Stage B against manifest order rather than this list,
# which is why all 16 "pilot" samples came from a single family.
SMOKE_2_ROSTER = (
    "absolute-clean_render_0029",
    "cubic-clean_render_0028",
)
PILOT_16_ROSTER = (
    "absolute-clean_render_0029",
    "absolute-clean_render_0030",
    "cubic-clean_render_0028",
    "cubic-clean_render_0029",
    "exponential-clean_render_0030",
    "exponential-clean_render_0031",
    "linear-clean_render_0016",
    "linear-clean_render_0017",
    "logarithmic-clean_render_0030",
    "logarithmic-clean_render_0031",
    "piecewise_linear-clean_render_0028",
    "piecewise_linear-clean_render_0030",
    "quadratic-clean_render_0027",
    "quadratic-clean_render_0029",
    "rational-clean_render_0029",
    "rational-clean_render_0030",
)


class AttemptCounter:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.http_attempts = 0

    def consume(self) -> None:
        if self.http_attempts >= self.cap:
            raise RuntimeError("llm_http_attempt_cap_exhausted")
        self.http_attempts += 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_image_name(sample: dict[str, Any]) -> str:
    family = sample["family"]
    sample_id = sample["sample_id"]
    profile_dir = f"{family}-clean_render"
    if sample_id.startswith(f"{profile_dir}/"):
        return sample_id
    return f"{profile_dir}/{sample_id}"


def selected_id_hash(samples: list[dict[str, Any]]) -> str:
    ids = [sample["sample_id"] for sample in samples]
    return sha256_hex_bytes(json.dumps(ids, separators=(",", ":")).encode("utf-8"))


def family_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        counts[str(sample["family"])] += 1
    return dict(sorted(counts.items()))


def plan_hash(manifest: dict[str, Any]) -> str:
    material = {k: v for k, v in manifest.items() if k not in {"created_at_utc", "plan_sha256"}}
    return sha256_hex_bytes(canonical_json_bytes(material))


def call_caps(sample_count: int) -> dict[str, int]:
    return {
        "sample_condition_job_cap": sample_count * len(ALL_CONDITIONS),
        "reasoning_llm_call_cap": sample_count * sum(REASONING_CALLS_BY_CONDITION.values()),
        "judge_llm_call_cap": sample_count * sum(JUDGE_CALLS_BY_CONDITION.values()),
        "llm_http_attempt_cap": sample_count * (sum(REASONING_CALLS_BY_CONDITION.values()) + len(ALL_CONDITIONS)),
        "ocr_http_attempt_cap": 0,
        "paddleocr_local_inference_job_cap": sample_count * 4,
    }


def build_manifest(metadata_path: Path = METADATA_AUDIT) -> dict[str, Any]:
    metadata = load_json(metadata_path)
    samples = metadata.get("selected_subset") or []
    caps = call_caps(len(samples))
    manifest = {
        "schema_version": "geosym_bench_controlled_glm46v_240x7_manifest_v1",
        "created_at_utc": utc_now(),
        "stage": "glm46v_geosym_bench_controlled_240x7_replication",
        "stage_status": "frozen_pending_paid_execution",
        "paid_calls_executed": 0,
        "hypothesis": "On the frozen 240-sample GeoSym-Bench controlled subset, GLM-4.6V can execute the same seven-condition experiment with acceptable completion for cross-backbone comparison.",
        "falsification_signals": [
            "E5 and E5_raw paired completion below 90%",
            "any condition has zero successful rows",
            "E5 and E5_raw cannot be compared on the same sample set",
        ],
        "reasoning_model_provenance": {
            "display_name": REASONING_MODEL,
            "api_model_id": REASONING_API_MODEL,
            "adapter": "evaluator.adapters.glm.GLMAdapter",
            "timeout_seconds": GLM_TIMEOUT_SECONDS,
            "max_tokens": GLM_MAX_TOKENS,
            "temperature": 0.0,
            "sdk_retries": 0,
        },
        "supersedes": {
            "plan_sha256": SUPERSEDED_PLAN_SHA256,
            "reason": "max_tokens=4096 exhausted by thinking tokens; empty outputs confounded every score",
            "superseded_artifacts": "outputs/paper_level_evidence_package/geosym_bench_controlled_glm46v_240x7",
        },
        "judge_model_provenance": {
            "display_name": JUDGE_MODEL,
            "api_model_id": JUDGE_API_MODEL,
            "adapter": "evaluator.adapters.qwen.QwenVLAdapter",
            "timeout_seconds": 90,
            "max_tokens": 2048,
            "temperature": 0.0,
            "sdk_retries": 0,
        },
        "ocr_backend": {
            "name": "PaddleOCR",
            "device": "gpu:0",
            "http_attempt_cap": 0,
            "scope": "E3/E4 only",
            "fail_closed": True,
        },
        "m0_policy": {
            "E0_E1_E2": "no M0/OCR facts",
            "E3_E4": "local PaddleOCR-backed OCR/rule baseline",
            "E5_E5_raw": "clean_grid_fallback_first only; no Mathpix fallback",
        },
        "conditions": list(ALL_CONDITIONS),
        "selected_ids": [sample["sample_id"] for sample in samples],
        "selected_samples": samples,
        "family_counts": family_counts(samples),
        "selected_id_sha256": selected_id_hash(samples),
        "source_artifacts": {
            "metadata_dedup_audit": {
                "path": str(metadata_path.resolve()),
                "sha256": stable_file_hash(metadata_path),
            }
        },
        "caps": caps,
        "retry_policy": "one counted HTTP transport attempt per semantic call; SDK retries disabled",
        "cache_policy": "checkpoint after every sample-condition row keyed by sample_id and condition",
        "sample_replacement": False,
        "interpretation_limits": [
            "Cross-backbone controlled replication only; not broad open-model superiority.",
            "E5 versus E5_raw is not claimed positive unless paired CI excludes zero.",
            "No real-world generalization claim from the synthetic controlled run.",
        ],
    }
    manifest["plan_sha256"] = plan_hash(manifest)
    return manifest


def review_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    findings = []
    if len(manifest.get("selected_samples") or []) != 240:
        findings.append("selected_subset_not_240")
    if manifest.get("selected_id_sha256") != ROSTER_SHA256:
        findings.append("selected_id_hash_mismatch")
    source_sha = manifest.get("source_artifacts", {}).get("metadata_dedup_audit", {}).get("sha256")
    if source_sha != METADATA_FILE_SHA256:
        findings.append("metadata_file_sha256_mismatch")
    if manifest.get("conditions") != list(ALL_CONDITIONS):
        findings.append("condition_list_mismatch")
    if manifest.get("reasoning_model_provenance", {}).get("api_model_id") != REASONING_API_MODEL:
        findings.append("reasoning_model_mismatch")
    if manifest.get("judge_model_provenance", {}).get("api_model_id") != JUDGE_API_MODEL:
        findings.append("judge_model_mismatch")
    if manifest.get("judge_model_provenance", {}).get("api_model_id") == manifest.get("reasoning_model_provenance", {}).get("api_model_id"):
        findings.append("judge_not_independent")
    if manifest.get("reasoning_model_provenance", {}).get("max_tokens") != GLM_MAX_TOKENS:
        findings.append("glm_max_tokens_mismatch")
    if manifest.get("reasoning_model_provenance", {}).get("timeout_seconds") != GLM_TIMEOUT_SECONDS:
        findings.append("glm_timeout_seconds_mismatch")
    caps = manifest.get("caps") or {}
    expected = call_caps(240)
    for key, value in expected.items():
        if caps.get(key) != value:
            findings.append(f"{key}_mismatch")
    if caps.get("ocr_http_attempt_cap") != 0:
        findings.append("ocr_http_cap_not_zero")
    return {
        "schema_version": "geosym_bench_controlled_glm46v_240x7_prepaid_review_v1",
        "created_at_utc": utc_now(),
        "status": "no_blocking_findings" if not findings else "blocking_findings_present",
        "blocking_findings": findings,
        "paid_calls_executed": 0,
    }


def build_dry_run(manifest: dict[str, Any]) -> dict[str, Any]:
    review = review_manifest(manifest)
    return {
        "schema_version": "geosym_bench_controlled_glm46v_240x7_dry_run_v1",
        "created_at_utc": utc_now(),
        "stage_status": "ready_for_paid_execution" if review["status"] == "no_blocking_findings" else "blocking_findings_present",
        "paid_calls_executed": 0,
        "conditions": manifest["conditions"],
        "sample_count": len(manifest["selected_samples"]),
        "caps": manifest["caps"],
        "review_status": review["status"],
        "blocking_findings": review["blocking_findings"],
    }


@contextmanager
def exclusive_writer_lock(result_path: Path) -> Iterator[None]:
    """Refuse to start if another process is already writing this result file.

    Two runners sharing one checkpoint interleave their writes: each reloads the
    file, appends its own row, and saves, so rows get recomputed and paid for
    twice while the recorded attempt count stays too low. Sharding makes
    concurrent runners normal, so this has to be enforced rather than assumed.
    """
    lock_path = result_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # O_EXCL is atomic on Windows and POSIX: exactly one process wins.
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = ""
        try:
            holder = lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
        raise RuntimeError(
            f"result_file_locked_by_another_run:{lock_path}"
            + (f" (holder: {holder})" if holder else "")
            + " — stop that process, or delete the lock if it is stale."
        )
    try:
        os.write(fd, f"pid={os.getpid()} started={utc_now()}\n".encode("utf-8"))
        os.close(fd)
        yield
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass


@contextmanager
def count_llm_attempts(counter: AttemptCounter) -> Iterator[None]:
    original = adapter_base.BaseLLMAdapter._call_with_retry

    def counted(self, image_b64: str, prompt: str) -> str:
        counter.consume()
        return self._call_api(image_b64, prompt)

    adapter_base.BaseLLMAdapter._call_with_retry = counted
    try:
        yield
    finally:
        adapter_base.BaseLLMAdapter._call_with_retry = original


@contextmanager
def clean_grid_only_m0() -> Iterator[None]:
    import evaluator.adapters.recognizer as recognizer_module
    from recognizer.core.clean_grid_fallback import extract_clean_cartesian_grid

    original = recognizer_module.RecognizerAdapter.recognize_intersections

    def recognize_clean_grid_only(self, image, gt=None, skip_enhancement=True, gemini_enabled=False):
        import cv2
        from pathlib import Path

        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                return [], {"success": False, "error": "clean_grid_unavailable"}, None
        extraction = extract_clean_cartesian_grid(image)
        if not extraction.success:
            return [], {"success": False, "error": "clean_grid_unavailable"}, None
        extraction.result["fallback_first"] = True
        extraction.result["clean_grid_only"] = True
        return extraction.coords, extraction.result, None

    recognizer_module.RecognizerAdapter.recognize_intersections = recognize_clean_grid_only
    try:
        yield
    finally:
        recognizer_module.RecognizerAdapter.recognize_intersections = original


def step_finish_reasons(pipeline: dict[str, Any]) -> dict[str, list[Any]]:
    """Return every semantic call's finish reason, keyed by step, in call order.

    The value is a LIST because a single step name can cover several calls: E2's
    ``SelfConsistency`` step runs ``SELF_CONSISTENCY_RUNS`` times but the pipeline
    only promotes the winning vote into ``steps``. Keying a scalar by step name
    therefore silently dropped all but one of E2's three finish reasons and
    understated its truncation rate in the 2026-09-15 run.
    """
    if not pipeline:
        return {}
    reasons: dict[str, list[Any]] = {}
    for step, data in (pipeline.get("steps") or {}).items():
        if not isinstance(data, dict):
            continue
        fallback = (pipeline.get("finish_reasons") or {}).get(step)
        reasons[step] = [data.get("finish_reason", fallback)]
    # Self-consistency keeps every sampled call under "samples"; prefer that full
    # record over the single promoted winner.
    samples = pipeline.get("samples")
    if isinstance(samples, list) and samples:
        step_name = next(iter(reasons), "SelfConsistency")
        reasons[step_name] = [
            sample.get("finish_reason") for sample in samples if isinstance(sample, dict)
        ]
    return reasons


def empty_output_steps(pipeline: dict[str, Any]) -> list[str]:
    """Steps whose call hit the token cap without emitting any visible text.

    GLM-4.6V spends tokens on internal thinking before producing output, so an
    exhausted budget yields ``finish_reason='length'`` with an empty body rather
    than a partial one. This is the primary health signal for this backbone and
    is distinct from a parse failure.
    """
    if not pipeline:
        return []
    empties: list[str] = []
    for step, data in (pipeline.get("steps") or {}).items():
        if not isinstance(data, dict):
            continue
        if data.get("finish_reason") == "length" and not (data.get("raw") or "").strip():
            empties.append(step)
    samples = pipeline.get("samples")
    if isinstance(samples, list):
        step_name = next(iter((pipeline.get("steps") or {})), "SelfConsistency")
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            if sample.get("finish_reason") == "length" and not (sample.get("raw") or "").strip():
                empties.append(f"{step_name}[{index}]")
        if step_name in empties and any(e.startswith(f"{step_name}[") for e in empties):
            empties.remove(step_name)
    return empties


def scoring_inputs(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Persist exactly what an offline re-score needs, and nothing more.

    Rows previously carried only ``raw_model_output_sha256``. A hash proves the
    output did not change but cannot be re-parsed, so when two scoring defects
    surfaced on 2026-09-19 -- narration captured as the expression, and
    ``\\log_{2}(u)`` parsed as a symbol times ``(u)`` -- the 1680-row artifact
    could not be corrected offline. Only 3.8% of rows were recoverable, from
    side artifacts that happened to keep ``steps``.

    Storing the per-layer raw text plus the parsed answer makes any future
    parser or metric repair a zero-cost re-analysis instead of a re-run. The
    hashes stay: they remain the integrity check, this is the payload.
    """
    if not pipeline:
        return {}
    steps = pipeline.get("steps") or {}
    raw_by_step = {
        name: data.get("raw")
        for name, data in steps.items()
        if isinstance(data, dict) and data.get("raw")
    }
    return {
        # Keyed by layer so a defect in one layer's parser can be re-run alone.
        "step_raw_text": raw_by_step,
        # The parsed answer as scored, so a re-score can be diffed against the
        # original rather than silently replacing it.
        "final_parsed": pipeline.get("final_parsed"),
    }


def condition_row(sample: dict[str, Any], cache: dict[str, Any], condition: str, llm_before: int, llm_after: int, ocr_jobs: int) -> dict[str, Any]:
    cond = (cache.get("conditions") or {}).get(condition) or {}
    ev = cond.get("eval") or {}
    judge = cond.get("judge") or {}
    pipeline = cond.get("pipeline") or {}
    chain_text = build_chain_text(pipeline) if pipeline else ""
    error = cond.get("error")
    error_reason = None
    if error:
        error_reason = error.get("message") if isinstance(error, dict) else str(error)
        if "clean_grid_unavailable" in str(error_reason):
            error_reason = "clean_grid_unavailable"
    return {
        "sample_id": sample["sample_id"],
        "image_name": sample_image_name(sample),
        "family": sample["family"],
        "dedup_key": sample.get("dedup_key"),
        "condition": condition,
        "status": cond.get("status"),
        "total_score": ev.get("total_score"),
        "dimension_scores": ev.get("dimension_scores"),
        "chain_text_sha256": sha256_hex_bytes(canonical_json_bytes(chain_text)) if chain_text else None,
        "raw_model_output_sha256": sha256_hex_bytes(canonical_json_bytes(pipeline)) if pipeline else None,
        # Re-scorable payload behind those hashes. See `scoring_inputs`.
        "scoring_inputs": scoring_inputs(pipeline),
        "judge_result_sha256": sha256_hex_bytes(canonical_json_bytes(judge)) if judge else None,
        "reasoning_finish_reasons": step_finish_reasons(pipeline),
        "judge_finish_reason": judge.get("finish_reason") if judge else None,
        "empty_output_steps": empty_output_steps(pipeline),
        "empty_output_count": len(empty_output_steps(pipeline)),
        "parse_gates": {"pipeline_present": bool(pipeline), "eval_present": bool(ev), "judge_present": bool(judge)},
        "error": error,
        "error_reason": error_reason,
        "llm_http_attempts": llm_after - llm_before,
        "ocr_http_attempts": 0,
        "paddleocr_local_inference_jobs": ocr_jobs,
    }


def write_result(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]], attempts: int, status: str) -> dict[str, Any]:
    result = {
        "schema_version": "geosym_bench_controlled_glm46v_240x7_result_v1",
        "created_at_utc": utc_now(),
        "stage_status": status,
        "plan_sha256": manifest["plan_sha256"],
        "paid_calls_executed": attempts,
        "actual_llm_http_attempts": attempts,
        "actual_ocr_http_attempts": 0,
        "completed_row_count": len(rows),
        "sample_condition_job_cap": manifest["caps"]["sample_condition_job_cap"],
        "llm_http_attempt_cap": manifest["caps"]["llm_http_attempt_cap"],
        "conditions": manifest["conditions"],
        "summary": summarize(rows),
        "results": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def row_finish_reasons(row: dict[str, Any]) -> list[Any]:
    reasons = row.get("reasoning_finish_reasons") or {}
    if isinstance(reasons, dict):
        flat: list[Any] = []
        for value in reasons.values():
            flat.extend(value if isinstance(value, list) else [value])
        return flat
    return list(reasons)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    summary: dict[str, Any] = {}
    for condition, items in by_condition.items():
        scored = [float(row["total_score"]) for row in items if row.get("total_score") is not None]
        calls = [reason for row in items for reason in row_finish_reasons(row)]
        summary[condition] = {
            "rows": len(items),
            "success": sum(1 for row in items if row.get("status") == "success"),
            # Reported alongside its denominator: a mean over successful rows
            # without the completion count is not interpretable.
            "scored_rows": len(scored),
            "mean_total_score": round(sum(scored) / len(scored), 6) if scored else None,
            "errors": sum(1 for row in items if row.get("error_reason")),
            "semantic_calls": len(calls),
            "truncated_calls": sum(1 for reason in calls if reason == "length"),
            "truncated_rows": sum(
                1 for row in items if any(r == "length" for r in row_finish_reasons(row))
            ),
            "empty_output_calls": sum(int(row.get("empty_output_count") or 0) for row in items),
            "empty_output_rows": sum(1 for row in items if row.get("empty_output_count")),
            "judge_length_calls": sum(1 for row in items if row.get("judge_finish_reason") == "length"),
        }
    return summary


STAGE_EXPECTED_SAMPLES = {
    "smoke_2": len(SMOKE_2_ROSTER),
    "pilot_16": len(PILOT_16_ROSTER),
}

# Rates carried over from the pilot_16 gate, which expressed the same thresholds
# as counts against a 16-sample roster. Restating them as rates keeps full_240
# governed by the SAME pre-registered numbers rather than new ones chosen after
# the pilot scores were visible: 15/16 successes is 0.9375, and the empty-output
# and truncation thresholds are already rates.
FULL_MIN_SUCCESS_RATE = 15 / 16
FULL_MAX_EMPTY_RATE = 0.05
FULL_MAX_TRUNCATION_RATE = 0.15
EXPECTED_FAMILY_COUNT = 8

# A row that fails because a LOCAL resource gave out is an infrastructure defect,
# not a property of the backbone. Both are recorded as a diagnosed `error_reason`
# and are therefore indistinguishable to the generic warning path, which is how a
# GPU out-of-memory event could be written up as "GLM-4.6V cannot do E3/E4".
# Hard rule 5 ("failures are reported as backbone/provider-path limitations")
# applies to remote failures only; these must block instead.
INFRASTRUCTURE_ERROR_MARKERS = (
    "out of memory",
    "outofmemory",
    "cuda",
    "cudnn",
    "resourceexhausted",
    "memoryerror",
    "llm_http_attempt_cap_exhausted",
    "no space left",
    "oserror",
)

# Not every PaddleOCR error is an infrastructure fault. E3/E4 ARE the OCR rule
# baseline, so "OCR ran and could not read the axis labels" is that baseline's
# measured capability — exactly the result the condition exists to produce, and
# suppressing it would overstate the baseline we compare against (invariant 6).
# Only failures where the OCR engine could not RUN are infrastructure. The
# generic "paddleocr" substring conflated the two and blocked the gate on four
# legitimate baseline failures.
BASELINE_CAPABILITY_MARKERS = (
    "paddleocr_rule_baseline_no_coordinates",
    "paddleocr_rule_baseline_no_ticks",
    "paddleocr_rule_baseline_ocr_empty",
    # OCR ran and read SOME labels, but too few on one axis to fit the ruler
    # (paddleocr_rule_baseline.py:167 needs >= 2 per axis). Observed as
    # "x=1:y=10": a reading limit of the baseline, not an engine fault.
    "paddleocr_insufficient_tick_labels",
)


def infrastructure_failures(rows: list[dict[str, Any]]) -> list[str]:
    """Rows whose failure is local-infrastructure, not backbone or baseline behaviour."""
    hits = []
    for row in rows:
        if row.get("status") == "success":
            continue
        reason = str(row.get("error_reason") or "").lower()
        if any(marker in reason for marker in BASELINE_CAPABILITY_MARKERS):
            continue
        if any(marker in reason for marker in INFRASTRUCTURE_ERROR_MARKERS):
            hits.append(f"{row.get('condition')}:{row.get('sample_id')}:{row.get('error_reason')}")
    return hits


def gate_report(rows: list[dict[str, Any]], *, stage: str, expected_samples: int | None = None) -> dict[str, Any]:
    """Evaluate the pre-registered stage gate. Failures must block expansion.

    The 2026-09-15 run breached the <15% E5/E5_raw truncation gate at the pilot
    (18.8%) and expanded to the full run anyway, because the gate lived only in
    the plan document. It is enforced in code here.
    """
    summary = summarize(rows)
    samples = {row["sample_id"] for row in rows}
    failures: list[str] = []

    # Coverage is checked against the frozen roster size, not against whatever
    # rows happen to exist. Deriving the denominator from the data would let a
    # crashed half-run report green, which is how the 2026-09-15 Stage C came
    # to be launched on a pilot that had not actually cleared its gate.
    required = expected_samples if expected_samples is not None else STAGE_EXPECTED_SAMPLES.get(stage)
    if required is not None and len(samples) < required:
        failures.append(f"incomplete_roster:{len(samples)}/{required}")
    expected_rows = required * len(ALL_CONDITIONS) if required is not None else None
    if expected_rows is not None and len(rows) < expected_rows:
        failures.append(f"incomplete_rows:{len(rows)}/{expected_rows}")

    calls = sum(s["semantic_calls"] for s in summary.values())
    empties = sum(s["empty_output_calls"] for s in summary.values())
    empty_rate = empties / calls if calls else 0.0
    judge_length = sum(s["judge_length_calls"] for s in summary.values())
    if judge_length:
        failures.append(f"judge_finish_reason_length:{judge_length}")

    if stage == "smoke_2":
        # Zero tolerance: if 16384 tokens cannot clear two samples, the mitigation
        # is insufficient and the answer is analysis, not a larger budget.
        if empties:
            failures.append(f"empty_outputs_present:{empties}")
        needed = required if required is not None else len(samples)
        for condition in ALL_CONDITIONS:
            stats = summary.get(condition)
            if not stats or stats["rows"] < needed:
                failures.append(f"{condition}_missing_rows")
                continue
            # The pre-registered gate reads: "All seven conditions produce
            # parseable rows for both samples, OR any failure has a row-level
            # error_reason that identifies the failing component." A diagnosed
            # failure therefore passes; an undiagnosable one does not.
            undiagnosed = [
                row
                for row in rows
                if row.get("condition") == condition
                and row.get("status") != "success"
                and not row.get("error_reason")
            ]
            if undiagnosed:
                failures.append(f"{condition}_undiagnosed_failure_rows:{len(undiagnosed)}")
    elif stage == "pilot_16":
        if empty_rate >= 0.05:
            failures.append(f"empty_output_rate_ge_5pct:{empty_rate:.3f}")
        for condition in ALL_CONDITIONS:
            stats = summary.get(condition)
            if not stats:
                failures.append(f"{condition}_missing")
                continue
            if stats["success"] < 15:
                failures.append(f"{condition}_success_below_15:{stats['success']}")
            if condition in ("E5", "E5_raw"):
                # Row-level, not per-call: averaging L1/L2/L3 together hides a
                # badly truncated L3 behind two cheap, reliable early layers.
                # The 2026-09-15 pilot was 8.3% per-call but 25% per-row.
                rate = stats["truncated_rows"] / stats["rows"] if stats["rows"] else 0.0
                if rate >= 0.15:
                    failures.append(f"{condition}_truncation_ge_15pct:{rate:.3f}")
        paired = sum(
            1
            for sample_id in samples
            if all(
                any(r["sample_id"] == sample_id and r["condition"] == c and r.get("status") == "success" for r in rows)
                for c in ("E5", "E5_raw")
            )
        )
        if paired < 15:
            failures.append(f"e5_paired_completion_below_15:{paired}")
    elif stage == "full_240":
        # Before 2026-09-18 this branch did not exist, so `full_240` fell through
        # to the shared checks alone. A merged result in which every row was a
        # diagnosed failure, or every call returned an empty body, still reported
        # `green` and exit 0 — the exact confound that invalidated three earlier
        # runs passed unnoticed at the only stage that spends the full budget.
        # Thresholds are the pilot's, restated as rates; none are new.
        if empty_rate >= FULL_MAX_EMPTY_RATE:
            failures.append(f"empty_output_rate_ge_5pct:{empty_rate:.3f}")
        for condition in ALL_CONDITIONS:
            stats = summary.get(condition)
            if not stats:
                failures.append(f"{condition}_missing")
                continue
            if not stats["rows"]:
                failures.append(f"{condition}_zero_rows")
                continue
            success_rate = stats["success"] / stats["rows"]
            if success_rate < FULL_MIN_SUCCESS_RATE:
                failures.append(
                    f"{condition}_success_rate_below_94pct:{success_rate:.3f}"
                    f"({stats['success']}/{stats['rows']})"
                )
            # A mean over survivors is only interpretable if survivors were
            # actually scored; an unscored majority silently reweights the mean.
            if stats["scored_rows"] < stats["success"]:
                failures.append(
                    f"{condition}_unscored_successes:{stats['success'] - stats['scored_rows']}"
                )
            if condition in ("E5", "E5_raw"):
                rate = stats["truncated_rows"] / stats["rows"]
                if rate >= FULL_MAX_TRUNCATION_RATE:
                    failures.append(f"{condition}_truncation_ge_15pct:{rate:.3f}")
        # The paired E5/E5_raw contrast is the run's primary deliverable, so its
        # completion is gated directly rather than inferred from per-condition
        # totals: two conditions can each clear 94% while overlapping poorly.
        paired = sum(
            1
            for sample_id in samples
            if all(
                any(
                    r["sample_id"] == sample_id and r["condition"] == c and r.get("status") == "success"
                    for r in rows
                )
                for c in ("E5", "E5_raw")
            )
        )
        if required and paired / required < FULL_MIN_SUCCESS_RATE:
            failures.append(f"e5_paired_completion_below_94pct:{paired}/{required}")
        # R3 requires reporting all eight families or none, so a family lost
        # wholesale to a crashed shard must stop the run rather than silently
        # narrow the per-family analysis.
        families = {str(row.get("family")) for row in rows if row.get("status") == "success"}
        if len(families) < EXPECTED_FAMILY_COUNT:
            failures.append(f"families_incomplete:{len(families)}/{EXPECTED_FAMILY_COUNT}")

    # Local-infrastructure failures are never a finding about the backbone.
    # Checked at every stage: mislabelling one of these is a scientific-integrity
    # problem, not a completion problem.
    infra = infrastructure_failures(rows)
    if infra:
        failures.append(f"infrastructure_failures:{len(infra)}")

    ocr_http = sum(int(row.get("ocr_http_attempts") or 0) for row in rows)
    if ocr_http:
        failures.append(f"ocr_http_attempts_nonzero:{ocr_http}")

    # Diagnosed failures do not block, but they are never silent: a transport
    # timeout that recurs at scale is a completion problem even when each
    # individual row is explainable.
    warnings = [
        f"{row.get('condition')}:{row.get('sample_id')}:{row.get('error_reason')}"
        for row in rows
        if row.get("status") != "success" and row.get("error_reason")
    ]

    return {
        "stage": stage,
        "status": "green" if not failures else "failed",
        "failures": failures,
        "warnings": warnings,
        "diagnosed_failure_rows": len(warnings),
        "infrastructure_failure_rows": infra,
        "sample_count": len(samples),
        "semantic_calls": calls,
        "empty_output_calls": empties,
        "empty_output_rate": round(empty_rate, 6),
        "ocr_http_attempts": ocr_http,
    }


def select_roster(manifest: dict[str, Any], roster_ids: list[str] | None) -> list[dict[str, Any]]:
    """Restrict execution to an explicit roster, preserving frozen manifest order.

    Roster membership is decided here and nowhere else. ``--max-new-rows`` is a
    spend brake only: using it to bound a pilot is what made the 2026-09-15
    Stage B run 16 consecutive ``absolute`` samples instead of the 16
    pre-registered IDs spanning 8 families.
    """
    samples = manifest["selected_samples"]
    if not roster_ids:
        return samples
    wanted = list(dict.fromkeys(roster_ids))
    by_id = {sample["sample_id"]: sample for sample in samples}
    missing = [sample_id for sample_id in wanted if sample_id not in by_id]
    if missing:
        raise ValueError(f"roster_ids_not_in_manifest:{missing}")
    selected = [sample for sample in samples if sample["sample_id"] in set(wanted)]
    if len(selected) != len(wanted):
        raise ValueError(f"roster_selection_size_mismatch:{len(selected)}!={len(wanted)}")
    return selected


def shard_roster(samples: list[dict[str, Any]], index: int, count: int) -> list[dict[str, Any]]:
    """Deterministic contiguous-stride slice of the roster.

    Sharding is by SAMPLE, never by condition *within a wave*: concurrent workers
    must not touch the same sample, because ``results/single_image_reasoning``
    stores one non-atomically-written JSON per (image, model) and two writers
    would clobber each other. ``--conditions`` splits work across sequential
    waves instead, which keeps that guarantee — see ``select_conditions``.
    """
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"invalid_shard:{index}/{count}")
    return [sample for position, sample in enumerate(samples) if position % count == index]


# E3/E4 are the only conditions that touch PaddleOCR, and PaddleOCR costs ~4.7 GB
# of VRAM per process (CUDA context + cuDNN kernels; capping the allocator pool
# only reclaims ~0.6 GB). On a 16 GB card three concurrent loaders already leave
# under 2 GB, so the 8-shard launch this file was written for would exhaust the
# device. Running conditions in waves keeps PaddleOCR in exactly one process:
# the OCR wave is serial, and the other five conditions never construct a reader
# because PaddleOCRRuleBaseline loads lazily on first use (~4 MiB until then).
OCR_CONDITIONS = ("E3", "E4")
NON_OCR_CONDITIONS = tuple(c for c in ALL_CONDITIONS if c not in OCR_CONDITIONS)


def select_conditions(manifest: dict[str, Any], requested: list[str] | None) -> list[str]:
    """Conditions this worker executes, preserving frozen manifest order."""
    declared = list(manifest["conditions"])
    if not requested:
        return declared
    wanted = list(dict.fromkeys(requested))
    unknown = [c for c in wanted if c not in declared]
    if unknown:
        raise ValueError(f"conditions_not_in_manifest:{unknown}")
    return [c for c in declared if c in set(wanted)]


def run_paid(
    manifest: dict[str, Any],
    output_path: Path,
    *,
    max_new_rows: int | None = None,
    roster_ids: list[str] | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
    conditions: list[str] | None = None,
) -> dict[str, Any]:
    review = review_manifest(manifest)
    if review["status"] != "no_blocking_findings":
        raise ValueError(f"prepaid_review_blocking_findings:{review['blocking_findings']}")
    manifest_max_tokens = manifest["reasoning_model_provenance"]["max_tokens"]
    if manifest_max_tokens != GLM_MAX_TOKENS:
        raise ValueError(f"manifest_max_tokens_diverges_from_runner:{manifest_max_tokens}!={GLM_MAX_TOKENS}")
    manifest_timeout = manifest["reasoning_model_provenance"]["timeout_seconds"]
    if manifest_timeout != GLM_TIMEOUT_SECONDS:
        raise ValueError(f"manifest_timeout_diverges_from_runner:{manifest_timeout}!={GLM_TIMEOUT_SECONDS}")
    roster = select_roster(manifest, roster_ids)
    if shard_count:
        roster = shard_roster(roster, shard_index or 0, shard_count)
    active_conditions = select_conditions(manifest, conditions)
    reasoning_key = _get_saved_key(REASONING_MODEL)
    judge_key = _get_saved_key(JUDGE_MODEL)
    if not reasoning_key:
        raise ValueError(f"saved API key not found for {REASONING_MODEL}")
    if not judge_key:
        raise ValueError(f"saved API key not found for {JUDGE_MODEL}")

    rows: list[dict[str, Any]] = []
    if output_path.exists():
        checkpoint = load_json(output_path)
        if checkpoint.get("plan_sha256") != manifest["plan_sha256"]:
            raise ValueError("checkpoint_plan_sha256_mismatch")
        rows = list(checkpoint.get("results") or [])
    completed = {(row["sample_id"], row["condition"]) for row in rows}
    # Budget by the work actually assigned, not by a flat division of the global
    # cap. `ceil(5040/8)` would over-fund a condition-restricted wave, letting one
    # wave spend budget that belongs to another; summing this per-slice cost over
    # disjoint (shard x condition) slices reproduces the global cap exactly.
    #
    # Headroom is deliberately zero: 5040 = 240 x 21 is the exact zero-retry cost
    # (hard rule 5 forbids retries), so any slack would breach the frozen cap.
    # Exhaustion therefore means unbudgeted spend happened, and the gate treats
    # `llm_http_attempt_cap_exhausted` as an infrastructure failure that blocks —
    # never as a backbone limitation.
    attempt_cap = manifest["caps"]["llm_http_attempt_cap"]
    if shard_count or conditions:
        per_sample = sum(
            REASONING_CALLS_BY_CONDITION[c] + JUDGE_CALLS_BY_CONDITION[c]
            for c in active_conditions
        )
        attempt_cap = len(roster) * per_sample
    counter = AttemptCounter(attempt_cap)
    counter.http_attempts = sum(int(row.get("llm_http_attempts") or 0) for row in rows)
    ocr_baseline = PaddleOCRRuleBaseline(device="gpu:0")
    original_env = {name: os.environ.get(name) for name in [
        "GEOSYM_GLM_TIMEOUT_SECONDS", "GEOSYM_GLM_MAX_TOKENS", "GEOSYM_GLM_MAX_RETRIES",
        "GEOSYM_QWEN_TIMEOUT_SECONDS", "GEOSYM_QWEN_MAX_TOKENS", "GEOSYM_QWEN_MAX_RETRIES",
    ]}
    os.environ["GEOSYM_GLM_TIMEOUT_SECONDS"] = str(manifest_timeout)
    os.environ["GEOSYM_GLM_MAX_TOKENS"] = str(manifest_max_tokens)
    os.environ["GEOSYM_GLM_MAX_RETRIES"] = "0"
    os.environ["GEOSYM_QWEN_TIMEOUT_SECONDS"] = "90"
    os.environ["GEOSYM_QWEN_MAX_TOKENS"] = "2048"
    os.environ["GEOSYM_QWEN_MAX_RETRIES"] = "0"
    new_rows = 0
    try:
        with exclusive_writer_lock(output_path), count_llm_attempts(counter), clean_grid_only_m0():
            for sample in roster:
                image_name = sample_image_name(sample)
                for condition in active_conditions:
                    if (sample["sample_id"], condition) in completed:
                        continue
                    if max_new_rows is not None and new_rows >= max_new_rows:
                        return write_result(output_path, manifest, rows, counter.http_attempts, "partial_paid_run")
                    before = counter.http_attempts
                    before_ocr_jobs = ocr_baseline.local_inference_jobs
                    try:
                        gen = run_reasoning_all_conditions(
                            image_name=image_name,
                            model_name=REASONING_MODEL,
                            api_key=reasoning_key,
                            judge_model_name=JUDGE_MODEL,
                            judge_api_key=judge_key,
                            lang="en",
                            force_rerun=True,
                            target_conditions=[condition],
                            ocr_baseline=ocr_baseline,
                        )
                        for _ in gen:
                            pass
                        cache = load_reasoning_cache(image_name, REASONING_MODEL) or {}
                    except Exception as exc:
                        cache = {"conditions": {condition: {"status": "pipeline_error", "error": {"type": type(exc).__name__, "message": str(exc)}}}}
                    after = counter.http_attempts
                    ocr_jobs = ocr_baseline.local_inference_jobs - before_ocr_jobs
                    rows.append(condition_row(sample, cache, condition, before, after, ocr_jobs))
                    completed.add((sample["sample_id"], condition))
                    new_rows += 1
                    result = write_result(output_path, manifest, rows, counter.http_attempts, "partial_paid_run")
                    print(
                        f"completed_row={len(rows)}/{manifest['caps']['sample_condition_job_cap']} "
                        f"sample_id={sample['sample_id']} condition={condition} "
                        f"llm_http_attempts={counter.http_attempts}/{manifest['caps']['llm_http_attempt_cap']}",
                        flush=True,
                    )
    finally:
        for name, value in original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    # A worker is complete when it finished ITS slice; the full-run job cap is
    # only the right denominator for an unsharded, all-conditions invocation.
    expected_rows = len(roster) * len(active_conditions)
    status = (
        "glm46v_geosym_bench_controlled_240x7_completed"
        if len(rows) == expected_rows
        else "partial_paid_run"
    )
    return write_result(output_path, manifest, rows, counter.http_attempts, status)


def merge_shards(shard_paths: list[Path], manifest: dict[str, Any]) -> dict[str, Any]:
    """Combine per-shard checkpoints into one result, refusing silent row loss.

    Duplicate ``(sample_id, condition)`` keys across shards mean the shards
    overlapped, which would double-count attempts and corrupt paired contrasts.
    That is an error, not something to dedupe away quietly.
    """
    rows: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], Path] = {}
    attempts = 0
    plan_hashes: set[str] = set()
    for path in shard_paths:
        payload = load_json(path)
        plan_hashes.add(str(payload.get("plan_sha256")))
        attempts += int(payload.get("actual_llm_http_attempts") or 0)
        for row in payload.get("results") or []:
            key = (str(row.get("sample_id")), str(row.get("condition")))
            if key in seen:
                raise ValueError(f"duplicate_row_across_shards:{key}:{seen[key].name}+{path.name}")
            seen[key] = path
            rows.append(row)
    if len(plan_hashes) > 1:
        raise ValueError(f"shards_disagree_on_plan_sha256:{sorted(plan_hashes)}")
    if plan_hashes and manifest["plan_sha256"] not in plan_hashes:
        raise ValueError("shard_plan_sha256_does_not_match_manifest")
    rows.sort(key=lambda r: (str(r.get("sample_id")), ALL_CONDITIONS.index(str(r.get("condition")))))
    expected = manifest["caps"]["sample_condition_job_cap"]
    status = (
        "glm46v_geosym_bench_controlled_240x7_completed"
        if len(rows) == expected
        else "partial_paid_run"
    )
    return {
        "schema_version": "geosym_bench_controlled_glm46v_240x7_result_v1",
        "created_at_utc": utc_now(),
        "stage_status": status,
        "plan_sha256": manifest["plan_sha256"],
        "paid_calls_executed": attempts,
        "actual_llm_http_attempts": attempts,
        "actual_ocr_http_attempts": sum(int(r.get("ocr_http_attempts") or 0) for r in rows),
        "completed_row_count": len(rows),
        "sample_condition_job_cap": expected,
        "llm_http_attempt_cap": manifest["caps"]["llm_http_attempt_cap"],
        "conditions": manifest["conditions"],
        "merged_from": [str(p) for p in shard_paths],
        "summary": summarize(rows),
        "results": rows,
    }


def render_analysis(result: dict[str, Any]) -> str:
    lines = [
        "# GLM-4.6V GeoSym-Bench Controlled 240x7",
        "",
        f"- Stage status: `{result['stage_status']}`",
        f"- Completed rows: `{result['completed_row_count']}/{result['sample_condition_job_cap']}`",
        f"- Paid calls executed: `{result['paid_calls_executed']}`",
        f"- OCR HTTP attempts: `{result['actual_ocr_http_attempts']}`",
        "",
        "## Condition Summary",
        "",
        "Completion and score are reported together; a mean without its denominator",
        "is not interpretable. `empty` counts calls that hit the token cap without",
        "emitting visible text — the GLM-4.6V thinking-budget failure mode.",
        "",
        "| Condition | success/rows | scored | mean score | truncated calls | empty calls | errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in ALL_CONDITIONS:
        row = result.get("summary", {}).get(condition)
        if not row:
            lines.append(f"| {condition} | 0/0 | 0 | — | — | — | — |")
            continue
        mean = row.get("mean_total_score")
        calls = row.get("semantic_calls") or 0
        trunc = row.get("truncated_calls") or 0
        trunc_pct = f"{trunc}/{calls} ({100 * trunc / calls:.1f}%)" if calls else "—"
        lines.append(
            f"| {condition} | {row['success']}/{row['rows']} | {row.get('scored_rows', 0)} | "
            f"{f'{mean:.2f}' if mean is not None else '—'} | {trunc_pct} | "
            f"{row.get('empty_output_calls', 0)} | {row['errors']} |"
        )
    gate = result.get("gate_report")
    if gate:
        lines += [
            "",
            "## Gate",
            "",
            f"- Stage: `{gate['stage']}`",
            f"- Status: `{gate['status']}`",
            f"- Empty-output rate: {gate['empty_output_rate']:.3%} ({gate['empty_output_calls']}/{gate['semantic_calls']} calls)",
            f"- OCR HTTP attempts: {gate['ocr_http_attempts']}",
        ]
        if gate["failures"]:
            lines.append("- Failures:")
            lines += [f"  - `{failure}`" for failure in gate["failures"]]
    return "\n".join(lines) + "\n"


def resolve_roster(args: argparse.Namespace) -> tuple[list[str] | None, str]:
    if args.stage == "smoke_2":
        return list(SMOKE_2_ROSTER), "smoke_2"
    if args.stage == "pilot_16":
        return list(PILOT_16_ROSTER), "pilot_16"
    if args.roster_file:
        ids = [line.strip() for line in Path(args.roster_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        return ids, "custom"
    if args.roster_ids:
        ids: list[str] = []
        for chunk in args.roster_ids:
            ids.extend(part.strip() for part in chunk.split(",") if part.strip())
        return ids, "custom"
    return None, "full_240"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--result-subdir")
    parser.add_argument(
        "--stage",
        choices=["smoke_2", "pilot_16", "full_240"],
        default="full_240",
        help="Frozen stage roster. smoke_2 and pilot_16 use pre-registered sample IDs.",
    )
    parser.add_argument("--roster-ids", action="append", help="Explicit sample IDs (repeatable or comma-separated).")
    parser.add_argument("--roster-file", help="File with one sample_id per line.")
    parser.add_argument("--dry-run", action="store_true", help="Explicit dry-run alias for the safe default.")
    parser.add_argument("--run-paid", action="store_true")
    parser.add_argument(
        "--max-new-rows",
        type=int,
        help="Spend brake only. Never use this to define a roster; use --stage or --roster-ids.",
    )
    parser.add_argument("--merge-shards", action="store_true", help="Merge shard files, gate the result, and exit.")
    parser.add_argument(
        "--conditions",
        action="append",
        help="Restrict this worker to these conditions (repeatable or comma-separated). "
        "Used to run E3/E4 as a serial OCR wave so only one process loads PaddleOCR. "
        "Waves must not overlap in time: concurrent workers on the same sample would "
        "race its reasoning cache file.",
    )
    parser.add_argument("--shard-index", type=int, help="0-based shard of this worker.")
    parser.add_argument(
        "--shard-count",
        type=int,
        help="Total shards. Splits by sample (all 7 conditions stay together) and "
        "divides the attempt cap so concurrent workers cannot exceed it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "frozen_manifest.json"
    if args.run_paid and manifest_path.exists():
        manifest = load_json(manifest_path)
    else:
        manifest = build_manifest()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    dry_run = build_dry_run(manifest)
    review = review_manifest(manifest)
    (output_dir / "runner_dry_run.json").write_text(json.dumps(dry_run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "prepaid_review.json").write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote frozen manifest: {manifest_path}")
    print(f"dry_run_status={dry_run['stage_status']}")
    print(f"prepaid_review_status={review['status']}")
    print("paid_calls_executed=0")
    # Merging is a zero-cost local operation, so it must not require --run-paid.
    # It did until 2026-09-18, which meant the documented merge command exited
    # here after printing `paid_calls_executed=0` and a success code, having
    # merged nothing — indistinguishable from a clean merge in a console log.
    if not args.run_paid and not args.merge_shards:
        return 0 if dry_run["stage_status"] == "ready_for_paid_execution" and review["status"] == "no_blocking_findings" else 2
    roster_ids, stage_name = resolve_roster(args)
    subdir = args.result_subdir or stage_name
    if args.merge_shards:
        shard_dir = output_dir / subdir
        # Matches both sharded workers (`paid_result.shard00of08.<wave>.json`)
        # and an unsharded wave (`paid_result.E3-E4.json`), while excluding the
        # merge output `paid_result.json` itself, which has no middle segment.
        # A narrower `shard*` glob would drop the serial OCR wave and merge a
        # 5/7-condition result that still looked internally consistent.
        shard_paths = sorted(shard_dir.glob("paid_result.*.json"))
        if not shard_paths:
            print(f"no shard files found in {shard_dir}")
            return 2
        print(f"merging {len(shard_paths)} worker files:")
        for path in shard_paths:
            print(f"  {path.name}")
        merged = merge_shards(shard_paths, manifest)
        gate = gate_report(
            merged["results"],
            stage=stage_name,
            expected_samples=len(manifest["selected_samples"]),
        )
        merged["gate_report"] = gate
        (shard_dir / "paid_result.json").write_text(
            json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (shard_dir / "analysis.md").write_text(render_analysis(merged), encoding="utf-8")
        print(f"merged {len(shard_paths)} shards -> {merged['completed_row_count']} rows")
        print(f"stage_status={merged['stage_status']}")
        print(f"paid_calls_executed={merged['paid_calls_executed']}")
        print(f"gate_status={gate['status']}")
        for failure in gate["failures"]:
            print(f"  gate_failure={failure}")
        return 0 if gate["status"] == "green" else 3
    requested_conditions: list[str] | None = None
    if args.conditions:
        requested_conditions = []
        for chunk in args.conditions:
            requested_conditions.extend(part.strip() for part in chunk.split(",") if part.strip())
    # Each shard checkpoints to its own file; merge_shards() combines them. The
    # wave tag keeps a condition-restricted run from overwriting the checkpoint
    # of the same shard index in another wave — merge_shards() globs shard*.json
    # and would otherwise silently see only whichever wave wrote last.
    wave = f".{'-'.join(requested_conditions)}" if requested_conditions else ""
    filename = (
        f"paid_result.shard{args.shard_index:02d}of{args.shard_count:02d}{wave}.json"
        if args.shard_count
        else f"paid_result{wave}.json" if wave else "paid_result.json"
    )
    result_path = output_dir / subdir / filename
    if roster_ids:
        print(f"stage={stage_name} roster_size={len(roster_ids)}")
    if args.shard_count:
        print(f"shard={args.shard_index}/{args.shard_count}")
    if requested_conditions:
        print(f"conditions={','.join(requested_conditions)}")
    result = run_paid(
        manifest,
        result_path,
        max_new_rows=args.max_new_rows,
        roster_ids=roster_ids,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        conditions=requested_conditions,
    )
    analysis_dir = result_path.parent
    if args.shard_count or requested_conditions:
        # This worker covers part of the roster, part of the condition set, or
        # both, so stage gates are not evaluable here — gating a wave that ran
        # only E3/E4 against all seven conditions would always fail. Merge first,
        # then gate the merged result. Writing the canonical paid_result.json
        # from a partial worker would also hide the other waves from the merge.
        print(f"stage_status={result['stage_status']}")
        print(f"completed_row_count={result['completed_row_count']}")
        print(f"paid_calls_executed={result['paid_calls_executed']}")
        print("gate_status=deferred_to_merge")
        return 0
    gate = gate_report(
        result["results"],
        stage=stage_name,
        expected_samples=len(roster_ids) if roster_ids else len(manifest["selected_samples"]),
    )
    result["gate_report"] = gate
    (analysis_dir / "paid_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (analysis_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (analysis_dir / "analysis.md").write_text(render_analysis(result), encoding="utf-8")
    print(f"stage_status={result['stage_status']}")
    print(f"completed_row_count={result['completed_row_count']}")
    print(f"paid_calls_executed={result['paid_calls_executed']}")
    print(f"gate_status={gate['status']}")
    if gate["failures"]:
        for failure in gate["failures"]:
            print(f"  gate_failure={failure}")
        # A failed gate must block expansion rather than leave it to judgement.
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
