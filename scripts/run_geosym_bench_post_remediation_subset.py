#!/usr/bin/env python
"""Run Track B GeoSym-Bench post-remediation subset rerun.

Safe default: write dry-run artifacts only. Paid mode executes the frozen
diversity-audited paper subset from
``outputs/paper_level_evidence_package/geosym_bench_controlled/post_remediation_rerun_manifest.json``.
The runner checkpoints after every sample and supports ``--max-new-samples``.
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
from app.reasoning_pipeline import build_chain_text  # noqa: E402
from app.reasoning_cache import load_reasoning_cache  # noqa: E402
from scripts.batch_experiment import ALL_CONDITIONS  # noqa: E402
from scripts.prepare_public_benchmark_pilot import canonical_json_bytes, sha256_hex_bytes, stable_file_hash  # noqa: E402


TRACK_B_DIR = ROOT / "outputs" / "paper_level_evidence_package" / "geosym_bench_controlled"
DEFAULT_MANIFEST = TRACK_B_DIR / "post_remediation_rerun_manifest.json"
DEFAULT_OUTPUT = TRACK_B_DIR / "post_remediation_paid_result.json"
DEFAULT_ANALYSIS = TRACK_B_DIR / "post_remediation_analysis.json"
REASONING_CALLS_PER_SAMPLE = 14
JUDGE_CALLS_PER_SAMPLE = 7
TRANSPORT_RETRY_CAP_PER_SEMANTIC_CALL = 1
OCR_DEPENDENT_CONDITIONS = {"E3", "E4", "E5", "E5_raw"}
OCR_HTTP_ATTEMPTS_PER_SAMPLE_UPPER_BOUND = len(OCR_DEPENDENT_CONDITIONS)
M0_EXECUTION_MODE_DEFAULT = "mathpix_then_clean_grid_fallback"
M0_EXECUTION_MODE_FALLBACK_FIRST = "clean_grid_fallback_first"
REASONING_CALLS_BY_CONDITION = {
    "E0": 1,
    "E1": 1,
    "E2": 3,
    "E3": 1,
    "E4": 1,
    "E5": 4,
    "E5_raw": 3,
}
JUDGE_CALLS_BY_CONDITION = {condition: 1 for condition in ALL_CONDITIONS}


class AttemptCounter:
    def __init__(self) -> None:
        self.http_attempts = 0


class OcrAttemptCounter:
    def __init__(self, cap: int) -> None:
        self.http_attempts = 0
        self.cap = cap


@contextmanager
def count_mathpix_http_attempts(counter: OcrAttemptCounter) -> Iterator[None]:
    """Count Mathpix transport attempts and force one attempt per OCR call."""
    import recognizer.mathpix.mathpix_image_ocr as mathpix_ocr

    original_post = mathpix_ocr.requests.post
    original_retries = mathpix_ocr._MAX_RETRIES

    def counted_post(*args: Any, **kwargs: Any) -> Any:
        if counter.http_attempts >= counter.cap:
            raise RuntimeError("ocr_http_attempt_cap_exhausted")
        counter.http_attempts += 1
        return original_post(*args, **kwargs)

    mathpix_ocr.requests.post = counted_post
    mathpix_ocr._MAX_RETRIES = 1
    try:
        yield
    finally:
        mathpix_ocr.requests.post = original_post
        mathpix_ocr._MAX_RETRIES = original_retries


@contextmanager
def count_llm_http_attempts(counter: AttemptCounter) -> Iterator[None]:
    """Count actual LLM transport attempts made through BaseLLMAdapter."""
    original_call_with_retry = adapter_base.BaseLLMAdapter._call_with_retry

    def counted_call_with_retry(self, image_b64: str, prompt: str) -> str:
        last_exc = None
        for attempt in range(1, adapter_base._MAX_API_RETRIES + 1):
            counter.http_attempts += 1
            try:
                return self._call_api(image_b64, prompt)
            except Exception as exc:
                last_exc = exc
                if attempt >= adapter_base._MAX_API_RETRIES or not adapter_base.is_transient_error(exc):
                    raise
                if adapter_base._is_rate_limit_error(exc):
                    delay = min(
                        adapter_base._BASE_RETRY_DELAY * (3 ** (attempt - 1)),
                        adapter_base._MAX_RETRY_DELAY,
                    )
                else:
                    delay = min(
                        adapter_base._BASE_RETRY_DELAY * (2 ** (attempt - 1)),
                        adapter_base._MAX_RETRY_DELAY,
                    )
                delay *= 0.5 + adapter_base.random.random()
                adapter_base.logger.warning(
                    "[%s] API attempt %d/%d failed (%s: %s), retry in %.1fs",
                    self.model_name, attempt, adapter_base._MAX_API_RETRIES,
                    type(exc).__name__, str(exc)[:200], delay,
                )
                time.sleep(delay)
        raise last_exc

    adapter_base.BaseLLMAdapter._call_with_retry = counted_call_with_retry
    try:
        yield
    finally:
        adapter_base.BaseLLMAdapter._call_with_retry = original_call_with_retry


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_requires_reauthorization(checkpoint: dict[str, Any]) -> bool:
    status = str(checkpoint.get("stage_status") or "")
    blockers = [str(finding) for finding in (checkpoint.get("blocking_findings") or [])]
    has_unfinished_sample = checkpoint.get("in_progress_sample") is not None
    unknown_blocker = any("unknown" in finding.lower() or "unaccount" in finding.lower() for finding in blockers)
    unknown_status = "unknown" in status or "unaccount" in status
    return bool(status == "running_paid_run" or has_unfinished_sample or unknown_blocker or unknown_status)


def sample_image_name(sample: dict[str, Any]) -> str:
    family = sample["family"]
    sample_id = sample["sample_id"]
    profile_dir = f"{family}-clean_render"
    if sample_id.startswith(f"{profile_dir}/"):
        return sample_id
    return f"{profile_dir}/{sample_id}"


def parse_conditions(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return list(ALL_CONDITIONS)
    conditions = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [condition for condition in conditions if condition not in ALL_CONDITIONS]
    if invalid:
        raise ValueError(f"unsupported Track B conditions: {', '.join(invalid)}")
    if len(set(conditions)) != len(conditions):
        raise ValueError("duplicate Track B conditions are not allowed")
    return conditions


def selected_conditions(manifest: dict[str, Any]) -> list[str]:
    return parse_conditions(",".join(manifest.get("conditions") or list(ALL_CONDITIONS)))


def m0_execution_mode(manifest: dict[str, Any]) -> str:
    mode = str(manifest.get("m0_execution_mode") or M0_EXECUTION_MODE_DEFAULT)
    if mode not in {M0_EXECUTION_MODE_DEFAULT, M0_EXECUTION_MODE_FALLBACK_FIRST}:
        raise ValueError(f"unsupported m0_execution_mode: {mode}")
    return mode


def call_caps(sample_count: int, conditions: list[str], *, clean_grid_fallback_first: bool = False) -> dict[str, int]:
    ocr_conditions = set() if clean_grid_fallback_first else (OCR_DEPENDENT_CONDITIONS & set(conditions))
    return {
        "sample_condition_job_cap": sample_count * len(conditions),
        "reasoning_llm_call_cap": sample_count * sum(REASONING_CALLS_BY_CONDITION[condition] for condition in conditions),
        "judge_llm_call_cap": sample_count * sum(JUDGE_CALLS_BY_CONDITION[condition] for condition in conditions),
        "ocr_http_attempt_cap": sample_count * len(ocr_conditions),
    }


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("sample_id"))


def plan_hash(manifest: dict[str, Any]) -> str:
    material = {k: v for k, v in manifest.items() if k != "created_at_utc"}
    return sha256_hex_bytes(canonical_json_bytes(material))


def build_dry_run(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    subset = manifest.get("source_artifacts", {}).get("metadata_dedup_audit", {})
    samples = load_json(Path(subset["path"])).get("selected_subset") if subset.get("path") else []
    conditions = selected_conditions(manifest)
    mode = m0_execution_mode(manifest)
    clean_grid_first = mode == M0_EXECUTION_MODE_FALLBACK_FIRST
    expected_caps = call_caps(len(samples or []), conditions, clean_grid_fallback_first=clean_grid_first)
    findings = []
    if manifest.get("stage_status") not in {
        "geosym_bench_rerun_manifest_frozen_pending_paid_execution",
        "geosym_bench_blocked_missing_post_remediation_subset_runner",
    }:
        findings.append(f"manifest_not_ready:{manifest.get('stage_status')}")
    if len(samples or []) != manifest.get("N_paper_subset"):
        findings.append("selected_subset_count_mismatch")
    ocr_dependent = [] if clean_grid_first else sorted(OCR_DEPENDENT_CONDITIONS & set(conditions))
    if manifest.get("ocr_http_attempt_cap") == 0 and ocr_dependent:
        findings.append(f"ocr_zero_cap_conflicts_with_current_conditions:{','.join(ocr_dependent)}")
    if manifest.get("ocr_http_attempt_cap") is not None and manifest.get("ocr_http_attempt_cap") < expected_caps["ocr_http_attempt_cap"]:
        findings.append("ocr_cap_below_conservative_upper_bound")
    if manifest.get("reasoning_llm_call_cap") != expected_caps["reasoning_llm_call_cap"]:
        findings.append("reasoning_cap_mismatch")
    if manifest.get("judge_llm_call_cap") != expected_caps["judge_llm_call_cap"]:
        findings.append("judge_cap_mismatch")
    return {
        "schema_version": "geosym_bench_post_remediation_subset_dry_run_v1",
        "created_at_utc": utc_now(),
        "stage": "track_b_geosym_bench_post_remediation_subset_dry_run",
        "stage_status": "ready_for_paid_execution" if not findings else "blocking_findings_present",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": stable_file_hash(manifest_path),
        "plan_sha256": plan_hash(manifest),
        "paid_calls_executed": 0,
        "actual_reasoning_semantic_calls": 0,
        "actual_judge_semantic_calls": 0,
        "actual_ocr_http_attempts": 0,
        "N_paper_subset": len(samples or []),
        "conditions": conditions,
        "m0_execution_mode": mode,
        "output_conditions": list(ALL_CONDITIONS),
        "sample_condition_job_cap": expected_caps["sample_condition_job_cap"],
        "reasoning_llm_call_cap": manifest.get("reasoning_llm_call_cap"),
        "judge_llm_call_cap": manifest.get("judge_llm_call_cap"),
        "ocr_http_attempt_cap": manifest.get("ocr_http_attempt_cap"),
        "ocr_dependent_conditions": ocr_dependent,
        "transport_retry_cap_per_semantic_call": TRANSPORT_RETRY_CAP_PER_SEMANTIC_CALL,
        "blocking_findings": findings,
    }


def condition_row(sample: dict[str, Any], cache: dict[str, Any], condition: str) -> dict[str, Any]:
    cond = (cache.get("conditions") or {}).get(condition) or {}
    ev = cond.get("eval") or {}
    judge = cond.get("judge") or {}
    pipeline = cond.get("pipeline") or {}
    chain_text = build_chain_text(pipeline) if pipeline else ""
    return {
        "sample_id": sample["sample_id"],
        "image_name": sample_image_name(sample),
        "family": sample["family"],
        "dedup_key": sample.get("dedup_key"),
        "condition": condition,
        "status": cond.get("status"),
        "cached": cond.get("cached", False),
        "total_score": ev.get("total_score"),
        "dimension_scores": ev.get("dimension_scores"),
        "judge_inference_score": (ev.get("sub_scores") or {}).get("judge_inference_score", {}).get("score"),
        "chain_text_sha256": sha256_hex_bytes(canonical_json_bytes(chain_text)),
        "raw_judge_response_sha256": sha256_hex_bytes(canonical_json_bytes(judge.get("raw_judge_response") or "")) if judge else None,
        "error": cond.get("error"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family_condition: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_family_condition[row["family"]][row["condition"]].append(row)

    def row_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        scores = [float(row["total_score"]) for row in items if row.get("total_score") is not None]
        return {
            "rows": len(items),
            "success": sum(1 for row in items if row.get("status") == "success"),
            "mean_total_score": round(sum(scores) / len(scores), 6) if scores else None,
        }

    return {
        "overall_by_condition": {condition: row_summary(rows) for condition, rows in by_condition.items()},
        "by_family_condition": {
            family: {condition: row_summary(items) for condition, items in conds.items()}
            for family, conds in by_family_condition.items()
        },
    }


def write_result(
    manifest: dict[str, Any],
    manifest_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    completed_samples: list[dict[str, Any]],
    status: str,
    started_at_utc: str,
    actual_llm_http_attempts: int,
    actual_ocr_http_attempts: int,
    in_progress_sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conditions = selected_conditions(manifest)
    cap_sample_count = len(completed_samples) + (1 if in_progress_sample is not None else 0)
    clean_grid_first = m0_execution_mode(manifest) == M0_EXECUTION_MODE_FALLBACK_FIRST
    nominal_caps = call_caps(cap_sample_count, conditions, clean_grid_fallback_first=clean_grid_first)
    nominal_semantic_calls = nominal_caps["reasoning_llm_call_cap"] + nominal_caps["judge_llm_call_cap"]
    result = {
        "schema_version": "geosym_bench_post_remediation_subset_result_v1",
        "created_at_utc": utc_now(),
        "stage": "track_b_geosym_bench_post_remediation_subset_rerun",
        "stage_status": status,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": stable_file_hash(manifest_path),
        "plan_sha256": plan_hash(manifest),
        "started_at_utc": started_at_utc,
        "N_paper_subset": manifest.get("N_paper_subset"),
        "completed_sample_count": len(completed_samples),
        "in_progress_sample": in_progress_sample,
        "conditions": conditions,
        "m0_execution_mode": m0_execution_mode(manifest),
        "output_conditions": list(ALL_CONDITIONS),
        "paid_calls_executed": actual_llm_http_attempts,
        "paid_call_accounting_mode": "actual_llm_http_attempts_with_retry_cap_1",
        "actual_reasoning_semantic_calls": None,
        "actual_judge_semantic_calls": None,
        "actual_llm_http_attempts": actual_llm_http_attempts,
        "nominal_reasoning_semantic_call_upper_bound": nominal_caps["reasoning_llm_call_cap"],
        "nominal_judge_semantic_call_upper_bound": nominal_caps["judge_llm_call_cap"],
        "nominal_llm_semantic_call_upper_bound": nominal_semantic_calls,
        "actual_ocr_http_attempts": actual_ocr_http_attempts,
        "ocr_http_attempt_accounting_mode": "mathpix_requests_post_counter",
        "reasoning_llm_call_cap": manifest.get("reasoning_llm_call_cap"),
        "judge_llm_call_cap": manifest.get("judge_llm_call_cap"),
        "sample_condition_job_cap": manifest.get("sample_condition_job_cap"),
        "governance": {
            "within_nominal_reasoning_call_cap": nominal_caps["reasoning_llm_call_cap"] <= int(manifest.get("reasoning_llm_call_cap") or 0),
            "within_nominal_judge_call_cap": nominal_caps["judge_llm_call_cap"] <= int(manifest.get("judge_llm_call_cap") or 0),
            "within_total_http_attempt_cap": actual_llm_http_attempts <= (
                int(manifest.get("reasoning_llm_call_cap") or 0)
                + int(manifest.get("judge_llm_call_cap") or 0)
            ),
            "within_ocr_http_attempt_cap": actual_ocr_http_attempts <= int(manifest.get("ocr_http_attempt_cap") or 0),
            "ocr_http_attempt_accounting_mode": "mathpix_requests_post_counter",
            "sample_replacement": False,
            "conditions_frozen": selected_conditions(manifest) == conditions,
        },
        "summary": summarize(rows),
        "completed_samples": completed_samples,
        "results": rows,
        "interpretation_limits": manifest.get("interpretation_limits", []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def run_paid(
    manifest: dict[str, Any],
    manifest_path: Path,
    output_path: Path,
    max_new_samples: int | None,
    restart_after_unaccounted_interruption: bool = False,
) -> dict[str, Any]:
    dry_run = build_dry_run(manifest, manifest_path)
    if dry_run["stage_status"] != "ready_for_paid_execution":
        raise ValueError(f"dry_run_blocking_findings:{dry_run['blocking_findings']}")
    reasoning_model = manifest["reasoning_model_provenance"]["display_name"]
    judge_model = manifest["judge_model_provenance"]["display_name"]
    reasoning_key = _get_saved_key(reasoning_model)
    judge_key = _get_saved_key(judge_model)
    if not reasoning_key:
        raise ValueError(f"saved API key not found for reasoning model {reasoning_model}")
    if not judge_key:
        raise ValueError(f"saved API key not found for judge model {judge_model}")

    subset_path = Path(manifest["source_artifacts"]["metadata_dedup_audit"]["path"])
    samples = load_json(subset_path)["selected_subset"]
    conditions = selected_conditions(manifest)
    rows: list[dict[str, Any]] = []
    completed_samples: list[dict[str, Any]] = []
    attempt_counter = AttemptCounter()
    ocr_attempt_counter = OcrAttemptCounter(int(manifest.get("ocr_http_attempt_cap") or 0))
    if output_path.exists():
        checkpoint = load_json(output_path)
        if checkpoint_requires_reauthorization(checkpoint) and not restart_after_unaccounted_interruption:
            raise ValueError(
                "checkpoint_has_unknown_attempt_accounting;"
                " explicit_reauthorization_required_before_restart"
            )
        if checkpoint.get("plan_sha256") != plan_hash(manifest):
            raise ValueError("checkpoint_plan_hash_mismatch")
        rows = list(checkpoint.get("results") or [])
        completed_samples = list(checkpoint.get("completed_samples") or [])
        attempt_counter.http_attempts = int(checkpoint.get("actual_llm_http_attempts") or 0)
        ocr_attempt_counter.http_attempts = int(checkpoint.get("actual_ocr_http_attempts") or 0)
    completed_ids = {row_key(row) for row in completed_samples}
    new_samples = 0
    started_at_utc = utc_now()
    original_retries = adapter_base._MAX_API_RETRIES
    adapter_base._MAX_API_RETRIES = TRANSPORT_RETRY_CAP_PER_SEMANTIC_CALL
    os.environ["GEOSYM_OPENAI_TIMEOUT_SECONDS"] = str(manifest.get("openai_timeout_seconds") or 120)
    os.environ["GEOSYM_QWEN_TIMEOUT_SECONDS"] = str(manifest.get("qwen_timeout_seconds") or 90)
    clean_grid_first = m0_execution_mode(manifest) == M0_EXECUTION_MODE_FALLBACK_FIRST
    original_fallback_first = os.environ.get("GEOSYM_CLEAN_GRID_FALLBACK_FIRST")
    if clean_grid_first:
        os.environ["GEOSYM_CLEAN_GRID_FALLBACK_FIRST"] = "1"
    in_progress_sample: dict[str, Any] | None = None
    try:
        with count_llm_http_attempts(attempt_counter):
            with count_mathpix_http_attempts(ocr_attempt_counter):
                for sample in samples:
                    if sample["sample_id"] in completed_ids:
                        continue
                    if max_new_samples is not None and new_samples >= max_new_samples:
                        return write_result(
                            manifest, manifest_path, output_path, rows, completed_samples,
                            "partial_paid_run", started_at_utc, attempt_counter.http_attempts,
                            ocr_attempt_counter.http_attempts,
                        )
                    next_caps = call_caps(
                        len(completed_samples) + 1,
                        conditions,
                        clean_grid_fallback_first=clean_grid_first,
                    )
                    next_reasoning = next_caps["reasoning_llm_call_cap"]
                    next_judge = next_caps["judge_llm_call_cap"]
                    next_total_http_attempt_cap = next_reasoning + next_judge
                    if (
                        next_reasoning > int(manifest["reasoning_llm_call_cap"])
                        or next_judge > int(manifest["judge_llm_call_cap"])
                        or attempt_counter.http_attempts >= next_total_http_attempt_cap
                    ):
                        return write_result(
                            manifest, manifest_path, output_path, rows, completed_samples,
                            "stopped_at_paid_cap", started_at_utc, attempt_counter.http_attempts,
                            ocr_attempt_counter.http_attempts,
                        )
                    image_name = sample_image_name(sample)
                    sample_started = time.perf_counter()
                    attempts_before_sample = attempt_counter.http_attempts
                    ocr_attempts_before_sample = ocr_attempt_counter.http_attempts
                    in_progress_sample = {
                        "sample_id": sample["sample_id"],
                        "family": sample["family"],
                        "image_name": image_name,
                        "started_at_utc": utc_now(),
                    }
                    write_result(
                        manifest, manifest_path, output_path, rows, completed_samples,
                        "running_paid_run", started_at_utc, attempt_counter.http_attempts,
                        ocr_attempt_counter.http_attempts,
                        in_progress_sample,
                    )
                    print(f"starting_sample={sample['sample_id']} image_name={image_name}", flush=True)
                    error = ""
                    try:
                        gen = run_reasoning_all_conditions(
                            image_name=image_name,
                            model_name=reasoning_model,
                            api_key=reasoning_key,
                            judge_model_name=judge_model,
                            judge_api_key=judge_key,
                            lang="en",
                            force_rerun=True,
                            target_conditions=conditions,
                        )
                        for _ in gen:
                            pass
                        cache = load_reasoning_cache(image_name, reasoning_model) or {}
                    except Exception as exc:
                        cache = {"conditions": {}}
                        error = f"{type(exc).__name__}: {exc}"
                    sample_rows = [condition_row(sample, cache, condition) for condition in ALL_CONDITIONS]
                    if error:
                        for row in sample_rows:
                            row["status"] = row.get("status") or "runner_error"
                            row["error"] = row.get("error") or {"type": "RunnerError", "message": error}
                    rows.extend(sample_rows)
                    completed = {
                        "sample_id": sample["sample_id"],
                        "family": sample["family"],
                        "image_name": image_name,
                        "duration_s": round(time.perf_counter() - sample_started, 3),
                        "status": "success" if not error else "runner_error",
                        "error": error,
                        "llm_http_attempts": attempt_counter.http_attempts - attempts_before_sample,
                        "ocr_http_attempts": ocr_attempt_counter.http_attempts - ocr_attempts_before_sample,
                    }
                    completed_samples.append(completed)
                    completed_ids.add(sample["sample_id"])
                    new_samples += 1
                    result = write_result(
                        manifest, manifest_path, output_path, rows, completed_samples,
                        "partial_paid_run", started_at_utc, attempt_counter.http_attempts,
                        ocr_attempt_counter.http_attempts,
                    )
                    in_progress_sample = None
                    print(
                        f"completed_sample={sample['sample_id']} completed_samples={result['completed_sample_count']}/{manifest['N_paper_subset']} "
                        f"semantic_calls={result['paid_calls_executed']}/{int(manifest['reasoning_llm_call_cap']) + int(manifest['judge_llm_call_cap'])} "
                        f"http_attempts={result['actual_llm_http_attempts']}/{int(manifest['reasoning_llm_call_cap']) + int(manifest['judge_llm_call_cap'])} "
                        f"ocr_attempts={result['actual_ocr_http_attempts']}/{int(manifest['ocr_http_attempt_cap'])}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        return write_result(
            manifest, manifest_path, output_path, rows, completed_samples,
            "interrupted_paid_run", started_at_utc, attempt_counter.http_attempts,
            ocr_attempt_counter.http_attempts,
            in_progress_sample,
        )
    finally:
        adapter_base._MAX_API_RETRIES = original_retries
        if original_fallback_first is None:
            os.environ.pop("GEOSYM_CLEAN_GRID_FALLBACK_FIRST", None)
        else:
            os.environ["GEOSYM_CLEAN_GRID_FALLBACK_FIRST"] = original_fallback_first
    status = "geosym_bench_post_remediation_rerun_completed" if len(completed_samples) == manifest.get("N_paper_subset") else "partial_paid_run"
    return write_result(
        manifest, manifest_path, output_path, rows, completed_samples, status,
        started_at_utc, attempt_counter.http_attempts, ocr_attempt_counter.http_attempts,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--analysis", default=str(DEFAULT_ANALYSIS))
    parser.add_argument("--run-paid", action="store_true")
    parser.add_argument("--max-new-samples", type=int)
    parser.add_argument("--conditions", help="Comma-separated Track B conditions to execute; default is all conditions.")
    parser.add_argument(
        "--restart-after-unaccounted-interruption",
        action="store_true",
        help="Allow resuming/restarting from a checkpoint with unknown attempt accounting. Use only after explicit cap reauthorization.",
    )
    args = parser.parse_args(argv)
    try:
        parse_conditions(args.conditions)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)
    if args.conditions:
        conditions = parse_conditions(args.conditions)
        subset = manifest.get("source_artifacts", {}).get("metadata_dedup_audit", {})
        samples = load_json(Path(subset["path"])).get("selected_subset") if subset.get("path") else []
        manifest["conditions"] = conditions
        manifest.update(
            call_caps(
                len(samples or []),
                conditions,
                clean_grid_fallback_first=m0_execution_mode(manifest) == M0_EXECUTION_MODE_FALLBACK_FIRST,
            )
        )
    output_path = Path(args.output)
    analysis_path = Path(args.analysis)
    dry_run = build_dry_run(manifest, manifest_path)
    dry_path = output_path.parent / "post_remediation_dry_run.json"
    dry_path.write_text(json.dumps(dry_run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote Track B dry-run: {dry_path}")
    print(f"dry_run_status={dry_run['stage_status']}")
    print(f"paid_calls_executed=0")
    if not args.run_paid:
        return 0 if dry_run["stage_status"] == "ready_for_paid_execution" else 2
    result = run_paid(
        manifest,
        manifest_path,
        output_path,
        args.max_new_samples,
        args.restart_after_unaccounted_interruption,
    )
    analysis_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"stage_status={result['stage_status']}")
    print(f"completed_sample_count={result['completed_sample_count']}")
    print(f"paid_calls_executed={result['paid_calls_executed']}")
    print(f"actual_ocr_http_attempts={result['actual_ocr_http_attempts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
