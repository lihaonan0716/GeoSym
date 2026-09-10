"""Bounded paid pilot for the P1 matched baseline prompts.

Default mode is dry-run and performs no paid calls. Use ``--run-paid`` only
after the frozen cap and sample roster in this file are acceptable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluator.adapters.base as adapter_base  # noqa: E402
from app.helpers import _create_llm_adapter, _get_saved_key  # noqa: E402
from app.reasoning_pipeline import CONDITION_MAP, run_single_condition  # noqa: E402


MODEL_NAME = "GPT-5.5"
SAMPLE_ID = "quadratic-clean_render/quadratic-clean_render_0000"
CONDITIONS = ("E0", "E1", "E3", "E4")
MAX_SAMPLE_LEVEL_JOBS = 4
MAX_HTTP_ATTEMPTS_PER_JOB = 1
OUTPUT_PATH = ROOT / "outputs" / "p1_3_prompt_pilot" / "pilot_report.json"


def _load_ground_truth(sample_id: str) -> tuple[Path, dict[str, Any]]:
    image_path = ROOT / "dataset" / "images" / f"{sample_id}.png"
    truth_path = ROOT / "dataset" / "ground_truth" / f"{sample_id}.json"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not truth_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {truth_path}")
    gt = json.loads(truth_path.read_text(encoding="utf-8"))
    gt_data = gt.get("question", {}).get("ground_truth", {})
    if not gt_data:
        raise ValueError(f"Missing question.ground_truth in {truth_path}")
    return image_path, gt_data


def frozen_plan() -> dict[str, Any]:
    return {
        "pilot": "P1-3 matched baseline prompt paid pilot",
        "hypothesis": (
            "After P1-1/P1-2, each modified E0/E1/E3/E4 baseline prompt can "
            "complete one real chart reasoning call and return parseable final JSON."
        ),
        "interpretation_limits": [
            "This pilot only validates prompt/API usability.",
            "It does not estimate benchmark accuracy.",
            "It does not validate E2 separately because E2 inherits the E1 prompt and uses three calls.",
            "It does not authorize full reruns or P2.",
        ],
        "model_name": MODEL_NAME,
        "sample_roster": [SAMPLE_ID],
        "conditions": list(CONDITIONS),
        "sample_level_job_cap": MAX_SAMPLE_LEVEL_JOBS,
        "http_attempt_cap_per_job": MAX_HTTP_ATTEMPTS_PER_JOB,
        "total_http_attempt_cap": MAX_SAMPLE_LEVEL_JOBS * MAX_HTTP_ATTEMPTS_PER_JOB,
        "decoding": "adapter/provider default; no temperature override in this script",
        "output_path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
        "cost_estimate": "4 vision calls through 9router; expected to be a small pilot cost, below a full benchmark rerun by >100x.",
        "stopping_rule": "Stop after the four frozen condition jobs, regardless of success or failure.",
    }


def _missing_fields(parsed: dict[str, Any]) -> list[str]:
    required = ["expression", "domain", "range", "inference_explanation"]
    return [field for field in required if not parsed.get(field)]


def run_paid_pilot() -> dict[str, Any]:
    if len(CONDITIONS) > MAX_SAMPLE_LEVEL_JOBS:
        raise RuntimeError("Frozen condition list exceeds sample-level job cap.")

    key = _get_saved_key(MODEL_NAME)
    if not key:
        raise RuntimeError("Missing API key for GPT-5.5 / ninerouter.")

    image_path, gt_data = _load_ground_truth(SAMPLE_ID)
    adapter = _create_llm_adapter(MODEL_NAME, key)
    if adapter is None:
        raise RuntimeError(f"Unsupported model: {MODEL_NAME}")

    original_max_retries = adapter_base._MAX_API_RETRIES
    adapter_base._MAX_API_RETRIES = MAX_HTTP_ATTEMPTS_PER_JOB
    jobs: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc).isoformat()
    try:
        for condition in CONDITIONS:
            job_started = time.perf_counter()
            error = None
            parsed: dict[str, Any] | None = None
            raw_chars = 0
            try:
                pipeline = run_single_condition(
                    adapter,
                    image_path,
                    gt_data,
                    CONDITION_MAP[condition],
                )
                parsed = pipeline.get("final_parsed") or {}
                raw = ""
                for step in (pipeline.get("steps") or {}).values():
                    raw += str(step.get("raw", ""))
                raw_chars = len(raw)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            elapsed_ms = round((time.perf_counter() - job_started) * 1000, 2)
            missing = _missing_fields(parsed or {})
            jobs.append(
                {
                    "condition": condition,
                    "model": getattr(adapter, "model_name", MODEL_NAME),
                    "sample_id": SAMPLE_ID,
                    "http_attempt_cap": MAX_HTTP_ATTEMPTS_PER_JOB,
                    "elapsed_ms": elapsed_ms,
                    "raw_chars": raw_chars,
                    "parseable": parsed is not None and error is None,
                    "missing_required_fields": missing,
                    "passed": error is None and not missing,
                    "error": error,
                }
            )
    finally:
        adapter_base._MAX_API_RETRIES = original_max_retries

    report = {
        "plan": frozen_plan(),
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_sample_level_jobs": len(jobs),
        "actual_http_attempt_cap": len(jobs) * MAX_HTTP_ATTEMPTS_PER_JOB,
        "jobs": jobs,
        "passed": all(job["passed"] for job in jobs) and len(jobs) == len(CONDITIONS),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run or print the frozen P1-3 matched-prompt pilot."
    )
    parser.add_argument(
        "--run-paid",
        action="store_true",
        help="Perform the four frozen paid calls. Omit for dry-run.",
    )
    args = parser.parse_args()

    if not args.run_paid:
        print(json.dumps(frozen_plan(), indent=2, ensure_ascii=False))
        return 0

    report = run_paid_pilot()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
