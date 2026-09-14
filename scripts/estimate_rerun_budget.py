"""Estimate the P1-4 matched-baseline full-rerun budget.

This script is zero-cost and performs no model calls. It exists to make the
full-rerun approval boundary explicit after P1-3 passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "dataset"
CONDITIONS = ("E0", "E1", "E2", "E3", "E4")
FAMILIES = (
    "absolute",
    "cubic",
    "exponential",
    "linear",
    "logarithmic",
    "piecewise_linear",
    "quadratic",
    "rational",
)
PROFILE = "clean_render"
SAMPLES_PER_FAMILY = 60
REASONING_CALLS_PER_CONDITION = {
    "E0": 1,
    "E1": 1,
    "E2": 3,
    "E3": 1,
    "E4": 1,
}
JUDGE_CALLS_PER_CONDITION = {
    "E0": 1,
    "E1": 1,
    "E2": 1,
    "E3": 1,
    "E4": 1,
}
ADAPTER_HTTP_ATTEMPT_CAP_PER_CALL = 5


def _count_samples(dataset_dir: Path, family: str) -> int:
    image_dir = dataset_dir / "images" / f"{family}-{PROFILE}"
    if not image_dir.exists():
        return 0
    return len(list(image_dir.glob("*.png")))


def estimate_budget(dataset_dir: Path = DEFAULT_DATASET_DIR) -> dict[str, object]:
    family_counts = {family: _count_samples(dataset_dir, family) for family in FAMILIES}
    capped_family_counts = {
        family: min(count, SAMPLES_PER_FAMILY)
        for family, count in family_counts.items()
    }
    total_samples = sum(capped_family_counts.values())
    condition_jobs = total_samples * len(CONDITIONS)
    reasoning_calls = total_samples * sum(
        REASONING_CALLS_PER_CONDITION[condition]
        for condition in CONDITIONS
    )
    judge_calls = total_samples * sum(
        JUDGE_CALLS_PER_CONDITION[condition]
        for condition in CONDITIONS
    )
    http_calls = reasoning_calls + judge_calls
    return {
        "scope": "P1-4 matched-baseline full rerun estimate",
        "dataset_dir": dataset_dir.relative_to(ROOT).as_posix()
        if dataset_dir.is_relative_to(ROOT)
        else str(dataset_dir),
        "profile": PROFILE,
        "families": list(FAMILIES),
        "conditions": list(CONDITIONS),
        "samples_per_family_cap": SAMPLES_PER_FAMILY,
        "available_samples_by_family": family_counts,
        "planned_samples_by_family": capped_family_counts,
        "total_samples": total_samples,
        "condition_jobs": condition_jobs,
        "reasoning_calls": reasoning_calls,
        "judge_calls": judge_calls,
        "nominal_http_calls": http_calls,
        "adapter_http_attempt_cap_per_call": ADAPTER_HTTP_ATTEMPT_CAP_PER_CALL,
        "worst_case_http_attempt_cap": http_calls * ADAPTER_HTTP_ATTEMPT_CAP_PER_CALL,
        "notes": [
            "E2 performs three deterministic prompt-variant calls and one judge call.",
            "E0/E1/E3/E4 each perform one reasoning call and one judge call.",
            "This estimate excludes E5/E5_raw and does not authorize running the rerun.",
            "A future run must freeze model, judge model, cache policy, retry cap, and output path.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the zero-cost P1-4 matched-baseline rerun budget."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()

    print(json.dumps(estimate_budget(args.dataset_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
