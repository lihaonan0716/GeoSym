#!/usr/bin/env python
"""Audit a frozen PlotQA public benchmark plan before paid execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_plotqa_public_benchmark_pilot import (  # noqa: E402
    DEFAULT_CONDITIONS,
    SUPPORTED_CHART_TYPES,
    SUPPORTED_REASONING_TYPES,
    validate_plotqa_sample_semantics,
)


def audit_plan(plan: dict[str, Any]) -> dict[str, Any]:
    violations = []
    frozen = plan.get("frozen_before_paid_run", {})
    selection = frozen.get("sample_selection", {})
    official = frozen.get("official_files", {})
    model_provenance = frozen.get("model_provenance")
    samples = plan.get("selected_samples") or []
    if plan.get("benchmark") != "PlotQA":
        violations.append("benchmark_not_plotqa")
    if plan.get("paid_run_approved") is not True:
        violations.append("paid_run_not_approved")
    if plan.get("zero_cost_only") is True and plan.get("paid_run_approved") is True:
        violations.append("paid_approved_plan_marked_zero_cost_only")
    if selection.get("sampling_frame") != "official_plotqa_full_split_hash_pool":
        violations.append("not_official_full_split_hash_pool")
    for key in ("annotations_sha256", "qa_pairs_sha256", "image_manifest_sha256"):
        if not official.get(key):
            violations.append(f"missing_{key}")
    sample_count = len(samples)
    if sample_count not in {8, 50}:
        violations.append("selected_roster_not_supported_size")
    if len({sample.get("metadata", {}).get("image_index") for sample in samples}) != len(samples):
        violations.append("selected_image_index_not_unique")
    if frozen.get("conditions") != list(DEFAULT_CONDITIONS):
        violations.append("unexpected_condition_list")
    expected_jobs = sample_count * len(list(DEFAULT_CONDITIONS))
    expected_llm_attempts = sample_count * 5
    if sample_count in {8, 50} and frozen.get("sample_condition_job_cap") != expected_jobs:
        violations.append("sample_condition_job_cap_mismatch")
    if sample_count in {8, 50} and frozen.get("llm_http_attempt_cap") != expected_llm_attempts:
        violations.append("llm_http_attempt_cap_mismatch")
    if frozen.get("ocr_http_attempt_cap") != 0:
        violations.append("ocr_not_disabled")
    required_model_keys = (
        "display_name",
        "provider",
        "adapter",
        "api_model_id",
        "api_key_name",
        "gateway_policy",
        "config_path",
        "config_sha256",
    )
    if not isinstance(model_provenance, dict):
        violations.append("missing_model_provenance")
        model_provenance = {}
    for key in required_model_keys:
        if not model_provenance.get(key):
            violations.append(f"missing_model_provenance_{key}")
    if model_provenance.get("api_model_id") and frozen.get("reasoning_model") != model_provenance.get("api_model_id"):
        violations.append("reasoning_model_does_not_match_model_provenance")
    if plan.get("paid_calls_executed") != 0:
        violations.append("plan_records_paid_calls")
    excluded_summary = plan.get("excluded_summary", {})
    if isinstance(excluded_summary, dict) and excluded_summary.get("eligible"):
        violations.append("excluded_summary_contains_eligible")

    per_sample = []
    for sample in samples:
        metadata = sample.get("metadata") or {}
        sample_violations = []
        for key in ("sample_id", "image_path", "question", "answer", "answer_type", "normalized_answer", "metadata"):
            if key not in sample:
                sample_violations.append(f"missing_{key}")
        if metadata.get("chart_type") not in SUPPORTED_CHART_TYPES:
            sample_violations.append(f"unsupported_chart_type:{metadata.get('chart_type')}")
        for key in ("image_readable", "axis_visible", "has_x_axis", "has_y_axis", "has_tick_labels", "has_data_table"):
            if metadata.get(key) is not True:
                sample_violations.append(f"{key}_not_true")
        if metadata.get("legend_required") is True and metadata.get("legend_available") is not True:
            sample_violations.append("legend_required_missing")
        if metadata.get("reasoning_type") not in SUPPORTED_REASONING_TYPES:
            sample_violations.append(f"unsupported_reasoning_type:{metadata.get('reasoning_type')}")
        if metadata.get("quantitative_reasoning_required") is not True:
            sample_violations.append("quantitative_reasoning_required_not_true")
        sample_violations.extend(validate_plotqa_sample_semantics(sample))
        if metadata.get("geosym_compatibility") != "presenter_candidate":
            sample_violations.append(f"geosym_compatibility:{metadata.get('geosym_compatibility')}")
        if metadata.get("image_index") in {None, ""} or metadata.get("qa_index") in {None, ""}:
            sample_violations.append("missing_stable_plotqa_indices")
        if sample_violations:
            violations.extend(f"{sample.get('sample_id')}:{item}" for item in sample_violations)
        per_sample.append({"sample_id": sample.get("sample_id"), "violations": sample_violations})

    return {
        "schema_version": "plotqa_plan_audit_v1",
        "valid": not violations,
        "violations": violations,
        "per_sample": per_sample,
        "selected_ids": [sample.get("sample_id") for sample in samples],
        "paid_calls_executed": 0,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    report = audit_plan(plan)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote PlotQA plan audit: {output}")
    print(f"valid={report['valid']}")
    print("paid_calls_executed=0")
    if report["violations"]:
        print("violations=" + ",".join(report["violations"]))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
