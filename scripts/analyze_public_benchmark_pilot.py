#!/usr/bin/env python
"""Analyze a public benchmark G0 smoke result without additional paid calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_public_benchmark_pilot import (  # noqa: E402
    EXCLUDED_CHART_TYPES,
    SUPPORTED_CHART_TYPES as CHARTQA_SUPPORTED_CHART_TYPES,
    SUPPORTED_REASONING_TYPES as CHARTQA_SUPPORTED_REASONING_TYPES,
    infer_chart_type_from_question,
)
from scripts.prepare_plotqa_public_benchmark_pilot import (  # noqa: E402
    SUPPORTED_CHART_TYPES as PLOTQA_SUPPORTED_CHART_TYPES,
    SUPPORTED_REASONING_TYPES as PLOTQA_SUPPORTED_REASONING_TYPES,
    validate_plotqa_sample_semantics,
)
from scripts.run_public_benchmark_pilot import validate_trace_record  # noqa: E402


def _rate(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def _stable_unit_interval(seed: str) -> float:
    value = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16)
    return value / float(16**16)


def selection_validity(plan: dict[str, Any]) -> dict[str, Any]:
    violations = []
    frozen = plan.get("frozen_before_paid_run", {})
    sample_selection = frozen.get("sample_selection", {})
    if plan.get("paid_run_approved") is not True:
        violations.append("paid_run_not_approved")
    if plan.get("zero_cost_only") is True and plan.get("paid_run_approved") is True:
        violations.append("paid_approved_plan_marked_zero_cost_only")
    benchmark = plan.get("benchmark")
    expected_frame = "official_plotqa_full_split_hash_pool" if benchmark == "PlotQA" else "full_split_hash_pool"
    supported_chart_types = PLOTQA_SUPPORTED_CHART_TYPES if benchmark == "PlotQA" else CHARTQA_SUPPORTED_CHART_TYPES
    supported_reasoning_types = PLOTQA_SUPPORTED_REASONING_TYPES if benchmark == "PlotQA" else CHARTQA_SUPPORTED_REASONING_TYPES
    if sample_selection.get("sampling_frame") != expected_frame:
        violations.append("not_full_split_hash_pool")
    if benchmark != "PlotQA" and sample_selection.get("full_split_export") is not True:
        violations.append("manifest_not_full_split_export")
    if benchmark == "PlotQA":
        if not plan.get("frozen_before_paid_run", {}).get("official_files"):
            violations.append("missing_official_file_hashes")
        model_provenance = plan.get("frozen_before_paid_run", {}).get("model_provenance")
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
        if model_provenance.get("api_model_id") and plan.get("frozen_before_paid_run", {}).get("reasoning_model") != model_provenance.get("api_model_id"):
            violations.append("reasoning_model_does_not_match_model_provenance")
        excluded_summary = plan.get("excluded_summary", {})
        if isinstance(excluded_summary, dict) and excluded_summary.get("eligible"):
            violations.append("excluded_summary_contains_eligible")
    selected_ids = sample_selection.get("selected_ids") or []
    selected_samples = plan.get("selected_samples") or []
    expected_count = 50 if len(selected_samples) == 50 else 8
    if len(selected_ids) != expected_count or len(selected_samples) != expected_count:
        violations.append(f"selected_roster_not_{expected_count}")
    if [sample.get("sample_id") for sample in selected_samples] != selected_ids:
        violations.append("selected_ids_do_not_match_samples")

    per_sample = []
    quantitative_count = 0
    image_indices = []
    for sample in selected_samples:
        metadata = sample.get("metadata") or {}
        sample_violations = []
        sample_id = sample.get("sample_id")
        chart_type = metadata.get("chart_type")
        question_chart_type = infer_chart_type_from_question(sample.get("question", ""))
        if question_chart_type in EXCLUDED_CHART_TYPES:
            sample_violations.append(f"question_cues_excluded_chart_type:{question_chart_type}")
        if chart_type not in supported_chart_types:
            sample_violations.append(f"unsupported_chart_type:{chart_type}")
        if metadata.get("axis_visible") is not True:
            sample_violations.append("axis_visible_not_true")
        if metadata.get("image_readable") is not True:
            sample_violations.append("image_readable_not_true")
        if metadata.get("geosym_compatibility") != "presenter_candidate":
            sample_violations.append(f"geosym_compatibility:{metadata.get('geosym_compatibility')}")
        if metadata.get("reasoning_type") not in supported_reasoning_types:
            sample_violations.append(f"unsupported_reasoning_type:{metadata.get('reasoning_type')}")
        if benchmark == "PlotQA":
            for key in ("has_x_axis", "has_y_axis", "has_tick_labels", "has_data_table"):
                if metadata.get(key) is not True:
                    sample_violations.append(f"{key}_not_true")
            if metadata.get("legend_required") is True and metadata.get("legend_available") is not True:
                sample_violations.append("legend_required_missing")
            sample_violations.extend(validate_plotqa_sample_semantics(sample))
            image_indices.append(metadata.get("image_index"))
        if metadata.get("quantitative_reasoning_required") is True:
            quantitative_count += 1
        if sample_violations:
            violations.extend(f"{sample_id}:{item}" for item in sample_violations)
        per_sample.append({"sample_id": sample_id, "violations": sample_violations})
    if quantitative_count < 4:
        violations.append("fewer_than_4_quantitative_reasoning_samples")
    if benchmark == "PlotQA" and len(set(image_indices)) != len(image_indices):
        violations.append("plotqa_selected_images_not_unique")
    return {
        "valid": not violations,
        "violations": violations,
        "per_sample": per_sample,
        "quantitative_reasoning_required": quantitative_count,
    }


def bootstrap_accuracy_delta(
    sample_ids: list[str],
    g0_correct: dict[str, bool],
    baseline_correct: dict[str, bool],
    iterations: int = 1000,
) -> dict[str, Any]:
    if not sample_ids:
        return {"delta": 0.0, "ci95": [0.0, 0.0], "iterations": 0}
    deltas = []
    n = len(sample_ids)
    for iteration in range(iterations):
        total = 0
        for slot in range(n):
            idx = int(_stable_unit_interval(f"{iteration}:{slot}:{sample_ids[slot]}") * n)
            sid = sample_ids[min(idx, n - 1)]
            total += int(g0_correct.get(sid) is True) - int(baseline_correct.get(sid) is True)
        deltas.append(total / n)
    deltas.sort()
    low = deltas[int(0.025 * (iterations - 1))]
    high = deltas[int(0.975 * (iterations - 1))]
    observed = sum(
        int(g0_correct.get(sid) is True) - int(baseline_correct.get(sid) is True)
        for sid in sample_ids
    ) / n
    return {"delta": observed, "ci95": [low, high], "iterations": iterations}


def build_report(plan: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    results = result.get("results", [])
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_condition[row["condition"]].append(row)

    condition_summary = {}
    for condition, rows in sorted(by_condition.items()):
        condition_summary[condition] = {
            "jobs": len(rows),
            "completed": sum(1 for row in rows if row.get("completed")),
            "parseable": sum(1 for row in rows if row.get("parseable")),
            "correct": sum(1 for row in rows if row.get("correct") is True),
            "accuracy": _rate(sum(1 for row in rows if row.get("correct") is True), len(rows)),
        }

    benchmark = plan.get("benchmark")
    g0_condition = "G0_plotqa_evidence_trace" if benchmark == "PlotQA" else "G0_public_evidence_trace"
    direct_conditions = ("E0_plotqa_direct", "E1_plotqa_cot") if benchmark == "PlotQA" else ("E0_public_direct", "E1_public_cot")
    g0_rows = by_condition.get(g0_condition, [])
    direct_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for condition in direct_conditions:
        for row in by_condition.get(condition, []):
            direct_by_sample[row["sample_id"]].append(row)

    rescues = 0
    degradations = 0
    for row in g0_rows:
        direct_correct = any(d.get("correct") is True for d in direct_by_sample.get(row["sample_id"], []))
        direct_all_wrong = direct_by_sample.get(row["sample_id"]) and not direct_correct
        if row.get("correct") is True and direct_all_wrong:
            rescues += 1
        if row.get("correct") is not True and direct_correct:
            degradations += 1

    attribution_counts = Counter(row.get("attribution_label", "missing") for row in g0_rows)
    trace_schema_by_sample = [
        {"sample_id": row.get("sample_id"), "violations": validate_trace_record(row)}
        for row in g0_rows
    ]
    strict_parseable_trace_records = sum(1 for row in trace_schema_by_sample if not row["violations"])
    useful_labels = {
        "correct",
        "extraction_failure",
        "curation_failure",
        "reasoning_failure",
        "proxy_extraction_failure",
        "proxy_curation_failure",
        "proxy_reasoning_failure",
        "normalization_failure",
    }
    useful_attribution = sum(count for label, count in attribution_counts.items() if label in useful_labels)
    wrong_g0_rows = [row for row in g0_rows if row.get("correct") is not True]
    wrong_g0_unknown = sum(1 for row in wrong_g0_rows if row.get("attribution_label") == "attribution_unknown")
    selected_samples = plan.get("selected_samples", [])
    sample_count = len(selected_samples)
    selected_sample_ids = [sample.get("sample_id") for sample in selected_samples]
    quantitative_samples = sum(
        1 for sample in selected_samples if (sample.get("metadata") or {}).get("quantitative_reasoning_required") is True
    )
    selection_gate = selection_validity(plan)
    if benchmark == "PlotQA" and sample_count >= 50:
        completion_threshold = 45
        trace_threshold = 45
        direct_threshold = 45
        degradation_cap = 10
    else:
        completion_threshold = 7
        trace_threshold = 7
        direct_threshold = 7
        degradation_cap = 1
    operational_go_gate = {
        "within_attempt_cap": result.get("actual_llm_http_attempts", 10**9) <= result.get("llm_http_attempt_cap", 0)
        and result.get("actual_ocr_http_attempts") == 0,
        f"g0_completed_at_least_{completion_threshold}": sum(1 for row in g0_rows if row.get("completed")) >= completion_threshold,
        f"g0_parseable_trace_at_least_{trace_threshold}": strict_parseable_trace_records >= trace_threshold,
        f"direct_completed_at_least_{direct_threshold}_each": all(
            condition_summary.get(condition, {}).get("completed", 0) >= direct_threshold
            for condition in direct_conditions
        ),
        f"g0_degradations_no_more_than_{degradation_cap}": degradations <= degradation_cap,
    }
    if benchmark == "PlotQA" and sample_count >= 50:
        operational_go_gate["wrong_g0_attribution_unknown_no_more_than_half"] = (
            True if not wrong_g0_rows else wrong_g0_unknown <= len(wrong_g0_rows) / 2
        )
    else:
        operational_go_gate["useful_attribution_at_least_6"] = useful_attribution >= 6
    go_gate = {"selection_validity_gate": selection_gate["valid"], **operational_go_gate}
    official_files = plan.get("frozen_before_paid_run", {}).get("official_files", {})
    correctness_by_condition = {
        condition: {row.get("sample_id"): row.get("correct") is True for row in rows}
        for condition, rows in by_condition.items()
    }
    paired_accuracy_delta = {}
    if g0_condition in correctness_by_condition:
        for condition in direct_conditions:
            if condition in correctness_by_condition:
                paired_accuracy_delta[f"{g0_condition}_minus_{condition}"] = bootstrap_accuracy_delta(
                    selected_sample_ids,
                    correctness_by_condition[g0_condition],
                    correctness_by_condition[condition],
                )
    strata = {
        "chart_type": dict(Counter((sample.get("metadata") or {}).get("chart_type", "unknown") for sample in selected_samples)),
        "reasoning_type": dict(Counter((sample.get("metadata") or {}).get("reasoning_type", "unknown") for sample in selected_samples)),
        "answer_derivation": dict(Counter((sample.get("metadata") or {}).get("answer_derivation", "unknown") for sample in selected_samples)),
    }
    if benchmark == "PlotQA":
        if not selection_gate["valid"]:
            experiment_status = "invalidated_selector_diagnostic"
        elif sample_count >= 50:
            g0_correct = condition_summary.get(g0_condition, {}).get("correct", 0)
            strongest_direct = max(
                condition_summary.get(condition, {}).get("correct", 0)
                for condition in direct_conditions
            )
            if g0_correct > strongest_direct:
                experiment_status = "plotqa_diagnostic_complete_limited_attribution"
            else:
                experiment_status = "plotqa_diagnostic_no_superiority"
        elif all(go_gate.values()):
            experiment_status = "plotqa_smoke_go"
        else:
            experiment_status = "plotqa_access_or_selector_diagnostic"
    else:
        experiment_status = "go" if all(go_gate.values()) else "invalidated_selector_diagnostic"

    return {
        "schema_version": "plotqa_g0_smoke_analysis_v1" if benchmark == "PlotQA" else "chartqa_g0_smoke_analysis_v1",
        "benchmark": plan.get("benchmark"),
        "split": plan.get("split"),
        "manifest_sha256": plan.get("manifest_sha256"),
        "official_files": official_files,
        "candidate_pool_sha256": plan.get("candidate_pool_sha256"),
        "plan_sha256": plan.get("plan_sha256"),
        "selected_ids": plan.get("frozen_before_paid_run", {}).get("sample_selection", {}).get("selected_ids", []),
        "conditions": result.get("conditions", []),
        "model": result.get("reasoning_model"),
        "model_provenance": plan.get("frozen_before_paid_run", {}).get("model_provenance", {}),
        "prompt_hashes": result.get("prompt_hashes", {}),
        "attempts": {
            "llm_cap": result.get("llm_http_attempt_cap"),
            "llm_actual": result.get("actual_llm_http_attempts"),
            "ocr_cap": result.get("ocr_http_attempt_cap"),
            "ocr_actual": result.get("actual_ocr_http_attempts"),
        },
        "sample_profile": {
            "samples": sample_count,
            "quantitative_reasoning_required": quantitative_samples,
            "strata": strata,
        },
        "selection_validity": selection_gate,
        "condition_summary": condition_summary,
        "g0_trace": {
            "completed": sum(1 for row in g0_rows if row.get("completed")),
            "parseable_trace_records": strict_parseable_trace_records,
            "trace_schema_by_sample": trace_schema_by_sample,
            "trace_schema_violation_count": sum(len(row["violations"]) for row in trace_schema_by_sample),
            "attribution_counts": dict(attribution_counts),
            "useful_attribution_records": useful_attribution,
        },
        "delta_proxy": {
            "rescues": rescues,
            "degradations": degradations,
            "wrong_g0_cases": len(wrong_g0_rows),
            "wrong_g0_attribution_unknown": wrong_g0_unknown,
        },
        "paired_accuracy_delta": paired_accuracy_delta,
        "go_gate": go_gate,
        "experiment_status": experiment_status,
        "interpretation_limits": result.get("interpretation_limits", []),
    }


def render_markdown(report: dict[str, Any]) -> str:
    sample_count = report.get("sample_profile", {}).get("samples", 0)
    run_label = "Diagnostic" if sample_count >= 50 else "Smoke"
    lines = [
        f"# {report['benchmark']} G0 {run_label} Analysis",
        "",
        f"- Benchmark: {report['benchmark']} / {report['split']}",
        f"- Model: {report['model']}",
        f"- Plan SHA256: `{report['plan_sha256']}`",
        f"- Selected IDs: {', '.join(report['selected_ids'])}",
        f"- Experiment status: {report['experiment_status']}",
        "",
    ]
    model_provenance = report.get("model_provenance") or {}
    if model_provenance:
        lines.extend(
            [
                "## Model Provenance",
                "",
                f"- Display name: `{model_provenance.get('display_name')}`",
                f"- Provider: `{model_provenance.get('provider')}`",
                f"- Adapter: `{model_provenance.get('adapter')}`",
                f"- API model ID: `{model_provenance.get('api_model_id')}`",
                f"- Gateway policy: `{model_provenance.get('gateway_policy')}`",
                f"- Base URL key: `{model_provenance.get('base_url_key', '')}`",
                f"- Config SHA256: `{model_provenance.get('config_sha256')}`",
                "",
            ]
        )
    official_files = report.get("official_files") or {}
    if official_files:
        lines.extend(
            [
                "## Official Files",
                "",
                f"- annotations.json SHA256: `{official_files.get('annotations_sha256')}`",
                f"- qa_pairs.json SHA256: `{official_files.get('qa_pairs_sha256')}`",
                f"- image manifest SHA256: `{official_files.get('image_manifest_sha256')}`",
                "",
            ]
        )
    elif report.get("manifest_sha256"):
        lines.extend(
            [
                "## Manifest",
                "",
                f"- Manifest SHA256: `{report['manifest_sha256']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Selection Validity",
            "",
            f"- Valid: {report['selection_validity']['valid']}",
            f"- Violations: {json.dumps(report['selection_validity']['violations'], sort_keys=True)}",
            "",
            "## Attempts",
            "",
            f"- LLM: {report['attempts']['llm_actual']} / {report['attempts']['llm_cap']}",
            f"- OCR: {report['attempts']['ocr_actual']} / {report['attempts']['ocr_cap']}",
            "",
            "## Condition Summary",
            "",
        ]
    )
    for condition, row in report["condition_summary"].items():
        lines.append(
            f"- {condition}: completed {row['completed']}/{row['jobs']}, parseable {row['parseable']}/{row['jobs']}, "
            f"correct {row['correct']}/{row['jobs']}."
        )
    lines.extend(["", "## Sample Strata", ""])
    for name, counts in report["sample_profile"].get("strata", {}).items():
        lines.append(f"- {name}: {json.dumps(counts, sort_keys=True)}")
    lines.extend(["", "## Paired Accuracy Delta", ""])
    for name, row in report.get("paired_accuracy_delta", {}).items():
        lines.append(
            f"- {name}: delta {row['delta']:.3f}, 95% bootstrap CI [{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]"
        )
    lines.extend(
        [
            "",
            "## Trace And Attribution",
            "",
            f"- G0 completed: {report['g0_trace']['completed']}/{report['sample_profile']['samples']}",
            f"- G0 parseable trace records: {report['g0_trace']['parseable_trace_records']}/{report['sample_profile']['samples']}",
            f"- Trace schema violations: {report['g0_trace']['trace_schema_violation_count']}",
            f"- Attribution labels: {json.dumps(report['g0_trace']['attribution_counts'], sort_keys=True)}",
            f"- Rescues: {report['delta_proxy']['rescues']}",
            f"- Degradations: {report['delta_proxy']['degradations']}",
            "",
            "## Interpretation Limits",
            "",
        ]
    )
    for item in report["interpretation_limits"]:
        lines.append(f"- {item}")
    lines.append("")
    status = report["experiment_status"]
    if status in {"go", "plotqa_smoke_go"}:
        lines.append("This smoke supports operational compatibility, not a broad public-benchmark performance claim.")
    elif status == "plotqa_diagnostic_no_superiority":
        lines.append(
            "This diagnostic package has valid access and selection, but it does not support a public-benchmark "
            "superiority claim."
        )
    elif status == "plotqa_diagnostic_complete_limited_attribution":
        lines.append(
            "This diagnostic package has valid access and selection, with remaining attribution limits that must be "
            "reported explicitly."
        )
    else:
        lines.append("This smoke is diagnostic evidence; do not treat it as a clean Phase 1 go result.")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    report = build_report(plan, result)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote analysis JSON: {output_json}")
    print(f"wrote analysis Markdown: {output_md}")
    print(f"g0_completed={report['g0_trace']['completed']}")
    print(f"useful_attribution_records={report['g0_trace']['useful_attribution_records']}")
    print(f"degradations={report['delta_proxy']['degradations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
