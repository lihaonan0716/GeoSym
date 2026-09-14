#!/usr/bin/env python
"""Freeze the larger confirmation manifest for M1/M2 selective propagation.

This script prepares the 48-job confirmation boundary only. It never executes
paid calls; execution requires explicit paid-cap approval.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_m1m2_selective_paid_pilot_manifest import (  # noqa: E402
    INPUT_TOKEN_LIMIT,
    LOCAL_PRICE_INPUT_PER_MILLION,
    LOCAL_PRICE_OUTPUT_PER_MILLION,
    MAX_OUTPUT_TOKENS,
    SEED,
    estimate_tokens,
    model_registry_value,
    stable_file_hash_from_text,
)
from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402
from scripts.selective_propagation import SELECTIVE_CONDITIONS, facts_for_selective_condition, selective_answer_prompt  # noqa: E402

SCHEMA_VERSION = "m1m2_selective_confirmation_manifest_v1"
DEFAULT_MODEL_DISPLAY_NAME = "GPT-5.5"
CONFIRMATION_SAMPLE_CAP = 48


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(
    stress_roster_path: Path,
    pilot_analysis_path: Path,
    zero_cost_path: Path,
    measured_records_path: Path,
    output_dir: Path,
    model_display_name: str,
) -> dict[str, Any]:
    roster = load_json(stress_roster_path)
    pilot_analysis = load_json(pilot_analysis_path)
    zero_cost = load_json(zero_cost_path)
    records = {str(row["sample_id"]): row for row in load_json(measured_records_path)}
    findings = []
    selected = sorted(roster.get("jobs", []), key=lambda row: (str(row.get("perturbation_family")), str(row.get("stress_id"))))
    prompt_token_estimates: dict[str, list[int]] = {condition: [] for condition in SELECTIVE_CONDITIONS}
    prompt_hashes: dict[str, list[str]] = {condition: [] for condition in SELECTIVE_CONDITIONS}
    selected_roster = []
    for job in selected:
        record = records.get(str(job["source_sample_id"]))
        if record is None:
            findings.append(f"missing_measured_record:{job['source_sample_id']}")
            continue
        schema = job["stress_schema"]
        for condition in SELECTIVE_CONDITIONS:
            facts = facts_for_selective_condition(schema, condition, SEED, str(job["stress_id"]))
            prompt = selective_answer_prompt(record["question"], facts, condition, str(job["perturbation_family"]))
            tokens = estimate_tokens(prompt)
            prompt_token_estimates[condition].append(tokens)
            if tokens > INPUT_TOKEN_LIMIT:
                findings.append(f"prompt_token_limit_exceeded:{job['stress_id']}:{condition}")
            prompt_hashes[condition].append(stable_file_hash_from_text(prompt))
        selected_roster.append(
            {
                "stress_id": job["stress_id"],
                "source_sample_id": job["source_sample_id"],
                "perturbation_family": job["perturbation_family"],
                "expected_answerability": job["expected_answerability"],
                "stress_hash": (job.get("stress_schema") or {}).get("stress_hash"),
            }
        )
    llm_cap = len(selected_roster) * len(SELECTIVE_CONDITIONS)
    total_input_tokens = sum(sum(values) for values in prompt_token_estimates.values())
    estimated_cost = (
        total_input_tokens * LOCAL_PRICE_INPUT_PER_MILLION / 1_000_000
        + llm_cap * MAX_OUTPUT_TOKENS * LOCAL_PRICE_OUTPUT_PER_MILLION / 1_000_000
    )
    max_input_tokens = max((max(values) for values in prompt_token_estimates.values() if values), default=0)
    registry_id = model_registry_value(model_display_name)
    if not registry_id:
        findings.append("model_registry_id_unavailable")
    if estimated_cost <= 0:
        findings.append("paid_cost_estimate_unavailable")
    gates = {
        "pilot_supported": pilot_analysis.get("stage_status") == "m1m2_selective_propagation_supported",
        "stress_roster_frozen": roster.get("stage_status") == "selective_stress_roster_frozen",
        "zero_cost_ready": zero_cost.get("stage_status") == "ready_for_paid_manifest",
        "plotqa_guardrail_preserved": (zero_cost.get("plotqa_guardrail") or {}).get("passed") is True,
        "selected_roster_48": len(selected_roster) == CONFIRMATION_SAMPLE_CAP,
        "family_counts_8_each": all(value == 8 for value in Counter(row["perturbation_family"] for row in selected_roster).values()),
        "llm_cap_matches_frozen_conditions": llm_cap == CONFIRMATION_SAMPLE_CAP * len(SELECTIVE_CONDITIONS),
        "ocr_cap_zero": True,
        "max_prompt_tokens_within_limit": max_input_tokens <= INPUT_TOKEN_LIMIT,
        "cost_estimate_available": estimated_cost > 0,
        "blocking_findings_empty": not findings,
    }
    ready = all(gates.values())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_selective_confirmation_manifest",
        "stage_status": "selective_confirmation_manifest_frozen_pending_approval" if ready else "selective_confirmation_manifest_failed_closed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "requires_explicit_paid_cap_approval_before_execution": True,
        "approval_required_for_llm_http_attempt_cap": llm_cap,
        "input_mode": "evidence_only",
        "sample_level_job_cap": len(selected_roster),
        "condition_list": list(SELECTIVE_CONDITIONS),
        "llm_http_attempt_cap": llm_cap,
        "ocr_http_attempt_cap": 0,
        "retry_cap_per_sample_condition": 1,
        "model_display_name": model_display_name,
        "model_registry_id_or_config_value": registry_id,
        "temperature_or_decoding_note": "config-resolved model default with temperature 0 when supported",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "prompt_token_estimates_by_condition": {
            condition: {"max": max(values) if values else 0, "sum": sum(values), "count": len(values)}
            for condition, values in prompt_token_estimates.items()
        },
        "estimated_cost_usd_upper_bound": round(estimated_cost, 4),
        "pricing_source_or_registry_snapshot": {
            "source": "local_protocol_default_estimate",
            "input_usd_per_million_tokens": LOCAL_PRICE_INPUT_PER_MILLION,
            "output_usd_per_million_tokens": LOCAL_PRICE_OUTPUT_PER_MILLION,
        },
        "prompt_template_hashes_by_condition": {
            condition: {"count": len(values), "unique_count": len(set(values)), "first_hash": values[0] if values else None}
            for condition, values in prompt_hashes.items()
        },
        "source_artifact_hashes": {
            "stress_roster": stable_file_hash(stress_roster_path),
            "pilot_analysis": stable_file_hash(pilot_analysis_path),
            "zero_cost_projection": stable_file_hash(zero_cost_path),
            "measured_records": stable_file_hash(measured_records_path),
        },
        "selected_roster": selected_roster,
        "perturbation_family_counts": dict(sorted(Counter(row["perturbation_family"] for row in selected_roster).items())),
        "expected_answerability_counts": dict(sorted(Counter(row["expected_answerability"] for row in selected_roster).items())),
        "random_seed": SEED,
        "safe_default": "do_not_execute_without_explicit_paid_cap_approval",
        "stop_rule": f"Stop after the frozen {llm_cap} LLM attempts regardless of interim results.",
        "expansion_rule": "No automatic expansion beyond this confirmation.",
        "blocking_findings": findings,
        "gates": gates,
        "next_required_action": "request_explicit_confirmation_paid_cap_approval" if ready else "repair_confirmation_manifest",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frozen_confirmation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "frozen_confirmation_manifest.md").write_text(render_markdown(manifest), encoding="utf-8")
    return manifest


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Selective Propagation Confirmation Manifest",
        "",
        f"- Stage status: `{manifest['stage_status']}`",
        "- Paid calls executed: 0",
        f"- Requires explicit paid-cap approval: {manifest['requires_explicit_paid_cap_approval_before_execution']}",
        f"- Input mode: `{manifest['input_mode']}`",
        f"- Sample-level job cap: {manifest['sample_level_job_cap']}",
        f"- LLM HTTP attempt cap: {manifest['llm_http_attempt_cap']}",
        f"- OCR HTTP attempt cap: {manifest['ocr_http_attempt_cap']}",
        f"- Estimated cost upper bound USD: {manifest['estimated_cost_usd_upper_bound']}",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in manifest["gates"].items())
    lines.extend(["", "## Blocking Findings", ""])
    lines.extend(f"- `{item}`" for item in manifest["blocking_findings"] or ["none"])
    lines.extend(["", "## Next Required Action", "", f"- `{manifest['next_required_action']}`", ""])
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-roster", default="outputs/m1m2_selective_propagation/stress_roster/selective_stress_roster.json")
    parser.add_argument("--pilot-analysis", default="outputs/m1m2_selective_propagation/paid_pilot_analysis/analysis.json")
    parser.add_argument("--zero-cost", default="outputs/m1m2_selective_propagation/zero_cost_projection/selective_zero_cost_projection.json")
    parser.add_argument("--measured-records", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/m0_measurement_records.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_selective_propagation/confirmation_manifest")
    parser.add_argument("--model-display-name", default=DEFAULT_MODEL_DISPLAY_NAME)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = build_manifest(
        Path(args.stress_roster),
        Path(args.pilot_analysis),
        Path(args.zero_cost),
        Path(args.measured_records),
        Path(args.output_dir),
        args.model_display_name,
    )
    print(f"wrote selective confirmation manifest: {Path(args.output_dir) / 'frozen_confirmation_manifest.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
