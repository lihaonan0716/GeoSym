#!/usr/bin/env python
"""Freeze the first paid pilot manifest for M1/M2 selective propagation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.helpers import MODEL_REGISTRY  # noqa: E402
from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402
from scripts.selective_propagation import SELECTIVE_CONDITIONS, facts_for_selective_condition, selective_answer_prompt  # noqa: E402

SCHEMA_VERSION = "m1m2_selective_paid_pilot_manifest_v1"
SEED = 20260830
SAMPLE_CAP = 12
MAX_OUTPUT_TOKENS = 256
INPUT_TOKEN_LIMIT = 12000
DEFAULT_MODEL_DISPLAY_NAME = "GPT-5.5"
LOCAL_PRICE_INPUT_PER_MILLION = 2.0
LOCAL_PRICE_OUTPUT_PER_MILLION = 8.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def select_balanced_roster(roster: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in sorted(roster.get("jobs", []), key=lambda row: (str(row.get("perturbation_family")), str(row.get("stress_id")))):
        by_family[str(job.get("perturbation_family"))].append(job)
    selected = []
    findings = []
    for family in sorted(by_family):
        rows = by_family[family]
        if len(rows) < 2:
            findings.append(f"stress_paid_manifest_family_underfilled:{family}")
            continue
        selected.extend(rows[:2])
    if len(selected) != SAMPLE_CAP:
        findings.append("selected_roster_not_12")
    return selected, findings


def model_registry_value(model_display_name: str) -> str:
    value = MODEL_REGISTRY.get(model_display_name)
    if not value:
        return ""
    return str(value[-1])


def build_manifest(stress_roster_path: Path, zero_cost_path: Path, measured_records_path: Path, output_dir: Path, model_display_name: str) -> dict[str, Any]:
    roster = load_json(stress_roster_path)
    zero_cost = load_json(zero_cost_path)
    records = {str(row["sample_id"]): row for row in load_json(measured_records_path)}
    selected, findings = select_balanced_roster(roster)
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
    max_input_tokens = max((max(values) for values in prompt_token_estimates.values() if values), default=0)
    total_input_tokens = sum(sum(values) for values in prompt_token_estimates.values())
    llm_cap = SAMPLE_CAP * len(SELECTIVE_CONDITIONS)
    estimated_cost = (
        total_input_tokens * LOCAL_PRICE_INPUT_PER_MILLION / 1_000_000
        + llm_cap * MAX_OUTPUT_TOKENS * LOCAL_PRICE_OUTPUT_PER_MILLION / 1_000_000
    )
    registry_id = model_registry_value(model_display_name)
    if not registry_id:
        findings.append("model_registry_id_unavailable")
    if estimated_cost <= 0:
        findings.append("paid_cost_estimate_unavailable")
    gates = {
        "stress_roster_frozen": roster.get("stage_status") == "selective_stress_roster_frozen",
        "zero_cost_ready_for_manifest": zero_cost.get("stage_status") == "ready_for_paid_manifest",
        "plotqa_guardrail_preserved": (zero_cost.get("plotqa_guardrail") or {}).get("passed") is True,
        "selected_roster_12": len(selected_roster) == SAMPLE_CAP,
        "two_per_family": all(value == 2 for value in Counter(row["perturbation_family"] for row in selected_roster).values()),
        "llm_cap_matches_frozen_conditions": llm_cap == SAMPLE_CAP * len(SELECTIVE_CONDITIONS),
        "ocr_cap_zero": True,
        "max_prompt_tokens_within_limit": max_input_tokens <= INPUT_TOKEN_LIMIT,
        "cost_estimate_available": estimated_cost > 0,
        "blocking_findings_empty": not findings,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_selective_paid_pilot_manifest",
        "stage_status": "selective_paid_pilot_manifest_frozen" if all(gates.values()) else "selective_paid_pilot_manifest_failed_closed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "input_mode": "evidence_only",
        "sample_level_job_cap": SAMPLE_CAP,
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
            "zero_cost_projection": stable_file_hash(zero_cost_path),
            "measured_records": stable_file_hash(measured_records_path),
        },
        "selected_roster": selected_roster,
        "perturbation_family_counts": dict(sorted(Counter(row["perturbation_family"] for row in selected_roster).items())),
        "expected_answerability_counts": dict(sorted(Counter(row["expected_answerability"] for row in selected_roster).items())),
        "random_seed": SEED,
        "safe_default": "dry_run_no_paid_calls",
        "stop_rule": f"Stop after the frozen {llm_cap} LLM attempts regardless of interim results.",
        "expansion_rule": "No automatic expansion; larger confirmation requires a new reviewed manifest and explicit cap approval.",
        "blocking_findings": findings,
        "gates": gates,
        "next_required_action": "run_selective_paid_pilot_dry_run" if all(gates.values()) else "repair_selective_paid_manifest",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "frozen_manifest.md").write_text(render_markdown(manifest), encoding="utf-8")
    return manifest


def stable_file_hash_from_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Selective Propagation Paid Pilot Manifest",
        "",
        f"- Stage status: `{manifest['stage_status']}`",
        "- Paid calls executed: 0",
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
    parser.add_argument("--zero-cost", default="outputs/m1m2_selective_propagation/zero_cost_projection/selective_zero_cost_projection.json")
    parser.add_argument("--measured-records", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/m0_measurement_records.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_selective_propagation/paid_pilot_manifest")
    parser.add_argument("--model-display-name", default=DEFAULT_MODEL_DISPLAY_NAME)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = build_manifest(Path(args.stress_roster), Path(args.zero_cost), Path(args.measured_records), Path(args.output_dir), args.model_display_name)
    print(f"wrote selective paid pilot manifest: {Path(args.output_dir) / 'frozen_manifest.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
