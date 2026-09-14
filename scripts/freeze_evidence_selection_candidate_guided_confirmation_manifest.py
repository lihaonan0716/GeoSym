#!/usr/bin/env python
"""Prepare a larger candidate-guided M1/M2 confirmation manifest.

The manifest is zero-cost and approval-gated. It deliberately cannot be passed
to the paid runner until the maintainer approves the cap and the stage status is
promoted by an explicit future step.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_evidence_selection_candidate_guided_roster import clone_control_job, deterministic_candidate  # noqa: E402
from scripts.freeze_evidence_selection_selective_propagation_pilot_manifest import (  # noqa: E402
    INPUT_TOKEN_LIMIT,
    LOCAL_PRICE_INPUT_PER_MILLION,
    LOCAL_PRICE_OUTPUT_PER_MILLION,
    MAX_OUTPUT_TOKENS,
    SEED,
    estimate_tokens,
    model_registry_value,
    stable_file_hash_from_text,
)
from scripts.prepare_public_benchmark_pilot import stable_file_hash, stable_payload_hash  # noqa: E402
from scripts.selective_propagation import SELECTIVE_CONDITIONS, facts_for_selective_condition, selective_answer_prompt  # noqa: E402

SCHEMA_VERSION = "m1m2_candidate_guided_confirmation_manifest_v1"
DEFAULT_CONFIG_PATH = ROOT / "config" / "default_config.yaml"
DEFAULT_SAMPLE_CAP = 16


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_default_model_provenance(config_path: Path | None = None) -> dict[str, Any]:
    import yaml

    config_path = DEFAULT_CONFIG_PATH if config_path is None else config_path
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for provider, provider_cfg in (cfg.get("models") or {}).items():
        models = provider_cfg.get("models") or {}
        for display_name, registry_id in models.items():
            return {
                "provider": str(provider),
                "model_display_name": str(display_name),
                "model_registry_id_or_config_value": str(registry_id),
                "config_path": str(config_path.resolve()),
                "config_sha256": stable_file_hash(config_path),
                "temperature_or_decoding_note": "config-resolved model default with temperature 0 when supported",
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            }
    raise ValueError(f"no configured models found in {config_path}")


def guarded_by_sample(guarded: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in guarded.get("results") or []}


def build_answerable_jobs(schema_outputs: dict[str, Any], guarded: dict[str, Any], records: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    guarded_rows = guarded_by_sample(guarded)
    jobs = []
    for schema in sorted(schema_outputs.get("outputs") or [], key=lambda row: str(row.get("sample_id"))):
        sample_id = str(schema.get("sample_id"))
        record = records.get(sample_id)
        guarded_row = guarded_rows.get(sample_id)
        if record is None or guarded_row is None:
            continue
        selected = schema.get("selected_evidence_set") or {}
        raw_text = str(guarded_row.get("raw_text") or "")
        if selected.get("propagation_policy") != "solve_deterministically":
            continue
        if not re.fullmatch(r"\s*deterministic_m3_prediction=.+?\s*", raw_text):
            continue
        if not selected.get("selected_fact_ids") or not selected.get("suppressed_fact_ids"):
            continue
        candidate = deterministic_candidate(raw_text, str(record.get("answer_type")))
        stress_schema = copy.deepcopy(schema)
        stress_schema["expected_answerability"] = "answerable"
        stress_schema["redesign_hypothesis"] = "candidate_guided_selective_propagation_v3_1"
        stress_schema["route_precedence"] = ["contradiction_on_required_key", "missing_required_fact", "answer_with_candidate_and_selected_facts"]
        selected_set = stress_schema.setdefault("selected_evidence_set", {})
        selected_set["required_fact_coverage_complete"] = True
        selected_set["blocking_gap_keys"] = []
        selected_set["surfaced_gap_ids"] = []
        selected_set["propagation_policy"] = "answer_with_candidate_and_selected_facts"
        selected_set["answer_candidate"] = {
            "normalized_prediction": candidate,
            "answer_type": record.get("answer_type"),
            "source": "deterministic_m3_from_selected_facts",
            "source_raw_text_hash": stable_payload_hash({"raw_text": raw_text}),
            "supporting_fact_ids": list(selected.get("selected_fact_ids") or []),
            "selection_rule": "schema_policy_solve_deterministically_and_deterministic_m3_raw_text_present",
        }
        stress_schema["stress_hash"] = stable_payload_hash(stress_schema)
        jobs.append(
            {
                "stress_id": f"candidate_guided_confirmation_answerable_{len(jobs):03d}",
                "source": "plotqa_guarded_projection_schema_join",
                "source_sample_id": sample_id,
                "perturbation_family": "spurious_low_confidence_fact",
                "seed": SEED,
                "answer_independent_selection": True,
                "expected_answerability": "answerable",
                "stress_schema": stress_schema,
                "stress_hash": stress_schema["stress_hash"],
            }
        )
        if len(jobs) >= limit:
            break
    return jobs


def build_control_jobs(source_roster: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    source = [
        job
        for job in sorted(source_roster.get("jobs") or [], key=lambda row: (str(row.get("perturbation_family")), str(row.get("stress_id"))))
        if job.get("expected_answerability") != "answerable"
    ]
    selected = []
    families_seen: set[str] = set()
    for job in source:
        family = str(job.get("perturbation_family"))
        if family in families_seen:
            continue
        selected.append(job)
        families_seen.add(family)
        if len(selected) == limit:
            break
    selected_ids = {str(job.get("stress_id")) for job in selected}
    for job in source:
        if len(selected) == limit:
            break
        if str(job.get("stress_id")) in selected_ids:
            continue
        selected.append(job)
    return [clone_control_job(job, index) for index, job in enumerate(selected)]


def build_manifest(
    schema_outputs_path: Path,
    guarded_projection_path: Path,
    pilot_analysis_path: Path,
    source_roster_path: Path,
    measured_records_path: Path,
    output_dir: Path,
    model_display_name: str | None,
    sample_cap: int,
) -> dict[str, Any]:
    model_provenance = load_default_model_provenance()
    if model_display_name is None:
        model_display_name = model_provenance["model_display_name"]
    else:
        registry_id = model_registry_value(model_display_name)
        model_provenance.update(
            {
                "model_display_name": model_display_name,
                "model_registry_id_or_config_value": registry_id,
            }
        )
    schema_outputs = load_json(schema_outputs_path)
    guarded = load_json(guarded_projection_path)
    pilot_analysis = load_json(pilot_analysis_path)
    source_roster = load_json(source_roster_path)
    records = {str(row["sample_id"]): row for row in load_json(measured_records_path)}
    target_answerable = sample_cap // 2
    target_controls = sample_cap - target_answerable
    answerable = build_answerable_jobs(schema_outputs, guarded, records, target_answerable)
    controls = build_control_jobs(source_roster, target_controls)
    jobs = answerable + controls
    findings = []
    if len(answerable) < target_answerable:
        findings.append(f"answerable_jobs_underfilled:{len(answerable)}_of_{target_answerable}")
    if len(controls) < target_controls:
        findings.append(f"control_jobs_underfilled:{len(controls)}_of_{target_controls}")
    prompt_token_estimates: dict[str, list[int]] = {condition: [] for condition in SELECTIVE_CONDITIONS}
    prompt_hashes: dict[str, list[str]] = {condition: [] for condition in SELECTIVE_CONDITIONS}
    selected_roster = []
    for job in jobs:
        record = records.get(str(job.get("source_sample_id")))
        if record is None:
            findings.append(f"missing_measured_record:{job.get('source_sample_id')}")
            continue
        for condition in SELECTIVE_CONDITIONS:
            facts = facts_for_selective_condition(job["stress_schema"], condition, SEED, str(job["stress_id"]))
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
    answerability_counts = Counter(row["expected_answerability"] for row in selected_roster)
    gates = {
        "pilot_supported": pilot_analysis.get("stage_status") == "m1m2_candidate_guided_pilot_supported",
        "schema_outputs_ready": schema_outputs.get("stage_status") == "ready_for_zero_cost_projection",
        "guarded_projection_preserved": guarded.get("stage_status") == "plotqa_guardrail_preserved",
        "plotqa_guardrail_preserved": (guarded.get("guardrail") or {}).get("passed") is True,
        "source_roster_frozen": source_roster.get("stage_status") == "answerability_preserving_roster_frozen",
        "selected_roster_matches_sample_cap": len(selected_roster) == sample_cap,
        "answerable_count_target": answerability_counts.get("answerable") == target_answerable,
        "llm_cap_exceeds_current_authorization": llm_cap > 72,
        "ocr_cap_zero": True,
        "max_prompt_tokens_within_limit": max_input_tokens <= INPUT_TOKEN_LIMIT,
        "cost_estimate_available": estimated_cost > 0,
        "blocking_findings_empty": not findings,
    }
    approval_ready = all(gates.values())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "m1m2_candidate_guided_confirmation_manifest",
        "stage_status": "candidate_guided_confirmation_manifest_frozen_pending_paid_cap_approval" if approval_ready else "candidate_guided_confirmation_manifest_failed_closed",
        "redesign_name": "candidate_guided_selective_propagation_v3_1",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "requires_explicit_paid_cap_approval_before_execution": True,
        "approval_required_for_llm_http_attempt_cap": llm_cap,
        "current_standing_authorization_llm_cap": 72,
        "input_mode": "evidence_only",
        "sample_level_job_cap": len(selected_roster),
        "condition_list": list(SELECTIVE_CONDITIONS),
        "llm_http_attempt_cap": llm_cap,
        "ocr_http_attempt_cap": 0,
        "retry_cap_per_sample_condition": 1,
        "model_display_name": model_display_name,
        "model_registry_id_or_config_value": model_provenance["model_registry_id_or_config_value"],
        "model_provenance": model_provenance,
        "temperature_or_decoding_note": model_provenance["temperature_or_decoding_note"],
        "max_output_tokens": model_provenance["max_output_tokens"],
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
            "schema_outputs": stable_file_hash(schema_outputs_path),
            "guarded_projection": stable_file_hash(guarded_projection_path),
            "pilot_analysis": stable_file_hash(pilot_analysis_path),
            "source_roster": stable_file_hash(source_roster_path),
            "measured_records": stable_file_hash(measured_records_path),
        },
        "selected_roster": selected_roster,
        "expected_answerability_counts": dict(sorted(answerability_counts.items())),
        "perturbation_family_counts": dict(sorted(Counter(row["perturbation_family"] for row in selected_roster).items())),
        "random_seed": SEED,
        "safe_default": "do_not_execute_without_explicit_paid_cap_approval",
        "stop_rule": f"Stop after the frozen {llm_cap} LLM attempts regardless of interim results.",
        "expansion_rule": "No automatic expansion beyond this confirmation.",
        "blocking_findings": findings,
        "gates": gates,
        "next_required_action": "request_explicit_confirmation_paid_cap_approval" if approval_ready else "repair_candidate_guided_confirmation_manifest",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    roster = {
        "schema_version": "m1m2_candidate_guided_confirmation_roster_v1",
        "created_at_utc": manifest["created_at_utc"],
        "stage": "m1m2_candidate_guided_confirmation_roster",
        "stage_status": "candidate_guided_confirmation_roster_frozen",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_manifest_sha256": stable_payload_hash(manifest),
        "jobs": jobs,
    }
    (output_dir / "confirmation_roster.json").write_text(json.dumps(roster, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "frozen_confirmation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "frozen_confirmation_manifest.md").write_text(render_markdown(manifest), encoding="utf-8")
    return manifest


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# M1/M2 Candidate-Guided Confirmation Manifest",
        "",
        f"- Stage status: `{manifest['stage_status']}`",
        "- Paid calls executed: 0",
        f"- Requires explicit paid-cap approval: {manifest['requires_explicit_paid_cap_approval_before_execution']}",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-outputs", default="outputs/m1m2_recovery/schema_outputs/m1m2_schema_outputs.json")
    parser.add_argument("--guarded-projection", default="outputs/m1m2_recovery/guarded_zero_cost_projection/guarded_projection_result.json")
    parser.add_argument("--pilot-analysis", default="outputs/m1m2_candidate_guided_v31/paid_pilot_analysis/analysis.json")
    parser.add_argument("--source-roster", default="outputs/m1m2_answerability_preserving/roster/answerability_preserving_roster.json")
    parser.add_argument("--measured-records", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/m0_measurement_records.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_candidate_guided_v31/confirmation_manifest")
    parser.add_argument("--model-display-name", default=None, help="Override the config-resolved default display name.")
    parser.add_argument(
        "--sample-cap",
        type=int,
        default=DEFAULT_SAMPLE_CAP,
        help="Balanced confirmation size. Default is the largest fully supported zero-cost candidate-guided confirmation: 8 answerable and 8 controls.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_manifest(
        Path(args.schema_outputs),
        Path(args.guarded_projection),
        Path(args.pilot_analysis),
        Path(args.source_roster),
        Path(args.measured_records),
        Path(args.output_dir),
        args.model_display_name,
        args.sample_cap,
    )
    print(f"wrote candidate-guided confirmation manifest: {Path(args.output_dir) / 'frozen_confirmation_manifest.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    print(f"approval_required_for_llm_http_attempt_cap={result['approval_required_for_llm_http_attempt_cap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
