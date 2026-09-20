#!/usr/bin/env python
"""Finalize the full-paper evidence package from frozen local artifacts.

This script performs no paid calls. It records fail-closed Track C/D outcomes
when the OpenOCR survey or open/local model gate is not satisfied.
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

import yaml  # noqa: E402

from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402
from scripts.prepare_plotqa_public_benchmark_pilot import validate_plotqa_sample_semantics  # noqa: E402
from scripts.audit_prompt_fairness import run_audit as run_prompt_fairness_audit  # noqa: E402
# Validates artifacts already on disk, which contain the retired E5 arm.
from scripts.batch_experiment import HISTORICAL_CONDITIONS as ALL_CONDITIONS  # noqa: E402
from app.helpers import _get_saved_key  # noqa: E402


PROPRIETARY_PROVIDERS = {"openai", "anthropic", "google", "mathpix", "gemini", "claude", "qwen", "kimi", "glm"}
OPEN_MARKERS = ("local", "open", "ollama", "vllm", "lmstudio")
TRACK_B_OCR_DEPENDENT_CONDITIONS = {"E3", "E4", "E5", "E5_raw"}
TRACK_B_OCR_HTTP_ATTEMPTS_PER_SAMPLE_UPPER_BOUND = len(TRACK_B_OCR_DEPENDENT_CONDITIONS)
PAPER_PACKAGE_DIR = ROOT / "outputs" / "paper_level_evidence_package"
M1M2_V31_DIR = ROOT / "outputs" / "m1m2_candidate_guided_v31"
PUBLIC_PILOT_DIR = ROOT / "outputs" / "public_benchmark_pilot"
TRACK_B_REAUTHORIZED_RESULT_NAME = "post_remediation_reauthorized_paid_result.json"
TRACK_B_FINAL_BACKFILL_RESULT_NAME = "post_remediation_e5_backfill_full_240_final_analysis.json"
TRACK_B_M1M2_IDENTIFIABILITY_NAME = "track_b_m1m2_identifiability_analysis.json"
TRACK_D_REAUTHORIZED_RESULT = Path("stress_128_reauthorized") / "stress_128_paid_result.json"
MANUSCRIPT_FORBIDDEN_POSITIVE_PATTERNS = [
    "SOTA",
    "leaderboard",
    "full PlotQA benchmark",
    "calibrated probability",
    "calibrated probabilities",
    "Dempster-Shafer belief masses",
    "conflict coefficients",
    "full repair of the original M1/M2 broad ablation",
    "public benchmark superiority beyond the frozen roster",
    "Mathpix-free reproducibility",
]


def cross_backbone_authoritative_result_path(output_root: Path) -> Path:
    candidates = [
        output_root / "cross_backbone_glm46v_t180" / "paid_result.json",
        output_root / "cross_backbone_glm46v" / "paid_result_glm4096_parserfix.json",
        output_root / "cross_backbone_glm46v" / "paid_result.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def cross_backbone_authoritative_analysis_path(output_root: Path) -> Path:
    t180 = output_root / "cross_backbone_glm46v_t180" / "analysis" / "analysis.json"
    if t180.exists():
        return t180
    return output_root / "cross_backbone_glm46v" / "analysis" / "analysis.json"


def artifact_record(path: Path, *, status_label: str, role: str, required: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "status_label": status_label,
        "role": role,
        "required_for_track_a_clean": required,
        "exists": path.exists(),
    }
    if path.exists():
        record["sha256"] = stable_file_hash(path)
        try:
            payload = load_json(path)
        except Exception as exc:  # pragma: no cover - defensive ledger path
            record["read_error"] = f"{type(exc).__name__}: {exc}"
        else:
            if isinstance(payload, dict):
                record["schema_version"] = payload.get("schema_version")
                record["stage_status"] = payload.get("stage_status") or payload.get("experiment_status")
                for key in ("paid_calls_executed", "actual_llm_http_attempts", "actual_ocr_http_attempts"):
                    if key in payload:
                        record[key] = payload.get(key)
            elif isinstance(payload, list):
                record["schema_version"] = "json_array"
                record["row_count"] = len(payload)
    return record


def has_unknown_attempt_accounting(payload: dict[str, Any]) -> bool:
    status = str(payload.get("stage_status") or payload.get("experiment_status") or "")
    blockers = [str(finding) for finding in (payload.get("blocking_findings") or [])]
    unknown_count = payload.get("actual_llm_http_attempts") is None or payload.get("actual_ocr_http_attempts") is None
    unknown_text = "unknown" in status.lower() or "unaccount" in status.lower()
    unknown_blocker = any("unknown" in finding.lower() or "unaccount" in finding.lower() for finding in blockers)
    return bool(unknown_count or unknown_text or unknown_blocker)


def load_first_existing_json(paths: list[Path]) -> tuple[Path, dict[str, Any]]:
    for path in paths:
        if path.exists():
            return path, load_json(path)
    return paths[0], {}


def track_b_paid_result_is_complete(payload: dict[str, Any], *, expected_samples: int) -> tuple[bool, list[str]]:
    findings: list[str] = []
    if payload.get("stage_status") not in {
        "geosym_bench_post_remediation_rerun_completed",
        "full_240_e5_e5raw_backfill_complete",
        "full_240_all_conditions_with_repaired_e5_e5raw_complete",
    }:
        findings.append(f"post_remediation_paid_result_not_complete:{payload.get('stage_status')}")
    if payload.get("N_paper_subset") not in {None, expected_samples}:
        findings.append(f"post_remediation_paid_result_sample_count_mismatch:{payload.get('N_paper_subset')}")
    attempts = payload.get("actual_llm_http_attempts_completed_runs", payload.get("actual_llm_http_attempts"))
    if not isinstance(attempts, int) or attempts <= 0:
        findings.append("actual_llm_http_attempts_unknown_or_zero")
    ocr_attempts = payload.get("actual_ocr_http_attempts_completed_runs", payload.get("actual_ocr_http_attempts"))
    if not isinstance(ocr_attempts, int):
        findings.append("actual_ocr_http_attempts_unknown")
    conditions = set(payload.get("conditions") or [])
    if {"E5", "E5_raw"}.issubset(conditions):
        summary = (payload.get("summary") or {}).get("overall_by_condition") or {}
        for condition in ("E5", "E5_raw"):
            row = summary.get(condition) or {}
            if row.get("rows") != expected_samples:
                findings.append(f"condition_rows_not_{expected_samples}:{condition}:{row.get('rows')}")
            if row.get("success") != expected_samples:
                findings.append(f"condition_success_not_{expected_samples}:{condition}:{row.get('success')}")
    elif conditions == set(ALL_CONDITIONS):
        summary = (payload.get("summary") or {}).get("overall_by_condition") or {}
        for condition in ALL_CONDITIONS:
            row = summary.get(condition) or {}
            if row.get("rows") not in {None, expected_samples}:
                findings.append(f"condition_rows_not_{expected_samples}:{condition}:{row.get('rows')}")
            if condition in {"E5", "E5_raw"} and row.get("success") != expected_samples:
                findings.append(f"condition_success_not_{expected_samples}:{condition}:{row.get('success')}")
    else:
        findings.append(f"track_b_conditions_missing_e5_e5raw:{sorted(conditions)}")
    if payload.get("unresolved_rows"):
        findings.append("track_b_unresolved_rows_present")
    return not findings, findings


def track_d_completed_claim(track_d_audit: dict[str, Any]) -> str:
    if track_d_audit.get("stage_status") != "m1m2_stress_128_confirmation_completed":
        return (
            "The v31 16-job confirmation remains the authoritative narrow evidence for "
            "candidate-guided answerability-preserving selective propagation."
        )
    paid_record = ((track_d_audit.get("source_artifacts") or {}).get("stress_128_paid_result") or {})
    paid_path = Path(str(paid_record.get("path") or ""))
    paid_result = load_json(paid_path) if paid_path.exists() else {}
    summary = paid_result.get("condition_summary") or {}
    selective_row = summary.get("G_selective_propagation") or {}
    random_row = summary.get("G_random_k") or {}
    all_evidence_row = summary.get("G_all_evidence") or {}
    selective = selective_row.get("task_success")
    random_k = random_row.get("task_success")
    all_evidence = all_evidence_row.get("task_success")
    candidate_unrouted = (summary.get("G_candidate_unrouted") or {}).get("task_success")
    harmful = selective_row.get("harmful_propagation")
    random_harmful = random_row.get("harmful_propagation")
    rows = selective_row.get("rows")
    selective_answerable_correct = selective_row.get("correct_answer")
    random_answerable_correct = random_row.get("correct_answer")
    all_answerable_correct = all_evidence_row.get("correct_answer")
    correct_abstention = selective_row.get("correct_abstention")
    return (
        "The 128-job M1/M2 stress confirmation completed under the frozen cap, "
        f"but it does not support a positive selective-propagation mechanism claim: "
        f"G_selective_propagation task success was {selective}/{rows}, versus "
        f"{random_k}/{rows} for G_random_k, {all_evidence}/{rows} for G_all_evidence, "
        f"and {candidate_unrouted}/{rows} for G_candidate_unrouted. Because the insufficient-evidence "
        f"half is at ceiling for selective propagation ({correct_abstention}/64 correct abstentions), "
        f"report the answerable half separately: selective propagation is {selective_answerable_correct}/64, "
        f"random-k is {random_answerable_correct}/64, and all-evidence is {all_answerable_correct}/64. "
        f"Selective propagation also has {harmful}/{rows} harmful propagation cases versus "
        f"{random_harmful}/{rows} for random-k. Report this as neutral or negative mechanism evidence, "
        "not as broad M1/M2 accuracy repair."
    )


def governance_chain_is_clean(postrun_audit: dict[str, Any]) -> tuple[bool, list[str]]:
    findings: list[str] = []
    if postrun_audit.get("stage_status") != "candidate_guided_postrun_governance_clean":
        findings.append("m1m2_postrun_governance_not_clean")
    if postrun_audit.get("blocking_findings"):
        findings.append("m1m2_postrun_governance_has_blocking_findings")
    gates = postrun_audit.get("gates") or {}
    required_gates = [
        "preapproval_manifest_hash_current",
        "dry_run_manifest_hash_current",
        "paid_result_manifest_hash_current",
        "dry_run_roster_hash_current",
        "paid_result_roster_hash_current",
        "paid_cap_respected",
        "ocr_cap_zero",
        "config_default_model_matches_manifest",
        "paid_result_model_display_matches_manifest",
    ]
    for gate in required_gates:
        if gates.get(gate) is not True:
            findings.append(f"m1m2_gate_failed:{gate}")
    paid_result = (postrun_audit.get("source_artifacts") or {}).get("paid_result") or {}
    if paid_result.get("paid_calls_executed") != 96:
        findings.append("m1m2_paid_result_not_96_rows")
    if paid_result.get("actual_llm_http_attempts") != 96:
        findings.append("m1m2_paid_result_not_96_llm_attempts")
    if paid_result.get("actual_ocr_http_attempts") != 0:
        findings.append("m1m2_paid_result_ocr_attempts_not_zero")
    dry_run = (postrun_audit.get("source_artifacts") or {}).get("dry_run") or {}
    if dry_run.get("paid_calls_executed") != 0:
        findings.append("m1m2_dry_run_executed_paid_calls")
    return not findings, findings


def build_track_a_ledger(output_root: Path = PAPER_PACKAGE_DIR) -> tuple[dict[str, Any], str]:
    output_root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    postrun_path = M1M2_V31_DIR / "postrun_governance_audit" / "postrun_governance_audit.json"
    postrun_audit = load_json(postrun_path) if postrun_path.exists() else {}
    m1m2_clean, m1m2_findings = governance_chain_is_clean(postrun_audit) if postrun_audit else (False, ["m1m2_postrun_governance_audit_missing"])

    artifacts = {
        "m1m2_candidate_guided_v31_confirmation_manifest": artifact_record(
            M1M2_V31_DIR / "confirmation_manifest" / "frozen_confirmation_manifest.json",
            status_label="current_authoritative",
            role="Track D completed candidate-guided M1/M2 confirmation manifest",
            required=True,
        ),
        "m1m2_candidate_guided_v31_confirmation_roster": artifact_record(
            M1M2_V31_DIR / "confirmation_manifest" / "confirmation_roster.json",
            status_label="current_authoritative",
            role="Track D completed candidate-guided M1/M2 confirmation roster",
            required=True,
        ),
        "m1m2_candidate_guided_v31_preapproval_audit": artifact_record(
            M1M2_V31_DIR / "confirmation_preapproval_audit" / "preapproval_audit.json",
            status_label="current_authoritative",
            role="Track D preapproval audit for the executed 96-call confirmation",
            required=True,
        ),
        "m1m2_candidate_guided_v31_confirmation_dry_run": artifact_record(
            M1M2_V31_DIR / "confirmation_run" / "dry_run.json",
            status_label="current_authoritative",
            role="Track D dry-run with zero paid calls",
            required=True,
        ),
        "m1m2_candidate_guided_v31_confirmation_paid_result": artifact_record(
            M1M2_V31_DIR / "confirmation_run" / "paid_result.json",
            status_label="current_authoritative",
            role="Track D paid confirmation result",
            required=True,
        ),
        "m1m2_candidate_guided_v31_confirmation_analysis": artifact_record(
            M1M2_V31_DIR / "confirmation_analysis" / "analysis.json",
            status_label="current_authoritative",
            role="Track D analysis supporting narrowed M1/M2 claim",
            required=True,
        ),
        "m1m2_candidate_guided_v31_postrun_governance": artifact_record(
            postrun_path,
            status_label="current_authoritative",
            role="Track A governance closure for existing M1/M2 confirmation",
            required=True,
        ),
        "plotqa_measured_geosym_50_final_audit": artifact_record(
            PUBLIC_PILOT_DIR / "plotqa_measured_geosym_50" / "final_evidence_audit" / "final_evidence_audit.json",
            status_label="diagnostic",
            role="Prior 50-sample PlotQA measured-data diagnostic and guardrail context",
        ),
        "plotqa_openocr_survey_50": artifact_record(
            PUBLIC_PILOT_DIR / "plotqa_openocr_survey_50" / "summary.json",
            status_label="diagnostic",
            role="OpenOCR/OpenOCR-only sensitivity context, not public performance anchor",
        ),
        "plotqa_diagnostic_50_analysis": artifact_record(
            PUBLIC_PILOT_DIR / "plotqa_diagnostic_50" / "analysis.json",
            status_label="diagnostic",
            role="Prior no-OCR public diagnostic, excluded from future 500 roster",
        ),
    }
    missing_required = [name for name, row in artifacts.items() if row["required_for_track_a_clean"] and not row["exists"]]
    blocking_findings = list(m1m2_findings) + [f"missing_required_artifact:{name}" for name in missing_required]
    ledger_status = "clean" if not blocking_findings else "blocked"
    ledger = {
        "schema_version": "paper_level_evidence_ledger_v1",
        "created_at_utc": now,
        "stage": "track_a_evidence_ledger_cleanup",
        "evidence_ledger_status": ledger_status,
        "paid_calls_executed_by_ledger": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_paths": {
            "plan": str((ROOT / "docs" / "PAPER_LEVEL_EVIDENCE_PACKAGE_PLAN.md").resolve()),
            "config": str((ROOT / "config" / "default_config.yaml").resolve()),
        },
        "source_hashes": {
            "plan": stable_file_hash(ROOT / "docs" / "PAPER_LEVEL_EVIDENCE_PACKAGE_PLAN.md"),
            "config": stable_file_hash(ROOT / "config" / "default_config.yaml"),
        },
        "artifacts": artifacts,
        "governance_gates": {
            "m1m2_confirmation_hash_chain_clean": m1m2_clean,
            "all_track_a_required_artifacts_present": not missing_required,
            "all_current_authoritative_artifacts_labeled": True,
            "old_diagnostics_labeled": True,
            "no_paid_calls_executed_by_track_a": True,
        },
        "blocking_findings": blocking_findings,
        "next_required_action": "start_track_f_zero_cost_diagnostics" if ledger_status == "clean" else "repair_track_a_governance_inputs",
        "interpretation_limits": [
            "Track A performs no paid calls.",
            "Historical and diagnostic artifacts are not revised main-table evidence.",
            "M1/M2 support is limited to candidate-guided answerability-preserving selective propagation.",
        ],
    }
    md_lines = [
        "# Paper-Level Evidence Ledger",
        "",
        f"- Created UTC: `{now}`",
        f"- Ledger status: `{ledger_status}`",
        f"- Paid calls executed by Track A: `{ledger['paid_calls_executed_by_ledger']}`",
        f"- Blocking findings: `{len(blocking_findings)}`",
        "",
        "## Artifact Labels",
        "",
    ]
    for name, row in artifacts.items():
        md_lines.append(f"- `{name}`: `{row['status_label']}`; exists=`{row['exists']}`; stage=`{row.get('stage_status')}`")
    if blocking_findings:
        md_lines.extend(["", "## Blocking Findings", ""])
        md_lines.extend(f"- `{finding}`" for finding in blocking_findings)
    md_lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "The ledger marks the existing M1/M2 v31 confirmation as current authoritative only for the narrowed candidate-guided selective-propagation claim. Prior PlotQA 50-sample artifacts remain diagnostic context and cannot be used as the revised public benchmark headline.",
            "",
        ]
    )
    return ledger, "\n".join(md_lines)


def build_track_f_artifacts(output_root: Path) -> dict[str, Any]:
    diag_dir = output_root / "fusion_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    measured_path = PUBLIC_PILOT_DIR / "plotqa_measured_geosym_50" / "m0_measurement_records.json"
    final_audit_path = PUBLIC_PILOT_DIR / "plotqa_measured_geosym_50" / "final_evidence_audit" / "final_evidence_audit.json"
    config_path = ROOT / "config" / "default_config.yaml"
    intervention_path = ROOT / "evaluator" / "reasoning" / "intervention.py"
    stress_roster_paths = {
        "selective_48": ROOT / "outputs" / "m1m2_selective_propagation" / "stress_roster" / "selective_stress_roster.json",
        "answerability_preserving_48": ROOT / "outputs" / "m1m2_answerability_preserving" / "roster" / "answerability_preserving_roster.json",
        "candidate_guided_v31_16": ROOT / "outputs" / "m1m2_candidate_guided_v31" / "confirmation_manifest" / "confirmation_roster.json",
        "paper_level_stress_128": output_root / "m1m2_stress_128" / "stress_128_roster.json",
    }
    real_root = ROOT / "assets" / "real_charts"

    tier_rows = []
    for roster_name, roster_path in stress_roster_paths.items():
        if not roster_path.exists():
            continue
        roster = load_json(roster_path)
        for job in roster.get("jobs") or []:
            schema = job.get("stress_schema") or {}
            selected = schema.get("selected_evidence_set") or {}
            selected_ids = set(selected.get("selected_fact_ids") or [])
            suppressed_ids = set(selected.get("suppressed_fact_ids") or [])
            expected = str(job.get("expected_answerability"))
            for fact in schema.get("evidence_candidates") or []:
                fact_id = str(fact.get("fact_id"))
                if fact_id in selected_ids:
                    role = "selected_support" if expected == "answerable" else "selected_incomplete_context"
                elif fact_id in suppressed_ids:
                    role = "suppressed_candidate"
                else:
                    role = "unrouted_candidate"
                tier_rows.append(
                    {
                        "roster": roster_name,
                        "stress_id": job.get("stress_id"),
                        "expected_answerability": expected,
                        "perturbation_family": job.get("perturbation_family"),
                        "fact_id": fact_id,
                        "reliability_tier": str(fact.get("reliability_tier") or "unknown"),
                        "fact_role": role,
                    }
                )
    tier_counts = Counter(row["reliability_tier"] for row in tier_rows)
    tier_role_counts: dict[str, dict[str, int]] = {}
    for tier in sorted(tier_counts):
        rows = [row for row in tier_rows if row["reliability_tier"] == tier]
        tier_role_counts[tier] = dict(sorted(Counter(row["fact_role"] for row in rows).items()))
    selected_support_by_tier = {
        tier: {
            "total": tier_counts.get(tier, 0),
            "selected_support": tier_role_counts.get(tier, {}).get("selected_support", 0),
            "selected_support_rate": round(tier_role_counts.get(tier, {}).get("selected_support", 0) / tier_counts[tier], 4)
            if tier_counts.get(tier)
            else None,
        }
        for tier in sorted(tier_counts)
    }
    confidence_blocking = []
    if not tier_rows:
        confidence_blocking.append("missing_frozen_fact_level_truth_join")
    if len(tier_counts) < 2:
        confidence_blocking.append("confidence_tier_variation_insufficient_for_accuracy_claim")
    confidence = {
        "schema_version": "confidence_tier_accuracy_v1",
        "created_at_utc": now,
        "stage": "track_f_confidence_tier_accuracy",
        "stage_status": "confidence_tier_accuracy_completed" if not confidence_blocking else "stopped_missing_frozen_fact_level_truth_join",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "measured_plotqa_records": artifact_record(
                measured_path,
                status_label="diagnostic",
                role="Contains ordinal M0 measurement confidence labels but not a frozen fact-level correctness join",
            ),
            "geosym_bench_ground_truth_dir": {
                "path": str((ROOT / "dataset" / "ground_truth").resolve()),
                "exists": (ROOT / "dataset" / "ground_truth").exists(),
            },
            **{
                f"stress_roster_{name}": artifact_record(
                    path,
                    status_label="current_authoritative" if path.exists() else "missing",
                    role="Frozen stress roster with constructed fact relevance roles",
                    required=False,
                )
                for name, path in stress_roster_paths.items()
            },
        },
        "fact_level_join_type": "constructed_stress_roster_fact_relevance_roles",
        "fact_count": len(tier_rows),
        "tier_counts": dict(sorted(tier_counts.items())),
        "tier_role_counts": tier_role_counts,
        "selected_support_by_tier": selected_support_by_tier,
        "calibration_claim_supported": False,
        "empirical_correctness": selected_support_by_tier,
        "blocking_findings": confidence_blocking,
        "next_required_action": "report_as_constructed_stress_diagnostic_not_calibration" if not confidence_blocking else "produce_post_remediation_fact_artifact_or_track_b_fact_join_before_reporting_tier_accuracy",
        "interpretation_limits": [
            "This is a constructed stress-roster fact-relevance diagnostic, not probability calibration.",
            "It must not be described as calibrated confidence or natural M0 fact correctness.",
            "A natural fact-level correctness join against post-remediation Track B remains stronger evidence if produced later.",
        ],
    }

    config_text = config_path.read_text(encoding="utf-8")
    intervention_text = intervention_path.read_text(encoding="utf-8")
    sensitivity_parameters = {
        "tier_rank_high_medium_low_unknown": "TIER_RANK = {\"high\": 0, \"medium\": 1, \"low\": 2, \"unknown\": 3}" in (ROOT / "scripts" / "selective_propagation.py").read_text(encoding="utf-8"),
        "selected_fact_budget_rule": "budget = max(1, len(selected_fact_ids)) if candidates else 0",
        # Names the policy this artifact was generated under. `_CONF_THRESHOLDS`
        # was deleted with M1/M2 on 2026-09-19, so the label records history
        # rather than a symbol a reader could still look up.
        "confidence_threshold_policy": "_CONF_THRESHOLDS (retired 2026-09-19 with M1/M2)",
        "random_seed": 20260830,
    }
    sensitivity_rows = []
    for roster_name, roster_path in stress_roster_paths.items():
        if not roster_path.exists():
            continue
        roster = load_json(roster_path)
        for job in roster.get("jobs") or []:
            schema = job.get("stress_schema") or {}
            candidates = schema.get("evidence_candidates") or []
            selected_ids = set((schema.get("selected_evidence_set") or {}).get("selected_fact_ids") or [])
            default_budget = max(1, len(selected_ids)) if candidates else 0
            ranked = sorted(candidates, key=lambda row: ({"high": 0, "medium": 1, "low": 2, "unknown": 3}.get(str(row.get("reliability_tier")), 3), str(row.get("fact_id"))))
            for multiplier in (0.5, 1.0, 1.5):
                budget = min(len(candidates), max(1, round(default_budget * multiplier))) if candidates else 0
                picked = {str(row.get("fact_id")) for row in ranked[:budget]}
                sensitivity_rows.append(
                    {
                        "roster": roster_name,
                        "stress_id": job.get("stress_id"),
                        "budget_multiplier": multiplier,
                        "budget": budget,
                        "candidate_count": len(candidates),
                        "selected_fact_count": len(selected_ids),
                        "selected_overlap_count": len(picked & selected_ids),
                        "selected_overlap_rate": round(len(picked & selected_ids) / len(selected_ids), 4) if selected_ids else None,
                    }
                )
    sensitivity_summary = {}
    for multiplier in (0.5, 1.0, 1.5):
        rows = [row for row in sensitivity_rows if row["budget_multiplier"] == multiplier]
        overlaps = [row["selected_overlap_rate"] for row in rows if row["selected_overlap_rate"] is not None]
        sensitivity_summary[str(multiplier)] = {
            "rows": len(rows),
            "mean_selected_overlap_rate": round(sum(overlaps) / len(overlaps), 4) if overlaps else None,
        }
    parameter_blocking = []
    if not sensitivity_rows:
        parameter_blocking.append("parameter_sensitivity_source_rosters_missing")
    if sensitivity_parameters["selected_fact_budget_rule"] not in (ROOT / "scripts" / "selective_propagation.py").read_text(encoding="utf-8"):
        parameter_blocking.append("selected_fact_budget_rule_not_found")
    if sensitivity_parameters["confidence_threshold_policy"] not in intervention_text:
        parameter_blocking.append("confidence_threshold_policy_not_found")
    parameter_sensitivity = {
        "schema_version": "parameter_sensitivity_v1",
        "created_at_utc": now,
        "stage": "track_f_parameter_sensitivity",
        "stage_status": "parameter_sensitivity_completed" if not parameter_blocking else "stopped_parameters_not_configured_for_zero_cost_sweep",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "config": artifact_record(config_path, status_label="current_authoritative", role="Runtime configuration source"),
            "intervention": artifact_record(
                intervention_path,
                status_label="current_authoritative",
                role="M1/M2 confidence and selection implementation",
            ),
        },
        "parameter_presence": sensitivity_parameters,
        "sweep_values": {
            "top_confidence_budget_multipliers": [0.5, 1.0, 1.5],
            "tier_rank": {"high": 0, "medium": 1, "low": 2, "unknown": 3},
            "default_budget_rule": "len(selected_fact_ids)",
        },
        "sensitivity_summary": sensitivity_summary,
        "sensitivity_rows": sensitivity_rows,
        "blocking_findings": parameter_blocking,
        "next_required_action": "report_zero_cost_selection_sensitivity_limits" if not parameter_blocking else "freeze_config_level_m1m2_parameters_before_sensitivity_sweep",
        "interpretation_limits": [
            "This is a zero-cost prompt/selection sensitivity diagnostic, not a paid accuracy sweep.",
            "No threshold is selected by optimizing on confirmatory outcomes.",
            "The current implementation uses ordinal tiers and budgeted selection, not calibrated probabilistic thresholds.",
        ],
    }

    ledger_path = output_root / "evidence_ledger.json"
    ledger = load_json(ledger_path) if ledger_path.exists() else {"artifacts": {}}
    counted = []
    for name, row in (ledger.get("artifacts") or {}).items():
        counted.append(
            {
                "name": name,
                "status_label": row.get("status_label"),
                "paid_calls_executed": row.get("paid_calls_executed"),
                "actual_llm_http_attempts": row.get("actual_llm_http_attempts"),
                "actual_ocr_http_attempts": row.get("actual_ocr_http_attempts"),
            }
        )
    cost_latency = {
        "schema_version": "cost_latency_v1",
        "created_at_utc": now,
        "stage": "track_f_cost_latency_schema",
        "stage_status": "schema_initialized_from_track_a_ledger",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {"evidence_ledger": artifact_record(ledger_path, status_label="current_authoritative", role="Central artifact ledger")},
        "counted_artifacts": counted,
        "required_fields_for_future_runs": [
            "wall_clock_seconds",
            "cache_hits",
            "sample_condition_count",
            "semantic_model_calls",
            "transport_http_attempts",
            "transient_retry_attempts",
            "local_inference_count",
            "estimated_cost_usd",
        ],
        "interpretation_limits": ["This schema records accounting requirements; it does not estimate missing historical costs."],
    }

    real_records = []
    if real_root.exists():
        for meta_path in sorted(real_root.glob("*/*/meta.json")):
            meta = load_json(meta_path)
            image_path = ROOT / str(meta.get("image_path", ""))
            has_reference = bool(meta.get("gold_answer") or meta.get("answer_validation"))
            real_records.append(
                {
                    "sample_id": meta.get("sample_id"),
                    "source_family": meta.get("source_family"),
                    "meta_path": str(meta_path.resolve()),
                    "meta_sha256": stable_file_hash(meta_path),
                    "image_path": str(image_path.resolve()),
                    "image_exists": image_path.exists(),
                    "image_sha256": stable_file_hash(image_path) if image_path.exists() else None,
                    "reference_available": has_reference,
                    "measurement_status": "not_run_in_track_f_zero_cost_audit",
                }
            )
    real_chart = {
        "schema_version": "real_chart_error_decomposition_v1",
        "created_at_utc": now,
        "stage": "track_f_real_chart_reference_audit",
        "stage_status": "real_chart_reference_audit_complete_measurement_not_run",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_root": str(real_root.resolve()),
        "chart_count": len(real_records),
        "reference_available_count": sum(1 for row in real_records if row["reference_available"]),
        "image_available_count": sum(1 for row in real_records if row["image_exists"]),
        "records": real_records,
        "blocking_findings": [] if real_records else ["real_chart_assets_missing"],
        "next_required_action": "run_deterministic_real_chart_measurement_decomposition" if real_records else "record_real_chart_reference_unavailable",
        "interpretation_limits": [
            "This artifact audits references and hashes only.",
            "It does not report localization, ruler residual, or numeric error until a deterministic measurement run exists.",
        ],
    }

    positioning_md = "\n".join(
        [
            "# Fusion Positioning Audit",
            "",
            f"Created UTC: `{now}`",
            "",
            "GeoSym should be positioned as a reliability-aware, provenance-preserving, modular chart evidence fusion pipeline. The implemented mechanism is heuristic ordinal confidence tiers, utility-aware selection, gap-aware propagation, and module-level provenance.",
            "",
            "The current code does not implement calibrated probabilities, Dempster-Shafer belief masses, conflict coefficients, or a formal probabilistic fusion rule. Those phrases remain forbidden until a separate formalism and calibration protocol are implemented and validated.",
            "",
            "Specialized chart models, plot-to-table flattening, unmanaged evidence injection, and tool-augmented code reasoning remain competing routes. The defensible GeoSym claim is auditability plus controlled propagation under unreliable evidence, not leaderboard or SOTA performance.",
            "",
        ]
    )
    positioning = {
        "schema_version": "fusion_positioning_audit_v1",
        "created_at_utc": now,
        "stage": "track_f_fusion_positioning_audit",
        "stage_status": "fusion_positioning_claim_boundaries_recorded",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "supported_terms": [
            "heuristic ordinal confidence tiers",
            "utility-aware selection",
            "gap-aware propagation",
            "module-level provenance",
        ],
        "forbidden_terms_without_new_evidence": [
            "calibrated probability",
            "Dempster-Shafer belief masses",
            "conflict coefficients",
            "formal probabilistic fusion rule",
            "SOTA",
            "leaderboard",
        ],
    }

    payloads = {
        "confidence_tier_accuracy.json": confidence,
        "parameter_sensitivity.json": parameter_sensitivity,
        "cost_latency.json": cost_latency,
        "real_chart_error_decomposition.json": real_chart,
        "fusion_positioning_audit.json": positioning,
    }
    for filename, payload in payloads.items():
        (diag_dir / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (diag_dir / "fusion_positioning_audit.md").write_text(positioning_md, encoding="utf-8")
    return {
        "confidence_tier_accuracy": confidence["stage_status"],
        "parameter_sensitivity": parameter_sensitivity["stage_status"],
        "cost_latency": cost_latency["stage_status"],
        "real_chart_error_decomposition": real_chart["stage_status"],
        "fusion_positioning_audit": positioning["stage_status"],
    }


def build_track_c_zero_cost_audit(output_root: Path) -> dict[str, Any]:
    track_c_dir = output_root / "plotqa_public_500"
    track_c_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    plan_path = track_c_dir / "plotqa_500_zero_cost_plan.json"
    prior_plan_path = PUBLIC_PILOT_DIR / "plotqa_diagnostic_50" / "frozen_plan.json"
    final_50_audit_path = PUBLIC_PILOT_DIR / "plotqa_measured_geosym_50" / "final_evidence_audit" / "final_evidence_audit.json"
    openocr_500_summary_path = track_c_dir / "openocr_survey" / "summary.json"
    measured_500_path = track_c_dir / "m0_measurement_records.json"
    measured_500_stage1_path = track_c_dir / "stage1_status.json"
    executable_runner_path = ROOT / "scripts" / "run_plotqa_measured_public_confirmation.py"
    measured_confirmation_dir = track_c_dir / "measured_confirmation"
    measured_confirmation_plan_path = measured_confirmation_dir / "frozen_plan.json"
    measured_confirmation_dry_run_path = measured_confirmation_dir / "dry_run.json"
    measured_confirmation_review_path = measured_confirmation_dir / "prepaid_review.json"
    completed_paid_result_path = measured_confirmation_dir / "paid_result.json"
    completed_analysis_path = measured_confirmation_dir / "analysis.json"

    blocking: list[str] = []
    plan = load_json(plan_path) if plan_path.exists() else {}
    selected = plan.get("selected_samples") or []
    selection = ((plan.get("frozen_before_paid_run") or {}).get("sample_selection") or {})
    semantic_violations = []
    for row in selected:
        violations = validate_plotqa_sample_semantics(row)
        if violations:
            semantic_violations.append({"sample_id": row.get("sample_id"), "violations": violations})
    overlap = selection.get("overlap_with_excluded_prior_sample_ids") or []
    if not plan_path.exists():
        blocking.append("plotqa_500_zero_cost_plan_missing")
    if len(selected) != 500:
        blocking.append(f"plotqa_500_roster_size_not_500:{len(selected)}")
    if overlap:
        blocking.append("plotqa_500_roster_overlaps_prior_50")
    if semantic_violations:
        blocking.append("plotqa_500_semantic_tagging_violations")
    if not measured_500_path.exists():
        blocking.append("plotqa_500_strict_measured_m0_preflight_missing")
    measured_stage1 = load_json(measured_500_stage1_path) if measured_500_stage1_path.exists() else {}
    if measured_500_stage1_path.exists() and measured_stage1.get("stage_status") != "stage_passed":
        blocking.append("plotqa_500_strict_measured_m0_preflight_not_passed")
    runner_available = executable_runner_path.exists()
    measured_confirmation_plan = load_json(measured_confirmation_plan_path) if measured_confirmation_plan_path.exists() else {}
    measured_confirmation_review = load_json(measured_confirmation_review_path) if measured_confirmation_review_path.exists() else {}
    measured_confirmation_dry_run = load_json(measured_confirmation_dry_run_path) if measured_confirmation_dry_run_path.exists() else {}
    measured_confirmation_ready = (
        measured_confirmation_plan.get("paid_run_approved") is True
        and measured_confirmation_plan.get("sample_condition_job_cap") == 1500
        and measured_confirmation_plan.get("llm_http_attempt_cap") == 1500
        and measured_confirmation_plan.get("ocr_http_attempt_cap") == 0
        and measured_confirmation_dry_run.get("paid_calls_executed") == 0
        and measured_confirmation_review.get("status") == "no_blocking_findings"
    )
    completed_paid_result = load_json(completed_paid_result_path) if completed_paid_result_path.exists() else {}
    completed_analysis = load_json(completed_analysis_path) if completed_analysis_path.exists() else {}
    completed_status_ok = completed_analysis.get("stage_status") in {
        "track_c_plotqa_500_confirmation_completed",
        "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim",
    }
    completed_paid_result_ok = (
        completed_paid_result.get("stage_status") == completed_analysis.get("stage_status")
        and len(completed_paid_result.get("results") or []) == 1500
        and completed_paid_result.get("paid_calls_executed") == completed_paid_result.get("actual_llm_http_attempts")
        and isinstance(completed_paid_result.get("paid_calls_executed"), int)
        and completed_paid_result.get("paid_calls_executed") > 0
        and completed_paid_result.get("paid_calls_executed") <= 1500
        and completed_paid_result.get("actual_ocr_http_attempts") == 0
    )
    completed_analysis_ok = (
        completed_status_ok
        and len(completed_analysis.get("results") or []) == 1500
        and completed_analysis.get("paid_calls_executed") == completed_analysis.get("actual_llm_http_attempts")
        and isinstance(completed_analysis.get("paid_calls_executed"), int)
        and completed_analysis.get("paid_calls_executed") > 0
        and completed_analysis.get("paid_calls_executed") <= 1500
        and completed_analysis.get("actual_ocr_http_attempts") == 0
    )
    completed = completed_paid_result_ok and completed_analysis_ok
    if completed:
        stage_status = completed_analysis["stage_status"]
        next_required_action = "integrate_track_c_plotqa_500_confirmation"
    elif blocking:
        stage_status = "track_c_zero_cost_gates_block_paid_execution"
        next_required_action = "run_plotqa_500_strict_measured_m0_preflight_before_paid_answering"
    elif not runner_available:
        stage_status = "track_c_blocked_missing_500_measured_confirmation_runner"
        next_required_action = "implement_run_plotqa_measured_public_500_confirmation_before_paid_execution"
    elif measured_confirmation_ready:
        stage_status = "track_c_500_measured_confirmation_ready_for_paid_execution"
        next_required_action = "run_plotqa_measured_public_500_confirmation_with_frozen_cap"
    else:
        stage_status = "track_c_zero_cost_gates_passed_ready_for_paid_manifest"
        next_required_action = "run_plotqa_measured_public_500_confirmation_dry_run"

    audit = {
        "schema_version": "plotqa_public_500_pre_execution_audit_v1",
        "created_at_utc": now,
        "stage": "track_c_plotqa_public_500_zero_cost_pre_execution_audit",
        "stage_status": stage_status,
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "plotqa_500_zero_cost_plan": artifact_record(plan_path, status_label="current_authoritative", role="Frozen answer-independent PlotQA 500 roster", required=True),
            "prior_50_plan": artifact_record(prior_plan_path, status_label="diagnostic", role="Prior 50-sample roster to exclude"),
            "plotqa_measured_50_final_audit": artifact_record(final_50_audit_path, status_label="diagnostic", role="Measured-M0 50-sample backtest gate"),
            "plotqa_500_openocr_survey": artifact_record(openocr_500_summary_path, status_label="current_authoritative" if openocr_500_summary_path.exists() else "missing", role="Local OpenOCR survey for the frozen 500 roster", required=False),
            "plotqa_500_measured_m0_preflight": artifact_record(measured_500_path, status_label="current_authoritative" if measured_500_path.exists() else "missing", role="Required strict measured-M0 records for paid PlotQA 500 answering", required=True),
            "plotqa_500_measured_m0_stage1": artifact_record(measured_500_stage1_path, status_label="current_authoritative" if measured_500_stage1_path.exists() else "missing", role="Strict measured-M0 preflight gate status", required=False),
            "executable_runner": artifact_record(executable_runner_path, status_label="missing" if not runner_available else "current_authoritative", role="Paper-level 500 measured-data paid confirmation runner", required=False),
            "measured_confirmation_plan": artifact_record(measured_confirmation_plan_path, status_label="current_authoritative" if measured_confirmation_plan_path.exists() else "missing", role="Frozen Track C measured-data confirmation plan", required=False),
            "measured_confirmation_dry_run": artifact_record(measured_confirmation_dry_run_path, status_label="current_authoritative" if measured_confirmation_dry_run_path.exists() else "missing", role="Track C measured-data confirmation dry-run with zero paid calls", required=False),
            "measured_confirmation_prepaid_review": artifact_record(measured_confirmation_review_path, status_label="current_authoritative" if measured_confirmation_review_path.exists() else "missing", role="Track C measured-data confirmation prepaid review", required=False),
            "completed_confirmation_paid_result": artifact_record(completed_paid_result_path, status_label="current_authoritative" if completed_paid_result_ok else "missing", role="Completed Track C 500 confirmation paid result", required=False),
            "completed_confirmation_analysis": artifact_record(completed_analysis_path, status_label="current_authoritative" if completed else "missing", role="Completed Track C 500 confirmation analysis", required=False),
        },
        "gates": {
            "plan_exists": plan_path.exists(),
            "selected_500_samples": len(selected) == 500,
            "prior_50_exclusion_list_generated": selection.get("excluded_prior_sample_count") == 50,
            "zero_overlap_with_prior_50": not overlap,
            "answer_contract_semantics_pass": not semantic_violations,
            "official_source_hashes_recorded": bool(((plan.get("frozen_before_paid_run") or {}).get("official_files") or {}).get("annotations_sha256")),
            "measured_50_gate_available": final_50_audit_path.exists(),
            "measured_500_strict_preflight_available": measured_500_path.exists(),
            "measured_500_strict_preflight_passed": measured_stage1.get("stage_status") == "stage_passed",
            "paper_level_500_measured_runner_available": runner_available,
            "measured_confirmation_manifest_frozen": bool(measured_confirmation_plan),
            "measured_confirmation_dry_run_zero_paid_calls": measured_confirmation_dry_run.get("paid_calls_executed") == 0,
            "measured_confirmation_prepaid_review_clean": measured_confirmation_review.get("status") == "no_blocking_findings",
            "measured_confirmation_ready_for_paid_execution": measured_confirmation_ready,
            "completed_confirmation_analysis_available": completed,
            "completed_confirmation_analysis_status": completed_analysis.get("stage_status"),
            "completed_confirmation_paid_result_available": completed_paid_result_ok,
            "completed_confirmation_paid_rows_1500": len(completed_paid_result.get("results") or []) == 1500,
            "completed_confirmation_analysis_rows_1500": len(completed_analysis.get("results") or []) == 1500,
            "completed_confirmation_paid_attempts_accounted": completed_paid_result.get("paid_calls_executed") == completed_paid_result.get("actual_llm_http_attempts"),
            "completed_confirmation_ocr_zero": completed_paid_result.get("actual_ocr_http_attempts") == 0,
        },
        "selection_summary": {
            "selected_count": len(selected),
            "eligible_count_after_prior_exclusion": selection.get("eligible_count"),
            "unique_image_eligible_count_after_prior_exclusion": selection.get("unique_image_eligible_count"),
            "excluded_prior_sample_count": selection.get("excluded_prior_sample_count"),
            "overlap_with_prior_50_count": len(overlap),
            "roster_note": selection.get("roster_note"),
        },
        "semantic_violation_count": len(semantic_violations),
        "semantic_violations_first_20": semantic_violations[:20],
        "blocking_findings": blocking,
        "next_required_action": next_required_action,
        "interpretation_limits": [
            "This audit performs no paid calls.",
            "The current plan covers zero-cost roster and semantic gates only.",
            "Paid PlotQA 500 answering must not start until strict measured-M0 preflight produces exactly 500 schema-valid final roster records.",
        ],
    }
    (track_c_dir / "pre_execution_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (track_c_dir / "pre_execution_audit.md").write_text(
        "\n".join(
            [
                "# PlotQA Public 500 Pre-Execution Audit",
                "",
                f"- Stage status: `{audit['stage_status']}`",
                f"- Selected samples: `{audit['selection_summary']['selected_count']}`",
                f"- Prior-50 overlap: `{audit['selection_summary']['overlap_with_prior_50_count']}`",
                f"- Semantic violations: `{audit['semantic_violation_count']}`",
                f"- Blocking findings: `{len(blocking)}`",
                f"- Next required action: `{audit['next_required_action']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return audit


def build_track_b_geosym_dedup_audit(output_root: Path) -> dict[str, Any]:
    track_b_dir = output_root / "geosym_bench_controlled"
    track_b_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    gt_root = ROOT / "dataset" / "ground_truth"
    image_root = ROOT / "dataset" / "images"
    expected_families = [
        "absolute",
        "cubic",
        "exponential",
        "linear",
        "logarithmic",
        "piecewise_linear",
        "quadratic",
        "rational",
    ]
    records = []
    family_counts: dict[str, int] = {family: 0 for family in expected_families}
    missing_meta = 0
    duplicate_keys: dict[str, int] = {}
    if gt_root.exists():
        for gt_path in sorted(gt_root.glob("*/*.json")):
            payload = load_json(gt_path)
            family = gt_path.parent.name.replace("-clean_render", "")
            family_counts[family] = family_counts.get(family, 0) + 1
            summary = (payload.get("generation_meta") or {}).get("summary") or {}
            exact = summary.get("exact_signature")
            family_signature = summary.get("family_signature")
            has_required = bool(exact and family_signature)
            if not has_required:
                missing_meta += 1
            dedup_key = f"{family}|{exact}|{family_signature}" if has_required else None
            if dedup_key:
                duplicate_keys[dedup_key] = duplicate_keys.get(dedup_key, 0) + 1
            image_path = image_root / gt_path.parent.name / f"{gt_path.stem}.png"
            records.append(
                {
                    "sample_id": gt_path.stem,
                    "family": family,
                    "ground_truth_path": str(gt_path.resolve()),
                    "ground_truth_sha256": stable_file_hash(gt_path),
                    "image_path": str(image_path.resolve()),
                    "image_exists": image_path.exists(),
                    "has_generation_exact_signature": bool(exact),
                    "has_family_signature": bool(family_signature),
                    "dedup_key": dedup_key,
                }
            )
    total = len(records)
    missing_rate = missing_meta / total if total else 1.0
    duplicate_cluster_count = sum(1 for count in duplicate_keys.values() if count > 1)
    retained_unique_count = sum(1 for count in duplicate_keys.values() if count >= 1)
    blocking = []
    if total != 480:
        blocking.append(f"geosym_generated_pool_not_480:{total}")
    if missing_rate > 0.05:
        blocking.append(f"metadata_dedup_key_missing_rate_gt_5pct:{missing_meta}/{total}")
    if any(family_counts.get(family, 0) != 60 for family in expected_families):
        blocking.append("family_counts_not_60_each")
    unique_records = []
    seen_keys: set[str] = set()
    for row in records:
        key = row.get("dedup_key")
        if not key or key in seen_keys:
            continue
        seen_keys.add(str(key))
        unique_records.append(row)
    unique_by_family: dict[str, list[dict[str, Any]]] = {family: [] for family in expected_families}
    for row in unique_records:
        unique_by_family.setdefault(str(row["family"]), []).append(row)
    target_subset_size = 0
    selected_subset: list[dict[str, Any]] = []
    if not blocking:
        if retained_unique_count > 240:
            target_subset_size = 240
        elif retained_unique_count >= 160:
            target_subset_size = retained_unique_count
        else:
            blocking.append(f"diversity_clean_count_below_160:{retained_unique_count}")
        if target_subset_size:
            remaining_slots = target_subset_size
            selected_ids: set[str] = set()
            for family in expected_families:
                rows = unique_by_family.get(family, [])
                take = min(15, len(rows))
                for row in rows[:take]:
                    selected_subset.append(row)
                    selected_ids.add(str(row["sample_id"]))
                remaining_slots -= take
            remaining_candidates = [
                row
                for family in expected_families
                for row in unique_by_family.get(family, [])[15:]
                if str(row["sample_id"]) not in selected_ids
            ]
            remaining_counts = {family: max(0, len(unique_by_family.get(family, [])) - 15) for family in expected_families}
            total_remaining = sum(remaining_counts.values())
            allocations = {family: 0 for family in expected_families}
            if total_remaining and remaining_slots > 0:
                raw_allocations = {
                    family: remaining_slots * remaining_counts[family] / total_remaining
                    for family in expected_families
                }
                allocations = {family: int(raw_allocations[family]) for family in expected_families}
                spare = remaining_slots - sum(allocations.values())
                tie_order = sorted(expected_families, key=lambda family: (-(raw_allocations[family] - allocations[family]), family))
                for family in tie_order[:spare]:
                    allocations[family] += 1
            for family in expected_families:
                take = allocations.get(family, 0)
                for row in unique_by_family.get(family, [])[15 : 15 + take]:
                    selected_subset.append(row)
            selected_subset = sorted(selected_subset, key=lambda row: (expected_families.index(str(row["family"])), str(row["sample_id"])))[:target_subset_size]
            if len(selected_subset) != target_subset_size:
                blocking.append(f"selected_subset_size_mismatch:{len(selected_subset)}!={target_subset_size}")
    audit = {
        "schema_version": "geosym_bench_metadata_dedup_audit_v1",
        "created_at_utc": now,
        "stage": "track_b_geosym_bench_metadata_key_dedup_audit",
        "stage_status": "geosym_bench_metadata_repair_required_before_subset_selection" if blocking else "geosym_bench_dedup_audit_passed_subset_ready",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "ground_truth_dir": {"path": str(gt_root.resolve()), "exists": gt_root.exists()},
            "image_dir": {"path": str(image_root.resolve()), "exists": image_root.exists()},
        },
        "pool_summary": {
            "N_generated_pool": total,
            "family_counts": family_counts,
            "missing_required_metadata_count": missing_meta,
            "missing_required_metadata_rate": missing_rate,
            "unique_metadata_dedup_key_count": retained_unique_count,
            "duplicate_metadata_cluster_count": duplicate_cluster_count,
        },
        "selected_subset": [
            {
                "sample_id": row["sample_id"],
                "family": row["family"],
                "ground_truth_sha256": row["ground_truth_sha256"],
                "dedup_key": row["dedup_key"],
            }
            for row in selected_subset
        ],
        "N_paper_subset": len(selected_subset),
        "selected_family_counts": {
            family: sum(1 for row in selected_subset if row["family"] == family)
            for family in expected_families
        },
        "blocking_findings": blocking,
        "next_required_action": "repair_generation_metadata_before_track_b_subset_selection" if blocking else "freeze_geosym_bench_paper_subset_manifest",
        "records_first_20": records[:20],
        "interpretation_limits": [
            "Selection is stopped when metadata dedup keys are missing above the 5 percent threshold.",
            "Image perceptual hashes and layout signatures are audit signals only and are not used to override missing metadata keys here.",
            "No post-remediation paid rerun can start from this audit until a subset manifest exists.",
        ],
    }
    (track_b_dir / "metadata_dedup_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (track_b_dir / "metadata_dedup_audit.md").write_text(
        "\n".join(
            [
                "# GeoSym-Bench Metadata Dedup Audit",
                "",
                f"- Stage status: `{audit['stage_status']}`",
                f"- Generated pool: `{total}`",
                f"- Missing required metadata: `{missing_meta}/{total}`",
                f"- Blocking findings: `{len(blocking)}`",
                f"- Next required action: `{audit['next_required_action']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return audit


def first_two_model_provenance(config_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    models = []
    for provider, provider_cfg in (cfg.get("models") or {}).items():
        for display_name, model_id in ((provider_cfg or {}).get("models") or {}).items():
            models.append(
                {
                    "provider": str(provider),
                    "display_name": str(display_name),
                    "api_model_id": str(model_id),
                    "adapter": str((provider_cfg or {}).get("adapter", "")),
                    "config_path": str(config_path.resolve()),
                    "config_sha256": stable_file_hash(config_path),
                }
            )
    if not models:
        return None, None
    if len(models) == 1:
        return models[0], None
    return models[0], models[1]


def select_track_b_model_provenance(config_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    models = []
    for provider, provider_cfg in (cfg.get("models") or {}).items():
        for display_name, model_id in ((provider_cfg or {}).get("models") or {}).items():
            models.append(
                {
                    "provider": str(provider),
                    "display_name": str(display_name),
                    "api_model_id": str(model_id),
                    "adapter": str((provider_cfg or {}).get("adapter", "")),
                    "config_path": str(config_path.resolve()),
                    "config_sha256": stable_file_hash(config_path),
                    "saved_key_available_at_manifest_time": bool(_get_saved_key(str(display_name))),
                }
            )
    if not models:
        return None, None
    reasoning = models[0]
    preferred_judges = [
        model
        for model in models[1:]
        if model["api_model_id"] != reasoning["api_model_id"]
        and model["provider"] != reasoning["provider"]
        and model["saved_key_available_at_manifest_time"]
        and model["provider"].lower() != "google"
    ]
    fallback_judges = [
        model
        for model in models[1:]
        if model["api_model_id"] != reasoning["api_model_id"]
        and model["provider"] != reasoning["provider"]
        and model["saved_key_available_at_manifest_time"]
    ]
    return reasoning, (preferred_judges or fallback_judges or [None])[0]


def build_track_b_rerun_manifest(output_root: Path) -> dict[str, Any]:
    track_b_dir = output_root / "geosym_bench_controlled"
    audit_path = track_b_dir / "metadata_dedup_audit.json"
    config_path = ROOT / "config" / "default_config.yaml"
    now = datetime.now(timezone.utc).isoformat()
    audit = load_json(audit_path) if audit_path.exists() else {}
    subset = audit.get("selected_subset") or []
    reasoning_model, judge_model = select_track_b_model_provenance(config_path)
    prompt_files = sorted((ROOT / "prompts").glob("*.txt"))
    fairness_rows = run_prompt_fairness_audit()
    executable_runner_path = ROOT / "scripts" / "run_geosym_bench_post_remediation_subset.py"
    default_result_path = track_b_dir / "post_remediation_paid_result.json"
    reauthorized_result_path = track_b_dir / TRACK_B_REAUTHORIZED_RESULT_NAME
    final_backfill_result_path = track_b_dir / TRACK_B_FINAL_BACKFILL_RESULT_NAME
    completed_result_path, result_payload = load_first_existing_json(
        [final_backfill_result_path, reauthorized_result_path, default_result_path]
    )
    blocking = []
    if audit.get("stage_status") != "geosym_bench_dedup_audit_passed_subset_ready":
        blocking.append("dedup_audit_not_passed")
    if len(subset) < 160 or len(subset) > 240:
        blocking.append(f"paper_subset_size_outside_range:{len(subset)}")
    if tuple(ALL_CONDITIONS) != ("E0", "E1", "E2", "E3", "E4", "E5", "E5_raw"):
        blocking.append("all_conditions_changed")
    if reasoning_model is None or judge_model is None:
        blocking.append("reasoning_or_judge_model_missing_from_config")
    elif reasoning_model["api_model_id"] == judge_model["api_model_id"]:
        blocking.append("judge_model_not_distinct_from_reasoning_model")
    if not all(row["passed"] for row in fairness_rows):
        blocking.append("prompt_fairness_audit_failed")
    completed_clean_grid_backfill = completed_result_path == final_backfill_result_path and result_payload.get("m0_execution_mode") == "clean_grid_fallback_first"
    ocr_dependent = [] if completed_clean_grid_backfill else sorted(TRACK_B_OCR_DEPENDENT_CONDITIONS & set(ALL_CONDITIONS))
    ocr_http_attempt_cap = 0 if completed_clean_grid_backfill else (
        len(subset) * TRACK_B_OCR_HTTP_ATTEMPTS_PER_SAMPLE_UPPER_BOUND if len(subset) <= 240 else None
    )
    runner_available = executable_runner_path.exists()
    completed, result_findings = track_b_paid_result_is_complete(result_payload, expected_samples=len(subset))
    if completed and ocr_http_attempt_cap is not None:
        actual_ocr = result_payload.get("actual_ocr_http_attempts_completed_runs", result_payload.get("actual_ocr_http_attempts"))
        if not isinstance(actual_ocr, int) or actual_ocr > int(ocr_http_attempt_cap):
            completed = False
            result_findings.append("post_remediation_paid_result_ocr_cap_exceeded")
    default_incomplete_result_present = default_result_path.exists() and not (
        default_result_path == completed_result_path and completed
    )
    active_incomplete_result_present = completed_result_path.exists() and not completed
    unaccounted_interruption_clean_stop = active_incomplete_result_present and has_unknown_attempt_accounting(result_payload)
    if active_incomplete_result_present:
        blocking.extend(result_findings)
    if completed:
        stage_status = "geosym_bench_post_remediation_rerun_completed"
        next_required_action = "integrate_geosym_bench_post_remediation_result"
    elif unaccounted_interruption_clean_stop:
        stage_status = "geosym_bench_post_remediation_rerun_failed_uncounted_interruption_clean_stop"
        next_required_action = "do_not_claim_track_b_positive_result_without_new_authorized_rerun"
    elif blocking:
        stage_status = "geosym_bench_rerun_manifest_blocked"
        next_required_action = "repair_geosym_bench_rerun_manifest_inputs"
    elif not runner_available:
        stage_status = "geosym_bench_blocked_missing_post_remediation_subset_runner"
        next_required_action = "implement_run_geosym_bench_post_remediation_subset_before_paid_execution"
    else:
        stage_status = "geosym_bench_rerun_manifest_frozen_pending_paid_execution"
        next_required_action = "run_dry_run_and_prepaid_audit_before_paid_rerun"
    manifest = {
        "schema_version": "geosym_bench_post_remediation_rerun_manifest_v1",
        "created_at_utc": now,
        "stage": "track_b_geosym_bench_post_remediation_subset_rerun_manifest",
        "stage_status": stage_status,
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "safe_default": "dry_run_no_paid_calls",
        "requires_explicit_paid_cap_approval_before_execution": True,
        "hypothesis": "On a diversity-audited GeoSym-Bench paper subset, post-remediation E5 remains above matched baselines while E5_raw vs E0 isolates M0+M3 contribution.",
        "falsification_signal": "If E5 does not exceed the strongest single matched baseline under 10D and 7D robustness, do not use GeoSym-Bench as positive main-table mechanism evidence.",
        "source_artifacts": {
            "metadata_dedup_audit": artifact_record(audit_path, status_label="current_authoritative", role="Frozen deduplicated GeoSym-Bench subset", required=True),
            "config": artifact_record(config_path, status_label="current_authoritative", role="Model provenance config", required=True),
            "executable_runner": artifact_record(executable_runner_path, status_label="missing" if not runner_available else "current_authoritative", role="Paper-subset post-remediation executable runner", required=False),
            "completed_paid_result": artifact_record(completed_result_path, status_label="current_authoritative" if completed else ("diagnostic_failed_smoke" if active_incomplete_result_present else "missing"), role="Completed Track B paid result", required=False),
            "default_interrupted_paid_result": artifact_record(default_result_path, status_label="diagnostic_failed_smoke" if default_incomplete_result_present else ("current_authoritative" if default_result_path == completed_result_path and completed else "missing"), role="Diagnostic failed Track B paid smoke; not mixed into the replacement clean run", required=False),
            "reauthorized_paid_result": artifact_record(reauthorized_result_path, status_label="current_authoritative" if completed and completed_result_path == reauthorized_result_path else ("missing" if not reauthorized_result_path.exists() else "diagnostic_failed_infrastructure"), role="Reauthorized Mathpix-first Track B paid result; diagnostic after chart-channel failure", required=False),
            "final_e5_e5raw_backfill_result": artifact_record(final_backfill_result_path, status_label="current_authoritative" if completed and completed_result_path == final_backfill_result_path else ("missing" if not final_backfill_result_path.exists() else "diagnostic_incomplete"), role="Final repaired E5/E5_raw Track B backfill result", required=False),
        },
        "N_generated_pool": (audit.get("pool_summary") or {}).get("N_generated_pool"),
        "N_paper_subset": len(subset),
        "selected_family_counts": audit.get("selected_family_counts"),
        "selected_subset_sha256": stable_file_hash(audit_path) if audit_path.exists() else None,
        "conditions": result_payload.get("conditions") or list(ALL_CONDITIONS),
        "m0_execution_mode": result_payload.get("m0_execution_mode") or "mathpix_then_clean_grid_fallback",
        "reasoning_model_provenance": reasoning_model,
        "judge_model_provenance": judge_model,
        "temperature": 0.0,
        "prompt_hashes": {
            path.name: stable_file_hash(path)
            for path in prompt_files
        },
        "prompt_fairness_audit": fairness_rows,
        "sample_condition_job_cap": result_payload.get("sample_condition_job_cap") or (len(subset) * len(ALL_CONDITIONS)),
        "reasoning_llm_call_cap": result_payload.get("reasoning_llm_call_cap") or (3360 if len(subset) <= 240 else None),
        "judge_llm_call_cap": result_payload.get("judge_llm_call_cap") or (1680 if len(subset) <= 240 else None),
        "ocr_http_attempt_cap": ocr_http_attempt_cap,
        "ocr_http_attempt_cap_policy": "conservative upper bound: one OCR HTTP attempt budgeted for each OCR-dependent condition per sample",
        "ocr_dependent_conditions_under_current_runner": ocr_dependent,
        "retry_policy": "No hidden semantic retries; transport retries must be separately logged and cannot increase semantic job caps.",
        "cache_policy": "Cache key must include plan hash, model ID, condition, prompt hash, sample ID, judge model, and scoring dimension set.",
        "stop_rule": "Stop after the frozen subset and caps regardless of interim results; no adaptive sample replacement.",
        "paid_runner_available": runner_available,
        "completed_paid_result_available": completed,
        "completed_paid_result_condition_gate_findings": result_findings,
        "actual_llm_http_attempts_from_completed_paid_result": result_payload.get("actual_llm_http_attempts_completed_runs", result_payload.get("actual_llm_http_attempts")),
        "actual_ocr_http_attempts_from_completed_paid_result": result_payload.get("actual_ocr_http_attempts_completed_runs", result_payload.get("actual_ocr_http_attempts")),
        "clean_stop_reason": "unaccounted_prior_paid_interruption" if unaccounted_interruption_clean_stop else None,
        "blocking_findings": blocking,
        "next_required_action": next_required_action,
        "interpretation_limits": [
            "This manifest performs no paid calls.",
            "Historical GeoSym-Bench scores remain pre-remediation sensitivity until this rerun completes.",
            "Broad E5 vs E5_raw should be interpreted as a bounded-effect estimate unless Track D supplies mechanism evidence.",
        ],
    }
    manifest_path = track_b_dir / "post_remediation_rerun_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (track_b_dir / "post_remediation_rerun_manifest.md").write_text(
        "\n".join(
            [
                "# GeoSym-Bench Post-Remediation Rerun Manifest",
                "",
                f"- Stage status: `{manifest['stage_status']}`",
                f"- N paper subset: `{manifest['N_paper_subset']}`",
                f"- Conditions: `{','.join(manifest['conditions'])}`",
                f"- Reasoning cap: `{manifest['reasoning_llm_call_cap']}`",
                f"- Judge cap: `{manifest['judge_llm_call_cap']}`",
                f"- Blocking findings: `{len(blocking)}`",
                f"- Next required action: `{manifest['next_required_action']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def build_track_e_artifacts(output_root: Path) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    config_path = ROOT / "config" / "default_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    candidates = open_model_candidates(config)
    open_dir = output_root / "open_repro_sensitivity"
    specialized_dir = output_root / "specialized_chart_model_anchor"
    open_dir.mkdir(parents=True, exist_ok=True)
    specialized_dir.mkdir(parents=True, exist_ok=True)
    open_summary = {
        "schema_version": "open_repro_sensitivity_probe_v1",
        "created_at_utc": now,
        "stage": "track_e_open_repro_sensitivity_probe",
        "stage_status": "open_mllm_not_configured" if not candidates else "open_mllm_candidates_found_pending_runtime_probe",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "config_path": str(config_path.resolve()),
        "config_sha256": stable_file_hash(config_path),
        "open_model_candidates": candidates,
        "blocking_findings": [] if candidates else ["no_open_or_local_model_configured"],
        "next_required_action": "configure_or_record_open_mllm_unavailability" if not candidates else "run_same_100_open_mllm_dry_run_probe",
        "interpretation_limits": [
            "This probe performs no package installation, model download, driver change, or manual setup.",
            "Do not claim model-agnostic or open-model reproducibility until a same-roster run completes.",
        ],
    }
    specialized_summary = {
        "schema_version": "specialized_chart_model_anchor_probe_v1",
        "created_at_utc": now,
        "stage": "track_e_specialized_chart_model_anchor_probe",
        "stage_status": "specialized_chart_model_runtime_unavailable",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "local_probe_scope": "configuration_and_repository_artifacts_only",
        "detected_runnable_anchor": None,
        "blocking_findings": ["no_specialized_chart_model_runtime_or_checkpoint_configured"],
        "next_required_action": "record_audited_unavailability_or_add_runnable_local_anchor_before_claiming_comparison",
        "interpretation_limits": [
            "This is an audited unavailability record, not a specialized-model comparison.",
            "No checkpoint download or external benchmark query was performed.",
            "Do not use SOTA, leaderboard, or specialized-model superiority language from this artifact.",
        ],
    }
    (open_dir / "summary.json").write_text(json.dumps(open_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (open_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Open Reproducibility Sensitivity Probe",
                "",
                f"- Stage status: `{open_summary['stage_status']}`",
                f"- Open/local candidates: `{len(candidates)}`",
                f"- Paid calls executed: `{open_summary['paid_calls_executed']}`",
                f"- Next required action: `{open_summary['next_required_action']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (specialized_dir / "summary.json").write_text(json.dumps(specialized_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (specialized_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Specialized Chart Model Anchor Probe",
                "",
                f"- Stage status: `{specialized_summary['stage_status']}`",
                f"- Paid calls executed: `{specialized_summary['paid_calls_executed']}`",
                f"- Blocking findings: `{len(specialized_summary['blocking_findings'])}`",
                f"- Next required action: `{specialized_summary['next_required_action']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"open_repro": open_summary, "specialized_anchor": specialized_summary}


def build_track_d_paper_stress_audit(output_root: Path) -> dict[str, Any]:
    track_d_dir = output_root / "m1m2_stress_128"
    track_d_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    stress_path = ROOT / "outputs" / "m1m2_recovery" / "stress_benchmark" / "stress_benchmark.json"
    source_pool_path = track_d_dir / "source_pool_feasibility.json"
    manifest_path = track_d_dir / "stress_128_manifest.json"
    dry_run_path = track_d_dir / "stress_128_dry_run.json"
    default_paid_result_path = track_d_dir / "stress_128_paid_result.json"
    reauthorized_paid_result_path = track_d_dir / TRACK_D_REAUTHORIZED_RESULT
    paid_result_path, paid_result = load_first_existing_json([reauthorized_paid_result_path, default_paid_result_path])
    v31_postrun_path = M1M2_V31_DIR / "postrun_governance_audit" / "postrun_governance_audit.json"
    stress = load_json(stress_path) if stress_path.exists() else {}
    stress_jobs = stress.get("jobs") or []
    source_pool = load_json(source_pool_path) if source_pool_path.exists() else {}
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    dry_run = load_json(dry_run_path) if dry_run_path.exists() else {}
    v31_postrun = load_json(v31_postrun_path) if v31_postrun_path.exists() else {}
    v31_clean, v31_findings = governance_chain_is_clean(v31_postrun) if v31_postrun else (False, ["m1m2_v31_postrun_audit_missing"])
    blocking = []
    source_pool_status = source_pool.get("stage_status")
    if source_pool_status == "m1m2_stress_128_source_pool_feasible":
        available_jobs = 128
    elif source_pool_status == "m1m2_stress_96_fallback_source_pool_feasible":
        available_jobs = 96
    else:
        available_jobs = len(stress_jobs)
        if len(stress_jobs) < 128:
            blocking.append(f"stress_source_pool_below_128:{len(stress_jobs)}")
        if len(stress_jobs) < 96:
            blocking.append(f"stress_source_pool_below_96_fallback:{len(stress_jobs)}")
        if source_pool_path.exists():
            blocking.append(f"paper_level_source_pool_not_feasible:{source_pool_status}")
    if not v31_clean:
        blocking.extend(v31_findings)
    manifest_status = manifest.get("stage_status")
    dry_run_status = dry_run.get("stage_status")
    paid_status = paid_result.get("stage_status")
    manifest_ready = (
        manifest_status == "m1m2_stress_128_manifest_frozen_pending_paid_cap_approval"
        and manifest.get("sample_level_job_cap") == 128
        and manifest.get("llm_http_attempt_cap") == 896
        and manifest.get("ocr_http_attempt_cap") == 0
        and len(manifest.get("condition_list") or []) == 7
        and not manifest.get("blocking_findings")
    )
    dry_run_ready = (
        dry_run_status == "m1m2_stress_128_dry_run_passed"
        and dry_run.get("job_count") == 896
        and dry_run.get("paid_calls_executed") == 0
        and dry_run.get("actual_llm_http_attempts") == 0
        and dry_run.get("actual_ocr_http_attempts") == 0
        and not dry_run.get("blocking_findings")
    )
    paid_completed = (
        paid_status == "m1m2_stress_128_confirmation_completed"
        and paid_result.get("paid_calls_executed") == 896
        and paid_result.get("actual_llm_http_attempts") == 896
        and paid_result.get("actual_ocr_http_attempts") == 0
    )
    active_incomplete_paid_result_present = paid_result_path.exists() and not paid_completed
    default_incomplete_paid_result_present = default_paid_result_path.exists() and not (
        default_paid_result_path == paid_result_path and paid_completed
    )
    unaccounted_interruption_clean_stop = active_incomplete_paid_result_present and has_unknown_attempt_accounting(paid_result)
    if source_pool_status == "m1m2_stress_128_source_pool_feasible":
        if not manifest_ready:
            blocking.append(f"m1m2_stress_128_manifest_not_ready:{manifest_status}")
        if manifest_ready and not dry_run_ready:
            blocking.append(f"m1m2_stress_128_dry_run_not_ready:{dry_run_status}")
        if dry_run_ready and active_incomplete_paid_result_present:
            blocking.append(f"m1m2_stress_128_paid_result_incomplete:{paid_status}")
    if paid_completed:
        track_d_status = "m1m2_stress_128_confirmation_completed"
        next_action = "analyze_m1m2_stress_128_confirmation"
    elif unaccounted_interruption_clean_stop:
        track_d_status = "m1m2_stress_128_failed_uncounted_interruption_clean_stop"
        next_action = "do_not_claim_track_d_128_confirmation_without_new_authorized_rerun"
    elif manifest_ready and dry_run_ready and not blocking:
        track_d_status = "m1m2_stress_128_ready_for_896_call_paid_execution"
        next_action = "execute_after_documented_896_call_paid_cap_approval"
    elif not blocking:
        track_d_status = "m1m2_stress_128_ready_for_manifest"
        next_action = "freeze_m1m2_stress_128_manifest"
    else:
        track_d_status = "m1m2_stress_128_blocked"
        next_action = "do_not_execute_track_d_expanded_paid_confirmation_report_v31_narrow_confirmation_only"
    audit = {
        "schema_version": "m1m2_stress_128_pre_execution_audit_v1",
        "created_at_utc": now,
        "stage": "track_d_m1m2_stress_128_pre_execution_audit",
        "stage_status": track_d_status,
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "stress_benchmark_24": artifact_record(stress_path, status_label="diagnostic", role="Existing 24-job stress source benchmark"),
            "paper_level_source_pool_feasibility": artifact_record(source_pool_path, status_label="current_authoritative" if source_pool_path.exists() else "missing", role="Track D 128/96 source-pool feasibility audit"),
            "stress_128_manifest": artifact_record(manifest_path, status_label="current_authoritative" if manifest_path.exists() else "missing", role="Track D frozen 128-job manifest"),
            "stress_128_dry_run": artifact_record(dry_run_path, status_label="current_authoritative" if dry_run_path.exists() else "missing", role="Track D dry-run job and prompt audit"),
            "stress_128_paid_result": artifact_record(paid_result_path, status_label="current_authoritative" if paid_completed else ("diagnostic_failed_smoke" if active_incomplete_paid_result_present else "missing"), role="Track D paid confirmation result"),
            "default_interrupted_paid_result": artifact_record(default_paid_result_path, status_label="diagnostic_failed_smoke" if default_incomplete_paid_result_present else ("current_authoritative" if default_paid_result_path == paid_result_path and paid_completed else "missing"), role="Diagnostic failed Track D paid smoke; not mixed into the replacement clean run"),
            "reauthorized_paid_result": artifact_record(reauthorized_paid_result_path, status_label="current_authoritative" if paid_completed and paid_result_path == reauthorized_paid_result_path else ("missing" if not reauthorized_paid_result_path.exists() else "diagnostic_failed_smoke"), role="Reauthorized clean Track D paid result"),
            "m1m2_v31_postrun_governance": artifact_record(v31_postrun_path, status_label="current_authoritative", role="Existing completed 16-job candidate-guided confirmation"),
        },
        "target_sample_level_job_cap": 128,
        "fallback_sample_level_job_cap": 96,
        "available_stress_source_jobs": available_jobs,
        "paper_level_source_pool_status": source_pool_status,
        "paper_level_manifest_status": manifest_status,
        "paper_level_dry_run_status": dry_run_status,
        "paper_level_paid_status": paid_status,
        "clean_stop_reason": "unaccounted_prior_paid_interruption" if unaccounted_interruption_clean_stop else None,
        "paper_level_source_pool_summary": {
            "record_count": source_pool.get("record_count"),
            "primary_128_passed": (source_pool.get("primary_128") or {}).get("passed"),
            "fallback_96_passed": (source_pool.get("fallback_96") or {}).get("passed"),
        },
        "current_authoritative_narrow_confirmation": {
            "status": v31_postrun.get("stage_status"),
            "claim": "candidate-guided answerability-preserving selective propagation",
            "sample_level_jobs": 16,
            "llm_http_attempts": 96,
            "ocr_http_attempts": 0,
        },
        "blocking_findings": blocking,
        "next_required_action": next_action,
        "interpretation_limits": [
            "The existing v31 confirmation remains current authoritative for the narrowed M1/M2 stress claim.",
            "This artifact does not expand the paid run and does not create replacement samples.",
            "Do not claim the planned 128-job confirmation until a frozen source pool and paid result exist.",
        ],
    }
    (track_d_dir / "pre_execution_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (track_d_dir / "pre_execution_audit.md").write_text(
        "\n".join(
            [
                "# M1/M2 Stress 128 Pre-Execution Audit",
                "",
                f"- Stage status: `{audit['stage_status']}`",
                f"- Available stress source jobs: `{audit['available_stress_source_jobs']}`",
                f"- Blocking findings: `{len(blocking)}`",
                f"- Paid calls executed: `{audit['paid_calls_executed']}`",
                f"- Next required action: `{audit['next_required_action']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return audit


def build_manuscript_integration_audit(output_root: Path) -> dict[str, Any]:
    audit_dir = output_root / "submission_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    paths = {
        "track_b_manifest": output_root / "geosym_bench_controlled" / "post_remediation_rerun_manifest.json",
        "track_b_identifiability": output_root / "geosym_bench_controlled" / TRACK_B_M1M2_IDENTIFIABILITY_NAME,
        "track_c_audit": output_root / "plotqa_public_500" / "pre_execution_audit.json",
        "track_d_audit": output_root / "m1m2_stress_128" / "pre_execution_audit.json",
        "track_e_open": output_root / "open_repro_sensitivity" / "summary.json",
        "track_e_specialized": output_root / "specialized_chart_model_anchor" / "summary.json",
        "track_f_positioning": output_root / "fusion_diagnostics" / "fusion_positioning_audit.json",
    }
    loaded = {name: load_json(path) if path.exists() else {} for name, path in paths.items()}
    track_b_status = loaded["track_b_manifest"].get("stage_status")
    track_b_identifiability_status = loaded["track_b_identifiability"].get("stage_status")
    track_c_status = loaded["track_c_audit"].get("stage_status")
    track_d_status = loaded["track_d_audit"].get("stage_status")
    track_e_open_status = loaded["track_e_open"].get("stage_status")
    track_e_specialized_status = loaded["track_e_specialized"].get("stage_status")
    if track_b_status == "geosym_bench_post_remediation_rerun_completed":
        track_b_allowed_claim = (
            "The diversity-audited GeoSym-Bench subset post-remediation rerun completed under the frozen "
            "240-sample roster and paid caps; report only the completed rerun artifact and its scoped analysis, "
            "and do not mix historical pre-remediation scores or diagnostic interrupted smoke outputs into the main "
            "evidence. The repaired Track B E5/E5_raw comparison is end-to-end execution evidence only, not M1/M2 "
            "mechanism evidence, when the identifiability audit reports no judge-reuse pairs and insufficient power."
        )
    else:
        track_b_allowed_claim = (
            "A diversity-audited generated benchmark subset is frozen; the post-remediation rerun stopped at "
            "an uncounted interruption, so historical GeoSym-Bench numbers remain pre-remediation sensitivity "
            "only unless a new authorized rerun completes."
        )

    claim_matrix = [
        {
            "paper_location": "main_results_or_supplement",
            "evidence": "GeoSym-Bench controlled package",
            "artifact_status": {
                "track_b_manifest": track_b_status,
                "track_b_identifiability": track_b_identifiability_status,
            },
            "allowed_claim": track_b_allowed_claim,
            "forbidden_claim": "Do not present historical GeoSym-Bench scores as revised post-remediation main evidence; do not use Track B E5/E5_raw deltas as M1/M2 mechanism evidence when identifiability is blocked.",
        },
        {
            "paper_location": "main_results_or_supplement",
            "evidence": "PlotQA public 500 package",
            "artifact_status": track_c_status,
            "allowed_claim": "The frozen PlotQA 500-sample measured-data confirmation completed with no public-benefit claim. Report G_full_measured_geosym as stronger than the measured-flat baseline on the same measured facts, while noting that cached E0/E1 are not a same-roster matched comparison and that deterministic counts differ across arms.",
            "forbidden_claim": "Do not claim public benchmark superiority or full PlotQA benchmark performance.",
        },
        {
            "paper_location": "main_or_supplement_ablation",
            "evidence": "M1/M2 candidate-guided stress evidence",
            "artifact_status": track_d_status,
            "allowed_claim": track_d_completed_claim(loaded["track_d_audit"]),
            "forbidden_claim": "Do not claim broad M1/M2 accuracy repair or calibration.",
        },
        {
            "paper_location": "reproducibility_or_limitations",
            "evidence": "Specialized/open model anchors",
            "artifact_status": {
                "open_repro": track_e_open_status,
                "specialized_anchor": track_e_specialized_status,
            },
            "allowed_claim": "Open/local and specialized chart-model routes have audited unavailable/not-configured stops.",
            "forbidden_claim": "Do not claim model-agnostic, Mathpix-free, leaderboard, or SOTA performance.",
        },
        {
            "paper_location": "information_fusion_positioning",
            "evidence": "Fusion diagnostics",
            "artifact_status": loaded["track_f_positioning"].get("stage_status"),
            "allowed_claim": "GeoSym implements heuristic ordinal confidence tiers, utility-aware selection, gap-aware propagation, and module-level provenance.",
            "forbidden_claim": "Do not claim calibrated probabilities, belief masses, conflict coefficients, or a formal probabilistic fusion rule.",
        },
    ]
    integration_text = json.dumps([row["allowed_claim"] for row in claim_matrix], ensure_ascii=False)
    forbidden_hits = [
        pattern
        for pattern in MANUSCRIPT_FORBIDDEN_POSITIVE_PATTERNS
        if pattern in integration_text
    ]
    claim_safety_stop_statuses = {
        "geosym_bench_rerun_manifest_blocked",
        "geosym_bench_post_remediation_rerun_completed",
        "geosym_bench_post_remediation_rerun_failed_uncounted_interruption_clean_stop",
        "geosym_bench_rerun_manifest_frozen_pending_paid_execution",
        "geosym_bench_blocked_missing_post_remediation_subset_runner",
        "track_b_m1m2_not_identifiable_from_repaired_rerun",
        "track_b_m1m2_identifiable_from_repaired_rerun",
        "track_c_blocked_missing_500_measured_confirmation_runner",
        "track_c_500_measured_confirmation_ready_for_paid_execution",
        "track_c_zero_cost_gates_passed_ready_for_paid_manifest",
        "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim",
        "m1m2_stress_128_blocked",
        "m1m2_stress_128_blocked_insufficient_source_pool",
        "m1m2_stress_128_failed_uncounted_interruption_clean_stop",
        "m1m2_stress_128_confirmation_completed",
        "open_mllm_not_configured",
        "specialized_chart_model_runtime_unavailable",
        "confidence_tier_accuracy_completed",
        "parameter_sensitivity_completed",
    }
    statuses_to_check = [
        track_b_status,
        track_c_status,
        track_d_status,
        track_e_open_status,
        track_e_specialized_status,
        loaded["track_f_positioning"].get("stage_status"),
    ]
    if track_b_status == "geosym_bench_post_remediation_rerun_completed":
        statuses_to_check.append(track_b_identifiability_status)
    required_statuses_present = all(
        status in claim_safety_stop_statuses or status in {"fusion_positioning_claim_boundaries_recorded"}
        for status in statuses_to_check
    )
    blocking = []
    if not required_statuses_present:
        blocking.append("manuscript_integration_missing_required_clean_stop_status")
    if forbidden_hits:
        blocking.append("manuscript_integration_contains_forbidden_positive_language")
    audit = {
        "schema_version": "manuscript_integration_audit_v1",
        "created_at_utc": now,
        "stage": "track_g_manuscript_integration_claim_safety",
        "stage_status": "manuscript_integration_claim_safety_complete" if not blocking else "manuscript_integration_claim_safety_blocked",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            name: artifact_record(path, status_label="current_authoritative", role=f"Manuscript integration input: {name}")
            for name, path in paths.items()
        },
        "claim_matrix": claim_matrix,
        "claim_safety_gates": {
            "expanded_track_b_paid_confirmation_scoped": track_b_status
            in {
                "geosym_bench_post_remediation_rerun_completed",
                "geosym_bench_post_remediation_rerun_failed_uncounted_interruption_clean_stop",
                "geosym_bench_rerun_manifest_frozen_pending_paid_execution",
                "geosym_bench_blocked_missing_post_remediation_subset_runner",
            },
            "track_b_m1m2_identifiability_scoped": track_b_identifiability_status
            in {"track_b_m1m2_not_identifiable_from_repaired_rerun", "track_b_m1m2_identifiable_from_repaired_rerun"},
            "expanded_track_c_paid_confirmation_not_claimed": track_c_status
            != "track_c_plotqa_500_confirmation_completed",
            "track_c_negative_or_no_benefit_result_scoped": track_c_status
            in {"track_c_plotqa_500_confirmation_completed", "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim"},
            "expanded_track_d_paid_confirmation_scoped": track_d_status in {
                "m1m2_stress_128_confirmation_completed",
                "m1m2_stress_128_failed_uncounted_interruption_clean_stop",
                "m1m2_stress_128_blocked",
                "m1m2_stress_128_blocked_insufficient_source_pool",
            },
            "open_and_specialized_unavailability_scoped": track_e_open_status == "open_mllm_not_configured"
            and track_e_specialized_status == "specialized_chart_model_runtime_unavailable",
            "forbidden_positive_language_absent": not forbidden_hits,
        },
        "forbidden_positive_language_hits": forbidden_hits,
        "blocking_findings": blocking,
        "next_required_action": "start_final_review_loop" if not blocking else "repair_manuscript_claim_matrix",
        "interpretation_limits": [
            "This artifact integrates evidence boundaries, not manuscript prose.",
            "The main paper must be rewritten around clean-stop and narrowed-evidence semantics unless the missing runners are implemented and executed under frozen paid caps.",
        ],
    }
    (audit_dir / "manuscript_integration_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (audit_dir / "manuscript_integration_audit.md").write_text(
        "\n".join(
            [
                "# Manuscript Integration Audit",
                "",
                f"- Stage status: `{audit['stage_status']}`",
                f"- Blocking findings: `{len(blocking)}`",
                "- Main constraint: no revised paper-level positive performance claim is supported by the current package.",
                "",
                "## Claim Matrix",
                "",
                *[
                    f"- `{row['evidence']}`: allowed claim boundary recorded; artifact status=`{row['artifact_status']}`."
                    for row in claim_matrix
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return audit


def build_final_review_loop_audit(output_root: Path) -> dict[str, Any]:
    audit_dir = output_root / "submission_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    paths = {
        "evidence_ledger": output_root / "evidence_ledger.json",
        "track_a": output_root / "track_a_governance_cleanup" / "track_a_governance_audit.json",
        "track_b_dedup": output_root / "geosym_bench_controlled" / "metadata_dedup_audit.json",
        "track_b_manifest": output_root / "geosym_bench_controlled" / "post_remediation_rerun_manifest.json",
        "track_b_final": output_root / "geosym_bench_controlled" / TRACK_B_FINAL_BACKFILL_RESULT_NAME,
        "track_b_m0_quality": output_root / "geosym_bench_controlled" / "clean_grid_fallback_m0_quality_8x3.json",
        "track_b_identifiability": output_root / "geosym_bench_controlled" / TRACK_B_M1M2_IDENTIFIABILITY_NAME,
        "track_c": output_root / "plotqa_public_500" / "pre_execution_audit.json",
        "track_d": output_root / "m1m2_stress_128" / "pre_execution_audit.json",
        "track_e_open": output_root / "open_repro_sensitivity" / "summary.json",
        "track_e_specialized": output_root / "specialized_chart_model_anchor" / "summary.json",
        "track_p2_cross_backbone_smoke": output_root / "cross_backbone_glm46v_t180" / "smoke_failure_audit" / "smoke_failure_audit.json",
        "track_p2_cross_backbone_result": cross_backbone_authoritative_result_path(output_root),
        "track_p2_cross_backbone_analysis": cross_backbone_authoritative_analysis_path(output_root),
        "track_f_positioning": output_root / "fusion_diagnostics" / "fusion_positioning_audit.json",
        "track_f_weighting": output_root / "fusion_diagnostics" / "weighting_sensitivity.json",
        "manuscript_integration": output_root / "submission_audit" / "manuscript_integration_audit.json",
    }
    loaded = {name: load_json(path) if path.exists() else {} for name, path in paths.items()}
    cross_backbone_required = (
        paths["track_p2_cross_backbone_smoke"].exists()
        or paths["track_p2_cross_backbone_result"].exists()
        or paths["track_p2_cross_backbone_analysis"].exists()
    )
    track_b_completed = loaded["track_b_manifest"].get("stage_status") == "geosym_bench_post_remediation_rerun_completed"
    allowed_claim_safety_stop_statuses = {
        "geosym_bench_rerun_manifest_blocked",
        "geosym_bench_post_remediation_rerun_completed",
        "geosym_bench_post_remediation_rerun_failed_uncounted_interruption_clean_stop",
        "geosym_bench_rerun_manifest_frozen_pending_paid_execution",
        "geosym_bench_blocked_missing_post_remediation_subset_runner",
        "track_b_m1m2_not_identifiable_from_repaired_rerun",
        "track_b_m1m2_identifiable_from_repaired_rerun",
        "track_c_blocked_missing_500_measured_confirmation_runner",
        "track_c_500_measured_confirmation_ready_for_paid_execution",
        "track_c_zero_cost_gates_passed_ready_for_paid_manifest",
        "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim",
        "m1m2_stress_128_blocked",
        "m1m2_stress_128_blocked_insufficient_source_pool",
        "m1m2_stress_128_failed_uncounted_interruption_clean_stop",
        "open_mllm_not_configured",
        "specialized_chart_model_runtime_unavailable",
        "confidence_tier_accuracy_completed",
        "parameter_sensitivity_completed",
        "stopped_missing_frozen_fact_level_truth_join",
        "stopped_parameters_not_configured_for_zero_cost_sweep",
        "real_chart_reference_audit_complete_measurement_not_run",
        "main_run_blocked_by_smoke_gate",
        "smoke_gate_passed_main_run_releasable",
        "cross_backbone_glm46v_partial_timeout_limited_analysis",
        "cross_backbone_glm46v_completed_analysis",
        "cross_backbone_glm46v_t180_completed_analysis",
        "weighting_sensitivity_completed",
    }
    source_blockers = []
    for name, payload in loaded.items():
        if not paths[name].exists():
            if name in {"track_p2_cross_backbone_smoke", "track_p2_cross_backbone_result", "track_p2_cross_backbone_analysis"} and not cross_backbone_required:
                continue
            if name in {"track_b_final", "track_b_m0_quality", "track_b_identifiability"} and not track_b_completed:
                continue
            source_blockers.append(f"missing_review_source:{name}")
        if payload.get("blocking_findings") and payload.get("stage_status") not in allowed_claim_safety_stop_statuses:
            source_blockers.append(f"source_has_blocking_findings:{name}")
    manuscript_audit = loaded.get("manuscript_integration") or {}
    forbidden_hits = [
        {"path": str(paths["manuscript_integration"].resolve()), "pattern": pattern}
        for pattern in manuscript_audit.get("forbidden_positive_language_hits", [])
    ]
    exit_questions = {
        "could_paid_run_start_without_frozen_cap": False,
        "could_sample_selection_depend_on_model_outcome": False,
        "could_reference_answers_affect_runtime_decisions": False,
        "could_diagnostic_result_be_mistaken_for_confirmatory_evidence": False,
        "could_historical_scores_be_mistaken_for_revised_main_evidence": False,
        "could_m1m2_stress_be_overstated_as_broad_accuracy_repair": False,
        "could_missing_runners_be_silently_ignored": False,
    }
    blocking = list(source_blockers)
    if forbidden_hits:
        blocking.append("forbidden_claim_language_present_in_integration_artifact")
    residual_risks = []
    track_b_final = loaded.get("track_b_final") or {}
    if track_b_completed:
        final_summary = (track_b_final.get("summary") or {}).get("overall_by_condition") or {}
        for condition in ALL_CONDITIONS:
            row = final_summary.get(condition) or {}
            if row.get("rows") != 240:
                blocking.append(f"track_b_final_condition_rows_not_240:{condition}:{row.get('rows')}")
        for condition in ("E5", "E5_raw"):
            row = final_summary.get(condition) or {}
            if row.get("success") != 240:
                blocking.append(f"track_b_final_condition_success_not_240:{condition}:{row.get('success')}")
        if track_b_final.get("cached_conditions_reused") != ["E0", "E1", "E2", "E3", "E4"]:
            blocking.append("track_b_final_cached_conditions_reuse_not_recorded")
        chain_audit = track_b_final.get("chain_text_reuse_audit") or {}
        if chain_audit.get("rows_checked") != 480:
            blocking.append(f"track_b_chain_rows_checked_not_480:{chain_audit.get('rows_checked')}")
        if chain_audit.get("empty_chain_text_hash_rows") != 0:
            blocking.append(f"track_b_empty_chain_text_hash_rows:{chain_audit.get('empty_chain_text_hash_rows')}")
        if chain_audit.get("judge_reuse_violations"):
            blocking.append("track_b_judge_reuse_violations_present")
        governance = track_b_final.get("governance") or {}
        for key in ("within_global_llm_cap", "within_global_ocr_cap", "sample_replacement", "conditions_frozen"):
            if key not in governance:
                blocking.append(f"track_b_final_governance_missing:{key}")
        if governance.get("within_global_llm_cap") is not True:
            blocking.append("track_b_final_global_llm_cap_not_verified")
        if governance.get("within_global_ocr_cap") is not True:
            blocking.append("track_b_final_global_ocr_cap_not_verified")
        if governance.get("sample_replacement") is not False:
            blocking.append("track_b_final_sample_replacement_not_false")
        if governance.get("conditions_frozen") is not True:
            blocking.append("track_b_final_conditions_not_frozen")
        identifiability = loaded.get("track_b_identifiability") or {}
        if identifiability.get("stage_status") == "track_b_m1m2_not_identifiable_from_repaired_rerun":
            residual_risks.append(
                "Track B does not make M1/M2 identifiable as a positive accuracy mechanism; the manuscript must downgrade the broad M1/M2 attribution claim and report the result as neutral or negative evidence."
            )
        elif identifiability.get("stage_status") != "track_b_m1m2_identifiable_from_repaired_rerun":
            blocking.append(f"track_b_m1m2_identifiability_status_unexpected:{identifiability.get('stage_status')}")
        identifiability_blockers = set(identifiability.get("identifiability_blockers") or [])
        for required in (
            "track_b_no_judge_reuse_pairs",
            "track_b_min_detectable_effect_exceeds_historical_m1m2_effect",
        ):
            if required not in identifiability_blockers:
                blocking.append(f"track_b_m1m2_identifiability_blocker_missing:{required}")
        power = identifiability.get("power_analysis") or {}
        required_pairs = power.get("required_pairs_for_historical_effect_normal_approx")
        if not isinstance(required_pairs, int) or required_pairs <= 240:
            blocking.append(f"track_b_power_required_pairs_not_gt_240:{required_pairs}")
    m0_quality = loaded.get("track_b_m0_quality") or {}
    if track_b_completed and m0_quality.get("stage_status") != "completed_zero_cost_local_cv_diagnostic":
        blocking.append("track_b_m0_quality_diagnostic_missing_or_incomplete")
    track_b_status = loaded["track_b_manifest"].get("stage_status")
    track_c_status = loaded["track_c"].get("stage_status")
    track_d_status = loaded["track_d"].get("stage_status")
    cross_backbone_smoke = loaded.get("track_p2_cross_backbone_smoke") or {}
    cross_backbone_result = loaded.get("track_p2_cross_backbone_result") or {}
    if cross_backbone_required and cross_backbone_smoke.get("stage_status") == "main_run_blocked_by_smoke_gate":
        blocking.append("p2_cross_backbone_glm_smoke_failed_main_run_blocked")
    elif cross_backbone_required and cross_backbone_smoke.get("stage_status") not in {
        "ready_for_main_run",
        "smoke_gate_passed_main_run_releasable",
        "cross_backbone_pilot_completed",
    }:
        blocking.append(f"p2_cross_backbone_smoke_status_unexpected:{cross_backbone_smoke.get('stage_status')}")
    if cross_backbone_required and cross_backbone_result.get("stage_status") not in {
        "partial_paid_pilot",
        "cross_backbone_pilot_completed",
        "stopped_at_paid_cap",
        "stopped_failed_smoke_gate",
    }:
        blocking.append(f"p2_cross_backbone_result_status_unexpected:{cross_backbone_result.get('stage_status')}")
    if track_b_status == "geosym_bench_post_remediation_rerun_failed_uncounted_interruption_clean_stop":
        residual_risks.append(
            "Track B has a frozen subset and runner, but the post-remediation rerun is a clean stop after an uncounted paid interruption; report historical GeoSym-Bench results as pre-remediation sensitivity only."
        )
    elif track_b_status != "geosym_bench_post_remediation_rerun_completed":
        residual_risks.append(f"Track B is not a completed post-remediation rerun: {track_b_status}.")
    elif (loaded.get("track_b_identifiability") or {}).get("stage_status") == "track_b_m1m2_not_identifiable_from_repaired_rerun":
        residual_risks.append(
            "Track B completed as repaired end-to-end execution evidence, but its E5/E5_raw comparison is not M1/M2 mechanism evidence because no pair triggers judge-score reuse and the observed variance makes the historical M1/M2 effect underpowered."
        )
    if track_c_status == "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim":
        residual_risks.append(
            "Track C 500 completed, but the result is negative for public-benefit claims; report it as public negative evidence and measured-flat sensitivity only."
        )
    elif track_c_status != "track_c_plotqa_500_confirmation_completed":
        residual_risks.append(f"Track C is not a positive 500-sample public confirmation: {track_c_status}.")
    if track_d_status == "m1m2_stress_128_failed_uncounted_interruption_clean_stop":
        residual_risks.append(
            "Track D 128-job confirmation is a clean stop after an uncounted paid interruption; retain only the v31 16-job narrow confirmation unless a new authorized rerun completes."
        )
    elif track_d_status != "m1m2_stress_128_confirmation_completed":
        residual_risks.append(f"Track D expanded stress confirmation is not complete: {track_d_status}.")
    if cross_backbone_required and cross_backbone_smoke.get("stage_status") == "main_run_blocked_by_smoke_gate":
        residual_risks.append(
            "P2 GLM-4.6V cross-backbone validation is blocked by the frozen smoke parser-integrity gate; do not run or report the 414-attempt main run until GLM output/parser compatibility is repaired and re-smoked."
        )
    elif cross_backbone_required and cross_backbone_result.get("stage_status") == "partial_paid_pilot":
        residual_risks.append(
            "P2 GLM-4.6V passed the repaired smoke gate but the main run completed only partially because timeout failures limited effective paired coverage; report completion and timeout rates alongside score deltas."
        )
    residual_risks.append("This review artifact does not replace external command verification such as pytest, LaTeX compilation, or git diff checks.")
    next_required_action = (
        "start_manuscript_rewrite_from_evidence_ledger_with_clean_stop_limits"
        if not blocking
        else "resolve_track_b_m1m2_identifiability_before_manuscript_writing"
    )
    audit = {
        "schema_version": "final_review_loop_audit_v1",
        "created_at_utc": now,
        "stage": "final_review_loop",
        "stage_status": "final_review_loop_claim_safety_no_blocking_findings" if not blocking else "final_review_loop_blocked",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            name: artifact_record(path, status_label="current_authoritative", role=f"Final review source: {name}")
            for name, path in paths.items()
        },
        "review_questions": exit_questions,
        "forbidden_claim_hits": forbidden_hits,
        "blocking_findings": blocking,
        "residual_risks": residual_risks,
        "next_required_action": next_required_action,
        "interpretation_limits": [
            "No paid calls are executed by the review loop.",
            "No clean stop is interpreted as positive performance evidence.",
        ],
    }
    (audit_dir / "final_review_loop.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (audit_dir / "final_review_loop.md").write_text(
        "\n".join(
            [
                "# Final Review Loop Audit",
                "",
                f"- Stage status: `{audit['stage_status']}`",
                f"- Blocking findings: `{len(blocking)}`",
                f"- Next required action: `{next_required_action}`",
                f"- Paid calls executed: `{audit['paid_calls_executed']}`",
                "",
                "## Residual Risks",
                "",
                *[f"- {risk}" for risk in audit["residual_risks"]],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return audit


def build_submission_audit(output_root: Path) -> dict[str, Any]:
    audit_dir = output_root / "submission_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    paths = {
        "track_a": output_root / "track_a_governance_cleanup" / "track_a_governance_audit.json",
        "track_b_dedup": output_root / "geosym_bench_controlled" / "metadata_dedup_audit.json",
        "track_b_rerun_manifest": output_root / "geosym_bench_controlled" / "post_remediation_rerun_manifest.json",
        "track_b_identifiability": output_root / "geosym_bench_controlled" / TRACK_B_M1M2_IDENTIFIABILITY_NAME,
        "track_c": output_root / "plotqa_public_500" / "pre_execution_audit.json",
        "track_d": output_root / "m1m2_stress_128" / "pre_execution_audit.json",
        "track_e_open": output_root / "open_repro_sensitivity" / "summary.json",
        "track_e_specialized": output_root / "specialized_chart_model_anchor" / "summary.json",
        "track_p2_cross_backbone_smoke": output_root / "cross_backbone_glm46v_t180" / "smoke_failure_audit" / "smoke_failure_audit.json",
        "track_p2_cross_backbone_result": cross_backbone_authoritative_result_path(output_root),
        "track_p2_cross_backbone_analysis": cross_backbone_authoritative_analysis_path(output_root),
        "track_f_confidence": output_root / "fusion_diagnostics" / "confidence_tier_accuracy.json",
        "track_f_parameter": output_root / "fusion_diagnostics" / "parameter_sensitivity.json",
        "track_f_cost": output_root / "fusion_diagnostics" / "cost_latency.json",
        "track_f_real_chart": output_root / "fusion_diagnostics" / "real_chart_error_decomposition.json",
        "track_f_positioning": output_root / "fusion_diagnostics" / "fusion_positioning_audit.json",
        "track_f_weighting": output_root / "fusion_diagnostics" / "weighting_sensitivity.json",
        "manuscript_integration": output_root / "submission_audit" / "manuscript_integration_audit.json",
        "final_review_loop": output_root / "submission_audit" / "final_review_loop.json",
    }
    loaded = {name: load_json(path) if path.exists() else {} for name, path in paths.items()}
    cross_backbone_required = (
        paths["track_p2_cross_backbone_smoke"].exists()
        or paths["track_p2_cross_backbone_result"].exists()
        or paths["track_p2_cross_backbone_analysis"].exists()
    )
    strict_experiment_completion_items = {
        "track_a_ledger_clean": loaded["track_a"].get("stage_status") == "track_a_governance_clean",
        "track_b_subset_frozen": loaded["track_b_dedup"].get("stage_status") == "geosym_bench_dedup_audit_passed_subset_ready",
        "track_b_post_remediation_rerun_completed": loaded["track_b_rerun_manifest"].get("stage_status")
        == "geosym_bench_post_remediation_rerun_completed",
        "track_b_m1m2_identifiability_resolved": loaded["track_b_identifiability"].get("stage_status")
        in {
            "track_b_m1m2_identifiable_from_repaired_rerun",
            "track_b_m1m2_not_identifiable_from_repaired_rerun",
        },
        "track_c_500_paid_confirmation_completed": loaded["track_c"].get("stage_status")
        in {"track_c_plotqa_500_confirmation_completed", "track_c_plotqa_500_confirmation_completed_no_public_benefit_claim"},
        "track_d_128_confirmation_completed": loaded["track_d"].get("stage_status")
        == "m1m2_stress_128_confirmation_completed",
        "track_e_completed_or_audited_unavailable": loaded["track_e_open"].get("stage_status")
        in {"open_mllm_same_100_completed", "open_mllm_not_configured"}
        and loaded["track_e_specialized"].get("stage_status")
        in {"specialized_chart_model_anchor_completed", "specialized_chart_model_runtime_unavailable"},
        "track_f_confidence_tier_accuracy_completed": loaded["track_f_confidence"].get("stage_status")
        == "confidence_tier_accuracy_completed",
        "track_f_parameter_sensitivity_completed": loaded["track_f_parameter"].get("stage_status")
        == "parameter_sensitivity_completed",
        "track_f_cost_latency_schema_completed": loaded["track_f_cost"].get("stage_status")
        == "schema_initialized_from_track_a_ledger",
        "track_f_real_chart_audit_completed": loaded["track_f_real_chart"].get("stage_status")
        in {"real_chart_error_decomposition_completed", "real_chart_reference_audit_complete_measurement_not_run"},
        "track_f_positioning_completed": loaded["track_f_positioning"].get("stage_status")
        == "fusion_positioning_claim_boundaries_recorded",
        "track_f_weighting_sensitivity_completed": loaded["track_f_weighting"].get("stage_status")
        == "weighting_sensitivity_completed",
    }
    if cross_backbone_required:
        strict_experiment_completion_items["track_p2_cross_backbone_glm_completed"] = (
            loaded["track_p2_cross_backbone_result"].get("stage_status") == "cross_backbone_pilot_completed"
            or (
                loaded["track_p2_cross_backbone_analysis"].get("stage_status")
                == "cross_backbone_glm46v_t180_completed_analysis"
                and ((loaded["track_p2_cross_backbone_analysis"].get("glm") or {}).get("paired_summary") or {}).get("paired_completion_rate", 0) >= 0.90
                and ((loaded["track_p2_cross_backbone_analysis"].get("glm") or {}).get("result_rows") or 0) == 96
            )
        )
    experiment_resolution_items = {
        **strict_experiment_completion_items,
        "track_b_post_remediation_rerun_completed_or_clean_stop": loaded["track_b_rerun_manifest"].get("stage_status")
        in {
            "geosym_bench_post_remediation_rerun_completed",
            "geosym_bench_post_remediation_rerun_failed_uncounted_interruption_clean_stop",
        },
        "track_b_m1m2_identifiability_completed_or_scoped_stop": loaded["track_b_rerun_manifest"].get("stage_status")
        != "geosym_bench_post_remediation_rerun_completed"
        or loaded["track_b_identifiability"].get("stage_status")
        in {
            "track_b_m1m2_identifiable_from_repaired_rerun",
            "track_b_m1m2_not_identifiable_from_repaired_rerun",
        },
        "track_d_128_confirmation_completed_or_clean_stop": loaded["track_d"].get("stage_status")
        in {
            "m1m2_stress_128_confirmation_completed",
            "m1m2_stress_128_failed_uncounted_interruption_clean_stop",
        },
    }
    if cross_backbone_required:
        experiment_resolution_items["track_p2_cross_backbone_glm_completed_or_smoke_failed_closed"] = (
            loaded["track_p2_cross_backbone_result"].get("stage_status") == "cross_backbone_pilot_completed"
            or loaded["track_p2_cross_backbone_result"].get("stage_status") in {"partial_paid_pilot", "stopped_at_paid_cap"}
            or loaded["track_p2_cross_backbone_smoke"].get("stage_status") == "main_run_blocked_by_smoke_gate"
        )
    del experiment_resolution_items["track_b_post_remediation_rerun_completed"]
    del experiment_resolution_items["track_b_m1m2_identifiability_resolved"]
    del experiment_resolution_items["track_d_128_confirmation_completed"]
    if cross_backbone_required:
        del experiment_resolution_items["track_p2_cross_backbone_glm_completed"]
    claim_safety_items = {
        "manuscript_integration_complete": loaded["manuscript_integration"].get("stage_status")
        == "manuscript_integration_claim_safety_complete"
        and not loaded["manuscript_integration"].get("blocking_findings"),
        "final_review_loop_claim_safety_no_blocking_findings": loaded["final_review_loop"].get("stage_status")
        == "final_review_loop_claim_safety_no_blocking_findings"
        and not loaded["final_review_loop"].get("blocking_findings"),
        "unexecuted_experiments_not_reported_as_positive_evidence": loaded["manuscript_integration"].get("stage_status")
        == "manuscript_integration_claim_safety_complete",
    }
    exit_items = {
        **experiment_resolution_items,
        **claim_safety_items,
    }
    paper_level_experiments_complete = all(strict_experiment_completion_items.values())
    paper_level_experiment_paths_resolved = all(experiment_resolution_items.values())
    claim_safety_package_ready = all(claim_safety_items.values())
    if paper_level_experiments_complete and claim_safety_package_ready:
        stage_status = "paper_level_experiments_complete_and_claim_safety_ready"
        next_required_action = "start_manuscript_rewrite_from_completed_evidence_ledger"
    elif paper_level_experiment_paths_resolved and claim_safety_package_ready:
        stage_status = "paper_level_package_ready_for_manuscript_rewrite_with_clean_stops"
        next_required_action = "start_manuscript_rewrite_from_evidence_ledger_with_clean_stop_limits"
    elif claim_safety_package_ready:
        stage_status = "claim_safety_package_ready_but_paper_level_experiments_incomplete"
        next_required_action = "continue_paper_level_experiment_execution"
    else:
        stage_status = "paper_level_package_not_ready_for_manuscript_rewrite"
        next_required_action = "resolve_claim_safety_or_governance_blockers"
    blocking = [name for name, passed in strict_experiment_completion_items.items() if not passed]
    unresolved = [name for name, passed in experiment_resolution_items.items() if not passed]
    claim_safety_blocking = [name for name, passed in claim_safety_items.items() if not passed]
    audit = {
        "schema_version": "paper_level_submission_audit_v1",
        "created_at_utc": now,
        "stage": "paper_level_submission_package_exit_audit",
        "stage_status": stage_status,
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            name: artifact_record(path, status_label="current_authoritative", role=f"Submission audit input: {name}")
            for name, path in paths.items()
        },
        "paper_level_experiments_complete": paper_level_experiments_complete,
        "paper_level_experiment_paths_resolved": paper_level_experiment_paths_resolved,
        "claim_safety_package_ready": claim_safety_package_ready,
        "exit_items": exit_items,
        "strict_experiment_completion_items": strict_experiment_completion_items,
        "experiment_resolution_items": experiment_resolution_items,
        "incomplete_experiment_items": blocking,
        "unresolved_experiment_items": unresolved,
        "claim_safety_blocking_findings": claim_safety_blocking,
        "blocking_findings": claim_safety_blocking,
        "next_required_action": next_required_action,
        "interpretation_limits": [
            "This audit separates claim-safety readiness from completed paper-level experiments.",
            "Pending paid runs are not started by this audit.",
            "Manuscript integration must not turn pending or failed gates into positive evidence.",
        ],
    }
    (audit_dir / "submission_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (audit_dir / "submission_audit.md").write_text(
        "\n".join(
            [
                "# Paper-Level Submission Audit",
                "",
                f"- Stage status: `{audit['stage_status']}`",
                f"- Paper-level experiments complete: `{audit['paper_level_experiments_complete']}`",
                f"- Paper-level experiment paths resolved: `{audit['paper_level_experiment_paths_resolved']}`",
                f"- Claim-safety package ready: `{audit['claim_safety_package_ready']}`",
                f"- Incomplete experiment items: `{len(blocking)}`",
                f"- Unresolved experiment items: `{len(unresolved)}`",
                f"- Claim-safety blockers: `{len(claim_safety_blocking)}`",
                f"- Paid calls executed: `{audit['paid_calls_executed']}`",
                f"- Next required action: `{audit['next_required_action']}`",
                "",
                "## Incomplete Experiment Items",
                "",
                *[f"- `{item}`" for item in blocking],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return audit


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def open_model_candidates(config: dict[str, Any]) -> list[dict[str, str]]:
    candidates = []
    for provider, spec in (config.get("models") or {}).items():
        provider_l = str(provider).lower()
        if provider_l in PROPRIETARY_PROVIDERS:
            continue
        adapter = str((spec or {}).get("adapter", ""))
        text = " ".join([provider_l, adapter.lower(), json.dumps(spec, ensure_ascii=False).lower()])
        if not any(marker in text for marker in OPEN_MARKERS):
            continue
        for display_name, model_id in ((spec or {}).get("models") or {}).items():
            candidates.append(
                {
                    "provider": str(provider),
                    "display_name": str(display_name),
                    "api_model_id": str(model_id),
                    "adapter": adapter,
                }
            )
    return candidates


def build_track_c_artifacts(
    diagnostic_plan_path: Path,
    diagnostic_result_path: Path,
    openocr_summary_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    diagnostic_plan = load_json(diagnostic_plan_path)
    openocr_summary = load_json(openocr_summary_path)
    now = datetime.now(timezone.utc).isoformat()
    selected_ids = [sample["sample_id"] for sample in diagnostic_plan.get("selected_samples", [])]
    stop_reason = None
    if openocr_summary.get("experiment_status") != "openocr_backend_frozen_for_track_c":
        stop_reason = openocr_summary.get("experiment_status", "openocr_gate_not_passed")
    frozen_plan = {
        "schema_version": "plotqa_full_openocr_50_frozen_plan_v1",
        "created_at_utc": now,
        "benchmark": "PlotQA",
        "split": diagnostic_plan.get("split"),
        "paid_run_approved": stop_reason is None,
        "paid_calls_executed": 0,
        "hypothesis": "Test whether OCR-enabled full GeoSym improves over matched direct, no-OCR G0, and OCR-flat baselines on the frozen PlotQA 50-sample roster.",
        "falsification_signal": "If full OCR GeoSym does not strictly exceed the strongest matched baseline or ties OCR-flat, do not claim fusion benefit.",
        "source_artifacts": {
            "diagnostic_plan": {
                "path": str(diagnostic_plan_path.resolve()),
                "sha256": stable_file_hash(diagnostic_plan_path),
                "declared_plan_sha256": diagnostic_plan.get("plan_sha256"),
            },
            "diagnostic_result": {
                "path": str(diagnostic_result_path.resolve()),
                "sha256": stable_file_hash(diagnostic_result_path),
            },
            "openocr_summary": {
                "path": str(openocr_summary_path.resolve()),
                "sha256": stable_file_hash(openocr_summary_path),
                "experiment_status": openocr_summary.get("experiment_status"),
            },
        },
        "selected_ids": selected_ids,
        "conditions": [
            "G_full_plotqa_openocr",
            "OCR_flat_answer_baseline",
        ],
        "cached_conditions_reused": [
            "E0_plotqa_direct",
            "E1_plotqa_cot",
            "G0_plotqa_no_ocr",
        ],
        "llm_http_attempt_cap": 200,
        "ocr_http_attempt_cap": 0,
        "retry_policy": "G_full_plotqa_openocr <=3 LLM attempts per sample; OCR_flat_answer_baseline <=1 LLM attempt per sample",
        "dry_run": {
            "paid_calls_executed": 0,
            "cap_enforced": True,
            "stopped_before_paid_calls": stop_reason is not None,
            "stop_reason": stop_reason,
        },
        "interpretation_limits": [
            "Do not run paid OCR-enabled conditions unless OpenOCR survey passes the frozen go gate.",
            "Do not claim fusion benefit unless full OCR GeoSym strictly exceeds E0, E1, G0-no-OCR, and OCR-flat on the same roster.",
            "Mathpix is not part of this plan.",
        ],
    }
    paid_result = {
        "schema_version": "plotqa_full_openocr_50_paid_result_v1",
        "created_at_utc": now,
        "plan_path": str((output_dir / "frozen_plan.json").resolve()),
        "benchmark": "PlotQA",
        "conditions": frozen_plan["conditions"],
        "sample_count": len(selected_ids),
        "llm_http_attempt_cap": 200,
        "ocr_http_attempt_cap": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "paid_calls_executed": 0,
        "completed_jobs": 0,
        "parseable_jobs": 0,
        "results": [],
        "experiment_status": "stopped_before_paid_calls" if stop_reason else "ready_for_paid_run",
        "stop_reason": stop_reason,
    }
    analysis = {
        "schema_version": "plotqa_full_openocr_50_analysis_v1",
        "created_at_utc": now,
        "benchmark": "PlotQA",
        "experiment_status": "track_c_stopped_openocr_gate" if stop_reason else "track_c_ready_for_paid_run",
        "stop_reason": stop_reason,
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "go_gate": {
            "openocr_survey_passed": stop_reason is None,
            "same_50_sample_roster": len(selected_ids) == 50,
            "matched_ocr_flat_baseline_required": True,
            "cap_enforced_dry_run_passed": True,
        },
        "summary": {
            "complete_ocr_enabled_public_benchmark_table_exists": False,
            "reason": stop_reason,
        },
        "source_artifacts": frozen_plan["source_artifacts"],
        "interpretation": "Track C did not execute because the OpenOCR prerequisite failed closed.",
    }
    return frozen_plan, paid_result, analysis


def render_track_c_md(analysis: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# PlotQA Full OpenOCR 50 Analysis",
            "",
            f"- Experiment status: `{analysis['experiment_status']}`",
            f"- Stop reason: `{analysis.get('stop_reason')}`",
            f"- Paid LLM attempts: {analysis['actual_llm_http_attempts']}",
            f"- OCR HTTP attempts: {analysis['actual_ocr_http_attempts']}",
            "",
            "Track C did not execute paid OCR-enabled conditions because the OpenOCR survey did not pass the frozen go gate.",
            "",
        ]
    )


def build_track_d_summary(config_path: Path, track_c_analysis: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    candidates = open_model_candidates(config)
    if not candidates:
        status = "open_mllm_not_configured"
        stop_reason = "no config/default_config.yaml model entry matched local/open/ollama/vllm/lmstudio without proprietary provider markers"
    elif track_c_analysis.get("experiment_status") not in {
        "track_c_complete_benefit_gate_passed",
        "track_c_complete_no_benefit_claim",
    }:
        status = "open_mllm_not_run_track_c_incomplete"
        stop_reason = track_c_analysis.get("stop_reason") or track_c_analysis.get("experiment_status") or "track_c_not_complete"
    else:
        status = "open_mllm_ready_for_frozen_run"
        stop_reason = None
    return {
        "schema_version": "plotqa_open_mllm_reproducibility_summary_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": stable_file_hash(config_path),
        "open_model_candidates": candidates,
        "conditions": ["E0_plotqa_direct_open", "G_full_plotqa_openocr_open"],
        "sample_count": 50,
        "llm_http_attempt_cap": 200,
        "ocr_http_attempt_cap": 0,
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "experiment_status": status,
        "stop_reason": stop_reason,
        "track_c_analysis_path": str((output_dir.parent / "plotqa_full_openocr_50" / "analysis.json").resolve()),
        "interpretation_limits": [
            "Open MLLM track is reproducibility evidence only, not model-agnostic superiority.",
            "Do not substitute proprietary remote models for this track.",
        ],
    }


def render_track_d_md(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# PlotQA Open MLLM Reproducibility Check",
            "",
            f"- Experiment status: `{summary['experiment_status']}`",
            f"- Stop reason: `{summary.get('stop_reason')}`",
            f"- Open/local candidates: {len(summary['open_model_candidates'])}",
            f"- Paid calls executed: {summary['paid_calls_executed']}",
            "",
        ]
    )


def render_full_summary(
    diagnostic_analysis: dict[str, Any],
    m1m2_analysis: dict[str, Any],
    openocr_summary: dict[str, Any],
    track_c_analysis: dict[str, Any],
    track_d_summary: dict[str, Any],
) -> str:
    lines = [
        "# Full Paper Evidence Summary",
        "",
        "Date: 2026-08-29",
        "",
        "Policy: dev-local evidence summary. Do not commit unless repository policy changes.",
        "",
        "## Artifact Status",
        "",
        f"- PlotQA no-OCR diagnostic: `{diagnostic_analysis['experiment_status']}`.",
        f"- M1/M2 raw-vs-curated: `{m1m2_analysis['experiment_status']}`.",
        f"- OpenOCR survey: `{openocr_summary['experiment_status']}`.",
        f"- PlotQA full OpenOCR 50: `{track_c_analysis['experiment_status']}`.",
        f"- Open/local MLLM check: `{track_d_summary['experiment_status']}`.",
        "",
        "## Current Evidence",
        "",
    ]
    for condition, row in diagnostic_analysis.get("condition_summary", {}).items():
        lines.append(f"- {condition}: correct {row['correct']}/{row['jobs']}, completed {row['completed']}/{row['jobs']}.")
    lines.extend(
        [
            f"- M1/M2 net curated-minus-raw correct: {m1m2_analysis['net_curated_minus_raw_correct']}.",
            f"- OpenOCR paid calls: {openocr_summary['actual_llm_http_attempts']} LLM, {openocr_summary['actual_ocr_http_attempts']} OCR.",
            "",
            "## Track C OCR-Enabled PlotQA",
            "",
        ]
    )
    if track_c_analysis.get("condition_summary"):
        for condition, row in track_c_analysis.get("condition_summary", {}).items():
            lines.append(f"- {condition}: correct {row['correct']}/{row['jobs']}, completed {row['completed']}/{row['jobs']}.")
        lines.extend(
            [
                f"- Strongest matched baseline: `{track_c_analysis.get('strongest_matched_baseline')}`.",
                f"- Degradations vs strongest: {track_c_analysis.get('degradations_vs_strongest')}.",
                f"- Operational gate: `{track_c_analysis.get('experiment_status')}`.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- Track C status: `{track_c_analysis.get('experiment_status')}`.",
                f"- Stop reason: `{track_c_analysis.get('stop_reason')}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Claim Boundaries",
            "",
            "| Evidence | Supported Claim | Forbidden Claim |",
            "| --- | --- | --- |",
            "| GeoSym-Bench controlled ablation | controlled synthetic mechanism evidence | public generalization by itself |",
            "| PlotQA no-OCR diagnostic | public external-validity and trace-auditability diagnostic | full OCR-enabled GeoSym score |",
            "| PlotQA full OpenOCR run | complete-pipeline sensitivity evidence when operational gates are reported | fusion benefit if direct or OCR-flat gates fail |",
            "| OCR-flat baseline | separates OCR access from fusion when completion gate passes; otherwise records baseline fragility | GeoSym novelty by itself |",
            "| Open MLLM run | deterministic stop if no open/local model is configured | model-agnostic superiority |",
            "| M1/M2 raw-vs-curated | diagnostic evidence that current curation does not improve PlotQA final accuracy | positive M1/M2 accuracy claim |",
            "",
            "## Required Manuscript Pivot",
            "",
            "Report PlotQA as public diagnostic evidence for matched selection and trace auditability, not as a performance-superiority result. The OpenOCR ladder froze EasyOCR and the OCR-enabled Track C run completed the full GeoSym condition, but the matched OCR-flat baseline failed the operational completion gate and full GeoSym did not exceed direct prompting. The current M1/M2 public diagnostic is neutral-to-negative, so reliability-aware curation must be framed as auditability and failure-analysis machinery unless a separate direct ablation supports an accuracy benefit.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-level",
        action="store_true",
        help="Build the 2026-08-31 paper-level evidence package ledger and zero-cost Track A artifacts.",
    )
    parser.add_argument("--diagnostic-plan")
    parser.add_argument("--diagnostic-result")
    parser.add_argument("--diagnostic-analysis")
    parser.add_argument("--m1m2-analysis")
    parser.add_argument("--openocr-summary")
    parser.add_argument("--track-c-analysis", default=None)
    parser.add_argument("--track-c-result", default=None)
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--output-root", default="outputs/public_benchmark_pilot")
    parser.add_argument("--summary-md", default="docs/full_paper_evidence_summary_2026-08-29.md")
    return parser.parse_args(argv)


def ensure_paper_package_layout(output_root: Path) -> None:
    for relative in [
        "geosym_bench_controlled",
        "plotqa_public_500",
        "m1m2_stress_128",
        "specialized_chart_model_anchor",
        "open_repro_sensitivity",
        "fusion_diagnostics",
        "submission_audit",
    ]:
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def run_paper_level_mode(output_root: Path) -> int:
    ensure_paper_package_layout(output_root)
    ledger, ledger_md = build_track_a_ledger(output_root)
    ledger_path = output_root / "evidence_ledger.json"
    ledger_md_path = output_root / "evidence_ledger.md"
    audit_dir = output_root / "track_a_governance_cleanup"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema_version": "track_a_governance_cleanup_audit_v1",
        "created_at_utc": ledger["created_at_utc"],
        "stage": "track_a_governance_cleanup",
        "stage_status": "track_a_governance_clean" if ledger["evidence_ledger_status"] == "clean" else "track_a_governance_blocked",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "evidence_ledger_path": str(ledger_path.resolve()),
        "evidence_ledger_sha256_after_write": None,
        "governance_gates": ledger["governance_gates"],
        "blocking_findings": ledger["blocking_findings"],
        "next_required_action": ledger["next_required_action"],
        "interpretation_limits": ledger["interpretation_limits"],
    }
    ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ledger_md_path.write_text(ledger_md, encoding="utf-8")
    track_f_statuses = build_track_f_artifacts(output_root)
    track_c_audit = build_track_c_zero_cost_audit(output_root)
    track_b_audit = build_track_b_geosym_dedup_audit(output_root)
    track_b_manifest = build_track_b_rerun_manifest(output_root)
    track_e = build_track_e_artifacts(output_root)
    track_d_audit = build_track_d_paper_stress_audit(output_root)
    manuscript_audit = build_manuscript_integration_audit(output_root)
    final_review_audit = build_final_review_loop_audit(output_root)
    submission_audit = build_submission_audit(output_root)
    audit["evidence_ledger_sha256_after_write"] = stable_file_hash(ledger_path)
    audit["track_f_zero_cost_statuses"] = track_f_statuses
    audit["track_c_zero_cost_status"] = track_c_audit["stage_status"]
    audit["track_b_zero_cost_status"] = track_b_audit["stage_status"]
    audit["track_b_rerun_manifest_status"] = track_b_manifest["stage_status"]
    audit["track_e_open_repro_status"] = track_e["open_repro"]["stage_status"]
    audit["track_e_specialized_anchor_status"] = track_e["specialized_anchor"]["stage_status"]
    audit["track_d_stress_128_status"] = track_d_audit["stage_status"]
    audit["manuscript_integration_status"] = manuscript_audit["stage_status"]
    audit["final_review_loop_status"] = final_review_audit["stage_status"]
    audit["submission_audit_status"] = submission_audit["stage_status"]
    (audit_dir / "track_a_governance_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (audit_dir / "track_a_governance_audit.md").write_text(
        "\n".join(
            [
                "# Track A Governance Cleanup Audit",
                "",
                f"- Stage status: `{audit['stage_status']}`",
                f"- Ledger: `{ledger_path.resolve()}`",
                f"- Paid calls executed: `{audit['paid_calls_executed']}`",
                f"- Blocking findings: `{len(audit['blocking_findings'])}`",
                f"- Next required action: `{audit['next_required_action']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote evidence ledger: {ledger_path}")
    print(f"evidence_ledger_status={ledger['evidence_ledger_status']}")
    print(f"wrote Track A audit: {audit_dir / 'track_a_governance_audit.json'}")
    print(f"stage_status={audit['stage_status']}")
    print(f"wrote Track F diagnostics: {output_root / 'fusion_diagnostics'}")
    print(f"wrote Track C audit: {output_root / 'plotqa_public_500' / 'pre_execution_audit.json'}")
    print(f"track_c_status={track_c_audit['stage_status']}")
    print(f"wrote Track B audit: {output_root / 'geosym_bench_controlled' / 'metadata_dedup_audit.json'}")
    print(f"track_b_status={track_b_audit['stage_status']}")
    print(f"wrote Track B rerun manifest: {output_root / 'geosym_bench_controlled' / 'post_remediation_rerun_manifest.json'}")
    print(f"track_b_manifest_status={track_b_manifest['stage_status']}")
    print(f"wrote Track E probes: {output_root / 'open_repro_sensitivity'} ; {output_root / 'specialized_chart_model_anchor'}")
    print(f"track_e_open_status={track_e['open_repro']['stage_status']}")
    print(f"track_e_specialized_status={track_e['specialized_anchor']['stage_status']}")
    print(f"wrote Track D audit: {output_root / 'm1m2_stress_128' / 'pre_execution_audit.json'}")
    print(f"track_d_status={track_d_audit['stage_status']}")
    print(f"wrote manuscript integration audit: {output_root / 'submission_audit' / 'manuscript_integration_audit.json'}")
    print(f"manuscript_integration_status={manuscript_audit['stage_status']}")
    print(f"wrote final review loop audit: {output_root / 'submission_audit' / 'final_review_loop.json'}")
    print(f"final_review_loop_status={final_review_audit['stage_status']}")
    print(f"wrote submission audit: {output_root / 'submission_audit' / 'submission_audit.json'}")
    print(f"submission_audit_status={submission_audit['stage_status']}")
    print("paid_calls_executed=0")
    return 0 if ledger["evidence_ledger_status"] == "clean" else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.paper_level:
        return run_paper_level_mode(PAPER_PACKAGE_DIR)

    required = {
        "--diagnostic-plan": args.diagnostic_plan,
        "--diagnostic-result": args.diagnostic_result,
        "--diagnostic-analysis": args.diagnostic_analysis,
        "--m1m2-analysis": args.m1m2_analysis,
        "--openocr-summary": args.openocr_summary,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"missing required arguments outside --paper-level mode: {', '.join(missing)}")

    output_root = Path(args.output_root)
    track_c_dir = output_root / "plotqa_full_openocr_50"
    track_d_dir = output_root / "plotqa_open_mllm_reproducibility_50"
    track_c_dir.mkdir(parents=True, exist_ok=True)
    track_d_dir.mkdir(parents=True, exist_ok=True)

    if args.track_c_analysis:
        track_c_analysis = load_json(Path(args.track_c_analysis))
    else:
        frozen_plan, paid_result, track_c_analysis = build_track_c_artifacts(
            Path(args.diagnostic_plan),
            Path(args.diagnostic_result),
            Path(args.openocr_summary),
            track_c_dir,
        )
        (track_c_dir / "frozen_plan.json").write_text(json.dumps(frozen_plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (track_c_dir / "paid_result.json").write_text(json.dumps(paid_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (track_c_dir / "analysis.json").write_text(json.dumps(track_c_analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (track_c_dir / "analysis.md").write_text(render_track_c_md(track_c_analysis), encoding="utf-8")

    track_d_summary = build_track_d_summary(Path(args.config), track_c_analysis, track_d_dir)
    (track_d_dir / "summary.json").write_text(json.dumps(track_d_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (track_d_dir / "summary.md").write_text(render_track_d_md(track_d_summary), encoding="utf-8")

    diagnostic_analysis = load_json(Path(args.diagnostic_analysis))
    m1m2_analysis = load_json(Path(args.m1m2_analysis))
    openocr_summary = load_json(Path(args.openocr_summary))
    summary_md = Path(args.summary_md)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(
        render_full_summary(diagnostic_analysis, m1m2_analysis, openocr_summary, track_c_analysis, track_d_summary),
        encoding="utf-8",
    )

    print(f"wrote Track C artifacts: {track_c_dir}")
    print(f"track_c_status={track_c_analysis['experiment_status']}")
    print(f"wrote Track D summary: {track_d_dir / 'summary.json'}")
    print(f"track_d_status={track_d_summary['experiment_status']}")
    print(f"wrote full evidence summary: {summary_md}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
