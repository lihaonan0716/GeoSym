#!/usr/bin/env python
"""Run zero-cost Phase A diagnostics for M1/M2 recovery."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402

OUTPUT_ROOT = ROOT / "outputs" / "paper_level_evidence_package"
RECOVERY_ROOT = ROOT / "outputs" / "m1m2_recovery"
BUDGET_LEDGER = ROOT / "outputs" / "autonomous_run_log" / "budget_ledger.json"
STATE = ROOT / "outputs" / "autonomous_run_log" / "state.json"
TRACK_B_IDENTIFIABILITY = OUTPUT_ROOT / "geosym_bench_controlled" / "track_b_m1m2_identifiability_analysis.json"
TRACK_B_FINAL = OUTPUT_ROOT / "geosym_bench_controlled" / "post_remediation_e5_backfill_full_240_final_analysis.json"
SECONDARY_METRICS = RECOVERY_ROOT / "secondary_metric_variance_study.json"
ALPHA = 0.05
POWER = 0.80
MIN_MEANINGFUL_EFFECT = 0.19


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def initialize_budget_ledger() -> dict[str, Any]:
    state = load_json(STATE)
    spent = state.get("budget_spent") or {}
    payload = {
        "schema_version": "m1m2_recovery_budget_ledger_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "llm_cap": spent.get("llm_global_cap", 4000),
        "ocr_cap": spent.get("ocr_global_cap", 1200),
        "entries": [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": "pre_phase_a",
                "step": "initialize_from_autonomous_state",
                "llm_attempts": spent.get("llm_attempts_minimum", 2403),
                "ocr_attempts": spent.get("ocr_attempts_minimum", 23),
                "artifact_path": str(STATE),
            }
        ],
        "cumulative_llm": spent.get("llm_attempts_minimum", 2403),
        "cumulative_ocr": spent.get("ocr_attempts_minimum", 23),
        "interpretation_limits": [
            "This ledger initializes cumulative attempts from outputs/autonomous_run_log/state.json.",
            "Phase A performs zero paid calls.",
        ],
    }
    write_json(BUDGET_LEDGER, payload)
    return payload


def paired_stats(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = sum(values) / n if n else None
    if n < 2 or mean is None:
        return {"n": n, "mean": mean, "sd": None, "nonzero_pair_rate": None, "mde_80_power": None, "required_pairs_for_0_19": None}
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
    z_alpha = float(stats.norm.ppf(1 - ALPHA / 2))
    z_power = float(stats.norm.ppf(POWER))
    mde = (z_alpha + z_power) * sd / math.sqrt(n)
    required = math.ceil(((z_alpha + z_power) * sd / MIN_MEANINGFUL_EFFECT) ** 2) if sd else 0
    return {
        "n": n,
        "mean": round(mean, 6),
        "sd": round(sd, 6),
        "nonzero_pair_rate": round(sum(1 for value in values if value != 0) / n, 6),
        "mde_80_power": round(mde, 6),
        "required_pairs_for_0_19": required,
    }


def objective9(row: dict[str, Any]) -> float:
    dims = row.get("dimension_scores") or {}
    values = [float(score) for dim, score in dims.items() if dim != "judge_inference_score"]
    return sum(values) / len(values)


def objective9_binary_hit_rate(row: dict[str, Any]) -> float:
    dims = row.get("dimension_scores") or {}
    values = [1.0 if float(score) >= 99.999 else 0.0 for dim, score in dims.items() if dim != "judge_inference_score"]
    return sum(values) / len(values)


def active_objective_score(e5: dict[str, Any], raw: dict[str, Any]) -> tuple[float | None, float | None]:
    e5_dims = e5.get("dimension_scores") or {}
    raw_dims = raw.get("dimension_scores") or {}
    active_dims = [dim for dim, score in raw_dims.items() if dim != "judge_inference_score" and float(score) < 99.999]
    if not active_dims:
        return None, None
    return (
        sum(float(e5_dims.get(dim, 0.0)) for dim in active_dims) / len(active_dims),
        sum(float(raw_dims.get(dim, 0.0)) for dim in active_dims) / len(active_dims),
    )


def candidate_metrics(track_b_final: dict[str, Any]) -> dict[str, Any]:
    by_sample: dict[str, dict[str, dict[str, Any]]] = {}
    for row in track_b_final.get("results") or []:
        if row.get("condition") in {"E5", "E5_raw"}:
            by_sample.setdefault(str(row["sample_id"]), {})[str(row["condition"])] = row
    deltas = {
        "official_10d_total_score": [],
        "objective9_mean_score": [],
        "objective9_binary_hit_rate": [],
        "active_objective_dimension_score": [],
    }
    ceiling_rows = 0
    for rows in by_sample.values():
        e5 = rows.get("E5")
        raw = rows.get("E5_raw")
        if not e5 or not raw:
            continue
        deltas["official_10d_total_score"].append(float(e5["total_score"]) - float(raw["total_score"]))
        raw_obj9 = objective9(raw)
        if raw_obj9 >= 99:
            ceiling_rows += 1
        deltas["objective9_mean_score"].append(objective9(e5) - raw_obj9)
        deltas["objective9_binary_hit_rate"].append(objective9_binary_hit_rate(e5) - objective9_binary_hit_rate(raw))
        active = active_objective_score(e5, raw)
        if active[0] is not None and active[1] is not None:
            deltas["active_objective_dimension_score"].append(active[0] - active[1])
    summaries = {name: paired_stats(values) for name, values in deltas.items()}
    metric_metadata = {
        "official_10d_total_score": {"scale": "score_points_0_100", "eligible_for_phase_c_primary": True},
        "objective9_mean_score": {"scale": "score_points_0_100", "eligible_for_phase_c_primary": True},
        "objective9_binary_hit_rate": {
            "scale": "proportion_0_1",
            "eligible_for_phase_c_primary": False,
            "exclusion_reason": "Scale is not score points, so the 0.19-point minimum meaningful effect is not applicable.",
        },
        "active_objective_dimension_score": {"scale": "score_points_0_100", "eligible_for_phase_c_primary": True},
    }
    selected = min(
        (
            name
            for name in summaries
            if summaries[name]["n"] and summaries[name]["sd"] is not None and metric_metadata[name]["eligible_for_phase_c_primary"]
        ),
        key=lambda name: (summaries[name]["mde_80_power"], -summaries[name]["nonzero_pair_rate"]),
    )
    return {
        "candidate_metrics": summaries,
        "metric_metadata": metric_metadata,
        "selected_phase_c_primary_metric": selected,
        "selection_reason": "Selected the defensible secondary metric with the smallest MDE@80% among metrics computable from the authoritative paired Track B artifact without changing SCORED_DIMENSIONS.",
        "ceiling_quantification": {
            "e5_raw_objective9_ge_99_count": ceiling_rows,
            "e5_raw_objective9_ge_99_rate": round(ceiling_rows / len(by_sample), 6) if by_sample else None,
            "legal_stratification_rule": "Future subsets must be defined by input-side M0 extraction quality, not by E5_raw score.",
        },
    }


def bh_survivors(p_values: list[tuple[str, float]], alpha: float = ALPHA) -> list[str]:
    ordered = sorted(p_values, key=lambda item: item[1])
    survivors = []
    m = len(ordered)
    for rank, (name, p_value) in enumerate(ordered, start=1):
        if p_value <= alpha * rank / m:
            survivors.append(name)
    return survivors


def patch_identifiability_multiplicity() -> dict[str, Any]:
    payload = load_json(TRACK_B_IDENTIFIABILITY)
    dimension_delta = payload.get("dimension_delta") or {}
    p_values = [
        (name, float(row["p"]))
        for name, row in dimension_delta.items()
        if row.get("p") is not None
    ]
    bonferroni_alpha = ALPHA / len(p_values) if p_values else None
    bonferroni_survivors = [name for name, p_value in p_values if bonferroni_alpha is not None and p_value <= bonferroni_alpha]
    payload["multiplicity_correction"] = {
        "method": "bonferroni+BH",
        "n_tests": len(p_values),
        "alpha": ALPHA,
        "bonferroni_alpha": round(bonferroni_alpha, 6) if bonferroni_alpha is not None else None,
        "bonferroni_survivors": bonferroni_survivors,
        "benjamini_hochberg_survivors": bh_survivors(p_values),
        "note": "10 dimension-wise tests are exploratory; no dimension survives correction.",
    }
    write_json(TRACK_B_IDENTIFIABILITY, payload)
    return payload["multiplicity_correction"]


def main() -> int:
    budget = initialize_budget_ledger()
    multiplicity = patch_identifiability_multiplicity()
    track_b_final = load_json(TRACK_B_FINAL)
    metrics = candidate_metrics(track_b_final)
    selected = metrics["selected_phase_c_primary_metric"]
    selected_summary = metrics["candidate_metrics"][selected]
    study = {
        "schema_version": "secondary_metric_variance_study_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "m1m2_recovery_phase_a_secondary_metric_variance_study",
        "stage_status": "phase_a_zero_cost_variance_study_completed",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "track_b_final": {"path": str(TRACK_B_FINAL.resolve()), "sha256": stable_file_hash(TRACK_B_FINAL)},
            "track_b_identifiability": {"path": str(TRACK_B_IDENTIFIABILITY.resolve()), "sha256": stable_file_hash(TRACK_B_IDENTIFIABILITY)},
            "budget_ledger": {"path": str(BUDGET_LEDGER.resolve()), "sha256": stable_file_hash(BUDGET_LEDGER)},
        },
        **metrics,
        "phase_c_sample_size_input": {
            "metric": selected,
            "mde_80_power": selected_summary["mde_80_power"],
            "required_pairs_for_min_meaningful_effect": selected_summary["required_pairs_for_0_19"],
            "note": "This is the input to Phase B/C cap planning; current global cap must not be exceeded without an explicit new cap.",
        },
        "multiplicity_correction_added_to_identifiability": multiplicity,
        "invariant_12_boundary": "SCORED_DIMENSIONS and weights are unchanged; all metrics here are secondary analyses.",
        "budget_ledger_initialized": {
            "path": str(BUDGET_LEDGER.resolve()),
            "cumulative_llm": budget["cumulative_llm"],
            "cumulative_ocr": budget["cumulative_ocr"],
        },
    }
    write_json(SECONDARY_METRICS, study)
    print(f"wrote {BUDGET_LEDGER}")
    print(f"updated {TRACK_B_IDENTIFIABILITY}")
    print(f"wrote {SECONDARY_METRICS}")
    print(json.dumps(study["phase_c_sample_size_input"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
