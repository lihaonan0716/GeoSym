#!/usr/bin/env python
"""Freeze a pilot-informed Phase C repair-round manifest without paid calls."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402

RECOVERY_ROOT = ROOT / "outputs" / "m1m2_recovery"
ROUND2_RESULT = RECOVERY_ROOT / "phase_c_pilot_repair_round_2_paid_result.json"
PROGRESS_AUDIT = RECOVERY_ROOT / "recovery_progress_audit.json"
PREREG = RECOVERY_ROOT / "preregistration.json"
LEDGER = ROOT / "outputs" / "autonomous_run_log" / "budget_ledger.json"
ROUND3_MANIFEST = RECOVERY_ROOT / "phase_c_pilot_repair_round_3_manifest.json"
PREREG_REVISION = RECOVERY_ROOT / "preregistration_pilot_informed_revision.json"
ALPHA = 0.05
POWER = 0.80
MIN_MEANINGFUL_EFFECT = 0.19


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def paired_deltas(result: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in result.get("results") or []:
        if row.get("status") != "success":
            continue
        if row.get("objective9_mean_score") is None:
            continue
        jobs[(str(row["sample_id"]), str(row["quality_level"]))][str(row["condition"])] = row
    deltas = []
    for (sample_id, quality_level), by_condition in sorted(jobs.items()):
        e5 = by_condition.get("E5")
        raw = by_condition.get("E5_raw")
        if not e5 or not raw:
            continue
        deltas.append(
            {
                "sample_id": sample_id,
                "quality_level": quality_level,
                "delta": float(e5["objective9_mean_score"]) - float(raw["objective9_mean_score"]),
            }
        )
    return deltas


def paired_stats(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["delta"]) for row in deltas]
    n = len(values)
    mean = sum(values) / n if n else None
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1)) if n > 1 and mean is not None else None
    se = sd / math.sqrt(n) if sd is not None and n else None
    z_alpha = float(stats.norm.ppf(1 - ALPHA / 2))
    z_power = float(stats.norm.ppf(POWER))
    required_019 = math.ceil(((z_alpha + z_power) * sd / MIN_MEANINGFUL_EFFECT) ** 2) if sd else None
    mde_44_pairs = (z_alpha + z_power) * sd / math.sqrt(44) if sd else None
    mde_132_pairs = (z_alpha + z_power) * sd / math.sqrt(132) if sd else None
    return {
        "n": n,
        "mean_delta": round(mean, 6) if mean is not None else None,
        "sd": round(sd, 6) if sd is not None else None,
        "se": round(se, 6) if se is not None else None,
        "ci95": [round(mean - 1.959963984540054 * se, 6), round(mean + 1.959963984540054 * se, 6)] if mean is not None and se is not None else [None, None],
        "exact_zero_deltas": sum(1 for value in values if value == 0),
        "required_pairs_for_0_19": required_019,
        "mde_80_power_at_44_pairs_per_quality_level": round(mde_44_pairs, 6) if mde_44_pairs is not None else None,
        "mde_80_power_at_132_pooled_pairs": round(mde_132_pairs, 6) if mde_132_pairs is not None else None,
    }


def build_revision() -> tuple[dict[str, Any], dict[str, Any]]:
    now = utc_now()
    result = load_json(ROUND2_RESULT)
    audit = load_json(PROGRESS_AUDIT)
    prereg = load_json(PREREG)
    ledger = load_json(LEDGER)
    deltas = paired_deltas(result)
    stats_row = paired_stats(deltas)
    global_remaining = int(ledger.get("llm_cap") or 0) - int(ledger.get("cumulative_llm") or 0)
    remaining_jobs = max(0, 24 - int((audit.get("phase_c_summary") or {}).get("completed_sample_quality_jobs") or 0))
    round3_cap = math.ceil(remaining_jobs * 9 * 1.2)
    revision = {
        "schema_version": "m1m2_recovery_pilot_informed_preregistration_revision_v1",
        "created_at_utc": now,
        "stage": "m1m2_recovery_pilot_informed_preregistration_revision",
        "stage_status": "pilot_informed_revision_frozen_pending_round3",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "original_preregistration": {"path": str(PREREG.resolve()), "sha256": stable_file_hash(PREREG)},
        "pilot_source": {"path": str(ROUND2_RESULT.resolve()), "sha256": stable_file_hash(ROUND2_RESULT)},
        "revision_reason": "Phase A used Track B objective9 marginal paired deltas for planning. Completed Phase C pilot pairs provide a direct paired-delta variance estimate for the actual degraded-quality protocol.",
        "pilot_informed_variance": stats_row,
        "governance_notes": [
            "This is a pilot-informed revision, not the original preregistration.",
            "The revision uses all completed successful E5/E5_raw pairs available before round 3; it does not select samples by outcome.",
            "The small n and high exact-zero rate are negative/neutral mechanism evidence and must be reported, not hidden.",
        ],
        "full_phase_c_feasibility_under_current_global_cap": {
            "global_llm_cap": ledger.get("llm_cap"),
            "cumulative_llm_after_accounting_repairs": ledger.get("cumulative_llm"),
            "global_remaining_llm_attempts": global_remaining,
            "attempts_per_pair": 9,
            "approx_pairs_affordable": global_remaining // 9,
            "target_pairs_per_quality_level": 44,
            "total_target_pairs": 132,
            "mde_80_power_at_44_pairs_per_quality_level": stats_row["mde_80_power_at_44_pairs_per_quality_level"],
            "mde_80_power_at_132_pooled_pairs": stats_row["mde_80_power_at_132_pooled_pairs"],
        },
        "interpretation_limits": [
            "The expected conclusion remains open; the current pilot CI is already close to the minimum meaningful effect bound.",
            "If full Phase C is run, a narrow negative result is a valid outcome.",
        ],
    }
    manifest = {
        "schema_version": "m1m2_recovery_phase_c_repair_round_manifest_v1",
        "created_at_utc": now,
        "stage": "m1m2_recovery_phase_c_repair_round_3_manifest",
        "stage_status": "repair_round_3_frozen_pending_paid_execution",
        "round_label": "pilot_repair_round_3",
        "reason_for_new_round": "Round 2 made partial progress but ended with an unclosed in-progress checkpoint under a gateway hang. Round 3 is limited to closing the original 8-sample pilot.",
        "original_preregistration": {"path": str(PREREG.resolve()), "sha256": stable_file_hash(PREREG)},
        "pilot_informed_revision": {"path": str(PREREG_REVISION.resolve())},
        "source_artifacts": {
            "progress_audit": {"path": str(PROGRESS_AUDIT.resolve()), "sha256": stable_file_hash(PROGRESS_AUDIT)},
            "round2_result": {"path": str(ROUND2_RESULT.resolve()), "sha256": stable_file_hash(ROUND2_RESULT)},
            "budget_ledger": {"path": str(LEDGER.resolve()), "sha256": stable_file_hash(LEDGER)},
        },
        "conditions": ["E5", "E5_raw"],
        "remaining_sample_quality_jobs": remaining_jobs,
        "semantic_call_graph": {"E5": 5, "E5_raw": 4},
        "cap": {
            "llm_http_attempt_cap": round3_cap,
            "ocr_http_attempt_cap": 0,
            "failure_margin": "20%",
            "global_remaining_before_round": global_remaining,
        },
        "transport_policy": {
            "base_adapter_retry_cap": 1,
            "openai_sdk_max_retries": 0,
            "qwen_sdk_max_retries": 0,
            "rationale": "Counted semantic attempts must correspond to at most one SDK network attempt per BaseLLMAdapter call.",
        },
        "stopping_rule": "Run only until the original 24 pilot sample-quality jobs are closed or this round-3 cap is exhausted. Do not start full Phase C from this manifest.",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
    }
    manifest["pilot_informed_revision"]["sha256_after_write"] = None
    return revision, manifest


def main() -> int:
    revision, manifest = build_revision()
    write_json(PREREG_REVISION, revision)
    manifest["pilot_informed_revision"]["sha256_after_write"] = stable_file_hash(PREREG_REVISION)
    write_json(ROUND3_MANIFEST, manifest)
    print(f"wrote {PREREG_REVISION}")
    print(f"wrote {ROUND3_MANIFEST}")
    print(f"round3_cap={manifest['cap']['llm_http_attempt_cap']}")
    print(f"global_remaining_before_round={manifest['cap']['global_remaining_before_round']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
