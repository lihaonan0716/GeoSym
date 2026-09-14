#!/usr/bin/env python
"""Freeze a bounded Phase C repair-round-4 manifest without paid calls."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_m1m2_recovery_round3 import paired_deltas, paired_stats  # noqa: E402
from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402

RECOVERY_ROOT = ROOT / "outputs" / "m1m2_recovery"
ROUND3_RESULT = RECOVERY_ROOT / "phase_c_pilot_repair_round_3_paid_result.json"
PROGRESS_AUDIT = RECOVERY_ROOT / "recovery_progress_audit.json"
PREREG = RECOVERY_ROOT / "preregistration.json"
LEDGER = ROOT / "outputs" / "autonomous_run_log" / "budget_ledger.json"
ROUND4_MANIFEST = RECOVERY_ROOT / "phase_c_pilot_repair_round_4_manifest.json"
PREREG_REVISION = RECOVERY_ROOT / "preregistration_pilot_informed_revision_round4.json"
CONDITION_CALL_CAPS = {"E5": 5, "E5_raw": 4}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def failed_condition_cap(result: dict[str, Any]) -> tuple[int, int, list[dict[str, Any]]]:
    jobs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in result.get("results") or []:
        jobs[(str(row["sample_id"]), str(row["quality_level"]))][str(row["condition"])] = row
    failed_conditions = []
    for (sample_id, quality_level), by_condition in sorted(jobs.items()):
        for condition in ("E5", "E5_raw"):
            row = by_condition.get(condition)
            if row is None or row.get("status") != "success":
                failed_conditions.append(
                    {
                        "sample_id": sample_id,
                        "quality_level": quality_level,
                        "condition": condition,
                        "prior_status": None if row is None else row.get("status"),
                        "prior_error": None if row is None else row.get("error"),
                        "condition_llm_http_attempt_cap": CONDITION_CALL_CAPS[condition],
                    }
                )
    base_cap = sum(int(row["condition_llm_http_attempt_cap"]) for row in failed_conditions)
    return len(failed_conditions), base_cap, failed_conditions


def main() -> int:
    now = utc_now()
    result = load_json(ROUND3_RESULT)
    audit = load_json(PROGRESS_AUDIT)
    ledger = load_json(LEDGER)
    failed_count, base_cap, failed_conditions = failed_condition_cap(result)
    round4_cap = math.ceil(base_cap * 1.2)
    global_remaining = int(ledger.get("llm_cap") or 0) - int(ledger.get("cumulative_llm") or 0)
    stats_row = paired_stats(paired_deltas(result))
    revision = {
        "schema_version": "m1m2_recovery_pilot_informed_preregistration_revision_v1",
        "created_at_utc": now,
        "stage": "m1m2_recovery_pilot_informed_preregistration_revision_round4",
        "stage_status": "pilot_informed_revision_frozen_pending_round4",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "original_preregistration": {"path": str(PREREG.resolve()), "sha256": stable_file_hash(PREREG)},
        "pilot_source": {"path": str(ROUND3_RESULT.resolve()), "sha256": stable_file_hash(ROUND3_RESULT)},
        "pilot_informed_variance": stats_row,
        "governance_notes": [
            "This revision uses all successful pairs in the round-3 checkpoint before round 4.",
            "It is a pilot-informed repair revision, not the original preregistration.",
            "The high exact-zero rate remains negative/neutral mechanism evidence.",
        ],
    }
    write_json(PREREG_REVISION, revision)
    manifest = {
        "schema_version": "m1m2_recovery_phase_c_repair_round_manifest_v1",
        "created_at_utc": now,
        "stage": "m1m2_recovery_phase_c_repair_round_4_manifest",
        "stage_status": "repair_round_4_frozen_pending_paid_execution",
        "round_label": "pilot_repair_round_4",
        "reason_for_new_round": "Round 3 completed 18/24 pilot sample-quality jobs. Round 4 is limited to retrying failed or missing conditions while preserving successful condition rows.",
        "original_preregistration": {"path": str(PREREG.resolve()), "sha256": stable_file_hash(PREREG)},
        "pilot_informed_revision": {"path": str(PREREG_REVISION.resolve()), "sha256_after_write": stable_file_hash(PREREG_REVISION)},
        "source_artifacts": {
            "progress_audit": {"path": str(PROGRESS_AUDIT.resolve()), "sha256": stable_file_hash(PROGRESS_AUDIT)},
            "round3_result": {"path": str(ROUND3_RESULT.resolve()), "sha256": stable_file_hash(ROUND3_RESULT)},
            "budget_ledger": {"path": str(LEDGER.resolve()), "sha256": stable_file_hash(LEDGER)},
        },
        "conditions": ["E5", "E5_raw"],
        "failed_or_missing_conditions": failed_conditions,
        "failed_or_missing_condition_count": failed_count,
        "semantic_call_graph": CONDITION_CALL_CAPS,
        "cap": {
            "llm_http_attempt_cap": round4_cap,
            "ocr_http_attempt_cap": 0,
            "base_failed_condition_cap": base_cap,
            "failure_margin": "20%",
            "global_remaining_before_round": global_remaining,
        },
        "transport_policy": {
            "base_adapter_retry_cap": 1,
            "openai_sdk_max_retries": 0,
            "qwen_sdk_max_retries": 0,
        },
        "stopping_rule": "Retry only failed or missing conditions needed to close the original 24 pilot jobs; do not start full Phase C from this manifest.",
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
    }
    write_json(ROUND4_MANIFEST, manifest)
    print(f"wrote {PREREG_REVISION}")
    print(f"wrote {ROUND4_MANIFEST}")
    print(f"round4_cap={round4_cap}")
    print(f"failed_or_missing_condition_count={failed_count}")
    print(f"global_remaining_before_round={global_remaining}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
