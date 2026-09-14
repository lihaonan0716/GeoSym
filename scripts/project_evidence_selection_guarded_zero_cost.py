#!/usr/bin/env python
"""Zero-cost projection for the opt-in guarded M1/M2 PlotQA arm."""

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

from scripts.prepare_public_benchmark_pilot import stable_file_hash  # noqa: E402

PROTECTED_CONDITION = "G_full_measured_geosym"
PROJECTED_CONDITION = "G_full_guarded"
SCHEMA_VERSION = "m1m2_guarded_zero_cost_projection_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def protected_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in result.get("results", []) if row.get("condition") == PROTECTED_CONDITION]


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "completed": sum(1 for row in rows if row.get("completed")),
        "parseable": sum(1 for row in rows if row.get("parseable")),
        "correct": sum(1 for row in rows if row.get("correct")),
        "deterministic": sum(1 for row in rows if row.get("deterministic_m3") or row.get("deterministic_m3_rescore")),
    }


def build_projection(protected_result_path: Path, schema_outputs_path: Path, output_dir: Path) -> dict[str, Any]:
    protected_result = load_json(protected_result_path)
    schema_outputs = load_json(schema_outputs_path)
    schema_by_id = {row["sample_id"]: row for row in schema_outputs.get("outputs", [])}
    projected_rows = []
    regressions = []
    policy_counts: Counter[str] = Counter()
    for row in protected_rows(protected_result):
        sample_id = row["sample_id"]
        schema_row = schema_by_id.get(sample_id)
        policy = None
        selected_count = 0
        suppressed_count = 0
        surfaced_gap_count = 0
        if schema_row:
            selected = schema_row.get("selected_evidence_set") or {}
            policy = selected.get("propagation_policy")
            selected_count = len(selected.get("selected_fact_ids") or [])
            suppressed_count = len(selected.get("suppressed_fact_ids") or [])
            surfaced_gap_count = len(selected.get("surfaced_gap_ids") or [])
            if policy:
                policy_counts[str(policy)] += 1
        projected = dict(row)
        projected["condition"] = PROJECTED_CONDITION
        projected["projection_source"] = {
            "source_condition": PROTECTED_CONDITION,
            "source": "protected_pass_through_zero_cost",
            "schema_condition": schema_outputs.get("condition"),
            "propagation_policy": policy,
            "selected_fact_count": selected_count,
            "suppressed_fact_count": suppressed_count,
            "surfaced_gap_count": surfaced_gap_count,
        }
        projected["protected_default_output_changed"] = False
        projected_rows.append(projected)
        if projected.get("normalized_prediction") != row.get("normalized_prediction") or projected.get("correct") != row.get("correct"):
            regressions.append(sample_id)
    summary = summarize(projected_rows)
    guardrail_passed = (
        summary["correct"] >= 48
        and summary["completed"] == 50
        and summary["parseable"] == 50
        and not regressions
        and schema_outputs.get("stage_status") == "ready_for_zero_cost_projection"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "stage": "p4_m1m2_guarded_zero_cost_projection",
        "stage_status": "plotqa_guardrail_preserved" if guardrail_passed else "plotqa_guardrail_failed",
        "condition": PROJECTED_CONDITION,
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "source_artifacts": {
            "protected_result": {
                "path": str(protected_result_path.resolve()),
                "sha256": stable_file_hash(protected_result_path),
                "stage_status": protected_result.get("stage_status"),
            },
            "schema_outputs": {
                "path": str(schema_outputs_path.resolve()),
                "sha256": stable_file_hash(schema_outputs_path),
                "stage_status": schema_outputs.get("stage_status"),
            },
        },
        "summary": {PROJECTED_CONDITION: summary},
        "policy_counts": dict(sorted(policy_counts.items())),
        "guardrail": {
            "required_correct": 48,
            "required_completed": 50,
            "required_parseable": 50,
            "protected_output_regressions": len(regressions),
            "protected_output_regression_sample_ids": regressions,
            "passed": guardrail_passed,
        },
        "results": projected_rows,
        "next_required_action": "freeze_bounded_paid_pilot_manifest" if guardrail_passed else "keep_guarded_arm_diagnostic_and_repair_projection",
        "interpretation_limits": [
            "This is a zero-cost pass-through projection, not a paid confirmation.",
            "It proves the opt-in guarded arm can preserve the protected PlotQA output under pass-through policy.",
            "It does not prove the M1/M2 stress mechanism gates.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "guarded_projection_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "guarded_projection_audit.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"][PROJECTED_CONDITION]
    lines = [
        "# M1/M2 Guarded Zero-Cost Projection",
        "",
        f"- Stage status: `{result['stage_status']}`",
        "- Paid calls executed: 0",
        f"- {PROJECTED_CONDITION}: correct {summary['correct']}/50, completed {summary['completed']}/50, parseable {summary['parseable']}/50",
        f"- Protected output regressions: {result['guardrail']['protected_output_regressions']}",
        "",
        "## Propagation Policies",
        "",
    ]
    for policy, count in result["policy_counts"].items():
        lines.append(f"- {policy}: {count}")
    lines.extend(["", "## Next Required Action", "", f"- `{result['next_required_action']}`", "", "## Limits", ""])
    lines.extend(f"- {line}" for line in result["interpretation_limits"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected-result", default="outputs/public_benchmark_pilot/plotqa_measured_geosym_50/stage5_measured_flat_deterministic_rescore/rescore_result.json")
    parser.add_argument("--schema-outputs", default="outputs/m1m2_recovery/schema_outputs/m1m2_schema_outputs.json")
    parser.add_argument("--output-dir", default="outputs/m1m2_recovery/guarded_zero_cost_projection")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = build_projection(Path(args.protected_result), Path(args.schema_outputs), Path(args.output_dir))
    print(f"wrote guarded projection: {Path(args.output_dir) / 'guarded_projection_result.json'}")
    print(f"stage_status={result['stage_status']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
