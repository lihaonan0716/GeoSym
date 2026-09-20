#!/usr/bin/env python
"""Recompute frozen benchmark scores under the strict year rule. Zero paid calls.

The generic numeric rule is ``tolerance = max(1.0, |ref| * 0.05)``, which admits
+/-100.35 years at ``ref=2007``. Year answers are identifying labels, so this
rescores them as exact matches via ``score_prediction(..., question=...)``.

Two properties this script guarantees, both verified in its own output:

1. **No paid artifact is modified.** Every input is hashed before and after the
   read and the two manifests are compared. Results are written to a new
   directory. A paid artifact is the only record of money already spent, and its
   condition labels are part of the data.
2. **Only year rows can move.** Every non-year row is rescored too and asserted
   to be unchanged, so a regression in the generic path shows up here rather
   than in a paper table.

The per-arm deltas are the point: the defect inflated arms by *different*
amounts, which distorts between-arm comparison rather than adding a constant.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_public_benchmark_pilot import is_year_answer, score_prediction  # noqa: E402

PACKAGE = ROOT / "outputs" / "paper_level_evidence_package"
PLAN = PACKAGE / "plotqa_public_500" / "plotqa_500_zero_cost_plan.json"

# Artifacts carrying PlotQA numeric rows keyed by a resolvable sample id.
# `m1m2_stress_128` is deliberately absent: its rows are keyed by
# `source_sample_id` under retired selective-propagation conditions, so a
# question cannot be attached without re-deriving a retired arm's roster.
TARGETS = (
    Path("plotqa_public_500/direct_baselines/paid_result.json"),
    Path("plotqa_public_500/measured_confirmation/paid_result.json"),
    Path("m1m2_plotqa_attribution/paid_result.json"),
    Path("plotqa_glm_measured_pilot/paid_result.json"),
    Path("plotqa_glm_pilot_bar_repair/paid_result.json"),
    Path("plotqa_glm_layered_pilot/paid_result.json"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def question_index() -> dict[str, str]:
    plan = load_json(PLAN)
    return {s["sample_id"]: s["question"] for s in plan["selected_samples"]}


def rescore_rows(rows: list[dict[str, Any]],
                 questions: dict[str, str]) -> dict[str, Any]:
    by_condition: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"numeric": 0, "year": 0, "lenient_correct": 0,
                 "strict_correct": 0, "non_year_drift": 0})
    flipped: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict) or row.get("answer_type") != "numeric":
            continue
        condition = row.get("condition") or "unknown"
        stats = by_condition[condition]
        stats["numeric"] += 1

        reference = row.get("reference")
        question = questions.get(row.get("sample_id"))
        prediction = row.get("normalized_prediction")
        recorded = bool(row.get("correct"))

        rescored = score_prediction(prediction, reference, "numeric", question=question)
        is_year = is_year_answer(question, reference)

        if is_year:
            stats["year"] += 1
            stats["lenient_correct"] += recorded
            stats["strict_correct"] += bool(rescored["correct"])
            if recorded != bool(rescored["correct"]):
                flipped.append({
                    "sample_id": row.get("sample_id"),
                    "condition": condition,
                    "question": question,
                    "reference": reference,
                    "prediction": prediction,
                    "was_correct": recorded,
                    "now_correct": bool(rescored["correct"]),
                })
        else:
            # Invariant check: the repair must not move a single non-year row.
            if recorded != bool(rescored["correct"]):
                stats["non_year_drift"] += 1

    return {"by_condition": {k: dict(v) for k, v in sorted(by_condition.items())},
            "flipped_rows": flipped}


def run(output_dir: Path) -> dict[str, Any]:
    questions = question_index()
    before = {str(rel): sha256_of_file(PACKAGE / rel) for rel in TARGETS}

    artifacts: dict[str, Any] = {}
    for rel in TARGETS:
        document = load_json(PACKAGE / rel)
        artifacts[str(rel)] = rescore_rows(document.get("results") or [], questions)

    after = {str(rel): sha256_of_file(PACKAGE / rel) for rel in TARGETS}
    unmodified = before == after

    drift = sum(stats["non_year_drift"]
                for art in artifacts.values()
                for stats in art["by_condition"].values())

    results = {
        "schema_version": "plotqa_year_rule_rescore_v1",
        "created_at_utc": utc_now(),
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "rule": {
            "before": "tolerance = max(1.0, abs(ref) * 0.05)",
            "after": "year answers require an exact match; all other numbers unchanged",
            "year_detection": "question asks 'which year' AND reference is an "
                              "integral value in [1800, 2100]; both required",
        },
        "paid_artifacts_unmodified": unmodified,
        "input_sha256_before": before,
        "input_sha256_after": after,
        "non_year_rows_that_moved": drift,
        "artifacts": artifacts,
        "interpretation_limits": [
            "Rescoring changes reported scores, not any stored paid artifact. "
            "The originals are byte-identical; this directory holds the "
            "recomputation.",
            "The strict rule widens the gap between GeoSym and the image-only "
            "baseline rather than narrowing it. It is applied because it is "
            "correct, not because of which direction it moves the result.",
            "m1m2_stress_128 is excluded: its rows carry `source_sample_id` "
            "under retired selective-propagation conditions, so no question "
            "can be attached without re-deriving a retired arm's roster.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "year_rule_rescore.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def render(results: dict[str, Any]) -> str:
    lines = [
        "# PlotQA year-rule rescore (zero cost)", "",
        f"Created: {results['created_at_utc']}",
        f"Paid calls: {results['paid_calls_executed']}",
        f"Paid artifacts byte-identical after run: "
        f"**{results['paid_artifacts_unmodified']}**",
        f"Non-year rows that moved: **{results['non_year_rows_that_moved']}** "
        f"(must be 0)", "",
        "Year answers are identifying labels. The generic rule "
        "`max(1.0, |ref| * 0.05)` admits +/-100.35 years at ref=2007.", "",
    ]
    for name, art in results["artifacts"].items():
        lines += [f"## {name}", "",
                  "| condition | numeric | year rows | lenient | strict | delta |",
                  "|---|---:|---:|---:|---:|---:|"]
        for condition, s in art["by_condition"].items():
            lines.append(
                f"| {condition} | {s['numeric']} | {s['year']} | "
                f"{s['lenient_correct']} | {s['strict_correct']} | "
                f"{s['strict_correct'] - s['lenient_correct']:+d} |")
        lines.append("")
    lines += ["## Limits", ""]
    lines += [f"- {limit}" for limit in results["interpretation_limits"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=PACKAGE / "plotqa_year_rule_rescore")
    args = parser.parse_args(argv)
    results = run(args.output_dir)
    markdown = render(results)
    (args.output_dir / "year_rule_rescore.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    # Fail loudly if the safety properties do not hold.
    if not results["paid_artifacts_unmodified"]:
        print("FAIL: a paid artifact changed on disk", file=sys.stderr)
        return 1
    if results["non_year_rows_that_moved"]:
        print("FAIL: the repair moved a non-year row", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
