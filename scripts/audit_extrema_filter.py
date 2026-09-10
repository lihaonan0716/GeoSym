"""Zero-cost audit for the P0-2 extrema candidate filter.

This script exercises the frozen discontinuity-bridge rule on small synthetic
coordinate sets. It does not read benchmark labels, family names, reports,
private artifacts, or model outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.reasoning.intervention import (
    EXTREMA_FILTER_RULE_VERSION,
    classify_extrema_candidates,
)


AUDIT_CASES = {
    "true_parabola_max": {
        "coords": [(-2, 0), (-1, 3), (0, 4), (1, 3), (2, 0)],
        "expected": ["KEEP"],
        "purpose": "keeps a standard smooth local maximum",
    },
    "monotone_line": {
        "coords": [(-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2)],
        "expected": [],
        "purpose": "does not invent extrema on a monotone function",
    },
    "pole_bridge": {
        "coords": [(-2, -1), (-1, -2), (1, 2), (2, 1)],
        "expected": ["DROP", "DROP"],
        "purpose": "drops extrema-like turns adjacent to a discontinuity bridge",
    },
    "regular_large_gap_true_max": {
        "coords": [(-4, -12), (-2, 0), (0, 4), (2, 0), (4, -12)],
        "expected": ["KEEP"],
        "purpose": "keeps true extrema under regular wider sampling",
    },
}


def run_audit() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_name, case in AUDIT_CASES.items():
        decisions = classify_extrema_candidates(case["coords"])
        actual = [decision["decision"] for decision in decisions]
        rows.append(
            {
                "case": case_name,
                "purpose": case["purpose"],
                "rule": EXTREMA_FILTER_RULE_VERSION,
                "expected": case["expected"],
                "actual": actual,
                "passed": actual == case["expected"],
                "decisions": decisions,
            }
        )
    return rows


def _format_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Case | Purpose | Expected | Actual | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(
            f"| {row['case']} | {row['purpose']} | {row['expected']} | "
            f"{row['actual']} | {status} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit P0-2 extrema KEEP/DROP filtering on synthetic controls."
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    args = parser.parse_args()

    rows = run_audit()
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(_format_markdown(rows))

    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
