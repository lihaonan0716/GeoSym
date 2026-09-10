"""Static audit for matched domain/range prompt instructions.

The audit is intentionally zero-cost: it only reads prompt templates and checks
that baseline conditions E0-E4 carry the same domain/range caution that E5 L3
already had. It does not call any model or touch experiment outputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PromptCondition:
    condition: str
    prompt_path: Path
    inherited_from: str | None = None


PROMPT_CONDITIONS = (
    PromptCondition("E0", ROOT / "prompts" / "reasoning_direct_single_prompt.txt"),
    PromptCondition("E1", ROOT / "prompts" / "reasoning_direct_cot_prompt.txt"),
    PromptCondition(
        "E2",
        ROOT / "prompts" / "reasoning_direct_cot_prompt.txt",
        inherited_from="E1",
    ),
    PromptCondition("E3", ROOT / "prompts" / "reasoning_ocr_single_prompt.txt"),
    PromptCondition("E4", ROOT / "prompts" / "reasoning_ocr_cot_prompt.txt"),
    PromptCondition("E5_L3", ROOT / "prompts" / "reasoning_L3_prompt.txt"),
)


REQUIRED_MARKERS = {
    "mathematical_domain": ("mathematical domain",),
    "domain_not_visible_axis": ("not the visible x-axis range",),
    "domain_defined_values": ("all x-values", "mathematically defined"),
    "domain_not_boundaries": ("not from the image boundaries",),
    "mathematical_range": ("mathematical range",),
    "range_not_visible_axis": ("not the visible y-axis range",),
    "range_all_outputs": ("all possible output y-values",),
    "range_expression_extrema": ("expression", "extrema"),
}


def _has_all(text: str, markers: Iterable[str]) -> bool:
    lowered = text.lower()
    return all(marker.lower() in lowered for marker in markers)


def audit_prompt(condition: PromptCondition) -> dict[str, object]:
    text = condition.prompt_path.read_text(encoding="utf-8")
    marker_results = {
        name: _has_all(text, markers)
        for name, markers in REQUIRED_MARKERS.items()
    }
    return {
        "condition": condition.condition,
        "prompt": condition.prompt_path.relative_to(ROOT).as_posix(),
        "inherited_from": condition.inherited_from,
        "passed": all(marker_results.values()),
        "markers": marker_results,
    }


def run_audit() -> list[dict[str, object]]:
    return [audit_prompt(condition) for condition in PROMPT_CONDITIONS]


def _format_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Condition | Prompt | Inherits | Matched domain/range guidance |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        inherited = row["inherited_from"] or ""
        lines.append(
            f"| {row['condition']} | `{row['prompt']}` | {inherited} | {status} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit matched domain/range prompt guidance for E0-E5."
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
