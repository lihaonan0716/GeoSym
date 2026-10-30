#!/usr/bin/env python
"""Run every free M0 provider over both benchmarks and report what applies where.

Performs **no network calls, no OCR inference, and no paid calls**. Providers
that need a paid OCR backend are excluded by ``available_providers``; the PlotQA
chain reads frozen OCR boxes from disk.

Why this exists
---------------
The architecture claims M0 implementations are substitutable behind an evidence
interface. Until now nothing tested that claim, and nothing could: the three
extractors returned three different shapes and none imported another. A reviewer
reading the code would have found an interface asserted in prose and absent from
the type system.

This script is the check. It runs each provider against charts from **both**
benchmarks -- including the pairings that are expected to fail -- and reports the
applicability matrix. A provider declining a chart outside its domain is as much
a result as one succeeding inside it: together they establish that the domain
boundaries are real and machine-checkable rather than asserted.

What a reader should take from the output
-----------------------------------------
The diagonal (each provider on the benchmark it was built for) shows the
implementations work. The off-diagonal shows *why* a single implementation could
not have covered both: gridlines are a structural property of one benchmark's
charts, not a preference of one extractor.
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

from evaluator.evidence import (  # noqa: E402
    ChartEvidenceRequest,
    available_providers,
)

DEFAULT_OUTPUT = ROOT / "outputs" / "paper_level_evidence_package" / "evidence_provider_substitutability"
PLOTQA_PLAN = (
    ROOT / "outputs" / "paper_level_evidence_package" / "plotqa_public_500"
    / "plotqa_500_zero_cost_plan.json"
)
PLOTQA_RECORDS = (
    ROOT / "outputs" / "paper_level_evidence_package" / "plotqa_public_500"
    / "m0_measurement_records.json"
)
PLOTQA_OCR = (
    ROOT / "outputs" / "paper_level_evidence_package" / "plotqa_public_500"
    / "openocr_survey_paddleocr" / "raw_ocr.json"
)
GEOSYM_IMAGES = ROOT / "dataset" / "images"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_of(payload: Any) -> list[dict[str, Any]]:
    """Normalise the frozen artifacts' two shapes into a row list.

    ``raw_ocr.json`` is a bare list while ``m0_measurement_records.json`` wraps
    its rows in a key. Both are frozen inputs to paid experiments and must not
    be rewritten to match each other, so the reader adapts instead.
    """
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("records", "results", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def geosym_requests(per_family: int) -> list[ChartEvidenceRequest]:
    """Sample GeoSym-Bench charts, stratified by function family.

    Stratified rather than random because the families differ in exactly the
    property under test: every family here is rendered on the same grid, so a
    provider's verdict should not vary across them, and sampling all eight is
    what makes that checkable.
    """
    requests: list[ChartEvidenceRequest] = []
    if not GEOSYM_IMAGES.is_dir():
        return requests
    for family_dir in sorted(GEOSYM_IMAGES.iterdir()):
        if not family_dir.is_dir():
            continue
        for image in sorted(family_dir.glob("*.png"))[:per_family]:
            requests.append(
                ChartEvidenceRequest(image_path=image, sample_id=image.stem)
            )
    return requests


def plotqa_requests(limit: int) -> list[ChartEvidenceRequest]:
    """Build PlotQA requests from the frozen plan, records and OCR boxes."""
    if not (PLOTQA_PLAN.is_file() and PLOTQA_RECORDS.is_file() and PLOTQA_OCR.is_file()):
        return []

    plan = load_json(PLOTQA_PLAN)
    records = {r["sample_id"]: r for r in _rows_of(load_json(PLOTQA_RECORDS))}
    ocr_rows = {row["sample_id"]: row for row in _rows_of(load_json(PLOTQA_OCR))}

    requests: list[ChartEvidenceRequest] = []
    for sample in plan.get("selected_samples", [])[:limit]:
        sample_id = sample["sample_id"]
        requests.append(
            ChartEvidenceRequest(
                image_path=Path(sample["image_path"]),
                sample_id=sample_id,
                question=sample.get("question", ""),
                ocr_row=ocr_rows.get(sample_id),
                record=records.get(sample_id),
            )
        )
    return requests


def evaluate(benchmark: str, requests: list[ChartEvidenceRequest]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provider in available_providers():
        for request in requests:
            report = provider.applicability(request)
            row = {
                "benchmark": benchmark,
                "provider": provider.name,
                "sample_id": request.sample_id,
                "verdict": report.verdict.value,
                "reason": report.reason,
                "items": None,
                "measured_items": None,
            }
            if report.supported:
                bundle = provider.extract(request)
                row["items"] = len(bundle.items)
                row["measured_items"] = len(bundle.measured_items)
                row["succeeded"] = bundle.succeeded
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for row in rows:
        key = f"{row['benchmark']}::{row['provider']}"
        bucket = summary.setdefault(
            key,
            {
                "benchmark": row["benchmark"],
                "provider": row["provider"],
                "n": 0,
                "verdicts": Counter(),
                "reasons": Counter(),
                "measured_items_total": 0,
            },
        )
        bucket["n"] += 1
        bucket["verdicts"][row["verdict"]] += 1
        bucket["reasons"][row["reason"]] += 1
        bucket["measured_items_total"] += row.get("measured_items") or 0

    for bucket in summary.values():
        bucket["verdicts"] = dict(bucket["verdicts"])
        bucket["reasons"] = dict(bucket["reasons"])
    return summary


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# M0 provider substitutability — applicability matrix",
        "",
        f"Created: {payload['created_at_utc']}",
        f"Paid calls: 0 | LLM HTTP: 0 | OCR HTTP: 0",
        "",
        "Free providers only. The paid OCR-ruler provider is excluded by "
        "`available_providers`, so this sweep cannot bill a call.",
        "",
        "## Applicability by benchmark",
        "",
        "| benchmark | provider | n | supported | unsupported | unknown | measured items |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for bucket in payload["summary"].values():
        verdicts = bucket["verdicts"]
        lines.append(
            f"| {bucket['benchmark']} | `{bucket['provider']}` | {bucket['n']} "
            f"| {verdicts.get('supported', 0)} | {verdicts.get('unsupported', 0)} "
            f"| {verdicts.get('unknown', 0)} | {bucket['measured_items_total']} |"
        )

    lines += ["", "## Why each provider declined", ""]
    for bucket in payload["summary"].values():
        declined = {k: v for k, v in bucket["reasons"].items()}
        lines.append(f"- **{bucket['benchmark']} / `{bucket['provider']}`**: {declined}")

    lines += [
        "",
        "## Limits",
        "",
        "- Applicability is a structural precondition, not a quality measure. A "
        "supported chart can still be extracted badly.",
        "- The paid OCR-ruler provider is untested here by construction. Its "
        "precondition is tick readability, which only a paid call can settle.",
        "- Item counts are not comparable across providers: the three emit "
        "different kinds of item (grid intersections, measured points, plotted "
        "marks), which is recorded per bundle as `item_semantics`.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--geosym-per-family", type=int, default=3)
    parser.add_argument("--plotqa-limit", type=int, default=24)
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    geosym = geosym_requests(args.geosym_per_family)
    plotqa = plotqa_requests(args.plotqa_limit)

    if geosym:
        rows += evaluate("geosym_bench", geosym)
    if plotqa:
        rows += evaluate("plotqa", plotqa)

    payload = {
        "schema_version": "evidence_provider_substitutability_v1",
        "created_at_utc": utc_now(),
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "providers": [p.name for p in available_providers()],
        "excluded_paid_providers": [
            p.name for p in available_providers(include_paid=True) if p.requires_paid_ocr
        ],
        "geosym_samples": len(geosym),
        "plotqa_samples": len(plotqa),
        "summary": summarize(rows),
        "rows": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "substitutability.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(payload)
    (args.output_dir / "substitutability.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
