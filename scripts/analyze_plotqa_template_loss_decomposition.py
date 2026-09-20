#!/usr/bin/env python
"""Decompose the PlotQA measured-arm loss by official template and bucket.

Zero paid calls: every input is a frozen artifact already on disk.

Why this decomposition rather than the earlier four-quadrant one: the quadrant
split was cut on "is the evidence independently sufficient", so each cell mixed
five task templates whose mechanisms differ. Cutting on ``qa_template`` -- an
*official* PlotQA annotation field, not a variable we construct -- shows the loss
is concentrated rather than uniform.

Two views, both paired on ``sample_id``:

1. **By template.** Where the loss actually sits.
2. **By bucket**, using the official ``reachable`` flag:
   - rescue   : baseline wrong, answer reachable from evidence
   - risk     : baseline right, answer not reachable
   - redundant: baseline right, answer reachable

The redundant bucket is the interesting one. A loss there cannot be blamed on
missing evidence -- the answer was reachable and the image alone already
sufficed -- which points at operand selection rather than extraction.

Selecting strata by an official annotation field is not the post-hoc subset
selection invariant 7 prohibits: no cell is chosen by an outcome variable, and
every template is reported whether it helps the claim or not.
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

PACKAGE = ROOT / "outputs" / "paper_level_evidence_package"
BASELINES = PACKAGE / "plotqa_public_500" / "direct_baselines" / "paid_result.json"
MEASURED = PACKAGE / "plotqa_public_500" / "measured_confirmation" / "paid_result.json"
COMPLEMENTARITY = (PACKAGE / "plotqa_evidence_complementarity"
                   / "evidence_complementarity.json")

# PlotQA conditions. Never `E5`/`E5_raw`: those are GeoSym-Bench condition
# strings, and the two rosters must not share identifiers.
BASELINE_CONDITION = "E0_plotqa_direct"
MEASURED_CONDITION = "G_measured_raw"

TEMPLATE_ORDER = ("arithmetic", "min_max", "compound", "comparison", "data_retrieval")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def correctness(document: Any, condition: str) -> dict[str, bool]:
    return {row["sample_id"]: bool(row["correct"])
            for row in document.get("results") or []
            if row.get("condition") == condition}


def run(output_dir: Path) -> dict[str, Any]:
    baseline = correctness(load_json(BASELINES), BASELINE_CONDITION)
    measured = correctness(load_json(MEASURED), MEASURED_CONDITION)
    annotations = {row["sample_id"]: row
                   for row in load_json(COMPLEMENTARITY)["rows"]}

    paired = sorted(set(baseline) & set(measured) & set(annotations))

    by_template: dict[str, dict[str, Any]] = {}
    for template in TEMPLATE_ORDER:
        subset = [s for s in paired if annotations[s]["qa_template"] == template]
        if not subset:
            continue
        base_correct = sum(baseline[s] for s in subset)
        meas_correct = sum(measured[s] for s in subset)
        by_template[template] = {
            "n": len(subset),
            "baseline_correct": base_correct,
            "measured_correct": meas_correct,
            "baseline_accuracy": base_correct / len(subset),
            "measured_accuracy": meas_correct / len(subset),
            "net": meas_correct - base_correct,
        }

    def bucket_of(sample: str) -> str:
        reachable = bool(annotations[sample]["reachable"])
        if baseline[sample]:
            return "redundant" if reachable else "risk"
        return "rescue" if reachable else "dead"

    by_bucket: dict[str, dict[str, Any]] = {}
    bucket_by_template: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for bucket in ("rescue", "risk", "redundant", "dead"):
        subset = [s for s in paired if bucket_of(s) == bucket]
        if not subset:
            continue
        kept = sum(measured[s] for s in subset)
        by_bucket[bucket] = {
            "n": len(subset),
            "measured_correct": kept,
            "rate": kept / len(subset),
            "lost": len(subset) - kept if bucket in ("redundant", "risk") else None,
            "converted": kept if bucket in ("rescue", "dead") else None,
        }
        counter = collections.Counter(annotations[s]["qa_template"] for s in subset
                                     if not measured[s])
        bucket_by_template[bucket] = dict(sorted(counter.items()))

    total_net = sum(t["net"] for t in by_template.values())
    concentrated = sum(by_template[t]["net"] for t in ("arithmetic", "min_max")
                       if t in by_template)
    redundant_lost = by_bucket.get("redundant", {}).get("lost") or 0
    measured_total = sum(measured[s] for s in paired)

    results = {
        "schema_version": "plotqa_template_loss_decomposition_v1",
        "created_at_utc": utc_now(),
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "inputs": {
            "baselines": str(BASELINES.relative_to(ROOT)),
            "baselines_sha256": sha256_of_file(BASELINES),
            "measured": str(MEASURED.relative_to(ROOT)),
            "measured_sha256": sha256_of_file(MEASURED),
            "complementarity": str(COMPLEMENTARITY.relative_to(ROOT)),
        },
        "conditions": {"baseline": BASELINE_CONDITION, "measured": MEASURED_CONDITION},
        "paired_samples": len(paired),
        "headline": {
            "baseline_accuracy": sum(baseline[s] for s in paired) / len(paired),
            "measured_accuracy": measured_total / len(paired),
            "net": total_net,
        },
        "by_template": by_template,
        "by_bucket": by_bucket,
        "lost_rows_by_bucket_and_template": dict(bucket_by_template),
        "concentration": {
            "arithmetic_plus_min_max_net": concentrated,
            "total_net": total_net,
            "share_of_loss": (concentrated / total_net) if total_net else None,
        },
        "ceiling_if_redundant_bucket_repaired": {
            "redundant_lost": redundant_lost,
            "measured_accuracy_now": measured_total / len(paired),
            "measured_accuracy_if_recovered":
                (measured_total + redundant_lost) / len(paired),
            "baseline_accuracy": sum(baseline[s] for s in paired) / len(paired),
            "reading": (
                "Recovering the entire redundant bucket only reaches parity with "
                "the image-only baseline. Beating it additionally requires "
                "raising rescue conversion, which is lowest exactly where the "
                "loss is largest."
            ),
        },
        "interpretation_limits": [
            "Strata are official PlotQA `qa_template` values, not outcome-derived "
            "subsets. Every template is reported, including those that favour "
            "the baseline.",
            "These paid rows predate the M0' defect repairs, so they characterise "
            "the extractor as it was when the calls ran, not the current one.",
            "Scores here use the tolerance recorded in the artifact. The strict "
            "year rule is applied separately by "
            "scripts/rescore_year_answers_offline.py and widens this gap.",
            "A bucket label uses the official `reachable` flag, which is computed "
            "from evidence and ground truth; it measures whether the answer was "
            "obtainable, not whether the model should have found it.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "template_loss_decomposition.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def render(results: dict[str, Any]) -> str:
    head = results["headline"]
    lines = [
        "# PlotQA loss decomposition by official template (zero cost)", "",
        f"Created: {results['created_at_utc']}",
        f"Paid calls: {results['paid_calls_executed']}",
        f"Paired samples: {results['paired_samples']}",
        f"Conditions: `{results['conditions']['baseline']}` vs "
        f"`{results['conditions']['measured']}`", "",
        f"Headline: baseline {head['baseline_accuracy']:.3f} vs "
        f"measured {head['measured_accuracy']:.3f} (net {head['net']:+d})", "",
        "## 1. By official template", "",
        "| qa_template | n | baseline | measured | net |", "|---|---:|---:|---:|---:|",
    ]
    for template, s in results["by_template"].items():
        lines.append(f"| {template} | {s['n']} | {s['baseline_accuracy']:.3f} | "
                     f"{s['measured_accuracy']:.3f} | {s['net']:+d} |")

    conc = results["concentration"]
    share = f"{conc['share_of_loss']:.0%}" if conc["share_of_loss"] else "n/a"
    lines += [
        "",
        f"`arithmetic` + `min_max` account for {conc['arithmetic_plus_min_max_net']:+d} "
        f"of {conc['total_net']:+d} ({share} of the loss).", "",
        "## 2. By bucket", "",
        "| bucket | n | measured correct | rate |", "|---|---:|---:|---:|",
    ]
    for bucket, s in results["by_bucket"].items():
        lines.append(f"| {bucket} | {s['n']} | {s['measured_correct']} | {s['rate']:.3f} |")

    ceiling = results["ceiling_if_redundant_bucket_repaired"]
    lines += [
        "", "## 3. Ceiling if the redundant bucket were fully repaired", "",
        f"- redundant rows lost: **{ceiling['redundant_lost']}**",
        f"- measured now: {ceiling['measured_accuracy_now']:.3f}",
        f"- measured if recovered: **{ceiling['measured_accuracy_if_recovered']:.3f}**",
        f"- baseline: {ceiling['baseline_accuracy']:.3f}", "",
        ceiling["reading"], "", "## Limits", "",
    ]
    lines += [f"- {limit}" for limit in results["interpretation_limits"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=PACKAGE / "plotqa_template_loss_decomposition")
    args = parser.parse_args(argv)
    results = run(args.output_dir)
    markdown = render(results)
    (args.output_dir / "template_loss_decomposition.md").write_text(
        markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
