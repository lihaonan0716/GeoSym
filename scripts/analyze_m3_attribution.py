#!/usr/bin/env python
"""Analyse the M3 attribution experiment against its pre-registered criteria.

Zero paid calls: reads the runner's own result files, merging shards if present.

This script exists **before** the confirmation run, not after. The decision rule
it implements was frozen in the plan, so that "which test, on which subset" is
not a choice made once the numbers are visible -- the failure mode invariant 7
exists to prevent.

Decision rule, per the frozen plan
----------------------------------
A fixed point-difference threshold is not usable here. A cross-batch probe
measured per-cell noise at sd 31.9 on this benchmark, which puts the standard
error of an arm mean at ~3.0 for n=110 and ~11.3 for n=8 -- an order of
magnitude above the 1.0-point threshold originally written down. So:

    H_A (layering)   paired test of E5_raw - E6_flat_single over samples scored
                     in BOTH arms. Layering is an attributable contribution only
                     if the paired test is significant at alpha=0.05 AND the
                     effect is positive. Otherwise report "no worse than a
                     single pass, and auditable" -- never as a layering gain.

    H_B (form-first) paired test of E7_form_first - E5_raw on the transcendental
                     families. If not significant and positive, the
                     fit-instead-of-recognise hypothesis is unsupported.

Effect size and a 95% confidence interval are reported alongside p, because a p
value alone says nothing about magnitude. Both the parametric (paired t) and the
rank-based (Wilcoxon) result are reported: on n=110 with bounded scores they
should agree, and if they disagree that disagreement is the finding.

Rows whose model output was empty are excluded from scoring and reported
separately. An empty body is a failed call, not a failed inference, and the
truncation rate is not guaranteed equal across arms -- so folding those rows in
as zeros would bias the very contrast being measured (invariant 6).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REFERENCE_CONDITION = "E5_raw"
FLAT_CONDITION = "E6_flat_single"
FORM_FIRST_CONDITION = "E7_form_first"

# Families where the layer profile localised the `expression` decay. Frozen here
# because H_B is stated over them; selecting them after seeing results would be
# the post-hoc subset selection invariant 7 forbids.
TRANSCENDENTAL_FAMILIES = ("exponential", "logarithmic", "rational", "piecewise_linear")

ALPHA = 0.05
DEFAULT_DIR = ROOT / "outputs" / "m3_attribution_pilot"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows(directory: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge the unsharded result and every shard file found.

    A shard that is missing is not silently skipped: its absence changes the
    sample count, and a partial merge reported as a full run would overstate
    coverage.
    """
    sources: list[dict[str, Any]] = []
    single = directory / "paid_result.json"
    if single.is_file():
        sources.append({"path": single, "document": load_json(single)})
    for path in sorted(directory.glob("paid_result.shard*of*.json")):
        sources.append({"path": path, "document": load_json(path)})
    if not sources:
        raise FileNotFoundError(f"no result files under {directory}")

    identities = {s["document"].get("plan_sha256") for s in sources}
    if len(identities) > 1:
        raise ValueError(
            "result files disagree on plan_sha256; refusing to merge different "
            f"experiments: {sorted(str(i)[:12] for i in identities)}")

    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for source in sources:
        document = source["document"]
        rows.extend(document.get("results") or [])
        path = source["path"]
        try:
            display = str(path.relative_to(ROOT))
        except ValueError:
            # A result directory outside the repo is legitimate (a scratch copy,
            # a test fixture); reporting the absolute path beats refusing to run.
            display = str(path)
        provenance.append({
            "path": display,
            "sha256": hashlib.sha256(source["path"].read_bytes()).hexdigest(),
            "stage_status": document.get("stage_status"),
            "rows": len(document.get("results") or []),
            "attempts": document.get("actual_llm_http_attempts"),
            "cap_exhausted": document.get("cap_exhausted"),
        })
    return rows, provenance


def paired_contrast(rows: list[dict[str, Any]], treatment: str, baseline: str,
                    families: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Paired treatment-minus-baseline contrast over samples scored in both arms."""
    from scipy import stats

    def index(condition: str) -> dict[str, float]:
        return {r["sample_id"]: r["total_score"] for r in rows
                if r["condition"] == condition
                and r.get("status") == "success"
                and r.get("total_score") is not None
                and (families is None or r["sample_id"].split("-")[0] in families)}

    treated, control = index(treatment), index(baseline)
    paired_ids = sorted(set(treated) & set(control))
    deltas = [treated[i] - control[i] for i in paired_ids]
    n = len(deltas)
    if n < 2:
        return {"n_paired": n, "insufficient_data": True,
                "reading": "fewer than two paired samples; no test is meaningful"}

    mean_delta = sum(deltas) / n
    variance = sum((d - mean_delta) ** 2 for d in deltas) / (n - 1)
    sd = math.sqrt(variance)
    stderr = sd / math.sqrt(n)
    t_stat, t_p = stats.ttest_rel(
        [treated[i] for i in paired_ids], [control[i] for i in paired_ids])
    try:
        w_stat, w_p = stats.wilcoxon([treated[i] for i in paired_ids],
                                     [control[i] for i in paired_ids])
    except ValueError:
        # Wilcoxon rejects an all-zero difference vector.
        w_stat, w_p = float("nan"), float("nan")
    crit = stats.t.ppf(1 - ALPHA / 2, n - 1)

    significant = bool(t_p < ALPHA)
    positive = mean_delta > 0
    return {
        "treatment": treatment,
        "baseline": baseline,
        "families": list(families) if families else "all",
        "n_paired": n,
        "mean_delta": mean_delta,
        "sd_delta": sd,
        "stderr": stderr,
        "ci95": [mean_delta - crit * stderr, mean_delta + crit * stderr],
        # Cohen's d for paired data: mean over sd of the differences.
        "cohens_d_paired": (mean_delta / sd) if sd > 0 else None,
        "paired_t": {"t": float(t_stat), "p": float(t_p)},
        "wilcoxon": {"statistic": float(w_stat), "p": float(w_p)},
        "significant_at_alpha": significant,
        "direction_positive": positive,
        "supports_hypothesis": bool(significant and positive),
        "wins": sum(1 for d in deltas if d > 0),
        "losses": sum(1 for d in deltas if d < 0),
        "ties": sum(1 for d in deltas if d == 0),
    }


def empty_output_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition = collections.Counter(
        r["condition"] for r in rows
        if str(r.get("error", "")).startswith("empty_model_output"))
    totals = collections.Counter(r["condition"] for r in rows)
    detail = {
        condition: {
            "rows": totals[condition],
            "empty": by_condition.get(condition, 0),
            "rate": (by_condition.get(condition, 0) / totals[condition])
            if totals[condition] else None,
        }
        for condition in sorted(totals)
    }
    rates = [v["empty"] for v in detail.values()]
    return {
        "by_condition": detail,
        "total_empty": sum(rates),
        "evenly_spread": len(set(rates)) <= 1,
        "reading": (
            "An empty body is a failed call, not a failed inference. These rows "
            "are excluded from the paired tests. If the rate differs across "
            "arms, the exclusion protects the contrast; folding them in as "
            "zeros would bias it (invariant 6)."
        ),
    }


def run(directory: Path) -> dict[str, Any]:
    rows, provenance = collect_rows(directory)
    scored = [r for r in rows if r.get("status") == "success"
              and r.get("total_score") is not None]

    per_condition = {}
    for condition in sorted({r["condition"] for r in rows}):
        values = [r["total_score"] for r in scored if r["condition"] == condition]
        per_condition[condition] = {
            "n_scored": len(values),
            "mean_total_score": (sum(values) / len(values)) if values else None,
        }

    h_a = paired_contrast(scored, REFERENCE_CONDITION, FLAT_CONDITION)
    h_b = paired_contrast(scored, FORM_FIRST_CONDITION, REFERENCE_CONDITION,
                          families=TRANSCENDENTAL_FAMILIES)

    n_samples = len({r["sample_id"] for r in rows})
    underpowered = n_samples < 30

    return {
        "schema_version": "m3_attribution_analysis_v1",
        "created_at_utc": utc_now(),
        "paid_calls_executed": 0,
        "inputs": provenance,
        "samples_seen": n_samples,
        "rows_seen": len(rows),
        "per_condition": per_condition,
        "empty_output": empty_output_report(rows),
        "H_A_layering": {
            **h_a,
            "hypothesis": "Given identical M0 evidence, the three-layer chain "
                          "scores higher than a single pass.",
            "verdict": (
                "layering_is_attributable" if h_a.get("supports_hypothesis")
                else "no_worse_than_single_pass_and_auditable"),
            "falsification_note": (
                "If not significant and positive, the plan requires reporting "
                "'no worse than a single pass, and auditable', never a layering "
                "gain."
            ),
        },
        "H_B_form_first": {
            **h_b,
            "hypothesis": "On transcendental families, naming the function form "
                          "before solving for parameters beats the standard L3.",
            "verdict": (
                "form_first_supported" if h_b.get("supports_hypothesis")
                else "fit_instead_of_recognise_unsupported"),
        },
        "power_warning": (
            f"{n_samples} samples: the standard error of an arm mean is about "
            "11 points at n=8 and about 3 at n=110. Treat this as a path check, "
            "not an effect estimate." if underpowered else None),
        "interpretation_limits": [
            "Both hypotheses were pre-registered with their falsification "
            "signals before any paid call; this script implements those rules "
            "rather than choosing a test after seeing the numbers.",
            "All three arms receive identical M0 evidence, so each contrast "
            "isolates one factor: E6 removes the layering, E7 swaps only the "
            "L3 template.",
            "E5_raw is re-run in the same batch as E6/E7 rather than reused "
            "from disk, so no contrast is confounded by between-run drift.",
            "Scores use the ten fixed, unequally weighted dimensions from "
            "config/default_config.yaml. Coordinates are M0 intermediate data "
            "and are deliberately not scored.",
        ],
    }


def render(results: dict[str, Any]) -> str:
    lines = [
        "# M3 attribution analysis (zero cost)", "",
        f"Created: {results['created_at_utc']}",
        f"Samples: {results['samples_seen']} | rows: {results['rows_seen']}", "",
    ]
    if results["power_warning"]:
        lines += [f"> **{results['power_warning']}**", ""]

    lines += ["## Inputs", "",
              "| file | status | rows | attempts | cap exhausted |",
              "|---|---|---:|---:|---|"]
    for source in results["inputs"]:
        lines.append(f"| `{source['path']}` | {source['stage_status']} | "
                     f"{source['rows']} | {source['attempts']} | "
                     f"{source['cap_exhausted']} |")

    lines += ["", "## Arm means", "", "| condition | n scored | mean |",
              "|---|---:|---:|"]
    for condition, stats in results["per_condition"].items():
        mean = stats["mean_total_score"]
        lines.append(f"| {condition} | {stats['n_scored']} | "
                     f"{mean:.2f} |" if mean is not None
                     else f"| {condition} | {stats['n_scored']} | -- |")

    empty = results["empty_output"]
    lines += ["", "## Empty model output (excluded from tests)", "",
              "| condition | rows | empty | rate |", "|---|---:|---:|---:|"]
    for condition, stats in empty["by_condition"].items():
        rate = "--" if stats["rate"] is None else f"{stats['rate']:.1%}"
        lines.append(f"| {condition} | {stats['rows']} | {stats['empty']} | {rate} |")
    lines += ["", f"Evenly spread across arms: **{empty['evenly_spread']}**", "",
              empty["reading"], ""]

    for key, title in (("H_A_layering", "H_A - does the layering earn its cost?"),
                       ("H_B_form_first", "H_B - does form-first reverse the penalty?")):
        block = results[key]
        lines += [f"## {title}", "", f"*{block['hypothesis']}*", ""]
        if block.get("insufficient_data"):
            lines += [f"Insufficient paired data (n={block['n_paired']}).", ""]
            continue
        lines += [
            f"- contrast: `{block['treatment']}` minus `{block['baseline']}`"
            f" over {block['families']}",
            f"- paired n: {block['n_paired']}",
            f"- mean delta: **{block['mean_delta']:+.2f}** "
            f"(95% CI {block['ci95'][0]:+.2f} to {block['ci95'][1]:+.2f})",
            f"- paired t: t={block['paired_t']['t']:.3f}, p={block['paired_t']['p']:.4g}",
            f"- Wilcoxon: p={block['wilcoxon']['p']:.4g}",
            f"- Cohen's d (paired): "
            + (f"{block['cohens_d_paired']:.3f}" if block['cohens_d_paired'] is not None else "--"),
            f"- win/loss/tie: {block['wins']}/{block['losses']}/{block['ties']}",
            f"- **verdict: {block['verdict']}**", "",
        ]

    lines += ["## Limits", ""]
    lines += [f"- {limit}" for limit in results["interpretation_limits"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    results = run(args.result_dir)
    output_dir = args.output_dir or args.result_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "attribution_analysis.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = render(results)
    (output_dir / "attribution_analysis.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
