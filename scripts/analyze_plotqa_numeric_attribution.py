#!/usr/bin/env python
"""Analyse the PlotQA numeric-attribution run against its pre-registered rule.

Zero paid calls: reads the runner's own result file.

Written **before** the confirmation numbers were visible, so "which test, on
which subset" is not a choice made once the outcome is known -- the failure mode
invariant 7 exists to prevent, and the one that cost this project its first
submission.

Decision rule, frozen in
``docs/PLOTQA_NUMERIC_ATTRIBUTION_FROZEN_PLAN_2026-09-20.md``
-----------------------------------------------------------
Answers are binary here, so the paired contrast is over per-sample correctness:

    H_N1  E0_plotqa_direct vs G_measured_raw, paired over samples answered in
          BOTH arms. McNemar's exact test on the discordant pairs, because a
          paired t-test on 0/1 outcomes with n=80 is the wrong instrument.
          Reported alongside the paired-proportion difference and its CI.

    H_N2  G_measured_raw vs Measured_flat_answer_baseline, same treatment.

    H_N3  Within the measured arm, does `both_label_share` associate with
          correctness? Mann-Whitney U on the covariate split by outcome, plus
          the two group medians. This is an association, never a cause.

A claim requires significance at alpha=0.05 **and** the stated direction. A
significant result in the opposite direction is reported as a refutation, never
renamed into a softer phrase.

Rows whose status is not ``success`` are excluded and reported separately. A
timeout or an empty body is a failed call, not a failed inference, and the
failure rate is not guaranteed equal across arms -- folding them in as incorrect
would bias the very contrast being measured (invariant 6).

Every stratum is reported, including strata that favour the baseline
(invariant 8).
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

BASELINE = "E0_plotqa_direct"
MEASURED = "G_measured_raw"
EVIDENCE_ONLY = "Measured_flat_answer_baseline"

ALPHA = 0.05
DEFAULT_DIR = ROOT / "outputs" / "paper_level_evidence_package" / "plotqa_numeric_attribution"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    document = load_json(path)
    provenance = {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "stage": document.get("stage"),
        "stage_status": document.get("stage_status"),
        "plan_sha256": document.get("plan_sha256"),
        "llm_http_attempts": document.get("actual_llm_http_attempts"),
        "llm_http_attempt_cap": document.get("llm_http_attempt_cap"),
        "cap_exhausted": document.get("cap_exhausted"),
        # Measured by the patched transport in the runner, not a constant.
        "ocr_http_attempts_measured": document.get("actual_ocr_http_attempts"),
        "records_sha256": document.get("records_sha256"),
        "model_provenance": document.get("model_provenance"),
    }
    return list(document.get("results") or []), provenance


def paired_binary_contrast(rows: list[dict[str, Any]], treatment: str,
                           baseline: str,
                           templates: tuple[str, ...] | None = None) -> dict[str, Any]:
    """McNemar's exact test on paired binary outcomes."""
    from scipy import stats

    def index(condition: str) -> dict[str, bool]:
        return {
            r["sample_id"]: bool(r.get("correct"))
            for r in rows
            if r["condition"] == condition
            and r.get("status") == "success"
            and r.get("correct") is not None
            and (templates is None or r.get("qa_template") in templates)
        }

    treated, control = index(treatment), index(baseline)
    paired = sorted(set(treated) & set(control))
    n = len(paired)
    if n < 2:
        return {"n_paired": n, "insufficient_data": True,
                "reading": "fewer than two paired samples; no test is meaningful"}

    both = sum(1 for s in paired if treated[s] and control[s])
    neither = sum(1 for s in paired if not treated[s] and not control[s])
    only_treatment = sum(1 for s in paired if treated[s] and not control[s])
    only_baseline = sum(1 for s in paired if not treated[s] and control[s])

    discordant = only_treatment + only_baseline
    # Exact binomial on the discordant pairs: with n=80 and few discordances the
    # chi-square approximation is not trustworthy.
    p_value = (stats.binomtest(only_treatment, discordant, 0.5).pvalue
               if discordant else 1.0)

    acc_t = sum(1 for s in paired if treated[s]) / n
    acc_b = sum(1 for s in paired if control[s]) / n
    diff = acc_t - acc_b
    # Standard error of a paired proportion difference.
    stderr = (math.sqrt(discordant) / n) if discordant else 0.0
    half = 1.96 * stderr

    significant = bool(p_value < ALPHA)
    positive = diff > 0
    return {
        "treatment": treatment,
        "baseline": baseline,
        "templates": list(templates) if templates else "all",
        "n_paired": n,
        "accuracy_treatment": acc_t,
        "accuracy_baseline": acc_b,
        "paired_difference": diff,
        "ci95": [diff - half, diff + half],
        "discordant_pairs": {"only_treatment": only_treatment,
                             "only_baseline": only_baseline},
        "concordant_pairs": {"both_correct": both, "both_wrong": neither},
        "mcnemar_exact_p": float(p_value),
        "significant_at_alpha": significant,
        "direction_positive": positive,
        "treatment_ahead_significantly": bool(significant and positive),
        "baseline_ahead_significantly": bool(significant and not positive),
    }


def label_completeness_association(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """H_N3: does label completeness associate with correctness in the measured arm?"""
    from scipy import stats

    correct, wrong = [], []
    for r in rows:
        if r["condition"] != MEASURED or r.get("status") != "success":
            continue
        share = r.get("both_label_share")
        if share is None:
            continue
        (correct if r.get("correct") else wrong).append(float(share))

    if len(correct) < 2 or len(wrong) < 2:
        return {"insufficient_data": True, "n_correct": len(correct),
                "n_wrong": len(wrong)}

    u_stat, p_value = stats.mannwhitneyu(correct, wrong, alternative="two-sided")
    median_correct = float(sorted(correct)[len(correct) // 2])
    median_wrong = float(sorted(wrong)[len(wrong) // 2])
    significant = bool(p_value < ALPHA)
    positive = median_correct > median_wrong
    return {
        "n_correct": len(correct),
        "n_wrong": len(wrong),
        "median_share_when_correct": median_correct,
        "median_share_when_wrong": median_wrong,
        "mann_whitney_u": float(u_stat),
        "p": float(p_value),
        "significant_at_alpha": significant,
        "direction_positive": positive,
        "supports_hypothesis": bool(significant and positive),
    }


def failure_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = collections.Counter(r["condition"] for r in rows)
    failures = collections.Counter(
        r["condition"] for r in rows if r.get("status") != "success")
    empty = collections.Counter(
        r["condition"] for r in rows
        if str(r.get("error", "")).startswith("empty_model_output"))
    detail = {
        condition: {
            "rows": totals[condition],
            "failed": failures.get(condition, 0),
            "empty_output": empty.get(condition, 0),
            "rate": (failures.get(condition, 0) / totals[condition])
            if totals[condition] else None,
        }
        for condition in sorted(totals)
    }
    counts = [v["failed"] for v in detail.values()]
    return {
        "by_condition": detail,
        "total_failed": sum(counts),
        "evenly_spread": len(set(counts)) <= 1,
        "reading": (
            "A timeout or empty body is a failed call, not a failed inference. "
            "These rows are excluded from the paired tests. If the rate differs "
            "across arms the exclusion protects the contrast; scoring them as "
            "incorrect would bias it (invariant 6)."
        ),
    }


def per_stratum(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy per arm within each official template. Every stratum is kept."""
    out: dict[str, Any] = {}
    templates = sorted({r.get("qa_template") for r in rows if r.get("qa_template")})
    for template in templates:
        block: dict[str, Any] = {}
        for condition in (BASELINE, MEASURED, EVIDENCE_ONLY):
            scored = [r for r in rows
                      if r["condition"] == condition
                      and r.get("qa_template") == template
                      and r.get("status") == "success"]
            correct = sum(1 for r in scored if r.get("correct"))
            block[condition] = {
                "scored": len(scored),
                "correct": correct,
                "accuracy": (correct / len(scored)) if scored else None,
            }
        out[template] = block
    return out


def run(result_path: Path) -> dict[str, Any]:
    rows, provenance = collect(result_path)

    h_n1 = paired_binary_contrast(rows, MEASURED, BASELINE)
    h_n2 = paired_binary_contrast(rows, MEASURED, EVIDENCE_ONLY)
    h_n3 = label_completeness_association(rows)

    n_samples = len({r["sample_id"] for r in rows})

    return {
        "schema_version": "plotqa_numeric_attribution_analysis_v1",
        "created_at_utc": utc_now(),
        "paid_calls_executed": 0,
        "input": provenance,
        "samples_seen": n_samples,
        "rows_seen": len(rows),
        "failures": failure_report(rows),
        "per_stratum": per_stratum(rows),
        "H_N1_numeric_penalty_reproduces": {
            **h_n1,
            "hypothesis": (
                "On the current extractor, G_measured_raw scores BELOW "
                "E0_plotqa_direct on this numeric roster."),
            "verdict": _verdict_h_n1(h_n1),
        },
        "H_N2_evidence_not_independently_sufficient": {
            **h_n2,
            "hypothesis": (
                "Measured_flat_answer_baseline (evidence, no image) scores below "
                "G_measured_raw."),
            "verdict": _verdict_h_n2(h_n2),
        },
        "H_N3_label_completeness_associates": {
            **h_n3,
            "hypothesis": (
                "Within the measured arm, a higher share of facts carrying both "
                "labels associates with correctness."),
            "verdict": (
                "association_present" if h_n3.get("supports_hypothesis")
                else "association_absent_or_inverted"),
        },
        "attribution_not_victory": (
            "This run measures position, not cause, and contains no hypothesis "
            "that GeoSym beats the baseline on PlotQA. A favourable H_N1 result "
            "refutes a stated expectation; it is not a headline claim, and the "
            "roster is two templates x numeric answers only."
        ),
        "interpretation_limits": [
            "Roster is qa_template in {arithmetic, min_max} AND answer_type == "
            "numeric. Not representative of PlotQA overall.",
            "Six candidate mechanisms were falsified offline before this run; "
            "the surviving object was a stratification, not an explanation.",
            "N=80 puts the standard error of a paired proportion difference near "
            "0.05; smaller differences are not resolvable.",
            "Baselines were re-run in the same batch rather than reused, because "
            "the cached rows predate the M0' repairs.",
            "H_N3 is an association within one arm. It cannot establish that "
            "raising label completeness would raise accuracy.",
        ],
    }


def _verdict_h_n1(block: dict[str, Any]) -> str:
    if block.get("insufficient_data"):
        return "insufficient_data"
    if block.get("baseline_ahead_significantly"):
        return "numeric_penalty_confirmed_on_current_extractor"
    if block.get("treatment_ahead_significantly"):
        return "REFUTED_measured_arm_ahead_significantly"
    return "no_significant_difference_penalty_not_reproduced_at_this_n"


def _verdict_h_n2(block: dict[str, Any]) -> str:
    if block.get("insufficient_data"):
        return "insufficient_data"
    if block.get("treatment_ahead_significantly"):
        return "evidence_alone_insufficient_confirmed"
    if block.get("baseline_ahead_significantly"):
        return "REFUTED_evidence_only_arm_ahead_significantly"
    return "no_significant_difference_at_this_n"


def render(results: dict[str, Any]) -> str:
    inp = results["input"]
    lines = [
        "# PlotQA numeric attribution — analysis (zero cost)", "",
        f"Created: {results['created_at_utc']}",
        f"Input: `{inp['path']}` (sha256 `{inp['sha256'][:16]}`)",
        f"Stage: {inp['stage']} / {inp['stage_status']} | "
        f"plan `{str(inp['plan_sha256'])[:16]}`",
        f"LLM HTTP attempts: **{inp['llm_http_attempts']} / "
        f"{inp['llm_http_attempt_cap']}** | cap exhausted: {inp['cap_exhausted']}",
        f"OCR HTTP attempts (measured, not asserted): "
        f"**{inp['ocr_http_attempts_measured']}**",
        f"Samples: {results['samples_seen']} | rows: {results['rows_seen']}", "",
    ]

    lines += ["## Failed calls (excluded from tests)", "",
              "| condition | rows | failed | empty | rate |",
              "|---|---:|---:|---:|---:|"]
    for condition, stats in results["failures"]["by_condition"].items():
        rate = "--" if stats["rate"] is None else f"{stats['rate']:.1%}"
        lines.append(f"| `{condition}` | {stats['rows']} | {stats['failed']} | "
                     f"{stats['empty_output']} | {rate} |")
    lines += ["", f"Evenly spread across arms: **{results['failures']['evenly_spread']}**",
              "", results["failures"]["reading"], ""]

    lines += ["## Accuracy by official template (every stratum reported)", "",
              "| template | baseline | measured+image | evidence only |",
              "|---|---:|---:|---:|"]
    for template, block in results["per_stratum"].items():
        def cell(condition: str) -> str:
            stats = block[condition]
            if stats["accuracy"] is None:
                return "--"
            return f"{stats['accuracy']:.3f} ({stats['correct']}/{stats['scored']})"
        lines.append(f"| {template} | {cell(BASELINE)} | {cell(MEASURED)} | "
                     f"{cell(EVIDENCE_ONLY)} |")
    lines.append("")

    for key, title in (
        ("H_N1_numeric_penalty_reproduces",
         "H_N1 — does the numeric penalty reproduce on the current extractor?"),
        ("H_N2_evidence_not_independently_sufficient",
         "H_N2 — is the evidence independently sufficient?"),
    ):
        block = results[key]
        lines += [f"## {title}", "", f"*{block['hypothesis']}*", ""]
        if block.get("insufficient_data"):
            lines += [f"Insufficient paired data (n={block['n_paired']}).", ""]
            continue
        disc = block["discordant_pairs"]
        conc = block["concordant_pairs"]
        lines += [
            f"- contrast: `{block['treatment']}` vs `{block['baseline']}`",
            f"- paired n: {block['n_paired']}",
            f"- accuracy: {block['accuracy_treatment']:.3f} vs "
            f"{block['accuracy_baseline']:.3f}",
            f"- paired difference: **{block['paired_difference']:+.3f}** "
            f"(95% CI {block['ci95'][0]:+.3f} to {block['ci95'][1]:+.3f})",
            f"- discordant: treatment-only {disc['only_treatment']}, "
            f"baseline-only {disc['only_baseline']}",
            f"- concordant: both right {conc['both_correct']}, "
            f"both wrong {conc['both_wrong']}",
            f"- McNemar exact p: {block['mcnemar_exact_p']:.4g}",
            f"- **verdict: {block['verdict']}**", "",
        ]

    h3 = results["H_N3_label_completeness_associates"]
    lines += ["## H_N3 — does label completeness associate with correctness?", "",
              f"*{h3['hypothesis']}*", ""]
    if h3.get("insufficient_data"):
        lines += [f"Insufficient data (correct={h3['n_correct']}, "
                  f"wrong={h3['n_wrong']}).", ""]
    else:
        lines += [
            f"- median both-label share when correct: "
            f"{h3['median_share_when_correct']:.3f} (n={h3['n_correct']})",
            f"- median both-label share when wrong: "
            f"{h3['median_share_when_wrong']:.3f} (n={h3['n_wrong']})",
            f"- Mann-Whitney p: {h3['p']:.4g}",
            f"- **verdict: {h3['verdict']}**", "",
        ]

    lines += ["## What this run is not", "", results["attribution_not_victory"], "",
              "## Limits", ""]
    lines += [f"- {limit}" for limit in results["interpretation_limits"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path,
                        default=DEFAULT_DIR / "paid_result.confirmation.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    results = run(args.result)
    output_dir = args.output_dir or args.result.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.result.stem.replace("paid_result.", "")
    (output_dir / f"attribution_analysis.{stem}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = render(results)
    (output_dir / f"attribution_analysis.{stem}.md").write_text(
        markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
