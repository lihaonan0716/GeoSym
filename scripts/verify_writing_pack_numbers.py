#!/usr/bin/env python
"""Check every number in the writing pack against its artifact.

Zero paid calls, zero network. Exits non-zero when the prose and the data
disagree.

Why this exists
---------------
The writing pack is what the manuscript will be drafted from, so a stale figure
in it becomes a stale figure in the paper. A manual review found four defects
that a reader would not have caught:

* the table header claimed a uniform n=240 while n ranged 229-240;
* three different statistics (+10.82, +10.66, +10.83) were all circulating as
  "C1", differing by which samples enter;
* a "2.0x cost" figure counted the judge call, which evaluation runs and
  deployments do not -- the architectural cost is 3 reasoning calls against 1;
* an aggregate coverage gain concealed that it came from one repaired subtype
  while two others drifted down.

Each of those is now an assertion here. The prose may be rewritten freely; the
numbers inside it cannot drift without this failing.

Scope
-----
This checks the figures it knows about. It does not read English, so a claim
phrased in words it does not test can still be wrong -- the same limit as
``verify_m0_prose_matches_code.py``.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACK = ROOT / "docs" / "PAPER_WRITING_EVIDENCE_PACK_2026-09-23.md"
BLUEPRINT = ROOT / "docs" / "PAPER_BLUEPRINT.md"
BENCH = (ROOT / "outputs" / "paper_level_evidence_package"
         / "geosym_bench_controlled_glm46v_240x7_rerun" / "full_240"
         / "paid_result.json")
PLOTQA_M0 = (ROOT / "outputs" / "paper_level_evidence_package"
             / "plotqa_m0_chain_wiring_bar_repair"
             / "m0_chain_wiring_verification.json")
PLOTQA_ATTR = (ROOT / "outputs" / "paper_level_evidence_package"
               / "plotqa_numeric_attribution"
               / "attribution_analysis.confirmation.json")
PLOTQA_ATTR_PAID = (ROOT / "outputs" / "paper_level_evidence_package"
                    / "plotqa_numeric_attribution"
                    / "paid_result.confirmation.json")
R3_3 = (ROOT / "outputs" / "m3_attribution_pilot" / "json_repair_rescore"
        / "rescore_report.json")

#: Arms that share the result file with the live conditions but take no part
#: in any reported contrast. They are reported where present and must never
#: gate a complete case: a retired arm's failure would otherwise drop samples
#: from comparisons it does not enter. See the C2 wording note in NARRATIVE.md.
RETIRED_CONDITIONS = frozenset({"E5"})
R3_3_PAID = ROOT / "outputs" / "m3_attribution_pilot" / "paid_result.json"
LAYER_PROFILE = (ROOT / "reports" / "layer_capability_profile"
                 / "layer_capability_profile.json")
SUBSTITUTABILITY = (ROOT / "outputs" / "paper_level_evidence_package"
                    / "evidence_provider_substitutability" / "substitutability.json")
DEFECT_AUDIT = ROOT / "outputs" / "json_parse_defect_audit" / "audit.json"

OBJECTIVE_DIMENSIONS = (
    "intercepts", "extrema", "predictions", "rate_of_change", "monotonicity",
    "symmetry", "expression", "domain", "range")

_BANNER = re.compile(r"^=== (.*?) ===$", re.MULTILINE)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bench_condition_stats() -> dict[str, dict[str, Any]]:
    rows = [r for r in load(BENCH)["results"]
            if r.get("total_score") is not None and r.get("dimension_scores")]
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    stats = {}
    for condition, subset in grouped.items():
        zeros = sum(1 for r in subset
                    if all(float(r["dimension_scores"].get(d) or 0) == 0.0
                           for d in OBJECTIVE_DIMENSIONS))
        stats[condition] = {
            "n": len(subset),
            "mean": st.mean(r["total_score"] for r in subset),
            "all_zero": zeros,
        }
    return stats


def c1_variants() -> dict[str, float]:
    """Statistics for GeoSym minus the strongest baseline.

    C1 is locked to the paired mean difference (2026-09-23). The other two are
    computed so the pack can name each where it belongs, and so swapping one
    for another fails a check rather than passing silently.
    """
    from scipy import stats

    rows = [r for r in load(BENCH)["results"] if r.get("total_score") is not None]
    by_condition: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]][row["sample_id"]] = row["total_score"]
    geosym, baseline = by_condition["E5_raw"], by_condition["E2"]
    shared = sorted(set(geosym) & set(baseline))
    # The complete case is over the LIVE conditions. Intersecting every
    # condition in the file included the retired E5 arm, whose single failure
    # on `cubic-clean_render_0025` dropped a sample all six live conditions
    # scored -- shrinking the complete case to 224 and moving both this
    # statistic and every C2 layer delta computed on the same basis.
    live = {c: v for c, v in by_condition.items()
            if c not in RETIRED_CONDITIONS}
    every_condition = set.intersection(*(set(v) for v in live.values()))

    diffs = [geosym[s] - baseline[s] for s in shared]
    mean = st.mean(diffs)
    sd = st.stdev(diffs)
    se = sd / len(diffs) ** 0.5
    crit = float(stats.t.ppf(0.975, len(diffs) - 1))
    t_stat, t_p = stats.ttest_rel([geosym[s] for s in shared],
                                  [baseline[s] for s in shared])
    _w, w_p = stats.wilcoxon(diffs)
    wins = [d for d in diffs if d > 0]
    losses = [d for d in diffs if d < 0]

    return {
        "difference_of_means": st.mean(geosym.values()) - st.mean(baseline.values()),
        "paired": mean,
        "paired_n": float(len(shared)),
        "paired_ci_low": mean - crit * se,
        "paired_ci_high": mean + crit * se,
        "paired_t": float(t_stat),
        "paired_t_p": float(t_p),
        "paired_wilcoxon_p": float(w_p),
        "paired_cohens_dz": mean / sd,
        "paired_wins": float(len(wins)),
        "paired_losses": float(len(losses)),
        "paired_ties": float(len(diffs) - len(wins) - len(losses)),
        "paired_median": st.median(diffs),
        "mean_win_magnitude": st.mean(wins),
        "mean_loss_magnitude": st.mean(losses),
        "complete_case": st.mean(geosym[s] - baseline[s] for s in every_condition),
        "complete_case_n": float(len(every_condition)),
    }


def c1_normalisation_bound() -> float:
    """Paired C1 margin if every affected row on both arms were normalised.

    The generous direction: it lifts baseline rows that lost JSON-gated
    dimensions to full marks on those dimensions. GeoSym has no such rows, so
    the bound can only move against us -- which is what makes it a bound.
    """
    from app.reasoning_pipeline import load_layer_config

    layer_config = load_layer_config()
    effective: dict[str, float] = {}
    for _layer, config in layer_config.items():
        weight = config.get("weight", 0)
        for dimension, dimension_weight in (config.get("dimensions") or {}).items():
            effective[dimension] = effective.get(dimension, 0) + weight * dimension_weight
    total = sum(effective.values())
    gated = ("predictions", "rate_of_change", "monotonicity")
    gated_weight = sum(effective[g] for g in gated) / total

    rows = [r for r in load(BENCH)["results"]
            if r.get("total_score") is not None and r.get("dimension_scores")]
    by_condition: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]][row["sample_id"]] = row

    def carries_signature(row: dict[str, Any]) -> bool:
        scores = row["dimension_scores"]
        return (all(float(scores.get(k) or 0) == 0 for k in gated)
                and any(float(scores.get(k) or 0) > 0
                        for k in ("expression", "domain", "range")))

    geosym, baseline = by_condition["E5_raw"], by_condition["E2"]
    shared = sorted(set(geosym) & set(baseline))
    adjusted = []
    for sample in shared:
        g_score = geosym[sample]["total_score"] + (
            gated_weight * 100 if carries_signature(geosym[sample]) else 0)
        b_score = baseline[sample]["total_score"] + (
            gated_weight * 100 if carries_signature(baseline[sample]) else 0)
        adjusted.append(g_score - b_score)
    return st.mean(adjusted)


def r3_3_reasoning_calls() -> dict[str, float]:
    """Reasoning calls per row, counted from the chain banners.

    The stored ``llm_http_attempts`` also counts the judge call, which is
    evaluation infrastructure rather than method cost, so it understates the
    architectural ratio.
    """
    rows = [r for r in load(R3_3_PAID)["results"] if r.get("status") == "success"]
    steps: dict[str, list[int]] = collections.defaultdict(list)
    for row in rows:
        steps[row["condition"]].append(len(_BANNER.findall(row.get("chain_text") or "")))
    return {arm: st.mean(v) for arm, v in steps.items()}


def defect_rows_by_condition() -> dict[str, dict[str, Any]]:
    """Normalisation-affected row counts for the 240x7 matrix.

    Selected by exact artifact path: the audit also contains per-shard files
    and a single-condition file from the same directory, and matching loosely
    picks up one of those instead -- which is how a first pass at this lookup
    returned a table with no baselines in it.
    """
    wanted = ("outputs/paper_level_evidence_package/"
              "geosym_bench_controlled_glm46v_240x7_rerun/full_240/paid_result.json")
    for entry in load(DEFECT_AUDIT)["audits"]:
        if entry["artifact"].replace("\\", "/") == wanted:
            return entry["conditions"]
    raise KeyError(f"audit entry not found: {wanted}")


def build_assertions() -> list[dict[str, Any]]:
    """Each entry: a claim the pack makes, and the value the artifact gives."""
    bench = bench_condition_stats()
    c1 = c1_variants()
    m0 = load(PLOTQA_M0)
    attribution = load(PLOTQA_ATTR)
    rescore = load(R3_3)
    layering = rescore["contrasts"]["H1_layering"]["after_repair"]
    layer = load(LAYER_PROFILE)["paired_contrasts"]["E5_raw_minus_E2"]["metrics"]
    substitutability = load(SUBSTITUTABILITY)["summary"]
    defect = defect_rows_by_condition()
    calls = r3_3_reasoning_calls()
    evidence_contrast = plotqa_evidence_contrast()
    subtypes = m0["coverage_by_qa_type"]
    evidence = attribution["H_N2_evidence_not_independently_sufficient"]
    image = attribution["H_N1_numeric_penalty_reproduces"]

    checks: list[tuple[str, Any, Any, float]] = [
        # -- section 2: the table, including the n column the header once hid
        ("bench E0 n", bench["E0"]["n"], 240, 0),
        ("bench E1 n", bench["E1"]["n"], 239, 0),
        ("bench E2 n", bench["E2"]["n"], 240, 0),
        ("bench E3 n", bench["E3"]["n"], 229, 0),
        ("bench E4 n", bench["E4"]["n"], 229, 0),
        ("bench E5_raw n", bench["E5_raw"]["n"], 237, 0),
        ("bench E2 mean", bench["E2"]["mean"], 70.64, 0.01),
        ("bench E5_raw mean", bench["E5_raw"]["mean"], 81.46, 0.01),
        ("bench E5_raw all-objective-zero", bench["E5_raw"]["all_zero"], 0, 0),
        ("bench E3 all-objective-zero", bench["E3"]["all_zero"], 27, 0),
        # -- C1: all three statistics, so none can be quietly swapped
        ("C1 difference of means", c1["difference_of_means"], 10.82, 0.01),
        ("C1 paired (LOCKED)", c1["paired"], 10.66, 0.01),
        ("C1 paired n", c1["paired_n"], 237, 0),
        ("C1 complete-case", c1["complete_case"], 10.81, 0.01),
        ("C1 complete-case n", c1["complete_case_n"], 225, 0),
        # -- the full statistics the pack quotes for the locked figure
        ("C1 CI low", c1["paired_ci_low"], 7.14, 0.01),
        ("C1 CI high", c1["paired_ci_high"], 14.18, 0.01),
        ("C1 t", c1["paired_t"], 5.97, 0.01),
        ("C1 t p", c1["paired_t_p"], 8.8e-09, 1e-10),
        ("C1 Wilcoxon p", c1["paired_wilcoxon_p"], 7.3e-04, 1e-5),
        ("C1 Cohen's dz", c1["paired_cohens_dz"], 0.388, 0.001),
        ("C1 wins", c1["paired_wins"], 116, 0),
        ("C1 losses", c1["paired_losses"], 103, 0),
        ("C1 ties", c1["paired_ties"], 18, 0),
        # The median is 0.00 and the split is near-even, so the effect is
        # magnitude-driven. A reviewer computes this; the pack must say it.
        ("C1 median", c1["paired_median"], 0.0, 0.001),
        ("C1 mean win magnitude", c1["mean_win_magnitude"], 28.9, 0.05),
        ("C1 mean loss magnitude", c1["mean_loss_magnitude"], -8.05, 0.05),
        ("C1 normalisation bound", c1_normalisation_bound(), 9.90, 0.01),
        # -- C2: the layer decomposition, complete-case against E2
        ("C2 L1 delta", layer["L1_reading"]["delta"], 10.65, 0.01),
        ("C2 L2 delta", layer["L2_analysis"]["delta"], 18.35, 0.01),
        ("C2 L3 delta", layer["L3_inference"]["delta"], 5.29, 0.01),
        ("C2 L3 objective-only delta",
         layer["L3_inference_objective_only"]["delta"], 8.00, 0.01),
        ("C2 judge-only delta", layer["judge_only"]["delta"], -1.04, 0.01),
        ("C2 total delta", layer["total"]["delta"], 10.81, 0.01),
        ("C2 n", layer["total"]["n"], 225, 0),
        # -- C5: substitutability, including the zeros that define the boundary
        ("C5 clean-grid on GeoSym-Bench",
         substitutability["geosym_bench::clean_grid"]["measured_items_total"], 222, 0),
        ("C5 clean-grid on PlotQA",
         substitutability["plotqa::clean_grid"]["measured_items_total"], 0, 0),
        ("C5 mark-chain on PlotQA",
         substitutability["plotqa::plotqa_mark_chain"]["measured_items_total"], 772, 0),
        ("C5 mark-chain on GeoSym-Bench",
         substitutability["geosym_bench::plotqa_mark_chain"]["measured_items_total"],
         0, 0),
        # -- normalisation exposure per condition, quoted verbatim in section 5
        ("defect rows E0", defect["E0"]["signature"], 16, 0),
        ("defect rows E1", defect["E1"]["signature"], 11, 0),
        ("defect rows E2", defect["E2"]["signature"], 8, 0),
        ("defect rows E3", defect["E3"]["signature"], 23, 0),
        ("defect rows E4", defect["E4"]["signature"], 23, 0),
        ("defect rows E5_raw", defect["E5_raw"]["signature"], 0, 0),
        # -- section 3.1: coverage, and the subtype split behind it
        ("C4 samples", m0["samples_evaluated"], 499, 0),
        ("C4 coverage before", m0["wiring"]["legacy_mean_coverage"], 0.522, 0.001),
        ("C4 coverage after", m0["wiring"]["chain_mean_coverage"], 0.765, 0.001),
        ("C4 zero-coverage before", m0["wiring"]["legacy_zero_coverage"], 173, 0),
        ("C4 zero-coverage after", m0["wiring"]["chain_zero_coverage"], 36, 0),
        ("C4 full-coverage after", m0["wiring"]["chain_full_coverage"], 242, 0),
        ("C4 hbar before", subtypes["hbar"]["legacy_coverage"], 0.035, 0.001),
        ("C4 hbar after", subtypes["hbar"]["chain_coverage"], 0.779, 0.001),
        ("C4 vbar after", subtypes["vbar"]["chain_coverage"], 0.769, 0.001),
        ("C4 line after", subtypes["line"]["chain_coverage"], 0.508, 0.001),
        ("C4 dot after", subtypes["dot"]["chain_coverage"], 0.883, 0.001),
        # -- section 3.2: the retained PlotQA comparison.
        ("C8 evidence+image accuracy", image["accuracy_treatment"], 0.588, 0.001),
        ("C8 image-only accuracy", image["accuracy_baseline"], 0.637, 0.001),
        ("C8 paired n", image["n_paired"], 80, 0),
        ("C8 paired p", image["mcnemar_exact_p"], 0.388, 0.001),
        # -- section 4: R3-3, and the cost figure the pack now states as 3x
        ("C9 mean delta", layering["mean_delta"], 0.12, 0.01),
        ("C9 p", layering["paired_t_p"], 0.935, 0.001),
        ("C9 n", layering["n"], 104, 0),
        ("C9 wins", layering["wins"], 48, 0),
        ("C9 losses", layering["losses"], 48, 0),
        ("C9 reproduction untouched",
         rescore["reproduction_check"]["untouched_rows"], 276, 0),
        ("C9 reproduction mismatches",
         rescore["reproduction_check"]["mismatches"], 0, 0),
        ("C9 layered reasoning calls", calls["E5_raw"], 3.0, 0.001),
        ("C9 single-pass reasoning calls", calls["E6_flat_single"], 1.0, 0.001),
        # -- the contrast that isolates evidence, kept as a standing guard.
        # It runs against us; pinning both counts is what stops the deleted
        # C7 claim from being reconstructed out of the same artifact.
        ("PlotQA evidence-added wins", evidence_contrast["evidence_added_wins"], 4, 0),
        ("PlotQA evidence-added losses",
         evidence_contrast["evidence_added_losses"], 8, 0),
        # -- E1 belongs in the failure-mode table: the prose says "every baseline"
        ("bench E1 all-objective-zero", bench["E1"]["all_zero"], 16, 0),
        ("bench E1 n (failure-mode table)", bench["E1"]["n"], 239, 0),
    ]

    results = []
    for name, actual, expected, tolerance in checks:
        actual_value = float(actual)
        ok = abs(actual_value - float(expected)) <= tolerance
        results.append({
            "check": name,
            "artifact_value": round(actual_value, 6),
            "pack_value": expected,
            "tolerance": tolerance,
            "status": "OK" if ok else "MISMATCH",
        })
    return results


def plotqa_evidence_contrast() -> dict[str, float]:
    """The retained image-only versus measured-plus-image comparison."""
    rows = load(PLOTQA_ATTR_PAID)["results"]
    by_condition: dict[str, dict[str, bool]] = collections.defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]][row["sample_id"]] = bool(row.get("correct"))
    image = by_condition["E0_plotqa_direct"]
    both = by_condition["G_measured_raw"]
    shared = sorted(set(image) & set(both))
    return {
        "evidence_added_wins": float(sum(1 for s in shared
                                         if both[s] and not image[s])),
        "evidence_added_losses": float(sum(1 for s in shared
                                           if image[s] and not both[s])),
    }


def blueprint_assertions(text: str) -> list[dict[str, Any]]:
    """Statements the blueprint must and must not contain.

    The blueprint had no automated coverage when three defects reached it at
    once: an evidence-contribution claim the data does not support, a
    substitutability boundary citing one of two different refusal reasons, and
    a failure-mode table missing E1 while the prose quantified "every
    baseline". Each is an assertion here.
    """
    # Line-wrapping is invisible to a substring search, so normalise runs of
    # whitespace before matching. Without this, a required phrase fails purely
    # because the editor broke it across two lines.
    flat = re.sub(r"\s+", " ", text)

    required = [
        ("lists E1 in the failure-mode table", "| E1 | 16/239 (6.7%) | 1.90 |"),
        ("marks the judge range as per-condition", "67–83 on average per condition"),
        ("names the measured off-diagonal cell", "insufficient_grid_lines"),
        ("flags the unrun off-diagonal cell", "no_extracted_record_supplied"),
        ("warns off the evidence-gain reading",
         "Do not write that the measured evidence contributes a gain"),
        ("states the 3x reasoning cost", "3 reasoning calls against 1"),
        ("keeps the C1 statistic named", "+10.66 paired (n=237)"),
        # C4 (evidence, PlotQA) and C1 (answers, GeoSym-Bench) merged into one
        # "improves on both benchmarks" sentence would imply a PlotQA accuracy
        # result at the one place a reader trusts most -- the contribution list.
        ("separates the C4 and C1 contributions", "Keep these two separate"),
        ("records O-4 as open", "O-4 open"),
    ]
    # A prohibition quotes the sentence it bans, so a bare search for the
    # sentence fires on the rule that forbids it. Each pattern therefore
    # requires an asserting verb and rejects the negated form -- the blueprint
    # states its bans as "No sentence asserting X" and "Never write X".
    forbidden = [
        # The deleted C7. Any of these phrasings would restate it.
        ("restates the C7 evidence-gain claim",
         r"(?<!not )\bevidence\s+(contributes|adds|carries)\s+a?\s*"
         r"(large|significant|substantial)"),
        ("cites the +17.5 gap as an evidence result",
         r"\+17\.5[^.]{0,60}\bevidence\b"),
        ("asserts a PlotQA win",
         r"(?<!asserting )(?<!No sentence asserting )GeoSym "
         r"(beats|outperforms|exceeds) the image-only"),
        ("claims equal weighting",
         r"(?<!Never write \")(?<!No \")equally[- ]weighted dimensions"),
    ]
    results = []
    for name, needle in required:
        flat_needle = re.sub(r"\s+", " ", needle)
        results.append({
            "check": f"blueprint: {name}",
            "status": "OK" if flat_needle in flat else "MISSING",
            "detail": needle,
        })
    for name, pattern in forbidden:
        hit = re.search(pattern, flat, re.IGNORECASE)
        results.append({
            "check": f"blueprint: {name}",
            "status": "FORBIDDEN_PRESENT" if hit else "OK",
            "detail": pattern,
        })
    return results


def prose_assertions(pack_text: str) -> list[dict[str, Any]]:
    """Statements the pack must and must not contain."""
    required = [
        ("names the varying n", "it is not a uniform 240"),
        ("states the 3x reasoning cost", "3 reasoning calls against 1"),
        ("shows the per-subtype split", "Per-subtype detail"),
        ("keeps the image-only PlotQA baseline", "image-only"),
        ("names the locked C1 statistic", "paired mean difference, +10.66"),
        ("explains the win/loss split", "magnitude rather than frequency"),
        ("labels the C2 block complete-case", "Label these numbers"),
        ("reports the flat judge component", "judge score is unchanged"),
    ]
    forbidden = [
        # A directional PlotQA win is unsupported at p=0.39.
        ("asserts a PlotQA win", r"GeoSym (beats|outperforms) the image-only"),
        # The pre-normalisation contrast measures formatting, not layering.
        ("cites +13.28 as a layering result",
         r"\+13\.28[^.]{0,80}\b(supports|shows|demonstrates)\b"),
    ]
    results = []
    for name, needle in required:
        results.append({
            "check": f"prose: {name}",
            "status": "OK" if needle in pack_text else "MISSING",
            "detail": needle,
        })
    for name, pattern in forbidden:
        hit = re.search(pattern, pack_text, re.IGNORECASE)
        results.append({
            "check": f"prose: {name}",
            "status": "FORBIDDEN_PRESENT" if hit else "OK",
            "detail": pattern,
        })
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=PACK)
    parser.add_argument("--blueprint", type=Path, default=BLUEPRINT)
    args = parser.parse_args(argv)

    numbers = build_assertions()
    prose = prose_assertions(args.pack.read_text(encoding="utf-8"))
    prose += blueprint_assertions(args.blueprint.read_text(encoding="utf-8"))
    failures = [r for r in numbers + prose if r["status"] != "OK"]

    print("# Writing-surface verification (zero cost)\n")
    print(f"Created: {utc_now()}\n")
    print(f"**{len(numbers)} numeric checks, {len(prose)} prose checks "
          f"(pack + blueprint), {len(failures)} failure(s)**\n")

    if failures:
        print("## Failures\n")
        for entry in failures:
            print(f"- `{entry['check']}` -> {entry['status']}: "
                  f"artifact={entry.get('artifact_value')} "
                  f"pack={entry.get('pack_value', entry.get('detail'))}")
        print()

    print("## All numeric checks\n")
    print("| check | artifact | pack | status |")
    print("|---|---:|---:|---|")
    for entry in numbers:
        print(f"| {entry['check']} | {entry['artifact_value']} | "
              f"{entry['pack_value']} | {entry['status']} |")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
