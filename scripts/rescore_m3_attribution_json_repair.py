#!/usr/bin/env python
"""Re-score the R3-3 attribution artifact after repairing non-JSON answer syntax.

Zero paid calls. Zero network. The stored ``chain_text`` is re-parsed, the
objective dimensions are recomputed locally, and the judge score is taken from
the row it already belongs to -- the chain text is unchanged by a syntax
repair, so the judge would see byte-identical input and re-billing it would buy
nothing but variance.

Why this exists
---------------
The confirmation run produced a +13.8 point paired gap for E5_raw over
E6_flat_single. Inspection of the raw text showed the gap was not reasoning:
44 of 104 E6 rows emitted an answer that was *correct* but not valid JSON --

    "monotonicity": [{"interval": (-inf, 1], "direction": "decreasing"}]

Bare parentheses are not JSON, and a literal U+221E is not a JSON number.
``_loads_json_with_numeric_fraction_repair`` repairs ``inf`` spellings and bad
escapes, but neither of those two, so the whole object was discarded and only
the labeled-field fallback survived. Four dimensions then scored 0.

That penalty is arm-asymmetric: the bare-paren interval appears in 12 E6 rows
and in 0 rows of either layered arm. Scoring it as reasoning would attribute a
parser limitation to the absence of layering -- the precise confound the
experiment exists to avoid (invariant 6), on the project's own main open
problem.

What the repair may and may not do
----------------------------------
It is applied **identically to all three arms** and touches only syntax:

  * ``U+221E`` -> ``Infinity``, ``U+2212`` (minus sign) -> ``-``
  * an interval value written as a tuple ``(a, b]`` -> ``[a, b]``

Bracket STYLE is not scored -- ``_parse_monotonicity`` reads the two endpoints
and the direction string, never the open/closed flags -- so normalising the
style discards nothing the metric measures. No value is invented, no answer is
corrected, and a row whose content is wrong stays wrong.

Both score sets are written. The repaired set does not replace the original:
per invariant 8 a report must show what the repair did, not merely its result.
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

from evaluator.reasoning.parser import (  # noqa: E402
    repair_interval_tuples,
    repair_unicode_math_glyphs,
)

DEFAULT_INPUT = ROOT / "outputs" / "m3_attribution_pilot" / "paid_result.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "m3_attribution_pilot" / "json_repair_rescore"

ARMS = ("E5_raw", "E6_flat_single", "E7_form_first")
REFERENCE = "E5_raw"

#: Dimensions that are lost when the final JSON object fails to parse. They are
#: exactly the ones whose values live inside the object rather than in the
#: labeled-field fallback, which recovers only expression/domain/range.
JSON_GATED_DIMENSIONS = ("coordinates", "predictions", "rate_of_change", "monotonicity")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# the repair
# --------------------------------------------------------------------------

def repair_answer_syntax(text: str) -> str:
    """Make LLM answer text JSON-loadable without altering any value.

    Delegates to the repairs that now live in ``evaluator.reasoning.parser``,
    rather than keeping a second copy here. A local copy drifted once already:
    it rewrote the infinity glyph everywhere, including inside string literals,
    so it reported a repair on 133 rows whose only glyph was in explanatory
    prose -- text no parser ever needed repaired. Two implementations of the
    same rule cannot both be the rule.

    Idempotent: running it on already-valid text is a no-op.
    """
    if not text:
        return text
    return repair_interval_tuples(repair_unicode_math_glyphs(text))


_JSON_FENCE = re.compile(r'```(?:json)?\s*\n?(.*?)```', re.DOTALL)


def repair_applied(text: str) -> bool:
    """True when the repair changes text the PARSER would actually read.

    Scoped to fenced JSON blocks on purpose. A chain also contains prose --
    "decreasing on (-inf, -3]" written with real glyphs -- and rewriting there
    changes nothing a parser consumes. Measuring the whole chain flagged 56
    rows whose only glyph sat in an explanation, which would have mislabelled
    their provenance; an earlier version, repairing inside string literals too,
    flagged 133.
    """
    if not text:
        return False
    blocks = _JSON_FENCE.findall(text)
    if not blocks:
        blocks = [text]
    return any(repair_answer_syntax(b) != b for b in blocks)


# --------------------------------------------------------------------------
# re-parsing
# --------------------------------------------------------------------------

_LAYER_BANNER = re.compile(r"^=== (?P<name>.*?) ===$", re.MULTILINE)


def split_chain_sections(chain_text: str) -> dict[str, str]:
    """Split a stored chain into ``{step_name: raw_text}``.

    ``build_chain_text`` writes ``=== {step} ===\\n{raw}`` per step, so a
    layered run yields L1/L2/L3 and a flat run yields its single step. The
    artifact stores only this rendered text, not the ``steps`` dict, so the
    sections have to be recovered from the banners.
    """
    text = chain_text or ""
    matches = list(_LAYER_BANNER.finditer(text))
    if not matches:
        return {"": text}
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group("name").strip()] = text[match.end():end]
    return sections


def reparse(chain_text: str, *, repair: bool = True) -> dict[str, Any]:
    """Re-derive ``final_parsed`` the way the pipeline derived it.

    A layered arm does NOT put the whole answer in its last block: L1 carries
    ``coordinates`` and ``intercepts``, L2 carries ``predictions``,
    ``rate_of_change`` and ``monotonicity``, and L3 carries the expression,
    domain and range. Parsing only the final block therefore drops six
    dimensions from the layered arms while leaving the flat arm intact -- an
    arm-asymmetric error, and the exact class of defect this rescore exists to
    remove. The merge order below matches ``app/reasoning.py``.
    """
    from evaluator.reasoning.parser import (
        parse_comprehensive_answer,
        parse_L1_answer,
        parse_L2_answer,
        parse_L3_answer,
    )

    sections = split_chain_sections(chain_text)
    if repair:
        sections = {name: repair_answer_syntax(raw) for name, raw in sections.items()}

    if {"L1", "L2", "L3"} & set(sections):
        parsed: dict[str, Any] = {}
        parsed.update(parse_L1_answer(sections.get("L1", "")))
        parsed.update(parse_L2_answer(sections.get("L2", "")))
        parsed.update(parse_L3_answer(sections.get("L3", "")))
        return parsed

    # Flat arm: one section holding the whole answer.
    single = next(iter(sections.values())) if sections else ""
    return parse_comprehensive_answer(single)


def gated_fields_present(parsed: dict[str, Any] | None) -> bool:
    """True when every JSON-gated dimension survived parsing.

    Defined on the parsed OUTPUT rather than on one text block, so it means the
    same thing for a layered arm (whose gated fields are spread over L1 and L2)
    as for the flat arm (whose fields are all in one object).
    """
    parsed = parsed or {}
    return all(parsed.get(k) not in (None, "", [], {})
               for k in JSON_GATED_DIMENSIONS)


def json_object_parses_before(row: dict[str, Any]) -> bool:
    """What the RUN produced, read from the row rather than recomputed.

    The interval/glyph repairs now live in ``evaluator.reasoning.parser``, so
    re-parsing the stored text would apply them and report the post-repair
    state under a "before" label. The run's own ``final_parsed`` is the only
    faithful record of the pre-repair behaviour.
    """
    return gated_fields_present(row.get("final_parsed"))


# --------------------------------------------------------------------------
# re-scoring
# --------------------------------------------------------------------------

def rescore_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute dimension and total scores from repaired text.

    The judge score is carried over from the row, not recomputed: a syntax
    repair leaves ``chain_text`` byte-identical, so a fresh judge call would
    receive the same input and differ only by sampling noise.
    """
    from app.reasoning_pipeline import load_layer_config
    from evaluator.metrics.reasoning import evaluate_comprehensive

    from scripts.run_m3_attribution_pilot import load_ground_truth

    layer_config = load_layer_config()
    out: list[dict[str, Any]] = []
    gt_cache: dict[str, dict[str, Any]] = {}

    for row in rows:
        record = dict(row)
        record["rescore_status"] = "skipped_not_success"
        if row.get("status") != "success":
            out.append(record)
            continue

        sample_id = row["sample_id"]
        if sample_id not in gt_cache:
            _, gt_cache[sample_id] = load_ground_truth(sample_id)

        chain_text = row.get("chain_text") or ""
        parsed = reparse(chain_text)
        evaluation = evaluate_comprehensive(
            parsed, gt_cache[sample_id], layer_config, row.get("judge_result"))

        record["repair_applied"] = repair_applied(chain_text)
        record["json_parsed_before"] = json_object_parses_before(row)
        record["json_parsed_after"] = gated_fields_present(parsed)
        record["original_total_score"] = row.get("total_score")
        record["original_dimension_scores"] = row.get("dimension_scores")
        record["repaired_total_score"] = evaluation.get("total_score")
        record["repaired_dimension_scores"] = evaluation.get("dimension_scores")
        record["repaired_final_parsed"] = parsed
        record["rescore_status"] = "rescored"
        out.append(record)

    return out


# --------------------------------------------------------------------------
# paired statistics
# --------------------------------------------------------------------------

def paired_contrast(rows: list[dict[str, Any]], arm_a: str, arm_b: str,
                    score_key: str) -> dict[str, Any]:
    """Paired comparison of ``arm_a`` minus ``arm_b`` over samples scored in both."""
    scored = {(r["sample_id"], r["condition"]): r for r in rows
              if r.get("rescore_status") == "rescored"
              and r.get(score_key) is not None}
    ids = sorted({sid for sid, cond in scored if cond == arm_a}
                 & {sid for sid, cond in scored if cond == arm_b})
    deltas = [scored[(s, arm_a)][score_key] - scored[(s, arm_b)][score_key]
              for s in ids]
    if len(deltas) < 2:
        return {"n": len(deltas), "insufficient_data": True}

    mean = st.mean(deltas)
    sd = st.stdev(deltas)
    result: dict[str, Any] = {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n": len(deltas),
        "mean_delta": mean,
        "sd_delta": sd,
        "median_delta": st.median(deltas),
        "wins": sum(1 for d in deltas if d > 0),
        "losses": sum(1 for d in deltas if d < 0),
        "ties": sum(1 for d in deltas if d == 0),
        "mean_a": st.mean([scored[(s, arm_a)][score_key] for s in ids]),
        "mean_b": st.mean([scored[(s, arm_b)][score_key] for s in ids]),
    }

    try:
        from scipy import stats

        t_stat, t_p = stats.ttest_rel(
            [scored[(s, arm_a)][score_key] for s in ids],
            [scored[(s, arm_b)][score_key] for s in ids])
        result["paired_t"] = float(t_stat)
        result["paired_t_p"] = float(t_p)
        if any(d != 0 for d in deltas):
            w_stat, w_p = stats.wilcoxon(deltas)
            result["wilcoxon_stat"] = float(w_stat)
            result["wilcoxon_p"] = float(w_p)
        se = sd / (len(deltas) ** 0.5)
        crit = float(stats.t.ppf(0.975, len(deltas) - 1))
        result["ci95"] = [mean - crit * se, mean + crit * se]
        result["cohens_dz"] = mean / sd if sd else None
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        result["scipy_unavailable"] = True

    return result


def verdict(contrast: dict[str, Any]) -> str:
    """Apply the pre-registered rule. A result is never renamed after the fact."""
    if contrast.get("insufficient_data"):
        return "insufficient_data"
    p = contrast.get("paired_t_p")
    if p is None:
        return "not_testable"
    if p < 0.05 and contrast["mean_delta"] > 0:
        return "layering_is_attributable"
    if p < 0.05 and contrast["mean_delta"] < 0:
        return "REFUTED_single_pass_ahead_significantly"
    return "no_worse_than_single_pass_and_auditable"


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def parser_provenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect rows written AFTER the interval repairs entered the main parser.

    The artifact is a checkpointed resume, so rows produced before the parser
    was hardened carry the defect in their stored ``final_parsed`` while rows
    produced after it do not. Both are correct records of their own run, but a
    "before repair" column computed over the mixture understates the defect --
    the late rows never had it.

    A row is post-hardening when its text WOULD need the repair yet its stored
    parse already has every gated field: only the new parser produces that
    combination.
    """
    late = [r for r in rows
            if r.get("rescore_status") == "rescored"
            and r.get("repair_applied")
            and r.get("json_parsed_before")]
    by_arm = collections.Counter(r["condition"] for r in late)
    return {
        "rows_written_by_hardened_parser": len(late),
        "by_arm": dict(by_arm),
        "note": (
            "These rows were produced after evaluator/reasoning/parser.py "
            "gained the interval repairs, so their stored parse is already "
            "clean. They inflate the 'before repair' column toward the "
            "repaired state; the 'after repair' column is unaffected."
        ),
    }


def reproduction_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """On rows the repair does not touch, the rescore must reproduce the run.

    This is the guard that makes the repaired numbers trustworthy. If the
    re-parse path differed from the pipeline's own -- a different merge order,
    a missed layer, a wrong step name -- untouched rows would move, and any
    between-arm difference would then be an artifact of THIS script rather
    than of the repair. A large mismatch count invalidates the rescore.
    """
    untouched = [r for r in rows
                 if r.get("rescore_status") == "rescored"
                 and not r.get("repair_applied")]
    mismatches = [
        {
            "sample_id": r["sample_id"],
            "condition": r["condition"],
            "original": r["original_total_score"],
            "repaired": r["repaired_total_score"],
        }
        for r in untouched
        if r["original_total_score"] is not None
        and abs(r["repaired_total_score"] - r["original_total_score"]) > 1e-6
    ]
    return {
        "untouched_rows": len(untouched),
        "mismatches": len(mismatches),
        "max_abs_delta": max(
            (abs(m["repaired"] - m["original"]) for m in mismatches), default=0.0),
        "examples": mismatches[:10],
        "verdict": "reproduces_original" if not mismatches else "DOES_NOT_REPRODUCE",
    }


def build_report(payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r.get("rescore_status") == "rescored"]

    parse_health = {}
    for arm in ARMS:
        subset = [r for r in scored if r["condition"] == arm]
        parse_health[arm] = {
            "rows": len(subset),
            "json_parsed_before": sum(1 for r in subset if r["json_parsed_before"]),
            "json_parsed_after": sum(1 for r in subset if r["json_parsed_after"]),
            "repair_applied": sum(1 for r in subset if r["repair_applied"]),
            "recovered_by_repair": sum(
                1 for r in subset
                if r["json_parsed_after"] and not r["json_parsed_before"]),
            "mean_original": (
                st.mean([r["original_total_score"] for r in subset])
                if subset else None),
            "mean_repaired": (
                st.mean([r["repaired_total_score"] for r in subset])
                if subset else None),
        }

    contrasts = {}
    for label, (a, b) in {
        "H1_layering": ("E5_raw", "E6_flat_single"),
        "H2_form_first": ("E7_form_first", "E5_raw"),
    }.items():
        before = paired_contrast(rows, a, b, "original_total_score")
        after = paired_contrast(rows, a, b, "repaired_total_score")
        contrasts[label] = {
            "before_repair": before,
            "after_repair": after,
            "verdict_before_repair": verdict(before) if label == "H1_layering" else None,
            "verdict_after_repair": verdict(after) if label == "H1_layering" else None,
        }

    return {
        "schema_version": "m3_attribution_json_repair_rescore_v1",
        "created_at_utc": utc_now(),
        "source_artifact": str(DEFAULT_INPUT),
        "source_plan_sha256": payload.get("plan_sha256"),
        "source_stage_status": payload.get("stage_status"),
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "judge_scores_reused_from_source": True,
        "repair_description": {
            "infinity_glyph_to_json": "U+221E -> Infinity",
            "unicode_minus_to_ascii": "U+2212 -> -",
            "interval_tuple_to_list": '"interval": (a, b] -> "interval": [a, b]',
            "applied_to": list(ARMS),
            "symmetric": True,
        },
        "reproduction_check": reproduction_check(rows),
        "parser_provenance": parser_provenance(rows),
        "parse_health": parse_health,
        "contrasts": contrasts,
        "interpretation_limits": [
            "The repair changes answer SYNTAX only. It cannot make a wrong "
            "answer right, and a row whose content is wrong stays wrong.",
            "Judge scores are carried over, not recomputed: the repair leaves "
            "chain_text byte-identical, so a new judge call would differ only "
            "by sampling noise.",
            "Both score sets are reported. The repaired set is the one the "
            "pre-registered rule is applied to, because the unrepaired set "
            "penalises one arm for a parser limitation (invariant 6).",
            "Interval bracket style is normalised because the metric reads "
            "endpoints and direction only; open/closed flags are not scored.",
            "Ten of 330 cells failed with transport timeouts, all but one in "
            "the two longest-output families (rational, logarithmic), so the "
            "paired n is below the frozen 110 and the losses are not evenly "
            "spread across families.",
        ],
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# R3-3 attribution: JSON-syntax repair rescore (zero cost)", "",
        f"Created: {report['created_at_utc']}",
        f"Source: `{report['source_artifact']}` "
        f"(status `{report['source_stage_status']}`)",
        f"Paid calls: {report['paid_calls_executed']}", "",
        "## Reproduction check (rows the repair does not touch)", "",
        f"- untouched rows: {report['reproduction_check']['untouched_rows']}",
        f"- mismatches: {report['reproduction_check']['mismatches']} "
        f"(max |delta| {report['reproduction_check']['max_abs_delta']:.4g})",
        f"- **{report['reproduction_check']['verdict']}**", "",
        "## Parser provenance", "",
        f"- rows written by the hardened parser: "
        f"{report['parser_provenance']['rows_written_by_hardened_parser']} "
        f"{report['parser_provenance']['by_arm'] or ''}", "",
        "## Parse health by arm", "",
        "| arm | rows | JSON ok before | JSON ok after | recovered | "
        "mean before | mean after |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, health in report["parse_health"].items():
        before = health["mean_original"]
        after = health["mean_repaired"]
        lines.append(
            f"| {arm} | {health['rows']} | {health['json_parsed_before']} | "
            f"{health['json_parsed_after']} | {health['recovered_by_repair']} | "
            f"{before:.2f} | {after:.2f} |" if before is not None else
            f"| {arm} | {health['rows']} | - | - | - | - | - |")

    for label, entry in report["contrasts"].items():
        lines += ["", f"## {label}", "",
                  "| | n | mean delta | 95% CI | paired t p | Wilcoxon p | "
                  "wins/losses/ties |", "|---|---:|---:|---|---:|---:|---|"]
        for phase in ("before_repair", "after_repair"):
            c = entry[phase]
            if c.get("insufficient_data"):
                lines.append(f"| {phase} | {c['n']} | - | - | - | - | - |")
                continue
            ci = c.get("ci95")
            ci_text = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "-"
            lines.append(
                f"| {phase} | {c['n']} | {c['mean_delta']:+.2f} | {ci_text} | "
                f"{c.get('paired_t_p', float('nan')):.4g} | "
                f"{c.get('wilcoxon_p', float('nan')):.4g} | "
                f"{c['wins']}/{c['losses']}/{c['ties']} |")
        if entry.get("verdict_after_repair"):
            lines += ["",
                      f"- verdict before repair: `{entry['verdict_before_repair']}`",
                      f"- verdict after repair: **`{entry['verdict_after_repair']}`**"]

    lines += ["", "## Limits", ""]
    lines += [f"- {limit}" for limit in report["interpretation_limits"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = rescore_rows(payload.get("results") or [])
    report = build_report(payload, rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rescore_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "rescored_rows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = render(report)
    (args.output_dir / "rescore_report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
