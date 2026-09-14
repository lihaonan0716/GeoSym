"""Analyze P0-4 candidate-level extrema validation results.

This zero-API script reads the frozen P0-3 manifest and evaluates the
pre-registered H1 decision gate. It does not regenerate samples, change the
P0-2 rule, or tune thresholds.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs" / "p0_3_extrema_validation" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "p0_4_extrema_analysis" / "h1_report.json"
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 20260828


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _candidate_delta(counts: dict[str, int]) -> float | None:
    keep_total = counts.get("keep_true", 0) + counts.get("keep_false", 0)
    drop_total = counts.get("drop_true", 0) + counts.get("drop_false", 0)
    keep_precision = _ratio(counts.get("keep_true", 0), keep_total)
    drop_true_rate = _ratio(counts.get("drop_true", 0), drop_total)
    if keep_precision is None or drop_true_rate is None:
        return None
    return keep_precision - drop_true_rate


def _empty_counts() -> dict[str, int]:
    return {"keep_true": 0, "keep_false": 0, "drop_true": 0, "drop_false": 0}


def _add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key in target:
        target[key] += int(source.get(key, 0))


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take percentile of an empty list")
    position = (len(sorted_values) - 1) * q
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


def family_clustered_bootstrap(
    family_counts: dict[str, dict[str, int]],
    *,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    families = sorted(family_counts)
    if not families:
        raise ValueError("No families available for bootstrap")

    rng = random.Random(seed)
    values: list[float] = []
    skipped = 0
    for _ in range(rounds):
        aggregate = _empty_counts()
        for _ in families:
            sampled_family = rng.choice(families)
            _add_counts(aggregate, family_counts[sampled_family])
        delta = _candidate_delta(aggregate)
        if delta is None:
            skipped += 1
            continue
        values.append(delta)

    if not values:
        raise ValueError("All bootstrap draws lacked both KEEP and DROP candidates")
    values.sort()
    return {
        "rounds": rounds,
        "seed": seed,
        "valid_draws": len(values),
        "skipped_draws": skipped,
        "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
    }


def analyze_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    samples = manifest.get("samples", [])
    family_counts: dict[str, dict[str, int]] = {}
    aggregate = _empty_counts()

    for sample in samples:
        family = sample["family"]
        family_counts.setdefault(family, _empty_counts())
        classification = sample["classification"]
        sample_counts = {
            "keep_true": int(classification.get("keep_true", 0)),
            "keep_false": int(classification.get("keep_false", 0)),
            "drop_true": int(classification.get("drop_true", 0)),
            "drop_false": int(classification.get("drop_false", 0)),
        }
        _add_counts(family_counts[family], sample_counts)
        _add_counts(aggregate, sample_counts)

    keep_total = aggregate["keep_true"] + aggregate["keep_false"]
    drop_total = aggregate["drop_true"] + aggregate["drop_false"]
    keep_precision = _ratio(aggregate["keep_true"], keep_total)
    drop_true_rate = _ratio(aggregate["drop_true"], drop_total)
    drop_false_precision = _ratio(aggregate["drop_false"], drop_total)
    delta = _candidate_delta(aggregate)

    per_family: dict[str, dict[str, Any]] = {}
    negative_family_count = 0
    drop_family_count = 0
    keep_family_count = 0
    for family, counts in family_counts.items():
        family_keep_total = counts["keep_true"] + counts["keep_false"]
        family_drop_total = counts["drop_true"] + counts["drop_false"]
        family_delta = _candidate_delta(counts)
        if family_keep_total:
            keep_family_count += 1
        if family_drop_total:
            drop_family_count += 1
        if family_delta is not None and family_delta < 0:
            negative_family_count += 1
        per_family[family] = {
            **counts,
            "keep_precision": _ratio(counts["keep_true"], family_keep_total),
            "drop_true_rate": _ratio(counts["drop_true"], family_drop_total),
            "drop_false_precision": _ratio(counts["drop_false"], family_drop_total),
            "delta_keep_precision_minus_drop_true_rate": family_delta,
        }

    bootstrap = family_clustered_bootstrap(family_counts)
    ci_low, ci_high = bootstrap["ci95"]
    h1_eligible = bool(manifest.get("summary", {}).get("h1_eligible"))
    decision = "stop"
    if h1_eligible and delta is not None and keep_precision is not None:
        if delta >= 0.30 and keep_precision >= 0.80 and ci_low > 0 and negative_family_count <= 1:
            decision = "go"
        elif delta > 0:
            decision = "revise"

    warnings = []
    if drop_family_count < 2:
        warnings.append(
            "DROP evidence is concentrated in fewer than two families; report this as a coverage limit."
        )
    if not h1_eligible:
        warnings.append("P0-3 manifest is not H1 eligible; this report cannot support H1.")

    return {
        "name": "P0-4 extrema candidate-level H1 analysis",
        "source_manifest": manifest.get("name"),
        "sampling_strategy": manifest.get("sampling_strategy"),
        "rule_version": manifest.get("rule_version"),
        "sample_count": len(samples),
        "aggregate_counts": aggregate,
        "keep_precision": keep_precision,
        "drop_true_rate": drop_true_rate,
        "drop_false_precision": drop_false_precision,
        "delta_keep_precision_minus_drop_true_rate": delta,
        "family_counts": per_family,
        "keep_family_count": keep_family_count,
        "drop_family_count": drop_family_count,
        "negative_family_delta_count": negative_family_count,
        "family_clustered_bootstrap": bootstrap,
        "h1_eligible": h1_eligible,
        "h1_decision": decision,
        "claim_limit": (
            "H1 only supports the frozen extrema candidate KEEP/DROP rule on this "
            "new-seed within-family mechanism validation set; it does not prove "
            "M1/M2 end-to-end contribution or external validity."
        ),
        "warnings": warnings,
    }


def _format_report(report: dict[str, Any]) -> str:
    ci_low, ci_high = report["family_clustered_bootstrap"]["ci95"]
    return "\n".join(
        [
            f"H1 decision: {report['h1_decision']}",
            f"Samples: {report['sample_count']}",
            f"KEEP precision: {report['keep_precision']}",
            f"DROP true rate: {report['drop_true_rate']}",
            f"DROP false precision: {report['drop_false_precision']}",
            "Delta KEEP precision - DROP true rate: "
            f"{report['delta_keep_precision_minus_drop_true_rate']}",
            f"Family-clustered bootstrap 95% CI: [{ci_low}, {ci_high}]",
            f"Families with negative delta: {report['negative_family_delta_count']}",
            f"Warnings: {json.dumps(report['warnings'], ensure_ascii=False)}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = analyze_manifest(manifest)
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(_format_report(report))


if __name__ == "__main__":
    main()
