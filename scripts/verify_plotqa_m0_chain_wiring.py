#!/usr/bin/env python
"""Verify the answering path now consumes the M0' chain, and bound what it buys.

Performs **no network calls and no OCR inference**. Every input is a frozen
artifact already on disk.

Three questions, in order of what they gate:

1. **Is the chain actually wired in?** Until now W1/W2/W4 existed only inside
   the ``evaluate_*`` scripts; ``measured_prompt_facts`` -- the function that
   builds what a model actually sees -- still curated the frozen M0 record whose
   measured values were 16% correct. This recomputes coverage *through the
   answering path* rather than through an evaluation harness, so "what we
   measured" and "what we inject" are checked to be the same thing.

2. **Are the two arms symmetric?** Both must receive the same M0' evidence, the
   same fact budget, and prompts differing only in reliability semantics. The
   prompt lengths are compared directly.

3. **What is the ceiling?** (T6) Given *perfect* evidence -- the official
   per-mark annotations injected verbatim -- how many of the 500 questions could
   a flawless downstream stage answer? This is the honest upper bound on
   ``E5_raw > baseline`` and decides whether further M0 work (W3 line
   extraction, W7 residual axes) is worth funding at all. If the ceiling sits
   below the cached E0 baseline, no amount of extraction work closes that gap
   and the effort belongs on ``E5 > E5_raw`` instead.

Ground truth is read only to *measure*. Nothing in the pipeline modules reads an
answer; the separation is asserted by the reliability module's own tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_plotqa_mark_segmentation import ground_truth_marks  # noqa: E402
from scripts.evaluate_plotqa_ruler_refit import PACKAGE, PADDLEOCR, load_json  # noqa: E402
from scripts.prepare_plotqa_public_benchmark_pilot import SHARED_FINAL_ANSWER_INSTRUCTION  # noqa: E402
from scripts.run_plotqa_locate import answer_prompt, measured_prompt_facts  # noqa: E402

NUMERIC_REL_TOLERANCE = 0.05
NUMERIC_ABS_TOLERANCE = 1.0

CURATED = "G_full_measured_geosym"
RAW = "G_measured_raw"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def fact_values(facts: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for fact in facts:
        raw = fact.get("measured_value") if "measured_value" in fact else fact.get("y_value")
        if _is_number(raw):
            values.append(float(raw))
    return values


def values_match(candidate: float, target: float) -> bool:
    return abs(candidate - target) <= max(NUMERIC_ABS_TOLERANCE, abs(target) * NUMERIC_REL_TOLERANCE)


def coverage_of(facts: list[dict[str, Any]], truth: list[dict[str, Any]]) -> float:
    """Share of ground-truth cells present, with a correct value, in the payload.

    This is the quantity that bounds a reasoning stage: a derived answer needs
    its operand cells. ``answer_recoverable`` -- whether the reference answer
    itself appears -- is structurally capped for the 273/500 derived questions,
    whose answers are computed and never sit in a cell.
    """
    if not truth:
        return 0.0
    values = fact_values(facts)
    hits = sum(1 for mark in truth if any(values_match(v, mark["value"]) for v in values))
    return hits / len(truth)


def reference_number(sample: dict[str, Any]) -> float | None:
    if sample.get("answer_type") != "numeric":
        return None
    try:
        return float(str(sample.get("normalized_answer")).strip())
    except (TypeError, ValueError):
        return None


def run(output_dir: Path, ocr_path: Path,
        records_path: Path | None = None) -> dict[str, Any]:
    """Recompute coverage through the answering path.

    ``records_path`` selects which M0 measurement records to measure. It is a
    parameter rather than a constant because repairs to the extractor write a
    *new* records file beside the old one: the truncated-bar repair produced
    ``plotqa_public_500_bar_repair/m0_measurement_records.json`` while this
    script still read ``plotqa_public_500/``. Reporting the two together mixes
    pre- and post-repair inputs in one table, and the per-family rows then do
    not add up to the headline -- the first thing a reader recomputes.
    """
    plan = load_json(PACKAGE / "plotqa_500_zero_cost_plan.json")
    records_path = records_path or (PACKAGE / "m0_measurement_records.json")
    records = {r["sample_id"]: r for r in load_json(records_path)}
    ocr_rows = {row["sample_id"]: row for row in load_json(ocr_path)}
    annotations_path = Path(plan["frozen_before_paid_run"]["official_files"]["annotations_json"])
    annotations = {str(row.get("image_index")): row
                   for row in load_json(annotations_path) if isinstance(row, dict)}

    rows: list[dict[str, Any]] = []
    for sample in plan["selected_samples"]:
        sample_id = sample["sample_id"]
        record = records.get(sample_id)
        metadata = sample.get("metadata") or {}
        annotation = annotations.get(str(metadata.get("image_index")))
        if record is None or annotation is None:
            continue
        truth = ground_truth_marks(annotation)
        if not truth:
            continue

        ocr_row = ocr_rows.get(sample_id)
        # One arm since M1'/M2' retired (2026-09-19). The `curated_*` keys are
        # kept, now aliasing the single arm, so the emitted JSON stays
        # comparable with the runs recorded before the retirement.
        raw_payload = measured_prompt_facts(record, None, ocr_row=ocr_row)
        curated_payload = raw_payload
        legacy_facts = list(record.get("measured_points") or [])[:20]
        legacy_payload_chars = len(json.dumps({
            "ruler": record.get("ruler"),
            "chart_geometry": record.get("chart_geometry"),
            "measured_facts": legacy_facts,
        }, ensure_ascii=False))

        reference = reference_number(sample)
        truth_values = [mark["value"] for mark in truth]
        rows.append({
            "sample_id": sample_id,
            "qa_type": metadata.get("qa_type", "unknown"),
            "answer_type": sample.get("answer_type"),
            "derived": bool(metadata.get("qa_type_family") == "derived"
                            or metadata.get("question_type") not in (None, "direct")),
            "legacy_coverage": coverage_of(legacy_facts, truth),
            "raw_coverage": coverage_of(raw_payload["measured_facts"], truth),
            "curated_coverage": coverage_of(curated_payload["measured_facts"], truth),
            "tier": "retired_m1_no_longer_scored",
            "curated_injected": curated_payload["evidence_injected"],
            "raw_fact_count": len(raw_payload["measured_facts"]),
            "curated_fact_count": len(curated_payload["measured_facts"]),
            # T6: with perfect evidence, is the reference value even present?
            "oracle_answer_present": (
                reference is not None
                and any(values_match(v, reference) for v in truth_values)
            ),
            "has_numeric_reference": reference is not None,
            "legacy_payload_chars": legacy_payload_chars,
            "chain_payload_chars": len(json.dumps(curated_payload, ensure_ascii=False)),
        })

    def mean(values: list[float]) -> float | None:
        return statistics.mean(values) if values else None

    injected = [r for r in rows if r["curated_injected"]]
    withheld = [r for r in rows if not r["curated_injected"]]

    by_qa: dict[str, Any] = {}
    for qa_type in sorted({r["qa_type"] for r in rows}):
        subset = [r for r in rows if r["qa_type"] == qa_type]
        by_qa[qa_type] = {
            "n": len(subset),
            "legacy_coverage": mean([r["legacy_coverage"] for r in subset]),
            "chain_coverage": mean([r["raw_coverage"] for r in subset]),
        }

    numeric = [r for r in rows if r["has_numeric_reference"]]
    oracle_hits = [r for r in numeric if r["oracle_answer_present"]]

    prompts = {
        "raw": answer_prompt("{question}", {}, RAW),
        "curated": answer_prompt("{question}", {}, CURATED),
    }

    results = {
        "schema_version": "plotqa_m0_chain_wiring_verification_v1",
        "created_at_utc": utc_now(),
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "samples_evaluated": len(rows),
        "wiring": {
            "payload_built_by": "scripts.run_plotqa_locate.measured_prompt_facts",
            "chain_version": (rows[0].get("chain_version") if rows else None)
            or "plotqa_m0_prime_chain_v1",
            "legacy_mean_coverage": mean([r["legacy_coverage"] for r in rows]),
            "chain_mean_coverage": mean([r["raw_coverage"] for r in rows]),
            "legacy_zero_coverage": sum(1 for r in rows if r["legacy_coverage"] <= 0.0),
            "chain_zero_coverage": sum(1 for r in rows if r["raw_coverage"] <= 0.0),
            "legacy_full_coverage": sum(1 for r in rows if r["legacy_coverage"] >= 0.999),
            "chain_full_coverage": sum(1 for r in rows if r["raw_coverage"] >= 0.999),
        },
        "payload_size": {
            "legacy_median_chars": statistics.median(
                [r["legacy_payload_chars"] for r in rows]) if rows else None,
            "chain_median_chars": statistics.median(
                [r["chain_payload_chars"] for r in rows]) if rows else None,
            "note": (
                "Extractor bookkeeping -- BGR colours, component areas, bboxes, "
                "pixel coordinates, per-axis uncertainties -- is no longer injected. "
                "It buried the few numbers a question needs and invited reasoning "
                "from pixel coordinates instead of values."
            ),
        },
        "arm_symmetry": {
            "raw_prompt_chars": len(prompts["raw"]),
            "curated_prompt_chars": len(prompts["curated"]),
            "prompt_char_delta": len(prompts["curated"]) - len(prompts["raw"]),
            "task_guidance_is_shared": "prefer it over visual guesswork"
                                       in SHARED_FINAL_ANSWER_INSTRUCTION,
            "task_guidance_absent_from_arm_specific_text": all(
                "prefer it over visual guesswork" not in text.split(
                    SHARED_FINAL_ANSWER_INSTRUCTION)[-1]
                for text in prompts.values()
            ),
            "same_fact_budget": max((r["curated_fact_count"] for r in rows), default=0)
            <= max((r["raw_fact_count"] for r in rows), default=0),
            "deterministic_solver_on_paid_path": False,
        },
        "m2_gate": {
            "injected_n": len(injected),
            "withheld_n": len(withheld),
            "injected_mean_coverage": mean([r["curated_coverage"] for r in injected]),
            "withheld_mean_coverage": mean([r["raw_coverage"] for r in withheld]),
        },
        "coverage_by_qa_type": by_qa,
        "oracle_ceiling_t6": {
            "numeric_questions": len(numeric),
            "reference_value_present_in_perfect_evidence": len(oracle_hits),
            "share": (len(oracle_hits) / len(numeric)) if numeric else None,
            "reading": (
                "Share of numeric questions whose reference answer appears as a "
                "plotted mark value in the official annotations. The remainder "
                "are derived answers that must be computed; perfect extraction "
                "does not place them in evidence, so extraction work cannot "
                "raise them and the reasoning stage has to do the arithmetic."
            ),
        },
        # Provenance is emitted so a pre/post-repair mix-up is visible in the
        # artifact instead of having to be inferred from a hardcoded path.
        "inputs": {
            "records": str(records_path),
            "records_sha256": sha256_of_file(records_path),
            "ocr": str(ocr_path),
            "plan": str(PACKAGE / "plotqa_500_zero_cost_plan.json"),
        },
        "interpretation_limits": [
            "Coverage measures the evidence, not an answer. Whether better evidence "
            "produces better answers needs paid calls and is not claimed here.",
            "Every number here is relative to the `inputs.records` file named "
            "above. Repairs to the extractor write a new records file, so rows "
            "from two different records files must never share one table.",
            "The oracle ceiling bounds direct look-up questions only. It is not a "
            "predicted score: a model given perfect evidence may still answer wrong.",
            "M1'/M2' retired 2026-09-19. There is one arm now; every admitted "
            "fact is injected, so sections 2 and 3 report a single arm against "
            "itself and are kept only for comparability with earlier runs.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m0_chain_wiring_verification.json").write_text(
        json.dumps({**results, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def render(results: dict[str, Any]) -> str:
    wiring = results["wiring"]
    symmetry = results["arm_symmetry"]
    gate = results["m2_gate"]
    oracle = results["oracle_ceiling_t6"]

    def fmt(value: Any) -> str:
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    lines = [
        "# PlotQA M0' chain — wiring verification", "",
        f"Created: {results['created_at_utc']}",
        f"Paid calls: {results['paid_calls_executed']} | "
        f"LLM HTTP: {results['actual_llm_http_attempts']} | "
        f"OCR HTTP: {results['actual_ocr_http_attempts']}",
        f"Samples: {results['samples_evaluated']}",
        f"Records: `{results['inputs']['records']}`",
        f"Records sha256: `{results['inputs']['records_sha256'][:16]}`", "",
        "## 1. Is the chain wired into the answering path?", "",
        "Coverage measured through `measured_prompt_facts`, the function that "
        "builds what the model sees.", "",
        "| metric | legacy M0 | M0' chain |", "|---|---:|---:|",
        f"| mean cell coverage | {fmt(wiring['legacy_mean_coverage'])} | "
        f"**{fmt(wiring['chain_mean_coverage'])}** |",
        f"| fully covered samples | {wiring['legacy_full_coverage']} | "
        f"**{wiring['chain_full_coverage']}** |",
        f"| zero-coverage samples | {wiring['legacy_zero_coverage']} | "
        f"**{wiring['chain_zero_coverage']}** |",
        f"| median payload chars | {results['payload_size']['legacy_median_chars']:.0f} | "
        f"**{results['payload_size']['chain_median_chars']:.0f}** |", "",
        "## 2. Arm symmetry", "",
        f"- raw prompt: {symmetry['raw_prompt_chars']} chars",
        f"- curated prompt: {symmetry['curated_prompt_chars']} chars "
        f"(delta {symmetry['prompt_char_delta']:+d})",
        f"- task guidance moved to shared instruction: {symmetry['task_guidance_is_shared']}",
        f"- identical fact budget: {symmetry['same_fact_budget']}",
        f"- deterministic solver on paid path: {symmetry['deterministic_solver_on_paid_path']}",
        "",
        "## 3. M2' gate", "",
        f"- injected: {gate['injected_n']}, mean coverage {fmt(gate['injected_mean_coverage'])}",
        f"- withheld: {gate['withheld_n']}, mean coverage {fmt(gate['withheld_mean_coverage'])}",
        "",
        "## 4. Coverage by chart type", "",
        "| qa_type | n | legacy | M0' chain |", "|---|---:|---:|---:|",
    ]
    for qa_type, stats in results["coverage_by_qa_type"].items():
        lines.append(f"| {qa_type} | {stats['n']} | {fmt(stats['legacy_coverage'])} "
                     f"| {fmt(stats['chain_coverage'])} |")

    lines += [
        "", "## 5. Oracle ceiling (T6)", "",
        f"- numeric questions: {oracle['numeric_questions']}",
        f"- reference value present in perfect evidence: "
        f"**{oracle['reference_value_present_in_perfect_evidence']}** "
        f"({fmt(oracle['share'])})",
        "", oracle["reading"], "",
        "## Limits", "",
    ]
    lines += [f"- {limit}" for limit in results["interpretation_limits"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "paper_level_evidence_package"
                        / "plotqa_m0_chain_wiring")
    parser.add_argument("--ocr", type=Path, default=PADDLEOCR)
    parser.add_argument(
        "--records", type=Path, default=None,
        help="M0 measurement records to measure. Defaults to the pre-repair "
             "plotqa_public_500 file; pass the repaired file to report "
             "post-repair coverage. Never mix the two in one table.")
    args = parser.parse_args(argv)
    results = run(args.output_dir, args.ocr, records_path=args.records)
    markdown = render(results)
    (args.output_dir / "m0_chain_wiring_verification.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
