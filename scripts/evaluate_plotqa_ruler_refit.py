#!/usr/bin/env python
"""Evaluate the W1 ruler refit against official PlotQA annotations (zero cost).

Performs **no network calls and no OCR inference**. Inputs are all frozen
artifacts on disk: the M0 records, the cached PaddleOCR boxes, and the official
``annotations.json`` per-mark ground truth.

Tests the three pre-registered W1 criteria plus the falsification
condition. The criteria and the falsification threshold were fixed **before**
this script produced a number.

Criterion 1: the share of samples whose measured/truth scale is ~1.0 rises.
Criterion 2: value accuracy of detected marks rises from 0.160.
Criterion 3: ``answer_recoverable`` rises from 114/384.
Falsification: if value accuracy stays below 0.40, the ruler was not the main
cause; stop the plan and re-attribute rather than continuing to invest.

Ground truth is used only to *measure* the refit. The refit code itself
(``evaluator/reasoning/plotqa_ruler_refit.py``) never reads an answer, so no
oracle dependency is introduced into the pipeline.
"""

from __future__ import annotations

import argparse
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

from evaluator.reasoning.plotqa_ruler_refit import refit_record  # noqa: E402
from scripts.diagnose_plotqa_mark_segmentation import (  # noqa: E402
    MATCH_RADIUS_PX,
    detected_marks,
    ground_truth_marks,
    values_agree,
)

PACKAGE = ROOT / "outputs" / "paper_level_evidence_package" / "plotqa_public_500"
PADDLEOCR = PACKAGE / "openocr_survey_paddleocr" / "raw_ocr.json"
EASYOCR = PACKAGE / "openocr_survey" / "raw_ocr.json"

# Pre-registered thresholds (see plan section 2). Fixed before any result.
BASELINE_VALUE_ACCURACY = 0.160
# The plan's 114/384 came from scanning y_value *and* x_value. On PlotQA the
# non-value axis holds category coordinates -- typically years -- which
# sometimes coincide with the reference answer, so that figure counts category
# hits as recovered values. 114 is kept as the recorded historical number, but
# the criterion is judged against the like-for-like value-field-only count so
# the comparison is not decided by a scanning-rule difference.
BASELINE_RECOVERABLE_AS_RECORDED = 114
BASELINE_RECOVERABLE_VALUE_FIELD_ONLY = 59
BASELINE_RECOVERABLE_DENOMINATOR = 384
FALSIFICATION_VALUE_ACCURACY = 0.40

NUMERIC_REL_TOLERANCE = 0.05
NUMERIC_ABS_TOLERANCE = 1.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def match_pairs(truth: list[dict[str, Any]], found: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Greedy nearest-centre association, same rule as the diagnosis script."""
    unmatched = list(range(len(found)))
    pairs: list[tuple[int, int]] = []
    for truth_index, gt_mark in enumerate(truth):
        best_position = None
        best_distance = None
        for position, found_index in enumerate(unmatched):
            candidate = found[found_index]
            distance = (
                (candidate["center_x"] - gt_mark["center_x"]) ** 2
                + (candidate["center_y"] - gt_mark["center_y"]) ** 2
            ) ** 0.5
            if best_distance is None or distance < best_distance:
                best_distance, best_position = distance, position
        if best_position is not None and best_distance is not None and best_distance <= MATCH_RADIUS_PX:
            pairs.append((truth_index, unmatched.pop(best_position)))
    return pairs


def answer_recoverable(record: dict[str, Any], value_field_only: bool = True) -> bool | None:
    """Is the reference value present among the measured values?

    With ``value_field_only`` (the default) only the value-bearing field is
    consulted: ``measured_value`` on a refitted record, else ``y_value``. This
    is the like-for-like rule. Scanning the cross axis as well would count
    category coordinates -- on PlotQA usually years -- as recovered values
    whenever one happens to equal the answer, which inflates the baseline.

    ``value_field_only=False`` reproduces the looser historical rule, reported
    alongside for continuity.
    """
    if record.get("answer_type") != "numeric":
        return None
    reference = record.get("reference")
    if not isinstance(reference, (int, float)) or isinstance(reference, bool):
        return None
    tolerance = max(NUMERIC_ABS_TOLERANCE, abs(float(reference)) * NUMERIC_REL_TOLERANCE)
    for point in record.get("measured_points") or []:
        if value_field_only:
            keys = ("measured_value",) if "measured_value" in point else ("y_value",)
        else:
            keys = ("measured_value", "y_value", "x_value")
        for key in keys:
            value = point.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if abs(float(value) - float(reference)) <= tolerance:
                    return True
    return False


def measure(records: dict[str, dict[str, Any]], plan: dict[str, Any],
            annotations: dict[str, dict[str, Any]], label: str) -> dict[str, Any]:
    """Score one record set against ground truth."""
    matched_total = 0
    value_correct = 0
    gt_total = 0
    detected_total = 0
    scale_ratios_by_sample: list[float] = []
    constant_scale_samples = 0
    scale_near_one = 0
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    unmeasurable_points = 0
    total_points = 0

    for sample in plan["selected_samples"]:
        sample_id = sample["sample_id"]
        metadata = sample.get("metadata") or {}
        annotation = annotations.get(str(metadata.get("image_index")))
        record = records.get(sample_id)
        if annotation is None or record is None:
            continue
        truth = ground_truth_marks(annotation)
        if not truth:
            continue
        found = detected_marks(record)
        total_points += len(record.get("measured_points") or [])
        unmeasurable_points += sum(
            1 for p in record.get("measured_points") or []
            if p.get("value_status") == "unmeasurable_axis_not_admissible")

        gt_total += len(truth)
        detected_total += len(found)
        pairs = match_pairs(truth, found)
        matched_total += len(pairs)

        qa_type = metadata.get("qa_type", "unknown")
        ratios = []
        for truth_index, found_index in pairs:
            detected_value = found[found_index]["value"]
            truth_value = truth[truth_index]["value"]
            correct = values_agree(detected_value, truth_value)
            value_correct += 1 if correct else 0
            by_type[qa_type][0] += 1 if correct else 0
            by_type[qa_type][1] += 1
            if detected_value is not None and truth_value:
                ratios.append(detected_value / truth_value)

        if len(ratios) >= 3:
            median_ratio = statistics.median(ratios)
            if median_ratio:
                spread = max(abs(r / median_ratio - 1) for r in ratios)
                if spread <= 0.10:
                    constant_scale_samples += 1
                    scale_ratios_by_sample.append(median_ratio)
                    if 0.95 <= median_ratio <= 1.05:
                        scale_near_one += 1

    recoverable = defaultdict(int)
    recoverable_any_field = defaultdict(int)
    for record in records.values():
        recoverable[str(answer_recoverable(record))] += 1
        recoverable_any_field[str(answer_recoverable(record, value_field_only=False))] += 1
    numeric_total = recoverable["True"] + recoverable["False"]

    return {
        "label": label,
        "ground_truth_marks": gt_total,
        "detected_points": detected_total,
        "matched": matched_total,
        "detection_recall": matched_total / gt_total if gt_total else 0.0,
        "value_correct_among_matched": value_correct,
        "value_accuracy_given_detected": value_correct / matched_total if matched_total else 0.0,
        "end_to_end_usable_rate": value_correct / gt_total if gt_total else 0.0,
        "constant_scale_samples": constant_scale_samples,
        "scale_near_one": scale_near_one,
        "scale_near_one_rate": (scale_near_one / constant_scale_samples
                                if constant_scale_samples else 0.0),
        "scale_median": (statistics.median(scale_ratios_by_sample)
                         if scale_ratios_by_sample else None),
        "answer_recoverable_true": recoverable["True"],
        "answer_recoverable_denominator": numeric_total,
        "recoverable_rate": recoverable["True"] / numeric_total if numeric_total else 0.0,
        "answer_recoverable_any_field": recoverable_any_field["True"],
        "unmeasurable_points": unmeasurable_points,
        "total_points": total_points,
        "value_accuracy_by_qa_type": {
            qa_type: {"correct": ok, "matched": total,
                      "accuracy": ok / total if total else 0.0}
            for qa_type, (ok, total) in sorted(by_type.items(), key=lambda kv: -kv[1][1])
        },
    }


def run(output_dir: Path, ocr_path: Path) -> dict[str, Any]:
    plan = load_json(PACKAGE / "plotqa_500_zero_cost_plan.json")
    baseline_records = {r["sample_id"]: r
                        for r in load_json(PACKAGE / "m0_measurement_records.json")}
    ocr_rows = {row["sample_id"]: row for row in load_json(ocr_path)}
    annotations_path = Path(plan["frozen_before_paid_run"]["official_files"]["annotations_json"])
    annotations = {str(row.get("image_index")): row
                   for row in load_json(annotations_path) if isinstance(row, dict)}

    refitted_records = {
        sample_id: refit_record(record, ocr_rows.get(sample_id))
        for sample_id, record in baseline_records.items()
    }

    before = measure(baseline_records, plan, annotations, "baseline_m0")
    after = measure(refitted_records, plan, annotations, "ruler_refit_from_paddleocr_ticks")

    admissible = sum(1 for r in refitted_records.values()
                     if (r.get("ruler_refit") or {}).get("admissible"))
    reasons: dict[str, int] = defaultdict(int)
    for record in refitted_records.values():
        reasons[(record.get("ruler_refit") or {}).get("reason", "unknown")] += 1

    value_accuracy = after["value_accuracy_given_detected"]
    criteria = {
        "criterion_1_scale_near_one_rises": {
            "before": before["scale_near_one"],
            "after": after["scale_near_one"],
            "before_rate": before["scale_near_one_rate"],
            "after_rate": after["scale_near_one_rate"],
            "met": after["scale_near_one_rate"] > before["scale_near_one_rate"],
        },
        "criterion_2_value_accuracy_rises": {
            "baseline": BASELINE_VALUE_ACCURACY,
            "after": value_accuracy,
            "met": value_accuracy > BASELINE_VALUE_ACCURACY,
        },
        "criterion_3_recoverable_rises": {
            "baseline_like_for_like_value_field_only": (
                f"{BASELINE_RECOVERABLE_VALUE_FIELD_ONLY}/{BASELINE_RECOVERABLE_DENOMINATOR}"),
            "baseline_as_recorded_in_plan": (
                f"{BASELINE_RECOVERABLE_AS_RECORDED}/{BASELINE_RECOVERABLE_DENOMINATOR}"),
            "after": f"{after['answer_recoverable_true']}/{after['answer_recoverable_denominator']}",
            "met": after["answer_recoverable_true"] > BASELINE_RECOVERABLE_VALUE_FIELD_ONLY,
            "met_against_as_recorded": after["answer_recoverable_true"] > BASELINE_RECOVERABLE_AS_RECORDED,
            "note": (
                "The as-recorded 114 counts category-axis coordinates as recovered values; "
                "it is reported for continuity but is not the like-for-like comparison."
            ),
        },
        "falsification_value_accuracy_below_0_40": {
            "threshold": FALSIFICATION_VALUE_ACCURACY,
            "after": value_accuracy,
            "falsified": value_accuracy < FALSIFICATION_VALUE_ACCURACY,
            "action_if_falsified": (
                "Stop the W1 line and re-attribute. The ruler is not the main cause."
            ),
        },
    }

    results = {
        "schema_version": "plotqa_ruler_refit_evaluation_v1",
        "created_at_utc": utc_now(),
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "inputs": {
            "m0_records": str((PACKAGE / "m0_measurement_records.json").relative_to(ROOT)),
            "cached_ocr": str(ocr_path.relative_to(ROOT)),
            "ocr_backend": next((row.get("source_backend") for row in load_json(ocr_path)
                                 if row.get("source_backend")), "unknown"),
            "official_annotations": str(annotations_path),
        },
        "axis_admissibility": {
            "admissible_samples": admissible,
            "total_samples": len(refitted_records),
            "admissible_rate": admissible / len(refitted_records) if refitted_records else 0.0,
            "reasons": dict(reasons),
        },
        "before": before,
        "after": after,
        "pre_registered_criteria": criteria,
        "interpretation_limits": [
            "Ground truth measures the refit only; the refit code never reads an answer.",
            "A refitted axis that is inadmissible yields no value, so recoverability can "
            "fall for samples whose previous value was accidentally close.",
            "Accuracy gains here do not imply end-task gains; that needs paid calls.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ruler_refit_evaluation.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def render(results: dict[str, Any]) -> str:
    before, after = results["before"], results["after"]
    admissibility = results["axis_admissibility"]
    lines = [
        "# W1 — ruler refit from PaddleOCR ticks", "",
        f"Created: {results['created_at_utc']}",
        f"Paid calls: {results['paid_calls_executed']} | "
        f"OCR HTTP: {results['actual_ocr_http_attempts']} | "
        f"OCR backend: {results['inputs']['ocr_backend']}", "",
        "## Axis admissibility", "",
        f"- admissible: {admissibility['admissible_samples']}/"
        f"{admissibility['total_samples']} ({admissibility['admissible_rate']:.3f})",
        "- reasons: " + ", ".join(f"{k}={v}" for k, v in
                                  sorted(admissibility["reasons"].items(), key=lambda kv: -kv[1])),
        "", "## Headline", "",
        "| Metric | before | after |", "|---|---:|---:|",
        f"| detection recall | {before['detection_recall']:.3f} | {after['detection_recall']:.3f} |",
        f"| **value accuracy given detected** | {before['value_accuracy_given_detected']:.3f} "
        f"| **{after['value_accuracy_given_detected']:.3f}** |",
        f"| end-to-end usable rate | {before['end_to_end_usable_rate']:.3f} "
        f"| {after['end_to_end_usable_rate']:.3f} |",
        f"| scale ~1.0 among constant-scale | {before['scale_near_one']}/"
        f"{before['constant_scale_samples']} | {after['scale_near_one']}/"
        f"{after['constant_scale_samples']} |",
        f"| scale median | {before['scale_median']} | {after['scale_median']} |",
        f"| **answer recoverable (value field only)** | {before['answer_recoverable_true']}/"
        f"{before['answer_recoverable_denominator']} | "
        f"**{after['answer_recoverable_true']}/{after['answer_recoverable_denominator']}** |",
        f"| answer recoverable (any field, historical rule) | "
        f"{before['answer_recoverable_any_field']} | {after['answer_recoverable_any_field']} |",
        f"| unmeasurable points | {before['unmeasurable_points']}/{before['total_points']} "
        f"| {after['unmeasurable_points']}/{after['total_points']} |",
        "", "## Pre-registered criteria", "",
    ]
    for name, payload in results["pre_registered_criteria"].items():
        if name.startswith("falsification"):
            lines.append(f"- **{name}**: falsified={payload['falsified']} "
                         f"(value accuracy {payload['after']:.3f} vs threshold {payload['threshold']})")
        else:
            lines.append(f"- {name}: met={payload['met']}")
    lines += ["", "## Value accuracy by qa_type", "",
              "| qa_type | before | after |", "|---|---:|---:|"]
    for qa_type, stats in after["value_accuracy_by_qa_type"].items():
        was = before["value_accuracy_by_qa_type"].get(qa_type, {})
        lines.append(f"| {qa_type} | {was.get('accuracy', 0.0):.3f} | {stats['accuracy']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "paper_level_evidence_package"
                        / "plotqa_m0_ruler_refit")
    parser.add_argument("--ocr", type=Path, default=PADDLEOCR,
                        help="Cached OCR boxes. Defaults to the PaddleOCR survey.")
    args = parser.parse_args(argv)
    results = run(args.output_dir, args.ocr)
    markdown = render(results)
    (args.output_dir / "ruler_refit_evaluation.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
