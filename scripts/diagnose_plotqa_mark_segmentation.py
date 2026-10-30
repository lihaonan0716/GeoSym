#!/usr/bin/env python
"""Zero-cost diagnosis of PlotQA mark segmentation against official annotations.

Performs **no network calls and no OCR inference**. Reads the frozen M0 records
and the official PlotQA ``annotations.json``, which carries per-mark ground
truth: for every series, the plotted ``y`` values and the pixel ``bboxes`` of
each mark.

This answers an open question from the zero-cost chain analysis: the
bottleneck was attributed to mark segmentation and pixel read-out, but "it is
broken" is not actionable. Segmentation can fail in three distinct ways, and
they need entirely different fixes:

1. **Detection recall** -- the mark was never found (a segmentation problem).
2. **Value accuracy** -- the mark was found but its pixel was mapped to the
   wrong value (a ruler/calibration problem).
3. **Spurious marks** -- non-marks entered the evidence (a precision problem,
   which is what makes injected evidence actively harmful).

Ground truth is used only to *measure* the extractor. Nothing here feeds the
pipeline, so this does not create an oracle dependency.
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

PACKAGE = ROOT / "outputs" / "paper_level_evidence_package" / "plotqa_public_500"
# A detected mark counts as matching a ground-truth mark when their centres are
# within this many pixels. Marks are 13px wide in PlotQA's dot plots, so half a
# mark width is a generous but not permissive association radius.
MATCH_RADIUS_PX = 12.0
# Value agreement tolerance, matching the public scorer.
VALUE_REL_TOLERANCE = 0.05
VALUE_ABS_TOLERANCE = 1.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric_series(model: dict[str, Any]) -> list[Any] | None:
    """Return the annotation axis holding plotted values, or None.

    PlotQA writes the value axis as ``y`` for vertical charts and as ``x`` for
    horizontal bars, where ``y`` instead holds the category strings. Reading
    ``y`` unconditionally silently discarded every ``hbar`` sample -- 168 of the
    frozen 500 -- because its values parsed as non-numeric. Pick the axis whose
    entries are actually numbers.
    """
    for key in ("y", "x"):
        values = model.get(key)
        if not isinstance(values, list) or not values:
            continue
        if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            return values
    return None


def ground_truth_marks(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the official annotation into one entry per plotted mark."""
    marks: list[dict[str, Any]] = []
    for model in annotation.get("models") or []:
        label = model.get("label") or model.get("name")
        boxes = model.get("bboxes") or []
        values = _numeric_series(model)
        if values is None:
            continue
        for index, box in enumerate(boxes):
            if not isinstance(box, dict):
                continue
            value = values[index] if index < len(values) else None
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            marks.append({
                "series_label": label,
                "value": float(value),
                # PlotQA boxes are {x, y, w, h} with y measured from the top.
                "center_x": float(box.get("x", 0.0)) + float(box.get("w", 0.0)) / 2.0,
                "center_y": float(box.get("y", 0.0)) + float(box.get("h", 0.0)) / 2.0,
            })
    return marks


def detected_marks(record: dict[str, Any]) -> list[dict[str, Any]]:
    marks: list[dict[str, Any]] = []
    for point in record.get("measured_points") or []:
        center = point.get("component_center_pixel") or point.get("pixel")
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            continue
        try:
            cx, cy = float(center[0]), float(center[1])
        except (TypeError, ValueError):
            continue
        value = point.get("y_value")
        marks.append({
            "center_x": cx,
            "center_y": cy,
            "value": float(value) if isinstance(value, (int, float))
            and not isinstance(value, bool) else None,
            "source": point.get("source"),
        })
    return marks


def values_agree(detected: float | None, truth: float) -> bool:
    if detected is None:
        return False
    tolerance = max(VALUE_ABS_TOLERANCE, abs(truth) * VALUE_REL_TOLERANCE)
    return abs(detected - truth) <= tolerance


def match_sample(truth: list[dict[str, Any]], found: list[dict[str, Any]]) -> dict[str, Any]:
    """Greedy nearest-centre matching of detected marks to ground-truth marks.

    Greedy is adequate here because PlotQA marks are well separated relative to
    the match radius; an optimal assignment would not change the recall figure
    materially and would obscure the diagnosis.
    """
    unmatched_found = list(range(len(found)))
    matched: list[tuple[int, int, float]] = []
    for truth_index, gt_mark in enumerate(truth):
        best_index = None
        best_distance = None
        for position, found_index in enumerate(unmatched_found):
            candidate = found[found_index]
            distance = (
                (candidate["center_x"] - gt_mark["center_x"]) ** 2
                + (candidate["center_y"] - gt_mark["center_y"]) ** 2
            ) ** 0.5
            if best_distance is None or distance < best_distance:
                best_distance, best_index = distance, position
        if best_index is not None and best_distance is not None and best_distance <= MATCH_RADIUS_PX:
            matched.append((truth_index, unmatched_found[best_index], best_distance))
            unmatched_found.pop(best_index)

    value_correct = sum(
        1 for truth_index, found_index, _ in matched
        if values_agree(found[found_index]["value"], truth[truth_index]["value"]))
    ratios = []
    for truth_index, found_index, _ in matched:
        detected_value = found[found_index]["value"]
        truth_value = truth[truth_index]["value"]
        if detected_value is not None and truth_value:
            ratios.append(detected_value / truth_value)

    return {
        "ground_truth_marks": len(truth),
        "detected_marks": len(found),
        "matched": len(matched),
        "missed": len(truth) - len(matched),
        "spurious": len(found) - len(matched),
        "value_correct_among_matched": value_correct,
        "value_ratios": ratios,
    }


def run(output_dir: Path) -> dict[str, Any]:
    plan = load_json(PACKAGE / "plotqa_500_zero_cost_plan.json")
    records = {r["sample_id"]: r for r in load_json(PACKAGE / "m0_measurement_records.json")}
    annotations_path = Path(plan["frozen_before_paid_run"]["official_files"]["annotations_json"])
    annotations = {str(row.get("image_index")): row
                   for row in load_json(annotations_path) if isinstance(row, dict)}

    per_sample: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
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
        stats = match_sample(truth, detected_marks(record))
        stats["sample_id"] = sample_id
        stats["qa_type"] = metadata.get("qa_type", "unknown")
        per_sample.append(stats)
        by_type[stats["qa_type"]].append(stats)

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        gt = sum(r["ground_truth_marks"] for r in rows)
        found = sum(r["detected_marks"] for r in rows)
        matched = sum(r["matched"] for r in rows)
        spurious = sum(r["spurious"] for r in rows)
        value_ok = sum(r["value_correct_among_matched"] for r in rows)
        ratios = [ratio for r in rows for ratio in r["value_ratios"]]
        return {
            "samples": len(rows),
            "ground_truth_marks": gt,
            "detected_marks": found,
            "matched": matched,
            "detection_recall": matched / gt if gt else 0.0,
            "detection_precision": matched / found if found else 0.0,
            "spurious_marks": spurious,
            "spurious_rate_of_detected": spurious / found if found else 0.0,
            "value_correct_among_matched": value_ok,
            "value_accuracy_given_detected": value_ok / matched if matched else 0.0,
            "end_to_end_usable_rate": value_ok / gt if gt else 0.0,
            "value_ratio_median": statistics.median(ratios) if ratios else None,
            "value_ratio_p10": (sorted(ratios)[int(len(ratios) * 0.10)] if ratios else None),
            "value_ratio_p90": (sorted(ratios)[int(len(ratios) * 0.90)] if ratios else None),
        }

    overall = aggregate(per_sample)
    results = {
        "schema_version": "plotqa_mark_segmentation_diagnosis_v1",
        "created_at_utc": utc_now(),
        "paid_calls_executed": 0,
        "actual_llm_http_attempts": 0,
        "actual_ocr_http_attempts": 0,
        "inputs": {
            "m0_records": str((PACKAGE / "m0_measurement_records.json").relative_to(ROOT)),
            "official_annotations": str(annotations_path),
            "match_radius_px": MATCH_RADIUS_PX,
        },
        "overall": overall,
        "by_qa_type": {qa_type: aggregate(rows)
                       for qa_type, rows in sorted(by_type.items(),
                                                   key=lambda kv: -len(kv[1]))},
        "interpretation": {
            "detection_recall": "fraction of real marks the segmenter found",
            "value_accuracy_given_detected": "of the marks found, fraction mapped to the right value",
            "end_to_end_usable_rate": "recall x value accuracy; the real ceiling on evidence usefulness",
            "spurious_rate_of_detected": "fraction of injected points that are not real marks",
        },
        "interpretation_limits": [
            "Ground truth is used only to measure the extractor; no pipeline stage reads it.",
            "Greedy centre matching within a fixed radius; a mark detected but displaced "
            "by more than the radius counts as missed plus spurious.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mark_segmentation_diagnosis.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def render(results: dict[str, Any]) -> str:
    overall = results["overall"]
    lines = [
        "# PlotQA mark segmentation — diagnosis against official annotations", "",
        f"Created: {results['created_at_utc']}",
        f"Paid calls: {results['paid_calls_executed']} | "
        f"OCR HTTP: {results['actual_ocr_http_attempts']}", "",
        "## Overall", "",
        f"- samples compared: {overall['samples']}",
        f"- ground-truth marks: {overall['ground_truth_marks']}",
        f"- detected points: {overall['detected_marks']}",
        f"- **detection recall: {overall['detection_recall']:.3f}**",
        f"- detection precision: {overall['detection_precision']:.3f}",
        f"- spurious rate of detected: {overall['spurious_rate_of_detected']:.3f}",
        f"- **value accuracy given detected: {overall['value_accuracy_given_detected']:.3f}**",
        f"- **end-to-end usable rate: {overall['end_to_end_usable_rate']:.3f}**",
        f"- value ratio p10/median/p90: {overall['value_ratio_p10']} / "
        f"{overall['value_ratio_median']} / {overall['value_ratio_p90']}",
        "", "## By qa_type", "",
        "| qa_type | samples | recall | value acc | usable | spurious |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for qa_type, stats in results["by_qa_type"].items():
        lines.append(
            f"| {qa_type} | {stats['samples']} | {stats['detection_recall']:.3f} "
            f"| {stats['value_accuracy_given_detected']:.3f} "
            f"| {stats['end_to_end_usable_rate']:.3f} "
            f"| {stats['spurious_rate_of_detected']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "paper_level_evidence_package"
                        / "plotqa_mark_segmentation_diagnosis")
    args = parser.parse_args(argv)
    results = run(args.output_dir)
    markdown = render(results)
    (args.output_dir / "mark_segmentation_diagnosis.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
