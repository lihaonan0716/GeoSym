"""Single-image GeoSym fact-diagnosis backend.

This module evaluates GeoSym extracted coordinates/facts against ground truth
without running the L1/L2/L3 reasoning pipeline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

from app.helpers import load_image_and_gt, _get_saved_key, _create_llm_adapter
from app.reasoning_pipeline import resolve_image_path, infer_visual_facts_with_llm
from evaluator.adapters.recognizer import RecognizerAdapter
from evaluator.metrics.reasoning import (
    evaluate_coordinates,
    evaluate_intercepts,
    evaluate_extrema,
    evaluate_monotonicity,
)


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def _build_gt_visible_facts(gt_data: Dict[str, Any]) -> Dict[str, Any]:
    visible = gt_data.get("visible_facts")
    if isinstance(visible, dict):
        return {
            "intercepts": visible.get("intercepts", {}),
            "extrema": visible.get("extrema", []),
            "monotonicity": visible.get("monotonicity", []),
        }
    # Backward-compatible fallback for old generated datasets.
    return {
        "intercepts": gt_data.get("intercepts", {}),
        "extrema": gt_data.get("extrema", []),
        "monotonicity": gt_data.get("monotonicity", []),
    }


def _normalize_coord_points(raw: Any) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    if not isinstance(raw, list):
        return pts
    for item in raw:
        x_val = y_val = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            x_val, y_val = item[0], item[1]
        elif isinstance(item, dict):
            x_val = item.get("coordinate_x", item.get("x"))
            y_val = item.get("coordinate_y", item.get("y"))
        try:
            pts.append((int(round(float(x_val))), int(round(float(y_val)))))
        except (TypeError, ValueError):
            continue
    return pts


def _normalize_extrema_points(raw: Any) -> List[Dict[str, Any]]:
    pts: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return pts
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            x = round(float(item.get("x")), 2)
            y = round(float(item.get("y")), 2)
        except (TypeError, ValueError):
            continue
        etype = str(item.get("type", "")).lower().strip()
        pts.append({"x": x, "y": y, "type": etype})
    return pts


def _match_coordinate_points(
    pred_points: List[Tuple[int, int]],
    gt_points: List[Tuple[int, int]],
    max_dist: float = 1.5,
) -> Dict[str, Any]:
    # Global closest-first matching to avoid suboptimal greedy ordering.
    candidates: List[Tuple[float, int, int]] = []
    for gi, (gx, gy) in enumerate(gt_points):
        for pi, (px, py) in enumerate(pred_points):
            d = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
            if d <= max_dist:
                candidates.append((d, gi, pi))
    candidates.sort()

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matched: List[Dict[str, Any]] = []
    for d, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matched.append({"gt": gt_points[gi], "pred": pred_points[pi], "dist": round(d, 2)})

    missed_gt = [g for i, g in enumerate(gt_points) if i not in used_gt]
    extra_pred = [p for i, p in enumerate(pred_points) if i not in used_pred]
    return {"matched": matched, "missed_gt": missed_gt, "extra_pred": extra_pred}


def _match_extrema_points(
    pred_extrema: List[Dict[str, Any]],
    gt_extrema: List[Dict[str, Any]],
    max_dist: float = 1.5,
) -> Dict[str, Any]:
    # Global closest-first matching (same strategy as coordinate matching).
    candidates: List[Tuple[float, int, int]] = []
    for gi, ge in enumerate(gt_extrema):
        gx, gy = ge["x"], ge["y"]
        for pi, pe in enumerate(pred_extrema):
            d = math.sqrt((float(pe["x"]) - gx) ** 2 + (float(pe["y"]) - gy) ** 2)
            if d <= max_dist:
                candidates.append((d, gi, pi))
    candidates.sort()

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matched: List[Dict[str, Any]] = []
    for d, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        ge = gt_extrema[gi]
        pe = pred_extrema[pi]
        matched.append({
            "gt": ge,
            "pred": pe,
            "dist": round(d, 2),
            "type_ok": str(pe.get("type", "")) == str(ge.get("type", "")),
        })

    missed_gt = [g for i, g in enumerate(gt_extrema) if i not in used_gt]
    extra_pred = [e for i, e in enumerate(pred_extrema) if i not in used_pred]
    return {"matched": matched, "missed_gt": missed_gt, "extra_pred": extra_pred}


def _fmt_list(items: List[str], lang: str, limit: int = 12) -> str:
    if not items:
        return 'none'
    show = items[:limit]
    if len(items) > limit:
        show.append(f"... +{len(items) - limit}")
    return "; ".join(show)


def _fmt_list_full(items: List[str], lang: str) -> str:
    """Format full list without truncation (used for coordinate display)."""
    if not items:
        return 'none'
    return "; ".join(items)


def _precision_only_score(matched: int, pred_total: int) -> float:
    """Precision-only score: only wrong predictions are penalized."""
    if pred_total <= 0:
        return 0.0
    return round(100.0 * matched / pred_total, 1)


def _precision_monotonicity_score(
    pred_mono: list,
    gt_mono: list,
) -> Dict[str, Any]:
    """Precision-oriented monotonicity scoring for facts diagnosis.

    For each *predicted* interval, check whether the GT agrees on the
    direction within the overlapping region.  Narrower predictions that
    are directionally correct receive full credit; regions the model
    didn't predict are NOT penalized.

    Returns a dict compatible with evaluate_monotonicity output format.
    """
    if not isinstance(gt_mono, list) or not gt_mono:
        return {"correct": True, "score": 100.0,
                "pred_display": "N/A", "gt_display": "N/A",
                "details": "No GT monotonicity data"}

    if not isinstance(pred_mono, list) or not pred_mono:
        return {"correct": False, "score": 0.0,
                "pred_display": "none",
                "gt_display": f"{len(gt_mono)} intervals",
                "details": "No monotonicity predicted"}

    def _interval_overlap(a: list, b: list) -> float:
        lo = max(a[0] if math.isfinite(a[0]) else -1e9,
                 b[0] if math.isfinite(b[0]) else -1e9)
        hi = min(a[1] if math.isfinite(a[1]) else 1e9,
                 b[1] if math.isfinite(b[1]) else 1e9)
        return max(0.0, hi - lo)

    pred_scores: list[float] = []
    for pi in pred_mono:
        p_int = pi.get("interval", [0, 0])
        p_dir = pi.get("direction", "")
        if not p_dir:
            pred_scores.append(0.0)
            continue

        best = 0.0
        for gi in gt_mono:
            g_int = gi.get("interval", [0, 0])
            g_dir = gi.get("direction", "")
            ovlp = _interval_overlap(p_int, g_int)
            if ovlp <= 0:
                continue
            if p_dir == g_dir:
                best = 1.0
                break
            else:
                best = max(best, 0.0)
        pred_scores.append(best)

    avg = sum(pred_scores) / len(pred_scores) * 100 if pred_scores else 0.0

    def _fmt_seg(seg):
        return f"[{seg.get('interval', [])}, {seg.get('direction', '')}]"
    pred_disp = "; ".join(_fmt_seg(s) for s in pred_mono[:5])
    gt_disp = "; ".join(_fmt_seg(s) for s in gt_mono[:5])

    n_correct = sum(1 for s in pred_scores if s >= 1.0)
    n_wrong = len(pred_scores) - n_correct

    return {
        "correct": avg >= 70.0,
        "score": round(avg, 1),
        "pred_display": pred_disp[:120] or "none",
        "gt_display": gt_disp[:120],
        "details": (
            f"precision_mono: {n_correct}/{len(pred_scores)} correct, "
            f"{n_wrong} direction-mismatch, score={avg:.1f}"
        ),
    }


def _coord_diag_summary(diag: Dict[str, Any]) -> str:
    matched = len(diag.get("matched", []))
    pred_total = matched + len(diag.get("extra_pred", []))
    gt_total = matched + len(diag.get("missed_gt", []))
    precision = (matched / pred_total) if pred_total > 0 else 0.0
    recall = (matched / gt_total) if gt_total > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return (
        f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} "
        f"matched={matched}/{gt_total}"
    )


def run_facts_diagnosis(
    image_name: str,
    gemini_enabled: bool = False,
    lang: str = "English",
    model_name: str = "",
) -> Tuple[str, str, str, Any]:
    """Run GeoSym fact diagnosis for a single image.

    Visual facts are inferred by the LLM specified by *model_name*.
    If *model_name* is empty, visual fact scoring is skipped.

    Returns:
        (summary_markdown, detail_markdown, raw_markdown, annotated_image_rgb)
    """
    if not image_name:
        return (
            f"{'### GeoSym Function-Specialized M0 Accuracy'}\n\n{'Please select an image before running diagnosis.'}",
            "",
            "",
            None,
        )

    img_path = resolve_image_path(image_name)
    if img_path is None:
        return (
            f"{'### GeoSym Function-Specialized M0 Accuracy'}\n\n"
            f"{'Image not found: `{image}`'.format(image=image_name)}",
            "",
            "",
            None,
        )

    _img, gt = load_image_and_gt(image_name)
    if not gt:
        return (
            f"{'### GeoSym Function-Specialized M0 Accuracy'}\n\n"
            f"{'Ground truth not found: `{image}`'.format(image=image_name)}",
            "",
            "",
            None,
        )

    gt_data = (gt.get("question") or {}).get("ground_truth") or {}
    gt_visible = _build_gt_visible_facts(gt_data)

    recognizer = RecognizerAdapter()
    coords, result, annotated = recognizer.recognize_intersections(
        str(img_path),
        gt,
        gemini_enabled=gemini_enabled,
    )
    annotated_rgb = None
    if annotated is not None:
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Infer visual facts via LLM (replaces former CV-based extraction)
    visual_facts: Dict[str, Any] = {}
    if model_name and coords:
        api_key = _get_saved_key(model_name)
        adapter = _create_llm_adapter(model_name, api_key)
        if adapter is not None:
            visual_facts = infer_visual_facts_with_llm(adapter, img_path, coords) or {}

    pred_intercepts = (visual_facts.get("intercepts") if isinstance(visual_facts, dict) else {}) or {}
    pred_extrema = (visual_facts.get("extrema") if isinstance(visual_facts, dict) else []) or []
    pred_mono = (visual_facts.get("monotonicity") if isinstance(visual_facts, dict) else []) or []

    s_coords = evaluate_coordinates(coords, gt_data.get("coordinates", []))
    s_intercepts = evaluate_intercepts(pred_intercepts, gt_visible.get("intercepts", {}))
    s_extrema = evaluate_extrema(pred_extrema, gt_visible.get("extrema", []))
    s_mono_original = evaluate_monotonicity(pred_mono, gt_visible.get("monotonicity", []))
    s_mono = _precision_monotonicity_score(pred_mono, gt_visible.get("monotonicity", []))

    gt_coords_norm = _normalize_coord_points(gt_data.get("coordinates", []))
    pred_coords_norm = _normalize_coord_points(coords)
    coord_diag = _match_coordinate_points(pred_coords_norm, gt_coords_norm)
    coord_precision_score = _precision_only_score(
        matched=len(coord_diag["matched"]),
        pred_total=len(pred_coords_norm),
    )

    gt_extrema_norm = _normalize_extrema_points(gt_visible.get("extrema", []))
    pred_extrema_norm = _normalize_extrema_points(pred_extrema)
    extrema_diag = _match_extrema_points(pred_extrema_norm, gt_extrema_norm)
    extrema_type_ok = sum(1 for m in extrema_diag["matched"] if m["type_ok"])
    gt_coords_list = [f"({x},{y})" for x, y in gt_coords_norm]
    coord_matched_list = [
        f"GT({m['gt'][0]},{m['gt'][1]})<->Pred({m['pred'][0]},{m['pred'][1]}),d={m['dist']}"
        for m in coord_diag["matched"]
    ]
    coord_missed_list = [f"({x},{y})" for x, y in coord_diag["missed_gt"]]
    coord_extra_list = [f"({x},{y})" for x, y in coord_diag["extra_pred"]]
    pred_coords_list = [f"({x},{y})" for x, y in pred_coords_norm]
    gt_extrema_list = [f"({e['x']},{e['y']},{e['type']})" for e in gt_extrema_norm]
    extrema_matched_list = [
        f"GT({m['gt']['x']},{m['gt']['y']},{m['gt']['type']})<->Pred({m['pred']['x']},{m['pred']['y']},{m['pred']['type']}),d={m['dist']}"
        for m in extrema_diag["matched"]
    ]
    extrema_missed_list = [f"({e['x']},{e['y']},{e['type']})" for e in extrema_diag["missed_gt"]]
    extrema_extra_list = [f"({e['x']},{e['y']},{e['type']})" for e in extrema_diag["extra_pred"]]

    score_map = {
        "coordinates": coord_precision_score,
        "intercepts": s_intercepts.get("score", 0.0),
        "extrema": s_extrema.get("score", 0.0),
        "monotonicity": s_mono.get("score", 0.0),
    }
    avg_score = round(sum(score_map.values()) / len(score_map), 1)

    summary_md = (
        f"{'### GeoSym Function-Specialized M0 Accuracy'}\n\n"
        f"- {'Image'}: `{image_name}`\n"
        f"- {'Overall (4-item average)'}: **{avg_score}**\n"
        f"- {'Coordinates'}: **{score_map['coordinates']}**\n"
        f"- {'Intercepts'}: **{score_map['intercepts']}**\n"
        f"- {'Extrema'}: **{score_map['extrema']}**\n"
        f"- {'Visible-range Monotonicity'}: **{score_map['monotonicity']}**\n"
    )

    detail_md = (
        f"{'### Prediction vs Ground Truth'}\n\n"
        f"#### {'Coordinates'}\n"
        f"- {'Pred'}: `{_fmt_list_full(pred_coords_list, lang)}`\n"
        f"- {'GT'}: `{_fmt_list(gt_coords_list, lang, limit=20)}`\n"
        f"- {'Details'}: {_coord_diag_summary(coord_diag)}\n"
        f"- Score policy: **precision-only** (max=100, no miss penalty), current score=`{coord_precision_score}`\n"
        f"- {'GT total'}: **{len(gt_coords_norm)}** | {'Pred total'}: **{len(pred_coords_norm)}** | "
        f"{'Matched'}: **{len(coord_diag['matched'])}** | {'Missed GT'}: **{len(coord_diag['missed_gt'])}** | "
        f"{'Extra Pred'}: **{len(coord_diag['extra_pred'])}**\n"
        f"- {'Matched pairs'}: `{_fmt_list(coord_matched_list, lang)}`\n"
        f"- {'Missed GT points'}: `{_fmt_list(coord_missed_list, lang)}`\n"
        f"- {'Extra predicted points'}: `{_fmt_list(coord_extra_list, lang)}`\n\n"
        f"#### {'Intercepts'}\n- {'Pred'}: `{s_intercepts.get('pred_display', '')}`\n- {'GT'}: `{s_intercepts.get('gt_display', '')}`\n- {'Details'}: {s_intercepts.get('details', '')}\n\n"
        f"#### {'Extrema'}\n"
        f"- {'Pred'}: `{s_extrema.get('pred_display', '')}`\n"
        f"- {'GT'}: `{_fmt_list(gt_extrema_list, lang, limit=20)}`\n"
        f"- {'Details'}: {s_extrema.get('details', '')}\n"
        f"- {'GT total'}: **{len(gt_extrema_norm)}** | {'Pred total'}: **{len(pred_extrema_norm)}** | "
        f"{'Matched'}: **{len(extrema_diag['matched'])}** | {'Type-correct matches'}: **{extrema_type_ok}** | "
        f"{'Missed GT'}: **{len(extrema_diag['missed_gt'])}** | {'Extra Pred'}: **{len(extrema_diag['extra_pred'])}**\n"
        f"- {'Matched pairs'}: `{_fmt_list(extrema_matched_list, lang)}`\n"
        f"- {'Missed GT points'}: `{_fmt_list(extrema_missed_list, lang)}`\n"
        f"- {'Extra predicted points'}: `{_fmt_list(extrema_extra_list, lang)}`\n\n"
        f"#### {'Visible-range Monotonicity'}\n- {'Pred'}: `{s_mono.get('pred_display', '')}`\n- {'GT'}: `{s_mono.get('gt_display', '')}`\n- {'Details'}: {s_mono.get('details', '')}\n"
    )

    raw_payload = {
        "image": image_name,
        "gt_visible_facts": gt_visible,
        "pred_coords": coords,
        "pred_reasoning_intersections": ((result or {}).get("result") or {}).get("reasoning_intersections", []),
        "pred_reasoning_debug": ((result or {}).get("result") or {}).get("reasoning_debug", {}),
        "pred_visual_facts": visual_facts,
        "recognizer_success": bool((result or {}).get("success", False)),
        "score_details": {
            "coordinates": {
                **s_coords,
                "score_replaced_for_facts_diagnosis": coord_precision_score,
                "scoring_policy": "precision_only_no_miss_penalty",
            },
            "intercepts": s_intercepts,
            "extrema": s_extrema,
            "monotonicity": s_mono,
            "monotonicity_original_recall_based": s_mono_original,
        },
        "diagnostics": {
            "coordinates": coord_diag,
            "extrema": {
                **extrema_diag,
                "type_correct_matches": extrema_type_ok,
            },
        },
    }
    raw_md = f"{'### Raw Diagnosis Data'}\n\n```json\n" + _safe_json(raw_payload) + "\n```"
    return summary_md, detail_md, raw_md, annotated_rgb

