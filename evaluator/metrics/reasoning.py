"""
Evaluate math reasoning answers with continuous (quantified) scoring.

Scoring dimensions (each 0~100, weighted sum → total_score):
  coordinates, intercepts, extrema, predictions, rate_of_change,
  monotonicity, symmetry, expression, domain, range, judge_inference_score

evaluate_comprehensive() returns per-dimension scores and a weighted total.
"""

import math
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sympy

logger = logging.getLogger(__name__)

from ..reasoning.parser import (
    normalize_interval_str,
    parse_expression_answer,
)

# ============================================================
# Interval parsing and continuous scoring
# ============================================================

_INF = float("inf")


def _is_real_line(s: str) -> bool:
    """Check if normalized string represents all real numbers."""
    return s in (
        "(-oo,+oo)", "(-oo,oo)",
        "R", "ℝ",
        "(-infinity,+infinity)", "(-infinity,infinity)",
        "allofrealnumbers", "allrealnumbers",
        "realnumbers",
    )


def _normalize_numbers_in_interval(s: str) -> str:
    """Normalize numeric values in interval strings (e.g., -4.0 -> -4)."""
    def _norm_num(m: re.Match) -> str:
        val = float(m.group(0))
        if val == int(val):
            return str(int(val))
        return str(val)
    return re.sub(r'-?\d+\.?\d*', _norm_num, s)


def _parse_interval_endpoints(s: str) -> Optional[List[Tuple[str, float, float, str]]]:
    """
    Parse normalized interval string into list of (left_bracket, lo, hi, right_bracket).
    Handles unions like (-oo,3)U(3,+oo).
    Returns None on parse failure.
    """
    if not s:
        return None

    norm = normalize_interval_str(s)
    norm = _normalize_numbers_in_interval(norm)

    if _is_real_line(norm):
        return [("(", -_INF, _INF, ")")]

    parts = norm.split("U")
    intervals = []
    for part in parts:
        part = part.strip()
        m = re.match(r'^([\(\[])([^,]+),([^)\]]+)([\)\]])$', part)
        if not m:
            return None
        lb, lo_s, hi_s, rb = m.groups()
        try:
            lo = -_INF if "oo" in lo_s else float(lo_s)
            hi = _INF if "oo" in hi_s else float(hi_s)
        except ValueError:
            return None
        intervals.append((lb, lo, hi, rb))
    return intervals if intervals else None


def _endpoint_distance_score(pred_val: float, gt_val: float, scale: float = 10.0) -> float:
    """Score for a single endpoint comparison. Returns 0~1."""
    if pred_val == gt_val:
        return 1.0
    if math.isinf(pred_val) and math.isinf(gt_val):
        return 1.0 if (pred_val > 0) == (gt_val > 0) else 0.0
    if math.isinf(pred_val) or math.isinf(gt_val):
        return 0.0
    dist = abs(pred_val - gt_val)
    return max(0.0, 1.0 - dist / scale)


def _bracket_match_score(pred_bracket: str, gt_bracket: str) -> float:
    """Score for bracket type matching. Returns 0 or 1."""
    return 1.0 if pred_bracket == gt_bracket else 0.0


def _interval_continuous_score(pred_str: str, gt_str: str, scale: float = 10.0) -> Tuple[float, str]:
    """
    Compute continuous score (0~100) for interval comparison.
    Components: endpoint distances (80%) + bracket matching (20%).
    Returns (score, details_string).
    """
    norm_p = normalize_interval_str(pred_str)
    norm_g = normalize_interval_str(gt_str)

    if not norm_p or not norm_g:
        return 0.0, "Empty interval"

    norm_pn = _normalize_numbers_in_interval(norm_p)
    norm_gn = _normalize_numbers_in_interval(norm_g)
    if norm_pn == norm_gn:
        return 100.0, "Exact match"

    if _is_real_line(norm_p) and _is_real_line(norm_g):
        return 100.0, "Both R"

    def _sort_union(s):
        return sorted(s.split('U'))
    if 'U'.join(_sort_union(norm_pn)) == 'U'.join(_sort_union(norm_gn)):
        return 100.0, "Union match (reordered)"

    pred_intervals = _parse_interval_endpoints(pred_str)
    gt_intervals = _parse_interval_endpoints(gt_str)

    if pred_intervals is None or gt_intervals is None:
        if _is_real_line(norm_pn) and _is_real_line(norm_gn):
            return 100.0, "Both R (fallback)"
        return 0.0, "Parse failed"

    if len(pred_intervals) != len(gt_intervals):
        return _union_overlap_score(pred_intervals, gt_intervals, scale)

    pred_sorted = sorted(pred_intervals, key=lambda x: x[1])
    gt_sorted = sorted(gt_intervals, key=lambda x: x[1])

    total_ep_score = 0.0
    total_br_score = 0.0
    n = len(gt_sorted)

    for (plb, plo, phi, prb), (glb, glo, ghi, grb) in zip(pred_sorted, gt_sorted):
        total_ep_score += _endpoint_distance_score(plo, glo, scale)
        total_ep_score += _endpoint_distance_score(phi, ghi, scale)
        total_br_score += _bracket_match_score(plb, glb)
        total_br_score += _bracket_match_score(prb, grb)

    ep_avg = total_ep_score / (2 * n) if n > 0 else 0
    br_avg = total_br_score / (2 * n) if n > 0 else 0

    score = (ep_avg * 0.8 + br_avg * 0.2) * 100
    return round(score, 3), f"endpoints={ep_avg:.2f} brackets={br_avg:.2f}"


def _union_overlap_score(pred_intervals, gt_intervals, scale: float) -> Tuple[float, str]:
    """Score when pred and gt have different numbers of sub-intervals."""
    if not gt_intervals:
        return (100.0, "Both empty") if not pred_intervals else (0.0, "GT empty")
    if not pred_intervals:
        return 0.0, "Pred empty"

    total = 0.0
    for glb, glo, ghi, grb in gt_intervals:
        best = 0.0
        for plb, plo, phi, prb in pred_intervals:
            ep = (_endpoint_distance_score(plo, glo, scale) +
                  _endpoint_distance_score(phi, ghi, scale)) / 2
            br = (_bracket_match_score(plb, glb) + _bracket_match_score(prb, grb)) / 2
            s = ep * 0.8 + br * 0.2
            best = max(best, s)
        total += best

    avg = total / len(gt_intervals)
    penalty = max(0, len(pred_intervals) - len(gt_intervals)) * 0.1
    score = max(0, avg - penalty) * 100
    return round(score, 3), f"union_overlap avg={avg:.2f} penalty={penalty:.2f}"


# ============================================================
# Coordinates: distance-based scoring
# ============================================================

def _coords_continuous_score(
    pred_coords: List[List[int]],
    gt_coords: List[List[int]],
    tol: float = 1.5,
    scale: float = 3.0,
) -> Dict[str, Any]:
    """
    Compare coordinate sets with distance-based partial credit.
    Each GT coordinate matched to closest pred; score decays with distance.
    """
    if not gt_coords and not pred_coords:
        return {"score": 100.0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "details": "Both empty", "matched": 0, "gt_total": 0, "pred_total": 0}
    if not gt_coords:
        return {"score": 0.0, "precision": 0.0, "recall": 1.0, "f1": 0.0,
                "details": "GT empty", "matched": 0, "gt_total": 0, "pred_total": len(pred_coords)}
    if not pred_coords:
        return {"score": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "details": "Pred empty", "matched": 0, "gt_total": len(gt_coords), "pred_total": 0}

    used_pred = set()
    match_scores = []

    for gx, gy in gt_coords:
        best_score = 0.0
        best_idx = -1
        for i, (px, py) in enumerate(pred_coords):
            if i in used_pred:
                continue
            dist = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
            s = max(0.0, 1.0 - dist / scale)
            if s > best_score:
                best_score = s
                best_idx = i
        if best_idx >= 0 and best_score > 0:
            used_pred.add(best_idx)
        match_scores.append(best_score)

    recall_score = sum(match_scores) / len(gt_coords)
    precision_penalty = max(0, len(pred_coords) - len(gt_coords)) / max(len(pred_coords), 1)
    score = recall_score * (1.0 - 0.3 * precision_penalty) * 100

    tp = sum(1 for s in match_scores if s >= (1.0 - tol / scale))
    precision = tp / len(pred_coords) if pred_coords else 0
    recall = tp / len(gt_coords) if gt_coords else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "score": round(min(100, score), 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "details": f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} matched={tp}/{len(gt_coords)}",
        "matched": tp,
        "gt_total": len(gt_coords),
        "pred_total": len(pred_coords),
    }


# ============================================================
# Sub-question evaluators
# ============================================================

def _regex_classify_type(expr_str: str) -> str:
    """Infer function type from raw expression string via regex heuristics.

    Used as a fallback when SymPy parsing fails.
    """
    s = expr_str.lower().replace(" ", "")
    if re.search(r"log[_2(]|ln\(|\\ln|\\log", s):
        return "logarithmic"
    if re.search(r"\bexp\(|e\^|e\*\*", s):
        return "exponential"
    if re.search(r"\d\^x|\d\*\*x|\d\^\(x", s):
        return "exponential"
    if re.search(r"\|.*x.*\||abs\(", s):
        return "absolute"
    if re.search(r"1/\(.*x|/\(.*x.*\)", s):
        return "rational"
    if re.search(r"x\*\*3|x\^3|x³", s):
        return "cubic"
    if re.search(r"x\*\*2|x\^2|x²", s):
        return "quadratic"
    if re.search(r"[+-]?\d*\.?\d*\s*x\s*[+-]", s) and "x**" not in s and "x^" not in s:
        return "linear"
    return "unknown"


def _normalized_edit_distance(a: str, b: str) -> float:
    """Normalized Levenshtein edit distance in [0, 1].  0 = identical."""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 1.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb] / max(la, lb)


def evaluate_expression(pred_str: str, gt_str: str, x_range=(-5, 5)) -> Dict[str, Any]:
    """Evaluate expression with multi-component scoring.

    Composite score = curve_fit (0-70) + type_match (0-15) + shape (0-15).
    Symbolic equivalence still yields 100.  Fallback chain handles SymPy
    parsing failures gracefully instead of binary 0/100.
    """
    from .expression import (
        parse_expression, symbolic_equivalence, compute_r2,
        compute_pearson_r, classify_type,
    )

    pred_clean = parse_expression_answer(pred_str)
    gt_clean = parse_expression_answer(gt_str)

    gt_expr = parse_expression(gt_clean)
    pred_expr = parse_expression(pred_clean)

    # ------------------------------------------------------------------
    # Path A: Both expressions parsed successfully
    # ------------------------------------------------------------------
    if pred_expr is not None and gt_expr is not None:
        sym_eq = symbolic_equivalence(pred_expr, gt_expr)
        if sym_eq:
            return {
                "correct": True,
                "score": 100.0,
                "pred_display": pred_str,
                "gt_display": gt_str,
                "details": "SymEq=True (symbolic equivalence)",
            }

        r2 = compute_r2(pred_expr, gt_expr, x_range)
        pr = compute_pearson_r(pred_expr, gt_expr, x_range)

        pred_type = classify_type(pred_expr)
        gt_type = classify_type(gt_expr)
        type_match = pred_type == gt_type and pred_type != "unknown"

        curve_pts = round(max(0.0, r2) * 70, 3) if r2 is not None else 0.0
        type_pts = 15.0 if type_match else 0.0
        shape_pts = round(max(0.0, pr) * 15, 3) if pr is not None else 0.0

        score = round(min(curve_pts + type_pts + shape_pts, 99.0), 3)
        correct = sym_eq or (r2 is not None and r2 > 0.95)

        detail_parts = []
        if r2 is not None:
            detail_parts.append(f"R²={r2:.3f}→{curve_pts}")
        if pr is not None:
            detail_parts.append(f"Pearson={pr:.3f}→{shape_pts}")
        detail_parts.append(
            f"Type={'✓' if type_match else '✗'}({pred_type}/{gt_type})→{type_pts}"
        )
        return {
            "correct": correct,
            "score": score,
            "pred_display": pred_str,
            "gt_display": gt_str,
            "details": " | ".join(detail_parts),
        }

    # ------------------------------------------------------------------
    # Path B: GT parsed but prediction failed — fallback chain
    # ------------------------------------------------------------------
    if gt_expr is not None and pred_expr is None:
        gt_type = classify_type(gt_expr)
        pred_type_regex = _regex_classify_type(pred_clean)
        type_match = (
            pred_type_regex == gt_type
            and pred_type_regex != "unknown"
        )
        type_pts = 15.0 if type_match else 0.0

        ned = _normalized_edit_distance(
            pred_clean.replace(" ", ""), gt_clean.replace(" ", ""),
        )
        edit_pts = round(max(0.0, 1.0 - ned) * 10, 3)

        score = round(min(type_pts + edit_pts, 25.0), 3)
        return {
            "correct": False,
            "score": score,
            "pred_display": pred_str,
            "gt_display": gt_str,
            "details": (
                f"ParseFail(pred) | RegexType={'✓' if type_match else '✗'}"
                f"({pred_type_regex}/{gt_type})→{type_pts}"
                f" | EditDist={ned:.2f}→{edit_pts}"
            ),
        }

    # ------------------------------------------------------------------
    # Path C: Both failed to parse — string similarity only
    # ------------------------------------------------------------------
    ned = _normalized_edit_distance(
        pred_clean.replace(" ", ""), gt_clean.replace(" ", ""),
    )
    exact = ned == 0.0
    edit_pts = round(max(0.0, 1.0 - ned) * 10, 3)
    score = 100.0 if exact else edit_pts
    return {
        "correct": exact,
        "score": score,
        "pred_display": pred_str,
        "gt_display": gt_str,
        "details": f"BothParseFail | EditDist={ned:.2f}→{score}",
    }


def evaluate_domain(pred_str: str, gt_str: str) -> Dict[str, Any]:
    """Evaluate domain — continuous score via endpoint distance."""
    score, details = _interval_continuous_score(pred_str, gt_str)
    return {
        "correct": score >= 95.0,
        "score": score,
        "pred_display": pred_str,
        "gt_display": gt_str,
        "details": details,
    }


def evaluate_range(pred_str: str, gt_str: str) -> Dict[str, Any]:
    """Evaluate range — continuous score via endpoint distance."""
    score, details = _interval_continuous_score(pred_str, gt_str)
    return {
        "correct": score >= 95.0,
        "score": score,
        "pred_display": pred_str,
        "gt_display": gt_str,
        "details": details,
    }


def evaluate_coordinates(pred_coords, gt_coords) -> Dict[str, Any]:
    """Evaluate coordinate recognition — distance-based matching."""
    if isinstance(pred_coords, str):
        pred_coords = _parse_coord_list(pred_coords)
    if not isinstance(pred_coords, list):
        pred_coords = []
    if isinstance(gt_coords, str):
        gt_coords = _parse_coord_list(gt_coords)
    if not isinstance(gt_coords, list):
        gt_coords = []

    pred_norm = _normalize_coord_list(pred_coords)
    gt_norm = _normalize_coord_list(gt_coords)

    result = _coords_continuous_score(pred_norm, gt_norm)
    pred_disp = ", ".join(f"({c[0]},{c[1]})" for c in pred_norm[:10])
    if len(pred_norm) > 10:
        pred_disp += f" +{len(pred_norm)-10}"
    gt_disp = f"{len(gt_norm)} points"
    return {
        "correct": result["score"] >= 95.0,
        "score": result["score"],
        "pred_display": pred_disp or "none",
        "gt_display": gt_disp,
        "details": result["details"],
    }


def _parse_coord_list(s: str) -> List[List[int]]:
    """Parse coordinate string like '(1,2), (3,4)' into list of [x,y]."""
    if not s:
        return []
    pairs = re.findall(r'\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?', s)
    return [[int(x), int(y)] for x, y in pairs]


def _normalize_coord_list(coords) -> List[List[int]]:
    """Normalize coordinates to list of [int, int]."""
    result = []
    for c in coords:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            try:
                result.append([int(round(c[0])), int(round(c[1]))])
            except (TypeError, ValueError):
                pass
        elif isinstance(c, dict):
            try:
                result.append([int(round(c.get("coordinate_x", c.get("x", 0)))),
                               int(round(c.get("coordinate_y", c.get("y", 0))))])
            except (TypeError, ValueError):
                pass
    return result


def _safe_float_key(s: str) -> Optional[float]:
    """Convert a prediction key string to float, handling Unicode minus etc."""
    s = s.replace("−", "-").replace("\u2013", "-").strip()
    s = re.sub(r'^[xX]\s*=\s*', '', s)
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _find_pred_value(pred_preds: dict, target_x: float) -> Any:
    """Find a prediction value whose key matches target_x numerically."""
    for pk, pv in pred_preds.items():
        pk_float = _safe_float_key(str(pk))
        if pk_float is not None and abs(pk_float - target_x) < 1e-6:
            return pv
    return None


def evaluate_predictions(
    pred_preds: Any,
    gt_preds: Any,
) -> Dict[str, Any]:
    """
    Score function-value predictions.

    For each target x, score = max(0, 100 - 20*|pred_y - gt_y|).
    Average across all targets.  Uses numerical key matching to handle
    format differences between GT keys (e.g. "3.0") and LLM keys (e.g. "3").
    """
    if not isinstance(gt_preds, dict) or not gt_preds:
        return {"correct": True, "score": 100.0,
                "pred_display": "N/A", "gt_display": "N/A",
                "details": "No prediction targets"}

    if not isinstance(pred_preds, dict):
        pred_preds = {}

    scores = []
    details_parts = []
    for x_str, gt_y in gt_preds.items():
        gt_x_float = _safe_float_key(x_str)
        if gt_x_float is None:
            scores.append(0.0)
            details_parts.append(f"x={x_str}: invalid GT key")
            continue

        pred_y = _find_pred_value(pred_preds, gt_x_float)
        if pred_y is None:
            scores.append(0.0)
            details_parts.append(f"x={x_str}: missing")
            continue
        try:
            err = abs(float(pred_y) - float(gt_y))
            sc = max(0.0, 100.0 - 20.0 * err)
            scores.append(sc)
            details_parts.append(f"x={x_str}: pred={pred_y}, gt={gt_y}, err={err:.2f}")
        except (TypeError, ValueError):
            scores.append(0.0)
            details_parts.append(f"x={x_str}: parse error")

    avg = sum(scores) / len(scores) if scores else 0.0
    pred_disp = ", ".join(f"f({k})={v}" for k, v in list(pred_preds.items())[:5])
    gt_disp = ", ".join(f"f({k})={v}" for k, v in list(gt_preds.items())[:5])

    return {
        "correct": avg >= 80.0,
        "score": round(avg, 3),
        "pred_display": pred_disp[:80] or "N/A",
        "gt_display": gt_disp[:80],
        "details": "; ".join(details_parts),
    }


def evaluate_rate_of_change(
    pred_roc: Any,
    gt_roc: Any,
) -> Dict[str, Any]:
    """
    Score average rate-of-change estimation.

    score = max(0, 100 - 25*|pred - gt|)
    """
    if not isinstance(gt_roc, dict) or "gt_value" not in gt_roc:
        return {"correct": True, "score": 100.0,
                "pred_display": "N/A", "gt_display": "N/A",
                "details": "No rate-of-change target"}

    gt_val = gt_roc["gt_value"]
    x1, x2 = gt_roc.get("x1", "?"), gt_roc.get("x2", "?")

    try:
        pred_val = float(pred_roc) if pred_roc is not None else None
    except (TypeError, ValueError):
        pred_val = None

    if pred_val is None:
        return {"correct": False, "score": 0.0,
                "pred_display": "N/A",
                "gt_display": f"{gt_val} (x∈[{x1},{x2}])",
                "details": "No prediction provided"}

    err = abs(pred_val - gt_val)
    sc = max(0.0, 100.0 - 25.0 * err)
    return {
        "correct": sc >= 80.0,
        "score": round(sc, 3),
        "pred_display": f"{pred_val:.4g}",
        "gt_display": f"{gt_val} (x∈[{x1},{x2}])",
        "details": f"err={err:.4f}",
    }


# ============================================================
# L1 new: intercepts scoring
# ============================================================

def evaluate_intercepts(
    pred_intercepts: Any,
    gt_intercepts: Any,
) -> Dict[str, Any]:
    """
    Score intercept identification.

    x-intercepts: per-point distance scoring (like coordinates).
    y-intercept: absolute error scoring.
    Combined 50/50.
    """
    if not isinstance(gt_intercepts, dict):
        return {"correct": True, "score": 100.0,
                "pred_display": "N/A", "gt_display": "N/A",
                "details": "No intercepts in GT"}

    if not isinstance(pred_intercepts, dict):
        pred_intercepts = {}

    gt_xi = sorted(gt_intercepts.get("x_intercepts", []))
    gt_yi = gt_intercepts.get("y_intercept")
    pred_xi = sorted(pred_intercepts.get("x_intercepts", []))
    pred_yi = pred_intercepts.get("y_intercept")

    # x-intercepts: matched scoring
    xi_score = 100.0
    if gt_xi:
        if not pred_xi:
            xi_score = 0.0
        else:
            used = set()
            match_sum = 0.0
            for gx in gt_xi:
                best, best_idx = 0.0, -1
                for i, px in enumerate(pred_xi):
                    if i in used:
                        continue
                    s = max(0.0, 1.0 - abs(px - gx) / 3.0)
                    if s > best:
                        best, best_idx = s, i
                if best_idx >= 0:
                    used.add(best_idx)
                match_sum += best
            xi_score = match_sum / len(gt_xi) * 100

    # y-intercept: error scoring
    yi_score = 100.0
    if gt_yi is not None:
        if pred_yi is None:
            yi_score = 0.0
        else:
            try:
                err = abs(float(pred_yi) - float(gt_yi))
                yi_score = max(0.0, 100.0 - 20.0 * err)
            except (TypeError, ValueError):
                yi_score = 0.0

    has_xi = bool(gt_xi)
    has_yi = gt_yi is not None
    if has_xi and has_yi:
        score = 0.5 * xi_score + 0.5 * yi_score
    elif has_xi:
        score = xi_score
    elif has_yi:
        score = yi_score
    else:
        score = 100.0

    pred_disp = f"x:{pred_xi[:5]} y:{pred_yi}"
    gt_disp = f"x:{gt_xi[:5]} y:{gt_yi}"

    return {
        "correct": score >= 80.0,
        "score": round(score, 3),
        "pred_display": str(pred_disp)[:80],
        "gt_display": str(gt_disp)[:80],
        "details": f"xi={xi_score:.1f} yi={yi_score:.1f}",
    }


# ============================================================
# L1 new: extrema scoring
# ============================================================

def evaluate_extrema(
    pred_extrema: Any,
    gt_extrema: Any,
) -> Dict[str, Any]:
    """
    Score extrema identification (position + type).

    Each GT extremum is matched to the closest predicted one.
    Position: distance-based (Euclidean, scale=3).
    Type: exact match bonus (20% of per-point score).
    """
    if not isinstance(gt_extrema, list) or not gt_extrema:
        return {"correct": True, "score": 100.0,
                "pred_display": "N/A", "gt_display": "N/A",
                "details": "No extrema in GT"}

    if not isinstance(pred_extrema, list) or not pred_extrema:
        return {"correct": False, "score": 0.0,
                "pred_display": "none",
                "gt_display": f"{len(gt_extrema)} points",
                "details": "No extrema predicted"}

    used = set()
    scores = []
    for ge in gt_extrema:
        gx, gy, gtype = ge.get("x", 0), ge.get("y", 0), ge.get("type", "")
        best_s, best_i = 0.0, -1
        for i, pe in enumerate(pred_extrema):
            if i in used:
                continue
            px, py = pe.get("x", 0), pe.get("y", 0)
            dist = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
            pos_s = max(0.0, 1.0 - dist / 3.0)
            type_s = 1.0 if pe.get("type", "") == gtype else 0.0
            s = 0.8 * pos_s + 0.2 * type_s
            if s > best_s:
                best_s, best_i = s, i
        if best_i >= 0:
            used.add(best_i)
        scores.append(best_s)

    avg = sum(scores) / len(scores) * 100
    pred_disp = ", ".join(f"({e.get('x')},{e.get('y')})" for e in pred_extrema[:5])
    gt_disp = ", ".join(f"({e.get('x')},{e.get('y')},{e.get('type','')})" for e in gt_extrema[:5])

    return {
        "correct": avg >= 80.0,
        "score": round(avg, 3),
        "pred_display": pred_disp[:80] or "none",
        "gt_display": gt_disp[:80],
        "details": f"avg_match={avg:.1f}",
    }


# ============================================================
# L2 new: monotonicity scoring
# ============================================================

def evaluate_monotonicity(
    pred_mono: Any,
    gt_mono: Any,
) -> Dict[str, Any]:
    """
    Score monotonicity interval identification.

    For each GT interval, find the best-matching predicted interval.
    Score = direction_match * interval_overlap.
    """
    if not isinstance(gt_mono, list) or not gt_mono:
        return {"correct": True, "score": 100.0,
                "pred_display": "N/A", "gt_display": "N/A",
                "details": "No monotonicity data"}

    if not isinstance(pred_mono, list) or not pred_mono:
        return {"correct": False, "score": 0.0,
                "pred_display": "none",
                "gt_display": f"{len(gt_mono)} intervals",
                "details": "No monotonicity predicted"}

    def _bound_score(gt_val: float, pred_val: float, is_lower: bool,
                     ref_span: float) -> float:
        """Score a single interval boundary (0–1).

        *is_lower*=True  → checking left  end (pred should be ≤ gt)
        *is_lower*=False → checking right end (pred should be ≥ gt)
        """
        gt_inf = not math.isfinite(gt_val)
        pred_inf = not math.isfinite(pred_val)
        if gt_inf and pred_inf:
            return 1.0 if (gt_val < 0) == (pred_val < 0) else 0.0
        if gt_inf and not pred_inf:
            return 0.7
        if not gt_inf and pred_inf:
            if is_lower:
                return 1.0 if pred_val < 0 else 0.0
            return 1.0 if pred_val > 0 else 0.0
        if is_lower:
            if pred_val <= gt_val:
                return 1.0
            return max(0.0, 1.0 - (pred_val - gt_val) / max(ref_span, 1.0))
        else:
            if pred_val >= gt_val:
                return 1.0
            return max(0.0, 1.0 - (gt_val - pred_val) / max(ref_span, 1.0))

    def _overlap(gt_int: List[float], pred_int: List[float]) -> float:
        """Boundary-matching score: how well does pred_int cover gt_int?

        Each endpoint is scored independently: a prediction that extends
        at least as far as the GT gets full marks for that bound. This
        correctly rewards analytical extensions beyond the observed range
        and handles infinite intervals.
        """
        _elo = max(gt_int[0] if math.isfinite(gt_int[0]) else -1e6,
                   pred_int[0] if math.isfinite(pred_int[0]) else -1e6)
        _ehi = min(gt_int[1] if math.isfinite(gt_int[1]) else 1e6,
                   pred_int[1] if math.isfinite(pred_int[1]) else 1e6)
        if _ehi <= _elo:
            return 0.0

        fin_lo = gt_int[0] if math.isfinite(gt_int[0]) else 0.0
        fin_hi = gt_int[1] if math.isfinite(gt_int[1]) else 0.0
        ref = abs(fin_hi - fin_lo) if math.isfinite(fin_hi - fin_lo) and fin_hi != fin_lo else 20.0

        lo_s = _bound_score(gt_int[0], pred_int[0], is_lower=True, ref_span=ref)
        hi_s = _bound_score(gt_int[1], pred_int[1], is_lower=False, ref_span=ref)
        return (lo_s + hi_s) / 2.0

    used = set()
    scores = []
    for gi in gt_mono:
        g_int = gi.get("interval", [0, 0])
        g_dir = gi.get("direction", "")
        best_s, best_j = 0.0, -1
        for j, pi in enumerate(pred_mono):
            if j in used:
                continue
            p_int = pi.get("interval", [0, 0])
            p_dir = pi.get("direction", "")
            dir_match = 1.0 if p_dir == g_dir else 0.0
            ovlp = _overlap(g_int, p_int)
            s = dir_match * ovlp
            if s > best_s:
                best_s, best_j = s, j
        if best_j >= 0:
            used.add(best_j)
        scores.append(best_s)

    avg = sum(scores) / len(scores) * 100

    def _fmt_seg(seg):
        return f"[{seg.get('interval', [])}, {seg.get('direction', '')}]"
    pred_disp = "; ".join(_fmt_seg(s) for s in pred_mono[:3])
    gt_disp = "; ".join(_fmt_seg(s) for s in gt_mono[:3])

    return {
        "correct": avg >= 70.0,
        "score": round(avg, 3),
        "pred_display": pred_disp[:80] or "none",
        "gt_display": gt_disp[:80],
        "details": f"avg_recall={avg:.1f}",
    }


# ============================================================
# L2 new: symmetry scoring
# ============================================================

def evaluate_symmetry(
    pred_sym: Any,
    gt_sym: Any,
) -> Dict[str, Any]:
    """
    Score symmetry classification: even / odd / neither.

    Exact match = 100, wrong = 0.
    """
    if not isinstance(gt_sym, str) or not gt_sym:
        return {"correct": True, "score": 100.0,
                "pred_display": "N/A", "gt_display": "N/A",
                "details": "No symmetry data"}

    pred_s = str(pred_sym).strip().lower() if pred_sym else ""
    gt_s = gt_sym.strip().lower()

    # Normalise common synonyms
    _NORM = {"偶": "even", "奇": "odd", "无": "neither", "none": "neither",
             "no": "neither", "yes_even": "even", "yes_odd": "odd"}
    pred_s = _NORM.get(pred_s, pred_s)
    gt_s = _NORM.get(gt_s, gt_s)

    correct = pred_s == gt_s
    return {
        "correct": correct,
        "score": 100.0 if correct else 0.0,
        "pred_display": pred_s or "N/A",
        "gt_display": gt_s,
        "details": "match" if correct else "mismatch",
    }


# ============================================================
# L1 new: Feature Completeness scoring
# ============================================================

def evaluate_feature_completeness(
    pred_full: Dict[str, Any],
    gt: Dict[str, Any],
) -> Dict[str, Any]:
    """Score how completely the model identified key features (recall-oriented).

    Counts x-intercepts, y-intercept presence, and extrema in the GT,
    then checks how many were found by the model.  This dimension rewards
    tiered-injection conditions that prompt the LLM to actively supplement
    GeoSym's incomplete detections.
    """
    total_features = 0
    found_features = 0

    # x-intercepts
    gt_intercepts = gt.get("intercepts", {})
    gt_xi = gt_intercepts.get("x_intercepts", []) if isinstance(gt_intercepts, dict) else []
    pred_intercepts = pred_full.get("intercepts", {})
    pred_xi = pred_intercepts.get("x_intercepts", []) if isinstance(pred_intercepts, dict) else []

    for gx in gt_xi:
        total_features += 1
        for px in pred_xi:
            try:
                if abs(float(px) - float(gx)) < 1.5:
                    found_features += 1
                    break
            except (TypeError, ValueError):
                pass

    # y-intercept
    gt_yi = gt_intercepts.get("y_intercept") if isinstance(gt_intercepts, dict) else None
    pred_yi = pred_intercepts.get("y_intercept") if isinstance(pred_intercepts, dict) else None
    if gt_yi is not None:
        total_features += 1
        if pred_yi is not None:
            try:
                if abs(float(pred_yi) - float(gt_yi)) < 1.5:
                    found_features += 1
            except (TypeError, ValueError):
                pass

    # extrema
    gt_extrema = gt.get("extrema", [])
    pred_extrema = pred_full.get("extrema", [])
    if isinstance(gt_extrema, list):
        for ge in gt_extrema:
            if not isinstance(ge, dict):
                continue
            total_features += 1
            gx, gy = ge.get("x", 0), ge.get("y", 0)
            if isinstance(pred_extrema, list):
                for pe in pred_extrema:
                    if not isinstance(pe, dict):
                        continue
                    try:
                        dist = math.sqrt((float(pe.get("x", 0)) - float(gx)) ** 2 +
                                         (float(pe.get("y", 0)) - float(gy)) ** 2)
                        if dist < 2.0:
                            found_features += 1
                            break
                    except (TypeError, ValueError):
                        pass

    if total_features == 0:
        return {"correct": True, "score": 100.0,
                "pred_display": "N/A", "gt_display": "N/A",
                "details": "No features in GT to evaluate"}

    score = round(found_features / total_features * 100, 3)
    return {
        "correct": score >= 80.0,
        "score": score,
        "pred_display": f"found {found_features}/{total_features}",
        "gt_display": f"{total_features} features",
        "details": f"recall={found_features}/{total_features}",
    }


# ============================================================
# L3 new: Cross-Layer Consistency scoring
# ============================================================

def _safe_parse_expression(expr_str: str):
    """Try to parse an expression string into a callable and a sympy expression.

    Handles common LLM output formats: ``x^3 + 2x - 1``, ``(1/2)x^2``,
    LaTeX-ish notation, implicit multiplication, etc.
    """
    if not expr_str or not isinstance(expr_str, str):
        return None, None
    try:
        from evaluator.reasoning.parser import parse_expression_answer
        cleaned = parse_expression_answer(expr_str)
        if not cleaned:
            return None, None
    except Exception:
        cleaned = expr_str.strip()

    import re as _re
    # Strip residual LaTeX
    cleaned = _re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"((\1)/(\2))", cleaned)
    cleaned = cleaned.replace("\\cdot", "*").replace("\\times", "*")
    cleaned = cleaned.replace("\\pi", "pi").replace("\\sqrt", "sqrt")
    cleaned = _re.sub(r"\\[a-zA-Z]+", "", cleaned)
    cleaned = cleaned.replace("{", "(").replace("}", ")")

    # Convert ^ to ** (must happen before implicit-multiplication pass)
    cleaned = cleaned.replace("^", "**")

    # Insert explicit * for implicit multiplication patterns:
    #   "3x" -> "3*x",  ")x" -> ")*x",  "x(" -> "x*(",  ")(" -> ")*("
    cleaned = _re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", cleaned)
    cleaned = _re.sub(r"(\))([a-zA-Z\d(])", r"\1*\2", cleaned)
    cleaned = _re.sub(r"([a-zA-Z])(\()", r"\1*\2", cleaned)

    x = sympy.Symbol("x")
    local_dict = {"x": x, "e": sympy.E, "E": sympy.E, "pi": sympy.pi}

    # Try sympy.sympify first (handles most well-formed expressions)
    try:
        sym_expr = sympy.sympify(cleaned, locals=local_dict)
        fn = sympy.lambdify(x, sym_expr, modules=["numpy"])
        return fn, sym_expr
    except Exception:
        pass

    # Fallback: use sympy's parser with implicit-multiplication transforms
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
            convert_xor,
        )
        transforms = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )
        sym_expr = parse_expr(cleaned, local_dict=local_dict,
                              transformations=transforms)
        fn = sympy.lambdify(x, sym_expr, modules=["numpy"])
        return fn, sym_expr
    except Exception:
        return None, None


def evaluate_consistency(
    pred_expression: str,
    gt_val: Any,
    *,
    pred_full: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score cross-layer consistency: does the L3 expression match L1 features?

    Checks whether the model's own expression passes through its own detected
    intercepts and extrema.  This rewards conditions where L1 data is reliable
    and the reasoning chain is coherent.
    """
    if pred_full is None:
        return {"correct": True, "score": 100.0,
                "pred_display": "N/A", "gt_display": "N/A",
                "details": "No cross-layer data available"}

    fn, sym_expr = _safe_parse_expression(str(pred_expression))
    if fn is None:
        return {"correct": False, "score": 0.0,
                "pred_display": str(pred_expression)[:60],
                "gt_display": "self-consistency check",
                "details": "Expression parse failed"}

    checks_total = 0
    checks_passed = 0
    detail_parts: List[str] = []

    # Check 1: Expression at x-intercepts should be ≈ 0
    pred_intercepts = pred_full.get("intercepts", {})
    if isinstance(pred_intercepts, dict):
        pred_xi = pred_intercepts.get("x_intercepts", [])
        if isinstance(pred_xi, list):
            for xi_val in pred_xi:
                try:
                    xi_f = float(xi_val)
                    y_at_xi = float(fn(xi_f))
                    if np.isfinite(y_at_xi):
                        checks_total += 1
                        if abs(y_at_xi) < 1.0:
                            checks_passed += 1
                            detail_parts.append(f"f({xi_f:.1f})={y_at_xi:.2f} OK")
                        else:
                            detail_parts.append(f"f({xi_f:.1f})={y_at_xi:.2f} MISMATCH")
                except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                    pass

        # Check 2: Expression at x=0 should match y-intercept
        pred_yi = pred_intercepts.get("y_intercept")
        if pred_yi is not None:
            try:
                yi_f = float(pred_yi)
                y_at_0 = float(fn(0.0))
                if np.isfinite(y_at_0):
                    checks_total += 1
                    if abs(y_at_0 - yi_f) < 1.0:
                        checks_passed += 1
                        detail_parts.append(f"f(0)={y_at_0:.2f}≈yi={yi_f} OK")
                    else:
                        detail_parts.append(f"f(0)={y_at_0:.2f}≠yi={yi_f}")
            except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                pass

    # Check 3: Expression value at extrema positions
    pred_extrema = pred_full.get("extrema", [])
    if isinstance(pred_extrema, list):
        for ext in pred_extrema:
            if not isinstance(ext, dict):
                continue
            try:
                ex = float(ext.get("x", 0))
                ey = float(ext.get("y", 0))
                y_at_ex = float(fn(ex))
                if np.isfinite(y_at_ex):
                    checks_total += 1
                    if abs(y_at_ex - ey) < 1.5:
                        checks_passed += 1
                        detail_parts.append(f"f({ex:.1f})={y_at_ex:.2f}≈{ey} OK")
                    else:
                        detail_parts.append(f"f({ex:.1f})={y_at_ex:.2f}≠{ey}")
            except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                pass

    if checks_total == 0:
        return {"correct": True, "score": 100.0,
                "pred_display": "no checks possible",
                "gt_display": "self-consistency",
                "details": "No L1 features to verify against"}

    score = round(checks_passed / checks_total * 100, 3)
    return {
        "correct": score >= 75.0,
        "score": score,
        "pred_display": f"{checks_passed}/{checks_total} consistent",
        "gt_display": "self-consistency",
        "details": "; ".join(detail_parts[:5]),
    }


_SUB_DISPATCH = {
    "expression": evaluate_expression,
    "domain": evaluate_domain,
    "range": evaluate_range,
    "predictions": evaluate_predictions,
    "rate_of_change": evaluate_rate_of_change,
    "intercepts": evaluate_intercepts,
    "extrema": evaluate_extrema,
    "monotonicity": evaluate_monotonicity,
    "symmetry": evaluate_symmetry,
    "judge_inference_score": None,
}


# ============================================================
# Main experiment scoring groups
# ============================================================

SUB_QUESTION_KEYS = [
    "expression", "domain", "range",
    "predictions", "rate_of_change",
    "intercepts", "extrema", "monotonicity", "symmetry",
]

# All scored dimensions — equal weight (1/N each).
# 10 dimensions aligned with the paper (coordinates are M0 intermediate data, not scored).
SCORED_DIMENSIONS = [
    "intercepts", "extrema",
    "predictions", "rate_of_change", "monotonicity", "symmetry",
    "expression", "domain", "range",
    "judge_inference_score",
]

_N_DIMS = len(SCORED_DIMENSIONS)
_EQUAL_W = round(1.0 / _N_DIMS, 6)

DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    dim: _EQUAL_W for dim in SCORED_DIMENSIONS
}

# Internal config used by evaluate_comprehensive (flat: one group per dimension).
DEFAULT_GROUP_CONFIG = {
    dim: {"weight": _EQUAL_W, "dimensions": {dim: 1.0}}
    for dim in SCORED_DIMENSIONS
}

# Backward-compat aliases.
GROUP_STRUCTURE = {dim: [dim] for dim in SCORED_DIMENSIONS}
LAYER_STRUCTURE = GROUP_STRUCTURE
DEFAULT_LAYER_CONFIG = DEFAULT_GROUP_CONFIG

_JUDGE_SUB_KEYS = (
    "factual_accuracy",
    "inference_quality",
    "reasoning_coherence",
    "evidence_support",
)


def evaluate_judge_response(raw_scores: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise Judge LLM sub-scores (each 0-5) into a single 0-100 score.

    Returns ``{"judge_inference_score": float, "sub_scores": dict, "details": str}``.
    """
    total = 0.0
    for k in _JUDGE_SUB_KEYS:
        total += float(raw_scores.get(k, 0))
    score = round(total / (len(_JUDGE_SUB_KEYS) * 5) * 100, 3)
    return {
        "judge_inference_score": score,
        "judge_score": score,
        "sub_scores": {k: raw_scores.get(k, 0) for k in _JUDGE_SUB_KEYS},
        "details": raw_scores.get("comments", ""),
    }


def evaluate_comprehensive(
    pred: Dict[str, Any],
    gt: Dict[str, Any],
    layer_config: Optional[Dict] = None,
    judge_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate all dimensions and compute per-group + overall scores.

    Args:
        pred: Parsed model answers (6 keys).
        gt: Ground truth dict.
        layer_config: ``{layer_name: {weight, dimensions: {dim: w}}}``
                      from ``default_config.yaml``.  Falls back to
                      ``DEFAULT_LAYER_CONFIG``.
        judge_result: Output of ``evaluate_judge_response``. Optional.

    Returns::

        {
            "sub_scores":   {dim: {...}, ...},
            "group_scores": {group: {"score": float, "dimensions": {dim: float}}, ...},
            "total_score":  float (0~100),
            "weights":      flat {dim: effective_weight},
        }
    """
    lc = layer_config or DEFAULT_GROUP_CONFIG

    # --- evaluate all automatic dimensions ---
    _LIST_KEYS = {"extrema", "monotonicity"}
    _DICT_KEYS = {"predictions", "intercepts"}
    _NULLABLE_KEYS = {"rate_of_change"}

    sub_scores: Dict[str, Any] = {}
    for key in SUB_QUESTION_KEYS:
        if key in _LIST_KEYS:
            pred_val = pred.get(key, [])
        elif key in _DICT_KEYS:
            pred_val = pred.get(key, {})
        elif key in _NULLABLE_KEYS:
            pred_val = pred.get(key)
        else:
            pred_val = pred.get(key, "")

        if key == "predictions":
            gt_val = gt.get("predictions") or gt.get("prediction_targets", {})
        elif key in _LIST_KEYS:
            gt_val = gt.get(key, [])
        elif key in _DICT_KEYS:
            gt_val = gt.get(key, {})
        elif key in _NULLABLE_KEYS:
            gt_val = gt.get(key)
        else:
            gt_val = gt.get(key, "")

        evaluator = _SUB_DISPATCH.get(key)
        if evaluator is None:
            sub_scores[key] = {"correct": False, "score": 0.0,
                               "pred_display": str(pred_val)[:80],
                               "gt_display": str(gt_val)[:80],
                               "details": "No evaluator"}
            continue
        sub_scores[key] = evaluator(pred_val, gt_val)

    # --- inject judge_inference_score ---
    if judge_result and "judge_inference_score" in judge_result:
        score_val = judge_result["judge_inference_score"]
        sub_scores["judge_inference_score"] = {
            "correct": score_val >= 75.0,
            "score": score_val,
            "pred_display": f"{score_val:.3f}/100",
            "gt_display": "Inference Judge rubric",
            "details": judge_result.get("details", ""),
        }
    else:
        sub_scores["judge_inference_score"] = {
            "correct": False, "score": 0.0,
            "skipped": True,
            "pred_display": "N/A", "gt_display": "Judge not run",
            "details": "Judge evaluation skipped",
        }

    # --- per-group aggregation ---
    # Skipped dimensions (e.g. Judge not run) score 0; no weight redistribution.
    group_scores: Dict[str, Any] = {}
    overall_weighted = 0.0
    overall_weight_sum = 0.0

    for group_name, group_cfg in lc.items():
        group_w = group_cfg.get("weight", 0)
        dims = group_cfg.get("dimensions", {})

        group_dim_scores: Dict[str, float] = {}
        group_weighted_sum = 0.0
        for dim, dw in dims.items():
            sc = sub_scores.get(dim, {}).get("score", 0.0)
            group_dim_scores[dim] = sc
            group_weighted_sum += sc * dw

        dim_total_w = sum(dims.values()) or 1.0
        group_score = round(group_weighted_sum / dim_total_w, 3)
        group_scores[group_name] = {
            "score": group_score,
            "dimensions": group_dim_scores,
        }
        overall_weighted += group_score * group_w
        overall_weight_sum += group_w

    total_score = round(overall_weighted / overall_weight_sum, 3) if overall_weight_sum else 0.0

    # flat weights for backward compat
    flat_weights: Dict[str, float] = {}
    for group_cfg in lc.values():
        lw = group_cfg.get("weight", 0)
        dims = group_cfg.get("dimensions", {})
        dtw = sum(dims.values()) or 1.0
        for dim, dw in dims.items():
            flat_weights[dim] = lw * (dw / dtw)

    # --- dimension_scores: flat {dim: 0~100} for radar charts ---
    dimension_scores = {
        dim: sub_scores.get(dim, {}).get("score", 0.0)
        for dim in SCORED_DIMENSIONS
    }

    return {
        "sub_scores": sub_scores,
        "group_scores": group_scores,
        "layer_scores": group_scores,
        "total_score": total_score,
        "weights": flat_weights,
        "dimension_scores": dimension_scores,
    }


# ============================================================
# Legacy single-question interface (backward compatible)
# ============================================================

_DISPATCH = {
    "domain": lambda pred, gt, **kw: evaluate_domain(pred, gt),
    "range": lambda pred, gt, **kw: evaluate_range(pred, gt),
    "expr": lambda pred, gt, **kw: evaluate_expression(pred, gt, **kw),
}


def evaluate_answer(
    question_type: str,
    pred_raw: str,
    gt_answer: str,
    **kwargs,
) -> Dict[str, Any]:
    """Legacy entry point for single-question evaluation."""
    fn = _DISPATCH.get(question_type)
    if fn is None:
        return {"correct": False, "score": 0.0, "pred_display": pred_raw[:100],
                "gt_display": gt_answer, "details": f"Unknown question type: {question_type}"}
    return fn(pred_raw, gt_answer, **kwargs)
