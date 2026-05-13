"""
Metrics for evaluating function expression recognition.

Compares predicted function expressions against ground truth using
scientifically established metrics from symbolic regression literature:

  - R² (Coefficient of Determination): Primary numerical similarity.
    Standard in statistics and used in SRBench (La Cava et al., NeurIPS 2021).
  - Symbolic Equivalence: Exact algebraic match via SymPy simplification.
    Used in AI Feynman (Udrescu & Tegmark, Science Advances 2020).
  - Function Type Accuracy: Classification accuracy for function type.
  - NRMSE (Normalized RMSE): Scale-independent numerical error.
    Used in SRBench benchmark.

Multi-curve matching uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment)
for optimal bipartite assignment, standard in object detection evaluation (e.g. DETR).
"""

from __future__ import annotations

import math
import re
import signal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import sympy
    from sympy import Abs, Rational, exp, log, oo, sqrt, symbols, sympify
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    _SYMPY_OK = True
except ImportError:
    _SYMPY_OK = False

_SCIPY_OK = None  # lazy-checked on first use
_linear_sum_assignment = None


def _ensure_scipy():
    """Lazy-load scipy to avoid 111 MB of IDE indexing at import time."""
    global _SCIPY_OK, _linear_sum_assignment
    if _SCIPY_OK is not None:
        return
    try:
        from scipy.optimize import linear_sum_assignment as _lsa
        _linear_sum_assignment = _lsa
        _SCIPY_OK = True
    except ImportError:
        _SCIPY_OK = False


# ============================================================
# Constants
# ============================================================

_FUNC_TYPES = [
    "linear",
    "quadratic",
    "cubic",
    "exponential",
    "logarithmic",
    "rational",
    "absolute",
    "piecewise_linear",
    "unknown",
]

_N_SAMPLES = 200  # number of evaluation points for numerical metrics


# ============================================================
# Expression Parsing
# ============================================================

def _normalize_expr_str(s: str) -> str:
    """Normalize an expression string before SymPy parsing.

    Handles plain text, caret notation, and LaTeX fragments that commonly
    appear in ground-truth labels and LLM outputs.
    """
    s = s.strip()
    # Unicode normalization (LLMs and GT labels may use these)
    s = s.replace("\u2212", "-")   # − (EN DASH minus)
    s = s.replace("\u00d7", "*")   # × (multiplication sign)
    s = s.replace("\u00b7", "*")   # · (middle dot)
    s = s.replace("\u2013", "-")   # – (EN DASH)
    s = s.replace("\u2014", "-")   # — (EM DASH)
    s = s.replace("\u00b2", "**2") # ² (superscript 2)
    s = s.replace("\u00b3", "**3") # ³ (superscript 3)
    # Strip surrounding $ delimiters
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    # Remove leading y= or f(x)=
    s = re.sub(r"^[yf]\s*\(?\s*x?\s*\)?\s*=\s*", "", s, flags=re.IGNORECASE)

    # --- LaTeX structural commands ---
    # \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
    # \sqrt[n]{expr} -> (expr)**(1/(n))
    s = re.sub(r"\\sqrt\[([^\]]+)\]\{([^}]+)\}", r"(\2)**(1/(\1))", s)
    # \sqrt{expr} -> sqrt(expr)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    # \left( \right) -> ( )
    s = s.replace("\\left(", "(").replace("\\right)", ")")
    s = s.replace("\\left[", "[").replace("\\right]", "]")
    # \cdot, \times -> *
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    # LaTeX function names: \ln -> log, \log -> log, \sin -> sin, etc.
    s = re.sub(r"\\ln\b", "log", s)
    s = re.sub(r"\\log\b", "log", s)
    s = re.sub(r"\\exp\b", "exp", s)
    s = re.sub(r"\\sin\b", "sin", s)
    s = re.sub(r"\\cos\b", "cos", s)
    s = re.sub(r"\\tan\b", "tan", s)
    s = re.sub(r"\\pi\b", "pi", s)
    # Strip any remaining backslashes (e.g. \, spacing)
    s = s.replace("\\", "")

    # --- Caret / brace exponents ---
    # ^{expr} -> **(expr)  (must come BEFORE generic ^ replacement)
    s = re.sub(r"\^\{([^}]+)\}", r"**(\1)", s)
    # Plain ^ -> **
    s = s.replace("^", "**")

    # Remove any remaining LaTeX braces
    s = s.replace("{", "").replace("}", "")

    # --- Common notation normalization ---
    s = s.replace("ln(", "log(")
    s = s.replace("LOG(", "log(").replace("EXP(", "exp(")
    s = s.replace("SQRT(", "sqrt(")
    s = s.replace("ABS(", "Abs(").replace("abs(", "Abs(")
    s = s.replace("|x|", "Abs(x)")
    s = re.sub(r"\|([^|]+)\|", r"Abs(\1)", s)
    s = s.replace("SIN(", "sin(").replace("COS(", "cos(").replace("TAN(", "tan(")
    # e^(...) -> exp(...)
    s = re.sub(r"\be\s*\*\*\s*\(", "exp(", s)
    s = re.sub(r"\be\s*\*\*\s*x", "exp(x)", s)
    return s


def parse_expression(expr_str: str) -> Optional[Any]:
    """
    Parse an expression string into a SymPy expression.

    Handles various formats output by LLMs:
      - Standard SymPy notation: ``x**2 + 2*x + 1``
      - Caret notation: ``x^2 + 2x + 1``
      - LaTeX fragments: ``\\frac{1}{x}``

    Returns:
        SymPy expression or None if parsing fails.
    """
    if not _SYMPY_OK:
        return None

    s = _normalize_expr_str(expr_str)
    if not s:
        return None

    x = symbols("x")
    local_dict = {"x": x, "e": sympy.E, "pi": sympy.pi,
                  "Piecewise": sympy.Piecewise}

    # Strategy 1: direct sympify
    try:
        expr = sympify(s, locals=local_dict)
        return expr
    except Exception:
        pass

    # Strategy 2: parse_expr with implicit multiplication
    try:
        transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )
        expr = parse_expr(s, local_dict=local_dict, transformations=transformations)
        return expr
    except Exception:
        pass

    # Strategy 3: LaTeX-like fragments (simple cases)
    try:
        # \frac{a}{b} -> (a)/(b)
        s2 = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
        s2 = s2.replace("\\", "")
        expr = sympify(s2, locals=local_dict)
        return expr
    except Exception:
        pass

    return None


# ============================================================
# Per-Curve Metrics
# ============================================================

def _safe_evaluate(expr, x_vals: np.ndarray) -> np.ndarray:
    """Evaluate a SymPy expression at given x values, returning NaN for invalid points."""
    if not _SYMPY_OK:
        return np.full_like(x_vals, np.nan, dtype=float)

    x = symbols("x")
    f = sympy.lambdify(x, expr, modules=["numpy"])

    # Suppress numpy warnings for domain errors
    with np.errstate(all="ignore"):
        try:
            y = np.array(f(x_vals), dtype=float)
        except Exception:
            y = np.full_like(x_vals, np.nan, dtype=float)

    # Replace inf with nan
    y[~np.isfinite(y)] = np.nan
    return y


def compute_r2(
    pred_expr,
    gt_expr,
    x_range: Tuple[float, float],
    n_samples: int = _N_SAMPLES,
) -> float:
    """
    Compute R² (coefficient of determination) between predicted and ground truth
    expressions over the specified x range.

    R² = 1 - SS_res / SS_tot
      where SS_res = sum((y_gt - y_pred)²)
            SS_tot = sum((y_gt - mean(y_gt))²)

    Points where either expression is NaN/Inf are excluded.

    Returns:
        R² score. 1.0 = perfect, 0.0 = as good as predicting mean,
        negative = worse than mean.
    """
    x_vals = np.linspace(x_range[0], x_range[1], n_samples)

    y_gt = _safe_evaluate(gt_expr, x_vals)
    y_pred = _safe_evaluate(pred_expr, x_vals)

    # Keep only valid points
    valid = np.isfinite(y_gt) & np.isfinite(y_pred)
    if valid.sum() < 3:
        return float("-inf")

    y_gt_v = y_gt[valid]
    y_pred_v = y_pred[valid]

    ss_tot = np.sum((y_gt_v - np.mean(y_gt_v)) ** 2)
    if ss_tot < 1e-12:
        # Constant function
        ss_res = np.sum((y_gt_v - y_pred_v) ** 2)
        return 1.0 if ss_res < 1e-12 else 0.0

    ss_res = np.sum((y_gt_v - y_pred_v) ** 2)
    return float(1.0 - ss_res / ss_tot)


def compute_nrmse(
    pred_expr,
    gt_expr,
    x_range: Tuple[float, float],
    n_samples: int = _N_SAMPLES,
) -> float:
    """
    Compute Normalized RMSE between predicted and ground truth expressions.

    NRMSE = RMSE / (max(y_gt) - min(y_gt))

    Returns:
        NRMSE value. 0.0 = perfect. Lower is better.
    """
    x_vals = np.linspace(x_range[0], x_range[1], n_samples)

    y_gt = _safe_evaluate(gt_expr, x_vals)
    y_pred = _safe_evaluate(pred_expr, x_vals)

    valid = np.isfinite(y_gt) & np.isfinite(y_pred)
    if valid.sum() < 3:
        return float("inf")

    y_gt_v = y_gt[valid]
    y_pred_v = y_pred[valid]

    rmse = np.sqrt(np.mean((y_gt_v - y_pred_v) ** 2))
    y_range = y_gt_v.max() - y_gt_v.min()

    if y_range < 1e-12:
        return 0.0 if rmse < 1e-12 else float("inf")

    return float(rmse / y_range)


def compute_pearson_r(
    pred_expr,
    gt_expr,
    x_range: Tuple[float, float],
    n_samples: int = _N_SAMPLES,
) -> Optional[float]:
    """Pearson correlation between predicted and GT y-values.

    Measures *shape similarity* independent of vertical shift and scale.
    For example, ``y = x**2`` and ``y = 3*x**2 + 5`` yield r ~ 1.0.

    Returns:
        Pearson r in [-1, 1], or None if insufficient valid points or
        constant y-values (zero variance).
    """
    x_vals = np.linspace(x_range[0], x_range[1], n_samples)

    y_gt = _safe_evaluate(gt_expr, x_vals)
    y_pred = _safe_evaluate(pred_expr, x_vals)

    valid = np.isfinite(y_gt) & np.isfinite(y_pred)
    if valid.sum() < 5:
        return None

    y_gt_v = y_gt[valid]
    y_pred_v = y_pred[valid]

    gt_std = np.std(y_gt_v)
    pred_std = np.std(y_pred_v)
    if gt_std < 1e-12 or pred_std < 1e-12:
        return None

    r = float(np.corrcoef(y_gt_v, y_pred_v)[0, 1])
    if not np.isfinite(r):
        return None
    return r


def symbolic_equivalence(pred_expr, gt_expr, timeout: int = 5) -> bool:
    """
    Check if two expressions are symbolically equivalent using SymPy.

    Uses simplify(pred - gt) == 0 with a timeout to avoid hanging.

    Returns:
        True if expressions are algebraically equivalent, False otherwise.
    """
    if not _SYMPY_OK:
        return False

    if pred_expr is None or gt_expr is None:
        return False

    try:
        diff = sympy.simplify(pred_expr - gt_expr)
        return diff == 0
    except Exception:
        return False


def classify_type(expr) -> str:
    """
    Heuristically classify a SymPy expression into one of the standard function types.

    Returns one of the types listed in _FUNC_TYPES.
    """
    if not _SYMPY_OK or expr is None:
        return "unknown"

    x = symbols("x")

    # Piecewise → piecewise_linear (checked first, before string-based heuristics)
    if isinstance(expr, sympy.Piecewise):
        return "piecewise_linear"

    try:
        expr_expanded = sympy.expand(expr)
    except Exception:
        expr_expanded = expr

    s = str(expr)

    # Check for Abs
    if "Abs" in s:
        return "absolute"

    # Check for exp / b^x patterns
    if "exp" in s or (sympy.E in expr.atoms() and x in expr.free_symbols):
        return "exponential"
    try:
        for a in sympy.preorder_traversal(expr):
            if isinstance(a, sympy.Pow) and a.args[1].has(x):
                return "exponential"
    except Exception:
        pass

    # Check for log
    if "log" in s:
        return "logarithmic"

    # Check for rational (a/(x-h) patterns)
    try:
        if expr.has(sympy.Pow):
            has_x_denom = False
            for a in sympy.preorder_traversal(expr):
                if isinstance(a, sympy.Pow):
                    base, exponent = a.args
                    if base.has(x) and exponent == -1:
                        has_x_denom = True
            if has_x_denom:
                return "rational"
    except Exception:
        pass

    # Check polynomial degree
    try:
        poly = sympy.Poly(expr_expanded, x)
        degree = poly.degree()
        if degree == 1:
            return "linear"
        elif degree == 2:
            return "quadratic"
        elif degree == 3:
            return "cubic"
    except (sympy.PolynomialError, sympy.GeneratorsNeeded):
        pass

    # Fallback: check if it's a constant (linear with a=0)
    if x not in expr.free_symbols:
        return "linear"

    return "unknown"


# ============================================================
# Multi-Curve Matching (Hungarian Algorithm)
# ============================================================

def match_curves(
    pred_exprs: List,
    gt_exprs: List,
    x_range: Tuple[float, float],
) -> Dict[str, Any]:
    """
    Optimally match predicted expressions to ground truth using the Hungarian
    algorithm based on R² scores.

    Args:
        pred_exprs: List of parsed SymPy expressions (predictions).
        gt_exprs: List of parsed SymPy expressions (ground truth).
        x_range: (x_min, x_max) for numerical evaluation.

    Returns:
        {
            "matches": [(gt_idx, pred_idx, r2), ...],
            "unmatched_gt": [gt_idx, ...],     # missed
            "unmatched_pred": [pred_idx, ...],  # false positives
        }
    """
    n_gt = len(gt_exprs)
    n_pred = len(pred_exprs)

    if n_gt == 0 and n_pred == 0:
        return {"matches": [], "unmatched_gt": [], "unmatched_pred": []}
    if n_gt == 0:
        return {"matches": [], "unmatched_gt": [], "unmatched_pred": list(range(n_pred))}
    if n_pred == 0:
        return {"matches": [], "unmatched_gt": list(range(n_gt)), "unmatched_pred": []}

    # Build cost matrix: cost = 1 - R² (lower is better)
    # For invalid R² (=-inf), use a large cost
    cost_matrix = np.zeros((n_gt, n_pred))
    r2_matrix = np.zeros((n_gt, n_pred))

    for i, gt_e in enumerate(gt_exprs):
        for j, pred_e in enumerate(pred_exprs):
            if gt_e is not None and pred_e is not None:
                r2 = compute_r2(pred_e, gt_e, x_range)
                r2_matrix[i, j] = r2
                # Clamp R² to [-10, 1] to avoid extreme costs
                cost_matrix[i, j] = 1.0 - max(r2, -10.0)
            else:
                r2_matrix[i, j] = float("-inf")
                cost_matrix[i, j] = 11.0  # high cost for None

    _ensure_scipy()
    if _SCIPY_OK:
        gt_indices, pred_indices = _linear_sum_assignment(cost_matrix)
    else:
        # Fallback: greedy matching
        gt_indices = list(range(min(n_gt, n_pred)))
        pred_indices = list(range(min(n_gt, n_pred)))

    matches = []
    matched_gt = set()
    matched_pred = set()

    for gi, pi in zip(gt_indices, pred_indices):
        r2 = r2_matrix[gi, pi]
        matches.append((int(gi), int(pi), float(r2)))
        matched_gt.add(int(gi))
        matched_pred.add(int(pi))

    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_pred = [j for j in range(n_pred) if j not in matched_pred]

    return {
        "matches": matches,
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": unmatched_pred,
    }


# ============================================================
# Per-Image Aggregate Metrics
# ============================================================

def compute_image_expr_metrics(
    pred_strs: List[str],
    gt_functions: List[Dict[str, Any]],
    x_range: Tuple[float, float],
) -> Dict[str, Any]:
    """
    Compute expression recognition metrics for a single image.

    Args:
        pred_strs: List of predicted expression strings from LLM.
        gt_functions: List of ground truth function dicts (with 'expression_sympy', 'type').
        x_range: (x_min, x_max) for numerical evaluation.

    Returns:
        {
            "num_gt": int,
            "num_pred": int,
            "num_matched": int,
            "num_missed": int,
            "num_extra": int,
            "mean_r2": float,
            "sym_equiv_rate": float,
            "type_accuracy": float,
            "mean_nrmse": float,
            "per_curve": [{
                "gt_expr_str": str,
                "pred_expr_str": str,
                "gt_type": str,
                "pred_type": str,
                "r2": float,
                "nrmse": float,
                "type_match": bool,
                "sym_equiv": bool,
            }, ...],
        }
    """
    # Parse ground truth expressions
    gt_exprs = []
    gt_types = []
    gt_strs = []
    for func in gt_functions:
        expr_str = func.get("expression_sympy", "")
        gt_strs.append(expr_str)
        gt_exprs.append(parse_expression(expr_str))
        gt_types.append(func.get("type", "unknown"))

    # Parse predicted expressions
    pred_exprs = []
    for s in pred_strs:
        pred_exprs.append(parse_expression(s))

    n_gt = len(gt_exprs)
    n_pred = len(pred_exprs)

    if n_gt == 0 and n_pred == 0:
        return {
            "num_gt": 0, "num_pred": 0, "num_matched": 0,
            "num_missed": 0, "num_extra": 0,
            "mean_r2": 1.0, "sym_equiv_rate": 1.0,
            "type_accuracy": 1.0, "mean_nrmse": 0.0,
            "num_sym_equiv": 0, "num_type_match": 0,
            "sum_r2": 0.0, "sum_nrmse": 0.0,
            "per_curve": [],
        }

    if n_gt == 0:
        return {
            "num_gt": 0, "num_pred": n_pred, "num_matched": 0,
            "num_missed": 0, "num_extra": n_pred,
            "mean_r2": 0.0, "sym_equiv_rate": 0.0,
            "type_accuracy": 0.0, "mean_nrmse": 0.0,
            "num_sym_equiv": 0, "num_type_match": 0,
            "sum_r2": 0.0, "sum_nrmse": 0.0,
            "per_curve": [],
        }

    # Match curves
    matching = match_curves(pred_exprs, gt_exprs, x_range)

    per_curve = []
    r2_scores = []
    nrmse_scores = []
    type_matches = []
    sym_equivs = []

    for gt_idx, pred_idx, r2 in matching["matches"]:
        gt_e = gt_exprs[gt_idx]
        pred_e = pred_exprs[pred_idx]

        nrmse = compute_nrmse(pred_e, gt_e, x_range) if (pred_e is not None and gt_e is not None) else float("inf")
        pred_type = classify_type(pred_e) if pred_e is not None else "unknown"
        gt_type = gt_types[gt_idx]
        type_match = pred_type == gt_type
        sym_eq = symbolic_equivalence(pred_e, gt_e)

        r2_clamped = max(r2, 0.0)  # clamp negative R² to 0 for averaging
        r2_scores.append(r2_clamped)
        nrmse_scores.append(nrmse if np.isfinite(nrmse) else 1.0)
        type_matches.append(type_match)
        sym_equivs.append(sym_eq)

        per_curve.append({
            "gt_expr_str": gt_strs[gt_idx],
            "pred_expr_str": pred_strs[pred_idx] if pred_idx < len(pred_strs) else "",
            "gt_type": gt_type,
            "pred_type": pred_type,
            "r2": r2,
            "nrmse": nrmse if np.isfinite(nrmse) else 1.0,
            "type_match": type_match,
            "sym_equiv": sym_eq,
        })

    # Handle unmatched GT (missed) — these do NOT contribute to sum_r2/sum_nrmse
    for gt_idx in matching["unmatched_gt"]:
        per_curve.append({
            "gt_expr_str": gt_strs[gt_idx],
            "pred_expr_str": "(missed)",
            "gt_type": gt_types[gt_idx],
            "pred_type": "-",
            "r2": 0.0,
            "nrmse": 0.0,
            "type_match": False,
            "sym_equiv": False,
        })

    num_matched = len(matching["matches"])
    num_missed = len(matching["unmatched_gt"])
    num_extra = len(matching["unmatched_pred"])

    mean_r2 = float(np.mean(r2_scores)) if r2_scores else 0.0
    mean_nrmse = float(np.mean(nrmse_scores)) if nrmse_scores else 0.0
    sym_equiv_rate = sum(sym_equivs) / len(sym_equivs) if sym_equivs else 0.0
    type_acc = sum(type_matches) / len(type_matches) if type_matches else 0.0

    # Raw counts for micro aggregation across images
    num_sym_equiv = int(sum(sym_equivs))
    num_type_match = int(sum(type_matches))
    sum_r2 = float(sum(r2_scores))
    sum_nrmse = float(sum(nrmse_scores))

    return {
        "num_gt": n_gt,
        "num_pred": n_pred,
        "num_matched": num_matched,
        "num_missed": num_missed,
        "num_extra": num_extra,
        "mean_r2": mean_r2,
        "sym_equiv_rate": sym_equiv_rate,
        "type_accuracy": type_acc,
        "mean_nrmse": mean_nrmse,
        # Raw counts for micro aggregation
        "num_sym_equiv": num_sym_equiv,
        "num_type_match": num_type_match,
        "sum_r2": sum_r2,
        "sum_nrmse": sum_nrmse,
        "per_curve": per_curve,
    }


# ============================================================
# Dataset-Level Aggregation
# ============================================================

def aggregate_expr_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate expression metrics across multiple images.

    Computes both Macro (Mean) and Micro aggregations:
      - Mean (Macro): per-image metrics averaged (each image has equal weight).
      - Micro: pooled across all matched curves (each curve has equal weight).

    Args:
        results: List of per-image metric dicts (from compute_image_expr_metrics).

    Returns:
        {
            "num_images": int,
            "total_gt": int, "total_pred": int,
            "total_matched": int, "total_missed": int, "total_extra": int,
            # Macro averages (per-image then average)
            "mean_r2": float, "mean_sym_equiv_rate": float,
            "mean_type_accuracy": float, "mean_nrmse": float,
            # Micro averages (pooled across all matched curves)
            "micro_r2": float, "micro_sym_equiv_rate": float,
            "micro_type_accuracy": float, "micro_nrmse": float,
        }
    """
    _empty = {
        "num_images": 0,
        "total_gt": 0, "total_pred": 0,
        "total_matched": 0, "total_missed": 0, "total_extra": 0,
        "mean_r2": 0.0, "mean_sym_equiv_rate": 0.0,
        "mean_type_accuracy": 0.0, "mean_nrmse": 0.0,
        "micro_r2": 0.0, "micro_sym_equiv_rate": 0.0,
        "micro_type_accuracy": 0.0, "micro_nrmse": 0.0,
    }
    if not results:
        return _empty

    n = len(results)
    total_matched = sum(r["num_matched"] for r in results)

    # For Mean (Macro): only include images that have at least one matched curve.
    # Images with num_matched==0 (e.g. API returned empty) are skipped to avoid
    # distorting the average with default zero values.
    valid = [r for r in results if r["num_matched"] > 0]
    n_valid = len(valid) if valid else 1  # avoid division by zero

    # Micro: pool raw counts across all curves
    total_sym_equiv = sum(r.get("num_sym_equiv", 0) for r in results)
    total_type_match = sum(r.get("num_type_match", 0) for r in results)
    total_sum_r2 = sum(r.get("sum_r2", 0.0) for r in results)
    total_sum_nrmse = sum(r.get("sum_nrmse", 0.0) for r in results)

    micro_r2 = total_sum_r2 / total_matched if total_matched > 0 else 0.0
    micro_sym = total_sym_equiv / total_matched if total_matched > 0 else 0.0
    micro_type = total_type_match / total_matched if total_matched > 0 else 0.0
    micro_nrmse = total_sum_nrmse / total_matched if total_matched > 0 else 0.0

    return {
        "num_images": n,
        "num_valid_images": len(valid),
        "total_gt": sum(r["num_gt"] for r in results),
        "total_pred": sum(r["num_pred"] for r in results),
        "total_matched": total_matched,
        "total_missed": sum(r["num_missed"] for r in results),
        "total_extra": sum(r["num_extra"] for r in results),
        # Macro (Mean) — only from images with at least one matched curve
        "mean_r2": sum(r["mean_r2"] for r in valid) / n_valid if valid else 0.0,
        "mean_sym_equiv_rate": sum(r["sym_equiv_rate"] for r in valid) / n_valid if valid else 0.0,
        "mean_type_accuracy": sum(r["type_accuracy"] for r in valid) / n_valid if valid else 0.0,
        "mean_nrmse": sum(r["mean_nrmse"] for r in valid) / n_valid if valid else 0.0,
        # Micro
        "micro_r2": micro_r2,
        "micro_sym_equiv_rate": micro_sym,
        "micro_type_accuracy": micro_type,
        "micro_nrmse": micro_nrmse,
    }
