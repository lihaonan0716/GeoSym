"""
GeoSym Numerical Analyzer: compute a structured analysis report from
an expression string using SymPy/NumPy — no LLM involved.

This is the core computational component of the active GeoSym fusion pipeline,
producing numerical evidence that an LLM then uses for formal reasoning.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import sympy
    from sympy import (
        Symbol, oo, nan, zoo, S,
        solve, solveset, diff, limit,
        simplify, sympify, Reals,
        Interval, Union, FiniteSet,
        Piecewise, Abs,
    )
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


_x = Symbol("x") if HAS_SYMPY else None

_PARSE_TRANSFORMATIONS = (
    standard_transformations
    + (implicit_multiplication_application, convert_xor)
) if HAS_SYMPY else None


def _safe_parse(expr_str: str):
    """Parse expression string to SymPy expression, return None on failure."""
    if not HAS_SYMPY or not expr_str:
        return None

    s = expr_str.strip()
    s = re.sub(r'^[yf]\s*\(?\s*x?\s*\)?\s*=\s*', '', s, flags=re.IGNORECASE)
    s = s.replace('^', '**')
    s = s.replace('−', '-')
    s = s.replace('×', '*')
    s = s.replace('÷', '/')
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)

    try:
        return parse_expr(s, local_dict={"x": _x, "e": sympy.E, "pi": sympy.pi},
                          transformations=_PARSE_TRANSFORMATIONS)
    except Exception:
        pass

    try:
        return sympify(s, locals={"x": _x})
    except Exception:
        return None


def _safe_eval(expr, x_val: float) -> Optional[float]:
    """Evaluate expression at a point, return None if undefined."""
    try:
        val = complex(expr.subs(_x, x_val))
        if val.imag != 0:
            return None
        r = val.real
        if math.isnan(r) or math.isinf(r):
            return None
        return round(r, 6)
    except Exception:
        return None


def _find_singularities(expr, x_range: Tuple[float, float]) -> List[float]:
    """Find points where the function is undefined (denominator = 0, etc.)."""
    results = []
    try:
        combined = sympy.cancel(expr)
        denom = sympy.denom(combined)
        if denom != 1:
            sols = solve(denom, _x)
            for s in sols:
                try:
                    v = float(s)
                    if x_range[0] <= v <= x_range[1]:
                        results.append(round(v, 6))
                except (TypeError, ValueError):
                    pass
    except Exception:
        pass
    return sorted(set(results))


def _find_zeros(expr, x_range: Tuple[float, float]) -> List[float]:
    """Find zeros of expression via SymPy solve."""
    results = []
    try:
        sols = solve(expr, _x)
        for s in sols:
            try:
                v = float(s)
                if x_range[0] - 0.5 <= v <= x_range[1] + 0.5 and abs(v.imag if hasattr(v, 'imag') else 0) < 1e-9:
                    results.append(round(v, 6))
            except (TypeError, ValueError):
                pass
    except Exception:
        pass

    if not results:
        try:
            f_np = sympy.lambdify(_x, expr, modules=["numpy"])
            xs = np.linspace(x_range[0], x_range[1], 2000)
            ys = f_np(xs)
            ys = np.where(np.isfinite(ys), ys, np.nan)
            for i in range(len(ys) - 1):
                if np.isnan(ys[i]) or np.isnan(ys[i + 1]):
                    continue
                if ys[i] * ys[i + 1] <= 0:
                    z = xs[i] - ys[i] * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
                    results.append(round(z, 4))
        except Exception:
            pass

    return sorted(set(results))


def _find_critical_points(expr, x_range: Tuple[float, float]) -> List[Dict[str, Any]]:
    """Find local extrema via f'(x) = 0."""
    results = []
    try:
        fp = diff(expr, _x)
        sols = solve(fp, _x)
        fpp = diff(fp, _x)

        for s in sols:
            try:
                xv = float(s)
                if x_range[0] <= xv <= x_range[1]:
                    yv = _safe_eval(expr, xv)
                    if yv is None:
                        continue
                    second = _safe_eval(fpp, xv)
                    if second is not None and second > 0:
                        ext_type = "min"
                    elif second is not None and second < 0:
                        ext_type = "max"
                    else:
                        ext_type = "inflection"
                    results.append({"x": round(xv, 4), "y": round(yv, 4), "type": ext_type})
            except (TypeError, ValueError):
                pass
    except Exception:
        pass
    return results


def _compute_limits(expr) -> Dict[str, Optional[str]]:
    """Compute limits as x -> +/-inf for asymptotic behavior."""
    result = {}
    for direction, label in [(oo, "pos_inf"), (-oo, "neg_inf")]:
        try:
            lim = limit(expr, _x, direction)
            if lim in (oo, -oo, zoo, nan):
                result[label] = str(lim)
            else:
                result[label] = str(round(float(lim), 6))
        except Exception:
            result[label] = "undefined"
    return result


def _sample_values(expr, x_range: Tuple[float, float],
                   singularities: List[float], n_samples: int = 8) -> Dict[str, Optional[float]]:
    """Sample function values at evenly spaced and notable points."""
    lo, hi = x_range
    step = (hi - lo) / (n_samples - 1) if n_samples > 1 else 1
    xs = [round(lo + i * step, 2) for i in range(n_samples)]

    for s in singularities:
        for offset in [-0.1, 0.1]:
            v = round(s + offset, 2)
            if lo <= v <= hi and v not in xs:
                xs.append(v)
    xs = sorted(set(xs))

    samples = {}
    for xv in xs:
        if any(abs(xv - s) < 1e-6 for s in singularities):
            samples[str(xv)] = None
        else:
            samples[str(xv)] = _safe_eval(expr, xv)
    return samples


def analyze_expression(
    expr_str: str,
    x_range: Tuple[float, float] = (-10, 10),
) -> Dict[str, Any]:
    """
    Perform pure numerical/symbolic analysis on a function expression.
    No LLM involved — this is GeoSym's computational contribution.

    Args:
        expr_str: Function expression string (e.g. "x^2 - 4", "1/(x-3) + 1").
        x_range: Analysis domain (default [-10, 10]).

    Returns:
        {
            "expression": str,
            "parsed_ok": bool,
            "samples": {str_x: float_or_None, ...},
            "zeros": [float, ...],
            "singularities": [float, ...],
            "critical_points": [{"x":, "y":, "type":}, ...],
            "y_limits": {"neg_inf": str, "pos_inf": str},
            "report_text": str,
        }
    """
    if not HAS_SYMPY:
        return {
            "expression": expr_str,
            "parsed_ok": False,
            "samples": {},
            "zeros": [],
            "singularities": [],
            "critical_points": [],
            "y_limits": {},
            "report_text": f"[SymPy not available] Expression: {expr_str}",
        }

    expr = _safe_parse(expr_str)
    if expr is None:
        return {
            "expression": expr_str,
            "parsed_ok": False,
            "samples": {},
            "zeros": [],
            "singularities": [],
            "critical_points": [],
            "y_limits": {},
            "report_text": f"[Parse failed] Expression: {expr_str}",
        }

    singularities = _find_singularities(expr, x_range)
    zeros = _find_zeros(expr, x_range)
    critical_points = _find_critical_points(expr, x_range)
    y_limits = _compute_limits(expr)
    samples = _sample_values(expr, x_range, singularities)

    report = _build_report_text(expr_str, samples, zeros, singularities,
                                critical_points, y_limits)

    return {
        "expression": expr_str,
        "parsed_ok": True,
        "samples": samples,
        "zeros": zeros,
        "singularities": singularities,
        "critical_points": critical_points,
        "y_limits": y_limits,
        "report_text": report,
    }


def _build_report_text(
    expr_str: str,
    samples: Dict[str, Optional[float]],
    zeros: List[float],
    singularities: List[float],
    critical_points: List[Dict[str, Any]],
    y_limits: Dict[str, Optional[str]],
) -> str:
    """Build a human-readable analysis report for the LLM prompt."""
    lines = [f"Function: f(x) = {expr_str}", "", "Numerical Analysis:"]

    sample_parts = []
    for x_str, y_val in list(samples.items())[:10]:
        if y_val is None:
            sample_parts.append(f"f({x_str})=undefined")
        else:
            sample_parts.append(f"f({x_str})={y_val}")
    if sample_parts:
        lines.append(f"- Sample values: {', '.join(sample_parts)}")

    if zeros:
        lines.append(f"- Zeros found at: x = {', '.join(str(z) for z in zeros)}")
    else:
        lines.append("- No real zeros detected in the analysis range")

    if singularities:
        lines.append(f"- Singularities (function undefined) at: x = {', '.join(str(s) for s in singularities)}")
    else:
        lines.append("- No singularities detected (function appears continuous)")

    for cp in critical_points:
        label = {"min": "local minimum", "max": "local maximum", "inflection": "inflection point"}
        lines.append(f"- {label.get(cp['type'], cp['type'])} at ({cp['x']}, {cp['y']})")

    if y_limits:
        neg = y_limits.get("neg_inf", "?")
        pos = y_limits.get("pos_inf", "?")
        lines.append(f"- Asymptotic behavior: f(x) -> {neg} as x -> -inf, f(x) -> {pos} as x -> +inf")

    return "\n".join(lines)
