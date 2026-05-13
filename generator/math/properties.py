"""
Compute mathematical properties for each function type.

Base properties: domain, range, zeros.
Extended properties (for three-layer evaluation):
    intercepts  — {x_intercepts: [...], y_intercept: float|None}
    extrema     — [{x, y, type: "max"|"min"}, ...]
    monotonicity — [{interval: [a,b], direction: "increasing"|"decreasing"}, ...]
    symmetry    — "even" | "odd" | "neither"

All functions are procedurally generated with known parameters, so we derive
properties directly from parameters rather than using generic symbolic solvers.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

_INF = r"+\infty"
_NINF = r"-\infty"


def _fmt_interval(left: str, a: str, b: str, right: str) -> str:
    return f"{left}{a},\\;{b}{right}"


def _union(*intervals: str) -> str:
    return r" \cup ".join(intervals)


def _real_line() -> str:
    return _fmt_interval("(", _NINF, _INF, ")")


def _solve_quadratic(a: float, b: float, c: float) -> List[float]:
    if a == 0:
        return [-c / b] if b != 0 else []
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    if disc == 0:
        return [-b / (2 * a)]
    sq = math.sqrt(disc)
    return sorted([(-b - sq) / (2 * a), (-b + sq) / (2 * a)])


def _round_if_close(v: float, tol: float = 1e-9) -> float:
    r = round(v)
    return float(r) if abs(v - r) < tol else round(v, 6)


# ============================================================
# Per-type property computers
# ============================================================

def _props_linear(p: Dict[str, Any]) -> Dict[str, Any]:
    a, b = p["a"], p["b"]
    zeros = [_round_if_close(-b / a)] if a != 0 else []
    return {"domain": _real_line(), "range": _real_line(), "zeros": zeros}


def _props_quadratic(p: Dict[str, Any]) -> Dict[str, Any]:
    a, h, k = p["a"], p["h"], p["k"]
    vy = _round_if_close(k)

    if a > 0:
        rng = _fmt_interval("[", f"{vy}", _INF, ")")
    else:
        rng = _fmt_interval("(", _NINF, f"{vy}", "]")

    # vertex form → standard form for zero-finding: a*x² + (-2ah)*x + (ah²+k)
    b_std = -2 * a * h
    c_std = a * h * h + k
    zeros = [_round_if_close(z) for z in _solve_quadratic(a, b_std, c_std)]
    return {"domain": _real_line(), "range": rng, "zeros": zeros}


def _props_absolute(p: Dict[str, Any]) -> Dict[str, Any]:
    a, h, k = p["a"], p["h"], p["k"]

    if a > 0:
        rng = _fmt_interval("[", f"{_round_if_close(k)}", _INF, ")")
    else:
        rng = _fmt_interval("(", _NINF, f"{_round_if_close(k)}", "]")

    zeros = []
    if a != 0:
        val = -k / a
        if val > 0:
            zeros = sorted([_round_if_close(h - val), _round_if_close(h + val)])
        elif val == 0:
            zeros = [_round_if_close(h)]

    return {"domain": _real_line(), "range": rng, "zeros": zeros}


def _props_cubic(p: Dict[str, Any]) -> Dict[str, Any]:
    """y = a(x-h)³ + b(x-h) + k.  Let u = x-h → au³+bu+k = 0."""
    a, b = p["a"], p["b"]
    h = p.get("h", 0)
    k = p.get("k", 0)

    if k == 0:
        zeros = [_round_if_close(h)]
        if a != 0 and b != 0:
            ratio = -b / a
            if ratio > 0:
                sq = math.sqrt(ratio)
                zeros = sorted([_round_if_close(h - sq), _round_if_close(h),
                                _round_if_close(h + sq)])
    else:
        try:
            import numpy as np
            roots = np.roots([a, 0, b, k])
            zeros = sorted(
                _round_if_close(r.real + h) for r in roots if abs(r.imag) < 1e-6
            )
        except ImportError:
            zeros = []

    return {"domain": _real_line(), "range": _real_line(), "zeros": zeros}


def _props_power(p: Dict[str, Any]) -> Dict[str, Any]:
    a, n = p["a"], p["n"]

    if n == 0.5:
        domain = _fmt_interval("[", "0", _INF, ")")
        if a > 0:
            rng = _fmt_interval("[", "0", _INF, ")")
        else:
            rng = _fmt_interval("(", _NINF, "0", "]")
        zeros = [0.0]
    elif n == 1 / 3:
        domain = _real_line()
        rng = _real_line()
        zeros = [0.0]
    elif n == 2:
        domain = _real_line()
        if a > 0:
            rng = _fmt_interval("[", "0", _INF, ")")
        else:
            rng = _fmt_interval("(", _NINF, "0", "]")
        zeros = [0.0]
    elif n == 3:
        domain = _real_line()
        rng = _real_line()
        zeros = [0.0]
    else:
        domain = _real_line()
        rng = _real_line()
        zeros = [0.0]

    return {"domain": domain, "range": rng, "zeros": zeros}


def _props_exponential(p: Dict[str, Any]) -> Dict[str, Any]:
    a, b = p["a"], p["b"]
    h = p.get("h", 0)
    k = p.get("k", p.get("c", 0))
    kv = _round_if_close(k)

    if a > 0:
        rng = _fmt_interval("(", f"{kv}", _INF, ")")
    else:
        rng = _fmt_interval("(", _NINF, f"{kv}", ")")

    zeros = []
    target = -k / a if a != 0 else None
    if target is not None and target > 0 and b > 0 and b != 1:
        z = h + math.log(target) / math.log(b)
        zeros = [_round_if_close(z)]

    return {"domain": _real_line(), "range": rng, "zeros": zeros}


def _props_logarithmic(p: Dict[str, Any]) -> Dict[str, Any]:
    a = p["a"]
    base = p.get("base", 2)
    h = p.get("h", None)
    k = p.get("k", None)
    if h is None:
        d = p.get("d", 0)
        h = -d
    if k is None:
        k = p.get("c", 0)

    hv = _round_if_close(h)
    domain = _fmt_interval("(", f"{hv}", _INF, ")")

    zeros = []
    try:
        x_zero = h + base ** (-k / a)
        if x_zero > h:
            zeros = [_round_if_close(x_zero)]
    except (ValueError, ZeroDivisionError, OverflowError):
        pass

    return {"domain": domain, "range": _real_line(), "zeros": zeros}


def _props_rational(p: Dict[str, Any]) -> Dict[str, Any]:
    a, h, k = p["a"], p["h"], p["k"]
    hv = _round_if_close(h)
    kv = _round_if_close(k)

    domain = _union(
        _fmt_interval("(", _NINF, f"{hv}", ")"),
        _fmt_interval("(", f"{hv}", _INF, ")"),
    )
    rng = _union(
        _fmt_interval("(", _NINF, f"{kv}", ")"),
        _fmt_interval("(", f"{kv}", _INF, ")"),
    )

    zeros = []
    if k != 0:
        z = h - a / k
        zeros = [_round_if_close(z)]

    return {"domain": domain, "range": rng, "zeros": zeros}


def _props_piecewise_linear(p: Dict[str, Any]) -> Dict[str, Any]:
    """y = s1*(x-h)+k for x<h, s2*(x-h)+k for x≥h"""
    s1, s2, h, k = p["s1"], p["s2"], p["h"], p["k"]

    zeros: List[Any] = []
    z1 = h - k / s1 if s1 != 0 else None
    z2 = h - k / s2 if s2 != 0 else None
    if z1 is not None and z1 < h:
        zeros.append(_round_if_close(z1))
    if z2 is not None and z2 >= h:
        zeros.append(_round_if_close(z2))
    zeros.sort()

    return {"domain": _real_line(), "range": _real_line(), "zeros": zeros}


def _props_semicircle(p: Dict[str, Any]) -> Dict[str, Any]:
    """y = a*sqrt(r² - (x-h)²) + k"""
    a, r, h, k = p["a"], p["r"], p["h"], p["k"]
    hv = _round_if_close(h)
    kv = _round_if_close(k)

    domain = _fmt_interval("[", f"{_round_if_close(h - r)}", f"{_round_if_close(h + r)}", "]")

    if a > 0:
        rng = _fmt_interval("[", f"{kv}", f"{_round_if_close(k + a * r)}", "]")
    else:
        rng = _fmt_interval("[", f"{_round_if_close(k + a * r)}", f"{kv}", "]")

    zeros: List[Any] = []
    if abs(k) <= abs(a) * r:
        disc = r * r - (k / a) ** 2
        if disc >= 0:
            sq = math.sqrt(disc)
            z1, z2 = h - sq, h + sq
            zeros = sorted(set(_round_if_close(z) for z in [z1, z2]))

    return {"domain": domain, "range": rng, "zeros": zeros}


# ============================================================
# Dispatch
# ============================================================

_DISPATCH = {
    "linear": _props_linear,
    "quadratic": _props_quadratic,
    "absolute": _props_absolute,
    "cubic": _props_cubic,
    "power": _props_power,
    "exponential": _props_exponential,
    "logarithmic": _props_logarithmic,
    "rational": _props_rational,
    "piecewise_linear": _props_piecewise_linear,
    "semicircle": _props_semicircle,
}


def compute_properties(
    func_type: str,
    params: Dict[str, Any],
    expr_sympy: str = "",
) -> Dict[str, Any]:
    """
    Compute domain, range, and zeros for a function with known type and parameters.

    Returns:
        {
            "domain": str,       # LaTeX interval notation
            "range": str,        # LaTeX interval notation
            "zeros": list[float] # numerical zeros
        }
    """
    fn = _DISPATCH.get(func_type)
    if fn is None:
        return {"domain": _real_line(), "range": _real_line(), "zeros": []}
    return fn(params)


# ============================================================
# Extended properties: intercepts, extrema, monotonicity, symmetry
# ============================================================

def _intercepts(func_type: str, params: Dict[str, Any],
                sampler, x_min: float, x_max: float) -> Dict[str, Any]:
    """Compute x-intercepts and y-intercept."""
    base = _DISPATCH.get(func_type, lambda p: {"zeros": []})
    zeros = base(params).get("zeros", [])
    x_ints = [z for z in zeros if x_min <= z <= x_max]

    y_int = None
    try:
        val = sampler(0.0)
        if val is not None and math.isfinite(val):
            y_int = _round_if_close(val)
    except Exception:
        pass

    return {"x_intercepts": x_ints, "y_intercept": y_int}


def _extrema(func_type: str, params: Dict[str, Any],
             x_min: float, x_max: float) -> List[Dict[str, Any]]:
    """Compute local extrema within the visible range.

    Returns list of {x, y, type} dicts.  Only implemented for types
    where extrema are analytically known.
    """
    results: List[Dict[str, Any]] = []

    if func_type == "quadratic":
        a, h, k = params["a"], params["h"], params["k"]
        if x_min <= h <= x_max:
            results.append({
                "x": _round_if_close(h), "y": _round_if_close(k),
                "type": "min" if a > 0 else "max",
            })

    elif func_type == "absolute":
        a, h, k = params["a"], params["h"], params["k"]
        if x_min <= h <= x_max:
            results.append({
                "x": _round_if_close(h), "y": _round_if_close(k),
                "type": "min" if a > 0 else "max",
            })

    elif func_type == "cubic":
        a, b = params["a"], params["b"]
        h = params.get("h", 0)
        k = params.get("k", 0)
        if a != 0 and b != 0:
            ratio = -b / (3 * a)
            if ratio > 0:
                uc = math.sqrt(ratio)
                for uv in [-uc, uc]:
                    xv = uv + h
                    if x_min <= xv <= x_max:
                        yv = a * uv**3 + b * uv + k
                        ext_type = "max" if a * 6 * uv < 0 else "min"
                        results.append({
                            "x": _round_if_close(xv),
                            "y": _round_if_close(yv),
                            "type": ext_type,
                        })

    elif func_type == "piecewise_linear":
        s1, s2, h, k = params["s1"], params["s2"], params["h"], params["k"]
        if x_min <= h <= x_max:
            if (s1 > 0 and s2 < 0) or (s1 < 0 and s2 > 0):
                ext_type = "max" if s1 > 0 else "min"
                results.append({
                    "x": _round_if_close(h), "y": _round_if_close(k),
                    "type": ext_type,
                })

    elif func_type == "semicircle":
        a, r, h, k = params["a"], params["r"], params["h"], params["k"]
        apex_y = k + a * r
        if x_min <= h <= x_max:
            results.append({
                "x": _round_if_close(h), "y": _round_if_close(apex_y),
                "type": "max" if a > 0 else "min",
            })

    return results


def _monotonicity(func_type: str, params: Dict[str, Any],
                  x_min: float, x_max: float) -> List[Dict[str, Any]]:
    """Compute monotonicity intervals using the mathematical domain.

    *x_min* / *x_max* are only used as a fallback or to clip critical
    points that fall outside the visible range.  Interval endpoints use
    the true mathematical domain bounds (e.g. ``-inf`` / ``+inf``) so
    that analytical predictions are scored fairly at L2.

    Returns list of {interval: [a, b], direction: "increasing"|"decreasing"}.
    """
    _NEG_INF = float("-inf")
    _POS_INF = float("inf")

    if func_type == "linear":
        a = params["a"]
        return [{"interval": [_NEG_INF, _POS_INF],
                 "direction": "increasing" if a > 0 else "decreasing"}]

    if func_type == "quadratic":
        a, h = params["a"], params["h"]
        hf = _round_if_close(float(h))
        segs = []
        if a > 0:
            segs.append({"interval": [_NEG_INF, hf], "direction": "decreasing"})
            segs.append({"interval": [hf, _POS_INF], "direction": "increasing"})
        else:
            segs.append({"interval": [_NEG_INF, hf], "direction": "increasing"})
            segs.append({"interval": [hf, _POS_INF], "direction": "decreasing"})
        return segs

    if func_type == "absolute":
        a, h = params["a"], params["h"]
        hf = _round_if_close(float(h))
        segs = []
        if a > 0:
            segs.append({"interval": [_NEG_INF, hf], "direction": "decreasing"})
            segs.append({"interval": [hf, _POS_INF], "direction": "increasing"})
        else:
            segs.append({"interval": [_NEG_INF, hf], "direction": "increasing"})
            segs.append({"interval": [hf, _POS_INF], "direction": "decreasing"})
        return segs

    if func_type == "cubic":
        a, b = params["a"], params["b"]
        h = params.get("h", 0)
        if b == 0 or (a != 0 and -b / (3 * a) <= 0):
            return [{"interval": [_NEG_INF, _POS_INF],
                     "direction": "increasing" if a > 0 else "decreasing"}]
        ratio = -b / (3 * a)
        if ratio > 0:
            uc = math.sqrt(ratio)
            crit = sorted([_round_if_close(h - uc), _round_if_close(h + uc)])
            segs = []
            pts = [_NEG_INF] + crit + [_POS_INF]
            for i in range(len(pts) - 1):
                if math.isfinite(pts[i]) and math.isfinite(pts[i + 1]):
                    mid = (pts[i] + pts[i + 1]) / 2
                elif math.isfinite(pts[i]):
                    mid = pts[i] + 1
                else:
                    mid = pts[i + 1] - 1
                u_mid = mid - h
                deriv = 3 * a * u_mid ** 2 + b
                segs.append({"interval": [pts[i], pts[i + 1]],
                             "direction": "increasing" if deriv > 0 else "decreasing"})
            return segs
        return [{"interval": [_NEG_INF, _POS_INF],
                 "direction": "increasing" if a > 0 else "decreasing"}]

    if func_type == "piecewise_linear":
        s1, s2, h = params["s1"], params["s2"], params["h"]
        hf = _round_if_close(float(h))
        return [
            {"interval": [_NEG_INF, hf],
             "direction": "increasing" if s1 > 0 else "decreasing"},
            {"interval": [hf, _POS_INF],
             "direction": "increasing" if s2 > 0 else "decreasing"},
        ]

    if func_type == "exponential":
        a, b = params["a"], params["b"]
        d = "increasing" if (a > 0 and b > 1) or (a < 0 and 0 < b < 1) else "decreasing"
        return [{"interval": [_NEG_INF, _POS_INF], "direction": d}]

    if func_type == "logarithmic":
        a = params["a"]
        h = params.get("h", None)
        if h is None:
            d_val = params.get("d", 0)
            h = -d_val
        domain_lo = _round_if_close(h)
        return [{"interval": [domain_lo, _POS_INF],
                 "direction": "increasing" if a > 0 else "decreasing"}]

    if func_type == "rational":
        a = params["a"]
        h = params.get("h", 0)
        hf = _round_if_close(float(h))
        d = "decreasing" if a > 0 else "increasing"
        return [
            {"interval": [_NEG_INF, hf], "direction": d},
            {"interval": [hf, _POS_INF], "direction": d},
        ]

    if func_type == "semicircle":
        lo, hi = float(x_min), float(x_max)
        return [{"interval": [lo, hi], "direction": "decreasing"}]

    return []


def _symmetry(func_type: str, params: Dict[str, Any]) -> str:
    """Determine if the function is even, odd, or neither."""
    if func_type == "quadratic":
        h = params["h"]
        return "even" if h == 0 else "neither"

    if func_type == "absolute":
        h = params["h"]
        return "even" if h == 0 else "neither"

    if func_type == "cubic":
        h = params.get("h", 0)
        k = params.get("k", 0)
        return "odd" if h == 0 and k == 0 else "neither"

    if func_type == "power":
        n = params["n"]
        if n in (2,):
            return "even"
        if n in (3, 1.0 / 3):
            return "odd"
        return "neither"

    if func_type == "rational":
        h = params.get("h", 0)
        k = params.get("k", 0)
        if h == 0 and k == 0:
            return "odd"
        return "neither"

    if func_type == "semicircle":
        h = params.get("h", 0)
        return "even" if h == 0 else "neither"

    if func_type in ("linear", "piecewise_linear"):
        return "neither"

    if func_type in ("exponential", "logarithmic"):
        return "neither"

    return "neither"


def compute_extended_properties(
    func_type: str,
    params: Dict[str, Any],
    sampler,
    x_min: float,
    x_max: float,
) -> Dict[str, Any]:
    """
    Compute the four extended properties for three-layer evaluation.

    Returns:
        {
            "intercepts":    {x_intercepts: list, y_intercept: float|None},
            "extrema":       [{x, y, type}, ...],
            "monotonicity":  [{interval: [a,b], direction}, ...],
            "symmetry":      "even" | "odd" | "neither",
        }
    """
    return {
        "intercepts": _intercepts(func_type, params, sampler, x_min, x_max),
        "extrema": _extrema(func_type, params, x_min, x_max),
        "monotonicity": _monotonicity(func_type, params, x_min, x_max),
        "symmetry": _symmetry(func_type, params),
    }
