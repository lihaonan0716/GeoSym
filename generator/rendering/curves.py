"""
Random elementary function generation and curve drawing.
Each curve is a single basic elementary function type (no mixing):
  - linear, quadratic, cubic, exponential, logarithmic,
    rational (1/x), absolute value, sqrt, piecewise_linear

Functions are randomly parameterized within configurable ranges.
Expressions are stored in three formats for ground truth:
  - expression_latex:  human-readable LaTeX for papers
  - expression_sympy:  sympy-parseable canonical form for symbolic equivalence comparison
  - type + parameters: structured dict for exact reconstruction and parameter-level comparison

Comparison with LLM output:
  import sympy
  x = sympy.Symbol('x')
  gt_expr = sympy.sympify(gt["expression_sympy"])
  llm_expr = sympy.sympify(parsed_llm_expression)
  is_equivalent = sympy.simplify(gt_expr - llm_expr) == 0
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np


# ============================================================
# Function type registry — extensible via @register decorator
# ============================================================

@dataclass(frozen=True)
class FunctionTypeSpec:
    """Metadata for a registered function type."""
    name: str
    category: str         # "elementary" | "polynomial" | "transcendental" | "special"
    generator: Callable   # (x_min, x_max, y_min, y_max) -> dict

_REGISTRY: Dict[str, FunctionTypeSpec] = {}


def register(name: str, *, category: str = "polynomial"):
    """Decorator to register a function generator into the global registry."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = FunctionTypeSpec(
            name=name, category=category, generator=fn,
        )
        return fn
    return decorator


# ============================================================
# Helper: format a numeric coefficient for sympy string
# ============================================================

def _sympy_coeff(val: float, var: str = "", leading: bool = True) -> str:
    """
    Format a coefficient * variable for sympy expression.
    leading=True means this is the first term (no + prefix).
    """
    if var == "":
        # constant term
        if val == 0:
            return ""
        if leading:
            return f"{val:g}"
        return f"+ {val:g}" if val > 0 else f"- {abs(val):g}"

    if val == 0:
        return ""
    if val == 1:
        term = var
    elif val == -1:
        term = f"-{var}" if leading else var
        if not leading:
            return f"- {var}"
        return f"-{var}"
    else:
        term = f"{val:g}*{var}"

    if leading:
        return term
    return f"+ {term}" if val > 0 else f"- {abs(val):g}*{var}"


# ============================================================
# Function generators — each returns dict with sampler + all expression formats
# ============================================================

@register("linear", category="elementary")
def _make_linear(x_min: float, x_max: float, y_min: float, y_max: float) -> Dict[str, Any]:
    """y = ax + b — choose among several line families with grid-aligned roots."""
    slope_pool = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
    x_candidates = list(range(int(math.ceil(x_min)), int(math.floor(x_max)) + 1))
    y_candidates = list(range(int(math.ceil(y_min)), int(math.floor(y_max)) + 1))

    best_a, best_b = 1, 0
    best_score = -1e9
    candidate_pool: List[Tuple[float, int, int]] = []
    family = random.choice([
        "root_left",
        "root_center",
        "root_right",
        "intercept_pos",
        "intercept_neg",
        "steep_cross",
    ])

    def _visible_count(a_: int, b_: int) -> int:
        count = 0
        for xv in x_candidates:
            yv = a_ * xv + b_
            if y_min <= yv <= y_max:
                count += 1
        return count

    for _ in range(80):
        if family == "steep_cross":
            a = random.choice([-6, -5, -4, 4, 5, 6])
            root_pool = [x for x in x_candidates if -2 <= x <= 2]
            root_x = random.choice(root_pool) if root_pool else 0
            b = -a * root_x
        elif family == "root_left":
            a = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            root_pool = [x for x in x_candidates if x <= -2]
            root_x = random.choice(root_pool) if root_pool else random.choice(x_candidates)
            b = -a * root_x
        elif family == "root_right":
            a = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            root_pool = [x for x in x_candidates if x >= 2]
            root_x = random.choice(root_pool) if root_pool else random.choice(x_candidates)
            b = -a * root_x
        elif family == "intercept_pos":
            a = random.choice([-5, -4, -3, -2, 2, 3, 4, 5])
            b_pool = [y for y in y_candidates if y >= 2]
            b = random.choice(b_pool) if b_pool else max(y_candidates) if y_candidates else 0
        elif family == "intercept_neg":
            a = random.choice([-5, -4, -3, -2, 2, 3, 4, 5])
            b_pool = [y for y in y_candidates if y <= -2]
            b = random.choice(b_pool) if b_pool else min(y_candidates) if y_candidates else 0
        else:
            a = random.choice([-3, -2, -1, 1, 2, 3])
            root_pool = [x for x in x_candidates if -1 <= x <= 1]
            root_x = random.choice(root_pool) if root_pool else 0
            b = -a * root_x

        count = _visible_count(a, b)
        if count < 7:
            continue

        y_at_xmin = a * x_min + b
        y_at_xmax = a * x_max + b
        x_cov = count / max(1.0, x_max - x_min + 1)
        center_bias = abs((y_at_xmin + y_at_xmax) / 2.0) / max(1.0, (y_max - y_min) / 2.0)
        root_x = (-b / a) if a != 0 else 0.0
        score = 10.0 * x_cov - 0.6 * center_bias
        score += 0.35 * min(12, count)
        score += 0.18 * min(abs(b), 12)
        score += 0.28 * min(abs(a), 6)
        if family in {"root_left", "root_right"}:
            score += 0.8
        if family in {"intercept_pos", "intercept_neg"}:
            score += 0.6
        if abs(a) >= 2:
            score += 1.0
        if abs(a) >= 3:
            score += 0.8
        if b == 0:
            score -= 2.2
        if abs(root_x) < 0.5:
            score -= 1.0

        if score > best_score:
            best_score = score
            best_a, best_b = a, int(round(b))
        candidate_pool.append((score, a, int(round(b))))

    if candidate_pool:
        candidate_pool.sort(key=lambda item: item[0], reverse=True)
        top_score = candidate_pool[0][0]
        top_pool = [item for item in candidate_pool[:16] if item[0] >= top_score - 2.5]
        _, a, b = random.choice(top_pool)
    else:
        a, b = best_a, best_b

    def sampler(x: float) -> Optional[float]:
        return a * x + b

    # --- Sympy canonical form (RHS only): "a*x + b" ---
    sympy_terms = []
    if a != 0:
        sympy_terms.append(f"{a:g}*x" if a not in (1, -1) else ("x" if a == 1 else "-x"))
    if b != 0:
        if sympy_terms:
            sympy_terms.append(f"+ {b:g}" if b > 0 else f"- {abs(b):g}")
        else:
            sympy_terms.append(f"{b:g}")
    expr_sympy = " ".join(sympy_terms) if sympy_terms else "0"

    # --- LaTeX ---
    latex_parts = []
    if a == 1:
        latex_parts.append("x")
    elif a == -1:
        latex_parts.append("-x")
    elif a != 0:
        latex_parts.append(f"{a:g}x")
    if b > 0:
        latex_parts.append(f"+ {b:g}")
    elif b < 0:
        latex_parts.append(f"- {abs(b):g}")
    latex_rhs = " ".join(latex_parts) if latex_parts else "0"

    return {
        "type": "linear",
        "parameters": {"a": a, "b": b},
        "expression_latex": f"y = {latex_rhs}",
        "expression_sympy": expr_sympy,
        "sampler": sampler,
    }


@register("quadratic", category="elementary")
def _make_quadratic(x_min: float, x_max: float, y_min: float, y_max: float) -> Dict[str, Any]:
    """y = a(x - h)² + k  — vertex on grid; x-intercepts (if any) on grid.

    Uses a generate-and-score loop to guarantee both diversity and
    visual quality.  Quality = the fraction of the x-viewport that
    the visible portion of the parabola spans (>= 35 %).

    a ∈ {-2, -1, -½, ½, 1, 2} — fractional values give wider, more
    graceful parabolas and increase expression diversity.
    """
    from fractions import Fraction

    _A_POOL = [Fraction(-2), Fraction(-1), Fraction(-1, 2),
               Fraction(1, 2), Fraction(1), Fraction(2)]
    _MIN_X_COVERAGE = 0.35
    x_span = float(x_max - x_min)
    y_span = float(y_max - y_min)

    best_a, best_h, best_k = 1, 0, 0
    best_score = -999.0

    for _ in range(50):
        a_frac = random.choice(_A_POOL)
        a_f = float(a_frac)
        abs_a = abs(a_f)

        # --- h: tighter centering for steep parabolas ---
        margin = 3 if abs_a >= 2 else 2
        h_lo = int(x_min) + margin
        h_hi = int(x_max) - margin
        if h_lo > h_hi:
            h_lo, h_hi = int(x_min) + 1, int(x_max) - 1
        if h_lo > h_hi:
            continue
        h = random.choice(list(range(h_lo, h_hi + 1)))

        # --- k: bias toward the opposite boundary to maximise visible area ---
        if a_f > 0:
            k_lo = int(y_min) + 1
            k_hi = min(int(y_max) - 2, int(y_min) + int(y_span * 0.65))
        else:
            k_lo = max(int(y_min) + 2, int(y_max) - int(y_span * 0.65))
            k_hi = int(y_max) - 1
        if k_lo > k_hi:
            k_lo, k_hi = int(y_min) + 1, int(y_max) - 1

        # k values that give integer x-intercepts: k = -a·d²
        k_with_roots = []
        for d in range(0, 8):
            kv_raw = float(-a_frac * d * d)
            kv = int(round(kv_raw))
            if abs(kv - kv_raw) < 0.01 and k_lo <= kv <= k_hi:
                k_with_roots.append(kv)
        # k values with no real x-intercepts: sign(k) == sign(a)
        k_no_roots = [kv for kv in range(k_lo, k_hi + 1)
                      if (a_f > 0 and kv > 0) or (a_f < 0 and kv < 0)]
        k_pool = list(set(k_with_roots + k_no_roots))
        if not k_pool:
            k_pool = list(range(k_lo, k_hi + 1)) or [0]
        k = random.choice(k_pool)

        # --- score: visible x-coverage + centering bonus ---
        delta_y = float(y_max - k) if a_f > 0 else float(k - y_min)
        if delta_y <= 0:
            continue
        half_w = math.sqrt(delta_y / abs_a)
        vis_lo = max(float(x_min), h - half_w)
        vis_hi = min(float(x_max), h + half_w)
        x_cov = (vis_hi - vis_lo) / x_span if x_span else 0

        center_off = abs(h - (x_min + x_max) / 2) / (x_span / 2) if x_span else 0
        score = x_cov - 0.15 * center_off

        if score > best_score:
            best_score = score
            best_a, best_h, best_k = a_frac, h, k

        if x_cov >= _MIN_X_COVERAGE:
            break

    a, h, k = best_a, best_h, best_k
    a_f = float(a)

    def sampler(x: float) -> Optional[float]:
        return a_f * (x - h) ** 2 + k

    # --- SymPy expression ---
    core_s = "x**2" if h == 0 else (
        f"(x - {h})**2" if h > 0 else f"(x + {abs(h)})**2")

    if a == 1:
        main_s = core_s
    elif a == -1:
        main_s = f"-{core_s}"
    elif a == Fraction(1, 2):
        main_s = f"{core_s}/2"
    elif a == Fraction(-1, 2):
        main_s = f"-{core_s}/2"
    else:
        main_s = f"{a}*{core_s}"

    if k > 0:
        expr_sympy = f"{main_s} + {k}"
    elif k < 0:
        expr_sympy = f"{main_s} - {abs(k)}"
    else:
        expr_sympy = main_s

    # --- LaTeX expression ---
    core_l = "x^{2}" if h == 0 else (
        f"(x - {h})^{{2}}" if h > 0 else f"(x + {abs(h)})^{{2}}")

    if a == 1:
        main_l = core_l
    elif a == -1:
        main_l = f"-{core_l}"
    elif a == Fraction(1, 2):
        main_l = f"\\frac{{1}}{{2}}{core_l}"
    elif a == Fraction(-1, 2):
        main_l = f"-\\frac{{1}}{{2}}{core_l}"
    else:
        main_l = f"{a}{core_l}"

    if k > 0:
        expr_latex = f"{main_l} + {k}"
    elif k < 0:
        expr_latex = f"{main_l} - {abs(k)}"
    else:
        expr_latex = main_l

    return {
        "type": "quadratic",
        "parameters": {"a": a_f, "h": h, "k": k},
        "expression_latex": f"y = {expr_latex}",
        "expression_sympy": expr_sympy,
        "sampler": sampler,
    }


@register("cubic", category="polynomial")
def _make_cubic(x_min: float, x_max: float, y_min: float, y_max: float) -> Dict[str, Any]:
    """y = a(x-h)³ + b(x-h) + k  — shifted cubic with S-curve shape.

    Instead of accepting the first random parameter set, sample several
    candidates and prefer ones whose turning structure is easier to read:
      - good visible x coverage
      - less crowding around the origin / axis-label region
      - fewer cases where both extrema bunch up on one side of the y-axis

    This keeps diversity, but makes the generated cubics more OCR-friendly
    before any post-hoc guard logic needs to step in.
    """

    x_span = float(x_max - x_min) if x_max != x_min else 1.0
    y_span = float(y_max - y_min) if y_max != y_min else 1.0
    target_count = 5
    best_params: Tuple[int, int, int, int] | None = None
    best_score = -1e9
    candidate_pool: List[Tuple[float, Tuple[int, int, int, int]]] = []

    def _sampler(a_: int, b_: int, h_: int, k_: int):
        return lambda x: a_ * (x - h_) * (x - h_) * (x - h_) + b_ * (x - h_) + k_

    def _extrema(a_: int, b_: int, h_: int, k_: int) -> List[Tuple[float, float]]:
        if b_ == 0 or a_ * b_ >= 0:
            return []
        u = math.sqrt(abs(b_) / 3.0)
        pts = []
        for sign in (-1.0, 1.0):
            x = h_ + sign * u
            y = a_ * (x - h_) ** 3 + b_ * (x - h_) + k_
            pts.append((x, y))
        return pts

    def _score_candidate(a_: int, b_: int, h_: int, k_: int) -> float:
        sampler = _sampler(a_, b_, h_, k_)
        xs = np.linspace(x_min, x_max, 401)
        ys = np.array([sampler(float(x)) for x in xs], dtype=np.float64)
        visible = (ys >= y_min - 0.5) & (ys <= y_max + 0.5)
        if visible.sum() < 24:
            return -1e6

        visible_runs = 0
        in_run = False
        for flag in visible:
            if flag and not in_run:
                visible_runs += 1
                in_run = True
            elif not flag:
                in_run = False
        if visible_runs >= 2:
            return -1e6

        vis_x = xs[visible]
        x_cov = float((vis_x.max() - vis_x.min()) / x_span) if len(vis_x) > 1 else 0.0
        vis_center = float((vis_x.min() + vis_x.max()) / 2.0) if len(vis_x) > 1 else 0.0
        score = 4.0 * x_cov - 0.35 * abs(vis_center) / max(1.0, x_span / 2.0)

        extrema = _extrema(a_, b_, h_, k_)
        if extrema:
            ext_x = [p[0] for p in extrema]
            ext_y = [p[1] for p in extrema]
            spread = abs(ext_x[1] - ext_x[0])
            min_abs_x = min(abs(x) for x in ext_x)
            x_margin = max(0.55, 0.08 * x_span)
            y_margin = max(0.75, 0.10 * y_span)
            if any(x <= x_min + x_margin or x >= x_max - x_margin for x in ext_x):
                return -1e6
            if any(y <= y_min + y_margin or y >= y_max - y_margin for y in ext_y):
                return -1e6

            near_origin_extrema = sum(
                1 for x, y in extrema
                if abs(x) < 1.0 and (y_min + 0.5) < y < (y_max - 0.5)
            )
            top_headroom = min(y_max - y for y in ext_y)
            bottom_headroom = min(y - y_min for y in ext_y)

            score += min(spread, 2.0)
            score -= 0.8 * near_origin_extrema
            score += 0.35 * min(top_headroom, 1.5)
            score += 0.35 * min(bottom_headroom, 1.5)

            if min_abs_x < 0.8:
                score -= 1.6 * (0.8 - min_abs_x)
            if all(abs(x) < 2.2 for x in ext_x):
                score -= 0.8
            if all(x > -0.25 for x in ext_x) or all(x < 0.25 for x in ext_x):
                score -= 0.7

            # Especially brittle: both extrema packed on one side of the y-axis
            # while staying entirely on the same side of the x-axis.
            if all(x > -0.25 for x in ext_x) and max(ext_y) < 0.0:
                score -= 2.5
            if all(x < 0.25 for x in ext_x) and min(ext_y) > 0.0:
                score -= 2.5
        else:
            score += 0.3

        y0 = sampler(0.0)
        if y0 is not None and abs(y0) < 0.75:
            score -= 0.3 * (0.75 - abs(y0))

        int_hits = 0
        for xv in range(int(math.ceil(x_min)), int(math.floor(x_max)) + 1):
            yv = sampler(float(xv))
            if yv is not None and y_min <= yv <= y_max and abs(yv - round(yv)) < 1e-9:
                int_hits += 1
        score -= 2.0 * abs(int_hits - target_count)
        score += 0.12 * (abs(b_) + abs(h_) + abs(k_))

        return score

    a_pool = [-2, -1, 1, 2]
    b_pool = [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8]
    hk_pool = list(range(-4, 5))

    for _ in range(80):
        cand = (
            random.choice(a_pool),
            random.choice(b_pool),
            random.choice(hk_pool),
            random.choice(hk_pool),
        )
        score = _score_candidate(*cand)
        if score > best_score:
            best_score = score
            best_params = cand
        candidate_pool.append((score, cand))

    if candidate_pool:
        candidate_pool.sort(key=lambda item: item[0], reverse=True)
        top_score = candidate_pool[0][0]
        top_pool = [item for item in candidate_pool[:20] if item[0] >= top_score - 2.0]
        _, best_params = random.choice(top_pool)
    elif best_params is None:
        best_params = (
            random.choice(a_pool),
            random.choice(b_pool),
            random.choice(hk_pool),
            random.choice(hk_pool),
        )

    a, b, h, k = best_params

    def sampler(x: float) -> Optional[float]:
        u = x - h
        return a * u * u * u + b * u + k

    # --- helper for (x-h) term in sympy/latex ---
    def _u_sympy() -> str:
        if h == 0:
            return "x"
        return f"(x - {h})" if h > 0 else f"(x + {abs(h)})"

    def _u_latex() -> str:
        if h == 0:
            return "x"
        return f"(x - {h})" if h > 0 else f"(x + {abs(h)})"

    u_s, u_l = _u_sympy(), _u_latex()

    # --- Sympy ---
    if a == 1:
        main_s = f"{u_s}**3"
    elif a == -1:
        main_s = f"-{u_s}**3"
    else:
        main_s = f"{a:g}*{u_s}**3"

    s_parts = [main_s]
    if b != 0:
        if b == 1:
            s_parts.append(f"+ {u_s}")
        elif b == -1:
            s_parts.append(f"- {u_s}")
        elif b > 0:
            s_parts.append(f"+ {b:g}*{u_s}")
        else:
            s_parts.append(f"- {abs(b):g}*{u_s}")
    if k > 0:
        s_parts.append(f"+ {k:g}")
    elif k < 0:
        s_parts.append(f"- {abs(k):g}")
    expr_sympy = " ".join(s_parts)

    # --- LaTeX ---
    if a == 1:
        main_l = f"{u_l}^{{3}}"
    elif a == -1:
        main_l = f"-{u_l}^{{3}}"
    else:
        main_l = f"{a:g}{u_l}^{{3}}"

    l_parts = [main_l]
    if b == 1:
        l_parts.append(f"+ {u_l}")
    elif b == -1:
        l_parts.append(f"- {u_l}")
    elif b > 0:
        l_parts.append(f"+ {b:g}{u_l}")
    elif b < 0:
        l_parts.append(f"- {abs(b):g}{u_l}")
    if k > 0:
        l_parts.append(f"+ {k:g}")
    elif k < 0:
        l_parts.append(f"- {abs(k):g}")

    return {
        "type": "cubic",
        "parameters": {"a": a, "b": b, "h": h, "k": k},
        "expression_latex": f"y = {' '.join(l_parts)}",
        "expression_sympy": expr_sympy,
        "sampler": sampler,
    }



@register("exponential", category="transcendental")
def _make_exponential(x_min: float, x_max: float, y_min: float, y_max: float) -> Dict[str, Any]:
    """y = a · b^(x-h) + k — sampled from several growth/decay families."""
    best_params: Tuple[int, int, int, int] | None = None
    best_score = -1e9
    candidate_pool: List[Tuple[float, Tuple[int, int, int, int]]] = []

    a_pool = [-3, -2, -1, 1, 2, 3]
    b_pool = [2, 3, 4]
    h_pool = list(range(-5, 5))
    k_pool = list(range(-4, 5))

    def _sampler(a_: int, b_: int, h_: int, k_: int):
        def _fn(x: float) -> Optional[float]:
            try:
                val = a_ * (b_ ** (x - h_)) + k_
                if abs(val) > 1e6:
                    return None
                return val
            except (OverflowError, ValueError):
                return None
        return _fn

    def _score_candidate(a_: int, b_: int, h_: int, k_: int) -> float:
        sampler = _sampler(a_, b_, h_, k_)
        int_hits = 0
        xs = []
        for xv in range(int(math.ceil(x_min)), int(math.floor(x_max)) + 1):
            yv = sampler(float(xv))
            if yv is not None and y_min <= yv <= y_max and abs(yv - round(yv)) < 1e-9:
                int_hits += 1
                xs.append(xv)
        if int_hits < 3:
            return -1e6
        x_cov = (max(xs) - min(xs)) / max(1.0, x_max - x_min) if len(xs) > 1 else 0.0
        score = 9.0 * x_cov - 2.5 * abs(int_hits - 4)
        score += 0.18 * (abs(a_) + abs(k_))
        score += 0.22 * (b_ - 2)
        score -= 0.10 * abs(h_)
        return score

    for _ in range(80):
        cand = (
            random.choice(a_pool),
            random.choice(b_pool),
            random.choice(h_pool),
            random.choice(k_pool),
        )
        score = _score_candidate(*cand)
        if score > best_score:
            best_score = score
            best_params = cand
        candidate_pool.append((score, cand))

    if candidate_pool:
        candidate_pool.sort(key=lambda item: item[0], reverse=True)
        top_score = candidate_pool[0][0]
        top_pool = [item for item in candidate_pool[:20] if item[0] >= top_score - 1.5]
        _, best_params = random.choice(top_pool)
    elif best_params is None:
        best_params = (1, 2, 0, 0)

    a, b, h, k = best_params

    def sampler(x: float) -> Optional[float]:
        try:
            val = a * (b ** (x - h)) + k
            if abs(val) > 1e6:
                return None
            return val
        except (OverflowError, ValueError):
            return None

    # --- Sympy ---
    if h == 0:
        exp_s = f"{b:g}**x"
    elif h > 0:
        exp_s = f"{b:g}**(x - {h})"
    else:
        exp_s = f"{b:g}**(x + {abs(h)})"

    if a == 1:
        main_sympy = exp_s
    elif a == -1:
        main_sympy = f"-{exp_s}"
    else:
        main_sympy = f"{a:g}*{exp_s}"

    s_parts = [main_sympy]
    if k > 0:
        s_parts.append(f"+ {k:g}")
    elif k < 0:
        s_parts.append(f"- {abs(k):g}")
    expr_sympy = " ".join(s_parts)

    # --- LaTeX ---
    if h == 0:
        exp_l = "x"
    elif h > 0:
        exp_l = f"x - {h}"
    else:
        exp_l = f"x + {abs(h)}"

    if a == 1:
        a_latex = ""
    elif a == -1:
        a_latex = "-"
    else:
        a_latex = f"{a:g} \\cdot "

    l_parts = [f"{a_latex}{b:g}^{{{exp_l}}}"]
    if k > 0:
        l_parts.append(f"+ {k:g}")
    elif k < 0:
        l_parts.append(f"- {abs(k):g}")

    return {
        "type": "exponential",
        "parameters": {"a": a, "b": round(b, 10), "h": h, "k": k},
        "expression_latex": f"y = {' '.join(l_parts)}",
        "expression_sympy": expr_sympy,
        "sampler": sampler,
    }


@register("logarithmic", category="transcendental")
def _make_logarithmic(x_min: float, x_max: float, y_min: float, y_max: float) -> Dict[str, Any]:
    """y = a · log₂(x - h) + k  where x > h.

    Using base-2 logarithm so that f(x) is an exact integer whenever
    x - h is a power of 2 (1, 2, 4, 8 …), giving many strict grid
    intersections.  h is the vertical asymptote position (negative
    values shift domain leftward, exposing more of the curve).
    """
    best_params: Tuple[int, int, int] | None = None
    best_score = -1e9
    candidate_pool: List[Tuple[float, Tuple[int, int, int]]] = []

    a_pool = [-4, -3, -2, -1, 1, 2, 3, 4]
    h_pool = [-8, -7, -6, -5, -4, -3, -2, -1]
    k_pool = list(range(-4, 5))

    def _sampler(a_: int, h_: int, k_: int):
        def _fn(x: float) -> Optional[float]:
            arg = x - h_
            if arg <= 0:
                return None
            return a_ * math.log2(arg) + k_
        return _fn

    def _score_candidate(a_: int, h_: int, k_: int) -> float:
        sampler = _sampler(a_, h_, k_)
        int_hits = 0
        xs = []
        for xv in range(int(math.ceil(x_min)), int(math.floor(x_max)) + 1):
            yv = sampler(float(xv))
            if yv is not None and y_min <= yv <= y_max and abs(yv - round(yv)) < 1e-9:
                int_hits += 1
                xs.append(xv)
        if int_hits < 4:
            return -1e6
        x_cov = (max(xs) - min(xs)) / max(1.0, x_max - x_min) if len(xs) > 1 else 0.0
        score = 10.0 * x_cov - 2.0 * abs(int_hits - 4)
        score += 0.16 * abs(a_) + 0.08 * abs(k_)
        score += 0.06 * abs(h_)
        return score

    for _ in range(90):
        cand = (
            random.choice(a_pool),
            random.choice(h_pool),
            random.choice(k_pool),
        )
        score = _score_candidate(*cand)
        if score > best_score:
            best_score = score
            best_params = cand
        candidate_pool.append((score, cand))

    if candidate_pool:
        candidate_pool.sort(key=lambda item: item[0], reverse=True)
        top_score = candidate_pool[0][0]
        top_pool = [item for item in candidate_pool[:24] if item[0] >= top_score - 1.25]
        _, best_params = random.choice(top_pool)
    elif best_params is None:
        best_params = (1, -2, 0)

    a, h, k = best_params

    def sampler(x: float) -> Optional[float]:
        arg = x - h
        if arg <= 0:
            return None
        return a * math.log2(arg) + k

    # --- Sympy: log(expr, 2) is base-2 log in sympy ---
    if h == 0:
        arg_sympy = "x"
    elif h > 0:
        arg_sympy = f"x - {h}"
    else:
        arg_sympy = f"x + {abs(h)}"

    if a == 1:
        main_sympy = f"log({arg_sympy}, 2)"
    elif a == -1:
        main_sympy = f"-log({arg_sympy}, 2)"
    else:
        main_sympy = f"{a:g}*log({arg_sympy}, 2)"

    s_parts = [main_sympy]
    if k > 0:
        s_parts.append(f"+ {k:g}")
    elif k < 0:
        s_parts.append(f"- {abs(k):g}")
    expr_sympy = " ".join(s_parts)

    # --- LaTeX ---
    if h == 0:
        arg_latex = "x"
    elif h > 0:
        arg_latex = f"x - {h}"
    else:
        arg_latex = f"x + {abs(h)}"

    if a == 1:
        a_latex = ""
    elif a == -1:
        a_latex = "-"
    else:
        a_latex = f"{a:g}"

    l_parts = [f"{a_latex}\\log_{{2}}({arg_latex})"]
    if k > 0:
        l_parts.append(f"+ {k:g}")
    elif k < 0:
        l_parts.append(f"- {abs(k):g}")

    return {
        "type": "logarithmic",
        "parameters": {"a": a, "h": h, "k": k, "base": 2},
        "expression_latex": f"y = {' '.join(l_parts)}",
        "expression_sympy": expr_sympy,
        "sampler": sampler,
    }


@register("rational", category="special")
def _make_rational(x_min: float, x_max: float, y_min: float, y_max: float) -> Dict[str, Any]:
    """y = a / (x - h) + k  (hyperbola, asymptote at x=h, y=k)."""
    best_params: Tuple[int, int, int] | None = None
    best_score = -1e9
    candidate_pool: List[Tuple[float, Tuple[int, int, int]]] = []

    a_pool = [-10, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 10]
    h_pool = list(range(-4, 5))
    k_pool = list(range(-3, 4))

    def _sampler(a_: int, h_: int, k_: int):
        def _fn(x: float) -> Optional[float]:
            denom = x - h_
            if abs(denom) < 1e-9:
                return None
            val = a_ / denom + k_
            if abs(val) > 50:
                return None
            return val
        return _fn

    def _score_candidate(a_: int, h_: int, k_: int) -> float:
        sampler = _sampler(a_, h_, k_)
        int_hits = 0
        branches = set()
        for xv in range(int(math.ceil(x_min)), int(math.floor(x_max)) + 1):
            yv = sampler(float(xv))
            if yv is not None and y_min <= yv <= y_max and abs(yv - round(yv)) < 1e-9:
                int_hits += 1
                branches.add("L" if xv < h_ else "R")
        if int_hits < 5:
            return -1e6
        score = -2.5 * abs(int_hits - 6)
        score += 2.0 * len(branches)
        score += 0.08 * abs(a_) + 0.08 * abs(h_) + 0.05 * abs(k_)
        return score

    for _ in range(80):
        cand = (
            random.choice(a_pool),
            random.choice(h_pool),
            random.choice(k_pool),
        )
        score = _score_candidate(*cand)
        if score > best_score:
            best_score = score
            best_params = cand
        candidate_pool.append((score, cand))

    if candidate_pool:
        candidate_pool.sort(key=lambda item: item[0], reverse=True)
        top_score = candidate_pool[0][0]
        top_pool = [item for item in candidate_pool[:24] if item[0] >= top_score - 1.5]
        _, best_params = random.choice(top_pool)
    elif best_params is None:
        best_params = (2, 0, 0)

    a, h, k = best_params

    def sampler(x: float) -> Optional[float]:
        denom = x - h
        if abs(denom) < 1e-9:
            return None
        val = a / denom + k
        if abs(val) > 50:
            return None
        return val

    # --- Sympy: "a/(x - h) + k" or "a/x + k" ---
    denom_sympy = "x" if h == 0 else (f"(x - {h:g})" if h > 0 else f"(x + {abs(h):g})")
    main_sympy = f"{a:g}/{denom_sympy}"

    s_parts = [main_sympy]
    if k > 0:
        s_parts.append(f"+ {k:g}")
    elif k < 0:
        s_parts.append(f"- {abs(k):g}")
    expr_sympy = " ".join(s_parts)

    # --- LaTeX ---
    if a == 1:
        num_latex = "1"
    elif a == -1:
        num_latex = "-1"
    else:
        num_latex = f"{a:g}"
    denom_latex = "x" if h == 0 else (f"x - {h:g}" if h > 0 else f"x + {abs(h):g}")

    l_parts = [f"\\frac{{{num_latex}}}{{{denom_latex}}}"]
    if k > 0:
        l_parts.append(f"+ {k:g}")
    elif k < 0:
        l_parts.append(f"- {abs(k):g}")

    return {
        "type": "rational",
        "parameters": {"a": a, "h": h, "k": k},
        "expression_latex": f"y = {' '.join(l_parts)}",
        "expression_sympy": expr_sympy,
        "sampler": sampler,
    }


@register("absolute", category="elementary")
def _make_absolute(x_min: float, x_max: float, y_min: float, y_max: float,
                   step_x: float = 1.0, step_y: float = 1.0) -> Dict[str, Any]:
    """y = a|x - h| + k  — vertex on grid; x-intercepts (if any) on grid.

    Reverse-engineering: pick integer d ≥ 0, set k = -a·d so that
    x-intercepts land at h±d.  Also allow "no x-intercepts" by picking
    k with same sign as a (vertex above/below x-axis).
    """
    a = random.choice([-2, -1, 1, 2])

    h_candidates = [round(x_min + i * step_x, 6) for i in range(int((x_max - x_min) / step_x) + 1)]
    h_candidates = [v for v in h_candidates if x_min + step_x <= v <= x_max - step_x]
    h = random.choice(h_candidates) if h_candidates else 0

    k_lo = int(y_min + step_y)
    k_hi = int(y_max - step_y)
    # k values with integer x-intercepts: k = -a*d for d=0,1,2,...
    k_with_roots = []
    for d in range(0, 8):
        kv = -a * d
        if k_lo <= kv <= k_hi:
            k_with_roots.append(kv)
    # k values with no x-intercepts: sign(k) == sign(a), k ≠ 0
    k_no_roots = [kv for kv in range(k_lo, k_hi + 1)
                  if (a > 0 and kv > 0) or (a < 0 and kv < 0)]
    k_pool = list(set(k_with_roots + k_no_roots))
    if not k_pool:
        k_pool = [0]
    k = random.choice(k_pool)

    def sampler(x: float) -> Optional[float]:
        return a * abs(x - h) + k

    # --- Sympy: Abs(x - h) is sympy's absolute value ---
    arg_sympy = "x" if h == 0 else (f"x - {h:g}" if h > 0 else f"x + {abs(h):g}")
    if a == 1:
        main_sympy = f"Abs({arg_sympy})"
    elif a == -1:
        main_sympy = f"-Abs({arg_sympy})"
    else:
        main_sympy = f"{a:g}*Abs({arg_sympy})"

    s_parts = [main_sympy]
    if k > 0:
        s_parts.append(f"+ {k:g}")
    elif k < 0:
        s_parts.append(f"- {abs(k):g}")
    expr_sympy = " ".join(s_parts)

    # --- LaTeX ---
    arg_latex = "x" if h == 0 else (f"x - {h:g}" if h > 0 else f"x + {abs(h):g}")
    if a == 1:
        a_latex = ""
    elif a == -1:
        a_latex = "-"
    else:
        a_latex = f"{a:g}"

    l_parts = [f"{a_latex}|{arg_latex}|"]
    if k > 0:
        l_parts.append(f"+ {k:g}")
    elif k < 0:
        l_parts.append(f"- {abs(k):g}")

    return {
        "type": "absolute",
        "parameters": {"a": a, "h": h, "k": k},
        "expression_latex": f"y = {' '.join(l_parts)}",
        "expression_sympy": expr_sympy,
        "sampler": sampler,
    }


@register("piecewise_linear", category="elementary")
def _make_piecewise_linear(x_min: float, x_max: float, y_min: float, y_max: float,
                           step_x: float = 1.0, step_y: float = 1.0) -> Dict[str, Any]:
    """y = s1*(x-h)+k for x<h, s2*(x-h)+k for x≥h  — breakpoint on grid,
    x-intercepts on grid (k divisible by both s1 and s2).
    """
    _MIN_SLOPE_DIFF = 3
    slopes = [-4, -3, -2, -1, 1, 2, 3, 4]
    s1 = random.choice(slopes)
    s2 = random.choice([s for s in slopes if abs(s - s1) >= _MIN_SLOPE_DIFF and s != -s1])

    h_candidates = [round(x_min + i * step_x, 6)
                    for i in range(int((x_max - x_min) / step_x) + 1)]
    h_candidates = [v for v in h_candidates if x_min + 2 * step_x <= v <= x_max - 2 * step_x]
    h = random.choice(h_candidates) if h_candidates else 0

    # k must be divisible by both s1 and s2 so x-intercepts are integers
    lcm_s = abs(s1 * s2) // math.gcd(abs(s1), abs(s2))
    k_lo = int(y_min + 2 * step_y)
    k_hi = int(y_max - 2 * step_y)
    k_pool = [kv for kv in range(k_lo, k_hi + 1) if kv % lcm_s == 0]
    if not k_pool:
        k_pool = [0]
    k = random.choice(k_pool)

    for _ in range(10):
        y_left = s1 * (x_min - h) + k
        y_right = s2 * (x_max - h) + k
        if y_left > y_max + 2 and k > y_max + 2 and y_right > y_max + 2:
            k -= lcm_s
        elif y_left < y_min - 2 and k < y_min - 2 and y_right < y_min - 2:
            k += lcm_s
        else:
            break

    h_int, k_int = int(round(h)), int(round(k))

    def sampler(x: float) -> Optional[float]:
        if x < h:
            return s1 * (x - h) + k
        return s2 * (x - h) + k

    b1 = round(-s1 * h + k)
    b2 = round(-s2 * h + k)

    def _lin_sympy(slope: int, intercept: int) -> str:
        parts: List[str] = []
        if slope == 1:
            parts.append("x")
        elif slope == -1:
            parts.append("-x")
        elif slope != 0:
            parts.append(f"{slope:g}*x")
        if intercept > 0:
            parts.append(f"+ {intercept:g}" if parts else f"{intercept:g}")
        elif intercept < 0:
            parts.append(f"- {abs(intercept):g}")
        elif not parts:
            parts.append("0")
        return " ".join(parts)

    def _lin_latex(slope: int, intercept: int) -> str:
        parts: List[str] = []
        if slope == 1:
            parts.append("x")
        elif slope == -1:
            parts.append("-x")
        elif slope != 0:
            parts.append(f"{slope:g}x")
        if intercept > 0:
            parts.append(f"+ {intercept:g}" if parts else f"{intercept:g}")
        elif intercept < 0:
            parts.append(f"- {abs(intercept):g}")
        elif not parts:
            parts.append("0")
        return " ".join(parts)

    expr_sympy = (f"Piecewise(({_lin_sympy(s1, int(b1))}, x < {h_int}), "
                  f"({_lin_sympy(s2, int(b2))}, True))")

    expr_latex = (f"y = \\begin{{cases}} {_lin_latex(s1, int(b1))} & x < {h_int} \\\\\\ "
                  f"{_lin_latex(s2, int(b2))} & x \\geq {h_int} \\end{{cases}}")

    return {
        "type": "piecewise_linear",
        "parameters": {"s1": s1, "s2": s2, "h": h_int, "k": k_int},
        "expression_latex": expr_latex,
        "expression_sympy": expr_sympy,
        "sampler": sampler,
    }


# ============================================================
# Composite function types
# ============================================================


# ============================================================
# Derived registry exports
# ============================================================

FUNCTION_GENERATORS: Dict[str, Callable] = {
    name: spec.generator for name, spec in _REGISTRY.items()
}

# ── Function-type categories (informational grouping) ──────────────────

FUNCTION_TYPE_CATEGORIES: Dict[str, List[str]] = {
    "elementary":     ["linear", "quadratic", "absolute"],
    "polynomial":     ["cubic", "piecewise_linear"],
    "transcendental": ["exponential", "logarithmic"],
    "special":        ["rational"],
}

ALL_FUNCTION_TYPES: List[str] = [
    t for types in FUNCTION_TYPE_CATEGORIES.values() for t in types
]

# Per-type upper bound on the number of *strict* grid intersections that
# are mathematically achievable.  For types not listed here the global
# user-configured min_intersections is used as-is.
TYPE_INTERSECTION_CAP: Dict[str, int] = {
    "linear":            99,
    "quadratic":          5,
    "absolute":          99,
    "cubic":              5,
    "piecewise_linear":  99,
    "exponential":        3,
    "logarithmic":        4,
    "rational":           6,
}

# Per-function-type recommended coordinate ranges for "static" smart mode.
# Each entry is a list of {x_range, y_range} dicts optimised for visual clarity.
FUNCTION_COORD_POOL: Dict[str, List[Dict[str, list]]] = {
    "linear": [
        {"x_range": [-7, 7],  "y_range": [-7, 7]},
        {"x_range": [-6, 8],  "y_range": [-7, 7]},
        {"x_range": [-8, 6],  "y_range": [-6, 8]},
    ],
    "quadratic": [
        {"x_range": [-7, 7],  "y_range": [-6, 8]},
        {"x_range": [-6, 8],  "y_range": [-6, 8]},
        {"x_range": [-8, 6],  "y_range": [-6, 8]},
    ],
    "absolute": [
        {"x_range": [-7, 7],  "y_range": [-6, 8]},
        {"x_range": [-6, 8],  "y_range": [-5, 9]},
        {"x_range": [-8, 6],  "y_range": [-6, 8]},
    ],
    "cubic": [
        {"x_range": [-5, 5],  "y_range": [-6, 6]},
        {"x_range": [-6, 6],  "y_range": [-6, 6]},
        {"x_range": [-4, 6],  "y_range": [-6, 6]},
    ],
    "exponential": [
        {"x_range": [-4, 5],  "y_range": [-3, 8]},
        {"x_range": [-5, 4],  "y_range": [-3, 8]},
        {"x_range": [-4, 4],  "y_range": [-3, 8]},
    ],
    "logarithmic": [
        {"x_range": [-2, 10], "y_range": [-6, 6]},
        {"x_range": [-4, 12], "y_range": [-5, 7]},
        {"x_range": [-6, 10], "y_range": [-6, 8]},
        {"x_range": [-8, 8],  "y_range": [-6, 8]},
    ],
    "rational": [
        {"x_range": [-7, 7],  "y_range": [-7, 7]},
        {"x_range": [-6, 8],  "y_range": [-7, 7]},
        {"x_range": [-8, 6],  "y_range": [-7, 7]},
        {"x_range": [-7, 7],  "y_range": [-6, 8]},
    ],
    "piecewise_linear": [
        {"x_range": [-7, 7],  "y_range": [-7, 7]},
        {"x_range": [-6, 8],  "y_range": [-6, 8]},
        {"x_range": [-8, 6],  "y_range": [-6, 8]},
    ],
}

FUNCTION_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "linear":           "y = ax + b",
    "quadratic":        "y = a(x-h)² + k",
    "absolute":         "y = a|x-h| + k",
    "cubic":            "y = a(x-h)³ + b(x-h) + k",
    "piecewise_linear": "y = s₁(x-h)+k / s₂(x-h)+k",
    "exponential":      "y = a·bˣ⁻ʰ + k",
    "logarithmic":      "y = a·log₂(x-h) + k",
    "rational":         "y = a/(x-h) + k",
}

FUNCTION_RICHNESS_SPECS: Dict[str, Dict[str, int]] = {
    "linear": {"hard_min": 8, "preferred_min": 10, "preferred_max": 14},
    "quadratic": {"hard_min": 4, "preferred_min": 4, "preferred_max": 5},
    "absolute": {"hard_min": 7, "preferred_min": 9, "preferred_max": 14},
    "cubic": {"hard_min": 5, "preferred_min": 5, "preferred_max": 5},
    "piecewise_linear": {"hard_min": 8, "preferred_min": 10, "preferred_max": 14},
    "exponential": {"hard_min": 3, "preferred_min": 4, "preferred_max": 5},
    "logarithmic": {"hard_min": 4, "preferred_min": 4, "preferred_max": 4},
    "rational": {"hard_min": 6, "preferred_min": 6, "preferred_max": 6},
}


# ============================================================
# Public API: generate random functions
# ============================================================

def generate_random_function(
    func_type: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Dict[str, Any]:
    """
    Generate a random function of the given type.

    Returns dict with keys:
        type, parameters, expression_latex, expression_sympy, sampler
    """
    gen = FUNCTION_GENERATORS.get(func_type)
    if gen is None:
        raise ValueError(f"Unknown function type: {func_type}. Available: {list(FUNCTION_GENERATORS.keys())}")
    return gen(x_min, x_max, y_min, y_max)


def sample_function(
    func_info: Dict[str, Any],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    step: float = 0.02,
) -> List[Tuple[float, float]]:
    """
    Sample points from a generated function.

    Returns list of (x, y) in math coordinates, only within visible range.
    """
    sampler = func_info["sampler"]
    pts = []
    x = x_min
    while x <= x_max + step * 0.5:
        y = sampler(x)
        if y is not None and y_min - 0.5 <= y <= y_max + 0.5:
            pts.append((x, float(y)))
        x += step
    return pts


def get_richness_spec(func_type: str) -> Dict[str, int]:
    """Return per-function richness targets."""
    return dict(FUNCTION_RICHNESS_SPECS.get(func_type, {"hard_min": 4, "preferred_min": 5, "preferred_max": 8}))


def _bucket_number(value: float, step: float = 1.0) -> str:
    bucket = int(round(float(value) / max(step, 1e-9)))
    return str(bucket)


def _sign_token(value: float) -> str:
    if value > 0:
        return "pos"
    if value < 0:
        return "neg"
    return "zero"


def _root_regime_token(func_type: str, intersections: List[dict]) -> str:
    x_axis_hits = sum(1 for p in intersections if p.get("coordinate_y") == 0)
    if func_type in {"quadratic", "absolute"}:
        if x_axis_hits == 0:
            return "no_x_root"
        if x_axis_hits == 1:
            return "single_x_root"
        return "double_x_root"
    if func_type in {"linear", "piecewise_linear", "cubic", "rational"}:
        if x_axis_hits == 0:
            return "no_x_root"
        if x_axis_hits == 1:
            return "single_x_root"
        return "multi_x_root"
    return f"xroot_{x_axis_hits}"


def _family_tokens(
    func_type: str,
    params: Dict[str, Any],
    intersections: List[dict],
) -> List[str]:
    count = len(intersections)
    x_axis_hits = sum(1 for p in intersections if p.get("coordinate_y") == 0)
    y_axis_hits = sum(1 for p in intersections if p.get("coordinate_x") == 0)

    if func_type == "linear":
        a = float(params.get("a", 0))
        b = float(params.get("b", 0))
        root_x = (-b / a) if abs(a) > 1e-9 else 0.0
        return [
            f"slope_{_sign_token(a)}_{_bucket_number(abs(a), 1.0)}",
            f"bias_{_bucket_number(b, 1.0)}",
            f"root_x_{_bucket_number(root_x, 1.0)}",
            f"xaxis_{x_axis_hits}",
            f"yaxis_{y_axis_hits}",
            f"count_{count}",
        ]

    if func_type == "quadratic":
        a = float(params.get("a", 0))
        h = float(params.get("h", 0))
        k = float(params.get("k", 0))
        return [
            f"open_{_sign_token(a)}",
            f"curve_{_bucket_number(abs(a), 0.5)}",
            f"vertex_x_{_bucket_number(h, 2.0)}",
            f"vertex_y_{_bucket_number(k, 2.0)}",
            _root_regime_token(func_type, intersections),
            f"count_{count}",
        ]

    if func_type == "absolute":
        a = float(params.get("a", 0))
        h = float(params.get("h", 0))
        k = float(params.get("k", 0))
        return [
            f"open_{_sign_token(a)}",
            f"slope_{_bucket_number(abs(a), 1.0)}",
            f"vertex_x_{_bucket_number(h, 2.0)}",
            f"vertex_y_{_bucket_number(k, 2.0)}",
            _root_regime_token(func_type, intersections),
            f"count_{count}",
        ]

    if func_type == "cubic":
        a = float(params.get("a", 0))
        b = float(params.get("b", 0))
        h = float(params.get("h", 0))
        k = float(params.get("k", 0))
        extremum_regime = "two_extrema" if a * b < 0 and b != 0 else "monotone_like"
        return [
            f"lead_{_sign_token(a)}",
            f"tilt_{_sign_token(b)}_{_bucket_number(abs(b), 1.0)}",
            f"shift_x_{_bucket_number(h, 2.0)}",
            f"shift_y_{_bucket_number(k, 2.0)}",
            extremum_regime,
            _root_regime_token(func_type, intersections),
            f"count_{count}",
        ]

    if func_type == "piecewise_linear":
        s1 = float(params.get("s1", 0))
        s2 = float(params.get("s2", 0))
        h = float(params.get("h", 0))
        k = float(params.get("k", 0))
        turn = "same_sign" if s1 * s2 > 0 else "cross_sign"
        return [
            f"left_{_sign_token(s1)}_{_bucket_number(abs(s1), 1.0)}",
            f"right_{_sign_token(s2)}_{_bucket_number(abs(s2), 1.0)}",
            turn,
            f"break_x_{_bucket_number(h, 2.0)}",
            f"break_y_{_bucket_number(k, 2.0)}",
            f"count_{count}",
        ]

    if func_type == "exponential":
        a = float(params.get("a", 0))
        b = float(params.get("b", 2))
        h = float(params.get("h", 0))
        k = float(params.get("k", 0))
        return [
            f"amp_{_sign_token(a)}_{_bucket_number(abs(a), 1.0)}",
            f"base_{_bucket_number(b, 1.0)}",
            f"shift_x_{_bucket_number(h, 1.0)}",
            f"shift_y_{_bucket_number(k, 1.0)}",
            f"count_{count}",
        ]

    if func_type == "logarithmic":
        a = float(params.get("a", 0))
        h = float(params.get("h", 0))
        k = float(params.get("k", 0))
        return [
            f"amp_{_sign_token(a)}_{_bucket_number(abs(a), 1.0)}",
            f"asym_x_{_bucket_number(h, 1.0)}",
            f"shift_y_{_bucket_number(k, 1.0)}",
            f"count_{count}",
        ]

    if func_type == "rational":
        a = float(params.get("a", 0))
        h = float(params.get("h", 0))
        k = float(params.get("k", 0))
        return [
            f"amp_{_sign_token(a)}_{_bucket_number(abs(a), 2.0)}",
            f"asym_x_{_bucket_number(h, 1.0)}",
            f"asym_y_{_bucket_number(k, 1.0)}",
            _root_regime_token(func_type, intersections),
            f"count_{count}",
        ]

    return [func_type, f"count_{count}"]


def summarize_function_candidate(
    func_info: Dict[str, Any],
    intersections: List[dict],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Dict[str, Any]:
    """Summarize a candidate for richness scoring and diversity control."""
    func_type = func_info.get("type", "unknown")
    params = dict(func_info.get("parameters", {}))
    sampler = func_info.get("sampler")
    vis_pts = sample_function(func_info, x_min, x_max, y_min, y_max, step=0.25) if sampler is not None else []
    if vis_pts:
        vis_xs = [p[0] for p in vis_pts]
        vis_ys = [p[1] for p in vis_pts]
        x_coverage = (max(vis_xs) - min(vis_xs)) / max(1e-9, float(x_max - x_min))
        y_coverage = (max(vis_ys) - min(vis_ys)) / max(1e-9, float(y_max - y_min))
    else:
        x_coverage = 0.0
        y_coverage = 0.0

    family_tokens = _family_tokens(func_type, params, intersections)
    exact_tokens = [func_type, f"xspan_{int(round(x_max - x_min))}", f"yspan_{int(round(y_max - y_min))}"]
    for key in sorted(params):
        val = params[key]
        if isinstance(val, (int, float)):
            exact_tokens.append(f"{key}_{float(val):.4f}")
        else:
            exact_tokens.append(f"{key}_{val}")

    return {
        "func_type": func_type,
        "intersection_count": len(intersections),
        "x_axis_count": sum(1 for p in intersections if p.get("coordinate_y") == 0),
        "y_axis_count": sum(1 for p in intersections if p.get("coordinate_x") == 0),
        "x_coverage": round(float(x_coverage), 5),
        "y_coverage": round(float(y_coverage), 5),
        "family_signature": "|".join(family_tokens),
        "exact_signature": "|".join(exact_tokens),
    }


def summarize_generated_sample(gt: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize an already generated GT sample."""
    funcs = gt.get("functions", [])
    if not funcs:
        return {
            "func_type": "unknown",
            "intersection_count": 0,
            "x_axis_count": 0,
            "y_axis_count": 0,
            "x_coverage": 0.0,
            "y_coverage": 0.0,
            "family_signature": "unknown",
            "exact_signature": "unknown",
        }
    x_range = gt.get("x_range", [-5, 5])
    y_range = gt.get("y_range", [-5, 5])
    return summarize_function_candidate(
        funcs[0],
        gt.get("intersections", []),
        float(x_range[0]),
        float(x_range[1]),
        float(y_range[0]),
        float(y_range[1]),
    )


# ============================================================
# Drawing (unchanged interface)
# ============================================================

def _to_pixel(
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    cx: int,
    cy: int,
    cw: int,
    ch: int,
    padding: int,
) -> Tuple[int, int]:
    ix = cx + padding
    iy = cy + padding
    iw = cw - 2 * padding
    ih = ch - 2 * padding
    if x_max == x_min or y_max == y_min:
        return (ix + iw // 2, iy + ih // 2)
    px = ix + (x - x_min) / (x_max - x_min) * iw
    py = iy + (y_max - y) / (y_max - y_min) * ih
    return (int(round(px)), int(round(py)))


def draw_curves(
    img: np.ndarray,
    chart_rect: Tuple[int, int, int, int],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    curves_points: List[List[Tuple[float, float]]],
    color_bgr: Tuple[int, int, int],
    line_width: int,
    padding: int,
) -> None:
    """Draw curves on img in a single color (black). curves_points[i] = [(x,y), ...] in math coords."""
    cx, cy, cw, ch = chart_rect
    x_min, x_max = x_range
    y_min, y_max = y_range

    def to_px(x: float, y: float) -> Tuple[int, int]:
        return _to_pixel(x, y, x_min, x_max, y_min, y_max, cx, cy, cw, ch, padding)

    for pts in curves_points:
        visible = [(x, y) for x, y in pts if y_min - 0.5 <= y <= y_max + 0.5]
        pts_px = [to_px(x, y) for x, y in visible]
        for j in range(len(pts_px) - 1):
            dx = abs(pts_px[j + 1][0] - pts_px[j][0])
            dy = abs(pts_px[j + 1][1] - pts_px[j][1])
            max_jump = max(img.shape[0], img.shape[1]) * 0.3
            if dy > max_jump or dx > max_jump:
                continue
            cv2.line(img, pts_px[j], pts_px[j + 1], color_bgr, line_width, lineType=cv2.LINE_AA)
