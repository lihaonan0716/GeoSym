"""
Generate comprehensive math reasoning questions for benchmark images.

Three-layer (Curcio) evaluation structure — 10 scored dimensions:
  L1 Reading  — intercepts, extrema
  L2 Analysis — predictions, rate_of_change, monotonicity, symmetry
  L3 Reasoning — expression, domain, range, judge_inference_score
"""

from typing import Any, Dict, List

_STEM_TEXT = (
    "Examine the function graph presented below and complete the following "
    "analytical tasks."
)

_QUESTION_TEXT = (
    "Based on the function graph shown above, address the following tasks:\n"
    "(a) Determine the analytical expression of the function.\n"
    "(b) Determine the domain of the function.\n"
    "(c) Determine the range of the function.\n"
    "(d) Report all key intersections with the grid as ordered pairs (x, y)."
)

SUB_QUESTION_KEYS = [
    "expression", "domain", "range",
    "predictions", "rate_of_change",
    "intercepts", "extrema", "monotonicity", "symmetry",
]

_STEM_TASK_GROUPS = [
    "1. Identify key grid intersections and intercepts.",
    "2. Characterize extrema, monotonicity, and symmetry where applicable.",
    "3. Infer the expression, domain, and range of the function.",
    "4. Estimate predictive values and rates of change when relevant.",
]


def _build_stem_lines(lang: str = "en") -> List[str]:
    # The benchmark uses English-only prompts for consistency across samples.
    _ = lang
    return [_STEM_TEXT, *_STEM_TASK_GROUPS]


def generate_question(
    func_meta: Dict[str, Any],
    intersections: List[Dict[str, Any]] = None,
    lang: str = "en",
) -> Dict[str, Any]:
    """
    Generate a comprehensive question for one function image.

    Args:
        func_meta: Must contain 'expression_latex', 'expression_sympy',
                   and 'math_properties' (from compute_properties).
        intersections: List of intersection dicts from GT (coordinate_x, coordinate_y).
        lang: Reserved for compatibility; prompts are rendered in English only.

    Returns:
        {
            "question_type": "comprehensive",
            "question_text": str,
            "question_text_zh": str,
            "stem_lines": [str],
            "ground_truth": {
                "expression": str,
                "domain": str,
                "range": str,
                "zeros": list[float],       # retained for reference, not evaluated
                "coordinates": list[[int, int]],
            },
        }
    """
    props = func_meta.get("math_properties", {})
    expr_latex = func_meta.get("expression_latex", "")

    zeros = props.get("zeros", [])

    coords = []
    if intersections:
        coords = [[p["coordinate_x"], p["coordinate_y"]] for p in intersections]

    ground_truth = {
        "expression": expr_latex,
        "domain": props.get("domain", ""),
        "range": props.get("range", ""),
        "zeros": zeros,
        "coordinates": coords,
    }

    return {
        "question_type": "comprehensive",
        "question_text": _QUESTION_TEXT,
        "question_text_zh": _QUESTION_TEXT,
        "stem_lines": _build_stem_lines(lang),
        "ground_truth": ground_truth,
    }


def get_stem_lines(lang: str = "en") -> List[str]:
    """Get stem lines for rendering on image."""
    return _build_stem_lines(lang)
