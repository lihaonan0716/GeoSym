"""Image generation and dataset preview logic. Extracted from app.py."""

import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Tuple, List, Dict

import cv2
import numpy as np

from app.helpers import (
    _root, _config, _images_dir, _gt_dir,
    list_dataset_images, load_image_and_gt, clear_dataset,
    _resolve_coord_mode,
    get_next_index, clear_subset,
)
from generator.core import generate_one
from generator.benchmark import (
    BENCHMARK_PROFILES,
    build_subset_config,
    generate_and_save_diverse_image,
)

logger = logging.getLogger(__name__)
from generator.rendering.curves import (
    ALL_FUNCTION_TYPES,
    FUNCTION_TYPE_DESCRIPTIONS,
    TYPE_INTERSECTION_CAP,
)

FUNCTION_TYPE_LABELS: list[str] = [
    f"{ft} — {FUNCTION_TYPE_DESCRIPTIONS.get(ft, ft)}" for ft in ALL_FUNCTION_TYPES
]
DEFAULT_FUNC_TYPE_LABELS: list[str] = list(FUNCTION_TYPE_LABELS)
FUNCTION_TYPE_COUNT = len(ALL_FUNCTION_TYPES)
PROFILE_COUNT = len(BENCHMARK_PROFILES)
TOTAL_SUBSET_COUNT = FUNCTION_TYPE_COUNT * PROFILE_COUNT


class _NoOpProgress:
    def __call__(self, *args, **kwargs):
        return None

    def tqdm(self, iterable, **kwargs):
        return iterable

_LABEL_TO_FTYPE: dict[str, str] = {
    label: ft for ft, label in zip(ALL_FUNCTION_TYPES, FUNCTION_TYPE_LABELS)
}

PROFILE_LABELS: dict[str, str] = {
    "clean_render": "Clean Render - no synthetic degradation",
}
DEFAULT_PROFILE_ID = BENCHMARK_PROFILES[0]
DEFAULT_SAMPLE_PLAN_ID = "uniform"
RECOMMENDED_SUBSET_SAMPLES: dict[str, int] = {
    "linear": 12,
    "quadratic": 30,
    "absolute": 50,
    "cubic": 50,
    "piecewise_linear": 30,
    "exponential": 50,
    "logarithmic": 50,
    "rational": 35,
}


def parse_func_type_labels(labels: list[str]) -> list[str]:
    """Convert UI labels back to internal function type names."""
    return [_LABEL_TO_FTYPE.get(lb, lb.split(" —")[0].strip()) for lb in labels]


def profile_choices(lang: str) -> list[tuple[str, str]]:
    """Return benchmark profile choices as (label, value) tuples."""
    return [(PROFILE_LABELS.get(pid, pid), pid) for pid in BENCHMARK_PROFILES]


def generation_mode_choices(lang: str) -> list[str]:
    """Return generation mode labels with live subset counts."""
    all_label = f"All {TOTAL_SUBSET_COUNT} subsets (batch)"
    return ['Selected types (append)', all_label]


def sample_plan_choices(lang: str) -> list[tuple[str, str]]:
    """Return localized sample-plan choices."""
    return [
        ('Uniform count for every function type', "uniform"),
        ('Recommended benchmark allocation', "recommended"),
    ]


def _resolve_sample_plan(count: int, plan_id: str) -> dict[str, int]:
    if plan_id == "recommended":
        return dict(RECOMMENDED_SUBSET_SAMPLES)
    return {ft: int(count) for ft in ALL_FUNCTION_TYPES}


def sample_plan_markdown(
    lang: str,
    plan_id: str,
    count: int,
    func_type_labels: list[str] | None,
    mode: str,
) -> str:
    """Build a live sample-plan table for the generation UI."""
    selected = parse_func_type_labels(func_type_labels or [])
    if not selected:
        selected = ALL_FUNCTION_TYPES[:1]
    is_all = str(TOTAL_SUBSET_COUNT) in mode
    active_types = ALL_FUNCTION_TYPES if is_all else selected
    count_map = _resolve_sample_plan(count, plan_id)

    header_type = "Function"
    header_count = "Per-subset count"
    header_selected = "Included now"
    yes_text = "Yes"
    no_text = "No"
    total_images = sum(count_map[ft] for ft in active_types)
    if is_all:
        total_images *= PROFILE_COUNT
    plan_name = (
        'Recommended benchmark allocation'
        if plan_id == "recommended"
        else 'Uniform count for every function type'
    )

    lines = [
        f"**{'Per-function sample plan'}**",
        "",
        f"{'Active allocation'}: `{plan_name}`",
    ]
    if plan_id == "recommended":
        lines.append('Recommended mode uses the tested per-function allocation and ignores the uniform-count slider.')
    else:
        lines.append('Uniform mode: every included subset uses `{count}` images.'.format(count=int(count)))
    lines.extend([
        "",
        f"{'Estimated images for current selection'}: **{total_images}**",
        "",
        f"| {header_type} | {header_count} | {header_selected} |",
        "|---|---:|---|",
    ])
    active_set = set(active_types)
    for ftype in ALL_FUNCTION_TYPES:
        lines.append(
            f"| `{ftype}` | {count_map[ftype]} | {yes_text if ftype in active_set else no_text} |"
        )
    return "\n".join(lines)


def benchmark_dimensions_md(lang: str) -> str:
    """Build benchmark-dimension markdown with live counts."""
    return (
        f"**Benchmark Dimensions** ({FUNCTION_TYPE_COUNT} function families, "
        f"~1,200 samples, clean_render profile only)"
    )

def _generate_one_subset(
    func_type: str, profile_id: str,
    count: int, seed: int, is_overwrite: bool,
    adv_overrides: dict,
    progress_iter=None,
) -> Tuple[str, int]:
    """Generate images for a single (func_type, profile) subset."""
    sid = f"{func_type}-{profile_id}"
    config = build_subset_config(func_type, profile_id)

    for key_path, val in adv_overrides.items():
        keys = key_path.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val

    img_sub = _images_dir / sid
    gt_sub = _gt_dir / sid

    if is_overwrite:
        clear_subset(sid)

    img_sub.mkdir(parents=True, exist_ok=True)
    gt_sub.mkdir(parents=True, exist_ok=True)
    start_idx = get_next_index(img_sub)

    exact_seen: set[str] = set()
    family_counts: dict[str, int] = {}
    indices = progress_iter if progress_iter is not None else range(count)
    for i in indices:
        idx = start_idx + i
        stem = f"{sid}_{idx:04d}"
        img_path = str(img_sub / f"{stem}.png")
        gt_path = str(gt_sub / f"{stem}.json")
        s = seed + i if seed > 0 else None
        page_cfg = deepcopy(config.get("page", {}))
        coord_mode = config.get("coords", {}).get("coord_mode", "manual")
        generate_and_save_diverse_image(
            img_cfg=deepcopy(config),
            img_path=img_path,
            gt_path=gt_path,
            seed=s,
            sid=sid,
            ftype=func_type,
            profile_id=profile_id,
            idx=idx,
            coord_mode=coord_mode,
            page_choice=page_cfg,
            exact_seen=exact_seen,
            family_counts=family_counts,
        )

    return sid, count


def _build_type_cap_md(lang: str) -> str:
    """Build a markdown table showing per-type intersection caps."""
    header_type = "Function"
    header_cap = "Cap"
    lines = [f"| {header_type} | {header_cap} |", "|---|---|"]
    for ftype, cap in TYPE_INTERSECTION_CAP.items():
        display = str(cap) if cap < 99 else "unlimited"
        lines.append(f"| `{ftype}` | {display} |")
    return "\n".join(lines)


def clear_dataset_ui(lang: str) -> Tuple[str, str]:
    """Clear all generated data and return status messages."""
    clear_dataset()
    status = '✅ All images and ground truth cleared.'
    stats = ""
    return status, stats


def generate_images(
    count: int,
    seed: int,
    mode: str,
    sample_plan_id: str,
    func_type_labels: list[str],
    profile_id: str,
    coord_mode_label: str,
    page_width: int,
    page_height: int,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    noise_enabled: bool,
    min_intersections: int,
    lang: str,
    progress=None,
) -> Tuple[str, str]:
    """Generate images — selected subsets or all benchmark subsets depending on mode."""
    progress = progress or _NoOpProgress()
    # Guard against None values from Gradio slider state sync issues
    count = int(count or 10)
    seed = int(seed or 42)
    page_width = int(page_width or 1440)
    page_height = int(page_height or 1440)
    x_min = int(x_min if x_min is not None else -5)
    x_max = int(x_max if x_max is not None else 5)
    y_min = int(y_min if y_min is not None else -5)
    y_max = int(y_max if y_max is not None else 5)
    min_intersections = int(min_intersections or 0)

    is_all = str(TOTAL_SUBSET_COUNT) in mode
    count_map = _resolve_sample_plan(count, sample_plan_id)

    cm = _resolve_coord_mode(coord_mode_label)
    adv: dict = {
        "page.width": page_width,
        "page.height": page_height,
        "coords.coord_mode": cm,
        "coords.min_intersections": min_intersections,
        "noise.enabled": bool(noise_enabled),
    }
    if cm == "manual":
        adv["coords.x_range"] = [int(x_min), int(x_max)]
        adv["coords.y_range"] = [int(y_min), int(y_max)]

    if is_all:
        clear_dataset()
        from itertools import product as _product
        subsets = list(_product(ALL_FUNCTION_TYPES, BENCHMARK_PROFILES))
        generated_info: list[str] = []
        counter = 0
        total_images = sum(count_map.get(ft, int(count)) for ft, _ in subsets)
        for ft, pid in subsets:
            subset_count = count_map.get(ft, int(count))
            desc_inner = f"{ft} ({counter}/{total_images})"
            inner_iter = progress.tqdm(range(subset_count), desc=desc_inner)
            sid, n = _generate_one_subset(
                ft, pid, int(subset_count), int(seed), is_overwrite=False,
                adv_overrides=adv,
                progress_iter=inner_iter,
            )
            generated_info.append(sid)
            counter += n

        sid_str = ", ".join(generated_info[:10]) + (" ..." if len(generated_info) > 10 else "")
        status = f"✅ All {len(subsets)} subsets complete, {counter} images total"
        stats = f"Output: {_images_dir}\nSubsets: {sid_str}"
    else:
        func_types = parse_func_type_labels(func_type_labels or [])
        if not func_types:
            func_types = ALL_FUNCTION_TYPES[:1]
        selected_profile = profile_id or DEFAULT_PROFILE_ID

        desc = "Generating images"
        generated_info = []
        counter = 0
        for ft in func_types:
            subset_count = count_map.get(ft, int(count))
            prog_iter = progress.tqdm(range(int(subset_count)), desc=f"{desc} [{ft}]")
            sid, n = _generate_one_subset(
                ft, selected_profile, int(subset_count), int(seed), is_overwrite=False,
                adv_overrides=adv, progress_iter=prog_iter,
            )
            generated_info.append(sid)
            counter += n

        sid_str = ", ".join(generated_info)
        status = f"✅ Generated {counter} images ({len(func_types)} types)"
        stats = f"Subsets: {sid_str}"

    total_all = len(list(_images_dir.rglob("*.png")))
    stats += f"\nDataset total: {total_all}"

    plan_name = (
        'Recommended benchmark allocation'
        if sample_plan_id == "recommended"
        else 'Uniform count for every function type'
    )
    stats += f"\n{'Active allocation'}: {plan_name}"

    return status, stats


# ==================== Tab 2: Dataset Preview ====================

def format_expressions_rendered(gt: dict, lang: str) -> str:
    """Format function expressions as rendered LaTeX in Markdown."""
    if not gt or "functions" not in gt:
        return ""
    functions = gt["functions"]
    if not functions:
        return ""
    header = "**Function Expressions**"
    lines = [header, ""]
    for i, fn in enumerate(functions, 1):
        ftype = fn.get("type", "unknown")
        latex = fn.get("expression_latex", "")
        if latex:
            lines.append(f"**{i}.** `{ftype}` $\\quad$ $${latex}$$")
        else:
            lines.append(f"**{i}.** `{ftype}` — N/A")
    return "\n\n".join(lines)


# ==================== Rich Preview Markdown Builders ====================

def _fmt_number(v) -> str:
    """Format a number, rounding floats to 4 decimal places."""
    import math
    if isinstance(v, float):
        if math.isinf(v):
            return "∞" if v > 0 else "-∞"
        if math.isnan(v):
            return "NaN"
        if v == int(v):
            return str(int(v))
        return f"{v:.4g}"
    return str(v)


def _build_expression_md(gt: dict, lang: str) -> str:
    """Build the function expression section with LaTeX rendering."""
    functions = gt.get("functions", [])
    if not functions:
        return ""
    zh = False
    lines = []
    for i, fn in enumerate(functions, 1):
        ftype = fn.get("type", "unknown")
        latex = fn.get("expression_latex", "")
        sympy_expr = fn.get("expression_sympy", "")
        params = fn.get("parameters", {})

        lines.append(f"#### {'Curve'} {i} · `{ftype}`")
        if latex:
            lines.append(f"$${latex}$$")
        if sympy_expr:
            lines.append(f"{'Symbolic'}: `{sympy_expr}`")
        if params:
            param_str = ", ".join(f"`{k}`={_fmt_number(v)}" for k, v in params.items())
            lines.append(f"{'Parameters'}: {param_str}")
        lines.append("")
    return "\n".join(lines)


def _build_properties_md(gt: dict, lang: str) -> str:
    """Build math properties section (domain, range, zeros)."""
    functions = gt.get("functions", [])
    if not functions:
        return ""
    zh = False
    lines = []
    for i, fn in enumerate(functions, 1):
        mp = fn.get("math_properties", {})
        if not mp:
            continue
        ftype = fn.get("type", "unknown")
        lines.append(f"**{'Curve'} {i}** (`{ftype}`)")
        domain = mp.get("domain", "—")
        rng = mp.get("range", "—")
        zeros = mp.get("zeros", [])
        zeros_str = ", ".join(_fmt_number(z) for z in zeros) if zeros else ("None")
        lines.append(f"| {'Property'} | {'Value'} |")
        lines.append("|:---|:---|")
        lines.append(f"| {'Domain'} | ${domain}$ |")
        lines.append(f"| {'Range'} | ${rng}$ |")
        lines.append(f"| {'Zeros'} | {zeros_str} |")
        lines.append("")
    return "\n".join(lines)


def _build_extended_props_md(gt: dict, lang: str) -> str:
    """Build extended properties: intercepts, extrema, monotonicity, symmetry."""
    def _append_fact_bundle(lines: list[str], facts: dict, zh: bool, title: str):
        lines.append(f"### {title}")
        lines.append("")

        intercepts = facts.get("intercepts", {})
        x_ints = intercepts.get("x_intercepts", []) if isinstance(intercepts, dict) else []
        y_int = intercepts.get("y_intercept") if isinstance(intercepts, dict) else None
        lines.append(f"**{'Intercepts'}**")
        if x_ints:
            pts = ", ".join(f"({_fmt_number(x)}, 0)" for x in x_ints)
            lines.append(f"- x{'-intercepts'}: {pts}")
        else:
            lines.append(f"- x{'-intercepts'}: {'None'}")
        if y_int is not None:
            lines.append(f"- y{'-intercept'}: (0, {_fmt_number(y_int)})")
        else:
            lines.append(f"- y{'-intercept'}: {'None'}")
        lines.append("")

        extrema = facts.get("extrema", [])
        lines.append(f"**{'Extrema'}**")
        if extrema:
            lines.append(f"| {'Type'} | x | y |")
            lines.append("|:---|:---|:---|")
            for e in extrema:
                etype = e.get("type", "?")
                if zh:
                    etype = {"minimum": "极小值", "maximum": "极大值"}.get(etype, etype)
                lines.append(f"| {etype} | {_fmt_number(e.get('x', '?'))} | {_fmt_number(e.get('y', '?'))} |")
        else:
            lines.append(f"- {'None'}")
        lines.append("")

        mono = facts.get("monotonicity", [])
        lines.append(f"**{'Monotonicity'}**")
        if mono:
            for seg in mono:
                interval = seg.get("interval", [])
                direction = seg.get("direction", "?")
                if zh:
                    direction = {"increasing": "递增", "decreasing": "递减"}.get(direction, direction)
                if len(interval) == 2:
                    a, b = interval
                    lines.append(f"- $[{_fmt_number(a)},\\, {_fmt_number(b)}]$: {direction}")
        else:
            lines.append(f"- {'None'}")
        lines.append("")

    q = gt.get("question", {})
    gt_data = q.get("ground_truth", {})
    if not gt_data:
        return ""
    zh = False
    lines = []
    _append_fact_bundle(lines, gt_data, zh, "Mathematical Ground Truth (Global)")
    visible_facts = gt_data.get("visible_facts", {})
    if isinstance(visible_facts, dict) and visible_facts:
        _append_fact_bundle(
            lines,
            visible_facts,
            zh,
            "Visible-Range Ground Truth (Evaluation Scope)",
        )

    sym = gt_data.get("symmetry")
    if sym:
        sym_label = sym
        if zh:
            sym_label = {"even": "偶函数", "odd": "奇函数", "neither": "非奇非偶"}.get(sym, sym)
        lines.append(f"**{'Symmetry'}**: {sym_label}")
        lines.append("")

    return "\n".join(lines)


def _build_intersections_md(gt: dict, lang: str) -> str:
    """Build intersections / key coordinates section."""
    zh = False
    lines = []

    intersections = gt.get("intersections", [])
    if intersections:
        lines.append(f"**{'Intersection Points'}** ({len(intersections)}{''})")
        lines.append(f"| # | x | y | {'Type'} |")
        lines.append("|:---|:---|:---|:---|")
        for idx, c in enumerate(intersections, 1):
            ctype = c.get("type", "")
            lines.append(f"| {idx} | {_fmt_number(c['coordinate_x'])} | {_fmt_number(c['coordinate_y'])} | {ctype} |")
        lines.append("")

    return "\n".join(lines)


def _build_tasks_md(gt: dict, lang: str) -> str:
    """Build prediction targets and rate of change section."""
    q = gt.get("question", {})
    gt_data = q.get("ground_truth", {})
    if not gt_data:
        return ""
    zh = False
    lines = []

    q_text = q.get("question_text_zh" if zh else "question_text", "")
    if q_text:
        lines.append(f"**{'Question'}**")
        lines.append(f"> {q_text}")
        lines.append("")

    pred_tgt = gt_data.get("prediction_targets", {})
    if pred_tgt:
        lines.append(f"**{'Value Predictions'}**")
        lines.append(f"| x | f(x) |")
        lines.append("|:---|:---|")
        for x, y in pred_tgt.items():
            lines.append(f"| {x} | {_fmt_number(y)} |")
        lines.append("")

    roc = gt_data.get("rate_of_change", {})
    if roc and "gt_value" in roc:
        lines.append(f"**{'Rate of Change'}**")
        x1 = roc.get("x1", "?")
        x2 = roc.get("x2", "?")
        val = _fmt_number(roc["gt_value"])
        lines.append(f"$x \\in [{x1},\\, {x2}]$ → $\\Delta y / \\Delta x = {val}$")
        lines.append("")

    return "\n".join(lines)


def _build_meta_md(gt: dict, image_name: str, img_shape: tuple, lang: str) -> str:
    """Build image/grid metadata section."""
    zh = False
    lines = []

    lines.append(f"| {'Item'} | {'Value'} |")
    lines.append("|:---|:---|")
    lines.append(f"| {'Filename'} | `{image_name}` |")
    lines.append(f"| {'Dimensions'} | {img_shape[1]} × {img_shape[0]} px |")

    if gt:
        xr = gt.get("x_range", [])
        yr = gt.get("y_range", [])
        if xr:
            lines.append(f"| x{' range'} | [{xr[0]}, {xr[1]}] |")
        if yr:
            lines.append(f"| y{' range'} | [{yr[0]}, {yr[1]}] |")
        layout = gt.get("layout", "")
        if layout:
            lines.append(f"| {'Layout'} | `{layout}` |")
        bm = gt.get("benchmark_meta", {})
        if bm:
            ft = bm.get("func_type", bm.get("difficulty", ""))
            profile_id = bm.get("profile_id", "")
            sid = bm.get("subset_id", "")
            if sid:
                lines.append(f"| {'Subset'} | `{sid}` |")
            if ft:
                lines.append(f"| {'Func type'} | `{ft}` |")
            if profile_id:
                lines.append(f"| {'Profile'} | `{profile_id}` |")

    lines.append("")
    return "\n".join(lines)


def preview_image(image_name: str, lang: str) -> Tuple[
    np.ndarray | None, str, str, str, str, str, str
]:
    """Preview selected image with full structured ground truth.

    Returns:
        (image, expr_md, props_md, extended_md, intersections_md, tasks_md, meta_md)
    """
    empty = (None, "", "", "", "", "", "")
    if not image_name:
        return empty

    img, gt = load_image_and_gt(image_name)
    if img is None:
        return empty

    if not gt:
        meta = _build_meta_md({}, image_name, img.shape, lang)
        return img, "", "", "", "", "", meta

    expr_md = _build_expression_md(gt, lang)
    props_md = _build_properties_md(gt, lang)
    extended_md = _build_extended_props_md(gt, lang)
    ints_md = _build_intersections_md(gt, lang)
    tasks_md = _build_tasks_md(gt, lang)
    meta_md = _build_meta_md(gt, image_name, img.shape, lang)

    return img, expr_md, props_md, extended_md, ints_md, tasks_md, meta_md


def refresh_dataset():
    """Refresh image-selection state without depending on a UI framework."""
    images = list_dataset_images()
    first = images[0] if images else None
    selected = {"choices": images, "value": first, "allow_custom_value": True}
    empty = {"choices": images, "value": None, "allow_custom_value": True}
    return selected, empty, empty
