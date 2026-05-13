"""Expression recognition, evaluation, and Expression PK. Extracted from app.py."""

import json
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import numpy as np

from app.helpers import (
    _root, _images_dir,
    _get_saved_key,
    _create_llm_adapter, list_dataset_images, load_image_and_gt,
)
from i18n import t, LANG_EN, LANG_ZH
from evaluator.adapters.base import BaseLLMAdapter


__compute_image_expr_metrics = None
__aggregate_expr_metrics = None


def _ensure_expr_metrics():
    """Lazy-load evaluator.expr_metrics (avoids importing sympy ~48 MB at module load)."""
    global _compute_image_expr_metrics, _aggregate_expr_metrics
    if _compute_image_expr_metrics is not None:
        return
    from evaluator.metrics.expression import (
        compute_image_expr_metrics as _ciem,
        aggregate_expr_metrics as _aem,
    )
    _compute_image_expr_metrics = _ciem
    _aggregate_expr_metrics = _aem

EXPR_PK_PAGE_SIZE = 20
_expr_pk_results_path = _root / "reports" / "expr_pk_results.json"


def _load_expr_prompt_direct() -> str:
    """Load expression recognition prompt template (direct mode)."""
    p = _root / "prompts" / "expr_prompt_direct.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def _load_expr_prompt_geosym() -> str:
    """Load expression recognition prompt template (GeoSym mode)."""
    p = _root / "prompts" / "expr_prompt_geosym.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def build_expr_prompt_direct(gt: dict) -> str:
    """Build the direct mode expression recognition prompt."""
    template = _load_expr_prompt_direct()
    x_range = gt.get("x_range", [-5, 5])
    y_range = gt.get("y_range", [-5, 5])
    n_curves = len(gt.get("functions", []))
    return template.format(
        n_curves=n_curves,
        x_min=int(x_range[0]),
        x_max=int(x_range[1]),
        y_min=int(y_range[0]),
        y_max=int(y_range[1]),
    )


def build_expr_prompt_geosym(gt: dict, detected_points: list) -> str:
    """Build the GeoSym mode expression recognition prompt."""
    template = _load_expr_prompt_geosym()
    x_range = gt.get("x_range", [-5, 5])
    y_range = gt.get("y_range", [-5, 5])
    n_curves = len(gt.get("functions", []))
    # Format points list
    if detected_points:
        points_str = ", ".join(f"({x}, {y})" for x, y in detected_points)
    else:
        points_str = "(no points detected)"
    return template.format(
        n_curves=n_curves,
        x_min=int(x_range[0]),
        x_max=int(x_range[1]),
        y_min=int(y_range[0]),
        y_max=int(y_range[1]),
        points_list=points_str,
    )


def _run_expr_direct(method: str, api_key: str, image_path: str, gt: dict) -> Tuple[List[str], str]:
    """
    Run expression recognition in Direct mode (image only).
    Returns (list of expression strings, raw response).
    """
    key = api_key.strip() if api_key else ""
    if not key:
        key = _get_saved_key(method)
    is_claude = "Claude" in method
    if not key and not is_claude:
        return [], "API key is empty."

    adapter = _create_llm_adapter(method, key)
    if adapter is None:
        return [], f"Unknown method: {method}"

    prompt = build_expr_prompt_direct(gt)
    raw_response = adapter.call_with_prompt(image_path, prompt)
    exprs = BaseLLMAdapter._parse_expr_response(raw_response)
    return exprs, raw_response


def _run_expr_geosym(method: str, api_key: str, image_path: str, gt: dict,
                     gemini_enabled: bool = False) -> Tuple[List[str], str, list, "np.ndarray | None", bool]:
    """
    Run expression recognition in GeoSym mode:
      1. Run GeoSym coordinate recognition first
      2. Send GeoSym's annotated image + detected coordinates to LLM

    Returns:
        (expressions, raw_response, detected_coords, annotated_image, geosym_success)
        *geosym_success* is False when GeoSym extraction itself failed
        (as opposed to simply detecting zero coordinates).
    """
    key = api_key.strip() if api_key else ""
    if not key:
        key = _get_saved_key(method)
    is_claude = "Claude" in method
    if not key and not is_claude:
        return [], "API key is empty.", [], None, False

    # Step 1: Run GeoSym
    from evaluator.adapters.recognizer import RecognizerAdapter
    recognizer = RecognizerAdapter()
    detected_coords, result, annotated_image = recognizer.recognize_intersections(
        image_path, gt, gemini_enabled=gemini_enabled)

    geosym_success = result.get("success", False) if isinstance(result, dict) else bool(detected_coords)

    # Step 2: Build prompt with detected coordinates
    prompt = build_expr_prompt_geosym(gt, detected_coords)

    # Step 3: Send original image + coordinate-enriched prompt to LLM
    adapter = _create_llm_adapter(method, key)
    if adapter is None:
        return [], f"Unknown method: {method}", detected_coords, annotated_image, geosym_success

    raw_response = adapter.call_with_prompt(image_path, prompt)

    exprs = BaseLLMAdapter._parse_expr_response(raw_response)
    if not exprs and raw_response:
        import logging
        logging.getLogger(__name__).warning(
            "[ExprGeoSym] %s returned non-empty response but no expressions parsed", method)

    return exprs, raw_response, detected_coords, annotated_image, geosym_success


def _format_expr_result_single(
    gt_functions: list,
    direct_exprs: list,
    geosym_exprs: list,
    direct_metrics: dict,
    geosym_metrics: dict,
    direct_raw: str,
    geosym_raw: str,
    lang: str,
) -> str:
    """Format expression recognition results for single image display."""
    is_zh = lang == LANG_ZH

    lines = []

    # Summary comparison table
    if is_zh:
        lines.append("#### 表达式识别对比\n")
        lines.append("| 指标 | 直接输入 | GeoSym |")
    else:
        lines.append("#### Expression Recognition Comparison\n")
        lines.append("| Metric | Direct | GeoSym-Assisted |")
    lines.append("|--------|-------:|----------------:|")
    lines.append(f"| R² | {direct_metrics.get('mean_r2', 0):.4f} | {geosym_metrics.get('mean_r2', 0):.4f} |")
    lines.append(f"| {'符号等价率' if is_zh else 'Sym Equiv Rate'} | {direct_metrics.get('sym_equiv_rate', 0):.0%} | {geosym_metrics.get('sym_equiv_rate', 0):.0%} |")
    lines.append(f"| {'类型准确率' if is_zh else 'Type Accuracy'} | {direct_metrics.get('type_accuracy', 0):.0%} | {geosym_metrics.get('type_accuracy', 0):.0%} |")
    lines.append(f"| NRMSE | {direct_metrics.get('mean_nrmse', 0):.4f} | {geosym_metrics.get('mean_nrmse', 0):.4f} |")

    # Per-curve detail
    lines.append("")
    if is_zh:
        lines.append("#### 逐曲线详情\n")
        lines.append("| 真值表达式 | 直接预测 | GeoSym 预测 | R²(直接) | R²(GeoSym) | 符号等价(直接) | 符号等价(GeoSym) |")
    else:
        lines.append("#### Per-Curve Detail\n")
        lines.append("| GT Expression | Direct Pred | GeoSym Pred | R²(Direct) | R²(GeoSym) | SymEq(Direct) | SymEq(GeoSym) |")
    lines.append("|---|---|---|---:|---:|:---:|:---:|")

    direct_curves = direct_metrics.get("per_curve", [])
    geosym_curves = geosym_metrics.get("per_curve", [])

    n = max(len(direct_curves), len(geosym_curves))
    for i in range(n):
        dc = direct_curves[i] if i < len(direct_curves) else {}
        gc = geosym_curves[i] if i < len(geosym_curves) else {}
        gt_e = dc.get("gt_expr_str", gc.get("gt_expr_str", "-"))
        d_e = dc.get("pred_expr_str", "-")
        g_e = gc.get("pred_expr_str", "-")
        d_r2 = f"{dc.get('r2', 0):.4f}" if dc else "-"
        g_r2 = f"{gc.get('r2', 0):.4f}" if gc else "-"
        d_sym = "Yes" if dc.get("sym_equiv") else "No"
        g_sym = "Yes" if gc.get("sym_equiv") else "No"
        lines.append(f"| `{gt_e}` | `{d_e}` | `{g_e}` | {d_r2} | {g_r2} | {d_sym} | {g_sym} |")

    # Raw responses (collapsed)
    lines.append("")
    lines.append(f"<details><summary>{'直接模式原始响应' if is_zh else 'Direct Mode Raw Response'}</summary>\n\n```\n{direct_raw}\n```\n</details>")
    lines.append(f"<details><summary>{'GeoSym 模式原始响应' if is_zh else 'GeoSym Mode Raw Response'}</summary>\n\n```\n{geosym_raw}\n```\n</details>")

    return "\n".join(lines)


def _save_expr_pk_results(summary_text: str, pk_data: dict):
    """Save expression PK results to disk for persistence."""
    _expr_pk_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Build serializable version
    ser_map = {}
    for method_name, mode_dict in pk_data.get("method_mode_map", {}).items():
        ser_map[method_name] = {}
        for mode, img_dict in mode_dict.items():
            ser_map[method_name][mode] = {}
            for img_name, r in img_dict.items():
                # Serialize per_curve data for advanced charts
                per_curve_ser = []
                for pc in r.get("per_curve", []):
                    per_curve_ser.append({
                        "gt_type": pc.get("gt_type", "unknown"),
                        "pred_type": pc.get("pred_type", "unknown"),
                        "r2": float(pc.get("r2", 0.0)),
                        "nrmse": float(pc.get("nrmse", 0.0)),
                        "type_match": bool(pc.get("type_match", False)),
                        "sym_equiv": bool(pc.get("sym_equiv", False)),
                    })
                ser_map[method_name][mode][img_name] = {
                    "mean_r2": r.get("mean_r2", 0.0),
                    "sym_equiv_rate": r.get("sym_equiv_rate", 0.0),
                    "type_accuracy": r.get("type_accuracy", 0.0),
                    "mean_nrmse": r.get("mean_nrmse", 0.0),
                    "num_gt": r.get("num_gt", 0),
                    "num_pred": r.get("num_pred", 0),
                    "num_matched": r.get("num_matched", 0),
                    "per_curve": per_curve_ser,
                }

    # Serialize excluded info
    ser_excluded = {}
    for method_name, excl_list in pk_data.get("excluded", {}).items():
        ser_excluded[method_name] = [
            {"image": img_name, "reason": reason}
            for img_name, reason in excl_list
        ]

    save_obj = {
        "summary_text": summary_text,
        "images": pk_data.get("images", []),
        "methods": pk_data.get("methods", []),
        "method_mode_map": ser_map,
        "excluded": ser_excluded,
    }
    with open(_expr_pk_results_path, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, indent=2, ensure_ascii=False)


def _load_expr_pk_results() -> Tuple[str, dict, str, str]:
    """Load previously saved expression PK results."""
    if not _expr_pk_results_path.exists():
        return "", {}, "", "- / -"
    try:
        with open(_expr_pk_results_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        summary_text = saved.get("summary_text", "")
        pk_data = {
            "images": saved.get("images", []),
            "methods": saved.get("methods", []),
            "method_mode_map": saved.get("method_mode_map", {}),
        }
        images = pk_data["images"]
        total_pages = max(1, -(-len(images) // EXPR_PK_PAGE_SIZE))
        detail = _render_expr_pk_detail_page(pk_data, page=0, lang=LANG_ZH)
        return summary_text, pk_data, detail, f"1 / {total_pages}"
    except Exception:
        return "", {}, "", "- / -"


def _render_expr_pk_detail_page(
    pk_data: dict,
    page: int = 0,
    lang: str = LANG_ZH,
    search_query: str = "",
) -> str:
    """Render one page of the per-image expression PK detail table."""
    if not pk_data:
        return ""

    images = pk_data.get("images", [])
    methods = pk_data.get("methods", [])
    method_mode_map = pk_data.get("method_mode_map", {})

    if search_query.strip():
        query = search_query.strip().lower()
        images = [img for img in images if query in img.lower()]

    if not images:
        return "No matching images." if lang == LANG_EN else "没有匹配的图像。"

    total_pages = max(1, -(-len(images) // EXPR_PK_PAGE_SIZE))
    page = max(0, min(page, total_pages - 1))

    start = page * EXPR_PK_PAGE_SIZE
    end = min(start + EXPR_PK_PAGE_SIZE, len(images))
    page_images = images[start:end]

    is_zh = lang == LANG_ZH

    # Build header: for each method, two columns (Direct, GeoSym)
    header_cells = ["图像" if is_zh else "Image"]
    sep_cells = ["|------"]
    for method_name in methods:
        header_cells.append(f"{method_name} Direct")
        header_cells.append(f"{method_name} GeoSym")
        sep_cells.append("|------")
        sep_cells.append("|------")

    lines = [
        f"**{'每张图像详情' if is_zh else 'Per-Image Detail'}** — {'第' if is_zh else 'Page'} {page + 1}/{total_pages} ({'共' if is_zh else ''}{len(images)} {'张' if is_zh else 'images'})\n",
        "| " + " | ".join(header_cells) + " |",
        "".join(sep_cells) + "|",
    ]

    for image_name in page_images:
        cells = [f"| {image_name}"]
        for method_name in methods:
            modes = method_mode_map.get(method_name, {})
            for mode in ("direct", "geosym"):
                r = modes.get(mode, {}).get(image_name)
                if r:
                    cells.append(f"R²:{r.get('mean_r2', 0):.2f} SE:{r.get('sym_equiv_rate', 0):.0%}")
                else:
                    cells.append("-")
        lines.append(" | ".join(cells) + " |")

    return "\n".join(lines)


def run_expr_single(
    image_name: str,
    method: str,
    lang: str,
    gemini_enabled: bool = False,
) -> str:
    """Run expression recognition on a single image with both modes."""
    if not image_name:
        return t("no_image_selected", lang)
    if not method:
        return "Please select a model." if lang == LANG_EN else "请选择模型。"

    img, gt = load_image_and_gt(image_name)
    if img is None or gt is None:
        return t("image_not_found", lang)

    gt_functions = gt.get("functions", [])
    if not gt_functions:
        return "No functions in ground truth." if lang == LANG_EN else "真值中无函数。"

    x_range = tuple(gt.get("x_range", [-5, 5]))
    img_path = str(_images_dir / f"{image_name}.png")
    api_key = _get_saved_key(method)

    # Direct mode
    try:
        direct_exprs, direct_raw = _run_expr_direct(method, api_key, img_path, gt)
    except Exception as e:
        direct_exprs, direct_raw = [], f"Error: {e}"

    # GeoSym mode
    try:
        geosym_exprs, geosym_raw, geo_coords, geo_img, _ = _run_expr_geosym(
            method, api_key, img_path, gt, gemini_enabled=gemini_enabled)
    except Exception as e:
        geosym_exprs, geosym_raw = [], f"Error: {e}"

    # Compute metrics
    _ensure_expr_metrics()
    direct_metrics = __compute_image_expr_metrics(direct_exprs, gt_functions, x_range)
    geosym_metrics = __compute_image_expr_metrics(geosym_exprs, gt_functions, x_range)

    return _format_expr_result_single(
        gt_functions, direct_exprs, geosym_exprs,
        direct_metrics, geosym_metrics,
        direct_raw, geosym_raw, lang,
    )


def run_expr_multi_model(
    image_name: str,
    selected_methods: List[str],
    lang: str,
    gemini_enabled: bool = False,
) -> str:
    """Run expression recognition on a single image with multiple models."""
    _ensure_expr_metrics()
    if not image_name:
        return t("no_image_selected", lang)
    if not selected_methods:
        return "Please select at least one model." if lang == LANG_EN else "请至少选择一个模型。"

    img, gt = load_image_and_gt(image_name)
    if img is None or gt is None:
        return t("image_not_found", lang)

    gt_functions = gt.get("functions", [])
    if not gt_functions:
        return "No functions in ground truth." if lang == LANG_EN else "真值中无函数。"

    x_range = tuple(gt.get("x_range", [-5, 5]))
    img_path = str(_images_dir / f"{image_name}.png")
    is_zh = lang == LANG_ZH

    # Header
    header = ["模型" if is_zh else "Model", "模式" if is_zh else "Mode", "R²",
              "符号等价" if is_zh else "Sym Equiv", "类型准确" if is_zh else "Type Acc", "NRMSE"]
    lines = [
        f"#### {'多模型函数识别对比' if is_zh else 'Multi-Model Expression Recognition Comparison'} — {image_name}\n",
        "| " + " | ".join(header) + " |",
        "|---|---|---:|:---:|:---:|---:|",
    ]

    for method in selected_methods:
        api_key = _get_saved_key(method)
        is_claude = "Claude" in method
        if not api_key and not is_claude:
            lines.append(f"| {method} | - | - | - | - | - |")
            continue

        # Direct
        try:
            d_exprs, _ = _run_expr_direct(method, api_key, img_path, gt)
            d_m = _compute_image_expr_metrics(d_exprs, gt_functions, x_range)
            lines.append(f"| {method} | Direct | {d_m['mean_r2']:.4f} | {d_m['sym_equiv_rate']:.0%} | {d_m['type_accuracy']:.0%} | {d_m['mean_nrmse']:.4f} |")
        except Exception as e:
            lines.append(f"| {method} | Direct | Error | - | - | - |")

        # GeoSym
        try:
            g_exprs, _, _, _, _ = _run_expr_geosym(method, api_key, img_path, gt,
                                                    gemini_enabled=gemini_enabled)
            g_m = _compute_image_expr_metrics(g_exprs, gt_functions, x_range)
            lines.append(f"| {method} | GeoSym | {g_m['mean_r2']:.4f} | {g_m['sym_equiv_rate']:.0%} | {g_m['type_accuracy']:.0%} | {g_m['mean_nrmse']:.4f} |")
        except Exception as e:
            lines.append(f"| {method} | GeoSym | Error | - | - | - |")

    return "\n".join(lines)


def run_expr_batch(
    method: str,
    lang: str,
    gemini_enabled: bool = False,
    progress=gr.Progress(),
) -> Tuple[str, str]:
    """Run expression recognition on all images with a single model (both modes)."""
    _ensure_expr_metrics()
    images = list_dataset_images()
    is_zh = lang == LANG_ZH

    if not images:
        msg = "没有可评估的图像" if is_zh else "No images to evaluate"
        return msg, ""
    if not method:
        msg = "请选择模型" if is_zh else "Please select a model"
        return msg, ""

    api_key = _get_saved_key(method)
    is_claude = "Claude" in method
    if not api_key and not is_claude:
        return "No API Key", ""

    direct_results = []
    geosym_results = []
    per_image_details = []
    total = len(images)

    for idx, image_name in enumerate(images):
        progress((idx + 1) / total, desc=f"{'运行' if is_zh else 'Running'} {method}: {image_name}")

        img, gt = load_image_and_gt(image_name)
        if img is None or gt is None:
            continue

        gt_functions = gt.get("functions", [])
        if not gt_functions:
            continue

        x_range = tuple(gt.get("x_range", [-5, 5]))
        img_path = str(_images_dir / f"{image_name}.png")

        # Direct
        d_m = {"mean_r2": 0, "sym_equiv_rate": 0, "type_accuracy": 0, "mean_nrmse": 0,
               "num_gt": len(gt_functions), "num_pred": 0, "num_matched": 0, "num_missed": len(gt_functions), "num_extra": 0,
               "num_sym_equiv": 0, "num_type_match": 0, "sum_r2": 0.0, "sum_nrmse": 0.0}
        try:
            d_exprs, _ = _run_expr_direct(method, api_key, img_path, gt)
            d_m = _compute_image_expr_metrics(d_exprs, gt_functions, x_range)
        except Exception as e:
            print(f"[ExprBatch] Direct mode error for {method}/{image_name}: {e}")
        direct_results.append(d_m)

        # GeoSym
        g_m = {"mean_r2": 0, "sym_equiv_rate": 0, "type_accuracy": 0, "mean_nrmse": 0,
               "num_gt": len(gt_functions), "num_pred": 0, "num_matched": 0, "num_missed": len(gt_functions), "num_extra": 0,
               "num_sym_equiv": 0, "num_type_match": 0, "sum_r2": 0.0, "sum_nrmse": 0.0}
        try:
            g_exprs, _, _, _, _ = _run_expr_geosym(method, api_key, img_path, gt,
                                                    gemini_enabled=gemini_enabled)
            g_m = _compute_image_expr_metrics(g_exprs, gt_functions, x_range)
        except Exception as e:
            print(f"[ExprBatch] GeoSym mode error for {method}/{image_name}: {e}")
        geosym_results.append(g_m)

        per_image_details.append((image_name, d_m, g_m))

    # Aggregate
    d_agg = _aggregate_expr_metrics(direct_results)
    g_agg = _aggregate_expr_metrics(geosym_results)

    # Summary
    if is_zh:
        summary_lines = [
            f"**{method} 批量函数识别 — {len(images)} 张图像**\n",
            "| 模式 | 图像数 | R²(Mean) | R²(Micro) | 符号等价(Mean) | 符号等价(Micro) | 类型准确(Mean) | 类型准确(Micro) | NRMSE(Mean) | NRMSE(Micro) |",
            "|------|-------:|--------:|---------:|-----------:|------------:|-----------:|------------:|-----------:|------------:|",
            f"| 直接输入 | {d_agg['num_images']} | {d_agg['mean_r2']:.4f} | {d_agg['micro_r2']:.4f} | {d_agg['mean_sym_equiv_rate']:.0%} | {d_agg['micro_sym_equiv_rate']:.0%} | {d_agg['mean_type_accuracy']:.0%} | {d_agg['micro_type_accuracy']:.0%} | {d_agg['mean_nrmse']:.4f} | {d_agg['micro_nrmse']:.4f} |",
            f"| GeoSym | {g_agg['num_images']} | {g_agg['mean_r2']:.4f} | {g_agg['micro_r2']:.4f} | {g_agg['mean_sym_equiv_rate']:.0%} | {g_agg['micro_sym_equiv_rate']:.0%} | {g_agg['mean_type_accuracy']:.0%} | {g_agg['micro_type_accuracy']:.0%} | {g_agg['mean_nrmse']:.4f} | {g_agg['micro_nrmse']:.4f} |",
        ]
    else:
        summary_lines = [
            f"**{method} Batch Expression Recognition — {len(images)} images**\n",
            "| Mode | Images | R²(Mean) | R²(Micro) | SymEq(Mean) | SymEq(Micro) | TypeAcc(Mean) | TypeAcc(Micro) | NRMSE(Mean) | NRMSE(Micro) |",
            "|------|-------:|--------:|---------:|-----------:|------------:|-----------:|------------:|-----------:|------------:|",
            f"| Direct | {d_agg['num_images']} | {d_agg['mean_r2']:.4f} | {d_agg['micro_r2']:.4f} | {d_agg['mean_sym_equiv_rate']:.0%} | {d_agg['micro_sym_equiv_rate']:.0%} | {d_agg['mean_type_accuracy']:.0%} | {d_agg['micro_type_accuracy']:.0%} | {d_agg['mean_nrmse']:.4f} | {d_agg['micro_nrmse']:.4f} |",
            f"| GeoSym | {g_agg['num_images']} | {g_agg['mean_r2']:.4f} | {g_agg['micro_r2']:.4f} | {g_agg['mean_sym_equiv_rate']:.0%} | {g_agg['micro_sym_equiv_rate']:.0%} | {g_agg['mean_type_accuracy']:.0%} | {g_agg['micro_type_accuracy']:.0%} | {g_agg['mean_nrmse']:.4f} | {g_agg['micro_nrmse']:.4f} |",
        ]

    # Per-image detail table
    if is_zh:
        detail_lines = [
            "\n**逐图详情**\n",
            "| 图像 | R²(直接) | R²(GeoSym) | 符号等价(直接) | 符号等价(GeoSym) |",
            "|------|--------:|-----------:|:-----------:|:---------------:|",
        ]
    else:
        detail_lines = [
            "\n**Per-Image Details**\n",
            "| Image | R²(Direct) | R²(GeoSym) | SymEq(Direct) | SymEq(GeoSym) |",
            "|-------|----------:|-----------:|:-------------:|:-------------:|",
        ]

    for img_name, d_m, g_m in per_image_details:
        detail_lines.append(
            f"| {img_name} | {d_m['mean_r2']:.4f} | {g_m['mean_r2']:.4f} | {d_m['sym_equiv_rate']:.0%} | {g_m['sym_equiv_rate']:.0%} |"
        )

    return "\n".join(summary_lines), "\n".join(detail_lines)


def run_expr_pk(
    selected_methods: List[str],
    lang: str,
    gemini_enabled: bool = False,
    progress=gr.Progress(),
) -> Tuple[str, dict, str, str]:
    """Run expression recognition PK: all selected models on entire dataset, both modes."""
    _ensure_expr_metrics()
    images = list_dataset_images()
    is_zh = lang == LANG_ZH

    if not images:
        msg = "没有可评估的图像" if is_zh else "No images to evaluate"
        return msg, {}, "", "- / -"
    if not selected_methods:
        msg = "请至少选择一个模型" if is_zh else "Please select at least one model"
        return msg, {}, "", "- / -"

    # method_mode_map: {method: {"direct": {img: metrics}, "geosym": {img: metrics}}}
    method_mode_map = {}
    all_method_agg = []  # list of (method, d_agg, g_agg)

    total_steps = len(selected_methods) * len(images) * 2  # x2 for two modes
    current_step = 0

    for method in selected_methods:
        api_key = _get_saved_key(method)
        is_claude = "Claude" in method
        if not api_key and not is_claude:
            method_mode_map[method] = {"direct": {}, "geosym": {}}
            empty_agg = _aggregate_expr_metrics([])
            all_method_agg.append((method, empty_agg, empty_agg, "No API Key", []))
            current_step += len(images) * 2
            progress(current_step / total_steps, desc=f"{method}: skipped")
            continue

        direct_results = []
        geosym_results = []
        method_mode_map[method] = {"direct": {}, "geosym": {}}
        excluded_imgs = []  # [(image_name, reason), ...]

        for image_name in images:
            img, gt = load_image_and_gt(image_name)
            if img is None or gt is None:
                current_step += 2
                progress(current_step / total_steps)
                excluded_imgs.append((image_name, "image/GT load failed"))
                continue

            gt_functions = gt.get("functions", [])
            if not gt_functions:
                current_step += 2
                progress(current_step / total_steps)
                continue

            x_range = tuple(gt.get("x_range", [-5, 5]))
            img_path = str(_images_dir / f"{image_name}.png")

            # Direct mode
            current_step += 1
            progress(current_step / total_steps, desc=f"{method} Direct: {image_name}")
            d_m = None
            d_error = None
            try:
                d_exprs, _ = _run_expr_direct(method, api_key, img_path, gt)
                if d_exprs:
                    d_m = _compute_image_expr_metrics(d_exprs, gt_functions, x_range)
                else:
                    d_error = "Direct: LLM returned empty expressions"
            except Exception as e:
                d_error = f"Direct exception: {str(e)[:80]}"
                print(f"[ExprPK] Direct mode error for {method}/{image_name}: {e}")

            # GeoSym mode
            current_step += 1
            progress(current_step / total_steps, desc=f"{method} GeoSym: {image_name}")
            g_m = None
            g_error = None
            try:
                g_exprs, _, _, _, _ = _run_expr_geosym(method, api_key, img_path, gt,
                                                      gemini_enabled=gemini_enabled)
                if g_exprs:
                    g_m = _compute_image_expr_metrics(g_exprs, gt_functions, x_range)
                else:
                    g_error = "GeoSym: returned empty expressions"
            except Exception as e:
                g_error = f"GeoSym exception: {str(e)[:80]}"
                print(f"[ExprPK] GeoSym mode error for {method}/{image_name}: {e}")

            # If EITHER mode failed, exclude this image entirely for this method
            if d_m is None or g_m is None:
                reasons = []
                if d_error:
                    reasons.append(d_error)
                if g_error:
                    reasons.append(g_error)
                if d_m is None and d_error is None:
                    reasons.append("Direct: no predictions")
                if g_m is None and g_error is None:
                    reasons.append("GeoSym: no predictions")
                reason_str = "; ".join(reasons)
                excluded_imgs.append((image_name, reason_str))
                print(f"[ExprPK] Excluded {image_name} for {method}: {reason_str}")
                continue

            direct_results.append(d_m)
            method_mode_map[method]["direct"][image_name] = d_m
            geosym_results.append(g_m)
            method_mode_map[method]["geosym"][image_name] = g_m

        d_agg = _aggregate_expr_metrics(direct_results)
        g_agg = _aggregate_expr_metrics(geosym_results)
        all_method_agg.append((method, d_agg, g_agg, None, excluded_imgs))

    # Build summary table
    if is_zh:
        summary_lines = [
            f"**函数表达式 PK — {len(images)} 张图像**\n",
            "| 模型 | 模式 | 图像数 | R²(Mean) | R²(Micro) | 符号等价(Mean) | 符号等价(Micro) | 类型准确(Mean) | 类型准确(Micro) | NRMSE(Mean) | NRMSE(Micro) |",
            "|------|------|-------:|--------:|---------:|-----------:|------------:|-----------:|------------:|-----------:|------------:|",
        ]
    else:
        summary_lines = [
            f"**Expression PK — {len(images)} images**\n",
            "| Model | Mode | Images | R²(Mean) | R²(Micro) | SymEq(Mean) | SymEq(Micro) | TypeAcc(Mean) | TypeAcc(Micro) | NRMSE(Mean) | NRMSE(Micro) |",
            "|-------|------|-------:|--------:|---------:|-----------:|------------:|-----------:|------------:|-----------:|------------:|",
        ]

    for item in all_method_agg:
        if len(item) == 5:
            method, d_agg, g_agg, error, excl = item
        else:
            method, d_agg, g_agg, error = item
            excl = []
        if error:
            summary_lines.append(f"| {method} | - | - | - | - | - | - | - | - | - | {error} |")
        else:
            summary_lines.append(
                f"| {method} | Direct | {d_agg['num_images']} | {d_agg['mean_r2']:.4f} | {d_agg['micro_r2']:.4f} | "
                f"{d_agg['mean_sym_equiv_rate']:.0%} | {d_agg['micro_sym_equiv_rate']:.0%} | "
                f"{d_agg['mean_type_accuracy']:.0%} | {d_agg['micro_type_accuracy']:.0%} | "
                f"{d_agg['mean_nrmse']:.4f} | {d_agg['micro_nrmse']:.4f} |"
            )
            summary_lines.append(
                f"| {method} | GeoSym | {g_agg['num_images']} | {g_agg['mean_r2']:.4f} | {g_agg['micro_r2']:.4f} | "
                f"{g_agg['mean_sym_equiv_rate']:.0%} | {g_agg['micro_sym_equiv_rate']:.0%} | "
                f"{g_agg['mean_type_accuracy']:.0%} | {g_agg['micro_type_accuracy']:.0%} | "
                f"{g_agg['mean_nrmse']:.4f} | {g_agg['micro_nrmse']:.4f} |"
            )

    # Append excluded images info
    all_excluded = []
    for item in all_method_agg:
        if len(item) == 5:
            method, _, _, _, excl = item
        else:
            method = item[0]
            excl = []
        for img_name, reason in excl:
            all_excluded.append((method, img_name, reason))

    if all_excluded:
        if is_zh:
            summary_lines.append("\n---\n**排除的图像（识别失败/返回空值）**\n")
            summary_lines.append("| 模型 | 图像 | 原因 |")
            summary_lines.append("|------|------|------|")
        else:
            summary_lines.append("\n---\n**Excluded Images (recognition failed / empty results)**\n")
            summary_lines.append("| Model | Image | Reason |")
            summary_lines.append("|-------|-------|--------|")
        for method, img_name, reason in all_excluded:
            summary_lines.append(f"| {method} | {img_name} | {reason} |")

    summary_text = "\n".join(summary_lines)

    # Build excluded_map for persistence
    excluded_map = {}
    for item in all_method_agg:
        if len(item) == 5:
            method, _, _, _, excl = item
        else:
            method = item[0]
            excl = []
        excluded_map[method] = excl

    pk_data = {
        "images": images,
        "methods": list(selected_methods),
        "method_mode_map": method_mode_map,
        "excluded": excluded_map,
    }

    _save_expr_pk_results(summary_text, pk_data)

    detail_text = _render_expr_pk_detail_page(pk_data, page=0, lang=lang)
    total_pages = max(1, -(-len(images) // EXPR_PK_PAGE_SIZE))

    return summary_text, pk_data, detail_text, f"1 / {total_pages}"
