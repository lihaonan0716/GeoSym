"""Single-image math reasoning pipeline and result formatting."""

import json
import logging
import time
from datetime import datetime, timezone


from pathlib import Path
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

import numpy as np

from app.helpers import (
    _root, _images_dir,
    _get_saved_key,
    _create_llm_adapter, list_dataset_images, load_image_and_gt,
)
from app.reasoning_pipeline import (
    CONDITION_ORDER,
    CONDITION_MAP,
    resolve_image_path,
    extract_geosym_coords,
    compute_visual_facts_from_coords,
    extrema_channel_enabled,
    infer_visual_facts_with_llm,
    run_single_condition,
    load_layer_config,
    load_judge_config,
    ensure_judge_adapter,
    run_judge,
)
from app.reasoning_schema import (
    build_condition_record,
    build_reasoning_record,
    condition_extrema_flag,
    stale_extrema_conditions,
)
from i18n import t, LANG_EN, LANG_ZH
from generator.config_loader import load_config
from evaluator.reasoning.intervention import (
    DEFAULT_INTERVENTION_MODE,
    build_intervention_payload,
    filter_visual_facts_with_llm,
    verify_facts_with_llm,
    get_intervention_profile,
)


# ==================== Math Reasoning — Multi-Step Flow ====================

_L1_PROMPT_PATH = _root / "prompts" / "reasoning_L1_prompt.txt"
_L1_ASSISTED_PROMPT_PATH = _root / "prompts" / "reasoning_L1_assisted_prompt.txt"
_L1_TIERED_PROMPT_PATH = _root / "prompts" / "reasoning_L1_tiered_prompt.txt"
_L2_PROMPT_PATH = _root / "prompts" / "reasoning_L2_prompt.txt"
_L3_PROMPT_PATH = _root / "prompts" / "reasoning_L3_prompt.txt"


def _load_prompt_file(path: Path, fallback: str = "") -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


_resolve_image_path = resolve_image_path


def _inject_prediction_placeholders(prompt: str, gt_data: dict) -> str:
    """Replace {prediction_xs}, {prediction_json_example}, {rate_x1}, {rate_x2} in prompt."""
    pred_targets = gt_data.get("prediction_targets", {})
    rate_info = gt_data.get("rate_of_change", {})

    xs_str = ", ".join(str(x) for x in pred_targets.keys()) if pred_targets else "N/A"
    prompt = prompt.replace("{prediction_xs}", xs_str)

    if pred_targets:
        example_pairs = ", ".join(f'"{k}": <your_estimate>' for k in pred_targets.keys())
        json_example = "{" + example_pairs + "}"
    else:
        json_example = "{}"
    prompt = prompt.replace("{prediction_json_example}", json_example)

    prompt = prompt.replace("{rate_x1}", str(rate_info.get("x1", "?")))
    prompt = prompt.replace("{rate_x2}", str(rate_info.get("x2", "?")))
    return prompt


def _fmt_coords_for_prompt(coords: list) -> str:
    """Format a list of coordinate tuples/dicts into a prompt-friendly string."""
    if not coords:
        return "(no points detected)"
    return ", ".join(
        f"({p[0]}, {p[1]})" if isinstance(p, (list, tuple))
        else f"({p.get('coordinate_x', '?')}, {p.get('coordinate_y', '?')})"
        for p in coords
    )


def _run_reasoning_flow(
    adapter,
    img_path: Path,
    gt_data: dict,
    geosym_coords: list | None = None,
    visual_facts: dict | None = None,
    intervention_mode: str = DEFAULT_INTERVENTION_MODE,
    l1_prompt_path: Path | None = None,
    coord_text_override: str | None = None,
) -> dict:
    """
    Three-step reasoning flow aligned with Curcio layers.

    Args:
        adapter: LLM adapter instance.
        img_path: Path to the image file.
        gt_data: Ground truth dict (for placeholder injection).
        geosym_coords: If provided, injected into L1 prompt.
                       Also used as the final coordinates answer.
        intervention_mode: GeoSym fusion level for structured interventions.
        l1_prompt_path: Custom L1 prompt template (for baselines).
                        When set, overrides the default assisted prompt.
        coord_text_override: Pre-formatted coordinate text for baselines.

    Returns::

        {
            "steps": {
                "L1": {"raw": str, "parsed": dict, "prompt": str},
                "L2": {"raw": str, "parsed": dict, "prompt": str},
                "L3": {"raw": str, "parsed": dict, "prompt": str},
            },
            "final_parsed": dict,   # merged 10-dimension answers
            "geosym_coords": list,  # original coords (or [])
        }
    """
    from evaluator.reasoning.parser import parse_L1_answer, parse_L2_answer, parse_L3_answer

    img_str = str(img_path)
    steps: Dict[str, dict] = {}

    geosym_intervention = None
    facts_for_injection = visual_facts

    # ---- Step 1: L1 Reading ----
    if l1_prompt_path is not None and geosym_coords is not None:
        coord_text = coord_text_override or ", ".join(f"({x}, {y})" for x, y in geosym_coords)
        l1_tpl = _load_prompt_file(l1_prompt_path,
                                    "Detected points: {detected_coords}\nExtract coordinates, intercepts, extrema as JSON.")
        l1_prompt = l1_tpl.replace("{detected_coords}", coord_text)
    elif geosym_coords is not None:
        profile = get_intervention_profile(intervention_mode)
        verify_meta = None

        # Phase 1: optional LLM-based fact verification for GeoSym fusion modes
        if profile.get("llm_verify") and visual_facts is not None:
            facts_for_injection, verify_meta = verify_facts_with_llm(
                adapter,
                img_str,
                geosym_coords,
                visual_facts,
                fact_keys=profile.get("visual_fact_keys"),
            )
        # Backward-compatible support for deprecated llm_filter-style profiles
        elif profile.get("llm_filter") and visual_facts is not None:
            facts_for_injection, verify_meta = filter_visual_facts_with_llm(
                adapter,
                img_str,
                geosym_coords,
                visual_facts,
                mode=intervention_mode,
            )

        # Phase 2: Build intervention payload (flat or tiered)
        geosym_intervention = build_intervention_payload(
            geosym_coords, intervention_mode, visual_facts=facts_for_injection,
        )
        if verify_meta is not None:
            status = verify_meta.get("status", "ok")
            geosym_intervention["verify_metadata"] = verify_meta
            geosym_intervention["summary"].append(
                f"LLM verification status: {status}"
            )
            for note in verify_meta.get("notes", []):
                geosym_intervention["summary"].append(f"Verification: {note}")

        # Phase 3: Build L1 prompt (tiered or flat)
        use_tiered = geosym_intervention.get("use_tiered", False)
        if use_tiered:
            l1_tpl = _load_prompt_file(
                _L1_TIERED_PROMPT_PATH,
                "Tiered evidence:\n{tiered_evidence}\nExtract coordinates, intercepts, extrema as JSON.",
            )
            l1_prompt = l1_tpl.replace("{tiered_evidence}", geosym_intervention["tiered_text"])
            l1_prompt = l1_prompt.replace("{intervention_label}", geosym_intervention["label"])
        else:
            l1_tpl = _load_prompt_file(
                _L1_ASSISTED_PROMPT_PATH,
                "Detected coordinates: {detected_coords}\nStructured facts: {structured_facts}\nExtract coordinates, intercepts, extrema as JSON.",
            )
            l1_prompt = l1_tpl.replace("{detected_coords}", geosym_intervention["coordinate_text"])
            l1_prompt = l1_prompt.replace("{structured_facts}", geosym_intervention["facts_text"])
            l1_prompt = l1_prompt.replace("{intervention_mode}", geosym_intervention["mode"])
            l1_prompt = l1_prompt.replace("{intervention_label}", geosym_intervention["label"])
    else:
        l1_prompt = _load_prompt_file(_L1_PROMPT_PATH,
                                       "Extract coordinates, intercepts, extrema from the graph as JSON.")

    l1_raw = adapter.call_with_prompt(img_str, l1_prompt)
    l1_parsed = parse_L1_answer(l1_raw)
    steps["L1"] = {"raw": l1_raw, "parsed": l1_parsed, "prompt": l1_prompt}

    # ---- Step 2: L2 Analysis ----
    l1_json_str = json.dumps(l1_parsed, ensure_ascii=False)
    l2_tpl = _load_prompt_file(_L2_PROMPT_PATH,
                                "Given L1 results: {L1_results}\nPerform quantitative analysis as JSON.")
    l2_prompt = l2_tpl.replace("{L1_results}", l1_json_str)
    l2_prompt = _inject_prediction_placeholders(l2_prompt, gt_data)

    l2_img = img_str
    l2_raw = adapter.call_with_prompt(l2_img, l2_prompt)
    l2_parsed = parse_L2_answer(l2_raw)
    steps["L2"] = {"raw": l2_raw, "parsed": l2_parsed, "prompt": l2_prompt}

    # ---- Step 3: L3 Reasoning ----
    l2_json_str = json.dumps(l2_parsed, ensure_ascii=False)
    l3_tpl = _load_prompt_file(_L3_PROMPT_PATH,
                                "Given L1: {L1_results}, L2: {L2_results}\nDetermine expression, domain, range as JSON.")
    l3_prompt = l3_tpl.replace("{L1_results}", l1_json_str)
    l3_prompt = l3_prompt.replace("{L2_results}", l2_json_str)

    l3_img = img_str
    l3_raw = adapter.call_with_prompt(l3_img, l3_prompt)
    l3_parsed = parse_L3_answer(l3_raw)
    steps["L3"] = {"raw": l3_raw, "parsed": l3_parsed, "prompt": l3_prompt}

    # ---- Merge final answers (all 10 dimensions) ----
    final_parsed: Dict[str, Any] = {}
    final_parsed.update(l1_parsed)
    final_parsed.update(l2_parsed)
    final_parsed.update(l3_parsed)

    if geosym_coords is not None:
        final_parsed["coordinates"] = geosym_coords

    return {
        "steps": steps,
        "final_parsed": final_parsed,
        "geosym_coords": geosym_coords or [],
        "geosym_intervention": geosym_intervention,
    }


def _fmt_intercepts(ints: dict, is_zh: bool) -> str:
    if not isinstance(ints, dict):
        return "N/A"
    xi = ints.get("x_intercepts", [])
    yi = ints.get("y_intercept")
    parts = []
    if xi:
        parts.append(f"x: {xi}")
    if yi is not None:
        parts.append(f"y: (0, {yi})")
    return ", ".join(parts) if parts else ("无" if is_zh else "None")


def _fmt_extrema(exts: list, is_zh: bool) -> str:
    if not isinstance(exts, list) or not exts:
        return "无" if is_zh else "None"
    return ", ".join(f"({e.get('x')},{e.get('y')}) {e.get('type','')}" for e in exts[:5])


def _fmt_monotonicity(mono: list) -> str:
    if not isinstance(mono, list) or not mono:
        return "N/A"
    parts = []
    for seg in mono[:4]:
        iv = seg.get("interval", [])
        d = seg.get("direction", "?")
        d_sym = "↑" if d == "increasing" else "↓"
        parts.append(f"[{iv[0]},{iv[1]}]{d_sym}" if len(iv) == 2 else str(seg))
    return " ".join(parts)


# ==================== Reasoning Result Formatting ====================
# Modular section builders for structured display in single-image mode.
# Each _section_* function returns a markdown string (may be empty).

_DIM_LABELS_ZH = {
    "expression": "表达式", "domain": "定义域", "range": "值域",
    "intercepts": "截距", "extrema": "极值点",
    "predictions": "函数值预测", "rate_of_change": "变化率",
    "monotonicity": "单调性", "symmetry": "对称性",
    "judge_inference_score": "数学推断质量 (Judge)",
    "inference_explanation": "数学推断说明",
}
_DIM_LABELS_EN = {
    "expression": "Expression", "domain": "Domain", "range": "Range",
    "intercepts": "Intercepts", "extrema": "Extrema",
    "predictions": "Value Prediction", "rate_of_change": "Rate of Change",
    "monotonicity": "Monotonicity", "symmetry": "Symmetry",
    "judge_inference_score": "Inference Quality (Judge)",
    "inference_explanation": "Inference Explanation",
}
def _dim_label(dim: str, is_zh: bool) -> str:
    return (_DIM_LABELS_ZH if is_zh else _DIM_LABELS_EN).get(dim, dim)


def _section_layer_overview(eval_result: dict, layer_config: dict, lang: str) -> str:
    """Per-dimension score overview table (equal weights)."""
    from evaluator.metrics.reasoning import SCORED_DIMENSIONS

    is_zh = lang == LANG_ZH
    dim_scores = eval_result.get("dimension_scores", {})
    total = eval_result.get("total_score", 0)

    header = ("维度" if is_zh else "Dimension",
              "得分 (0~100)" if is_zh else "Score (0~100)")
    rows = [f"| {header[0]} | {header[1]} |", "|---|---:|"]
    for dim in SCORED_DIMENSIONS:
        sc = dim_scores.get(dim, 0)
        rows.append(f"| {_dim_label(dim, is_zh)} | {sc:.1f} |")
    rows.append(f"| **{'等权平均' if is_zh else 'Equal-weight Average'}** | **{total:.1f}** |")
    return "\n".join(rows)


def _section_dimension_detail(eval_result: dict, lang: str) -> str:
    """Per-dimension scoring comparison (prediction vs ground truth)."""
    from evaluator.metrics.reasoning import SCORED_DIMENSIONS

    is_zh = lang == LANG_ZH
    sub_scores = eval_result.get("sub_scores", {})

    def _esc(text: str) -> str:
        """Escape pipe characters so they don't break Markdown tables."""
        return text.replace("|", "\\|")

    h = ("维度" if is_zh else "Dimension",
         "得分" if is_zh else "Score", "预测值" if is_zh else "Predicted",
         "真值" if is_zh else "Ground Truth")
    lines = [f"| {h[0]} | {h[1]} | {h[2]} | {h[3]} |",
             "|---|---:|---|---|"]

    for dim in SCORED_DIMENSIONS:
        s = sub_scores.get(dim, {})
        if s.get("skipped"):
            continue
        score = s.get("score", 0)
        pred = _esc(s.get("pred_display", "N/A")[:60])
        gt_d = _esc(s.get("gt_display", "N/A")[:60])
        lines.append(f"| {_dim_label(dim, is_zh)} | {score:.1f} | {pred} | {gt_d} |")
    return "\n".join(lines)


def _section_parsed_answers(parsed: dict, lang: str) -> str:
    """Display the structured answers extracted from LLM output."""
    is_zh = lang == LANG_ZH
    title = "解析后的结构化答案" if is_zh else "Parsed Structured Answers"

    def _fmt_val(key: str, val) -> str:
        if key == "coordinates":
            if isinstance(val, list):
                pts = [f"({p[0]},{p[1]})" if isinstance(p, (list, tuple))
                       else f"({p.get('coordinate_x','?')},{p.get('coordinate_y','?')})"
                       for p in val[:20]]
                s = ", ".join(pts)
                if len(val) > 20:
                    s += f" +{len(val)-20}"
                return s or ("无" if is_zh else "None")
            return str(val)
        if key == "intercepts":
            return _fmt_intercepts(val, is_zh) if isinstance(val, dict) else str(val)
        if key == "extrema":
            return _fmt_extrema(val, is_zh) if isinstance(val, list) else str(val)
        if key == "monotonicity":
            return _fmt_monotonicity(val) if isinstance(val, list) else str(val)
        if key == "predictions":
            if isinstance(val, dict) and val:
                return ", ".join(f"f({x})={y}" for x, y in val.items())
            return "N/A"
        if key == "rate_of_change":
            return str(val) if val is not None else "N/A"
        if key == "inference_explanation":
            return str(val) if val else ("无" if is_zh else "None")
        return str(val) if val else ("无" if is_zh else "None")

    display_keys = [
        "expression", "domain", "range", "coordinates",
        "intercepts", "extrema", "predictions", "rate_of_change",
        "monotonicity", "symmetry", "inference_explanation",
    ]
    items = [f"- **{_dim_label(k, is_zh)}:** {_fmt_val(k, parsed.get(k, ''))}"
             for k in display_keys]
    body = "\n".join(items)
    return f"<details><summary>{title}</summary>\n\n{body}\n\n</details>"


def _section_prompt(prompt: str, lang: str) -> str:
    """Collapsible section showing the actual prompt sent to the LLM."""
    title = "发送给 LLM 的 Prompt" if lang == LANG_ZH else "Prompt Sent to LLM"
    safe = _escape_for_codeblock(prompt)
    return f"<details><summary>{title}</summary>\n\n```\n{safe}\n```\n\n</details>"


def _section_raw_response(raw: str, lang: str) -> str:
    """Collapsible section showing the LLM's raw output."""
    title = "LLM 原始回复" if lang == LANG_ZH else "Raw LLM Response"
    safe = _escape_for_codeblock(raw)
    return f"<details><summary>{title}</summary>\n\n```\n{safe}\n```\n\n</details>"


def _section_geosym_detail(detected_coords: list, lang: str, intervention: dict | None = None) -> str:
    """Collapsible section showing GeoSym coordinate detection details."""
    is_zh = lang == LANG_ZH
    title = "GeoSym 识别详情" if is_zh else "GeoSym Recognition Details"
    pts = [f"({p[0]},{p[1]})" if isinstance(p, (list, tuple))
           else f"({p.get('coordinate_x','?')},{p.get('coordinate_y','?')})"
           for p in detected_coords[:30]]
    coord_str = ", ".join(pts) if pts else ("无" if is_zh else "None")
    if len(detected_coords) > 30:
        coord_str += f" +{len(detected_coords)-30}"
    body_lines = [
        f"- **{'检测坐标数' if is_zh else 'Detected Points'}:** {len(detected_coords)}",
        f"- **{'坐标列表' if is_zh else 'Coordinate List'}:** {coord_str}",
    ]
    if intervention:
        body_lines.insert(0, f"- **{'干预模式' if is_zh else 'Intervention Mode'}:** {intervention.get('label', intervention.get('mode', 'N/A'))}")
        body_lines.append(f"- **{'说明' if is_zh else 'Description'}:** {intervention.get('description', '')}")
        for line in intervention.get("summary", []):
            body_lines.append(f"- **{'结构化摘要' if is_zh else 'Structured Summary'}:** {line}")
        facts_text = intervention.get("facts_text", "")
        if facts_text and facts_text != "Not provided in this intervention profile.":
            safe_facts = _escape_for_codeblock(facts_text)
            body_lines.append("")
            body_lines.append(f"**{'派生图表事实' if is_zh else 'Derived Chart Facts'}:**")
            body_lines.append(f"```json\n{safe_facts}\n```")
    body = "\n".join(body_lines)
    return f"<details><summary>{title}</summary>\n\n{body}\n\n</details>"


def _section_judge_detail(judge_result: dict | None, lang: str) -> str:
    """Collapsible section showing Judge LLM evaluation for mathematical inference."""
    if not judge_result:
        return ""
    is_zh = lang == LANG_ZH
    title = "Judge 评分详情" if is_zh else "Judge Evaluation Details"

    label_map = {
        "factual_accuracy": "事实准确性" if is_zh else "Factual Accuracy",
        "inference_quality": "推断质量" if is_zh else "Inference Quality",
        "reasoning_coherence": "推理连贯性" if is_zh else "Reasoning Coherence",
        "evidence_support": "证据支撑" if is_zh else "Evidence Support",
        "comments": "评语" if is_zh else "Comments",
    }

    sub_scores = judge_result.get("sub_scores", {})
    items = []
    for k in ("factual_accuracy", "inference_quality", "reasoning_coherence",
              "evidence_support"):
        v = sub_scores.get(k, 0)
        items.append(f"- **{label_map.get(k, k)}:** {v}/5")

    details_str = judge_result.get("details", "")
    if details_str:
        items.append(f"- **{label_map['comments']}:** {details_str}")

    body = "\n".join(items)
    score = judge_result.get("judge_inference_score", judge_result.get("judge_score", 0))
    header = f"**{'Judge 综合分' if is_zh else 'Judge Score'}:** {score:.3f}/100\n\n"

    raw_judge = judge_result.get("raw_judge_response", "")
    raw_section = ""
    if raw_judge:
        raw_title = "Judge 原始回复" if is_zh else "Judge Raw Response"
        safe_judge = _escape_for_codeblock(raw_judge)
        raw_section = f"\n\n<details><summary>{raw_title}</summary>\n\n```\n{safe_judge}\n```\n\n</details>"

    return f"<details><summary>{title}</summary>\n\n{header}{body}{raw_section}\n\n</details>"


def _escape_for_codeblock(text: str) -> str:
    """Escape triple-backtick sequences inside text so they don't break Markdown code fences."""
    return text.replace("```", "` ` `")


def _section_reasoning_steps(pipeline_result: dict, lang: str) -> str:
    """Render collapsible step details for both staged and single-step pipelines."""
    is_zh = lang == LANG_ZH
    step_labels = {
        "L1": "Step 1: Reading the Graph" if not is_zh else "Step 1: 读图",
        "L2": "Step 2: Quantitative Analysis" if not is_zh else "Step 2: 量化分析",
        "L3": "Step 3: Reasoning" if not is_zh else "Step 3: 推理",
        "DirectQA": "Direct QA" if not is_zh else "直接问答",
        "DirectQA-CoT": "Direct QA + CoT" if not is_zh else "直接问答 + CoT",
        "OCRRulesQA": "OCR + Rules QA" if not is_zh else "OCR + 规则问答",
        "OCRRules-CoT": "OCR + Rules + CoT" if not is_zh else "OCR + 规则 + CoT",
        "SelfConsistency": "Self-Consistency Aggregate" if not is_zh else "自一致性聚合",
    }
    parts = []
    steps = pipeline_result.get("steps", {})
    if {"L1", "L2", "L3"} & set(steps):
        ordered_keys = [key for key in ("L1", "L2", "L3") if key in steps]
    else:
        ordered_keys = list(steps.keys())

    for step_key in ordered_keys:
        step = steps.get(step_key, {})
        raw = step.get("raw", "")
        parsed = step.get("parsed", {})
        prompt = step.get("prompt", "")
        label = step_labels.get(step_key, step_key)

        parsed_str = json.dumps(parsed, ensure_ascii=False, indent=2) if parsed else "N/A"
        safe_parsed = _escape_for_codeblock(parsed_str)
        safe_prompt = _escape_for_codeblock(prompt)
        safe_raw = _escape_for_codeblock(raw)

        block = [
            f"<details><summary><b>{label}</b></summary>\n",
            f"**{'解析结果' if is_zh else 'Parsed Output'}:**",
            f"```json\n{safe_parsed}\n```",
            "",
            f"<details><summary>{'Prompt' if not is_zh else '提示词'}</summary>\n",
            f"```\n{safe_prompt}\n```",
            "\n</details>",
            "",
            f"<details><summary>{'Raw Response' if not is_zh else '原始回复'}</summary>\n",
            f"```\n{safe_raw}\n```",
            "\n</details>",
            "\n</details>",
        ]
        parts.append("\n".join(block))

    title = "Reasoning Steps" if not is_zh else "推理步骤"
    inner = "\n\n".join(parts)
    return f"<details><summary><b>{title}</b></summary>\n\n{inner}\n\n</details>"


def _format_path_result(
    *,
    path_label: str,
    eval_result: dict,
    layer_config: dict,
    parsed: dict,
    lang: str,
    pipeline_result: dict | None = None,
    judge_result: dict | None = None,
    detected_coords: list | None = None,
    geosym_intervention: dict | None = None,
) -> str:
    """
    Assemble the full markdown for one evaluation path (A or B).

    Sections are built by independent helpers for easy extension.
    """
    is_zh = lang == LANG_ZH
    total = eval_result.get("total_score", 0)
    sections = [
        f"### {path_label}",
        "",
        f"**{'总分' if is_zh else 'Total Score'}:** {total:.3f}/100",
        "",
        _section_layer_overview(eval_result, layer_config, lang),
        "",
        _section_dimension_detail(eval_result, lang),
        "",
    ]

    if detected_coords is not None:
        sections.append(_section_geosym_detail(detected_coords, lang, geosym_intervention))
        sections.append("")

    if judge_result:
        sections.append(_section_judge_detail(judge_result, lang))
        sections.append("")

    sections.append(_section_parsed_answers(parsed, lang))
    sections.append("")

    if pipeline_result:
        sections.append(_section_reasoning_steps(pipeline_result, lang))
        sections.append("")

    return "\n".join(sections)


_load_layer_config = load_layer_config
_load_judge_config = load_judge_config


def _load_intervention_config() -> dict:
    """Load reasoning intervention defaults for fusion modes and baselines."""
    cfg = load_config()
    default_cfg = {
        "default_path_b_mode": DEFAULT_INTERVENTION_MODE,
        "fusion_modes": ["visual_facts_flat", DEFAULT_INTERVENTION_MODE],
        "baseline_modes": [],
    }
    icfg = cfg.get("reasoning", {}).get("intervention", default_cfg)
    if "batch_modes" in icfg and "fusion_modes" not in icfg:
        icfg["fusion_modes"] = icfg.pop("batch_modes")
    return icfg


_ensure_judge_adapter = ensure_judge_adapter
_run_judge = run_judge


def _condition_labels(lang: str) -> dict:
    is_zh = lang == LANG_ZH
    return {
        "E0": "E0: " + ("DirectQA" if not is_zh else "直接视觉问答"),
        "E1": "E1: " + ("DirectQA-CoT" if not is_zh else "直接视觉问答 + CoT"),
        "E2": "E2: " + ("DirectQA-SelfConsistency" if not is_zh else "直接视觉问答 + 自一致性"),
        "E3": "E3: " + ("OCRRulesQA" if not is_zh else "OCR + 规则问答"),
        "E4": "E4: " + ("OCRRules-CoT" if not is_zh else "OCR + 规则问答 + CoT"),
        "E5": "E5: " + ("FullGeoSym" if not is_zh else "完整 GeoSym"),
        "E5_raw": "E5_raw: " + ("GeoSym-Raw" if not is_zh else "GeoSym 原始坐标"),
    }


def _empty_condition_outputs() -> tuple[str, str, str, str, str, str, str]:
    return ("", "", "", "", "", "", "")


def run_reasoning_comparison(
    image_name: str,
    model_name: str,
    api_key: str,
    judge_model_name: str,
    judge_api_key: str,
    lang: str,
    gemini_enabled: bool = False,
):
    """Run the quick comparison between `E0` and `E5` for one image."""
    from evaluator.metrics.reasoning import evaluate_comprehensive

    is_zh = lang == LANG_ZH
    blank_outputs = _empty_condition_outputs()
    condition_labels = _condition_labels(lang)

    if not image_name:
        msg = "请选择图像" if is_zh else "Please select an image"
        yield (msg, "") + blank_outputs
        return

    img, gt = load_image_and_gt(image_name)
    if img is None or gt is None:
        msg = "图像加载失败" if is_zh else "Failed to load image"
        yield (msg, "") + blank_outputs
        return

    question = gt.get("question", {})
    gt_data = question.get("ground_truth", {})
    if not gt_data:
        msg = "该图像无综合题目数据" if is_zh else "This image has no comprehensive question data"
        yield (msg, "") + blank_outputs
        return

    gt_coords = gt_data.get("coordinates", [])
    gt_coords_str = ", ".join(f"({c[0]},{c[1]})" for c in gt_coords[:15])
    if len(gt_coords) > 15:
        gt_coords_str += f" +{len(gt_coords)-15}"
    pred_tgt = gt_data.get("prediction_targets", {})
    pred_tgt_str = ", ".join(f"f({x})={y}" for x, y in pred_tgt.items()) if pred_tgt else ("无" if is_zh else "None")
    roc_info = gt_data.get("rate_of_change", {})
    roc_str = f"{roc_info.get('gt_value', 'N/A')} (x∈[{roc_info.get('x1', '?')},{roc_info.get('x2', '?')}])" if roc_info else ("无" if is_zh else "None")
    gt_lines = [
        f"### {'标准答案' if is_zh else 'Ground Truth'}",
        f"- **{'表达式' if is_zh else 'Expression'}:** {gt_data.get('expression', '')}",
        f"- **{'定义域' if is_zh else 'Domain'}:** {gt_data.get('domain', '')}",
        f"- **{'值域' if is_zh else 'Range'}:** {gt_data.get('range', '')}",
        f"- **{'坐标' if is_zh else 'Coordinates'}:** {gt_coords_str or ('无' if is_zh else 'None')}",
        f"- **{'截距' if is_zh else 'Intercepts'}:** {_fmt_intercepts(gt_data.get('intercepts', {}), is_zh)}",
        f"- **{'极值点' if is_zh else 'Extrema'}:** {_fmt_extrema(gt_data.get('extrema', []), is_zh)}",
        f"- **{'函数值预测' if is_zh else 'Prediction Targets'}:** {pred_tgt_str}",
        f"- **{'变化率' if is_zh else 'Rate of Change'}:** {roc_str}",
        f"- **{'单调性' if is_zh else 'Monotonicity'}:** {_fmt_monotonicity(gt_data.get('monotonicity', []))}",
        f"- **{'对称性' if is_zh else 'Symmetry'}:** {gt_data.get('symmetry', 'N/A')}",
    ]
    gt_md = "\n\n".join(gt_lines)

    key = api_key.strip() if api_key else ""
    if not key:
        key = _get_saved_key(model_name)
    if not key:
        msg = "请输入 API Key" if is_zh else "Please enter API Key"
        yield (gt_md, msg) + blank_outputs
        return

    img_path = _resolve_image_path(image_name)
    if img_path is None:
        msg = "图像文件未找到" if is_zh else "Image file not found"
        yield (gt_md, msg) + blank_outputs
        return

    adapter = _create_llm_adapter(model_name, key)
    if adapter is None:
        msg = f"不支持的模型: {model_name}" if is_zh else f"Unsupported model: {model_name}"
        yield (gt_md, msg) + blank_outputs
        return

    layer_config = _load_layer_config()
    judge_adapter = _ensure_judge_adapter(judge_model_name, judge_api_key, model_name, key)
    progress = "🔄 " + ("正在运行 E0 与 E5…" if is_zh else "Running E0 and E5…")
    yield (gt_md, progress) + blank_outputs

    def _run_e0():
        pipeline = run_single_condition(
            adapter, img_path, gt_data, CONDITION_MAP["E0"],
        )
        parsed = pipeline["final_parsed"]
        judge = _run_judge(judge_adapter, img_path, gt_data, pipeline) if judge_adapter else None
        return pipeline, parsed, judge, None

    def _run_e5():
        detected, vf, success, reason = extract_geosym_coords(
            model_name, key, img_path, gt, gemini_enabled=gemini_enabled,
        )
        if not success:
            raise RuntimeError(f"GeoSym extraction failed: {reason or 'unknown reason'}")
        pipeline = run_single_condition(
            adapter, img_path, gt_data, CONDITION_MAP["E5"],
            geosym_coords=detected or [],
            visual_facts=vf,
        )
        parsed = pipeline["final_parsed"]
        judge = _run_judge(judge_adapter, img_path, gt_data, pipeline) if judge_adapter else None
        return pipeline, parsed, judge, detected or []

    result_md = {key: "" for key in CONDITION_ORDER}
    for cond, run_fn in [("E0", _run_e0), ("E5", _run_e5)]:
        try:
            pipeline, parsed, judge, detected_coords = run_fn()
            ev = evaluate_comprehensive(parsed, gt_data, layer_config, judge)
            result_md[cond] = _format_path_result(
                path_label=condition_labels[cond],
                eval_result=ev,
                layer_config=layer_config,
                parsed=parsed,
                lang=lang,
                pipeline_result=pipeline,
                judge_result=judge,
                detected_coords=detected_coords,
                geosym_intervention=pipeline.get("geosym_intervention"),
            )
        except Exception as exc:
            result_md[cond] = f"**{cond} Error:** {exc}"

        progress = "✅ " + cond + " " + ("完成" if is_zh else "done")
        if cond == "E0":
            progress += " · 🔄 E5…"

        yield (
            gt_md,
            progress,
            result_md["E0"],
            result_md["E1"],
            result_md["E2"],
            result_md["E3"],
            result_md["E4"],
            result_md["E5"],
            result_md["E5_raw"],
        )

    done = "✅ " + ("全部完成" if is_zh else "All done")
    yield (
        gt_md,
        done,
        result_md["E0"],
        result_md["E1"],
        result_md["E2"],
        result_md["E3"],
        result_md["E4"],
        result_md["E5"],
        result_md["E5_raw"],
    )


def _build_progress_bar(
    condition_order: list,
    condition_labels: dict,
    current_step: str,
    completed: set,
    skipped: set,
    failed: set,
    is_zh: bool,
) -> str:
    """Build a live progress indicator showing each condition's status."""
    icons = {"done": "✅", "skip": "⏭️", "run": "🔄", "wait": "⏳", "fail": "❌"}
    lines = [
        f"### {'执行进度' if is_zh else 'Execution Progress'}",
        "",
    ]
    for ck in condition_order:
        label = condition_labels.get(ck, ck)
        if ck in failed:
            icon, status = icons["fail"], ("失败" if is_zh else "Failed")
        elif ck in completed:
            icon, status = icons["done"], ("完成" if is_zh else "Done")
        elif ck in skipped:
            icon, status = icons["skip"], ("缓存" if is_zh else "Cached")
        elif ck == current_step:
            icon, status = icons["run"], ("**运行中…**" if is_zh else "**Running…**")
        else:
            icon, status = icons["wait"], ("等待中" if is_zh else "Pending")
        lines.append(f"{icon} **{ck}** {label.split(': ', 1)[-1]} — {status}")
    lines.append("")
    lines.append(f"> {'当前步骤' if is_zh else 'Current'}: **{current_step}**")
    return "\n\n".join(lines)


def run_reasoning_all_conditions(
    image_name: str,
    model_name: str,
    api_key: str,
    judge_model_name: str,
    judge_api_key: str,
    lang: str,
    gemini_enabled: bool = False,
    force_rerun: bool = False,
    target_conditions: list[str] | None = None,
):
    """Run the unified six-condition matrix (E0-E5) on a single image."""
    from evaluator.metrics.reasoning import evaluate_comprehensive
    from app.reasoning_cache import (
        load_reasoning_cache, save_reasoning_cache,
    )

    is_zh = lang == LANG_ZH
    condition_order = list(CONDITION_ORDER)
    condition_labels = _condition_labels(lang)
    target_set = {ck for ck in (target_conditions or []) if ck in condition_order}
    requested_conditions = [ck for ck in condition_order if not target_set or ck in target_set]
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_started_perf = time.perf_counter()

    if target_conditions is not None and not requested_conditions:
        msg = "请选择有效实验条件" if is_zh else "Please select a valid condition"
        yield (msg,) + ("",) * 8
        return

    if not image_name:
        msg = "请选择图像" if is_zh else "Please select an image"
        yield (msg,) + ("",) * 8
        return

    img, gt = load_image_and_gt(image_name)
    if img is None or gt is None:
        msg = "图像加载失败" if is_zh else "Failed to load image"
        yield (msg,) + ("",) * 8
        return

    question = gt.get("question", {})
    gt_data = question.get("ground_truth", {})
    if not gt_data:
        msg = "该图像无综合题目数据" if is_zh else "This image has no comprehensive question data"
        yield (msg,) + ("",) * 8
        return

    gt_coords = gt_data.get("coordinates", [])
    gt_coords_str = ", ".join(f"({c[0]},{c[1]})" for c in gt_coords[:15])
    if len(gt_coords) > 15:
        gt_coords_str += f" +{len(gt_coords)-15}"
    pred_tgt = gt_data.get("prediction_targets", {})
    pred_tgt_str = ", ".join(f"f({x})={y}" for x, y in pred_tgt.items()) if pred_tgt else ("无" if is_zh else "None")
    roc_info = gt_data.get("rate_of_change", {})
    roc_str = f"{roc_info.get('gt_value', 'N/A')} (x∈[{roc_info.get('x1', '?')},{roc_info.get('x2', '?')}])" if roc_info else ("无" if is_zh else "None")
    gt_lines = [
        f"### {'标准答案' if is_zh else 'Ground Truth'}",
        f"- **{'表达式' if is_zh else 'Expression'}:** {gt_data.get('expression', '')}",
        f"- **{'定义域' if is_zh else 'Domain'}:** {gt_data.get('domain', '')}",
        f"- **{'值域' if is_zh else 'Range'}:** {gt_data.get('range', '')}",
        f"- **{'坐标' if is_zh else 'Coordinates'}:** {gt_coords_str or ('无' if is_zh else 'None')}",
        f"- **{'截距' if is_zh else 'Intercepts'}:** {_fmt_intercepts(gt_data.get('intercepts', {}), is_zh)}",
        f"- **{'极值点' if is_zh else 'Extrema'}:** {_fmt_extrema(gt_data.get('extrema', []), is_zh)}",
        f"- **{'函数值预测' if is_zh else 'Prediction Targets'}:** {pred_tgt_str}",
        f"- **{'变化率' if is_zh else 'Rate of Change'}:** {roc_str}",
        f"- **{'单调性' if is_zh else 'Monotonicity'}:** {_fmt_monotonicity(gt_data.get('monotonicity', []))}",
        f"- **{'对称性' if is_zh else 'Symmetry'}:** {gt_data.get('symmetry', 'N/A')}",
    ]
    gt_md = "\n\n".join(gt_lines)

    key = api_key.strip() if api_key else ""
    if not key:
        key = _get_saved_key(model_name)
    if not key:
        msg = "请输入 API Key" if is_zh else "Please enter API Key"
        yield (gt_md, msg) + ("",) * 7
        return

    img_path = _resolve_image_path(image_name)
    if img_path is None:
        msg = "图像文件未找到" if is_zh else "Image file not found"
        yield (gt_md, msg) + ("",) * 7
        return

    adapter = _create_llm_adapter(model_name, key)
    if adapter is None:
        msg = f"不支持的模型: {model_name}" if is_zh else f"Unsupported model: {model_name}"
        yield (gt_md, msg) + ("",) * 7
        return

    layer_config = load_layer_config()
    judge_adapter = ensure_judge_adapter(judge_model_name, judge_api_key, model_name, key)

    from evaluator.baselines.ocr_rule_baseline import OCRRuleBaseline

    ocr_bl = OCRRuleBaseline()

    from evaluator.metrics.reasoning import SCORED_DIMENSIONS
    _dim_keys = list(SCORED_DIMENSIONS)

    # ---- Load cache ----
    cached = None
    skip_conds: set = set()
    loaded_from_cache: set = set()
    current_extrema_flag = extrema_channel_enabled()
    # Conditions cached under the opposite extrema-channel state.  Held apart from
    # `skip_conds` so a flag flip forces a re-run of exactly the affected
    # conditions instead of discarding the whole file.
    stale_conds: set = set()
    should_load_cache = (not force_rerun) or bool(target_set)
    if should_load_cache:
        cached = load_reasoning_cache(image_name, model_name)
        if cached is not None:
            cached_judge = cached.get("judge_model", "")
            current_judge = judge_model_name or model_name or ""
            if cached_judge != current_judge:
                logger.info(
                    "Judge model changed (%s → %s) — invalidating cached results",
                    cached_judge, current_judge,
                )
                cached = None
            else:
                stale_conds = stale_extrema_conditions(cached, current_extrema_flag)
                if stale_conds:
                    logger.info(
                        "Extrema channel state changed (now %s) — invalidating cached "
                        "conditions: %s",
                        current_extrema_flag, ", ".join(sorted(stale_conds)),
                    )
                cached_conds = cached.get("conditions", {})
                if not force_rerun:
                    for ck in requested_conditions:
                        if ck in stale_conds:
                            continue
                        if cached_conds.get(ck, {}).get("status") == "success":
                            skip_conds.add(ck)
                    if skip_conds:
                        logger.info("Cache hit — skipping conditions: %s", ", ".join(sorted(skip_conds)))

    results_md: dict = {}
    results_eval: dict = {}
    results_pipeline: dict = {}
    results_parsed: dict = {}
    results_judge: dict = {}
    results_errors: dict = {}
    condition_elapsed_ms: dict = {}
    condition_retry_counts: dict = {}
    condition_cached_flags: dict = {}
    # Per-condition extrema-channel state.  Cached conditions keep whatever they
    # were produced under; freshly run ones take this process's setting.
    condition_extrema_flags: dict = {}
    judge_failed_conds: set = set()
    geosym_failed_conds: set = set()
    completed_conds: set = set()

    # Pre-populate cached condition records, except force-rerun targets.
    if cached is not None:
        cached_conds = cached.get("conditions", {})
        for ck, cond_data in cached_conds.items():
            if ck not in condition_order:
                continue
            if force_rerun and not target_set:
                continue
            if ck in target_set:
                continue
            # Stale-flag conditions are deliberately still carried over here.
            # Dropping them would throw away a paid result whenever this run does
            # not target them, and it is unnecessary: each condition records the
            # flag it was actually produced under, so a mixed record stays honest.
            # What the flag change does force is a re-run — see `skip_conds`.
            results_eval[ck] = cond_data.get("eval")
            results_md[ck] = cond_data.get("md", "")
            results_pipeline[ck] = cond_data.get("pipeline")
            results_parsed[ck] = cond_data.get("parsed")
            results_judge[ck] = cond_data.get("judge")
            results_errors[ck] = cond_data.get("error")
            condition_elapsed_ms[ck] = cond_data.get("elapsed_ms", 0)
            condition_retry_counts[ck] = cond_data.get("retry_count", 0)
            condition_cached_flags[ck] = True
            condition_extrema_flags[ck] = condition_extrema_flag(cond_data)
            loaded_from_cache.add(ck)
            if cond_data.get("status") == "success":
                completed_conds.add(ck)

    def _snapshot():
        """Build a yield tuple from current state."""
        return (
            gt_md,
            progress_md,
            results_md.get("E0", ""),
            results_md.get("E1", ""),
            results_md.get("E2", ""),
            results_md.get("E3", ""),
            results_md.get("E4", ""),
            results_md.get("E5", ""),
            results_md.get("E5_raw", ""),
        )

    # ---- Yield initial progress ----
    progress_md = _build_progress_bar(
        condition_order,
        condition_labels,
        current_step="INIT",
        completed=completed_conds,
        skipped=skip_conds,
        failed=geosym_failed_conds | judge_failed_conds,
        is_zh=is_zh,
    ) + "\n\n> " + ("正在初始化…" if is_zh else "Initializing…")
    yield _snapshot()

    cached_geosym = cached.get("geosym_artifacts", {}) if cached else {}
    geosym_coords: list = cached_geosym.get("coords", [])
    geosym_success = cached_geosym.get("success", False)
    geosym_failure_reason = cached_geosym.get("error")
    geosym_attempted = geosym_success
    # Always recompute visual facts from coords using the current algorithm,
    # so that code improvements to fact derivation take effect immediately.
    if geosym_coords:
        geosym_vf: dict | None = compute_visual_facts_from_coords(geosym_coords)
    else:
        geosym_vf: dict | None = cached_geosym.get("visual_facts")

    def _run_pipeline(cond_key: str, **kwargs):
        pipeline = run_single_condition(
            adapter,
            img_path,
            gt_data,
            CONDITION_MAP[cond_key],
            **kwargs,
        )
        parsed = pipeline["final_parsed"]
        judge_result = None
        if judge_adapter:
            judge_result = run_judge(judge_adapter, img_path, gt_data, pipeline)
        return pipeline, parsed, judge_result

    def _ensure_geosym():
        nonlocal geosym_coords, geosym_vf, geosym_success, geosym_failure_reason, geosym_attempted
        if geosym_attempted:
            return geosym_success
        geosym_attempted = True
        geosym_coords, _, geosym_success, geosym_failure_reason = extract_geosym_coords(
            model_name,
            key,
            img_path,
            gt,
            gemini_enabled=gemini_enabled,
        )
        if geosym_success and geosym_coords:
            # Use computation-grounded fact derivation (no LLM hallucination risk)
            geosym_vf = compute_visual_facts_from_coords(geosym_coords)
            if geosym_vf is None:
                # Fallback to LLM-based inference if computation fails
                adapter_for_vf = _create_llm_adapter(model_name, key)
                if adapter_for_vf is not None:
                    geosym_vf = infer_visual_facts_with_llm(adapter_for_vf, img_path, geosym_coords)
        return geosym_success

    cond_statuses: dict = {}
    for ck in condition_order:
        if cached and ck in cached.get("conditions", {}):
            cond_statuses[ck] = cached["conditions"][ck].get("status", "unknown")

    for cond_key in requested_conditions:
        if cond_key in skip_conds:
            cond_statuses[cond_key] = "success"
            continue

        # About to run for real, so any prepopulated cache entry for this
        # condition is superseded.  Matters for stale-flag conditions, which are
        # prepopulated *and* re-run — without this they would still be reported
        # under `summary.cached_conditions`.
        loaded_from_cache.discard(cond_key)

        progress_md = _build_progress_bar(
            condition_order,
            condition_labels,
            current_step=cond_key,
            completed=completed_conds,
            skipped=skip_conds,
            failed=geosym_failed_conds | judge_failed_conds,
            is_zh=is_zh,
        )
        yield _snapshot()

        try:
            started_cond_perf = time.perf_counter()
            kwargs = {}
            if cond_key in {"E3", "E4"}:
                kwargs["ocr_baseline"] = ocr_bl
            if cond_key == "E5":
                if not _ensure_geosym():
                    raise RuntimeError(f"GeoSym extraction failed: {geosym_failure_reason or 'unknown reason'}")
                kwargs["geosym_coords"] = geosym_coords
                kwargs["visual_facts"] = geosym_vf
            if cond_key == "E5_raw":
                if not _ensure_geosym():
                    raise RuntimeError(f"GeoSym extraction failed: {geosym_failure_reason or 'unknown reason'}")
                kwargs["geosym_coords"] = geosym_coords

            pipeline, parsed, j_result = _run_pipeline(cond_key, **kwargs)
            ev = evaluate_comprehensive(parsed, gt_data, layer_config, j_result)
            results_eval[cond_key] = ev
            results_pipeline[cond_key] = pipeline
            results_parsed[cond_key] = parsed
            results_judge[cond_key] = j_result
            results_errors[cond_key] = None
            condition_elapsed_ms[cond_key] = round((time.perf_counter() - started_cond_perf) * 1000, 2)
            condition_retry_counts[cond_key] = 0
            condition_cached_flags[cond_key] = False
            condition_extrema_flags[cond_key] = current_extrema_flag
            results_md[cond_key] = _format_path_result(
                path_label=condition_labels[cond_key],
                eval_result=ev,
                layer_config=layer_config,
                parsed=parsed,
                lang=lang,
                pipeline_result=pipeline,
                judge_result=j_result,
                detected_coords=geosym_coords if cond_key == "E5" else None,
                geosym_intervention=pipeline.get("geosym_intervention"),
            )
            cond_statuses[cond_key] = "success"
            completed_conds.add(cond_key)
        except Exception as exc:
            condition_elapsed_ms[cond_key] = round((time.perf_counter() - started_cond_perf) * 1000, 2)
            condition_retry_counts[cond_key] = 0
            condition_cached_flags[cond_key] = False
            condition_extrema_flags[cond_key] = current_extrema_flag
            results_errors[cond_key] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
            results_pipeline[cond_key] = None
            results_parsed[cond_key] = None
            results_judge[cond_key] = None
            if cond_key == "E5":
                geosym_failed_conds.add(cond_key)
                results_md[cond_key] = (
                    f"**{condition_labels[cond_key]}**\n\n"
                    f"❌ {'GeoSym 提取失败，该条件数据无效' if is_zh else 'GeoSym extraction failed — condition invalid'}"
                )
                cond_statuses[cond_key] = "geosym_failed"
            else:
                logger.warning("Condition %s failed: %s", cond_key, exc)
                results_md[cond_key] = f"**{cond_key} Error:** {exc}"
                cond_statuses[cond_key] = "pipeline_error"
            results_eval[cond_key] = None
        yield _snapshot()

    # ---- Summary Table ----
    dim_header = "维度" if is_zh else "Dimension"
    tbl = [
        f"### {'六条件评分对比' if is_zh else 'Six-Condition Score Comparison'}",
        "",
        f"| {dim_header} | " + " | ".join(condition_order) + " |",
        "|---|" + "---|" * len(condition_order),
    ]
    for dk in _dim_keys:
        vals = []
        for ck in condition_order:
            ev = results_eval.get(ck)
            if ev:
                sc = ev.get("sub_scores", {}).get(dk, {}).get("score", 0)
                vals.append(f"{sc:.3f}")
            else:
                vals.append("—")
        tbl.append(f"| {_dim_label(dk, is_zh)} | " + " | ".join(vals) + " |")
    total_vals = []
    for ck in condition_order:
        ev = results_eval.get(ck)
        if ev:
            total_vals.append(f"**{ev.get('total_score', 0):.3f}**")
        else:
            total_vals.append("—")
    tbl.append(f"| **Total** | " + " | ".join(total_vals) + " |")

    if results_eval.get("E0") and results_eval.get("E5"):
        base_total = results_eval["E0"].get("total_score", 0)
        full_total = results_eval["E5"].get("total_score", 0)
        delta = full_total - base_total
        tbl.append("")
        tbl.append(f"**Δ(E5 − E0) = {delta:+.3f}**")

    # Status line: show which conditions were cached vs freshly run
    if loaded_from_cache:
        cached_list = ", ".join(sorted(loaded_from_cache))
        fresh_list = ", ".join(sorted(set(requested_conditions) - skip_conds)) or ("无" if is_zh else "None")
        tbl.append("")
        if is_zh:
            tbl.append(f"> ℹ️ 已缓存: {cached_list} | 本次运行: {fresh_list}")
        else:
            tbl.append(f"> ℹ️ Cached: {cached_list} | Ran this time: {fresh_list}")

    if target_set:
        tbl.append("")
        selected_text = ", ".join(requested_conditions)
        if is_zh:
            tbl.append(f"> ℹ️ 本次仅重跑所选条件: {selected_text}")
        else:
            tbl.append(f"> ℹ️ Re-ran selected conditions only: {selected_text}")

    all_failed = judge_failed_conds | geosym_failed_conds
    if all_failed:
        tbl.append("")
        parts = []
        if judge_failed_conds:
            jf = ", ".join(sorted(judge_failed_conds))
            if is_zh:
                parts.append(f"条件 **{jf}** 的 Judge 评估失败（已重试 5 次）")
            else:
                parts.append(f"Judge evaluation failed for **{jf}** (after 5 retries)")
        if geosym_failed_conds:
            gf = ", ".join(sorted(geosym_failed_conds))
            if is_zh:
                parts.append(f"条件 **{gf}** 的 GeoSym 提取失败")
            else:
                parts.append(f"GeoSym extraction failed for **{gf}**")
        separator = "；" if is_zh else "; "
        suffix = "，标记为无效数据（—）。可点击「重跑失败条件」补全。" if is_zh else ". Marked as invalid (—). Click 'Re-run Failed' to retry."
        tbl.append(f"> ⚠️ {separator.join(parts)}{suffix}")

    summary_md = "\n".join(tbl)

    cache_conditions = {}
    for ck in condition_order:
        cache_conditions[ck] = build_condition_record(
            condition_id=ck,
            label=condition_labels[ck],
            status=cond_statuses.get(ck, "unknown"),
            cached=condition_cached_flags.get(ck, False),
            elapsed_ms=condition_elapsed_ms.get(ck, 0),
            retry_count=condition_retry_counts.get(ck, 0),
            error=results_errors.get(ck),
            eval_result=results_eval.get(ck),
            parsed=results_parsed.get(ck),
            judge=results_judge.get(ck),
            pipeline=results_pipeline.get(ck),
            md=results_md.get(ck, ""),
            extrema_channel_enabled=condition_extrema_flags.get(ck, current_extrema_flag),
        )

    cache_record = build_reasoning_record(
        image_name=image_name,
        image_path=str(img_path),
        reasoning_model=model_name,
        judge_model=judge_model_name or model_name or "",
        started_at=run_started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        duration_ms=round((time.perf_counter() - run_started_perf) * 1000, 2),
        requested_conditions=requested_conditions,
        run_settings={
            "gemini_enabled": gemini_enabled,
            "force_rerun": force_rerun,
            "target_conditions": requested_conditions,
            # Resolved boolean, not the raw env-var string: without it a stored
            # record cannot be told apart from one produced with the extrema
            # channel in the opposite state.  This is the *process* setting; a
            # condition reused from an earlier run keeps its own flag in
            # `conditions[ck].extrema_channel_enabled`, which is the field to
            # trust when the two disagree.
            "extrema_channel_enabled": current_extrema_flag,
        },
        ground_truth_snapshot=gt_data,
        score_table_markdown=summary_md,
        cached_conditions=sorted(loaded_from_cache),
        geosym_artifacts={
            "success": geosym_success,
            "coords": geosym_coords,
            "visual_facts": geosym_vf,
        },
        conditions=cache_conditions,
    )
    try:
        save_reasoning_cache(cache_record)
    except Exception as exc:
        logger.warning("Failed to save reasoning cache: %s", exc)

    # Final yield with the complete summary table replacing the progress bar
    yield (
        gt_md,
        summary_md,
        results_md.get("E0", ""),
        results_md.get("E1", ""),
        results_md.get("E2", ""),
        results_md.get("E3", ""),
        results_md.get("E4", ""),
        results_md.get("E5", ""),
        results_md.get("E5_raw", ""),
    )
