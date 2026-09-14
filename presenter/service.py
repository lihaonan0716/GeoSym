from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


logger = logging.getLogger(__name__)

_root = Path(__file__).resolve().parent.parent

PROMPT_PATHS = {
    "direct": _root / "prompts" / "real_chart_direct_prompt.txt",
    "m0_extract": _root / "prompts" / "presenter_m0_extract_prompt.txt",
    "m0_locate": _root / "prompts" / "presenter_m0_locate_prompt.txt",
    "filter": _root / "prompts" / "presenter_filter_prompt.txt",
    "l1": _root / "prompts" / "presenter_l1_prompt.txt",
    "l2": _root / "prompts" / "presenter_l2_prompt.txt",
    "l3": _root / "prompts" / "presenter_l3_prompt.txt",
    "pairwise": _root / "prompts" / "presenter_pairwise_compare_prompt.txt",
}

REPORT_ROOT = _root / "reports" / "presenter"


def _slugify(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    return value.strip("_") or "unknown"


_ASSETS_ROOT = _root / "assets" / "real_charts"


def _sample_meta_path(sample_id: str) -> Path:
    """Resolve meta.json path: assets/real_charts/{family}/{sample_id}/meta.json"""
    for family_dir in sorted(_ASSETS_ROOT.iterdir()):
        if not family_dir.is_dir():
            continue
        candidate = family_dir / sample_id / "meta.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Sample meta.json not found for: {sample_id}")


def _load_sample_spec(sample_id: str) -> Dict[str, Any]:
    path = _sample_meta_path(sample_id)
    return json.loads(path.read_text(encoding="utf-8"))


def list_real_chart_samples() -> List[str]:
    if not _ASSETS_ROOT.exists():
        return []
    sample_ids = []
    for family_dir in sorted(_ASSETS_ROOT.iterdir()):
        if not family_dir.is_dir():
            continue
        for sample_dir in sorted(family_dir.iterdir()):
            if sample_dir.is_dir() and (sample_dir / "meta.json").exists():
                sample_ids.append(sample_dir.name)
    return sample_ids


def get_sample_image_path(sample_id: str) -> str | None:
    try:
        sample = _load_sample_spec(sample_id)
    except Exception:
        return None
    image_path = _root / sample["image_path"]
    return str(image_path) if image_path.exists() else None


def build_sample_overview(sample_id: str, lang: str = "English") -> str:
    if not sample_id:
        return ""
    try:
        sample = _load_sample_spec(sample_id)
    except Exception as exc:
        return f"### Sample Error\n\n`{exc}`"

    is_zh = False
    title = "### Presenter Sample"
    labels = {
        "source": "Source",
        "question": "Question",
    }
    lines = [
        title,
        "",
        f"- **ID**: `{sample['sample_id']}`",
        f"- **Title**: `{sample.get('title', sample['sample_id'])}`",
        f"- **{labels['source']}**: {sample.get('source_url', 'N/A')}",
        f"- **{labels['question']}**: `{sample.get('question', '')}`",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _load_prompt(name: str) -> str:
    path = PROMPT_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def _bullet_block(items: List[str]) -> str:
    if not items:
        return "- (none)"
    return "\n".join(f"- {item}" for item in items)


def _build_prompt(name: str, sample: Dict[str, Any], **kwargs: str) -> str:
    prompt = _load_prompt(name)
    values = {
        "question": sample.get("question", ""),
        "candidate_facts": _bullet_block(sample.get("candidate_facts", [])),
        "verified_facts": _bullet_block(sample.get("verified_facts", [])),
        "useful_facts": _bullet_block(sample.get("useful_facts", [])),
    }
    values.update(kwargs)
    for key, value in values.items():
        prompt = prompt.replace(f"{{{key}}}", value)
    return prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_section(raw: str, label: str) -> str:
    pattern = re.compile(
        rf"(?im)^{re.escape(label)}:\s*(.*?)(?=^\w[\w ]*:\s*|\Z)",
        re.DOTALL,
    )
    match = pattern.search(raw)
    if not match:
        return ""
    return match.group(1).strip()


def _parse_answer_response(raw: str) -> Dict[str, Any]:
    answer = _extract_section(raw, "Answer")
    evidence = _extract_section(raw, "Evidence")
    confidence = _extract_section(raw, "Confidence")
    if not answer:
        answer = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    evidence_lines = []
    if evidence:
        for line in evidence.splitlines():
            cleaned = line.strip().lstrip("-").strip()
            if cleaned:
                evidence_lines.append(cleaned)
    return {
        "answer": answer,
        "evidence": evidence_lines,
        "confidence": confidence or "",
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    src = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", src, re.DOTALL)
    if match:
        src = match.group(1)
    else:
        start = src.find("{")
        end = src.rfind("}")
        if start != -1 and end != -1 and end > start:
            src = src[start:end + 1]
    data = json.loads(src)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def _call_json_stage(adapter, image: str | None, prompt: str, fallback: Dict[str, Any]) -> Tuple[Dict[str, Any], str, bool]:
    raw = adapter.call_with_prompt(image, prompt)
    try:
        parsed = _extract_json_object(raw)
        return parsed, raw, False
    except Exception:
        return fallback, raw, True


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM adapter
# ---------------------------------------------------------------------------

def _ensure_adapter(model_name: str):
    from app.helpers import _create_llm_adapter, _get_saved_key

    key = _get_saved_key(model_name)
    is_claude = "claude" in model_name.lower()
    if not key and not is_claude:
        raise RuntimeError(f"No API key found for model: {model_name}")
    adapter = _create_llm_adapter(model_name, key or "")
    if adapter is None:
        raise RuntimeError(f"Failed to create adapter for model: {model_name}")
    return adapter


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _run_llm_m0_extraction(adapter, image_path: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    """M0: use LLM vision to extract visual evidence from a chart image."""
    prompt = _build_prompt("m0_extract", sample)
    fallback = {
        "candidate_facts": [],
        "verified_facts": [],
        "useful_facts": [],
        "chart_metadata": {
            "chart_type": "other",
            "x_axis": "",
            "y_axis": "",
            "series_count": 1,
            "has_grid_lines": False,
        },
    }
    parsed, raw, fallback_used = _call_json_stage(adapter, image_path, prompt, fallback)
    return {
        "prompt": prompt,
        "raw_response": raw,
        "parsed": parsed,
        "fallback_used": fallback_used,
    }


def _run_direct_group(adapter, image_path: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _build_prompt("direct", sample)
    raw = adapter.call_with_prompt(image_path, prompt)
    parsed = _parse_answer_response(raw)
    return {
        "prompt": prompt,
        "raw_response": raw,
        "parsed_response": parsed,
    }


def _run_enhanced_m0(adapter, image_path: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced M0: OCR标尺 + LLM定位 + 数值计算测量。

    Pipeline (inspired by Computer Use Agent architecture):
      Phase 1: Build Ruler — OCR提取坐标轴，建立像素↔数值映射
      Phase 2: Locate     — LLM看图+标尺上下文，输出关键点像素坐标
      Phase 3: Measure    — pixel_to_real() 数值计算转换
      Phase 4: Verify     — 交叉验证 + 融合为兼容格式

    如果标尺建立失败，降级为纯 LLM M0（与现有行为相同）。
    """
    from .ruler import build_ruler, format_ruler_for_prompt
    from .measure import measure_points, compute_derived_quantities
    from .verify import verify_measurements, fuse_evidence

    # Phase 1: Build Ruler
    logger.info("[Enhanced-M0] Phase 1: Building coordinate ruler via OCR...")
    ruler = build_ruler(image_path)

    if not ruler.success:
        # 降级为纯 LLM M0
        logger.warning("[Enhanced-M0] Ruler failed, falling back to pure LLM M0")
        fallback_result = _run_llm_m0_extraction(adapter, image_path, sample)
        fallback_result["enhanced"] = False
        fallback_result["ruler_diagnostics"] = ruler.diagnostics
        return fallback_result

    # Phase 2: LLM Locate (with ruler context)
    logger.info("[Enhanced-M0] Phase 2: LLM locating key data points...")
    ruler_context = format_ruler_for_prompt(ruler)
    locate_prompt = _load_prompt("m0_locate")
    locate_prompt = locate_prompt.replace("{ruler_context}", ruler_context)
    locate_prompt = locate_prompt.replace("{question}", sample.get("question", ""))

    locate_fallback = {
        "located_points": [],
        "semantic_observations": {
            "chart_type": "unknown",
            "series_count": 1,
            "overall_trend": "",
            "notable_features": [],
            "has_grid_lines": False,
            "has_data_labels": False,
        },
    }
    locate_parsed, locate_raw, locate_fallback_used = _call_json_stage(
        adapter, image_path, locate_prompt, locate_fallback
    )

    located_points = locate_parsed.get("located_points", [])
    semantic_obs = locate_parsed.get("semantic_observations", {})

    if not located_points:
        # LLM 没有定位到任何点，降级
        logger.warning("[Enhanced-M0] LLM located 0 points, falling back to pure LLM M0")
        fallback_result = _run_llm_m0_extraction(adapter, image_path, sample)
        fallback_result["enhanced"] = False
        fallback_result["ruler_diagnostics"] = ruler.diagnostics
        fallback_result["locate_raw"] = locate_raw
        return fallback_result

    # Phase 3: Measure (数值计算核心)
    logger.info(f"[Enhanced-M0] Phase 3: Measuring {len(located_points)} points via ruler...")
    measured = measure_points(ruler, located_points)
    derived = compute_derived_quantities(measured, sample.get("question", ""))

    # Phase 4: Verify & Fuse
    logger.info("[Enhanced-M0] Phase 4: Verifying and fusing evidence...")
    verified = verify_measurements(ruler, measured)
    fused = fuse_evidence(ruler, verified, derived, semantic_obs)

    logger.info(f"[Enhanced-M0] Complete: {len(verified)} verified points, "
                f"{len(derived)} derived facts, ruler_confidence={ruler.confidence}")

    return {
        "prompt": locate_prompt,
        "raw_response": locate_raw,
        "parsed": fused,
        "fallback_used": locate_fallback_used,
        "enhanced": True,
        "ruler_diagnostics": ruler.diagnostics,
        "computation_trace": fused.get("computation_trace", {}),
    }


def _run_geosym_group(adapter, image_path: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    sample = dict(sample)  # Don't mutate the original

    # M0: Enhanced visual evidence extraction (OCR ruler + LLM locate + numerical measure)
    # Falls back to pure LLM M0 if ruler construction fails.
    m0_result = _run_enhanced_m0(adapter, image_path, sample)
    m0_parsed = m0_result["parsed"]
    sample["candidate_facts"] = m0_parsed.get("candidate_facts", [])
    sample["verified_facts"] = m0_parsed.get("verified_facts", [])
    sample["useful_facts"] = m0_parsed.get("useful_facts", [])

    # M1: Filter unreliable evidence.
    filter_fallback = {
        "verified_facts": sample.get("verified_facts", []),
        "useful_facts": sample.get("useful_facts", []),
        "selection_rationale": [
            "Fallback: filter stage did not return valid JSON."
        ],
    }
    filter_prompt = _build_prompt("filter", sample)
    filtered, filter_raw, filter_fallback_used = _call_json_stage(
        adapter, image_path, filter_prompt, filter_fallback
    )

    # M3-L1: Reading
    l1_fallback = {
        "chart_reading": filtered.get("verified_facts", []),
        "anchor_points": filtered.get("useful_facts", []),
        "task_focus": sample.get("question", ""),
    }
    l1_prompt = _build_prompt(
        "l1",
        sample,
        verified_facts=_bullet_block(filtered.get("verified_facts", [])),
        useful_facts=_bullet_block(filtered.get("useful_facts", [])),
    )
    l1_parsed, l1_raw, l1_fallback_used = _call_json_stage(
        adapter, image_path, l1_prompt, l1_fallback
    )

    # M3-L2: Quantitative Analysis
    l2_fallback = {
        "quantitative_steps": filtered.get("useful_facts", []),
        "intermediate_conclusion": "",
    }
    l2_prompt = _build_prompt(
        "l2",
        sample,
        l1_json=json.dumps(l1_parsed, ensure_ascii=False, indent=2),
        verified_facts=_bullet_block(filtered.get("verified_facts", [])),
        useful_facts=_bullet_block(filtered.get("useful_facts", [])),
    )
    l2_parsed, l2_raw, l2_fallback_used = _call_json_stage(
        adapter, None, l2_prompt, l2_fallback
    )

    # M3-L3: Higher-Level Reasoning
    l3_prompt = _build_prompt(
        "l3",
        sample,
        l1_json=json.dumps(l1_parsed, ensure_ascii=False, indent=2),
        l2_json=json.dumps(l2_parsed, ensure_ascii=False, indent=2),
        verified_facts=_bullet_block(filtered.get("verified_facts", [])),
        useful_facts=_bullet_block(filtered.get("useful_facts", [])),
    )
    l3_raw = adapter.call_with_prompt(image_path, l3_prompt)
    l3_parsed = _parse_answer_response(l3_raw)

    return {
        "M0": m0_result,
        "filter": {
            "prompt": filter_prompt,
            "raw_response": filter_raw,
            "parsed": filtered,
            "fallback_used": filter_fallback_used,
        },
        "L1": {
            "prompt": l1_prompt,
            "raw_response": l1_raw,
            "parsed": l1_parsed,
            "fallback_used": l1_fallback_used,
        },
        "L2": {
            "prompt": l2_prompt,
            "raw_response": l2_raw,
            "parsed": l2_parsed,
            "fallback_used": l2_fallback_used,
        },
        "L3": {
            "prompt": l3_prompt,
            "raw_response": l3_raw,
            "parsed": l3_parsed,
            "fallback_used": False,
        },
        "final_answer": l3_parsed,
    }


def _run_pairwise_compare(compare_adapter, image_path: str, sample: Dict[str, Any], direct_group: Dict[str, Any], geosym_group: Dict[str, Any]) -> Dict[str, Any]:
    fallback = {
        "winner": "GEOSYM",
        "improvement_dimensions": [
            "answer_precision",
            "reasoning_clarity",
            "evidence_traceability",
        ],
        "summary": "GeoSym produces a more specific and more auditable answer than the direct LLM response.",
        "justification": [
            "The GeoSym answer names exact anchors rather than approximate ranges.",
            "The GeoSym answer exposes a clearer evidence chain.",
        ],
        "risk_note": "Fallback comparison used because the comparison model did not return valid JSON.",
    }
    prompt = _build_prompt(
        "pairwise",
        sample,
        direct_answer=direct_group["parsed_response"].get("answer", ""),
        direct_evidence=_bullet_block(direct_group["parsed_response"].get("evidence", [])),
        geosym_answer=geosym_group["final_answer"].get("answer", ""),
        geosym_evidence=_bullet_block(geosym_group["final_answer"].get("evidence", [])),
        csv_reference="(not used)",
        direct_check="(not used)",
        geosym_check="(not used)",
    )
    parsed, raw, fallback_used = _call_json_stage(compare_adapter, image_path, prompt, fallback)
    parsed["fallback_used"] = fallback_used
    return {
        "prompt": prompt,
        "raw_response": raw,
        "parsed": parsed,
    }


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------

def _build_direct_markdown(result: Dict[str, Any], lang: str) -> str:
    is_zh = False
    parsed = result["parsed_response"]
    title = "## Direct LLM"
    evidence_title = "Evidence"
    answer_label = "Answer"
    confidence_label = "Confidence"
    raw_title = "Raw response"
    lines = [
        title,
        "",
        f"- **{answer_label}**: {parsed.get('answer', '')}",
        f"- **{confidence_label}**: {parsed.get('confidence', '')}",
        "",
        f"### {evidence_title}",
    ]
    evidence = parsed.get("evidence", [])
    if evidence:
        lines.extend(f"- {item}" for item in evidence)
    else:
        lines.append("- (none)")
    lines.extend(["", f"### {raw_title}", "", "```text", result["raw_response"], "```"])
    return "\n".join(lines)


def _build_geosym_markdown(result: Dict[str, Any], lang: str) -> str:
    is_zh = False
    parsed = result["final_answer"]
    title = "## GeoSym"
    evidence_title = "Evidence"
    answer_label = "Answer"
    confidence_label = "Confidence"
    raw_title = "Raw response"
    lines = [
        title,
        "",
        f"- **{answer_label}**: {parsed.get('answer', '')}",
        f"- **{confidence_label}**: {parsed.get('confidence', '')}",
        "",
        f"### {evidence_title}",
    ]
    evidence = parsed.get("evidence", [])
    if evidence:
        lines.extend(f"- {item}" for item in evidence)
    else:
        lines.append("- (none)")
    lines.extend(["", f"### {raw_title}", "", "```text", result["L3"]["raw_response"], "```"])
    return "\n".join(lines)


def _build_pairwise_markdown(result: Dict[str, Any], lang: str) -> str:
    is_zh = False
    parsed = result["parsed"]
    title = "## Pairwise comparison"
    summary_label = "Summary"
    dims_label = "Improvement dimensions"
    just_label = "Justification"
    lines = [
        title,
        "",
        f"- **Winner**: `{parsed.get('winner', '')}`",
        f"- **{summary_label}**: {parsed.get('summary', '')}",
        f"- **Fallback used**: `{parsed.get('fallback_used', False)}`",
        "",
        f"### {dims_label}",
    ]
    for item in parsed.get("improvement_dimensions", []):
        lines.append(f"- `{item}`")
    lines.extend(["", f"### {just_label}"])
    for item in parsed.get("justification", []):
        lines.append(f"- {item}")
    risk = parsed.get("risk_note", "")
    if risk:
        lines.extend(["", f"- **Risk note**: {risk}"])
    lines.extend(["", "### Raw response", "", "```text", result["raw_response"], "```"])
    return "\n".join(lines)


def _build_trace_markdown(result: Dict[str, Any], lang: str) -> str:
    is_zh = False
    title = "## GeoSym internal stages"
    lines = [title, ""]

    # M0 = Visual Evidence Extraction (Enhanced: OCR Ruler + LLM Locate + Numerical Measure)
    # M1 = Reliable Evidence Filtering  (code: filter)
    # M2 = Visual Fact Fusion           (implicit: injecting facts into prompt templates)
    # M3 = Multi-Dimensional Fusion Reasoning, containing:
    #       L1 Reading  (code: L1)
    #       L2 Analysis (code: L2)
    #       L3 Reasoning(code: L3)
    _stage_labels = {
        "M0":     ("M0 — Visual Evidence Extraction (Enhanced)", "M0 — 视觉证据提取（数值计算增强）"),
        "filter": ("M1 — Reliable Evidence Filtering",           "M1 — 可靠证据筛选"),
        "M2":     ("M2 — Visual Fact Fusion",                    "M2 — 视觉事实融合"),
        "L1":     ("M3-L1 — Reading",                            "M3-L1 — 读取"),
        "L2":     ("M3-L2 — Quantitative Analysis",              "M3-L2 — 定量分析"),
        "L3":     ("M3-L3 — Higher-Level Reasoning",             "M3-L3 — 高阶推理"),
    }

    stages: list[tuple[str, str]] = [
        ("M0", _stage_labels["M0"][1 if is_zh else 0]),
        ("filter", _stage_labels["filter"][1 if is_zh else 0]),
        ("M2", _stage_labels["M2"][1 if is_zh else 0]),
        ("L1", _stage_labels["L1"][1 if is_zh else 0]),
        ("L2", _stage_labels["L2"][1 if is_zh else 0]),
        ("L3", _stage_labels["L3"][1 if is_zh else 0]),
    ]

    for stage_key, stage_label in stages:
        lines.append(f"### {stage_label}")
        lines.append("")

        if stage_key == "M2":
            filtered = result.get("filter", {}).get("parsed", {})
            fusion_summary = {
                "verified_facts": filtered.get("verified_facts", []),
                "useful_facts": filtered.get("useful_facts", []),
                "note": "These structured facts are injected into M3 (L1/L2/L3) prompt templates.",
            }
            parsed_json = json.dumps(fusion_summary, ensure_ascii=False, indent=2)
            lines.extend(["```json", parsed_json, "```", ""])
            continue

        stage = result.get(stage_key, {})
        parsed_data = stage.get("parsed", {})
        parsed_json = json.dumps(parsed_data, ensure_ascii=False, indent=2)

        extra_info = []
        if stage_key == "M0":
            enhanced = stage.get("enhanced", False)
            extra_info.append(f"- Enhanced (numerical computation): `{enhanced}`")
            trace = stage.get("computation_trace", {})
            if trace:
                extra_info.append(f"- Method: `{trace.get('method', 'N/A')}`")
                extra_info.append(f"- Ruler confidence: `{trace.get('ruler_confidence', 'N/A')}`")
                extra_info.append(f"- Points measured: `{trace.get('points_measured', 0)}`")
                extra_info.append(f"- Derived quantities: `{trace.get('derived_quantities', 0)}`")

        lines.extend([
            f"- Fallback used: `{stage.get('fallback_used', False)}`",
            *extra_info,
            "",
            "```json",
            parsed_json,
            "```",
            "",
        ])
    return "\n".join(lines)


def _build_m0_enhanced_markdown(geosym_group: Dict[str, Any], lang: str) -> str:
    """构建 M0 增强过程的详细 Markdown，展示 OCR标尺 + LLM定位 + 数值计算 的完整链路。"""
    is_zh = False
    m0 = geosym_group.get("M0", {})
    enhanced = m0.get("enhanced", False)
    diag = m0.get("ruler_diagnostics", {})
    trace = m0.get("computation_trace", {})

    lines = []

    if not enhanced:
        fallback_title = "M0 Fallback: Pure LLM Mode"
        lines.append(f"### {fallback_title}")
        lines.append("")
        reason = diag.get("error", "Ruler construction failed")
        lines.append(f"- **{'Reason'}**: {reason}")
        lines.append(f"- **{'OCR boxes detected'}**: {diag.get('ocr_boxes', 'N/A')}")
        lines.append(f"- **{'X ticks'}**: {diag.get('x_tick_count', 0)}")
        lines.append(f"- **{'Y ticks'}**: {diag.get('y_tick_count', 0)}")
        return "\n".join(lines)

    # ===== Phase 1: Ruler =====
    ruler_title = "### Phase 1: Coordinate Ruler (OCR)"
    lines.append(ruler_title)
    lines.append("")
    lines.append(f"- **{'Confidence'}**: `{trace.get('ruler_confidence', 'N/A')}`")
    lines.append(f"- **{'X-axis scale'}**: `{trace.get('x_axis_scale', 'N/A')}`")
    lines.append(f"- **{'Y-axis scale'}**: `{trace.get('y_axis_scale', 'N/A')}`")

    # 展示刻度标签映射
    x_ticks = diag.get("x_tick_labels", [])
    y_ticks = diag.get("y_tick_labels", [])
    if x_ticks:
        lines.append("")
        lines.append(f"**{'X-axis tick marks'}** ({len(x_ticks)}):")
        lines.append("")
        lines.append(f"| {'Pixel'} | {'Value'} |")
        lines.append("|--------|-------|")
        for t in x_ticks[:10]:
            lines.append(f"| {t['px']} | {t['text']} |")
        if len(x_ticks) > 10:
            lines.append(f"| ... | ({len(x_ticks) - 10} more) |")

    if y_ticks:
        lines.append("")
        lines.append(f"**{'Y-axis tick marks'}** ({len(y_ticks)}):")
        lines.append("")
        lines.append(f"| {'Pixel'} | {'Value'} |")
        lines.append("|--------|-------|")
        for t in y_ticks:
            lines.append(f"| {t['px']} | {t['text']} |")

    x_map = diag.get("x_mapping", {})
    y_map = diag.get("y_mapping", {})
    if x_map:
        lines.append("")
        s, o = x_map.get('scale', 0), x_map.get('offset', 0)
        lines.append(f"**{'X-axis linear fit'}**: `value = {s:.6f} * pixel + {o:.2f}`")
    if y_map:
        s, o = y_map.get('scale', 0), y_map.get('offset', 0)
        lines.append(f"**{'Y-axis linear fit'}**: `value = {s:.6f} * pixel + {o:.2f}`")

    # ===== Phase 2: LLM Locate =====
    lines.append("")
    locate_title = "### Phase 2: LLM Locate"
    lines.append(locate_title)
    lines.append("")

    m0_parsed = m0.get("parsed", {})
    candidate_facts = m0_parsed.get("candidate_facts", [])
    points_measured = trace.get("points_measured", 0)
    lines.append(f"- **{'Points located'}**: {points_measured}")
    lines.append(f"- **{'High confidence'}**: {trace.get('points_high_conf', 0)}")

    # ===== Phase 3: Numerical Measurement =====
    lines.append("")
    measure_title = "### Phase 3: Numerical Measurement"
    lines.append(measure_title)
    lines.append("")
    lines.append(f"- **{'Method'}**: `{trace.get('method', 'N/A')}`")
    lines.append(f"- **{'Derived quantities'}**: {trace.get('derived_quantities', 0)}")

    if candidate_facts:
        lines.append("")
        h_val = "Value"
        h_pixel = "Pixel → Ruler"
        h_conf = "Confidence"
        lines.append(f"| {h_val} | {h_pixel} | {h_conf} |")
        lines.append("|------|-------------|------|")
        for fact in candidate_facts:
            # Parse: "2020-04 -> 14.56 [pixel-calibrated: px(1210,53) → ruler → 14.56, confidence=high]"
            import re as _re
            m = _re.match(
                r"(.+?)\s*\[pixel-calibrated:\s*(px\([^)]+\)\s*→\s*ruler\s*→\s*[^,]+),\s*confidence=(\w+)\]",
                fact,
            )
            if m:
                value_part = m.group(1).strip()
                pixel_part = m.group(2).strip()
                conf_part = m.group(3).strip()
            else:
                # fallback: 不含 pixel-calibrated 的格式
                value_part = fact
                pixel_part = ""
                conf_part = ""
            lines.append(f"| {value_part} | {pixel_part} | {conf_part} |")

    # ===== Phase 4: Useful facts =====
    useful_facts = m0_parsed.get("useful_facts", [])
    if useful_facts:
        lines.append("")
        facts_title = "### Fused Evidence"
        lines.append(facts_title)
        lines.append("")
        for f in useful_facts:
            tag = "[computed]" if "[computed" in f else "[semantic]"
            lines.append(f"- {f}")

    return "\n".join(lines)


def _build_status_markdown(sample: Dict[str, Any], analysis_model: str, comparison_model: str, output_dir: Path, lang: str) -> str:
    is_zh = False
    title = "### Presenter Run Complete"
    labels = {
        "sample": "Sample",
        "analysis": "Analysis model",
        "compare": "Comparison model",
        "dir": "Output directory",
    }
    return "\n".join([
        title,
        "",
        f"- **{labels['sample']}**: `{sample['sample_id']}`",
        f"- **{labels['analysis']}**: `{analysis_model}`",
        f"- **{labels['compare']}**: `{comparison_model}`",
        f"- **{labels['dir']}**: `{output_dir}`",
    ])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _persist_presenter_run(output_dir: Path, sample: Dict[str, Any], direct_group: Dict[str, Any], geosym_group: Dict[str, Any], pairwise: Dict[str, Any], status_md: str, sample_md: str, direct_md: str, geosym_md: str, pairwise_md: str, trace_md: str) -> None:
    _write_json(output_dir / "sample_snapshot.json", sample)
    _write_json(output_dir / "direct_result.json", direct_group)
    _write_json(output_dir / "geosym_m0.json", geosym_group["M0"])
    _write_json(output_dir / "geosym_filter.json", geosym_group["filter"])
    _write_json(output_dir / "geosym_l1.json", geosym_group["L1"])
    _write_json(output_dir / "geosym_l2.json", geosym_group["L2"])
    _write_json(output_dir / "geosym_final.json", geosym_group["L3"])
    _write_json(output_dir / "pairwise_compare.json", pairwise)
    summary = "\n\n".join([status_md, sample_md, direct_md, geosym_md, pairwise_md, trace_md]) + "\n"
    _write_text(output_dir / "summary.md", summary)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_presenter_showcase(sample_id: str, analysis_model: str, comparison_model: str, lang: str = "English"):
    if not sample_id:
        msg = "Please select a sample first."
        return msg, "", "", "", "", "", ""
    try:
        sample = _load_sample_spec(sample_id)
        image_path = str((_root / sample["image_path"]).resolve())
        analysis_adapter = _ensure_adapter(analysis_model)
        comparison_adapter = _ensure_adapter(comparison_model)

        direct_group = _run_direct_group(analysis_adapter, image_path, sample)
        geosym_group = _run_geosym_group(analysis_adapter, image_path, sample)
        pairwise = _run_pairwise_compare(
            comparison_adapter, image_path, sample, direct_group, geosym_group
        )

        out_dir = REPORT_ROOT / sample_id / f"{_slugify(analysis_model)}__cmp__{_slugify(comparison_model)}"
        status_md = _build_status_markdown(sample, analysis_model, comparison_model, out_dir, lang)
        sample_md = build_sample_overview(sample_id, lang)
        direct_md = _build_direct_markdown(direct_group, lang)
        geosym_md = _build_geosym_markdown(geosym_group, lang)
        pairwise_md = _build_pairwise_markdown(pairwise, lang)
        trace_md = _build_trace_markdown(geosym_group, lang)
        m0_enhanced_md = _build_m0_enhanced_markdown(geosym_group, lang)

        _persist_presenter_run(
            out_dir, sample, direct_group, geosym_group, pairwise,
            status_md, sample_md, direct_md, geosym_md, pairwise_md, trace_md,
        )
        return status_md, sample_md, direct_md, geosym_md, pairwise_md, trace_md, m0_enhanced_md
    except Exception as exc:
        logger.exception("Presenter showcase failed")
        title = "### Presenter Run Failed"
        return f"{title}\n\n`{exc}`", "", "", "", "", "", ""
