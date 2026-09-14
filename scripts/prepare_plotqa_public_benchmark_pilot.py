#!/usr/bin/env python
"""Prepare a frozen zero-cost PlotQA public benchmark pilot plan.

This script consumes the official local PlotQA export only: annotations JSON,
QA-pairs JSON, and a PNG image directory. It performs QA-pair-level joining,
deterministic eligibility tagging, full-split hash sorting, and 8-sample roster
selection without reading model outputs or spending paid calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_public_benchmark_pilot import (  # noqa: E402
    canonical_json_bytes,
    normalize_answer,
    sha256_hex_bytes,
    stable_file_hash,
)


SUPPORTED_CHART_TYPES = {"line", "bar", "simple_bar", "grouped_bar", "scatter"}
SUPPORTED_REASONING_TYPES = {"lookup", "comparison", "extrema", "trend", "range", "slope", "ratio"}
STRUCTURAL_QA_TEMPLATES = {"structural"}
DEFAULT_CONDITIONS = ("E0_plotqa_direct", "E1_plotqa_cot", "G0_plotqa_evidence_trace")
DEFAULT_SEED = "20260829"
DEFAULT_SAMPLE_COUNT = 8
STAGE2_SAMPLE_COUNT = 50
SCORER_VERSION = "plotqa_phase1_public_scorer_v1"
DEFAULT_CONFIG_PATH = ROOT / "config" / "default_config.yaml"
SHARED_FINAL_ANSWER_INSTRUCTION = (
    "Return JSON only with keys raw_text and normalized_prediction. "
    "The normalized_prediction value must be the final answer string."
)


@dataclass(frozen=True)
class PlotQACandidate:
    sample_id: str
    image_path: str
    question: str
    answer: str
    answer_type: str
    metadata: dict[str, Any]


def normalize_chart_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"line", "dot_line", "dotline"}:
        return "line"
    if text in {"bar", "bar_chart", "vbar", "hbar", "vbar_categorical", "hbar_categorical"}:
        return "bar"
    if text in {"scatter", "scatter_plot"}:
        return "scatter"
    if not text:
        return "unknown"
    return "other"


def normalize_question_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def infer_reasoning_type(question: str, qa_template: str = "", qa_type: str = "") -> str:
    question_q = normalize_question_text(question)
    template_q = normalize_question_text(qa_template)
    qa_type_q = normalize_question_text(qa_type)
    q = normalize_question_text(" ".join([question, qa_template, qa_type]))
    if template_q in STRUCTURAL_QA_TEMPLATES:
        return "text"
    text_cues = ("title", "caption", "legend label", "axis label", "x axis label", "y axis label", "color of", "colour of", "name of the chart")
    structural_cues = (
        "legend labels",
        "legend entries",
        "legend names",
        "labels are there",
        "bars are there",
        "lines are there",
        "plots are there",
        "how many legend",
        "how many labels",
    )
    unsupported_aggregate_cues = ("average", "mean", "median", "sum of", "total of")
    quantitative_cues = (
        "value",
        "amount",
        "number",
        "how many",
        "how much",
        "higher",
        "lower",
        "greater",
        "less",
        "most",
        "least",
        "increase",
        "decrease",
        "ratio",
        "percentage",
        "percent",
        "difference",
        "range",
        "slope",
        "rate",
    )
    if any(cue in question_q for cue in structural_cues):
        return "text"
    if any(cue in question_q for cue in unsupported_aggregate_cues):
        return "unknown"
    if any(cue in q for cue in text_cues) and not any(cue in q for cue in quantitative_cues):
        return "text"
    if re.search(r"\bdifference between (?:the )?(?:highest|maximum|largest|most) and (?:the )?(?:lowest|minimum|smallest|least)\b", question_q):
        return "range"
    extrema_markers = ("highest", "lowest", "maximum", "minimum", "peak", "largest", "smallest", "most", "least")
    comparison_markers = (
        "more than",
        "less than",
        "greater than",
        "fewer than",
        "difference between",
        "compare",
        "which is higher",
        "which is lower",
        "which is greater",
        "which is smaller",
    )
    if template_q == "min_max" or any(marker in question_q for marker in extrema_markers):
        return "extrema"
    if any(marker in question_q for marker in comparison_markers):
        return "comparison"
    if re.search(r"\b(?:ratio|proportion|times as|fraction of)\b", question_q) or "what percentage of" in question_q:
        return "ratio"
    ordered = [
        ("slope", ("slope", "rate of change", "increase per", "decrease per", "per year")),
        ("range", ("range", "difference between highest and lowest", "max minus min", "spread")),
        ("trend", ("trend", "increase", "decrease", "overall", "change over time", "from start to end")),
        (
            "lookup",
            (
                "value",
                "amount",
                "number",
                "how many",
                "how much",
                "what is the value",
                "what was the value",
                "what is the amount",
                "what was the amount",
                "what is the number",
                "what was the number",
            ),
        ),
    ]
    for label, markers in ordered:
        if any(marker in q for marker in markers):
            return label
    return "unknown"


def infer_answer_derivation(reasoning_type: str, question: str) -> str:
    q = normalize_question_text(question)
    if re.search(r"\bdifference between (?:the )?(?:highest|maximum|largest|most) and (?:the )?(?:lowest|minimum|smallest|least)\b", q):
        return "derived"
    derived = ("difference", "how many more", "how much more", "how many fewer", "how much fewer", "minus", "between")
    direct = ("which is higher", "which is lower", "which is greater", "which is smaller")
    if reasoning_type == "comparison":
        if any(cue in q for cue in derived):
            return "derived"
        if "how many" in q or "how much" in q:
            return "derived"
        if any(cue in q for cue in direct):
            return "direct"
        return "unknown"
    if reasoning_type in {"range", "slope", "ratio", "trend"}:
        return "derived"
    if reasoning_type in {"lookup", "extrema"}:
        return "direct"
    return "unknown"


def validate_plotqa_sample_semantics(sample: dict[str, Any]) -> list[str]:
    metadata = sample.get("metadata") or {}
    question = str(sample.get("question", ""))
    qa_template = str(metadata.get("qa_template", "") or "")
    qa_type = str(metadata.get("qa_type", "") or "")
    expected_reasoning = infer_reasoning_type(question, qa_template, qa_type)
    expected_derivation = infer_answer_derivation(expected_reasoning, question)
    violations = []
    if normalize_question_text(qa_template) in STRUCTURAL_QA_TEMPLATES:
        violations.append("structural_qa_template_excluded")
    if expected_reasoning == "text":
        violations.append("text_or_structural_question")
    if expected_reasoning == "unknown":
        violations.append("reasoning_type_unknown_by_cues")
    if metadata.get("reasoning_type") != expected_reasoning:
        violations.append(f"reasoning_type_mismatch:{metadata.get('reasoning_type')}!={expected_reasoning}")
    if metadata.get("answer_derivation") not in {"direct", "derived"}:
        violations.append(f"answer_derivation_unresolved:{metadata.get('answer_derivation')}")
    if expected_derivation != "unknown" and metadata.get("answer_derivation") != expected_derivation:
        violations.append(f"answer_derivation_mismatch:{metadata.get('answer_derivation')}!={expected_derivation}")
    expected_quantitative = expected_reasoning in SUPPORTED_REASONING_TYPES
    if metadata.get("quantitative_reasoning_required") is not expected_quantitative:
        violations.append(f"quantitative_flag_mismatch:{metadata.get('quantitative_reasoning_required')}!={expected_quantitative}")
    return violations


def infer_answer_type(answer: Any) -> str:
    try:
        normalize_answer(answer, "numeric")
        return "numeric"
    except Exception:
        return "text"


def _axis_has_ticks(axis: Any) -> bool:
    if not isinstance(axis, dict):
        return False
    tick_keys = ("major_ticks", "minor_ticks", "ticks", "tick_labels", "values")
    return any(bool(axis.get(key)) for key in tick_keys)


def _image_readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def image_manifest_hash(image_dir: Path) -> str:
    rows = []
    for path in sorted(image_dir.glob("*.png"), key=lambda p: p.name):
        rows.append({"name": path.name, "sha256": stable_file_hash(path)})
    return sha256_hex_bytes(canonical_json_bytes({"images": rows}))


def _canonical_sha(value: Any) -> str:
    return sha256_hex_bytes(canonical_json_bytes(value if isinstance(value, dict) else {"value": value}))


def resolve_model_provenance(config_path: Path, display_name: str | None = None) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    models_cfg = config.get("models")
    if not isinstance(models_cfg, dict) or not models_cfg:
        raise ValueError("model_config_missing: config/default_config.yaml has no models section")
    first_model: dict[str, Any] | None = None
    for provider, provider_cfg in models_cfg.items():
        if not isinstance(provider_cfg, dict):
            continue
        adapter = str(provider_cfg.get("adapter") or "")
        api_key_name = str(provider_cfg.get("api_key_name") or str(provider).lower())
        base_url_key = str(provider_cfg.get("base_url_key") or "")
        provider_models = provider_cfg.get("models")
        if not isinstance(provider_models, dict):
            continue
        for configured_display_name, api_model_id in provider_models.items():
            row = {
                "display_name": str(configured_display_name),
                "provider": str(provider),
                "adapter": adapter,
                "api_model_id": str(api_model_id),
                "api_key_name": api_key_name,
                "base_url_key": base_url_key,
                "gateway_policy": "base_url_key" if base_url_key else "provider_default",
                "config_path": str(config_path),
                "config_sha256": stable_file_hash(config_path),
            }
            if first_model is None:
                first_model = row
            if display_name and str(configured_display_name) == display_name:
                return row
    if display_name:
        raise ValueError(f"model_display_name_not_in_config: {display_name}")
    if first_model is None:
        raise ValueError("model_config_missing: no configured models found")
    return first_model


def build_candidate(
    annotation: dict[str, Any] | None,
    qa: dict[str, Any],
    qa_index: int,
    image_dir: Path,
    split: str,
    image_cache: dict[str, dict[str, Any]] | None = None,
    annotation_hash_cache: dict[str, str] | None = None,
) -> PlotQACandidate:
    image_index = str(qa.get("image_index", ""))
    question = str(qa.get("question_string", "")).strip()
    answer = str(qa.get("answer", "")).strip()
    qa_template = str(qa.get("template", "") or "")
    qa_type = str(qa.get("type", "") or "")
    image_path = image_dir / f"{image_index}.png"
    question_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
    if image_cache is not None and image_index in image_cache:
        image_info = image_cache[image_index]
    else:
        image_exists = image_path.exists()
        image_readable = image_exists and _image_readable(image_path)
        image_info = {
            "exists": image_exists,
            "readable": image_readable,
            "sha256": stable_file_hash(image_path) if image_readable else "",
        }
        if image_cache is not None:
            image_cache[image_index] = image_info
    image_exists = bool(image_info["exists"])
    image_readable = bool(image_info["readable"])
    image_sha256 = str(image_info["sha256"])
    if annotation_hash_cache is not None and image_index in annotation_hash_cache:
        annotation_sha256 = annotation_hash_cache[image_index]
    else:
        annotation_sha256 = _canonical_sha(annotation or {})
        if annotation_hash_cache is not None:
            annotation_hash_cache[image_index] = annotation_sha256
    qa_pair_sha256 = _canonical_sha(qa)
    payload = {
        "annotation_sha256": annotation_sha256,
        "image_index": image_index,
        "image_sha256": image_sha256,
        "qa_index": str(qa_index),
        "qa_pair_sha256": qa_pair_sha256,
        "question_sha256": question_sha256,
        "seed": DEFAULT_SEED,
    }
    sample_id = f"plotqa_{split}_{image_index}_{qa_index}_{sha256_hex_bytes(canonical_json_bytes(payload))[:8]}"

    chart_type = normalize_chart_type(annotation.get("type") if annotation else None)
    figure = annotation.get("general_figure_info") if annotation else None
    figure = figure if isinstance(figure, dict) else {}
    has_x_axis = isinstance(figure.get("x_axis"), dict)
    has_y_axis = isinstance(figure.get("y_axis"), dict)
    has_tick_labels = _axis_has_ticks(figure.get("x_axis")) or _axis_has_ticks(figure.get("y_axis"))
    axis_visible = has_x_axis and has_y_axis and has_tick_labels
    models = annotation.get("models") if annotation else None
    series_count: int | str = len(models) if isinstance(models, list) else "unknown"
    has_data_table = isinstance(models, list) and bool(models)
    legend_available = isinstance(figure.get("legend"), dict) and bool(figure.get("legend"))
    normalized_question = normalize_question_text(question)
    series_cues = ("legend", "series", "line for", "bar for", "category for", "group", "color", "colour")
    legend_required = (isinstance(series_count, int) and series_count > 1) or any(cue in normalized_question for cue in series_cues)
    reasoning_type = infer_reasoning_type(question, qa_template, qa_type)
    answer_derivation = infer_answer_derivation(reasoning_type, question)
    quantitative = reasoning_type in SUPPORTED_REASONING_TYPES

    if annotation is None:
        compatibility = "incompatible"
        note = "missing_annotation"
    elif not image_exists:
        compatibility = "incompatible"
        note = "missing_image"
    elif not image_readable:
        compatibility = "incompatible"
        note = "image_open_failed"
    elif chart_type == "unknown":
        compatibility = "unknown"
        note = "chart_type_unknown"
    elif chart_type not in SUPPORTED_CHART_TYPES:
        compatibility = "incompatible"
        note = "unsupported_chart_type"
    elif not has_x_axis or not has_y_axis:
        compatibility = "incompatible"
        note = "axis_metadata_missing"
    elif not has_tick_labels:
        compatibility = "incompatible"
        note = "tick_metadata_missing"
    elif not has_data_table or series_count == "unknown":
        compatibility = "incompatible"
        note = "data_model_missing"
    elif legend_required and not legend_available:
        compatibility = "incompatible"
        note = "legend_required_missing"
    elif reasoning_type == "text":
        compatibility = "incompatible"
        note = "text_only_answer"
    elif reasoning_type == "unknown":
        compatibility = "unknown"
        note = "reasoning_type_unknown"
    else:
        compatibility = "presenter_candidate"
        note = "eligible"

    metadata = {
        "benchmark": "PlotQA",
        "split": split,
        "image_index": image_index,
        "qa_index": str(qa_index),
        "question_sha256": question_sha256,
        "image_sha256": image_sha256,
        "annotation_sha256": annotation_sha256,
        "qa_pair_sha256": qa_pair_sha256,
        "answer_bbox_present": "answer_bbox" in qa,
        "qa_template": qa_template,
        "qa_type": qa_type,
        "chart_type": chart_type,
        "axis_visible": axis_visible,
        "image_readable": image_readable,
        "has_x_axis": has_x_axis,
        "has_y_axis": has_y_axis,
        "has_tick_labels": has_tick_labels,
        "has_data_table": has_data_table,
        "series_count": series_count,
        "legend_required": legend_required,
        "legend_available": legend_available,
        "reasoning_type": reasoning_type,
        "answer_derivation": answer_derivation,
        "quantitative_reasoning_required": quantitative,
        "geosym_compatibility": compatibility,
        "selection_note": note,
    }
    return PlotQACandidate(sample_id, str(image_path), question, answer, infer_answer_type(answer), metadata)


def load_candidates(annotations_path: Path, qa_pairs_path: Path, image_dir: Path, split: str) -> tuple[list[PlotQACandidate], dict[str, Any]]:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    qa_pairs = json.loads(qa_pairs_path.read_text(encoding="utf-8"))
    if isinstance(qa_pairs, dict) and isinstance(qa_pairs.get("qa_pairs"), list):
        qa_pairs = qa_pairs["qa_pairs"]
    if not isinstance(annotations, list) or not isinstance(qa_pairs, list):
        raise ValueError("PlotQA annotations and qa_pairs must both be JSON lists")
    annotations_by_image = {str(row.get("image_index")): row for row in annotations}
    image_cache: dict[str, dict[str, Any]] = {}
    annotation_hash_cache: dict[str, str] = {}
    candidates = [
        build_candidate(
            annotations_by_image.get(str(qa.get("image_index"))),
            qa,
            idx,
            image_dir,
            split,
            image_cache=image_cache,
            annotation_hash_cache=annotation_hash_cache,
        )
        for idx, qa in enumerate(qa_pairs)
    ]
    access = {
        "annotations_count": len(annotations),
        "qa_pair_count": len(qa_pairs),
        "qa_pairs_with_annotation": sum(1 for qa in qa_pairs if str(qa.get("image_index")) in annotations_by_image),
        "qa_pairs_with_readable_image": sum(1 for candidate in candidates if candidate.metadata["image_readable"] is True),
    }
    return candidates, access


def selection_hash(candidate: PlotQACandidate) -> str:
    metadata = candidate.metadata
    return sha256_hex_bytes(
        canonical_json_bytes(
            {
                "annotation_sha256": metadata["annotation_sha256"],
                "image_index": metadata["image_index"],
                "image_sha256": metadata["image_sha256"],
                "qa_index": metadata["qa_index"],
                "qa_pair_sha256": metadata["qa_pair_sha256"],
                "question_sha256": metadata["question_sha256"],
                "seed": DEFAULT_SEED,
            }
        )
    )


def eligible(candidate: PlotQACandidate) -> bool:
    return first_exclusion_reason(candidate) == "eligible"


def first_exclusion_reason(candidate: PlotQACandidate) -> str:
    metadata = candidate.metadata
    semantic_violations = validate_plotqa_sample_semantics(sample_row(candidate, normalize=False))
    if metadata["chart_type"] not in SUPPORTED_CHART_TYPES:
        return "chart_type_unknown" if metadata["chart_type"] == "unknown" else "unsupported_chart_type"
    if metadata["image_readable"] is not True:
        return "image_not_readable"
    if metadata["axis_visible"] is not True or metadata["has_x_axis"] is not True or metadata["has_y_axis"] is not True:
        return "axis_metadata_missing"
    if metadata["has_tick_labels"] is not True:
        return "tick_metadata_missing"
    if metadata["has_data_table"] is not True:
        return "data_model_missing"
    if metadata["legend_required"] is True and metadata["legend_available"] is not True:
        return "legend_required_missing"
    if "structural_qa_template_excluded" in semantic_violations or "text_or_structural_question" in semantic_violations:
        return "structural_or_text_question"
    if "reasoning_type_unknown_by_cues" in semantic_violations:
        return "reasoning_type_unknown"
    if metadata["reasoning_type"] not in SUPPORTED_REASONING_TYPES:
        return "unsupported_reasoning_type"
    if metadata["answer_derivation"] not in {"direct", "derived"}:
        return "answer_derivation_unknown"
    if semantic_violations:
        return "semantic_tag_conflict"
    if metadata["quantitative_reasoning_required"] is not True:
        return "not_quantitative_reasoning"
    if metadata["geosym_compatibility"] != "presenter_candidate":
        return "geosym_compatibility_unknown"
    return "eligible"


def select_roster(candidates: list[PlotQACandidate], count: int) -> tuple[list[PlotQACandidate], dict[str, Any]]:
    sorted_candidates = sorted(candidates, key=lambda row: (selection_hash(row), row.sample_id))
    initial_pool_size = 1000 if count >= STAGE2_SAMPLE_COUNT else 100
    expanded_pool_size = 5000 if count >= STAGE2_SAMPLE_COUNT else 200
    pool = sorted_candidates[:initial_pool_size]
    expanded = False
    pool_eligible = [row for row in pool if eligible(row)]
    if len(pool_eligible) < count:
        pool = sorted_candidates[:expanded_pool_size]
        expanded = True
        pool_eligible = [row for row in pool if eligible(row)]
    if len(pool_eligible) < count:
        raise ValueError(f"selector_insufficient: fewer than {count} eligible QA-pair samples ({len(pool_eligible)})")

    unique = []
    seen_images = set()
    for row in pool_eligible:
        image_index = row.metadata["image_index"]
        if image_index in seen_images:
            continue
        unique.append(row)
        seen_images.add(image_index)
    if len(unique) < count:
        raise ValueError(f"selector_insufficient: fewer than {count} unique-image eligible samples ({len(unique)})")

    if count >= STAGE2_SAMPLE_COUNT:
        selected = select_stage2_roster(unique, count)
        roster_note = "stage2_quota_satisfied" if len(selected) == count else "stage2_quota_underfilled"
    else:
        derived = [row for row in unique if row.metadata["answer_derivation"] == "derived"]
        roster_note = "derived_quota_satisfied"
        if len(derived) >= 4:
            selected = derived[:4]
            selected_ids = {row.sample_id for row in selected}
            for row in unique:
                if len(selected) >= count:
                    break
                if row.sample_id not in selected_ids:
                    selected.append(row)
                    selected_ids.add(row.sample_id)
        else:
            selected = unique[:count]
            roster_note = "derived_quota_insufficient"
    if len(selected) < count:
        raise ValueError(f"selector_insufficient: fewer than {count} quota-compatible samples ({len(selected)})")
    derived = [row for row in unique if row.metadata["answer_derivation"] == "derived"]
    return selected, {
        "candidate_count": len(sorted_candidates),
        "candidate_pool_size": len(pool),
        "expanded_to_200": expanded if count < STAGE2_SAMPLE_COUNT else False,
        "expanded_to_5000": expanded if count >= STAGE2_SAMPLE_COUNT else False,
        "eligible_count": len(pool_eligible),
        "unique_image_eligible_count": len(unique),
        "derived_eligible_count": len(derived),
        "roster_note": roster_note,
    }


def select_stage2_roster(unique: list[PlotQACandidate], count: int) -> list[PlotQACandidate]:
    selected: list[PlotQACandidate] = []
    selected_ids: set[str] = set()

    def add_rows(rows: Iterable[PlotQACandidate], limit: int) -> None:
        added = 0
        for row in rows:
            if added >= limit:
                break
            if row.sample_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row.sample_id)
            added += 1

    add_rows((row for row in unique if row.metadata["answer_derivation"] == "derived"), 20)
    add_rows((row for row in unique if row.metadata["answer_derivation"] == "direct"), 10)
    add_rows(
        (
            row
            for row in unique
            if row.metadata["reasoning_type"] in {"extrema", "lookup"}
            and row.metadata["answer_derivation"] == "direct"
        ),
        10,
    )
    add_rows(
        (
            row
            for row in unique
            if row.metadata["reasoning_type"] in {"comparison", "range"}
            and row.metadata["answer_derivation"] == "derived"
        ),
        10,
    )
    add_rows(unique, count - len(selected))
    return selected[:count]


def prompt_hashes() -> dict[str, str]:
    prompts = {
        "shared_final_answer_instruction": SHARED_FINAL_ANSWER_INSTRUCTION,
        "e0_prompt_template": e0_prompt("{question}"),
        "e1_prompt_template": e1_prompt("{question}"),
        "g0_extract_prompt_template": g0_extract_prompt("{question}"),
        "g0_curate_prompt_template": g0_curate_prompt("{question}", []),
        "g0_answer_prompt_template": g0_answer_prompt("{question}", []),
    }
    return {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in prompts.items()}


def e0_prompt(question: str) -> str:
    return f"{SHARED_FINAL_ANSWER_INSTRUCTION}\nUse the image to answer the PlotQA chart question.\nQuestion: {question}"


def e1_prompt(question: str) -> str:
    return (
        f"{SHARED_FINAL_ANSWER_INSTRUCTION}\nUse the image to answer the chart question. "
        "Include brief reasoning inside raw_text only.\n"
        f"Question: {question}"
    )


def g0_extract_prompt(question: str) -> str:
    return (
        "Return JSON only with key raw_observations. Each observation must have fact_id, source_step, text, "
        "numeric_entities, categorical_entities, and confidence. Use source_step='extract'. Do not answer the question.\n"
        f"Question: {question}"
    )


def g0_curate_prompt(question: str, raw_observations: list[dict[str, Any]]) -> str:
    return (
        "Return JSON only with keys retained_facts and rejected_or_low_confidence_facts. Preserve raw fact_id lineage. "
        "Do not add new visual facts and do not provide a final answer.\n"
        f"Question: {question}\nRaw observations: {json.dumps(raw_observations, ensure_ascii=False)}"
    )


def g0_answer_prompt(question: str, retained_facts: list[dict[str, Any]]) -> str:
    return (
        f"{SHARED_FINAL_ANSWER_INSTRUCTION}\nAnswer using the retained facts. "
        "Do not use reference answers or excluded-sample rationales.\n"
        f"Question: {question}\nRetained facts: {json.dumps(retained_facts, ensure_ascii=False)}"
    )


def excluded_summary(candidates: Iterable[PlotQACandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        if eligible(candidate):
            continue
        note = first_exclusion_reason(candidate)
        counts[note] = counts.get(note, 0) + 1
    return counts


def sample_row(candidate: PlotQACandidate, normalize: bool = True) -> dict[str, Any]:
    row = {
        "sample_id": candidate.sample_id,
        "image_path": candidate.image_path,
        "question": candidate.question,
        "answer": candidate.answer,
        "answer_type": candidate.answer_type,
        "metadata": candidate.metadata,
    }
    if normalize:
        row["normalized_answer"] = normalize_answer(candidate.answer, candidate.answer_type)
    return row


def load_excluded_sample_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("selected_samples") or payload.get("samples") or payload.get("selected_ids") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("exclude sample file must be a JSON object or array")
    excluded: set[str] = set()
    for row in rows:
        if isinstance(row, str):
            excluded.add(row)
        elif isinstance(row, dict) and row.get("sample_id"):
            excluded.add(str(row["sample_id"]))
    return excluded


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    annotations_path = Path(args.annotations).resolve()
    qa_pairs_path = Path(args.qa_pairs).resolve()
    image_dir = Path(args.image_dir).resolve()
    if not annotations_path.exists() or not qa_pairs_path.exists() or not image_dir.is_dir():
        raise ValueError("plotqa_access_boundary: required official PlotQA files are missing")
    candidates, access = load_candidates(annotations_path, qa_pairs_path, image_dir, args.split)
    excluded_ids = load_excluded_sample_ids(args.exclude_samples)
    if excluded_ids:
        candidates = [candidate for candidate in candidates if candidate.sample_id not in excluded_ids]
    selected, selection = select_roster(candidates, args.max_samples)
    model_provenance = resolve_model_provenance(Path(args.config), args.model)
    conditions = [condition.strip() for condition in args.conditions.split(",") if condition.strip()]
    unsupported = sorted(set(conditions) - set(DEFAULT_CONDITIONS))
    if unsupported:
        raise ValueError(f"unsupported PlotQA Phase 1 condition names: {unsupported}")
    sample_condition_jobs = len(selected) * len(conditions)
    llm_attempt_cap = len(selected) * sum(3 if condition == "G0_plotqa_evidence_trace" else 1 for condition in conditions)
    if sample_condition_jobs > args.max_paid_jobs:
        raise ValueError("sample-condition job cap exceeded")
    if llm_attempt_cap > args.max_llm_attempts:
        raise ValueError("LLM attempt cap exceeded")
    if len({row.metadata["image_index"] for row in selected}) != len(selected):
        raise ValueError("selector_insufficient: selected samples do not have unique image_index values")

    frozen = {
        "hypothesis": args.hypothesis,
        "official_files": {
            "annotations_json": str(annotations_path),
            "qa_pairs_json": str(qa_pairs_path),
            "image_dir": str(image_dir),
            "annotations_sha256": stable_file_hash(annotations_path),
            "qa_pairs_sha256": stable_file_hash(qa_pairs_path),
            "image_manifest_sha256": image_manifest_hash(image_dir),
        },
        "sample_selection": {
            "seed": DEFAULT_SEED,
            "sampling_frame": "official_plotqa_full_split_hash_pool",
            "selected_ids": [row.sample_id for row in selected],
            "excluded_prior_sample_ids": sorted(excluded_ids),
            "excluded_prior_sample_count": len(excluded_ids),
            "overlap_with_excluded_prior_sample_ids": sorted({row.sample_id for row in selected} & excluded_ids),
            **access,
            **selection,
        },
        "conditions": conditions,
        "sample_count": len(selected),
        "reasoning_model": model_provenance["api_model_id"],
        "model_provenance": model_provenance,
        "temperature": 0.0,
        "decoding": "provider default except temperature 0 when adapter exposes it",
        "prompt_hashes": prompt_hashes(),
        "scorer_version": SCORER_VERSION,
        "sample_condition_job_cap": sample_condition_jobs,
        "llm_http_attempt_cap": llm_attempt_cap,
        "ocr_http_attempt_cap": 0,
        "per_condition_attempt_caps": {
            "E0_plotqa_direct": {"llm": 1, "ocr": 0},
            "E1_plotqa_cot": {"llm": 1, "ocr": 0},
            "G0_plotqa_evidence_trace": {"llm": 3, "ocr": 0},
        },
        "retry_policy": "no retry expansion; one counted attempt per external request step",
        "cache_policy": "cache key includes sample_id, condition, model ID, prompt hashes, OCR flag, plan hash, and scorer version",
        "stop_rule": "stop after the frozen 8-sample roster and condition list; no adaptive replacement",
        "cost_estimate": "at most 24 sample-condition jobs and 40 LLM HTTP attempts; OCR disabled",
        "interpretation_limits": [
            "PlotQA compatibility and attribution smoke only",
            "not a broad public-benchmark performance claim",
            "proxy attribution labels are diagnostic and do not prove true causal module failure",
            "derived questions without operand schema remain attribution-limited",
        ],
    }
    plan = {
        "schema_version": "plotqa_g0_smoke_plan_v1",
        "benchmark": "PlotQA",
        "split": args.split,
        "created_date": args.created_date,
        "paid_run_approved": True,
        "paid_calls_executed": 0,
        "frozen_before_paid_run": frozen,
        "selected_samples": [sample_row(row) for row in selected],
        "excluded_summary": excluded_summary(candidates),
    }
    plan["plan_sha256"] = sha256_hex_bytes(
        canonical_json_bytes({key: value for key, value in plan.items() if key != "plan_sha256"})
    )
    return plan


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--qa-pairs", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--max-paid-jobs", type=int, default=24)
    parser.add_argument("--max-llm-attempts", type=int, default=40)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--exclude-samples", default=None, help="JSON plan/list whose sample_id values must be excluded before selection.")
    parser.add_argument("--model", default=None, help="Display-name model from config/default_config.yaml. Defaults to the first configured model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the config containing the paid-facing models section.")
    parser.add_argument("--created-date", default=date.today().isoformat())
    parser.add_argument(
        "--hypothesis",
        default=(
            "An 8-sample official PlotQA smoke can test whether GeoSym-style evidence traces run on "
            "public chart-QA metadata with strict attribution and paid caps."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        plan = build_plan(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    frozen = plan["frozen_before_paid_run"]
    print(f"wrote frozen zero-cost PlotQA plan: {output}")
    print(f"selected_samples={len(plan['selected_samples'])}")
    print(f"sample_condition_job_cap={frozen['sample_condition_job_cap']}")
    print(f"llm_http_attempt_cap={frozen['llm_http_attempt_cap']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
