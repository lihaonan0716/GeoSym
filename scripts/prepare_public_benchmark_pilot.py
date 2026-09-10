#!/usr/bin/env python
"""Prepare a frozen zero-cost ChartQA public benchmark pilot plan.

The script performs no model or OCR calls. It validates an exported local
manifest, derives deterministic question tags, applies the Phase 1 fixed
selection rule, and writes the paid-run plan only when the 8-sample roster can
be frozen without answer-dependent filtering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


SUPPORTED_ANSWER_TYPES = {"categorical", "numeric", "text", "open"}
SUPPORTED_CHART_TYPES = {"line", "scatter", "simple_bar", "grouped_bar"}
SUPPORTED_REASONING_TYPES = {"lookup", "comparison", "extrema", "trend", "range", "slope", "ratio"}
EXCLUDED_CHART_TYPES = {"pie", "infographic", "multi_panel", "other"}
DEFAULT_CONDITIONS = ("E0_public_direct", "E1_public_cot", "G0_public_evidence_trace")
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_SAMPLE_COUNT = 8
DEFAULT_SEED = "20260828"
SCORER_VERSION = "phase1_public_scorer_v1"
SHARED_FINAL_ANSWER_INSTRUCTION = (
    "Return JSON only with keys raw_text and normalized_prediction. "
    "The normalized_prediction value must be the final answer string."
)


@dataclass(frozen=True)
class PublicChartTask:
    sample_id: str
    image_path: str
    question: str
    answer: str
    answer_type: str
    metadata: dict[str, Any]


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_payload_hash(payload: dict[str, Any]) -> str:
    return sha256_hex_bytes(canonical_json_bytes(payload))


def stable_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_answer(answer: Any, answer_type: str) -> Any:
    if answer_type not in SUPPORTED_ANSWER_TYPES:
        raise ValueError(f"unsupported answer_type: {answer_type}")
    text = str(answer).strip()
    if answer_type == "numeric":
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text.replace(",", ""))
        if not match:
            raise ValueError(f"numeric answer has no parseable number: {answer!r}")
        return float(match.group(0))
    if answer_type in {"categorical", "text"}:
        return re.sub(r"\s+", " ", text).strip(" \t\r\n.,;:!?").lower()
    return text


def parse_metadata(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    parsed = raw if isinstance(raw, dict) else json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("metadata must be a JSON object")
    return parsed


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _image_readable(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def infer_chart_type_from_question(question: str) -> str | None:
    q = question.lower()
    exclusion_markers = {
        "pie": ("pie chart", "pie graph", "pie-chart", "slice of the pie"),
        "infographic": ("infographic",),
        "multi_panel": ("multi-panel", "multiple panels", "subplot"),
    }
    for chart_type, markers in exclusion_markers.items():
        if any(marker in q for marker in markers):
            return chart_type
    return None


def infer_reasoning_type(question: str) -> str:
    q = question.lower()
    ordered_rules = [
        ("ratio", ("ratio", "proportion", "times as", "percentage of", "percent of", "fraction of")),
        ("slope", ("slope", "rate of change", "increase per", "decrease per", "per year")),
        ("range", ("range", "difference between highest and lowest", "max minus min", "spread")),
        ("extrema", ("highest", "lowest", "maximum", "minimum", "peak", "largest", "smallest", "most", "least")),
        (
            "comparison",
            (
                "more than",
                "less than",
                "greater than",
                "fewer than",
                "difference between",
                "compare",
                "which is higher",
                "which is lower",
            ),
        ),
        ("trend", ("trend", "increase", "decrease", "overall", "change over time", "from start to end")),
    ]
    for label, markers in ordered_rules:
        if any(marker in q for marker in markers):
            return label
    if any(marker in q for marker in ("what is", "how many", "how much", "value", "amount", "number", "at ", "in ")):
        return "lookup"
    if any(marker in q for marker in ("legend", "title", "category", "label")):
        return "text"
    return "unknown"


def infer_answer_derivation(reasoning_type: str, question: str) -> str:
    q = question.lower()
    if reasoning_type in {"range", "slope", "ratio", "trend"}:
        return "derived"
    if reasoning_type == "comparison":
        derived_markers = (
            "difference",
            "how many more",
            "how much more",
            "how many fewer",
            "how much fewer",
            "minus",
            "between",
        )
        if any(marker in q for marker in derived_markers):
            return "derived"
        if any(marker in q for marker in ("which is higher", "which is lower", "which is greater", "which is smaller", "more", "less")):
            return "direct"
        return "unknown"
    if reasoning_type in {"lookup", "extrema"}:
        return "direct"
    return "unknown"


def enrich_metadata(task: PublicChartTask, manifest_dir: Path) -> dict[str, Any]:
    metadata = dict(task.metadata)
    image_path = Path(task.image_path)
    resolved_image = image_path if image_path.is_absolute() else manifest_dir / image_path
    reasoning_type = str(metadata.get("reasoning_type") or infer_reasoning_type(task.question))
    answer_derivation = str(metadata.get("answer_derivation") or infer_answer_derivation(reasoning_type, task.question))
    quantitative = reasoning_type in SUPPORTED_REASONING_TYPES
    image_readable = _coerce_bool(metadata.get("image_readable"))
    if image_readable is None:
        image_readable = _image_readable(resolved_image)
    axis_visible = _coerce_bool(metadata.get("axis_visible"))
    chart_type = str(metadata.get("chart_type") or "unknown")
    question_chart_type = infer_chart_type_from_question(task.question)
    if question_chart_type in EXCLUDED_CHART_TYPES:
        chart_type = question_chart_type

    required_unknown = chart_type == "unknown" or axis_visible is None or reasoning_type == "unknown"
    if (
        image_readable is True
        and axis_visible is True
        and chart_type in SUPPORTED_CHART_TYPES
        and reasoning_type in SUPPORTED_REASONING_TYPES
    ):
        compatibility = "presenter_candidate"
        selection_note = "eligible"
    elif required_unknown:
        compatibility = "unknown"
        selection_note = "selector_insufficient"
    else:
        compatibility = "incompatible"
        if chart_type in EXCLUDED_CHART_TYPES:
            selection_note = "unsupported_chart_type"
        elif axis_visible is False:
            selection_note = "axis_unreadable"
        elif image_readable is False:
            selection_note = "image_export_failed"
        elif reasoning_type == "text":
            selection_note = "text_only_answer"
        else:
            selection_note = "fixed_filter_excluded"

    metadata.update(
        {
            "benchmark": metadata.get("benchmark", "ChartQA"),
            "split": metadata.get("split", "test"),
            "chart_type": chart_type,
            "reasoning_type": reasoning_type,
            "answer_derivation": answer_derivation,
            "quantitative_reasoning_required": quantitative,
            "numeric_evidence_required": quantitative,
            "image_readable": image_readable,
            "axis_visible": axis_visible if axis_visible is not None else "unknown",
            "geosym_compatibility": compatibility,
            "selection_note": selection_note,
        }
    )
    return metadata


def coerce_task(row: dict[str, Any], manifest_dir: Path) -> PublicChartTask:
    required = ("sample_id", "image_path", "question", "answer", "answer_type")
    missing = [field for field in required if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    answer_type = str(row["answer_type"]).strip()
    if answer_type not in SUPPORTED_ANSWER_TYPES:
        raise ValueError(f"unsupported answer_type for {row['sample_id']}: {answer_type}")
    image_path = Path(str(row["image_path"]).strip())
    resolved_image = image_path if image_path.is_absolute() else manifest_dir / image_path
    if not resolved_image.exists():
        raise ValueError(f"image_path does not exist for {row['sample_id']}: {resolved_image}")
    normalize_answer(row["answer"], answer_type)
    base = PublicChartTask(
        sample_id=str(row["sample_id"]).strip(),
        image_path=str(image_path),
        question=str(row["question"]).strip(),
        answer=str(row["answer"]).strip(),
        answer_type=answer_type,
        metadata=parse_metadata(row.get("metadata")),
    )
    return PublicChartTask(base.sample_id, base.image_path, base.question, base.answer, base.answer_type, enrich_metadata(base, manifest_dir))


def read_manifest(path: Path) -> list[PublicChartTask]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("manifest must be .jsonl or .csv")
    tasks = [coerce_task(row, path.parent) for row in rows]
    sample_ids = [task.sample_id for task in tasks]
    duplicates = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate sample_id values: {duplicates}")
    if not tasks:
        raise ValueError("manifest contains no tasks")
    return tasks


def _hash_payload(task: PublicChartTask) -> dict[str, str]:
    return {
        "image_sha256": str(task.metadata.get("image_sha256") or ""),
        "question_sha256": str(task.metadata.get("question_sha256") or sha256_hex_bytes(task.question.encode("utf-8"))),
        "seed": DEFAULT_SEED,
        "source_index": str(task.metadata.get("source_index", task.metadata.get("original_index", task.sample_id))),
    }


def candidate_sort_key(task: PublicChartTask) -> tuple[str, str]:
    return stable_payload_hash(_hash_payload(task)), task.sample_id


def selection_sort_key(task: PublicChartTask) -> tuple[str, str]:
    return stable_payload_hash(_hash_payload(task)), task.sample_id


def build_candidate_pool(tasks: list[PublicChartTask], limit: int) -> list[PublicChartTask]:
    return sorted(tasks, key=candidate_sort_key)[:limit]


def eligible_for_phase1(task: PublicChartTask) -> bool:
    metadata = task.metadata
    return (
        metadata.get("chart_type") in SUPPORTED_CHART_TYPES
        and metadata.get("axis_visible") is True
        and metadata.get("image_readable") is True
        and metadata.get("geosym_compatibility") == "presenter_candidate"
        and metadata.get("reasoning_type") in SUPPORTED_REASONING_TYPES
    )


def deterministic_roster(tasks: Iterable[PublicChartTask], count: int = DEFAULT_SAMPLE_COUNT) -> list[PublicChartTask]:
    eligible = sorted([task for task in tasks if eligible_for_phase1(task)], key=selection_sort_key)
    quantitative = [task for task in eligible if task.metadata.get("quantitative_reasoning_required") is True]
    if len(eligible) < count:
        raise ValueError(f"selector_insufficient: fewer than {count} eligible samples ({len(eligible)})")
    if len(quantitative) < 4:
        raise ValueError(f"selector_insufficient: fewer than 4 quantitative samples ({len(quantitative)})")
    selected = quantitative[:4]
    selected_ids = {task.sample_id for task in selected}
    for task in eligible:
        if len(selected) >= count:
            break
        if task.sample_id not in selected_ids:
            selected.append(task)
            selected_ids.add(task.sample_id)
    return selected


def e0_prompt(question: str) -> str:
    return f"{SHARED_FINAL_ANSWER_INSTRUCTION}\nUse the image to answer the question.\nQuestion: {question}"


def e1_prompt(question: str) -> str:
    return (
        f"{SHARED_FINAL_ANSWER_INSTRUCTION}\nUse the image to answer the question. "
        "Include brief reasoning inside raw_text only.\n"
        f"Question: {question}"
    )


def g0_extract_prompt(question: str) -> str:
    return (
        "Return JSON only with key raw_observations. Each observation must have fact_id, source_step, text, "
        "numeric_entities, and confidence. Use source_step='extract'. Do not answer the question.\n"
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


def prompt_hashes() -> dict[str, str]:
    prompts = {
        "shared_final_answer_instruction": SHARED_FINAL_ANSWER_INSTRUCTION,
        "e0_prompt_template": e0_prompt("{question}"),
        "e1_prompt_template": e1_prompt("{question}"),
        "g0_extract_prompt_template": g0_extract_prompt("{question}"),
        "g0_curate_prompt_template": g0_curate_prompt("{question}", []),
        "g0_answer_prompt_template": g0_answer_prompt("{question}", []),
    }
    return {name: sha256_hex_bytes(text.encode("utf-8")) for name, text in prompts.items()}


def excluded_summary(tasks: Iterable[PublicChartTask]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        if eligible_for_phase1(task):
            continue
        note = str(task.metadata.get("selection_note", "unknown"))
        counts[note] = counts.get(note, 0) + 1
    return counts


def _sample_row(task: PublicChartTask) -> dict[str, Any]:
    return {
        "sample_id": task.sample_id,
        "image_path": task.image_path,
        "question": task.question,
        "answer": task.answer,
        "answer_type": task.answer_type,
        "normalized_answer": normalize_answer(task.answer, task.answer_type),
        "metadata": task.metadata,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.manifest).resolve()
    tasks = read_manifest(manifest)
    manifest_is_full_split = all((task.metadata.get("full_split_export") is True) for task in tasks)
    if not manifest_is_full_split and not args.allow_source_prefix_smoke:
        raise ValueError(
            "source_prefix_manifest: manifest is not marked as a full split export; "
            "rerun export without --limit or pass --allow-source-prefix-smoke for a non-paid diagnostic plan"
        )
    candidate_pool = build_candidate_pool(tasks, 100)
    expanded = False
    try:
        selected = deterministic_roster(candidate_pool, args.max_samples)
    except ValueError:
        candidate_pool = build_candidate_pool(tasks, 200)
        expanded = True
        selected = deterministic_roster(candidate_pool, args.max_samples)
    conditions = [condition.strip() for condition in args.conditions.split(",") if condition.strip()]
    unsupported = sorted(set(conditions) - set(DEFAULT_CONDITIONS))
    if unsupported:
        raise ValueError(f"unsupported Phase 1 condition names: {unsupported}")
    if args.max_samples != DEFAULT_SAMPLE_COUNT:
        raise ValueError("Phase 1 public smoke requires exactly 8 samples")

    sample_condition_jobs = len(selected) * len(conditions)
    llm_attempt_cap = len(selected) * sum(3 if condition == "G0_public_evidence_trace" else 1 for condition in conditions)
    if sample_condition_jobs > args.max_paid_jobs:
        raise ValueError(
            f"selected samples x conditions = {sample_condition_jobs}, exceeding --max-paid-jobs {args.max_paid_jobs}"
        )
    if llm_attempt_cap > args.max_llm_attempts:
        raise ValueError(f"LLM attempt cap {llm_attempt_cap} exceeds --max-llm-attempts {args.max_llm_attempts}")

    selected_ids = [task.sample_id for task in selected]
    sampling_frame = "full_split_hash_pool" if manifest_is_full_split else "source_prefix_diagnostic"
    frozen = {
        "hypothesis": args.hypothesis,
        "sample_selection": {
            "seed": DEFAULT_SEED,
            "candidate_pool_rule": "first 100 rows after deterministic hash sort; expand once to 200 if needed",
            "sampling_frame": sampling_frame,
            "full_split_export": manifest_is_full_split,
            "candidate_pool_size": len(candidate_pool),
            "expanded_to_200": expanded,
            "selected_ids": selected_ids,
            "selection_filter": {
                "chart_type": sorted(SUPPORTED_CHART_TYPES),
                "axis_visible": True,
                "geosym_compatibility": "presenter_candidate",
                "reasoning_type": sorted(SUPPORTED_REASONING_TYPES),
                "minimum_quantitative_reasoning_required": 4,
            },
        },
        "conditions": conditions,
        "reasoning_model": args.model,
        "temperature": 0.0,
        "decoding": "provider default except temperature 0 when adapter exposes it",
        "prompt_hashes": prompt_hashes(),
        "scorer_version": SCORER_VERSION,
        "sample_condition_job_cap": sample_condition_jobs,
        "llm_http_attempt_cap": llm_attempt_cap,
        "ocr_http_attempt_cap": 0,
        "per_condition_attempt_caps": {
            "E0_public_direct": {"llm": 1, "ocr": 0},
            "E1_public_cot": {"llm": 1, "ocr": 0},
            "G0_public_evidence_trace": {"llm": 3, "ocr": 0},
        },
        "retry_policy": "no retry expansion; one counted attempt per external request step",
        "cache_policy": "cache key includes sample_id, condition, model ID, prompt hashes, OCR flag, plan hash, and scorer version",
        "stop_rule": "stop after the frozen 8-sample roster and the frozen condition list; no adaptive replacement",
        "cost_estimate": "at most 24 sample-condition jobs and 40 LLM HTTP attempts; OCR disabled",
        "interpretation_limits": [
            "compatibility and attribution pilot only",
            "not a broad public-benchmark performance claim",
            "proxy attribution labels are diagnostic and do not prove true causal module failure",
            "G0 answer is text-only over retained facts, so image-access asymmetry is an interpretation limit",
            "source-prefix manifests are diagnostics only and are not approved for paid execution",
        ],
    }
    plan = {
        "schema_version": "chartqa_g0_smoke_plan_v1",
        "benchmark": args.benchmark,
        "split": "test",
        "created_date": args.created_date,
        "manifest_path": str(manifest),
        "manifest_sha256": stable_file_hash(manifest),
        "candidate_pool_sha256": sha256_hex_bytes("\n".join(task.sample_id for task in candidate_pool).encode("utf-8")),
        "zero_cost_only": True,
        "paid_run_approved": manifest_is_full_split,
        "paid_calls_executed": 0,
        "frozen_before_paid_run": frozen,
        "selected_samples": [_sample_row(task) for task in selected],
        "excluded_summary": excluded_summary(candidate_pool),
    }
    plan["plan_sha256"] = sha256_hex_bytes(
        json.dumps({k: v for k, v in plan.items() if k != "plan_sha256"}, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
    )
    return plan


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Local .jsonl or .csv ChartQA manifest.")
    parser.add_argument("--benchmark", default="ChartQA")
    parser.add_argument("--output", required=True, help="Frozen plan JSON output path.")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--max-paid-jobs", type=int, default=24)
    parser.add_argument("--max-llm-attempts", type=int, default=40)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--created-date", default=date.today().isoformat())
    parser.add_argument(
        "--allow-source-prefix-smoke",
        action="store_true",
        help="Allow a non-paid diagnostic plan from a source-prefix manifest. Never approves paid execution.",
    )
    parser.add_argument(
        "--hypothesis",
        default=(
            "An 8-sample ChartQA smoke can test whether a GeoSym-style evidence trace interface runs on "
            "public chart-QA data and yields useful proxy attribution without broad performance claims."
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
    print(f"wrote frozen zero-cost plan: {output}")
    print(f"selected_samples={len(plan['selected_samples'])}")
    print(f"sample_condition_job_cap={frozen['sample_condition_job_cap']}")
    print(f"llm_http_attempt_cap={frozen['llm_http_attempt_cap']}")
    print("paid_calls_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
