#!/usr/bin/env python
"""Rescore persisted reasoning outputs from stored raw responses.

This is a zero-cost repair tool: it never calls an LLM, OCR service, or API key.
It rebuilds parsed answers and Judge scores from persisted raw text, then reruns
the local deterministic scorer.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reasoning_cache import load_reasoning_cache  # noqa: E402
from app.reasoning_pipeline import build_chain_text, load_layer_config  # noqa: E402
from evaluator.metrics.reasoning import evaluate_comprehensive  # noqa: E402
from evaluator.reasoning.parser import (  # noqa: E402
    parse_L1_answer,
    parse_L2_answer,
    parse_L3_answer,
    parse_comprehensive_answer,
    parse_judge_response,
)
from evaluator.metrics.reasoning import evaluate_judge_response  # noqa: E402
from scripts.batch_experiment import ALL_CONDITIONS  # noqa: E402
from scripts.prepare_public_benchmark_pilot import canonical_json_bytes, sha256_hex_bytes  # noqa: E402


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def sample_image_name(row: dict[str, Any]) -> str:
    image_name = str(row.get("image_name") or "")
    if "/" in image_name:
        return image_name
    family = str(row["family"])
    sample_id = str(row["sample_id"])
    return f"{family}-clean_render/{sample_id}"


def is_all_zero_success(row: dict[str, Any]) -> bool:
    dims = row.get("dimension_scores") or {}
    if row.get("status") != "success" or row.get("total_score") not in {0, 0.0}:
        return False
    return bool(dims) and all(float(value or 0) == 0.0 for value in dims.values())


def extract_step_raw(pipeline: dict[str, Any], *names: str) -> str:
    steps = pipeline.get("steps") or {}
    for name in names:
        raw = (steps.get(name) or {}).get("raw")
        if isinstance(raw, str) and raw.strip():
            return raw
    if len(steps) == 1:
        only = next(iter(steps.values()))
        raw = only.get("raw") if isinstance(only, dict) else None
        if isinstance(raw, str):
            return raw
    return ""


def layer_step_names(pipeline: dict[str, Any]) -> set[str]:
    steps = pipeline.get("steps") or {}
    return {str(name) for name in steps.keys()} & {"L1", "L2", "L3"}


def rebuild_parsed(pipeline: dict[str, Any]) -> tuple[dict[str, Any], str]:
    pipeline_type = str(pipeline.get("pipeline_type") or "")
    if layer_step_names(pipeline):
        l1 = parse_L1_answer(extract_step_raw(pipeline, "L1", "Step 1", "Reading", "Reading the Graph"))
        l2 = parse_L2_answer(extract_step_raw(pipeline, "L2", "Step 2", "Quantitative Analysis"))
        l3 = parse_L3_answer(extract_step_raw(pipeline, "L3", "Step 3", "Reasoning"))
        parsed = {}
        parsed.update(l1)
        parsed.update(l2)
        parsed.update(l3)
        return parsed, "layer_raw_reparse"

    if pipeline_type in {"single_step", "self_consistency"}:
        raw = extract_step_raw(pipeline, "DirectQA", "CoT", "OCRBaseline", "SelfConsistency")
        return parse_comprehensive_answer(raw), "single_step_raw_reparse"

    if pipeline_type in {"reasoning_flow", "multi_step"}:
        l1 = parse_L1_answer(extract_step_raw(pipeline, "L1", "Step 1", "Reading", "Reading the Graph"))
        l2 = parse_L2_answer(extract_step_raw(pipeline, "L2", "Step 2", "Quantitative Analysis"))
        l3 = parse_L3_answer(extract_step_raw(pipeline, "L3", "Step 3", "Reasoning"))
        parsed = {}
        parsed.update(l1)
        parsed.update(l2)
        parsed.update(l3)
        return parsed, "layer_raw_reparse"

    final = pipeline.get("final_parsed")
    if isinstance(final, dict):
        return final, "stored_final_parsed_fallback"
    return {}, "missing_pipeline"


def rebuild_judge(cond: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    judge = cond.get("judge") or {}
    raw = judge.get("raw_judge_response")
    if isinstance(raw, str) and raw.strip():
        return evaluate_judge_response(parse_judge_response(raw)), "raw_judge_reparse"
    if judge and "judge_inference_score" in judge:
        return judge, "stored_judge_fallback"
    return None, "missing_judge"


BARE_INFINITY_RE = re.compile(r'(?<![\w])([+-]?(?:inf|infinity|Infinity|INF))(?![\w])')
JSON_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)


def candidate_json_objects(text: str) -> list[str]:
    candidates = []
    for match in re.finditer(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL):
        candidates.append(match.group(1).strip())

    depth = 0
    start = None
    for idx, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:idx + 1])
                start = None
    return candidates


def strict_json_parseable(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    for candidate in candidate_json_objects(text):
        try:
            loaded = json.loads(
                candidate,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
            if isinstance(loaded, dict):
                return True
        except (json.JSONDecodeError, ValueError):
            continue
    return False


def has_bare_infinity_token(text: str) -> bool:
    return any(BARE_INFINITY_RE.search(JSON_STRING_RE.sub('""', candidate)) for candidate in candidate_json_objects(text))


def raw_reasoning_outputs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    steps = pipeline.get("steps") or {}
    outputs = []
    for step_name, step in steps.items():
        if isinstance(step, dict) and isinstance(step.get("raw"), str) and step["raw"].strip():
            outputs.append({"step": str(step_name), "raw": step["raw"]})
    if pipeline.get("pipeline_type") == "self_consistency":
        samples = pipeline.get("samples") or []
        sample_outputs = [
            {"step": f"SelfConsistency.sample_{idx + 1}", "raw": sample.get("raw")}
            for idx, sample in enumerate(samples)
            if isinstance(sample, dict) and isinstance(sample.get("raw"), str) and sample.get("raw", "").strip()
        ]
        if sample_outputs:
            outputs = sample_outputs
    return outputs


def build_raw_json_parse_audit_from_conditions(condition_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_condition_step: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for item in condition_records:
        condition = str(item.get("condition"))
        cond = item.get("condition_record") or {}
        pipeline = cond.get("pipeline") or {}
        for output in raw_reasoning_outputs(pipeline):
            raw = output["raw"]
            record = {
                "sample_id": item.get("sample_id"),
                "step": output["step"],
                "has_bare_infinity": has_bare_infinity_token(raw),
                "strict_json_parseable": strict_json_parseable(raw),
            }
            by_condition[condition].append(record)
            by_condition_step[condition][output["step"]].append(record)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        bare = sum(1 for item in items if item["has_bare_infinity"])
        failed = sum(1 for item in items if not item["strict_json_parseable"])
        return {
            "raw_outputs": total,
            "bare_infinity_outputs": bare,
            "bare_infinity_rate": round(bare / total, 6) if total else None,
            "strict_json_fail_outputs": failed,
            "strict_json_fail_rate": round(failed / total, 6) if total else None,
        }

    return {
        "unit": "persisted_reasoning_raw_output",
        "strict_json_definition": "At least one JSON object or fenced JSON block in the raw output must load with json.loads without parser repairs.",
        "bare_infinity_definition": "Regex match for unquoted inf/infinity tokens in the raw output; this is diagnostic and may include explanatory prose.",
        "by_condition": {condition: summarize(items) for condition, items in sorted(by_condition.items())},
        "by_condition_step": {
            condition: {step: summarize(items) for step, items in sorted(steps.items())}
            for condition, steps in sorted(by_condition_step.items())
        },
    }


def rescore_condition(cond: dict[str, Any], gt: dict[str, Any], layer_config: dict[str, Any]) -> dict[str, Any]:
    if cond.get("status") != "success":
        return {
            "condition": cond.get("condition_id"),
            "status": cond.get("status"),
            "rescore_status": "skipped_non_success",
            "row": None,
            "parsed": cond.get("parsed"),
            "judge": cond.get("judge"),
            "eval": cond.get("eval"),
        }

    pipeline = cond.get("pipeline") or {}
    parsed, parsed_source = rebuild_parsed(pipeline)
    judge, judge_source = rebuild_judge(cond)
    evaluation = evaluate_comprehensive(parsed, gt, layer_config, judge)
    chain_text = build_chain_text(pipeline) if pipeline else ""
    raw_judge = (cond.get("judge") or {}).get("raw_judge_response") or ""
    return {
        "condition": cond.get("condition_id"),
        "status": cond.get("status"),
        "rescore_status": "rescored_success",
        "parsed_source": parsed_source,
        "judge_source": judge_source,
        "chain_text_sha256": sha256_hex_bytes(canonical_json_bytes(chain_text)) if chain_text else None,
        "raw_judge_response_sha256": sha256_hex_bytes(canonical_json_bytes(raw_judge)) if raw_judge else None,
        "parsed": parsed,
        "judge": judge,
        "eval": evaluation,
    }


def condition_row_from_cache(source_row: dict[str, Any], rescored: dict[str, Any]) -> dict[str, Any]:
    ev = rescored.get("eval") or {}
    row = dict(source_row)
    row["total_score_before_parser_repair"] = source_row.get("total_score")
    row["dimension_scores_before_parser_repair"] = source_row.get("dimension_scores")
    row["judge_inference_score_before_parser_repair"] = source_row.get("judge_inference_score")
    if ev:
        row["total_score"] = ev.get("total_score")
        row["dimension_scores"] = ev.get("dimension_scores")
        row["judge_inference_score"] = (ev.get("sub_scores") or {}).get("judge_inference_score", {}).get("score")
    if rescored.get("chain_text_sha256"):
        row["chain_text_sha256"] = rescored.get("chain_text_sha256")
    if rescored.get("raw_judge_response_sha256"):
        row["raw_judge_response_sha256"] = rescored.get("raw_judge_response_sha256")
    row["parser_repair_rescore_status"] = rescored.get("rescore_status")
    row["parser_repair_parsed_source"] = rescored.get("parsed_source")
    row["parser_repair_judge_source"] = rescored.get("judge_source")
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family_condition: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_condition[str(row.get("condition"))].append(row)
        by_family_condition[str(row.get("family"))][str(row.get("condition"))].append(row)

    def row_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        scores = [float(row["total_score"]) for row in items if row.get("total_score") is not None and math.isfinite(float(row["total_score"]))]
        return {
            "rows": len(items),
            "success": sum(1 for row in items if row.get("status") == "success"),
            "mean_total_score": round(sum(scores) / len(scores), 6) if scores else None,
            "status_counts": dict(Counter(str(row.get("status")) for row in items)),
        }

    return {
        "overall_by_condition": {condition: row_summary(items) for condition, items in by_condition.items()},
        "by_family_condition": {
            family: {condition: row_summary(items) for condition, items in conds.items()}
            for family, conds in by_family_condition.items()
        },
    }


def build_repair_audit(original_rows: list[dict[str, Any]], repaired_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition_original: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_condition_repaired: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in original_rows:
        by_condition_original[str(row.get("condition"))].append(row)
    for row in repaired_rows:
        by_condition_repaired[str(row.get("condition"))].append(row)

    condition_audit = {}
    for condition in sorted(set(by_condition_original) | set(by_condition_repaired)):
        originals = by_condition_original.get(condition, [])
        repaired = by_condition_repaired.get(condition, [])
        changed = 0
        repaired_zero = 0
        for old, new in zip(originals, repaired):
            old_score = old.get("total_score")
            new_score = new.get("total_score")
            if old_score != new_score:
                changed += 1
            if is_all_zero_success(old) and not is_all_zero_success(new):
                repaired_zero += 1
        condition_audit[condition] = {
            "rows": len(repaired),
            "all_zero_success_before": sum(1 for row in originals if is_all_zero_success(row)),
            "all_zero_success_after": sum(1 for row in repaired if is_all_zero_success(row)),
            "all_zero_success_repaired": repaired_zero,
            "rows_with_total_score_changed": changed,
        }
    return {
        "condition_audit": condition_audit,
        "total_all_zero_success_before": sum(1 for row in original_rows if is_all_zero_success(row)),
        "total_all_zero_success_after": sum(1 for row in repaired_rows if is_all_zero_success(row)),
        "total_rows_with_total_score_changed": sum(
            1 for old, new in zip(original_rows, repaired_rows)
            if old.get("total_score") != new.get("total_score")
        ),
    }


def rescore_artifact(input_path: Path, output_path: Path, cache_model: str) -> dict[str, Any]:
    artifact = load_json(input_path)
    layer_config = load_layer_config()
    repaired_rows = []
    unresolved = []
    original_rows = list(artifact.get("results") or [])
    raw_audit_records = []
    for source_row in original_rows:
        cache = load_reasoning_cache(sample_image_name(source_row), cache_model)
        cond = ((cache or {}).get("conditions") or {}).get(str(source_row.get("condition")))
        if not cache or not cond:
            row = dict(source_row)
            row["parser_repair_rescore_status"] = "missing_cache"
            repaired_rows.append(row)
            unresolved.append({"sample_id": source_row.get("sample_id"), "condition": source_row.get("condition"), "reason": "missing_cache"})
            continue
        raw_audit_records.append({
            "sample_id": source_row.get("sample_id"),
            "condition": source_row.get("condition"),
            "condition_record": cond,
        })
        rescored = rescore_condition(cond, cache.get("ground_truth_snapshot") or {}, layer_config)
        repaired_rows.append(condition_row_from_cache(source_row, rescored))

    payload = dict(artifact)
    payload.update({
        "schema_version": f"{artifact.get('schema_version', 'reasoning_artifact')}_parser_repair_rescore_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": f"{artifact.get('stage', 'reasoning_artifact')}_parser_repair_rescore",
        "stage_status": "parser_repair_offline_rescore_complete" if not unresolved else "parser_repair_offline_rescore_incomplete",
        "parser_repair": {
            "no_paid_calls": True,
            "input_path": str(input_path.resolve()),
            "cache_model": cache_model,
            "repairs": [
                "strict JSON parse first; after failure repair bare infinity tokens",
                "strict JSON parse first; after failure repair invalid JSON escapes such as LaTeX backslashes",
                "accept two-bound monotonicity interval strings",
            ],
            **build_repair_audit(original_rows, repaired_rows),
            "raw_json_parse_audit": build_raw_json_parse_audit_from_conditions(raw_audit_records),
            "unresolved_rows": unresolved,
        },
        "summary": summarize_rows(repaired_rows),
        "results": repaired_rows,
    })
    write_json(output_path, payload)
    return payload


def iter_archive_records(archive_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(archive_root.glob("*/*.json")):
        if path.name == "archive_manifest.json":
            continue
        payload = load_json(path)
        if isinstance(payload, dict) and payload.get("conditions"):
            payload["_source_archive_path"] = str(path)
            records.append(payload)
    return records


def rows_from_archive_record(record: dict[str, Any], layer_config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    image_name = str(record.get("image_name") or "")
    sample_id = Path(image_name).stem
    family = sample_id.split("-clean_render_", 1)[0] if "-clean_render_" in sample_id else image_name.split("/", 1)[0].replace("-clean_render", "")
    gt = record.get("ground_truth_snapshot") or {}
    for condition in ALL_CONDITIONS:
        cond = (record.get("conditions") or {}).get(condition) or {}
        source_row = {
            "sample_id": sample_id,
            "image_name": image_name,
            "family": family,
            "condition": condition,
            "status": cond.get("status"),
            "cached": cond.get("cached", False),
            "total_score": (cond.get("eval") or {}).get("total_score"),
            "dimension_scores": (cond.get("eval") or {}).get("dimension_scores"),
            "judge_inference_score": ((cond.get("eval") or {}).get("sub_scores") or {}).get("judge_inference_score", {}).get("score"),
            "error": cond.get("error"),
            "source_archive_path": record.get("_source_archive_path"),
        }
        rescored = rescore_condition(cond, gt, layer_config)
        rows.append(condition_row_from_cache(source_row, rescored))
    return rows


def rescore_archive_root(archive_root: Path, output_path: Path) -> dict[str, Any]:
    layer_config = load_layer_config()
    records = iter_archive_records(archive_root)
    original_rows = []
    repaired_rows = []
    raw_audit_records = []
    for record in records:
        for condition in ALL_CONDITIONS:
            cond = (record.get("conditions") or {}).get(condition) or {}
            sample_id = Path(str(record.get("image_name") or "")).stem
            original_rows.append({
                "sample_id": sample_id,
                "image_name": record.get("image_name"),
                "family": str(record.get("image_name") or "").split("/", 1)[0].replace("-clean_render", ""),
                "condition": condition,
                "status": cond.get("status"),
                "total_score": (cond.get("eval") or {}).get("total_score"),
                "dimension_scores": (cond.get("eval") or {}).get("dimension_scores"),
            })
            raw_audit_records.append({
                "sample_id": sample_id,
                "condition": condition,
                "condition_record": cond,
            })
        repaired_rows.extend(rows_from_archive_record(record, layer_config))

    payload = {
        "schema_version": "reasoning_archive_parser_repair_rescore_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "reasoning_archive_parser_repair_rescore",
        "stage_status": "parser_repair_offline_rescore_complete",
        "archive_root": str(archive_root.resolve()),
        "no_paid_calls": True,
        "records": len(records),
        "conditions": list(ALL_CONDITIONS),
        "parser_repair": {
            **build_repair_audit(original_rows, repaired_rows),
            "raw_json_parse_audit": build_raw_json_parse_audit_from_conditions(raw_audit_records),
        },
        "summary": summarize_rows(repaired_rows),
        "results": repaired_rows,
    }
    write_json(output_path, payload)
    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-artifact", type=Path)
    group.add_argument("--archive-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-model", default="GPT-5.5")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.input_artifact:
        payload = rescore_artifact(args.input_artifact, args.output, args.cache_model)
    else:
        payload = rescore_archive_root(args.archive_root, args.output)
    audit = payload.get("parser_repair") or {}
    print(f"wrote={args.output}")
    print(f"stage_status={payload.get('stage_status')}")
    print(f"total_all_zero_success_before={audit.get('total_all_zero_success_before')}")
    print(f"total_all_zero_success_after={audit.get('total_all_zero_success_after')}")
    print(f"total_rows_with_total_score_changed={audit.get('total_rows_with_total_score_changed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
