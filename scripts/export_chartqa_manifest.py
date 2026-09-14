#!/usr/bin/env python
"""Export ChartQA split rows to a local PNG-backed manifest.

This is a zero-cost data-preparation utility. It does not call any model, OCR
service, or scorer. If dataset metadata lacks chart-type or axis-visibility
fields, those tags are left unknown for ``prepare_public_benchmark_pilot.py`` to
fail closed before paid execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_public_benchmark_pilot import canonical_json_bytes, sha256_hex_bytes  # noqa: E402


DEFAULT_SOURCE = "HuggingFaceM4/ChartQA"


def _first_present(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _first_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return _first_scalar(value[0])
    return value


def _answer_type(answer: Any) -> str:
    text = str(_first_scalar(answer)).strip().replace(",", "")
    try:
        float(text)
        return "numeric"
    except ValueError:
        return "text"


def _metadata_value(row: dict[str, Any], names: Iterable[str]) -> Any:
    value = _first_present(row, names)
    if value is None:
        return "unknown"
    if isinstance(value, str):
        return value.strip().lower().replace(" ", "_")
    return value


def _is_full_split_export(limit: int, stopped_by_limit: bool) -> bool:
    return limit == 0 and not stopped_by_limit


def _save_png(image_value: Any, path: Path) -> bytes:
    if isinstance(image_value, Image.Image):
        image = image_value.convert("RGB")
    elif isinstance(image_value, dict) and image_value.get("bytes"):
        image = Image.open(BytesIO(image_value["bytes"])).convert("RGB")
    elif isinstance(image_value, (str, Path)):
        image = Image.open(image_value).convert("RGB")
    else:
        raise ValueError(f"unsupported image field type: {type(image_value).__name__}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    path.write_bytes(data)
    return data


def _local_sample_id(source_index: int, question_sha256: str, image_sha256: str) -> str:
    payload = {
        "image_sha256": image_sha256,
        "question_sha256": question_sha256,
        "seed": "20260828",
        "source_index": str(source_index),
    }
    return f"chartqa_test_{source_index}_{sha256_hex_bytes(canonical_json_bytes(payload))[:8]}"


def export_manifest(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets package is required to export ChartQA") from exc

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.source, split=args.split, streaming=args.streaming)
    rows = []
    stopped_by_limit = False
    for source_index, row in enumerate(dataset):
        if args.limit and source_index >= args.limit:
            stopped_by_limit = True
            break
        question = _first_present(row, ("query", "question", "Question"))
        answer = _first_scalar(_first_present(row, ("label", "answer", "Answer")))
        image_value = _first_present(row, ("image", "img", "png"))
        if question is None or answer is None or image_value is None:
            continue
        question_text = str(question)
        question_hash = hashlib.sha256(question_text.encode("utf-8")).hexdigest()
        provisional_image = image_dir / f"chartqa_test_{source_index:06d}.png"
        image_bytes = _save_png(image_value, provisional_image)
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        sample_id = _local_sample_id(source_index, question_hash, image_hash)
        final_image = image_dir / f"{sample_id}.png"
        if final_image != provisional_image:
            provisional_image.replace(final_image)
        metadata = {
            "benchmark": "ChartQA",
            "source": args.source,
            "split": args.split,
            "dataset_export_date": args.export_date,
            "full_split_export": not bool(args.limit),
            "source_row_limit": args.limit or None,
            "source_prefix_export": bool(args.limit),
            "source_index": str(source_index),
            "question_sha256": question_hash,
            "image_sha256": image_hash,
            "chart_type": _metadata_value(row, ("chart_type", "chartType", "type")),
            "axis_visible": _metadata_value(row, ("axis_visible", "axisVisible")),
        }
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": str(final_image.relative_to(output_dir)),
                "question": question_text,
                "answer": str(answer),
                "answer_type": _answer_type(answer),
                "metadata": metadata,
            }
        )

    manifest_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    full_split_export = _is_full_split_export(args.limit, stopped_by_limit)
    summary = {
        "schema_version": "chartqa_manifest_export_v1",
        "source": args.source,
        "split": args.split,
        "streaming": args.streaming,
        "export_date": args.export_date,
        "full_split_export": full_split_export,
        "source_prefix_export": stopped_by_limit,
        "source_row_limit": args.limit or None,
        "row_count": len(rows),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "paid_calls_executed": 0,
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="0 means no explicit row limit.")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--export-date", default=date.today().isoformat())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        summary = export_manifest(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
