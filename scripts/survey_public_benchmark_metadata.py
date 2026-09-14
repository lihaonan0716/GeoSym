#!/usr/bin/env python
"""Zero-cost metadata survey for candidate public chart benchmarks.

The survey reads public dataset metadata or local manifests only. It does not
call LLMs, OCR services, judges, or paid APIs. The goal is to decide whether a
benchmark can support a frozen 8-sample GeoSym-compatible public smoke before
any paid execution is considered.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_CHART_TYPES = {"line", "scatter", "simple_bar", "grouped_bar", "bar"}
PLOTQA_HF_MIRRORS = ("achang/plot_qa", "martinsinnona/plotqa")


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    rank: int
    official_sources: tuple[str, ...]
    official_schema_summary: dict[str, bool]
    notes: str


BENCHMARK_SPECS = [
    BenchmarkSpec(
        name="PlotQA",
        rank=1,
        official_sources=(
            "https://github.com/NiteshMethani/PlotQA/blob/master/PlotQA_Dataset.md",
            "https://arxiv.org/abs/1909.00997",
        ),
        official_schema_summary={
            "chart_type": True,
            "x_axis": True,
            "y_axis": True,
            "tick_labels": True,
            "legend": True,
            "plot_bbox": True,
            "data_table": True,
            "qa_answer_type": True,
        },
        notes="Best first target if the official JSON export is available locally; HF mirrors may not preserve QA fields.",
    ),
    BenchmarkSpec(
        name="CHART-Info / ICPR CHART",
        rank=2,
        official_sources=("https://chartinfo.github.io/index_2020.html", "https://github.com/kdavila/CHART_Info_2024"),
        official_schema_summary={
            "chart_type": True,
            "x_axis": True,
            "y_axis": True,
            "tick_labels": True,
            "legend": True,
            "plot_bbox": True,
            "data_table": True,
            "qa_answer_type": False,
        },
        notes="Strong parsing/evidence diagnostic, but not a native QA benchmark.",
    ),
    BenchmarkSpec(
        name="DVQA",
        rank=3,
        official_sources=("https://github.com/kushalkafle/DVQA_dataset", "https://kushalkafle.com/projects/dvqa.html"),
        official_schema_summary={
            "chart_type": True,
            "x_axis": True,
            "y_axis": True,
            "tick_labels": True,
            "legend": True,
            "plot_bbox": True,
            "data_table": True,
            "qa_answer_type": True,
        },
        notes="Useful bar-chart structured evidence target; weaker external-validity breadth than PlotQA.",
    ),
    BenchmarkSpec(
        name="FigureQA",
        rank=4,
        official_sources=("https://github.com/maluuba/figureqa", "https://www.microsoft.com/en-us/research/project/figureqa-dataset/"),
        official_schema_summary={
            "chart_type": True,
            "x_axis": True,
            "y_axis": True,
            "tick_labels": True,
            "legend": True,
            "plot_bbox": True,
            "data_table": True,
            "qa_answer_type": True,
        },
        notes="Good low-cost sanity or negative-control candidate; many questions are relational yes/no.",
    ),
    BenchmarkSpec(
        name="ChartQA-X",
        rank=5,
        official_sources=("https://huggingface.co/datasets/shamanthakhegde/ChartQA-X", "https://arxiv.org/html/2504.13275v1"),
        official_schema_summary={
            "chart_type": True,
            "x_axis": False,
            "y_axis": False,
            "tick_labels": False,
            "legend": False,
            "plot_bbox": False,
            "data_table": False,
            "qa_answer_type": True,
        },
        notes="Potential chart-type/explanation diagnostic; axis-level metadata still needs local probe.",
    ),
]


def _contains_any(keys: set[str], candidates: Iterable[str]) -> bool:
    return any(candidate in keys for candidate in candidates)


def _normalize_chart_type(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"vbar", "hbar", "bar_chart", "bar", "vbar_categorical", "hbar_categorical"}:
        return "bar"
    if text in {"dot_line", "dotline"}:
        return "line"
    if text in {"scatter", "scatter_plot"}:
        return "scatter"
    if text:
        return text if text in SUPPORTED_CHART_TYPES else "other"
    return text


def _extract_tags(text: str, tag: str) -> list[str]:
    return re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)


def infer_schema_from_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = set(row)
    text = str(row.get("text", ""))
    literal_text = None
    if text.startswith("{") and text.endswith("}"):
        try:
            literal_text = ast.literal_eval(text)
        except Exception:
            literal_text = None

    chart_type = None
    for key in ("chart_type", "type", "mark", "plot_type"):
        if row.get(key):
            chart_type = _normalize_chart_type(row[key])
            break
    if chart_type is None and literal_text:
        chart_type = _normalize_chart_type(literal_text.get("mark", "unknown"))

    has_tagged_x = bool(_extract_tags(text, "s_x"))
    has_tagged_y = bool(_extract_tags(text, "s_y"))
    has_tagged_bbox = "<s_bboxes>" in text or "<s_w>" in text or "<s_h>" in text
    has_tagged_legend = any(marker in text for marker in ("<s_name>", "<s_label>", "<s_color>"))
    has_vega_encoding = isinstance(literal_text, dict) and isinstance(literal_text.get("encoding"), dict)
    has_vega_data = isinstance(literal_text, dict) and isinstance(literal_text.get("data"), dict)

    return {
        "has_image": _contains_any(keys, ("image", "img", "png", "image_path")),
        "has_question": _contains_any(keys, ("question", "query", "question_string", "qa_question")),
        "has_answer": _contains_any(keys, ("answer", "label", "qa_answer", "answers")),
        "has_chart_type": chart_type not in {None, "", "unknown"} or _contains_any(keys, ("chart_type", "type", "mark")),
        "chart_type": chart_type or "unknown",
        "has_x_axis": _contains_any(keys, ("x_axis", "xaxis", "x_label")) or has_tagged_x or has_vega_encoding,
        "has_y_axis": _contains_any(keys, ("y_axis", "yaxis", "y_label")) or has_tagged_y or has_vega_encoding,
        "has_tick_labels": _contains_any(keys, ("ticks", "tick_labels", "x_ticks", "y_ticks")) or has_tagged_x or has_tagged_y,
        "has_legend": _contains_any(keys, ("legend", "legend_info")) or has_tagged_legend,
        "has_plot_bbox": _contains_any(keys, ("plot_bbox", "bbox", "bboxes")) or has_tagged_bbox,
        "has_data_table": _contains_any(keys, ("plot_info", "figure_info", "data", "data_table")) or has_tagged_x or has_vega_data,
        "has_qa_answer_type": _contains_any(keys, ("answer_type", "question_type", "qa_type")) or (
            _contains_any(keys, ("question", "query")) and _contains_any(keys, ("answer", "label", "answers"))
        ),
        "raw_keys": sorted(keys),
    }


def is_geosym_compatible(row_schema: dict[str, Any], require_qa: bool) -> bool:
    chart_type = row_schema.get("chart_type")
    chart_ok = chart_type in SUPPORTED_CHART_TYPES
    required = [
        row_schema.get("has_image"),
        chart_ok,
        row_schema.get("has_x_axis"),
        row_schema.get("has_y_axis"),
        row_schema.get("has_tick_labels"),
        row_schema.get("has_data_table"),
    ]
    if require_qa:
        required.extend([row_schema.get("has_question"), row_schema.get("has_answer"), row_schema.get("has_qa_answer_type")])
    return all(required)


def summarize_rows(rows: list[dict[str, Any]], require_qa: bool) -> dict[str, Any]:
    schemas = [infer_schema_from_row(row) for row in rows]
    coverage_keys = [
        "has_image",
        "has_question",
        "has_answer",
        "has_chart_type",
        "has_x_axis",
        "has_y_axis",
        "has_tick_labels",
        "has_legend",
        "has_plot_bbox",
        "has_data_table",
        "has_qa_answer_type",
    ]
    return {
        "sampled_rows": len(rows),
        "raw_keys": sorted({key for row in rows for key in row}),
        "coverage": {key: sum(1 for schema in schemas if schema[key]) for key in coverage_keys},
        "chart_type_counts": dict(Counter(schema["chart_type"] for schema in schemas)),
        "eligible_count": sum(1 for schema in schemas if is_geosym_compatible(schema, require_qa=require_qa)),
        "qa_required": require_qa,
    }


def plotqa_annotation_schema(annotation: dict[str, Any], qa: dict[str, Any] | None, image_exists: bool) -> dict[str, Any]:
    figure = annotation.get("general_figure_info") or {}
    chart_type = _normalize_chart_type(annotation.get("type", "unknown"))
    if chart_type in {"vbar_categorical", "hbar_categorical"}:
        chart_type = "bar"
    if chart_type == "dot_line":
        chart_type = "line"
    schema = {
        "has_image": image_exists,
        "has_question": bool(qa and qa.get("question_string")),
        "has_answer": bool(qa and "answer" in qa),
        "has_chart_type": chart_type != "unknown",
        "chart_type": chart_type,
        "has_x_axis": isinstance(figure.get("x_axis"), dict),
        "has_y_axis": isinstance(figure.get("y_axis"), dict),
        "has_tick_labels": bool(
            isinstance(figure.get("x_axis"), dict)
            and figure["x_axis"].get("major_ticks")
            or isinstance(figure.get("y_axis"), dict)
            and figure["y_axis"].get("major_ticks")
        ),
        "has_legend": isinstance(figure.get("legend"), dict),
        "has_plot_bbox": isinstance(figure.get("plot_info"), dict) or isinstance(figure.get("figure_info"), dict),
        "has_data_table": bool(annotation.get("models")),
        "has_qa_answer_type": bool(qa and qa.get("type")),
        "raw_keys": sorted(set(annotation) | (set(qa) if qa else set())),
    }
    return schema


def summarize_plotqa_official(
    annotations_path: Path, qa_pairs_path: Path, image_dir: Path | None, limit: int, require_qa: bool
) -> dict[str, Any]:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    qa_pairs = json.loads(qa_pairs_path.read_text(encoding="utf-8"))
    if isinstance(qa_pairs, dict) and isinstance(qa_pairs.get("qa_pairs"), list):
        qa_pairs = qa_pairs["qa_pairs"]
    if not isinstance(annotations, list) or not isinstance(qa_pairs, list):
        raise ValueError("PlotQA annotations and qa_pairs must both be JSON lists")
    annotations_by_image = {str(annotation.get("image_index")): annotation for annotation in annotations}
    schemas = []
    missing_annotations = 0
    for qa in qa_pairs[: limit or None]:
        image_index = str(qa.get("image_index"))
        annotation = annotations_by_image.get(image_index)
        if annotation is None:
            missing_annotations += 1
            continue
        image_exists = True
        if image_dir is not None:
            image_exists = (image_dir / f"{image_index}.png").exists()
        schemas.append(plotqa_annotation_schema(annotation, qa, image_exists))
    coverage_keys = [
        "has_image",
        "has_question",
        "has_answer",
        "has_chart_type",
        "has_x_axis",
        "has_y_axis",
        "has_tick_labels",
        "has_legend",
        "has_plot_bbox",
        "has_data_table",
        "has_qa_answer_type",
    ]
    return {
        "source": "official_plotqa_json",
        "annotations_path": str(annotations_path),
        "qa_pairs_path": str(qa_pairs_path),
        "image_dir": str(image_dir) if image_dir else None,
        "sampled_rows": len(schemas),
        "annotation_count": len(annotations),
        "qa_pair_count": len(qa_pairs),
        "missing_annotation_count": missing_annotations,
        "coverage": {key: sum(1 for schema in schemas if schema[key]) for key in coverage_keys},
        "chart_type_counts": dict(Counter(schema["chart_type"] for schema in schemas)),
        "eligible_count": sum(1 for schema in schemas if is_geosym_compatible(schema, require_qa=require_qa)),
        "qa_required": require_qa,
    }


def probe_hf_dataset(dataset_id: str, split: str, limit: int, require_qa: bool) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        return {"dataset_id": dataset_id, "split": split, "available": False, "error": f"ImportError: {exc}"}
    try:
        ds = load_dataset(dataset_id, split=split, streaming=True)
        rows = []
        for idx, row in enumerate(ds):
            if idx >= limit:
                break
            rows.append(row)
        summary = summarize_rows(rows, require_qa=require_qa)
        return {"dataset_id": dataset_id, "split": split, "available": True, **summary}
    except Exception as exc:
        return {"dataset_id": dataset_id, "split": split, "available": False, "error": f"{type(exc).__name__}: {exc}"}


def survey(args: argparse.Namespace) -> dict[str, Any]:
    hf_probes = [] if args.skip_hf else [probe_hf_dataset(dataset_id, args.split, args.limit, args.require_qa) for dataset_id in PLOTQA_HF_MIRRORS]
    local_plotqa = None
    if args.plotqa_annotations and args.plotqa_qa_pairs:
        try:
            local_plotqa = summarize_plotqa_official(
                Path(args.plotqa_annotations),
                Path(args.plotqa_qa_pairs),
                Path(args.plotqa_image_dir) if args.plotqa_image_dir else None,
                args.limit,
                args.require_qa,
            )
        except Exception as exc:
            local_plotqa = {"source": "official_plotqa_json", "available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "schema_version": "public_benchmark_metadata_survey_v1",
        "created_date": args.created_date,
        "paid_calls_executed": 0,
        "sample_limit_per_probe": args.limit,
        "require_qa_for_eligible_count": args.require_qa,
        "candidate_specs": [
            {
                "rank": spec.rank,
                "name": spec.name,
                "official_sources": list(spec.official_sources),
                "official_schema_summary": spec.official_schema_summary,
                "notes": spec.notes,
            }
            for spec in BENCHMARK_SPECS
        ],
        "hf_probes": hf_probes,
        "local_plotqa_probe": local_plotqa,
        "recommendation": recommend(hf_probes, local_plotqa),
    }


def recommend(hf_probes: list[dict[str, Any]], local_plotqa: dict[str, Any] | None = None) -> dict[str, Any]:
    if local_plotqa and local_plotqa.get("eligible_count", 0) >= 8:
        return {
            "next_benchmark": "PlotQA",
            "action": "freeze a PlotQA zero-cost roster protocol from the official local JSON export",
            "basis": f"official PlotQA JSON produced eligible_count={local_plotqa['eligible_count']} under the survey rule.",
        }
    plotqa_probe = next((probe for probe in hf_probes if probe.get("available") and probe.get("eligible_count", 0) >= 8), None)
    if plotqa_probe:
        return {
            "next_benchmark": "PlotQA",
            "action": "freeze a PlotQA zero-cost roster protocol using the available mirror only after verifying QA fields",
            "basis": f"{plotqa_probe['dataset_id']} produced eligible_count={plotqa_probe['eligible_count']} under the survey rule.",
        }
    return {
        "next_benchmark": "PlotQA",
        "action": "obtain the official PlotQA JSON export before paid work; current quick HF mirrors are metadata diagnostics only",
        "basis": "Official PlotQA schema matches GeoSym's metadata needs, but probed mirrors did not yet provide a QA-ready 8-sample roster.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Public Benchmark Metadata Survey",
        "",
        f"- Date: {report['created_date']}",
        f"- Paid calls executed: {report['paid_calls_executed']}",
        f"- Sample limit per probe: {report['sample_limit_per_probe']}",
        f"- Eligible count requires QA fields: {report['require_qa_for_eligible_count']}",
        "",
        "## Candidate Ranking",
        "",
    ]
    for spec in report["candidate_specs"]:
        schema = spec["official_schema_summary"]
        present = ", ".join(key for key, value in schema.items() if value)
        missing = ", ".join(key for key, value in schema.items() if not value) or "none"
        lines.append(f"- {spec['rank']}. {spec['name']}: present [{present}], missing [{missing}]. {spec['notes']}")
    lines.extend(["", "## PlotQA HF Probe", ""])
    for probe in report["hf_probes"]:
        if not probe.get("available"):
            lines.append(f"- {probe['dataset_id']}: unavailable ({probe.get('error')}).")
            continue
        lines.append(
            f"- {probe['dataset_id']} / {probe['split']}: sampled {probe['sampled_rows']}, "
            f"eligible_count {probe['eligible_count']}, keys {probe['raw_keys']}."
        )
        lines.append(f"  coverage: {json.dumps(probe['coverage'], sort_keys=True)}")
        lines.append(f"  chart types: {json.dumps(probe['chart_type_counts'], sort_keys=True)}")
    if report.get("local_plotqa_probe"):
        local = report["local_plotqa_probe"]
        lines.extend(["", "## Official PlotQA Local Probe", ""])
        if local.get("available") is False:
            lines.append(f"- unavailable ({local.get('error')}).")
        else:
            lines.append(
                f"- sampled {local['sampled_rows']}, eligible_count {local['eligible_count']}, "
                f"chart types {json.dumps(local['chart_type_counts'], sort_keys=True)}."
            )
            lines.append(f"  coverage: {json.dumps(local['coverage'], sort_keys=True)}")
    rec = report["recommendation"]
    lines.extend(["", "## Recommendation", "", f"- Next benchmark: {rec['next_benchmark']}", f"- Action: {rec['action']}", f"- Basis: {rec['basis']}"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--created-date", default=date.today().isoformat())
    parser.add_argument("--require-qa", action="store_true", default=True)
    parser.add_argument("--no-require-qa", dest="require_qa", action="store_false")
    parser.add_argument("--plotqa-annotations", help="Optional official PlotQA annotations.json path.")
    parser.add_argument("--plotqa-qa-pairs", help="Optional official PlotQA qa_pairs.json path.")
    parser.add_argument("--plotqa-image-dir", help="Optional official PlotQA png directory.")
    parser.add_argument("--skip-hf", action="store_true", help="Skip Hugging Face probes and use only local official files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = survey(args)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote survey JSON: {output_json}")
    print(f"wrote survey Markdown: {output_md}")
    print("paid_calls_executed=0")
    for probe in report["hf_probes"]:
        if probe.get("available"):
            print(f"{probe['dataset_id']} eligible_count={probe['eligible_count']} sampled_rows={probe['sampled_rows']}")
        else:
            print(f"{probe['dataset_id']} unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
