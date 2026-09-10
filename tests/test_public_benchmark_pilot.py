import base64
import json
from pathlib import Path

import pytest

from scripts.prepare_public_benchmark_pilot import (
    build_plan,
    canonical_json_bytes,
    deterministic_roster,
    infer_chart_type_from_question,
    infer_answer_derivation,
    infer_reasoning_type,
    normalize_answer,
    parse_args,
    read_manifest,
    stable_payload_hash,
)


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def write_png(path: Path) -> None:
    path.write_bytes(PNG_1X1)


def write_synthetic_bar_chart(path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (120, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.line((15, 10, 15, 75), fill="black", width=3)
    draw.line((15, 75, 110, 75), fill="black", width=3)
    for x0, top in [(30, 45), (55, 30), (80, 55)]:
        draw.rectangle((x0, top, x0 + 12, 74), fill=(40, 90, 180))
    image.save(path)


def write_manifest(tmp_path: Path, count: int = 8, eligible: bool = True) -> Path:
    image = tmp_path / "chart.png"
    write_png(image)
    rows = []
    for idx in range(count):
        metadata = {
            "benchmark": "ChartQA",
            "split": "test",
            "source_index": str(idx),
            "chart_type": "line" if eligible else "unknown",
            "axis_visible": True if eligible else "unknown",
            "image_readable": True,
            "full_split_export": True,
            "question_sha256": f"q{idx}",
            "image_sha256": f"i{idx}",
        }
        rows.append(
            {
                "sample_id": f"chartqa_test_{idx:04d}",
                "image_path": "chart.png",
                "question": "What is the highest value?",
                "answer": str(idx + 1),
                "answer_type": "numeric",
                "metadata": metadata,
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return manifest


def test_normalize_answer_numeric_and_text() -> None:
    assert normalize_answer("1,234.50 kg", "numeric") == 1234.5
    assert normalize_answer("  North   America ", "categorical") == "north america"
    assert normalize_answer("No.", "text") == "no"
    assert normalize_answer("  Keep Exact Case ", "open") == "Keep Exact Case"


def test_canonical_json_hash_is_byte_stable() -> None:
    left = {"source_index": "1", "seed": "20260828", "question_sha256": "q", "image_sha256": "i"}
    right = {"image_sha256": "i", "question_sha256": "q", "seed": "20260828", "source_index": "1"}
    assert canonical_json_bytes(left) == b'{"image_sha256":"i","question_sha256":"q","seed":"20260828","source_index":"1"}'
    assert stable_payload_hash(left) == stable_payload_hash(right)


def test_question_tagging_priority_and_answer_derivation() -> None:
    assert infer_reasoning_type("What is the ratio between A and B?") == "ratio"
    assert infer_reasoning_type("What is the difference between Lamb and Corn?") == "comparison"
    assert infer_reasoning_type("What is the peak value?") == "extrema"
    assert infer_reasoning_type("How many bars are shown?") == "lookup"
    assert infer_answer_derivation("comparison", "Which is higher, A or B?") == "direct"
    assert infer_answer_derivation("comparison", "What is the difference between A and B?") == "derived"
    assert infer_chart_type_from_question("Identify the percentage in the pie chart.") == "pie"


def test_question_chart_guard_overrides_supported_metadata_for_exclusion(tmp_path: Path) -> None:
    image = tmp_path / "chart.png"
    write_synthetic_bar_chart(image)
    manifest = tmp_path / "manifest.jsonl"
    row = {
        "sample_id": "pie_question",
        "image_path": "chart.png",
        "question": "Identify the percentage in the pie chart.",
        "answer": "10",
        "answer_type": "numeric",
        "metadata": {
            "chart_type": "simple_bar",
            "axis_visible": True,
            "image_readable": True,
            "full_split_export": True,
        },
    }
    manifest.write_text(json.dumps(row), encoding="utf-8")

    task = read_manifest(manifest)[0]

    assert task.metadata["chart_type"] == "pie"
    assert task.metadata["geosym_compatibility"] == "incompatible"


def test_read_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    image = tmp_path / "chart.png"
    write_png(image)
    row = {
        "sample_id": "dup",
        "image_path": "chart.png",
        "question": "q",
        "answer": "1",
        "answer_type": "numeric",
        "metadata": {"chart_type": "line", "axis_visible": True},
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate sample_id"):
        read_manifest(manifest)


def test_manifest_enrichment_fails_closed_when_visual_tags_unknown(tmp_path: Path) -> None:
    tasks = read_manifest(write_manifest(tmp_path, count=8, eligible=False))
    assert all(task.metadata["geosym_compatibility"] == "unknown" for task in tasks)
    with pytest.raises(ValueError, match="selector_insufficient"):
        deterministic_roster(tasks)


def test_build_plan_rejects_source_prefix_manifest_by_default(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["metadata"]["full_split_export"] = False
        row["metadata"]["source_prefix_export"] = True
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    args = parse_args(["--manifest", str(manifest), "--output", str(tmp_path / "plan.json")])

    with pytest.raises(ValueError, match="source_prefix_manifest"):
        build_plan(args)


def test_build_plan_freezes_phase1_g0_smoke(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    output = tmp_path / "plan.json"
    args = parse_args(["--manifest", str(manifest), "--output", str(output)])

    plan = build_plan(args)

    frozen = plan["frozen_before_paid_run"]
    assert plan["zero_cost_only"] is True
    assert plan["paid_calls_executed"] == 0
    assert frozen["conditions"] == ["E0_public_direct", "E1_public_cot", "G0_public_evidence_trace"]
    assert frozen["sample_condition_job_cap"] == 24
    assert frozen["llm_http_attempt_cap"] == 40
    assert frozen["ocr_http_attempt_cap"] == 0
    assert len(plan["selected_samples"]) == 8
    assert len(frozen["sample_selection"]["selected_ids"]) == 8
    assert "g0_extract_prompt_template" in frozen["prompt_hashes"]


def test_build_plan_rejects_unsupported_condition_name(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    args = parse_args(["--manifest", str(manifest), "--output", str(tmp_path / "plan.json"), "--conditions", "E5"])

    with pytest.raises(ValueError, match="unsupported Phase 1 condition"):
        build_plan(args)


def test_main_writes_frozen_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = write_manifest(tmp_path)
    output = tmp_path / "plan.json"
    from scripts.prepare_public_benchmark_pilot import main

    exit_code = main(["--manifest", str(manifest), "--output", str(output)])

    assert exit_code == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["benchmark"] == "ChartQA"
    assert written["paid_calls_executed"] == 0
    assert "paid_calls_executed=0" in capsys.readouterr().out
