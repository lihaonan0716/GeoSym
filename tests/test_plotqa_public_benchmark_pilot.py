import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_plotqa_public_benchmark_pilot import (
    build_plan,
    excluded_summary,
    infer_answer_derivation,
    infer_reasoning_type,
    load_candidates,
    normalize_chart_type,
    parse_args,
    select_roster,
    validate_plotqa_sample_semantics,
)
from scripts.run_public_benchmark_pilot import load_plan


def write_plotqa_bundle(tmp_path: Path, count: int = 8, chart_type: str = "line") -> tuple[Path, Path, Path]:
    image_dir = tmp_path / "png"
    image_dir.mkdir()
    annotations = []
    qa_pairs = []
    for idx in range(count):
        Image.new("RGB", (8, 8), "white").save(image_dir / f"{idx}.png")
        annotations.append(
            {
                "image_index": idx,
                "type": chart_type,
                "models": [{"name": "series", "x": [1, 2], "y": [3, 4]}],
                "general_figure_info": {
                    "x_axis": {"major_ticks": [{"label": "1"}]},
                    "y_axis": {"major_ticks": [{"label": "3"}]},
                    "legend": {"bbox": [0, 0, 1, 1]},
                    "plot_info": {"bbox": [0, 0, 10, 10]},
                },
            }
        )
        qa_pairs.append(
            {
                "image_index": idx,
                "question_string": "What is the difference between A and B?",
                "answer": "1",
                "answer_bbox": [0, 0, 1, 1],
                "template": "difference",
                "type": chart_type,
            }
        )
    annotations_path = tmp_path / "annotations.json"
    qa_path = tmp_path / "qa_pairs.json"
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    qa_path.write_text(json.dumps(qa_pairs), encoding="utf-8")
    return annotations_path, qa_path, image_dir


def write_model_config(tmp_path: Path) -> Path:
    path = tmp_path / "default_config.yaml"
    path.write_text(
        """
models:
  TestProvider:
    adapter: "tests.adapters.TestAdapter"
    api_key_name: "test_key"
    base_url_key: "test_base_url"
    models:
      "Test Model": "test-api-model"
""".strip(),
        encoding="utf-8",
    )
    return path


def test_plotqa_chart_type_and_question_rules() -> None:
    assert normalize_chart_type("vbar_categorical") == "bar"
    assert normalize_chart_type("dot_line") == "line"
    assert normalize_chart_type("") == "unknown"
    assert infer_reasoning_type("What is the ratio between A and B?") == "ratio"
    assert infer_reasoning_type("What is the difference between A and B?") == "comparison"
    assert infer_answer_derivation("comparison", "How many more A than B?") == "derived"


def test_plotqa_question_rules_exclude_structural_and_unsupported_aggregate() -> None:
    assert infer_reasoning_type("How many legend labels are there ?", "structural", "line") == "text"
    assert infer_reasoning_type("What is the average value per year?", "", "line") == "unknown"
    assert infer_reasoning_type("What is the minimum percentage of imports?", "min_max", "line") == "extrema"
    assert infer_reasoning_type("Is the difference between the percentage of A and B greater than C?", "compound", "bar") == "comparison"
    question = "What is the difference between the highest and lowest values?"
    assert infer_reasoning_type(question, "", "line") == "range"
    assert infer_answer_derivation("range", question) == "derived"
    assert infer_answer_derivation("comparison", "In how many countries is A greater than 10?") == "derived"


def test_plotqa_semantic_validator_catches_template_tag_conflicts() -> None:
    sample = {
        "question": "How many legend labels are there ?",
        "metadata": {
            "qa_template": "structural",
            "qa_type": "line",
            "reasoning_type": "lookup",
            "answer_derivation": "direct",
            "quantitative_reasoning_required": True,
        },
    }

    violations = validate_plotqa_sample_semantics(sample)

    assert "structural_qa_template_excluded" in violations
    assert "text_or_structural_question" in violations
    assert "reasoning_type_mismatch:lookup!=text" in violations
    assert "quantitative_flag_mismatch:True!=False" in violations


def test_plotqa_official_join_is_qa_pair_level(tmp_path: Path) -> None:
    annotations_path, qa_path, image_dir = write_plotqa_bundle(tmp_path, count=1)
    qa_pairs = json.loads(qa_path.read_text(encoding="utf-8"))
    qa_pairs.append({**qa_pairs[0], "question_string": "What is the value?", "template": "lookup"})
    qa_path.write_text(json.dumps(qa_pairs), encoding="utf-8")

    candidates, access = load_candidates(annotations_path, qa_path, image_dir, "test")

    assert len(candidates) == 2
    assert access["qa_pair_count"] == 2
    assert candidates[0].sample_id != candidates[1].sample_id


def test_plotqa_loader_accepts_wrapped_qa_pairs_json(tmp_path: Path) -> None:
    annotations_path, qa_path, image_dir = write_plotqa_bundle(tmp_path, count=1)
    qa_pairs = json.loads(qa_path.read_text(encoding="utf-8"))
    qa_path.write_text(json.dumps({"qa_pairs": qa_pairs}), encoding="utf-8")

    candidates, access = load_candidates(annotations_path, qa_path, image_dir, "test")

    assert len(candidates) == 1
    assert access["qa_pair_count"] == 1


def test_plotqa_unknown_chart_type_is_not_eligible(tmp_path: Path) -> None:
    annotations_path, qa_path, image_dir = write_plotqa_bundle(tmp_path, count=8, chart_type="")
    candidates, _ = load_candidates(annotations_path, qa_path, image_dir, "test")

    with pytest.raises(ValueError, match="selector_insufficient"):
        select_roster(candidates, 8)


def test_plotqa_build_plan_freezes_8_unique_images_and_cap(tmp_path: Path) -> None:
    annotations_path, qa_path, image_dir = write_plotqa_bundle(tmp_path, count=8)
    output = tmp_path / "plan.json"
    args = parse_args(
        [
            "--annotations",
            str(annotations_path),
            "--qa-pairs",
            str(qa_path),
            "--image-dir",
            str(image_dir),
            "--output",
            str(output),
            "--config",
            str(write_model_config(tmp_path)),
        ]
    )

    plan = build_plan(args)
    output.write_text(json.dumps(plan), encoding="utf-8")
    loaded = load_plan(output)

    assert plan["benchmark"] == "PlotQA"
    assert plan["paid_run_approved"] is True
    assert "zero_cost_only" not in plan
    assert plan["frozen_before_paid_run"]["reasoning_model"] == "test-api-model"
    assert plan["frozen_before_paid_run"]["model_provenance"]["display_name"] == "Test Model"
    assert plan["frozen_before_paid_run"]["model_provenance"]["provider"] == "TestProvider"
    assert plan["frozen_before_paid_run"]["sample_condition_job_cap"] == 24
    assert plan["frozen_before_paid_run"]["llm_http_attempt_cap"] == 40
    assert len({sample["metadata"]["image_index"] for sample in plan["selected_samples"]}) == 8
    assert loaded["frozen_before_paid_run"]["conditions"] == [
        "E0_plotqa_direct",
        "E1_plotqa_cot",
        "G0_plotqa_evidence_trace",
    ]


def test_plotqa_excluded_summary_never_counts_ineligible_as_eligible(tmp_path: Path) -> None:
    annotations_path, qa_path, image_dir = write_plotqa_bundle(tmp_path, count=1)
    qa_pairs = [
        {
            "image_index": 0,
            "question_string": "How many legend labels are there ?",
            "answer": "1",
            "template": "structural",
            "type": "line",
        }
    ]
    qa_path.write_text(json.dumps(qa_pairs), encoding="utf-8")
    candidates, _ = load_candidates(annotations_path, qa_path, image_dir, "test")

    summary = excluded_summary(candidates)

    assert "eligible" not in summary
    assert summary == {"structural_or_text_question": 1}
