import json
from pathlib import Path

from scripts.run_plotqa_direct_public_baselines import build_plan, review_plan


def test_plotqa_direct_public_500_plan_is_e0_e1_only(tmp_path: Path) -> None:
    samples = [
        {
            "sample_id": f"s{i}",
            "question": "What is the value?",
            "image_path": "image.png",
            "answer_type": "numeric",
            "normalized_answer": 1.0,
        }
        for i in range(500)
    ]
    source_plan = {
        "benchmark": "PlotQA",
        "selected_samples": samples,
        "frozen_before_paid_run": {
            "model_provenance": {"display_name": "GPT-5.5", "api_model_id": "gpt-5.5"},
            "reasoning_model": "gpt-5.5",
            "scorer_version": "test_scorer",
        },
    }
    plan_path = tmp_path / "plotqa_plan.json"
    plan_path.write_text(json.dumps(source_plan), encoding="utf-8")

    plan = build_plan(plan_path)
    review = review_plan(plan)

    assert plan["conditions"] == ["E0_plotqa_direct", "E1_plotqa_cot"]
    assert plan["sample_condition_job_cap"] == 1000
    assert plan["llm_http_attempt_cap"] == 1000
    assert plan["ocr_http_attempt_cap"] == 0
    assert review["status"] == "no_blocking_findings"
