"""The Judge must see what each arm actually generated, not a digest of it.

Before 2026-09-19 ``build_chain_text`` treated the two pipeline styles
differently: single-step conditions sent their raw model output verbatim, while
layered conditions were rebuilt from ``final_parsed`` -- final answers only,
with the L1 reading and L2 analysis discarded. Measured over the 240x7
GLM-4.6V artifact the Judge received 102-106% of what E0-E4 generated and
25-27% of what the layered arms generated, then was asked to score
``evidence_support`` on both.

That asymmetry is a harness defect, not a property of the method, and it is the
baseline-fairness failure CLAUDE.md invariant 6 forbids -- pointing the other
way from the usual case, since it penalised the GeoSym arm.

These tests lock the repair in both directions: layered arms must now reach the
Judge with their derivation intact, and the single-step arms' text must not
change, because their scores are reused rather than re-judged.
"""

import pytest

from app.reasoning_pipeline import (
    _build_single_step_pipeline,
    build_chain_text,
    render_step_transcript,
)


def _layered_pipeline(**overrides):
    pipeline = {
        "steps": {
            "L1": {"raw": "L1 READING: the curve passes through (0, -4) and (2, 0).",
                   "parsed": {"intercepts": {"y_intercept": -4.0}}},
            "L2": {"raw": "L2 ANALYSIS: monotone increasing on [0, 5]; slope ~2.",
                   "parsed": {"monotonicity": [{"interval": [0, 5]}]}},
            "L3": {"raw": "L3 INFERENCE: therefore f(x) = 2x - 4.",
                   "parsed": {"expression": "2*x - 4"}},
        },
        "final_parsed": {
            "expression": "2*x - 4",
            "domain": "all reals",
            "range": "all reals",
            "inference_explanation": "linear fit",
        },
    }
    pipeline.update(overrides)
    return pipeline


class TestLayeredArmsSendTheirDerivation:
    def test_every_layer_reaches_the_judge(self):
        text = build_chain_text(_layered_pipeline())
        assert "L1 READING" in text
        assert "L2 ANALYSIS" in text
        assert "L3 INFERENCE" in text

    def test_layers_appear_in_execution_order(self):
        text = build_chain_text(_layered_pipeline())
        assert text.index("L1 READING") < text.index("L2 ANALYSIS") < text.index("L3 INFERENCE")

    def test_transcript_is_substantially_longer_than_the_answers_alone(self):
        """The regression this guards against shrank the text to about a quarter."""
        pipeline = _layered_pipeline()
        answers_only = len(str(pipeline["final_parsed"]))
        assert len(build_chain_text(pipeline)) > answers_only

    def test_step_headers_match_the_single_step_format(self):
        """Both styles must be labelled the same way, or format becomes a cue."""
        text = build_chain_text(_layered_pipeline())
        assert "=== L1 ===" in text
        assert "=== L3 ===" in text


class TestSingleStepArmsAreUnchanged:
    """Their scores are reused, so any change here would silently invalidate them."""

    def test_raw_output_is_still_sent_verbatim(self):
        pipeline = _build_single_step_pipeline(
            step_name="DirectQA",
            prompt="p",
            raw_response="The answer is f(x) = x^2 - 4.",
            parsed={"expression": "x**2 - 4"},
            image_mode="direct_image",
        )
        assert build_chain_text(pipeline) == "=== DirectQA ===\nThe answer is f(x) = x^2 - 4."

    def test_an_explicit_override_still_wins(self):
        pipeline = _layered_pipeline(judge_chain_text="PRESET")
        assert build_chain_text(pipeline) == "PRESET"


class TestFallbackWhenNoTranscriptExists:
    def test_empty_steps_fall_back_to_the_parsed_answers(self):
        pipeline = {"steps": {}, "final_parsed": {"expression": "x + 1"}}
        text = build_chain_text(pipeline)
        assert "Expression: x + 1" in text

    def test_blank_raw_text_does_not_produce_an_empty_transcript(self):
        """A run whose steps stored empty strings must not send the Judge nothing."""
        pipeline = {"steps": {"L1": {"raw": "   "}},
                    "final_parsed": {"expression": "x + 1"}}
        assert render_step_transcript(pipeline) is None
        assert "Expression: x + 1" in build_chain_text(pipeline)

    def test_missing_steps_key_falls_back(self):
        assert render_step_transcript({"final_parsed": {}}) is None

    def test_partially_blank_steps_keep_the_usable_ones(self):
        pipeline = {"steps": {"L1": {"raw": ""}, "L2": {"raw": "kept"}}}
        text = render_step_transcript(pipeline)
        assert text == "=== L2 ===\nkept"


class TestCoverageParityAcrossArms:
    """The defect was a ratio between arms, so the test is a ratio too."""

    def test_layered_and_single_step_both_reach_full_coverage(self):
        layered = _layered_pipeline()
        layered_generated = sum(len(s["raw"].strip()) for s in layered["steps"].values())
        layered_sent = len(build_chain_text(layered))

        raw = "The answer is f(x) = x^2 - 4, because the vertex sits at (0, -4)."
        single = _build_single_step_pipeline(
            step_name="DirectQA", prompt="p", raw_response=raw,
            parsed={"expression": "x**2 - 4"}, image_mode="direct_image",
        )
        single_sent = len(build_chain_text(single))

        assert layered_sent >= layered_generated
        assert single_sent >= len(raw)
        # Neither arm may be more than modestly padded relative to the other.
        layered_ratio = layered_sent / layered_generated
        single_ratio = single_sent / len(raw)
        assert abs(layered_ratio - single_ratio) < 0.5


@pytest.mark.parametrize("step_names", [
    ("L1", "L2", "L3"),
    ("Step 1", "Step 2", "Step 3"),
    ("Reading", "Quantitative Analysis", "Reasoning"),
])
def test_alternate_step_namings_are_all_rendered(step_names):
    """Stored runs use several step-name conventions; none may be dropped."""
    pipeline = {"steps": {name: {"raw": f"content of {name}"} for name in step_names}}
    text = build_chain_text(pipeline)
    for name in step_names:
        assert f"content of {name}" in text
