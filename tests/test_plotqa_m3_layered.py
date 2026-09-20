"""Tests for PlotQA M3 layered reasoning.

The property that matters most here is that **L1 never sees the question**. A
reader who knows the target will find it in the pixels, and L1's output stops
being an independent reading. Several tests below attack that from different
angles because it cannot be enforced by review alone.

The second theme is baseline fairness (invariant 6). The GeoSym arm and the
image-only arm must differ in the evidence they receive and in nothing else; if
the layered structure itself were the advantage, a gain would be attributable to
multi-turn prompting rather than to evidence.
"""

from __future__ import annotations

import json

import pytest

from evaluator.reasoning.plotqa_m3_layered import (
    LAYER_ORDER,
    build_l1_prompt,
    build_l2_prompt,
    build_l3_prompt,
    parse_l1,
    parse_l2,
    parse_l3,
    prompt_hashes,
    render_chain_text,
    run_layered_answer,
)

QUESTION = "What is the ratio of exports in Sudan in 1988 to that in 1990?"
EVIDENCE = {"measured_facts": [{"series_label": "Sudan", "category_label": "1988", "measured_value": 12.5}]}


def _scripted(responses: dict[str, str]):
    """An ``ask`` that records the prompts it was given."""
    seen: dict[str, str] = {}

    def ask(layer: str, prompt: str) -> str:
        seen[layer] = prompt
        return responses.get(layer, "{}")

    return ask, seen


class TestL1NeverSeesTheQuestion:
    """The load-bearing constraint of the whole design."""

    def test_question_text_is_absent_from_the_l1_prompt(self):
        """The question string itself must not appear.

        Entity names may legitimately appear in L1 via evidence labels -- a fact
        labelled "Sudan" is a reading, not a leak -- so the assertion is on the
        question text, not on words that happen to occur in both.
        """
        ask, seen = _scripted({})
        run_layered_answer(ask, QUESTION, EVIDENCE)
        assert QUESTION not in seen["L1"]
        assert "{question}" not in seen["L1"]
        assert "ratio of exports" not in seen["L1"].lower()

    def test_no_question_leaks_when_evidence_is_unlabelled(self):
        """With no labels to confuse the check, nothing from the question shows."""
        ask, seen = _scripted({})
        run_layered_answer(ask, QUESTION, {"measured_facts": [{"measured_value": 12.5}]})
        assert "Sudan" not in seen["L1"]
        assert "1988" not in seen["L1"]

    def test_baseline_l1_also_excludes_the_question(self):
        ask, seen = _scripted({})
        run_layered_answer(ask, QUESTION, None)
        assert "Sudan" not in seen["L1"]

    def test_build_l1_prompt_cannot_accept_a_question(self):
        """Enforced by signature, not by convention."""
        import inspect

        params = inspect.signature(build_l1_prompt).parameters
        assert "question" not in params
        assert len(params) == 1

    def test_later_layers_do_receive_the_question(self):
        ask, seen = _scripted({})
        run_layered_answer(ask, QUESTION, EVIDENCE)
        assert "Sudan" in seen["L2"]
        assert "Sudan" in seen["L3"]


class TestLayersSeeOnlyPriorConclusions:
    def test_l2_receives_l1_output(self):
        l1 = '{"series": ["Sudan"], "readings": [{"series": "Sudan", "category": "1988", "value": 12.5}]}'
        ask, seen = _scripted({"L1": l1})
        run_layered_answer(ask, QUESTION, EVIDENCE)
        assert "12.5" in seen["L2"]

    def test_l3_receives_both_earlier_layers(self):
        l1 = '{"readings": [{"series": "Sudan", "category": "1988", "value": 12.5}]}'
        l2 = '{"operation": "ratio", "result": 1.48, "computation": "12.5 / 8.4"}'
        ask, seen = _scripted({"L1": l1, "L2": l2})
        run_layered_answer(ask, QUESTION, EVIDENCE)
        assert "12.5" in seen["L3"]
        assert "1.48" in seen["L3"]

    def test_evidence_reaches_l1_only(self):
        """Evidence must not leak forward unchecked past the reading layer."""
        ask, seen = _scripted({"L1": "{}"})
        run_layered_answer(ask, QUESTION, {"measured_facts": [{"measured_value": 987654.321}]})
        assert "987654.321" in seen["L1"]
        assert "987654.321" not in seen["L2"]


class TestBaselineFairness:
    def test_both_arms_share_the_l1_output_schema(self):
        measured, baseline = build_l1_prompt(EVIDENCE), build_l1_prompt(None)
        assert measured.split("```json")[1] == baseline.split("```json")[1]

    def test_both_arms_forbid_answering_at_l1(self):
        for prompt in (build_l1_prompt(EVIDENCE), build_l1_prompt(None)):
            assert "Do NOT answer the question yet" in prompt

    def test_l2_and_l3_are_byte_identical_across_arms(self):
        """Only L1 differs; the reasoning layers must not favour either arm."""
        l1 = {"readings": []}
        l2 = {"result": 1.0}
        assert build_l2_prompt(l1, QUESTION) == build_l2_prompt(l1, QUESTION)
        assert build_l3_prompt(l1, l2, QUESTION) == build_l3_prompt(l1, l2, QUESTION)

    def test_baseline_prompt_carries_no_evidence_placeholder(self):
        assert "{measured_evidence}" not in build_l1_prompt(None)

    def test_measured_prompt_substitutes_its_placeholder(self):
        assert "{measured_evidence}" not in build_l1_prompt(EVIDENCE)


class TestParsers:
    def test_l1_missing_keys_become_empty_containers(self):
        parsed = parse_l1("{}")
        assert parsed["readings"] == [] and parsed["series"] == []

    def test_l1_parses_json_wrapped_in_prose(self):
        raw = 'Here is my reading.\n```json\n{"series": ["A"], "readings": []}\n```\nDone.'
        assert parse_l1(raw)["series"] == ["A"]

    def test_l2_absent_sufficiency_is_none_not_false(self):
        """Unstated is not the same as insufficient."""
        assert parse_l2('{"operation": "ratio"}')["operands_sufficient"] is None

    def test_l2_explicit_insufficiency_is_preserved(self):
        assert parse_l2('{"operands_sufficient": false}')["operands_sufficient"] is False

    def test_l3_falls_back_to_raw_text_without_json(self):
        parsed = parse_l3("The answer is 1.48")
        assert parsed["normalized_prediction"] is None
        assert "1.48" in parsed["raw_text"]

    def test_l3_extracts_the_prediction(self):
        assert parse_l3('{"raw_text": "x", "normalized_prediction": "1.48"}')["normalized_prediction"] == "1.48"

    @pytest.mark.parametrize("raw", ["", "no json here", "{broken"])
    def test_parsers_never_raise(self, raw):
        parse_l1(raw), parse_l2(raw), parse_l3(raw)


class TestChainCompletionAndFailure:
    def test_a_complete_chain_reports_every_layer(self):
        ask, _ = _scripted({"L3": '{"normalized_prediction": "1.48"}'})
        result = run_layered_answer(ask, QUESTION, EVIDENCE)
        assert result["layers_completed"] == list(LAYER_ORDER)
        assert result["prediction_text"] == "1.48"
        assert result["failed_layer"] is None

    def test_a_mid_chain_failure_keeps_completed_layers(self):
        """A partial chain shows where reasoning broke; discarding it hides that."""
        def ask(layer: str, prompt: str) -> str:
            if layer == "L2":
                raise TimeoutError("boom")
            return "{}"

        result = run_layered_answer(ask, QUESTION, EVIDENCE)
        assert result["failed_layer"] == "L2"
        assert result["layers_completed"] == ["L1"]
        assert "TimeoutError" in result["steps"]["L2"]["error"]

    def test_failure_at_l1_stops_the_chain(self):
        def ask(layer: str, prompt: str) -> str:
            raise RuntimeError("no")

        result = run_layered_answer(ask, QUESTION, EVIDENCE)
        assert result["failed_layer"] == "L1"
        assert "L2" not in result["steps"]

    def test_exactly_three_calls_are_made(self):
        """Cost is predictable: one call per layer, never a retry loop."""
        calls: list[str] = []

        def ask(layer: str, prompt: str) -> str:
            calls.append(layer)
            return "{}"

        run_layered_answer(ask, QUESTION, EVIDENCE)
        assert calls == list(LAYER_ORDER)


class TestChainText:
    def test_chain_text_renders_raw_output_not_conclusions(self):
        ask, _ = _scripted({"L1": "L1 said this", "L2": "L2 said that", "L3": "L3 concluded"})
        text = render_chain_text(run_layered_answer(ask, QUESTION, EVIDENCE))
        assert "L1 said this" in text and "L2 said that" in text and "L3 concluded" in text

    def test_chain_text_preserves_execution_order(self):
        ask, _ = _scripted({"L1": "first", "L2": "second", "L3": "third"})
        text = render_chain_text(run_layered_answer(ask, QUESTION, EVIDENCE))
        assert text.index("first") < text.index("second") < text.index("third")

    def test_chain_text_of_a_partial_chain_is_not_empty(self):
        def ask(layer: str, prompt: str) -> str:
            if layer == "L2":
                raise TimeoutError("boom")
            return "L1 output"

        text = render_chain_text(run_layered_answer(ask, QUESTION, EVIDENCE))
        assert "L1 output" in text


class TestProvenance:
    def test_every_layer_template_is_hashed(self):
        hashes = prompt_hashes()
        assert set(hashes) == {"plotqa_L1_measured", "plotqa_L1_baseline", "plotqa_L2", "plotqa_L3"}
        assert all(len(h) == 64 for h in hashes.values())

    def test_the_two_l1_templates_differ(self):
        hashes = prompt_hashes()
        assert hashes["plotqa_L1_measured"] != hashes["plotqa_L1_baseline"]


class TestNoReliabilityScoring:
    """M1/M2 retired 2026-09-19; they must not return via a reasoning layer."""

    def test_no_layer_ranks_or_filters_evidence(self):
        import evaluator.reasoning.plotqa_m3_layered as module

        source = json.dumps(dir(module))
        for banned in ("score_evidence", "select_evidence", "filter_facts", "rank_facts"):
            assert banned not in source
