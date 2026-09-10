from scripts.selective_propagation import facts_for_selective_condition, selective_answer_prompt


def schema() -> dict:
    return {
        "condition": "G_full_guarded_schema_only",
        "expected_answerability": "answerable",
        "stress_hash": "h0",
        "evidence_candidates": [
            {"fact_id": "f_selected", "key": "measured_value", "reliability_tier": "high"},
            {"fact_id": "f_suppressed", "key": "measured_value", "reliability_tier": "low"},
        ],
        "evidence_gaps": [{"gap_id": "g0", "key": "measured_value", "needed_for_question": True, "severity": "blocking"}],
        "selected_evidence_set": {
            "selected_fact_ids": ["f_selected"],
            "suppressed_fact_ids": ["f_suppressed"],
            "surfaced_gap_ids": [],
            "blocking_gap_keys": [],
            "required_fact_coverage_complete": True,
            "propagation_policy": "answer_with_candidate_and_selected_facts",
            "answer_candidate": {
                "normalized_prediction": 7,
                "answer_type": "numeric",
                "supporting_fact_ids": ["f_selected"],
            },
        },
    }


def test_candidate_is_visible_only_to_selective_propagation() -> None:
    selective = facts_for_selective_condition(schema(), "G_selective_propagation", 20260830, "s0")
    no_routing = facts_for_selective_condition(schema(), "G_no_routing", 20260830, "s0")
    all_evidence = facts_for_selective_condition(schema(), "G_all_evidence", 20260830, "s0")

    assert selective["answer_candidate"]["normalized_prediction"] == 7
    assert selective["routing_policy"] == "answer_with_candidate_and_selected_facts"
    assert no_routing["answer_candidate"] is None
    assert all_evidence["answer_candidate"] is None


def test_candidate_guided_prompt_forbids_missing_fact_reaudit() -> None:
    facts = facts_for_selective_condition(schema(), "G_selective_propagation", 20260830, "s0")
    prompt = selective_answer_prompt("What is the value?", facts, "G_selective_propagation", "spurious_low_confidence_fact")

    assert "return answer_candidate.normalized_prediction exactly" in prompt
    assert "do not invent missing facts" in prompt.lower()
