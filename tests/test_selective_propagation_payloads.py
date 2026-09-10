from scripts.selective_propagation import facts_for_selective_condition, selective_answer_prompt


def stress_schema() -> dict:
    return {
        "condition": "G_selective_schema",
        "expected_answerability": "insufficient_evidence",
        "stress_hash": "abc",
        "evidence_candidates": [
            {"fact_id": "low", "reliability_tier": "low"},
            {"fact_id": "high", "reliability_tier": "high"},
            {"fact_id": "medium", "reliability_tier": "medium"},
        ],
        "evidence_gaps": [{"gap_id": "gap0", "key": "measured_value"}],
        "selected_evidence_set": {
            "selected_fact_ids": ["medium"],
            "surfaced_gap_ids": ["gap0"],
            "propagation_policy": "abstain_insufficient_evidence",
            "answer_candidate": {
                "normalized_prediction": 12,
                "supporting_fact_ids": ["medium"],
            },
        },
    }


def test_selective_condition_payload_semantics() -> None:
    row = stress_schema()

    selective = facts_for_selective_condition(row, "G_selective_propagation", 20260830, "stress_000")
    candidate_unrouted = facts_for_selective_condition(row, "G_candidate_unrouted", 20260830, "stress_000")
    no_gaps = facts_for_selective_condition(row, "G_selective_no_gaps", 20260830, "stress_000")
    all_evidence = facts_for_selective_condition(row, "G_all_evidence", 20260830, "stress_000")
    top = facts_for_selective_condition(row, "G_top_confidence", 20260830, "stress_000")
    random_k = facts_for_selective_condition(row, "G_random_k", 20260830, "stress_000")
    no_routing = facts_for_selective_condition(row, "G_no_routing", 20260830, "stress_000")

    assert [fact["fact_id"] for fact in selective["measured_facts"]] == ["medium"]
    assert selective["evidence_gaps"] == [{"gap_id": "gap0", "key": "measured_value"}]
    assert selective["routing_policy"] == "abstain_insufficient_evidence"
    assert selective["answer_candidate"]["normalized_prediction"] == 12
    assert [fact["fact_id"] for fact in candidate_unrouted["measured_facts"]] == ["medium"]
    assert candidate_unrouted["evidence_gaps"] == []
    assert candidate_unrouted["routing_policy"] == "hidden"
    assert candidate_unrouted["answer_candidate"]["supporting_fact_ids"] == ["medium"]
    assert no_gaps["evidence_gaps"] == []
    assert no_gaps["routing_policy"] == "answerable_by_default"
    assert no_gaps["answer_candidate"] is None
    assert [fact["fact_id"] for fact in all_evidence["measured_facts"]] == ["low", "high", "medium"]
    assert [fact["fact_id"] for fact in top["measured_facts"]] == ["high"]
    assert len(random_k["measured_facts"]) == 1
    assert no_routing["evidence_gaps"] == [{"gap_id": "gap0", "key": "measured_value"}]
    assert no_routing["routing_policy"] == "hidden"


def test_evidence_only_prompt_has_no_image_or_visual_remeasurement_instruction() -> None:
    facts = facts_for_selective_condition(stress_schema(), "G_selective_propagation", 20260830, "stress_000")

    prompt = selective_answer_prompt("What is the difference?", facts, "G_selective_propagation", "missing_required_fact")

    assert "No chart image is available" in prompt
    assert "Use only the supplied measured evidence JSON" in prompt
    assert "visually re-measuring" not in prompt
    assert "image_path" not in prompt
