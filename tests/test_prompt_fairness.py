from scripts.audit_prompt_fairness import run_audit


def test_domain_range_guidance_is_matched_across_conditions():
    rows = run_audit()

    failed = [
        (row["condition"], name)
        for row in rows
        for name, passed in row["markers"].items()
        if not passed
    ]

    assert failed == []
