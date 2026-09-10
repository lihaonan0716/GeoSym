from scripts.audit_extrema_filter import run_audit


def test_p0_2_extrema_filter_audit_cases_pass():
    rows = run_audit()

    assert [(row["case"], row["actual"]) for row in rows if not row["passed"]] == []
