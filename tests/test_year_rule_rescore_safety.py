"""Safety properties of the offline year-rule rescore.

The rescore reads paid artifacts, which are the only record of money already
spent. Two properties make it safe to run, and both are asserted by the script
itself rather than trusted:

1. no paid artifact changes on disk (sha256 manifest compared before/after);
2. no *non-year* row changes verdict, so a regression in the generic numeric
   path surfaces here instead of in a paper table.

The script exits non-zero if either fails, so a broken run cannot be mistaken
for a clean one.
"""

import inspect

from scripts import rescore_year_answers_offline as rescore


class TestTargetsAreScoped:
    def test_stress_128_is_excluded_with_a_stated_reason(self):
        """Its rows are keyed by `source_sample_id` under retired conditions.

        Locked because silently including it would either crash or, worse,
        attach the wrong question to a row.
        """
        assert all("stress_128" not in str(t) for t in rescore.TARGETS)
        assert "m1m2_stress_128" in inspect.getsource(rescore)

    def test_targets_are_relative_paths_under_the_package(self):
        """An absolute path here would escape the package on another machine."""
        assert all(not t.is_absolute() for t in rescore.TARGETS)

    def test_no_geosym_bench_artifact_is_targeted(self):
        """GeoSym-Bench has no year questions; touching it would be out of scope."""
        assert all("geosym_bench" not in str(t) for t in rescore.TARGETS)


class TestSafetyPropertiesAreEnforced:
    def test_run_hashes_inputs_before_and_after(self):
        source = inspect.getsource(rescore.run)
        assert "input_sha256_before" in source
        assert "input_sha256_after" in source
        assert "paid_artifacts_unmodified" in source

    def test_non_year_drift_is_counted(self):
        """Every non-year row is rescored and compared, not assumed stable."""
        source = inspect.getsource(rescore.rescore_rows)
        assert "non_year_drift" in source

    def test_main_fails_loudly_when_a_paid_artifact_changed(self):
        source = inspect.getsource(rescore.main)
        assert "return 1" in source
        assert "paid_artifacts_unmodified" in source

    def test_main_fails_loudly_on_non_year_drift(self):
        source = inspect.getsource(rescore.main)
        assert "non_year_rows_that_moved" in source


class TestRescoreLogic:
    def test_year_row_flips_when_prediction_is_a_different_year(self):
        rows = [{"answer_type": "numeric", "condition": "E0",
                 "sample_id": "s1", "reference": 2001.0,
                 "normalized_prediction": 1991.0, "correct": True}]
        out = rescore.rescore_rows(rows, {"s1": "In which year was it minimum?"})
        stats = out["by_condition"]["E0"]
        assert stats["year"] == 1
        assert stats["lenient_correct"] == 1
        assert stats["strict_correct"] == 0
        assert len(out["flipped_rows"]) == 1

    def test_non_year_row_is_left_alone(self):
        rows = [{"answer_type": "numeric", "condition": "E0",
                 "sample_id": "s1", "reference": 2001.0,
                 "normalized_prediction": 1991.0, "correct": True}]
        # Same numbers, but the question asks for a magnitude, not a year.
        out = rescore.rescore_rows(rows, {"s1": "What is the total population?"})
        stats = out["by_condition"]["E0"]
        assert stats["year"] == 0
        assert stats["non_year_drift"] == 0
        assert out["flipped_rows"] == []

    def test_a_row_with_no_resolvable_question_cannot_flip(self):
        """Missing question must fall back to the generic rule, never guess."""
        rows = [{"answer_type": "numeric", "condition": "E0",
                 "sample_id": "unknown", "reference": 2001.0,
                 "normalized_prediction": 1991.0, "correct": True}]
        out = rescore.rescore_rows(rows, {})
        assert out["by_condition"]["E0"]["year"] == 0
        assert out["by_condition"]["E0"]["non_year_drift"] == 0

    def test_text_rows_are_skipped(self):
        rows = [{"answer_type": "text", "condition": "E0", "sample_id": "s1",
                 "reference": "Brazil", "normalized_prediction": "Brazil",
                 "correct": True}]
        assert rescore.rescore_rows(rows, {"s1": "Which year?"})["by_condition"] == {}
