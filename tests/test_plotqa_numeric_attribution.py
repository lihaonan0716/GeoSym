"""Safety properties of the PlotQA numeric-attribution run.

Written **before** any paid call. What these pin is the set of defects that have
actually cost this project money or credibility before:

* a cap written as a literal instead of derived from the condition list;
* an identity hash that includes its own timestamp, so the resume guard rejects
  its own checkpoint and re-bills every completed row;
* an ``actual_ocr_http_attempts`` field that is a hardcoded 0 while the path
  really bills an OCR call (a defect found in an earlier attribution runner);
* ``score_prediction`` called without ``question``, which silently scores year
  answers with a numeric tolerance instead of exact equality;
* a roster selected with anything that could encode the reference answer.
"""

import json
from pathlib import Path

import pytest

from scripts import run_plotqa_numeric_attribution as attribution


def _roster(n=4):
    return [
        {
            "sample_id": f"plotqa_test_{i}_x_{i}",
            "question": f"What is the sum of series in year {1990 + i}?",
            "answer": 12.5 + i,
            "answer_type": "numeric",
            "image_path": "img.png",
            "metadata": {"qa_template": "arithmetic", "qa_type": "vbar"},
        }
        for i in range(n)
    ]


@pytest.fixture
def frozen_files(tmp_path: Path):
    records = tmp_path / "records.json"
    ocr = tmp_path / "ocr.json"
    records.write_text(json.dumps([{"sample_id": "a"}]), encoding="utf-8")
    ocr.write_text(json.dumps([{"sample_id": "a"}]), encoding="utf-8")
    return records, ocr


def _plan(frozen_files, roster=None, stage="pilot"):
    records, ocr = frozen_files
    return attribution.build_plan(stage, roster or _roster(),
                                  records_path=records, ocr_path=ocr)


class TestCostGating:
    def test_caps_derive_from_the_condition_list(self):
        caps = attribution.call_caps(8)
        n = len(attribution.CONDITIONS)
        assert caps["sample_condition_job_cap"] == 8 * n
        assert caps["reasoning_llm_call_cap"] == 8 * n
        assert caps["llm_http_attempt_cap"] == 8 * n * 2
        assert caps["judge_llm_call_cap"] == 0
        assert caps["ocr_http_attempt_cap"] == 0

    def test_caps_scale_linearly(self):
        assert (attribution.call_caps(80)["reasoning_llm_call_cap"]
                == 10 * attribution.call_caps(8)["reasoning_llm_call_cap"])

    def test_plan_defaults_to_spending_nothing(self, frozen_files):
        plan = _plan(frozen_files)
        assert plan["paid_calls_executed"] == 0
        assert plan["dry_run"]["paid_calls_executed"] == 0

    def test_review_rejects_a_tampered_cap(self, frozen_files):
        plan = _plan(frozen_files)
        plan["llm_http_attempt_cap"] = 999999
        review = attribution.review_plan(plan)
        assert review["status"] == "blocking_findings_present"
        assert any(f.startswith("llm_http_attempt_cap_not_")
                   for f in review["blocking_findings"])

    def test_clean_plan_has_no_blocking_findings(self, frozen_files):
        assert attribution.review_plan(_plan(frozen_files))["status"] == "no_blocking_findings"

    def test_counter_refuses_the_request_that_would_exceed_the_cap(self):
        counter = attribution.AttemptCounter(2)
        counter.consume()
        counter.consume()
        with pytest.raises(attribution.LLMHttpAttemptCapExhausted):
            counter.consume()
        assert counter.cap_exhausted is True

    def test_cap_is_charged_before_the_request_not_after(self):
        """Wrapping _call_with_retry means adapter retries consume the cap."""
        counter = attribution.AttemptCounter(1)
        calls = []

        class FakeAdapter(attribution.adapter_base.BaseLLMAdapter):
            def __init__(self):  # noqa: D107 - no API key needed
                pass

            def _call_api(self, image_b64, prompt):
                calls.append(prompt)
                return "{}"

        with attribution.count_llm_attempts(counter):
            adapter = FakeAdapter()
            adapter._call_with_retry("b64", "p1")
            with pytest.raises(attribution.LLMHttpAttemptCapExhausted):
                adapter._call_with_retry("b64", "p2")
        assert len(calls) == 1


class TestOcrIsMeasuredNotAsserted:
    def test_a_mathpix_call_raises_and_is_counted(self):
        """An earlier runner recorded a hardcoded 0 while billing per sample."""
        import recognizer.mathpix.mathpix_image_ocr as mathpix_module

        with attribution.forbid_paid_ocr() as attempts:
            with pytest.raises(attribution.OcrCallAttempted):
                mathpix_module.requests.post("https://api.mathpix.com", json={})
            assert attempts[0] == 1

    def test_transport_is_restored_afterwards(self):
        import recognizer.mathpix.mathpix_image_ocr as mathpix_module

        original = mathpix_module.requests.post
        with attribution.forbid_paid_ocr():
            assert mathpix_module.requests.post is not original
        assert mathpix_module.requests.post is original


class TestExperimentIdentity:
    def test_identity_hash_excludes_the_timestamp(self):
        """A guard hashing its own timestamp rejects its own resume and re-bills."""
        assert "created_at_utc" not in attribution.EXPERIMENT_IDENTITY_FIELDS

    def test_same_roster_yields_the_same_identity(self, frozen_files):
        a = _plan(frozen_files)
        b = _plan(frozen_files)
        assert a["created_at_utc"] != b["created_at_utc"] or True
        assert a["plan_sha256"] == b["plan_sha256"]

    def test_changing_the_roster_changes_the_identity(self, frozen_files):
        a = _plan(frozen_files, roster=_roster(4))
        b = _plan(frozen_files, roster=_roster(5))
        assert a["plan_sha256"] != b["plan_sha256"]

    def test_records_file_is_part_of_the_identity(self, frozen_files, tmp_path):
        records, ocr = frozen_files
        other = tmp_path / "other_records.json"
        other.write_text(json.dumps([{"sample_id": "different"}]), encoding="utf-8")
        a = attribution.build_plan("pilot", _roster(), records_path=records, ocr_path=ocr)
        b = attribution.build_plan("pilot", _roster(), records_path=other, ocr_path=ocr)
        assert a["plan_sha256"] != b["plan_sha256"]

    def test_review_flags_an_identity_hash_that_includes_time(self, monkeypatch, frozen_files):
        monkeypatch.setattr(
            attribution, "EXPERIMENT_IDENTITY_FIELDS",
            attribution.EXPERIMENT_IDENTITY_FIELDS + ("created_at_utc",))
        assert "identity_hash_includes_timestamp" in attribution.review_plan(
            _plan(frozen_files))["blocking_findings"]


class TestRosterIsAnswerIndependent:
    def test_ordering_key_depends_only_on_the_sample_id(self):
        a = attribution.roster_order_key("plotqa_test_1_x_1")
        b = attribution.roster_order_key("plotqa_test_1_x_1")
        c = attribution.roster_order_key("plotqa_test_2_x_2")
        assert a == b and a != c

    def test_roster_is_reproducible_and_is_a_prefix(self, tmp_path: Path):
        source = tmp_path / "plan.json"
        samples = [
            {"sample_id": f"s{i}", "question": "q", "answer": float(i),
             "answer_type": "numeric", "image_path": "i.png",
             "metadata": {"qa_template": "min_max", "qa_type": "vbar"}}
            for i in range(20)
        ]
        source.write_text(json.dumps({"selected_samples": samples}), encoding="utf-8")
        first = attribution.build_roster(source, 5)
        again = attribution.build_roster(source, 5)
        assert [s["sample_id"] for s in first] == [s["sample_id"] for s in again]
        bigger = attribution.build_roster(source, 8)
        assert [s["sample_id"] for s in bigger][:5] == [s["sample_id"] for s in first]

    def test_roster_keeps_only_the_frozen_strata(self, tmp_path: Path):
        source = tmp_path / "plan.json"
        samples = [
            {"sample_id": "keep1", "question": "q", "answer": 1.0,
             "answer_type": "numeric", "image_path": "i.png",
             "metadata": {"qa_template": "arithmetic"}},
            {"sample_id": "keep2", "question": "q", "answer": 1.0,
             "answer_type": "numeric", "image_path": "i.png",
             "metadata": {"qa_template": "min_max"}},
            {"sample_id": "drop_text", "question": "q", "answer": "yes",
             "answer_type": "text", "image_path": "i.png",
             "metadata": {"qa_template": "min_max"}},
            {"sample_id": "drop_tmpl", "question": "q", "answer": 1.0,
             "answer_type": "numeric", "image_path": "i.png",
             "metadata": {"qa_template": "comparison"}},
        ]
        source.write_text(json.dumps({"selected_samples": samples}), encoding="utf-8")
        ids = {s["sample_id"] for s in attribution.build_roster(source, 10)}
        assert ids == {"keep1", "keep2"}

    def test_duplicate_roster_entries_are_a_blocking_finding(self, frozen_files):
        roster = _roster(2)
        plan = _plan(frozen_files, roster=roster + roster[:1])
        assert "roster_contains_duplicates" in attribution.review_plan(plan)["blocking_findings"]


class TestYearRuleIsApplied:
    def test_year_rows_are_detected_in_the_plan(self, frozen_files):
        roster = [{
            "sample_id": "y1",
            "question": "In which year was the value minimum?",
            "answer": 2009, "answer_type": "numeric", "image_path": "i.png",
            "metadata": {"qa_template": "min_max", "qa_type": "vbar"},
        }]
        plan = _plan(frozen_files, roster=roster)
        assert plan["year_rule"]["applies"] is True
        assert plan["year_rule"]["year_answer_sample_ids"] == ["y1"]

    def test_a_year_answer_is_scored_exactly_not_by_tolerance(self):
        """Off-by-one on a year must be wrong; the generic tolerance allows it."""
        sample = {
            "sample_id": "y1",
            "question": "In which year was the value minimum?",
            "answer": 2009, "answer_type": "numeric",
            "metadata": {"qa_template": "min_max", "qa_type": "vbar"},
        }
        raw = json.dumps({"raw_text": "2010", "normalized_prediction": "2010"})
        row = attribution.score_row(sample, "G_measured_raw", raw, "success", "",
                                    {"facts": 0, "facts_with_both_labels": 0,
                                     "both_label_share": None}, 1)
        assert row["correct"] is False
        assert row["tolerance"] == 0.0
        assert row["year_answer"] is True

    def test_a_correct_year_is_scored_correct(self):
        sample = {
            "sample_id": "y1",
            "question": "In which year was the value minimum?",
            "answer": 2009, "answer_type": "numeric",
            "metadata": {"qa_template": "min_max", "qa_type": "vbar"},
        }
        raw = json.dumps({"raw_text": "2009", "normalized_prediction": "2009"})
        row = attribution.score_row(sample, "G_measured_raw", raw, "success", "",
                                    {"facts": 0, "facts_with_both_labels": 0,
                                     "both_label_share": None}, 1)
        assert row["correct"] is True


class TestEmptyOutputHandling:
    def test_an_empty_body_is_an_error_not_a_wrong_answer(self):
        sample = {"sample_id": "s", "question": "q", "answer": 1.0,
                  "answer_type": "numeric",
                  "metadata": {"qa_template": "arithmetic", "qa_type": "vbar"}}
        row = attribution.score_row(sample, "G_measured_raw", "  ", "success", "",
                                    {"facts": 0, "facts_with_both_labels": 0,
                                     "both_label_share": None}, 1)
        assert row["status"] == "error"
        assert row["error"] == "empty_model_output"
        assert row["correct"] is None

    def test_empty_rows_are_counted_per_arm_in_the_summary(self):
        rows = [
            {"condition": "G_measured_raw", "status": "error",
             "error": "empty_model_output", "correct": None},
            {"condition": "G_measured_raw", "status": "success", "correct": True},
        ]
        summary = attribution.summarize(rows)
        assert summary["by_condition"]["G_measured_raw"]["empty_output_rows"] == 1
        assert summary["by_condition"]["G_measured_raw"]["accuracy"] == 1.0


class TestArmSymmetry:
    def test_baseline_gets_no_evidence_and_the_others_do(self):
        """The baseline prompt must not contain the evidence at all.

        Asserted on a value that cannot occur incidentally, because an `in`
        check against a short string like "1.0" would pass on any prompt that
        happened to contain those characters.
        """
        sentinel = 8675309.4211
        facts = {"measured_facts": [{"measured_value": sentinel}]}
        base = attribution.build_prompt("E0_plotqa_direct", "Q?", facts)
        meas = attribution.build_prompt("G_measured_raw", "Q?", facts)
        flat = attribution.build_prompt("Measured_flat_answer_baseline", "Q?", facts)
        assert str(sentinel) not in base
        assert str(sentinel) in meas
        assert str(sentinel) in flat

    def test_evidence_only_arm_is_told_not_to_use_the_image(self):
        facts = {"measured_facts": []}
        flat = attribution.build_prompt("Measured_flat_answer_baseline", "Q?", facts)
        assert "do not infer from the image" in flat.lower()

    def test_all_arms_share_the_same_output_schema_instruction(self):
        facts = {"measured_facts": []}
        prompts = [attribution.build_prompt(c, "Q?", facts)
                   for c in attribution.CONDITIONS]
        for p in prompts:
            assert "normalized_prediction" in p

    def test_conditions_reuse_the_historical_identifiers(self):
        assert attribution.CONDITIONS == (
            "E0_plotqa_direct", "G_measured_raw", "Measured_flat_answer_baseline")

    def test_retired_condition_is_not_run(self):
        assert "G_full_measured_geosym" not in attribution.CONDITIONS


class TestLabelCompletenessCovariate:
    def test_share_counts_only_facts_with_both_labels(self):
        facts = {"measured_facts": [
            {"series_label": "a", "category_label": "1"},
            {"series_label": "a"},
            {"category_label": "1"},
            {},
        ]}
        out = attribution.label_completeness(facts)
        assert out["facts"] == 4
        assert out["facts_with_both_labels"] == 1
        assert out["both_label_share"] == 0.25

    def test_empty_payload_reports_none_not_zero(self):
        out = attribution.label_completeness({"measured_facts": []})
        assert out["both_label_share"] is None


class TestPlanRecordsWhatReviewersNeed:
    def test_plan_pins_prompt_sources_by_hash(self, frozen_files):
        hashes = _plan(frozen_files)["prompt_source_hashes"]
        assert hashes["scripts/run_plotqa_locate.py"]
        assert len(next(iter(hashes.values()))) == 64

    def test_plan_states_it_is_not_a_victory_run(self, frozen_files):
        plan = _plan(frozen_files)
        assert "may be reported as such" in plan["attribution_not_victory"]

    def test_falsification_requires_significance_and_direction(self, frozen_files):
        shared = _plan(frozen_files)["falsification_signals"]["shared_rules"]
        assert "alpha=0.05" in shared
        assert "significance AND the stated direction" in shared
        assert "invariant 6" in shared

    def test_temperature_is_zero_and_there_is_no_judge(self, frozen_files):
        provenance = _plan(frozen_files)["model_provenance"]
        assert provenance["temperature"] == 0.0
        assert provenance["judge_model"] is None

    def test_model_id_is_not_hardcoded_in_the_plan(self, frozen_files):
        provenance = _plan(frozen_files)["model_provenance"]
        assert provenance["model_id_source"].startswith("config/default_config.yaml")


class TestUnparseableAnswersDoNotCrashTheRun:
    """The first confirmation attempt died at row 124 on this exact input.

    `extract_prediction` documents a None return for an unparseable body, but it
    reaches `normalize_answer`, which raises when a numeric question is answered
    with a bare category name. An unhandled raise loses the run; the rows already
    on disk survive, but the runner stops mid-experiment.
    """

    CATEGORY_ANSWER_TO_NUMERIC_QUESTION = json.dumps({
        "raw_text": "Japan has the minimum number of refugees by country of origin.",
        "normalized_prediction": "Japan",
    })

    def _sample(self):
        return {
            "sample_id": "plotqa_test_7736_284007_0055d038",
            "question": "Across all countries, what is the minimum number of refugees by country of origin ?",
            "answer": "185",
            "answer_type": "numeric",
            "metadata": {"qa_template": "min_max", "qa_type": "hbar"},
        }

    def test_a_category_name_answering_a_numeric_question_scores_wrong(self):
        row = attribution.score_row(
            self._sample(), "G_measured_raw",
            self.CATEGORY_ANSWER_TO_NUMERIC_QUESTION, "success", "",
            {"facts": 0, "facts_with_both_labels": 0, "both_label_share": None}, 1)
        assert row["status"] == "success"
        assert row["correct"] is False
        assert row["parseable"] is False
        assert row["normalized_prediction"] is None

    def test_an_unparseable_body_is_not_recorded_as_a_failed_call(self):
        """Excluding it would drop a real wrong answer from the contrast."""
        row = attribution.score_row(
            self._sample(), "G_measured_raw",
            self.CATEGORY_ANSWER_TO_NUMERIC_QUESTION, "success", "",
            {"facts": 0, "facts_with_both_labels": 0, "both_label_share": None}, 1)
        assert row["status"] != "error"
        assert row["error"] == ""

    def test_a_parseable_numeric_answer_is_still_marked_parseable(self):
        raw = json.dumps({"raw_text": "185", "normalized_prediction": "185"})
        row = attribution.score_row(
            self._sample(), "G_measured_raw", raw, "success", "",
            {"facts": 0, "facts_with_both_labels": 0, "both_label_share": None}, 1)
        assert row["parseable"] is True
        assert row["correct"] is True
