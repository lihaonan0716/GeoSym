"""Tests for the R3-3 JSON-syntax repair rescore.

The rescore decides whether the layering claim survives, so its own failure
modes matter as much as its result. Two in particular:

  * a repair that alters CONTENT rather than syntax would manufacture the
    answer it is supposed to be measuring;
  * a re-parse that differs from the pipeline's own would move rows the repair
    never touched -- and because the layered and flat arms parse differently,
    that error would be arm-asymmetric, which is the exact confound the
    rescore exists to remove.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Dev-local script (`scripts/*` is gitignored): skip rather than abort the
# suite's collection in a clean clone.
_module = pytest.importorskip(  # noqa: E402
    "scripts.rescore_m3_attribution_json_repair")

JSON_GATED_DIMENSIONS = _module.JSON_GATED_DIMENSIONS
paired_contrast = _module.paired_contrast
parser_provenance = _module.parser_provenance
repair_answer_syntax = _module.repair_answer_syntax
repair_applied = _module.repair_applied
reparse = _module.reparse
reproduction_check = _module.reproduction_check
split_chain_sections = _module.split_chain_sections
verdict = _module.verdict


class TestRepairTouchesSyntaxOnly:
    def test_infinity_glyph_becomes_json_loadable(self):
        assert "∞" not in repair_answer_syntax('{"a": (-∞, 1]}')

    def test_interval_tuple_becomes_a_list(self):
        repaired = repair_answer_syntax('{"interval": (-5, 1], "d": "up"}')
        assert '"interval": [-5, 1]' in repaired

    def test_closed_open_interval_is_also_normalised(self):
        # The opening bracket may be `[` with a `)` close -- probe 6 missed
        # this form and under-counted the recoverable rows.
        repaired = repair_answer_syntax('{"interval": [1, 5)}')
        assert '"interval": [1, 5]' in repaired

    def test_numbers_are_never_altered(self):
        text = '{"rate_of_change": 0.3333, "predictions": {"1.5": -2.75}}'
        assert repair_answer_syntax(text) == text

    def test_an_expression_is_never_altered(self):
        text = '{"expression": "2*abs(x-1) - 4", "domain": "(-5, 5)"}'
        assert repair_answer_syntax(text) == text

    def test_a_plain_parenthesised_value_is_not_touched(self):
        # Only the `interval` key is rewritten. A domain string keeps its
        # brackets, because there the open/closed distinction is meaningful.
        assert repair_answer_syntax('{"domain": "(-inf, 3]"}') == '{"domain": "(-inf, 3]"}'

    def test_a_glyph_inside_prose_is_not_a_repair(self):
        # An explanation mentioning the range "(-∞, -1]" needs no repair: the
        # glyph sits inside a JSON string and loads fine. Treating it as a
        # repair mislabels the row's provenance -- this script's first version
        # flagged 133 such rows.
        text = '{"inference_explanation": "the range is (-∞, -1] overall"}'
        assert repair_applied(text) is False

    def test_repair_is_idempotent(self):
        once = repair_answer_syntax('{"interval": (-∞, 1]}')
        assert repair_answer_syntax(once) == once

    def test_repair_applied_is_false_for_clean_text(self):
        assert repair_applied('{"interval": [-5, 1]}') is False

    def test_repair_applied_is_true_for_broken_text(self):
        assert repair_applied('{"interval": (-5, 1]}') is True

    def test_a_glyph_in_prose_outside_the_json_fence_is_not_a_repair(self):
        # Layered chains interleave explanation with fenced JSON. Only the
        # fenced part is parsed, so a glyph in the narration changes nothing.
        chain = (
            "=== L3 ===\nThe function is decreasing on (-∞, -3].\n\n"
            '```json\n{"expression": "abs(x+3) - 1"}\n```'
        )
        assert repair_applied(chain) is False

    def test_a_broken_interval_inside_the_fence_is_still_a_repair(self):
        chain = (
            "=== L2 ===\nSome explanation with (-∞, 0].\n\n"
            '```json\n{"monotonicity": [{"interval": (-∞, 0], "direction": "up"}]}\n```'
        )
        assert repair_applied(chain) is True


class TestChainSectionsSplitByStep:
    def test_layered_chain_splits_into_three(self):
        chain = "=== L1 ===\na\n\n=== L2 ===\nb\n\n=== L3 ===\nc"
        sections = split_chain_sections(chain)
        assert set(sections) == {"L1", "L2", "L3"}
        assert sections["L2"].strip() == "b"

    def test_flat_chain_yields_its_single_step(self):
        sections = split_chain_sections("=== FlatSingleGeoSym ===\npayload")
        assert list(sections) == ["FlatSingleGeoSym"]

    def test_text_without_a_banner_is_returned_whole(self):
        assert split_chain_sections("bare")[""] == "bare"


class TestReparseMatchesThePipelineLayerSplit:
    """A layered answer is spread over L1/L2/L3, not held in the last block.

    Parsing only the final block silently drops `coordinates` (L1) and the
    three L2 quantities from BOTH layered arms while leaving the flat arm
    complete -- inflating the flat arm by comparison. The first version of the
    rescore did exactly that and reported E5_raw at 37.54 instead of 82.74.
    """

    LAYERED = (
        '=== L1 ===\n```json\n{"coordinates": "(0,1), (1,3)", '
        '"intercepts": {"x_intercepts": [], "y_intercept": 1}}\n```\n\n'
        '=== L2 ===\n```json\n{"predictions": {"0.5": 2}, '
        '"rate_of_change": 2, '
        '"monotonicity": [{"interval": (-∞, 1], "direction": "increasing"}]}\n```\n\n'
        '=== L3 ===\n```json\n{"expression": "2x+1", "domain": "(-inf, inf)", '
        '"range": "(-inf, inf)"}\n```'
    )

    def test_fields_from_every_layer_survive(self):
        parsed = reparse(self.LAYERED)
        assert parsed["coordinates"]
        assert parsed["predictions"] == {"0.5": 2.0}
        assert parsed["rate_of_change"] == 2.0
        assert parsed["expression"] == "2x+1"

    def test_the_repair_recovers_the_bad_interval(self):
        assert reparse(self.LAYERED)["monotonicity"]

    def test_the_parser_now_recovers_it_without_the_script_repair(self):
        # The same repairs were promoted into `evaluator.reasoning.parser`, so
        # the script-level repair is now belt-and-braces for this artifact
        # rather than the only thing standing between a correct answer and a
        # zero. Future runs never produce the defect in the first place.
        assert reparse(self.LAYERED, repair=False)["monotonicity"]

    def test_flat_answer_parses_from_its_single_block(self):
        flat = ('=== FlatSingleGeoSym ===\n```json\n'
                '{"coordinates": "(0,1)", "predictions": {"0.5": 2}, '
                '"rate_of_change": 2, '
                '"monotonicity": [{"interval": (-∞, 1], "direction": "up"}], '
                '"expression": "2x+1"}\n```')
        parsed = reparse(flat)
        assert parsed["expression"] == "2x+1"
        assert parsed["monotonicity"]


class TestReproductionCheckGuardsTheRescore:
    def test_untouched_rows_reproducing_exactly_is_the_pass_condition(self):
        rows = [
            {"rescore_status": "rescored", "repair_applied": False,
             "sample_id": "s", "condition": "E5_raw",
             "original_total_score": 80.0, "repaired_total_score": 80.0},
        ]
        assert reproduction_check(rows)["verdict"] == "reproduces_original"

    def test_a_moved_untouched_row_fails_the_check(self):
        rows = [
            {"rescore_status": "rescored", "repair_applied": False,
             "sample_id": "s", "condition": "E5_raw",
             "original_total_score": 80.0, "repaired_total_score": 37.5},
        ]
        result = reproduction_check(rows)
        assert result["verdict"] == "DOES_NOT_REPRODUCE"
        assert result["mismatches"] == 1

    def test_repaired_rows_are_excluded_from_the_check(self):
        # A repaired row is SUPPOSED to move; including it would make the
        # guard fire on success.
        rows = [
            {"rescore_status": "rescored", "repair_applied": True,
             "sample_id": "s", "condition": "E6_flat_single",
             "original_total_score": 45.0, "repaired_total_score": 90.0},
        ]
        assert reproduction_check(rows)["verdict"] == "reproduces_original"


class TestVerdictFollowsThePreRegisteredRule:
    @staticmethod
    def _contrast(mean, p):
        return {"mean_delta": mean, "paired_t_p": p, "n": 100}

    def test_significant_and_positive_supports_layering(self):
        assert verdict(self._contrast(5.0, 0.001)) == "layering_is_attributable"

    def test_not_significant_is_the_weaker_wording(self):
        assert verdict(self._contrast(0.11, 0.94)) == \
            "no_worse_than_single_pass_and_auditable"

    def test_significant_and_negative_is_reported_as_refutation(self):
        # A loss must not be renamable into a win, nor into a null result.
        assert verdict(self._contrast(-45.0, 1e-40)) == \
            "REFUTED_single_pass_ahead_significantly"

    def test_a_large_but_non_significant_effect_is_not_promoted(self):
        assert verdict(self._contrast(9.0, 0.06)) == \
            "no_worse_than_single_pass_and_auditable"


class TestPairedContrastUsesOnlyCompletePairs:
    ROWS = [
        {"sample_id": "a", "condition": "E5_raw", "rescore_status": "rescored",
         "repaired_total_score": 90.0},
        {"sample_id": "a", "condition": "E6_flat_single", "rescore_status": "rescored",
         "repaired_total_score": 80.0},
        {"sample_id": "b", "condition": "E5_raw", "rescore_status": "rescored",
         "repaired_total_score": 70.0},
        {"sample_id": "b", "condition": "E6_flat_single", "rescore_status": "rescored",
         "repaired_total_score": 75.0},
        # unpaired: present in one arm only, must be dropped
        {"sample_id": "c", "condition": "E5_raw", "rescore_status": "rescored",
         "repaired_total_score": 100.0},
    ]

    def test_unpaired_samples_are_excluded(self):
        result = paired_contrast(self.ROWS, "E5_raw", "E6_flat_single",
                                 "repaired_total_score")
        assert result["n"] == 2

    def test_error_rows_are_excluded(self):
        rows = self.ROWS + [
            {"sample_id": "d", "condition": "E5_raw",
             "rescore_status": "skipped_not_success", "repaired_total_score": None},
            {"sample_id": "d", "condition": "E6_flat_single",
             "rescore_status": "skipped_not_success", "repaired_total_score": None},
        ]
        assert paired_contrast(rows, "E5_raw", "E6_flat_single",
                               "repaired_total_score")["n"] == 2

    def test_direction_is_a_minus_b(self):
        result = paired_contrast(self.ROWS, "E5_raw", "E6_flat_single",
                                 "repaired_total_score")
        assert result["mean_delta"] == pytest.approx(2.5)


class TestParserProvenanceDetectsMixedRows:
    """The artifact is a checkpointed resume that spans the parser fix.

    Rows written after `evaluator/reasoning/parser.py` gained the interval
    repairs never carried the defect, so counting them in the "before repair"
    column would understate it. They are detectable: only the hardened parser
    produces text that WOULD need repair alongside a stored parse that is
    already complete.
    """

    def test_a_post_hardening_row_is_counted(self):
        rows = [{"rescore_status": "rescored", "condition": "E6_flat_single",
                 "repair_applied": True, "json_parsed_before": True}]
        result = parser_provenance(rows)
        assert result["rows_written_by_hardened_parser"] == 1
        assert result["by_arm"] == {"E6_flat_single": 1}

    def test_a_pre_hardening_defective_row_is_not_counted(self):
        # Text needs repair AND the stored parse is missing fields: this is
        # the old parser's signature, which is what the report is measuring.
        rows = [{"rescore_status": "rescored", "condition": "E6_flat_single",
                 "repair_applied": True, "json_parsed_before": False}]
        assert parser_provenance(rows)["rows_written_by_hardened_parser"] == 0

    def test_a_row_needing_no_repair_is_not_counted(self):
        rows = [{"rescore_status": "rescored", "condition": "E5_raw",
                 "repair_applied": False, "json_parsed_before": True}]
        assert parser_provenance(rows)["rows_written_by_hardened_parser"] == 0

    def test_error_rows_are_ignored(self):
        rows = [{"rescore_status": "skipped_not_success", "condition": "E5_raw",
                 "repair_applied": True, "json_parsed_before": True}]
        assert parser_provenance(rows)["rows_written_by_hardened_parser"] == 0


class TestGatedDimensionSet:
    def test_coordinates_is_listed_but_is_not_a_scored_dimension(self):
        # `coordinates` gates the parse, but per invariant 14 it is M0
        # intermediate data and is deliberately not scored. Keeping it in the
        # parse-health set while excluding it from score reasoning is
        # intentional; this test pins that distinction.
        from evaluator.metrics.reasoning import SCORED_DIMENSIONS

        assert "coordinates" in JSON_GATED_DIMENSIONS
        assert "coordinates" not in SCORED_DIMENSIONS
