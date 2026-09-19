"""Tests for the M1 extrema evidence channel (P0-1 / P0-1b).

Background: `compute_visual_facts_from_coords` hard-coded its extrema slot to
`[]`, so no extrema evidence ever reached the reasoning prompts, while a
working candidate generator sat unused in `evaluator.reasoning.intervention`.
P0-1b exposes that generator as `infer_extrema_candidates`; P0-1 wires it in
behind a default-off feature flag.

Two things are asserted here, and they are different in kind:

  * Flag OFF is a *contract*: the output must stay structurally identical to
    every run recorded so far. These assertions must never be relaxed.
  * Flag ON is a *characterization*: the current unfiltered candidates are
    locked in as-is, including the known false positives across
    discontinuities. These tests document present behaviour; they do not
    claim it is correct. Rejecting discontinuity crossings is P0-2 work, and
    the corresponding assertion here is expected to change when that lands.

The last section covers flag provenance rather than the channel itself: because
the flag changes E5's prompt, a cached E5 from the opposite setting must be
re-run, and each condition must record which setting produced it.
"""

import pytest

from app.reasoning_pipeline import (
    CONDITION_ORDER,
    EXTREMA_CHANNEL_ENV_VAR,
    compute_visual_facts_from_coords,
    extrema_channel_enabled,
)
from app.reasoning_schema import (
    build_condition_record,
    build_reasoning_record,
    condition_extrema_flag,
    stale_extrema_conditions,
    validate_reasoning_record,
)
from evaluator.reasoning.intervention import (
    classify_extrema_candidates,
    infer_extrema_candidates,
)


class _FakeRecognizer:
    result = {"success": False, "error": "tick labels missing"}

    def recognize_intersections(self, *_args, **_kwargs):
        return [], self.result, None


# A downward parabola sampled on the integer grid: y = 4 - x^2, one true max.
PARABOLA = [(-2, 0), (-1, 3), (0, 4), (1, 3), (2, 0)]

# Strictly increasing line: y = x, no extremum anywhere.
LINE = [(-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2)]

# y = 1/x sampled either side of the pole at x = 0. Nothing here is a turning
# point mathematically -- the branches are each monotone -- but the discrete
# three-point test sees the sign flip as one.
HYPERBOLA_ACROSS_POLE = [(-2, -1), (-1, -2), (1, 2), (2, 1)]

# A true maximum with wider regular sampling and large but smooth endpoint
# changes. This is a negative control against over-dropping genuine extrema
# merely because the chart is steep or coarsely sampled.
STEEP_PARABOLA_REGULAR_GAPS = [(-4, -12), (-2, 0), (0, 4), (2, 0), (4, -12)]


@pytest.fixture
def extrema_flag(monkeypatch):
    """Set the extrema channel flag for the duration of one test."""

    def _set(enabled: bool):
        monkeypatch.setenv(EXTREMA_CHANNEL_ENV_VAR, "true" if enabled else "false")

    return _set


# ── Case 1: flag off — the behaviour contract ─────────────────────────

def test_flag_defaults_to_off(monkeypatch):
    monkeypatch.delenv(EXTREMA_CHANNEL_ENV_VAR, raising=False)
    assert extrema_channel_enabled() is False


def test_geosym_extraction_surfaces_structured_failure_reason(monkeypatch, tmp_path):
    import app.reasoning_pipeline as pipeline

    monkeypatch.setattr(
        "evaluator.adapters.recognizer.RecognizerAdapter",
        lambda: _FakeRecognizer(),
    )

    coords, facts, success, reason = pipeline.extract_geosym_coords(
        "model", "key", tmp_path / "chart.png", {}, gemini_enabled=False
    )

    assert coords == []
    assert facts is None
    assert success is False
    assert reason == "tick labels missing"


def test_geosym_extraction_surfaces_exception_reason(monkeypatch, tmp_path):
    import app.reasoning_pipeline as pipeline

    class RaisingRecognizer:
        def recognize_intersections(self, *_args, **_kwargs):
            raise RuntimeError("chart channel unavailable")

    monkeypatch.setattr(
        "evaluator.adapters.recognizer.RecognizerAdapter",
        lambda: RaisingRecognizer(),
    )

    coords, facts, success, reason = pipeline.extract_geosym_coords(
        "model", "key", tmp_path / "chart.png", {}, gemini_enabled=False
    )

    assert coords == []
    assert facts is None
    assert success is False
    assert reason == "RuntimeError: chart channel unavailable"


def test_flag_off_yields_empty_extrema_even_when_candidates_exist(extrema_flag):
    """The parabola has an obvious max; with the flag off it must not appear."""
    extrema_flag(False)

    facts = compute_visual_facts_from_coords(PARABOLA)

    assert facts["extrema"] == []
    # Sanity check that the emptiness is the flag's doing, not a lack of signal.
    assert infer_extrema_candidates(PARABOLA) != []


def test_flag_off_preserves_field_set_and_order(extrema_flag):
    """Structural equivalence with the archived runs, field by field."""
    extrema_flag(False)

    facts = compute_visual_facts_from_coords(PARABOLA)

    assert list(facts.keys()) == ["intercepts", "extrema", "monotonicity", "source"]
    assert list(facts["intercepts"].keys()) == ["x_intercepts", "y_intercept"]
    assert facts["source"] == "computation_grounded"


def test_flag_off_leaves_other_categories_untouched(extrema_flag):
    """Toggling the flag must change the extrema slot and nothing else."""
    extrema_flag(False)
    off = compute_visual_facts_from_coords(PARABOLA)
    extrema_flag(True)
    on = compute_visual_facts_from_coords(PARABOLA)

    assert list(off.keys()) == list(on.keys())
    for field in ("intercepts", "monotonicity", "source"):
        assert off[field] == on[field], f"flag leaked into {field}"
    assert off["extrema"] != on["extrema"]


# ── Case 2: correct extrema ───────────────────────────────────────────

def test_flag_on_finds_the_true_maximum(extrema_flag):
    extrema_flag(True)

    facts = compute_visual_facts_from_coords(PARABOLA)

    assert facts["extrema"] == [{"x": 0, "y": 4, "type": "max"}]


# ── Case 3: no extrema ────────────────────────────────────────────────

def test_flag_on_reports_none_for_a_monotone_function(extrema_flag):
    extrema_flag(True)

    facts = compute_visual_facts_from_coords(LINE)

    assert facts["extrema"] == []


def test_too_few_points_returns_none_regardless_of_flag(extrema_flag):
    """`compute_visual_facts_from_coords` needs >= 2 points; unchanged here."""
    for enabled in (False, True):
        extrema_flag(enabled)
        assert compute_visual_facts_from_coords([(0, 0)]) is None
        assert compute_visual_facts_from_coords([]) is None


# ── Case 4: discontinuity crossing (P0-2 filter contract) ─────────────

def test_flag_off_suppresses_the_spurious_discontinuity_candidate(extrema_flag):
    """The contract half: whatever the generator thinks, flag off emits `[]`."""
    extrema_flag(False)

    assert compute_visual_facts_from_coords(HYPERBOLA_ACROSS_POLE)["extrema"] == []


def test_flag_on_drops_spurious_extrema_across_a_pole(extrema_flag):
    extrema_flag(True)

    facts = compute_visual_facts_from_coords(HYPERBOLA_ACROSS_POLE)

    assert facts["extrema"] == []


def test_extrema_filter_records_drop_reasons_for_discontinuity_bridge():
    decisions = classify_extrema_candidates(HYPERBOLA_ACROSS_POLE)

    assert [decision["decision"] for decision in decisions] == ["DROP", "DROP"]
    assert all("discontinuity_bridge" in decision["reason"] for decision in decisions)


def test_extrema_filter_keeps_true_extrema_with_regular_large_gaps():
    decisions = classify_extrema_candidates(STEEP_PARABOLA_REGULAR_GAPS)

    assert decisions == [
        {
            "candidate": {"x": 0, "y": 4, "type": "max"},
            "decision": "KEEP",
            "rule": "p0-2-discontinuity-bridge-v1",
            "reason": "local_turn_without_discontinuity_bridge",
            "metrics": {
                "typical_x_gap": 2.0,
                "typical_y_jump": 8.0,
                "left_x_gap": 2,
                "left_y_jump": 4,
                "right_x_gap": 2,
                "right_y_jump": 4,
            },
        }
    ]


# ── Case 5: duplicate / unordered coordinates ─────────────────────────

def test_candidates_are_invariant_to_input_order_and_duplicates():
    shuffled_with_dupes = [(1, 3), (0, 4), (-2, 0), (0, 4), (2, 0), (-1, 3), (1, 3)]

    assert infer_extrema_candidates(shuffled_with_dupes) == infer_extrema_candidates(PARABOLA)


def test_flag_on_handles_unordered_input_with_duplicates(extrema_flag):
    extrema_flag(True)

    facts = compute_visual_facts_from_coords(sorted(PARABOLA, reverse=True))

    assert facts["extrema"] == [{"x": 0, "y": 4, "type": "max"}]


# ── P0-2: the public interface returns filtered candidates ────────────

def test_public_interface_returns_kept_candidates_from_classifier():
    from evaluator.reasoning.intervention import _infer_extrema, normalize_coords

    assert _infer_extrema(normalize_coords(HYPERBOLA_ACROSS_POLE)) != []
    assert infer_extrema_candidates(HYPERBOLA_ACROSS_POLE) == []
    assert infer_extrema_candidates(PARABOLA) == [
        decision["candidate"]
        for decision in classify_extrema_candidates(PARABOLA)
        if decision["decision"] == "KEEP"
    ]


def test_public_interface_returns_only_the_candidate_schema():
    for candidate in infer_extrema_candidates(PARABOLA):
        assert set(candidate.keys()) == {"x", "y", "type"}
        assert candidate["type"] in {"max", "min"}


# NOTE (2026-09-19): five tests covering `format_tiered_facts_for_prompt`
# ("P0-1c: the tiered formatter's second gate") were removed here. They
# asserted on E5's tiered injection, which retired with M1/M2 — the formatter
# and the `visual_facts_full` profile they imported no longer exist. The
# extrema-candidate generation they guarded is still covered above; what was
# dropped is the E5 injection surface, not extrema coverage.


# ── Flag provenance across the condition cache ────────────────────────
#
# The flag decides what E5's prompt contains, so a cached E5 produced under the
# opposite setting is not reusable. Recording the process-level flag alone was
# not enough: a run could reuse a flag-off E5 and then store
# `extrema_channel_enabled=true` beside it, making the record claim an
# experimental condition it never ran. These tests pin down the invalidation
# rule and the per-condition provenance field that together prevent that.


def _cached(flags: dict) -> dict:
    """A minimal cache stub: condition id → the flag it was produced under.

    ``None`` means the field is absent entirely, as in every record archived
    before the flag existed.
    """
    conditions = {}
    for ck, flag in flags.items():
        cond = {"status": "success"}
        if flag is not None:
            cond["extrema_channel_enabled"] = flag
        conditions[ck] = cond
    return {"conditions": conditions}


def test_pre_flag_records_are_read_as_flag_off():
    """The 460 archived runs have no such field, and they were all flag-off.

    Defaulting to False is a fact about those runs, not a convenience: the
    extrema slot was hard-coded to `[]` when they were produced.
    """
    assert condition_extrema_flag({"status": "success"}) is False
    assert condition_extrema_flag(None) is False
    assert stale_extrema_conditions(_cached({"E5": None}), False) == set()


def test_pre_flag_record_is_stale_once_the_channel_is_on():
    """Turning the flag on must not silently reuse a pre-flag E5."""
    assert stale_extrema_conditions(_cached({"E5": None}), True) == {"E5"}


def test_flag_off_to_on_invalidates_e5():
    assert stale_extrema_conditions(_cached({"E5": False}), True) == {"E5"}


def test_flag_on_to_off_invalidates_e5():
    """Symmetric: reverting must not leave an experimental E5 in a control run."""
    assert stale_extrema_conditions(_cached({"E5": True}), False) == {"E5"}


def test_unchanged_flag_keeps_the_cache():
    assert stale_extrema_conditions(_cached({"E5": True}), True) == set()
    assert stale_extrema_conditions(_cached({"E5": False}), False) == set()


def test_only_e5_is_invalidated_by_a_flag_change():
    """E5_raw passes `visual_facts=None` and E0-E4 never see GeoSym facts.

    Re-running them on a flag change would cost real API spend for no
    information, so the invalidation must stay surgical.
    """
    cache = _cached({ck: False for ck in ("E0", "E1", "E2", "E3", "E4", "E5", "E5_raw")})

    assert stale_extrema_conditions(cache, True) == {"E5"}


def test_missing_or_malformed_cache_yields_nothing_stale():
    for bad in (None, {}, {"conditions": {}}, {"conditions": None}, "not a dict"):
        assert stale_extrema_conditions(bad, True) == set()


def test_condition_record_carries_its_own_flag():
    """Per-condition provenance, so a mixed record can still be read correctly."""
    record = build_condition_record(
        condition_id="E5",
        label="E5",
        status="success",
        cached=True,
        elapsed_ms=0,
        retry_count=0,
        error=None,
        eval_result=None,
        parsed=None,
        judge=None,
        pipeline=None,
        md="",
        extrema_channel_enabled=True,
    )

    assert record["extrema_channel_enabled"] is True


def test_condition_record_flag_defaults_to_off():
    """Callers that predate the field keep producing valid flag-off records."""
    record = build_condition_record(
        condition_id="E0",
        label="E0",
        status="success",
        cached=False,
        elapsed_ms=0,
        retry_count=0,
        error=None,
        eval_result=None,
        parsed=None,
        judge=None,
        pipeline=None,
        md="",
    )

    assert record["extrema_channel_enabled"] is False


def test_the_new_field_does_not_invalidate_archived_records():
    """`validate_reasoning_record` must not require it.

    Making it mandatory would reject every record archived before the flag
    existed, discarding caches that cost real API spend.
    """
    conditions = {}
    for ck in CONDITION_ORDER:
        cond = build_condition_record(
            condition_id=ck, label=ck, status="success", cached=False,
            elapsed_ms=0, retry_count=0, error=None, eval_result=None,
            parsed=None, judge=None, pipeline=None, md="",
        )
        cond.pop("extrema_channel_enabled")  # as archived, before the field existed
        conditions[ck] = cond

    record = build_reasoning_record(
        image_name="x.png", image_path="x.png", reasoning_model="m", judge_model="j",
        started_at="t", finished_at="t", duration_ms=0, requested_conditions=[],
        run_settings={}, ground_truth_snapshot={}, score_table_markdown="",
        cached_conditions=[], geosym_artifacts={}, conditions=conditions,
    )

    assert validate_reasoning_record(record) == []
