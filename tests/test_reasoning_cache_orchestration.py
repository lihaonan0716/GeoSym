"""Offline orchestration test for extrema-channel cache provenance.

`test_extrema_channel.py` covers the staleness *decision* as a pure function.
This file covers the thing that decision exists to control: what
`run_reasoning_all_conditions` actually does with a cache when the flag flips.
The two are separable, and the unit tests passing while the wiring is wrong is
exactly the failure mode worth guarding — a flag change that silently reuses a
cached E5 produces a record that claims an experimental condition it never ran.

Every paid and I/O boundary is stubbed, so this makes **no API calls**: no
adapter, no judge, no OCR, no image load, no GeoSym extraction, no disk cache.
`run_single_condition` is replaced by a recorder, which is what lets the test
assert *which* conditions were re-run rather than merely inspecting the output.
"""

import pytest

from app.reasoning_pipeline import EXTREMA_CHANNEL_ENV_VAR


PARABOLA = [(-2, 0), (-1, 3), (0, 4), (1, 3), (2, 0)]

# The cache on disk still holds E5 records; only E5_raw and E0-E4 can run now.
ALL_CONDITIONS = ["E0", "E1", "E2", "E3", "E4", "E5", "E5_raw"]
LIVE_CONDITIONS = [ck for ck in ALL_CONDITIONS if ck != "E5"]


def _cached_record(*, extrema_flag=None, judge_model="judge-x"):
    """A full success-in-every-condition cache, as an earlier run would leave it.

    ``extrema_flag=None`` omits the per-condition field entirely, reproducing the
    460 archived records written before the flag existed.
    """
    conditions = {}
    for ck in ALL_CONDITIONS:
        cond = {
            "condition_id": ck,
            "status": "success",
            "cached": False,
            "elapsed_ms": 1,
            "retry_count": 0,
            "error": None,
            "eval": {"total_score": 80.0, "dimension_scores": {}},
            "parsed": {},
            "judge": None,
            "pipeline": {},
            "md": f"**{ck}** cached",
        }
        if extrema_flag is not None:
            cond["extrema_channel_enabled"] = extrema_flag
        conditions[ck] = cond
    return {
        "judge_model": judge_model,
        "conditions": conditions,
        "geosym_artifacts": {"success": True, "coords": PARABOLA, "visual_facts": None},
    }


@pytest.fixture
def harness(monkeypatch):
    """Stub every external boundary and return a runner plus a call log."""
    import app.reasoning as R

    gt_data = {
        "expression": "4 - x^2",
        "domain": "[-2, 2]",
        "range": "[0, 4]",
        "coordinates": PARABOLA,
        "intercepts": {},
        "extrema": [],
        "prediction_targets": {},
        "rate_of_change": {},
        "monotonicity": [],
        "symmetry": "y-axis",
    }

    ran: list = []
    saved: dict = {}
    visual_facts_seen: dict = {}

    monkeypatch.setattr(
        R, "load_image_and_gt",
        lambda name: ({}, {"question": {"ground_truth": gt_data}}),
    )
    monkeypatch.setattr(R, "_resolve_image_path", lambda name: "img.png")
    monkeypatch.setattr(R, "_create_llm_adapter", lambda *a, **k: object())
    monkeypatch.setattr(R, "ensure_judge_adapter", lambda *a, **k: None)
    monkeypatch.setattr(R, "load_layer_config", lambda: {})
    monkeypatch.setattr(R, "_get_saved_key", lambda *a, **k: "key")
    # Would otherwise call Mathpix at construction/use time.
    monkeypatch.setattr(
        "evaluator.baselines.ocr_rule_baseline.OCRRuleBaseline.extract_coordinates",
        lambda self, path, **k: [],
    )

    def _fake_run_single_condition(adapter, img, gt, mode, **kwargs):
        ran.append(R.MODE_TO_SHORT.get(mode, mode) if hasattr(R, "MODE_TO_SHORT") else mode)
        if "visual_facts" in kwargs:
            visual_facts_seen[mode] = kwargs["visual_facts"]
        return {"final_parsed": {}, "geosym_intervention": None}

    monkeypatch.setattr(R, "run_single_condition", _fake_run_single_condition)
    # Imported inside the function, so it has to be patched at its own module.
    monkeypatch.setattr(
        "evaluator.metrics.reasoning.evaluate_comprehensive",
        lambda *a, **k: {"total_score": 90.0},
    )
    monkeypatch.setattr(R, "_format_path_result", lambda **k: "ok")
    # Never touch disk: the real extractor is a paid multi-step pipeline.
    monkeypatch.setattr(
        R, "extract_geosym_coords", lambda *a, **k: (PARABOLA, None, True),
    )

    def _run(cached, *, flag, force_rerun=False, target_conditions=None):
        monkeypatch.setenv(EXTREMA_CHANNEL_ENV_VAR, "true" if flag else "false")
        monkeypatch.setattr(
            "app.reasoning_cache.load_reasoning_cache", lambda *a, **k: cached,
        )
        monkeypatch.setattr(
            "app.reasoning_cache.save_reasoning_cache",
            lambda record: saved.update(record) or "path",
        )
        ran.clear()
        saved.clear()
        visual_facts_seen.clear()
        for _ in R.run_reasoning_all_conditions(
            image_name="img.png",
            model_name="model-x",
            api_key="key",
            judge_model_name="judge-x",
            judge_api_key="key",
            lang="en",
            force_rerun=force_rerun,
            target_conditions=target_conditions,
        ):
            pass
        return saved

    return _run, ran, visual_facts_seen


def _modes_to_conditions(ran):
    from app.reasoning_pipeline import MODE_TO_SHORT
    return {MODE_TO_SHORT.get(m, m) for m in ran}


def test_unchanged_flag_reruns_nothing(harness):
    """Baseline: a matching flag must leave the whole cache intact."""
    run, ran, _ = harness

    record = run(_cached_record(extrema_flag=False), flag=False)

    assert _modes_to_conditions(ran) == set()
    # Includes the retired E5: it is cached, not runnable, and must be carried.
    assert set(record["summary"]["cached_conditions"]) == set(ALL_CONDITIONS)


# NOTE (2026-09-19): five tests asserting that flipping the extrema flag re-runs
# E5 were removed with M1/M2. E5 was the only condition the flag could reach
# (`EXTREMA_CHANNEL_CONDITIONS`), and it is no longer runnable, so the re-run
# they described cannot occur. The surviving cache-preservation guarantees are
# re-expressed below against a live condition and a stale cached E5.


def test_flipping_the_flag_reruns_nothing_now_that_e5_is_retired(harness):
    """The extrema flag no longer reaches any runnable condition.

    It gated E5 alone. With E5 retired, flipping it must cost nothing -- if this
    ever fails, the flag has been wired into a live condition and that is a
    prompt change to the main arm, not a cache concern.
    """
    run, ran, _ = harness

    record = run(_cached_record(extrema_flag=False), flag=True)

    assert _modes_to_conditions(ran) == set()
    assert set(record["summary"]["cached_conditions"]) == set(ALL_CONDITIONS)


def test_a_stale_cached_e5_is_preserved_not_rerun_or_dropped(harness):
    """Archived E5 rows stay readable after their condition retired.

    The cache still holds E5 records produced under the old pipeline. They must
    survive untouched, keeping the flag they were produced under -- re-running
    is impossible and discarding would destroy paid history.
    """
    run, ran, _ = harness

    record = run(_cached_record(extrema_flag=False), flag=True)

    assert "E5" not in _modes_to_conditions(ran)
    assert record["conditions"]["E5"]["extrema_channel_enabled"] is False
    assert record["conditions"]["E5"]["md"] == "**E5** cached"


def test_targeted_rerun_of_one_condition_leaves_the_others_cached(harness):
    """A targeted run must not spend money on conditions out of scope."""
    run, ran, _ = harness

    record = run(
        _cached_record(extrema_flag=False), flag=True,
        force_rerun=True, target_conditions=["E0"],
    )

    assert _modes_to_conditions(ran) == {"E0"}
    for ck in set(LIVE_CONDITIONS) - {"E0"}:
        assert record["conditions"][ck]["cached"] is True, ck


def test_a_string_flag_is_not_read_as_true(harness):
    """`bool("false")` is True, so truthiness here would re-run all of E5.

    A record carrying the raw env-var string must be treated as flag-absent,
    i.e. flag-off, not as flag-on.
    """
    run, ran, _ = harness
    cached = _cached_record(extrema_flag=False)
    cached["conditions"]["E5"]["extrema_channel_enabled"] = "false"

    run(cached, flag=False)

    assert _modes_to_conditions(ran) == set()
