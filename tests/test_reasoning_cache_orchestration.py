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

ALL_CONDITIONS = ["E0", "E1", "E2", "E3", "E4", "E5", "E5_raw"]


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
    assert set(record["summary"]["cached_conditions"]) == set(ALL_CONDITIONS)


def test_turning_the_flag_on_reruns_only_e5(harness):
    """The core guarantee, end to end.

    E5 must be re-run because its prompt changes; everything else must be reused,
    because re-running it would cost real API spend for no information.
    """
    run, ran, _ = harness

    record = run(_cached_record(extrema_flag=False), flag=True)

    assert _modes_to_conditions(ran) == {"E5"}
    assert "E5" not in record["summary"]["cached_conditions"]
    assert set(record["summary"]["cached_conditions"]) == set(ALL_CONDITIONS) - {"E5"}


def test_pre_flag_cache_is_also_rerun_when_the_flag_goes_on(harness):
    """The 460 archived records have no flag field at all."""
    run, ran, _ = harness

    run(_cached_record(extrema_flag=None), flag=True)

    assert _modes_to_conditions(ran) == {"E5"}


def test_turning_the_flag_off_reruns_only_e5(harness):
    """Symmetric: an experimental E5 must not survive into a control run."""
    run, ran, _ = harness

    run(_cached_record(extrema_flag=True), flag=False)

    assert _modes_to_conditions(ran) == {"E5"}


def test_rerun_e5_receives_flag_on_extrema(harness):
    """The re-run must actually differ, not just happen.

    Facts are recomputed from the cached coordinates, so a stale
    `geosym_artifacts.visual_facts` cannot carry the old empty extrema forward.
    """
    run, _, visual_facts_seen = harness

    run(_cached_record(extrema_flag=False), flag=True)

    assert visual_facts_seen["full_geosym"]["extrema"] == [{"x": 0, "y": 4, "type": "max"}]


def test_reused_conditions_keep_their_own_flag(harness):
    """Provenance survives a mixed record.

    The re-run E5 is flag-on while every reused condition keeps the flag it was
    produced under, so the record stays readable rather than claiming one state
    for results produced under two.
    """
    run, _, _ = harness

    record = run(_cached_record(extrema_flag=False), flag=True)

    conds = record["conditions"]
    assert conds["E5"]["extrema_channel_enabled"] is True
    assert conds["E5"]["cached"] is False
    for ck in set(ALL_CONDITIONS) - {"E5"}:
        assert conds[ck]["extrema_channel_enabled"] is False, ck
        assert conds[ck]["cached"] is True, ck
    assert record["run_settings"]["extrema_channel_enabled"] is True


def test_targeted_rerun_of_another_condition_preserves_stale_e5(harness):
    """A targeted run must not spend money on, or discard, an untargeted E5.

    E5 is stale but out of scope here: it is neither re-run nor dropped, and its
    own flag still says how it was produced.  `force_rerun=True` alongside a
    target is the "re-run this one condition" path — a target on its own still
    honours the cache.
    """
    run, ran, _ = harness

    record = run(
        _cached_record(extrema_flag=False), flag=True,
        force_rerun=True, target_conditions=["E0"],
    )

    assert _modes_to_conditions(ran) == {"E0"}
    assert record["conditions"]["E5"]["extrema_channel_enabled"] is False
    assert record["conditions"]["E5"]["md"] == "**E5** cached"


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
