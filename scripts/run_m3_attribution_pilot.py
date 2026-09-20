#!/usr/bin/env python
"""M3 attribution pilot: does the layering earn its cost, and can form-first
reasoning reverse the evidence penalty on transcendental families?

Safe default: freeze a plan, review it, and exit with ``paid_calls_executed=0``.
``--run-paid`` is required to spend, and the frozen cap is enforced in code
before the first network call.

Three arms, all receiving the SAME M0 coordinate evidence so that a contrast
isolates one factor:

    E5_raw          layered L1->L2->L3          (GeoSym; the reference arm)
    E6_flat_single  same evidence, single pass  (removes the layering)
    E7_form_first   layered, L3 template swap   (identify family before fitting)

Why this pilot exists
---------------------
The layer profile (``scripts/analyze_layer_capability_profile.py``) showed the
evidence gain decaying at L3 (+18.41 at L2 -> +5.21 at L3) and localized the
decay to ``expression``. Two scoring defects found on 2026-09-19 -- narration
captured as the expression, and ``\\log_{2}(u)`` parsed as a symbol times
``(u)`` -- mean the per-family expression numbers behind "evidence hurts on
transcendental families" are not trustworthy. Only 3.8% of the 1680-row main
artifact could be re-scored offline, because it persisted hashes rather than
text.

So this pilot does two jobs at once: it measures the two attribution arms, and
it produces rows that carry their own raw text, making any future scoring
repair a re-analysis instead of a re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reasoning_pipeline import (  # noqa: E402
    ATTRIBUTION_CONDITION_MAP,
    CONDITION_MAP,
    DISPATCHABLE_CONDITION_MAP,
)
from scripts.prepare_public_benchmark_pilot import (  # noqa: E402
    canonical_json_bytes,
    sha256_hex_bytes,
    stable_file_hash,
)

REFERENCE_CONDITION = "E5_raw"
PILOT_CONDITIONS = (REFERENCE_CONDITION, "E6_flat_single", "E7_form_first")

# One sample per family. The pilot's job is to prove the path end to end and
# expose obvious breakage across every family shape, not to measure an effect --
# eight samples cannot separate arms, and this plan does not claim they can.
PILOT_8_ROSTER = (
    "absolute-clean_render_0031",
    "cubic-clean_render_0030",
    "exponential-clean_render_0032",
    "linear-clean_render_0018",
    "logarithmic-clean_render_0032",
    "piecewise_linear-clean_render_0031",
    "quadratic-clean_render_0030",
    "rational-clean_render_0031",
)

# Confirmation roster: 110 samples, stratified by family and weighted toward the
# transcendental families. The elementary families sit near the ceiling on this
# benchmark (linear 100.0, quadratic 96.5), so samples spent there buy almost no
# information; the transcendental families are where the layer profile localised
# the `expression` decay and where evidence measured as harmful.
#
# Weights are frozen here, before execution, for the reason invariant 7 exists:
# choosing strata after seeing results is how a favourable subset gets promoted
# to a headline.
CONFIRMATION_FAMILY_WEIGHTS = {
    "exponential": 20,      # evidence measured harmful -- main battleground
    "rational": 20,         # evidence measured harmful -- main battleground
    "logarithmic": 20,      # floor effect, needs confirming
    "piecewise_linear": 20,  # floor effect, needs confirming
    "cubic": 10,            # evidence measured helpful -- control
    "absolute": 10,         # evidence measured helpful -- control
    "quadratic": 5,         # ceiling control
    "linear": 5,            # ceiling control
}
CONFIRMATION_SAMPLE_COUNT = sum(CONFIRMATION_FAMILY_WEIGHTS.values())

# Reasoning calls per sample: E5_raw and E7 are three-layer chains, E6 is one
# pass. Judge adds one call per sample-condition.
REASONING_CALLS_BY_CONDITION = {
    "E5_raw": 3,
    "E6_flat_single": 1,
    "E7_form_first": 3,
}
JUDGE_CALLS_PER_SAMPLE_CONDITION = 1
TRANSPORT_RETRY_CAP = 1

# GLM bills thinking tokens against `max_tokens` and emits them before any
# visible text, so too small a budget returns an empty body with
# finish_reason=length. The adapter's own default is 4096, which produced 90.9%
# empty truncations on GeoSym-Bench; 16384 is the value the 240x7 run settled
# on. The runner must set this explicitly -- the first pilot did not, silently
# inherited 4096, and lost three rows to empty output.
GLM_MAX_TOKENS = 16384
GLM_DISPLAY_NAME = "GLM-4.6V"

OUTPUT_DIR = ROOT / "outputs" / "m3_attribution_pilot"

# What makes this experiment *this* experiment. The resume guard hashes exactly
# these fields and nothing else.
#
# `created_at_utc` must never be among them. A hash covering the whole plan
# changes on every invocation, so a resumed run fails its own identity check,
# discards the checkpoint, and re-bills every completed job -- the caller pays
# twice for rows already on disk. This is a repeat of a defect found during the
# M1/M2 retirement, where a resume guard rejected its own run for the same
# reason.
EXPERIMENT_IDENTITY_FIELDS = (
    "stage",
    "conditions",
    "reference_condition",
    "condition_modes",
    "selected_ids",
    "sample_count",
    "model_provenance",
    "prompt_hashes",
    "sample_condition_job_cap",
    "reasoning_llm_call_cap",
    "judge_llm_call_cap",
    "llm_http_attempt_cap",
    "ocr_http_attempt_cap",
    "hypotheses",
    "falsification_signals",
)


class LLMHttpAttemptCapExhausted(RuntimeError):
    """Raised at the HTTP layer when the frozen attempt cap is reached."""


class AttemptCounter:
    def __init__(self, cap: int) -> None:
        self.http_attempts = 0
        self.cap = cap
        self.cap_exhausted = False


@contextmanager
def count_llm_http_attempts(counter: AttemptCounter) -> Iterator[None]:
    """Count and cap every HTTP attempt, including SDK-internal retries.

    Wrapping ``_call_with_retry`` rather than the job loop is deliberate: a cap
    checked once per job silently under-counts whenever a transport retry
    fires, so a run can exceed its authorised spend without any single call
    looking wrong. The check runs *before* the request, making the cap a
    ceiling on money committed rather than on money already spent.
    """
    import evaluator.adapters.base as adapter_base

    original = adapter_base.BaseLLMAdapter._call_with_retry

    def counted(self, image_b64: str, prompt: str) -> str:
        last_exc = None
        for attempt in range(1, adapter_base._MAX_API_RETRIES + 1):
            if counter.http_attempts >= counter.cap:
                counter.cap_exhausted = True
                raise LLMHttpAttemptCapExhausted("llm_http_attempt_cap_exhausted")
            counter.http_attempts += 1
            try:
                return self._call_api(image_b64, prompt)
            except Exception as exc:
                last_exc = exc
                if (attempt >= adapter_base._MAX_API_RETRIES
                        or not adapter_base.is_transient_error(exc)):
                    raise
                time.sleep(0.1)
        raise last_exc

    adapter_base.BaseLLMAdapter._call_with_retry = counted
    try:
        yield
    finally:
        adapter_base.BaseLLMAdapter._call_with_retry = original


@contextmanager
def clean_grid_only_m0() -> Iterator[None]:
    """Force M0 extraction onto the free pure-OpenCV grid path.

    Without this the default route runs, and the default route is **paid**:
    ``RecognizerAdapter.recognize_intersections`` only tries the clean-grid
    extractor first when ``GEOSYM_CLEAN_GRID_FALLBACK_FIRST`` is set, which it is
    not, so it falls through to ``ChartExtractor`` and bills one Mathpix OCR call
    per sample. A probe measured exactly that: one ``requests.post`` to Mathpix
    per image, while this plan froze ``ocr_http_attempt_cap = 0``.

    The GeoSym-Bench 240x7 run wraps its whole loop the same way. Matching it
    also keeps the two runs' M0 evidence from coming from different extractors,
    which would confound any comparison between them.
    """
    import evaluator.adapters.recognizer as recognizer_module
    from recognizer.core.clean_grid_fallback import extract_clean_cartesian_grid

    original = recognizer_module.RecognizerAdapter.recognize_intersections

    def recognize_clean_grid_only(self, image, gt=None, skip_enhancement=True,
                                  gemini_enabled=False):
        import cv2

        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                return [], {"success": False, "error": "clean_grid_unavailable"}, None
        extraction = extract_clean_cartesian_grid(image)
        if not extraction.success:
            return [], {"success": False, "error": "clean_grid_unavailable"}, None
        extraction.result["fallback_first"] = True
        extraction.result["clean_grid_only"] = True
        return extraction.coords, extraction.result, None

    recognizer_module.RecognizerAdapter.recognize_intersections = recognize_clean_grid_only
    try:
        yield
    finally:
        recognizer_module.RecognizerAdapter.recognize_intersections = original


class OcrCallAttempted(RuntimeError):
    """Raised if anything on this path tries to bill an OCR call."""


@contextmanager
def forbid_paid_ocr() -> Iterator[list[int]]:
    """Make a paid OCR call impossible, and count attempts so zero is measured.

    ``actual_ocr_http_attempts`` used to be written as the literal ``0`` while
    the run really billed one Mathpix call per sample -- the field asserted a
    fact instead of recording one. Yields a one-element list holding the count
    so the caller reports a measured number.
    """
    import recognizer.mathpix.mathpix_image_ocr as mathpix_module

    attempts: list[int] = [0]
    original = mathpix_module.requests.post

    def blocked(*args: Any, **kwargs: Any):
        attempts[0] += 1
        raise OcrCallAttempted(
            "this plan froze ocr_http_attempt_cap = 0; M0 must use the free "
            "clean-grid path")

    mathpix_module.requests.post = blocked
    try:
        yield attempts
    finally:
        mathpix_module.requests.post = original


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def call_caps(sample_count: int) -> dict[str, int]:
    """Derive every cap from the condition list.

    Never hardcode a total here: a literal silently over-authorises the moment
    an arm is added or retired, which is exactly how a retired PlotQA arm kept
    500 calls of headroom until 2026-09-19.
    """
    reasoning = sample_count * sum(REASONING_CALLS_BY_CONDITION.values())
    judge = sample_count * len(PILOT_CONDITIONS) * JUDGE_CALLS_PER_SAMPLE_CONDITION
    return {
        "sample_condition_job_cap": sample_count * len(PILOT_CONDITIONS),
        "reasoning_llm_call_cap": reasoning,
        "judge_llm_call_cap": judge,
        # One transport retry per call, and no more.
        "llm_http_attempt_cap": (reasoning + judge) * (1 + TRANSPORT_RETRY_CAP),
        "ocr_http_attempt_cap": 0,
    }


def prompt_provenance() -> dict[str, str]:
    """Hash every prompt an arm can reach, so the plan pins the exact wording."""
    names = (
        "ablation_coords_only_L1_prompt.txt",
        "reasoning_L2_prompt.txt",
        "reasoning_L3_prompt.txt",
        "reasoning_L3_form_first_prompt.txt",
        "ablation_flat_single_geosym_prompt.txt",
    )
    return {
        name: stable_file_hash(ROOT / "prompts" / name)
        for name in names
        if (ROOT / "prompts" / name).is_file()
    }


def build_plan(
    roster: tuple[str, ...],
    reasoning_model: str,
    judge_model: str,
) -> dict[str, Any]:
    stop_reason = None
    if len(roster) != len(set(roster)):
        stop_reason = "roster_contains_duplicates"
    elif reasoning_model == judge_model:
        # Invariant 12: a model must not grade its own chain.
        stop_reason = "judge_model_must_differ_from_reasoning_model"

    missing = [
        sample_id for sample_id in roster
        if not (ROOT / "dataset" / "ground_truth"
                / f"{sample_id.split('-')[0]}-clean_render" / f"{sample_id}.json").is_file()
    ]
    if missing and stop_reason is None:
        stop_reason = f"missing_ground_truth:{missing[0]}"

    unknown = [c for c in PILOT_CONDITIONS if c not in DISPATCHABLE_CONDITION_MAP]
    if unknown and stop_reason is None:
        stop_reason = f"undispatchable_condition:{unknown[0]}"

    caps = call_caps(len(roster))
    plan = {
        "schema_version": "m3_attribution_pilot_plan_v1",
        "created_at_utc": utc_now(),
        "stage": "pilot_8_samples",
        "paid_run_approved": stop_reason is None,
        "paid_calls_executed": 0,

        "hypotheses": {
            "H1_layering": (
                "Given identical M0 evidence, the three-layer chain (E5_raw) "
                "scores higher than a single pass (E6_flat_single)."
            ),
            "H2_form_first": (
                "On transcendental families, identifying the function family "
                "before solving for parameters (E7_form_first) scores higher "
                "on `expression` than the standard L3 template (E5_raw)."
            ),
        },
        "falsification_signals": {
            # A fixed point-difference threshold is not usable on this
            # benchmark. A cross-batch probe measured per-cell noise at sd 31.9,
            # which puts the standard error of an arm mean near 3.0 at n=110 and
            # 11.3 at n=8 -- an order of magnitude above the 1.0-point threshold
            # this plan originally carried. The decision is therefore a paired
            # test, implemented by scripts/analyze_m3_attribution.py.
            "H1_layering": (
                "Paired test of E5_raw minus E6_flat_single over samples scored "
                "in both arms. Layering counts as an attributable contribution "
                "only if the paired test is significant at alpha=0.05 AND the "
                "effect is positive. Otherwise report 'no worse than a single "
                "pass, and auditable' -- never as a layering gain. Effect size "
                "and a 95% CI are reported alongside p."
            ),
            "H2_form_first": (
                "Paired test of E7_form_first minus E5_raw restricted to the "
                "transcendental families, frozen as exponential, logarithmic, "
                "rational, piecewise_linear. If not significant and positive, "
                "the fit-instead-of-recognise hypothesis is unsupported and a "
                "different mechanism must be sought."
            ),
            "shared_rules": (
                "Rows whose model output was empty are excluded from the tests "
                "and reported separately per arm: an empty body is a failed "
                "call, not a failed inference, and the truncation rate is not "
                "guaranteed equal across arms (invariant 6)."
            ),
        },
        # Experiments A and B run as one batch over one roster rather than as
        # two sequential runs. All three arms already execute per sample, so
        # E7 gets the full transcendental subset (n=80) instead of a separate
        # 60-sample run: 600 fewer paid calls and more power for H2, with no
        # change to either contrast, since every arm still receives identical
        # M0 evidence from the same batch.
        "merged_experiments": {
            "merged": ["A_layering", "B_form_first"],
            "h2_transcendental_sample_count": 80,
            "calls_saved_versus_sequential": 600,
        },
        "pilot_decides_nothing_about_hypotheses": (
            "Eight samples cannot separate arms. This stage verifies the path "
            "end to end, confirms per-layer raw text is persisted, and exposes "
            "breakage. Both falsification signals above apply to the "
            "confirmation stage, which requires separate approval."
        ),

        "conditions": list(PILOT_CONDITIONS),
        "reference_condition": REFERENCE_CONDITION,
        "condition_modes": {c: DISPATCHABLE_CONDITION_MAP[c] for c in PILOT_CONDITIONS},
        "selected_ids": list(roster),
        "sample_count": len(roster),
        "family_counts": _family_counts(roster),

        "model_provenance": {
            "reasoning_model": reasoning_model,
            "judge_model": judge_model,
            "temperature": 0.0,
            # Frozen in the plan, not left to an adapter default, and part of
            # the identity hash: a run at a different budget is a different
            # experiment and must not share a checkpoint.
            "max_tokens": GLM_MAX_TOKENS if reasoning_model == GLM_DISPLAY_NAME else None,
        },
        "prompt_hashes": prompt_provenance(),
        **caps,
        "retry_policy": (
            f"at most {TRANSPORT_RETRY_CAP} transport retry per call; "
            "no adaptive retries, no sample replacement"
        ),
        "stop_rule": (
            f"stop after {len(roster)} samples x {len(PILOT_CONDITIONS)} conditions "
            f"and at most {caps['llm_http_attempt_cap']} LLM HTTP attempts; "
            "report before any expansion"
        ),
        "cost_estimate": (
            f"{caps['reasoning_llm_call_cap']} reasoning calls + "
            f"{caps['judge_llm_call_cap']} judge calls = "
            f"{caps['reasoning_llm_call_cap'] + caps['judge_llm_call_cap']} paid calls; "
            f"hard ceiling {caps['llm_http_attempt_cap']} HTTP attempts"
        ),

        "fairness_controls": [
            "All three arms receive byte-identical M0 coordinate evidence, "
            "rendered by one shared formatter.",
            "E6 differs from E5_raw only by removing the layering; its prompt "
            "carries the same task guidance as the layered chain.",
            "E7 differs from E5_raw only by the L3 template; L1 and L2 are "
            "byte-identical.",
            "No arm receives a task hint absent from the others except the one "
            "factor under test, which is the manipulation itself (invariant 6).",
        ],
        "persistence_requirement": (
            "Every row must carry per-layer raw text and the parsed answer. "
            "The 240x7 artifact stored only hashes, so two scoring defects "
            "found on 2026-09-19 could not be repaired offline on 96.2% of rows."
        ),
        "interpretation_limits": [
            "Pilot scale. No arm comparison is reportable from these 8 samples.",
            "E6/E7 are attribution arms, not members of the six-condition headline matrix.",
            "Transcendental expression scores before this run were affected by two "
            "parser/metric defects repaired on 2026-09-19; pre-repair per-family "
            "expression figures are not comparable to these.",
            "A single sample per family cannot represent family-level behaviour.",
        ],
        "dry_run": {
            "paid_calls_executed": 0,
            "cap_enforced": True,
            "stopped_before_paid_calls": stop_reason is not None,
            "stop_reason": stop_reason,
        },
    }
    plan["plan_sha256"] = sha256_hex_bytes(
        canonical_json_bytes({k: plan[k] for k in EXPERIMENT_IDENTITY_FIELDS if k in plan})
    )
    return plan


def _family_counts(roster: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample_id in roster:
        family = sample_id.split("-")[0]
        counts[family] = counts.get(family, 0) + 1
    return counts


def review_plan(plan: dict[str, Any]) -> dict[str, Any]:
    findings: list[str] = []
    caps = call_caps(plan["sample_count"])

    if plan["sample_count"] != len(plan["selected_ids"]):
        findings.append("sample_count_does_not_match_roster")
    if plan.get("sample_condition_job_cap") != caps["sample_condition_job_cap"]:
        findings.append(f"job_cap_not_{caps['sample_condition_job_cap']}")
    if plan.get("llm_http_attempt_cap") != caps["llm_http_attempt_cap"]:
        findings.append(f"llm_cap_not_{caps['llm_http_attempt_cap']}")
    if plan.get("ocr_http_attempt_cap") != 0:
        findings.append("ocr_cap_not_zero")
    if plan.get("dry_run", {}).get("paid_calls_executed") != 0:
        findings.append("dry_run_executed_paid_calls")
    if set(plan.get("conditions") or []) != set(PILOT_CONDITIONS):
        findings.append("conditions_not_frozen")
    if plan["model_provenance"]["reasoning_model"] == plan["model_provenance"]["judge_model"]:
        findings.append("judge_model_equals_reasoning_model")
    if plan["model_provenance"]["temperature"] != 0.0:
        findings.append("temperature_not_zero")
    if not plan.get("prompt_hashes", {}).get("reasoning_L3_form_first_prompt.txt"):
        findings.append("form_first_prompt_missing")
    if not plan.get("prompt_hashes", {}).get("ablation_flat_single_geosym_prompt.txt"):
        findings.append("flat_single_prompt_missing")
    # The reference arm must be a real matrix condition, not an attribution arm.
    if REFERENCE_CONDITION not in CONDITION_MAP:
        findings.append("reference_condition_not_in_matrix")
    for arm in ("E6_flat_single", "E7_form_first"):
        if arm in CONDITION_MAP:
            findings.append(f"attribution_arm_leaked_into_matrix:{arm}")
        if arm not in ATTRIBUTION_CONDITION_MAP:
            findings.append(f"attribution_arm_unregistered:{arm}")

    return {
        "schema_version": "m3_attribution_pilot_prepaid_review_v1",
        "created_at_utc": utc_now(),
        "status": "no_blocking_findings" if not findings else "blocking_findings_present",
        "blocking_findings": findings,
        "paid_calls_executed": 0,
    }


def build_confirmation_roster(exclude: tuple[str, ...] = PILOT_8_ROSTER) -> tuple[str, ...]:
    """Select the 110-sample confirmation roster, answer-independently.

    Ordering within a family is by ``sha256(sample_id)``: deterministic across
    machines and independent of any score, coverage value, or answer. Selecting
    on an outcome would pick the samples the method happens to do well on, which
    is the benchmark-shaped selection invariant 5 forbids.

    The pilot's eight samples are excluded by default so the confirmation stage
    does not re-bill rows already on disk.
    """
    from hashlib import sha256

    excluded = set(exclude)
    roster: list[str] = []
    gt_root = ROOT / "dataset" / "ground_truth"
    for family, wanted in sorted(CONFIRMATION_FAMILY_WEIGHTS.items()):
        family_dir = gt_root / f"{family}-clean_render"
        available = sorted(
            (path.stem for path in family_dir.glob("*.json")
             if path.stem not in excluded),
            key=lambda name: sha256(name.encode("utf-8")).hexdigest(),
        )
        if len(available) < wanted:
            raise ValueError(
                f"insufficient_samples:{family} wants {wanted}, has {len(available)}")
        roster.extend(available[:wanted])
    return tuple(roster)


def shard_roster(roster: tuple[str, ...], index: int, count: int) -> tuple[str, ...]:
    """Deterministic stride slice of the roster.

    Sharding is by SAMPLE, never by condition: the three arms of one sample must
    stay in one worker. Splitting a sample across workers would let two
    processes write results for the same image, and the per-image result files
    are not written atomically -- one writer clobbers the other.

    Unlike the 240x7 run, no wave separation is needed here: none of these three
    arms constructs a PaddleOCR reader (only E3/E4 do, and they are not in this
    experiment), and coordinate extraction is pure OpenCV/NumPy on the CPU. The
    OOM hazard that forced serial OCR waves there does not exist here.
    """
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"invalid_shard:{index}/{count}")
    return tuple(sample for position, sample in enumerate(roster)
                 if position % count == index)


def dataset_subpath(sample_id: str) -> str:
    """Map ``absolute-clean_render_0031`` to ``<family-dir>/<sample_id>``.

    The dataset stores each family in its own directory, and the loaders join
    the name straight onto the root, so a bare sample id resolves to a path
    that does not exist. The plan's own existence check builds the family
    directory itself, which is why a roster can pass review and then fail to
    load -- the two used different path rules.
    """
    family_dir = f"{sample_id.split('-')[0]}-clean_render"
    return f"{family_dir}/{sample_id}"


def load_ground_truth(sample_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(gt, gt_data)``.

    These are different objects and are not interchangeable: the recognizer
    takes the whole record, while the reasoning pipeline and the judge take the
    nested ``question.ground_truth``. Passing one where the other is expected
    fails silently rather than loudly -- extraction simply finds nothing.
    """
    from app.helpers import load_image_and_gt

    _, record = load_image_and_gt(dataset_subpath(sample_id))
    if record is None:
        raise FileNotFoundError(f"ground truth not found for {sample_id}")
    gt_data = (record.get("question") or {}).get("ground_truth") or {}
    if not gt_data:
        raise ValueError(f"ground truth question missing for {sample_id}")
    return record, gt_data


def resolve_image(sample_id: str) -> Path:
    from app.reasoning_pipeline import resolve_image_path

    path = resolve_image_path(dataset_subpath(sample_id))
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"image not found for {sample_id}")
    return Path(path)


EMPTY_OUTPUT_CHAIN_CHARS = 64


def is_empty_output(pipeline: dict[str, Any] | None, chain_text: str) -> bool:
    """True when the model returned no visible answer at all.

    GLM emits thinking tokens before any text and bills them against
    `max_tokens`, so an exhausted budget yields a body of length zero with
    finish_reason=length. The chain then contains only its step header.

    This must be separated from a wrong answer. Scoring an empty body as 0
    attributes an infrastructure failure to the arm's reasoning, and the
    truncations do not fall evenly across arms -- so the bias lands on whichever
    arm happens to truncate more, which is exactly the between-arm comparison
    this experiment exists to make (invariant 6).
    """
    if pipeline is None:
        return True
    if len(chain_text or "") > EMPTY_OUTPUT_CHAIN_CHARS:
        return False
    parsed = pipeline.get("final_parsed") or {}
    return not any(str(parsed.get(k) or "").strip()
                   for k in ("expression", "domain", "range", "coordinates"))


def result_row(
    sample_id: str,
    condition: str,
    pipeline: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    judge: dict[str, Any] | None,
    *,
    judge_reused: bool,
    attempts_before: int,
    attempts_after: int,
    error: str = "",
) -> dict[str, Any]:
    from app.reasoning_pipeline import build_chain_text

    chain_text = build_chain_text(pipeline) if pipeline else ""
    if not error and is_empty_output(pipeline, chain_text):
        # Recorded as an error, not as a zero: an empty body is a failed call,
        # not a failed inference.
        error = "empty_model_output_finish_reason_length"
        evaluation = None
        judge = None
    return {
        "sample_id": sample_id,
        "family": sample_id.split("-")[0],
        "condition": condition,
        "status": "success" if error == "" else "error",
        "error": error,
        "total_score": (evaluation or {}).get("total_score"),
        "dimension_scores": (evaluation or {}).get("dimension_scores"),
        # The 1680-row artifact stored hashes only, so a scoring repair forced a
        # re-run instead of a re-analysis. These rows carry their own text.
        "chain_text": chain_text,
        "final_parsed": (pipeline or {}).get("final_parsed"),
        "judge_result": judge,
        "judge_reused": judge_reused,
        "llm_http_attempts": attempts_after - attempts_before,
    }


def write_result(
    plan: dict[str, Any],
    output_path: Path,
    rows: list[dict[str, Any]],
    counter: AttemptCounter,
    status: str,
    ocr_http_attempts: int = 0,
) -> dict[str, Any]:
    payload = {
        "schema_version": "m3_attribution_pilot_result_v1",
        "created_at_utc": utc_now(),
        "stage": plan["stage"],
        "stage_status": status,
        # Identity only. A checkpoint keyed on anything that changes per run --
        # a timestamp, most of all -- rejects its own resume and re-bills the
        # completed jobs.
        "plan_sha256": plan["plan_sha256"],
        "roster_sha256": plan.get("roster_sha256"),
        "conditions": list(plan["conditions"]),
        "reference_condition": REFERENCE_CONDITION,
        "model_provenance": plan["model_provenance"],
        "prompt_hashes": plan["prompt_hashes"],
        "paid_calls_executed": counter.http_attempts,
        "actual_llm_http_attempts": counter.http_attempts,
        # Measured by the patched Mathpix transport, not asserted. This field
        # read a hardcoded 0 while the default M0 route billed one OCR call per
        # sample; the number now comes from a counter that the run cannot evade.
        "actual_ocr_http_attempts": ocr_http_attempts,
        "llm_http_attempt_cap": counter.cap,
        "cap_exhausted": counter.cap_exhausted,
        "hypotheses": plan["hypotheses"],
        "falsification_signals": plan["falsification_signals"],
        "summary": summarize(rows),
        "results": rows,
        "interpretation_limits": [
            "Eight samples verify the path end to end; they cannot separate "
            "arms. Both falsification signals apply to the confirmation stage.",
            "All three arms receive identical M0 coordinate evidence, so a "
            "contrast isolates one factor: E6 removes the layering, E7 swaps "
            "only the L3 template.",
            "E5_raw is re-run here rather than reused from disk, so all three "
            "arms come from one batch and a contrast is not confounded by "
            "between-run drift.",
            "The judge model differs from the reasoning model, and judge "
            "results are reused when the chain text matches byte for byte.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for condition in PILOT_CONDITIONS:
        scored = [r for r in rows
                  if r["condition"] == condition and r["total_score"] is not None]
        subset = [r for r in rows if r["condition"] == condition]
        by_condition[condition] = {
            "n": len(subset),
            "scored": len(scored),
            "errors": sum(1 for r in subset if r["status"] != "success"),
            # Reported per arm because an uneven truncation rate biases the
            # between-arm contrast rather than adding constant noise.
            "empty_output_rows": sum(
                1 for r in subset
                if r.get("error", "").startswith("empty_model_output")),
            "mean_total_score": (
                sum(r["total_score"] for r in scored) / len(scored) if scored else None),
        }
    empty_total = sum(c["empty_output_rows"] for c in by_condition.values())
    return {
        "by_condition": by_condition,
        "path_check": {
            "every_arm_produced_a_scored_row": all(
                by_condition[c]["scored"] > 0 for c in PILOT_CONDITIONS),
            "empty_output_rows_total": empty_total,
            "empty_outputs_are_evenly_spread": (
                len({c["empty_output_rows"] for c in by_condition.values()}) == 1),
            "reading": (
                "This pilot reports whether the path runs, not which arm wins. "
                "Mean scores over eight samples are printed for inspection and "
                "must not be read as an effect."
            ),
        },
    }


def run_paid(plan: dict[str, Any], output_path: Path,
             max_new_jobs: int | None = None) -> dict[str, Any]:
    import evaluator.adapters.base as adapter_base
    from app.helpers import _create_llm_adapter, _get_saved_key
    from app.reasoning_pipeline import (
        DISPATCHABLE_CONDITION_MAP,
        build_chain_text,
        ensure_judge_adapter,
        extract_geosym_coords,
        load_layer_config,
        run_judge,
        run_single_condition,
    )
    from evaluator.metrics.reasoning import evaluate_comprehensive

    if plan.get("paid_run_approved") is not True:
        raise ValueError(plan.get("dry_run", {}).get("stop_reason") or "plan_not_approved")

    reasoning_model = plan["model_provenance"]["reasoning_model"]
    judge_model = plan["model_provenance"]["judge_model"]
    reasoning_key = _get_saved_key(reasoning_model)
    if not reasoning_key:
        raise ValueError(f"saved API key not found for {reasoning_model}")
    judge_key = _get_saved_key(judge_model)
    if not judge_key:
        raise ValueError(f"saved API key not found for {judge_model}")

    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str]] = set()
    if output_path.exists():
        checkpoint = load_json(output_path)
        if checkpoint.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("checkpoint plan hash mismatch; refusing to mix runs")
        rows = [r for r in checkpoint.get("results") or [] if r.get("status") == "success"]
        completed = {(r["sample_id"], r["condition"]) for r in rows}

    counter = AttemptCounter(plan["llm_http_attempt_cap"])
    counter.http_attempts = sum(int(r.get("llm_http_attempts") or 0) for r in rows)

    reasoning_adapter = _create_llm_adapter(reasoning_model, reasoning_key)
    if reasoning_adapter is None:
        raise ValueError(f"could not create adapter for {reasoning_model}")
    judge_adapter = ensure_judge_adapter(judge_model, judge_key, reasoning_model, reasoning_key)
    if judge_adapter is None:
        raise ValueError(f"could not create judge adapter for {judge_model}")

    layer_config = load_layer_config()
    judge_cache: dict[str, dict[str, Any]] = {}
    new_jobs = 0

    # Set the frozen budget before the first call. Leaving it to the adapter
    # default (4096) is what truncated three rows to empty output in the first
    # pilot run, and the truncations fell unevenly across arms.
    import os

    if plan["model_provenance"].get("max_tokens"):
        os.environ["GEOSYM_GLM_MAX_TOKENS"] = str(plan["model_provenance"]["max_tokens"])

    # One transport retry, matching the cap arithmetic. Retries must stay
    # symmetric across arms: retrying only one arm biases the contrast.
    original_retries = adapter_base._MAX_API_RETRIES
    adapter_base._MAX_API_RETRIES = 1 + TRANSPORT_RETRY_CAP
    try:
        with forbid_paid_ocr() as ocr_attempts, clean_grid_only_m0(), \
                count_llm_http_attempts(counter):
            for sample_id in plan["selected_ids"]:
                image_path = resolve_image(sample_id)
                gt, gt_data = load_ground_truth(sample_id)
                # Extraction is a local CV path, not a paid call; it takes the
                # whole record, while reasoning takes `question.ground_truth`.
                coords, _visual_facts, extraction_ok, extraction_reason = (
                    extract_geosym_coords(reasoning_model, reasoning_key, image_path, gt))
                if not extraction_ok:
                    # Fail the sample rather than silently running all three
                    # arms on empty evidence: the contrast would then compare
                    # three arms that each received nothing.
                    for condition in PILOT_CONDITIONS:
                        if (sample_id, condition) in completed:
                            continue
                        rows.append(result_row(
                            sample_id, condition, None, None, None,
                            judge_reused=False,
                            attempts_before=counter.http_attempts,
                            attempts_after=counter.http_attempts,
                            error=f"geosym_extraction_failed: {extraction_reason}"))
                    write_result(plan, output_path, rows, counter,
                                 "running_paid_run", ocr_attempts[0])
                    continue

                for condition in PILOT_CONDITIONS:
                    if (sample_id, condition) in completed:
                        continue
                    if max_new_jobs is not None and new_jobs >= max_new_jobs:
                        return write_result(plan, output_path, rows, counter,
                                            "partial_paid_run", ocr_attempts[0])

                    before = counter.http_attempts
                    try:
                        pipeline = run_single_condition(
                            reasoning_adapter,
                            image_path,
                            gt_data,
                            DISPATCHABLE_CONDITION_MAP[condition],
                            geosym_coords=coords,
                            # M1'/M2' retired 2026-09-19: no arm tiers or
                            # filters evidence, so every arm gets every fact.
                            visual_facts=None,
                        )
                        chain_text = build_chain_text(pipeline)
                        chain_hash = sha256_hex_bytes(canonical_json_bytes(chain_text))
                        judge_reused = chain_hash in judge_cache
                        if judge_reused:
                            judge = judge_cache[chain_hash]
                        else:
                            judge = run_judge(judge_adapter, image_path, gt_data,
                                              pipeline, max_retries=1)
                            judge_cache[chain_hash] = judge
                        evaluation = evaluate_comprehensive(
                            pipeline["final_parsed"], gt_data, layer_config, judge)
                        row = result_row(sample_id, condition, pipeline, evaluation, judge,
                                         judge_reused=judge_reused,
                                         attempts_before=before,
                                         attempts_after=counter.http_attempts)
                    except LLMHttpAttemptCapExhausted:
                        return write_result(plan, output_path, rows, counter,
                                            "stopped_at_paid_cap", ocr_attempts[0])
                    except Exception as exc:
                        row = result_row(sample_id, condition, None, None, None,
                                         judge_reused=False,
                                         attempts_before=before,
                                         attempts_after=counter.http_attempts,
                                         error=f"{type(exc).__name__}: {exc}")

                    rows.append(row)
                    new_jobs += 1
                    # Checkpoint after every job: an interrupted run that lost
                    # its rows would have to re-bill them.
                    write_result(plan, output_path, rows, counter,
                                 "running_paid_run", ocr_attempts[0])
    finally:
        adapter_base._MAX_API_RETRIES = original_retries

    return write_result(plan, output_path, rows, counter, "complete",
                        ocr_attempts[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoning-model", default="GLM-4.6V")
    parser.add_argument("--judge-model", default="Qwen3-VL-Plus")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--run-paid",
        action="store_true",
        help="Execute paid calls. Without this flag nothing is spent.",
    )
    parser.add_argument(
        "--max-new-jobs",
        type=int,
        default=None,
        help="Further limit jobs this invocation may add, under the frozen cap.",
    )
    parser.add_argument(
        "--stage",
        choices=("pilot", "confirmation"),
        default="pilot",
        help="pilot = 8 samples (path check). confirmation = 110 samples, "
             "which requires separate maintainer approval.",
    )
    parser.add_argument(
        "--shard-index", type=int,
        help="0-based shard of this worker. Sharding is by sample, so the three "
             "arms of one sample always stay in one worker.")
    parser.add_argument(
        "--shard-count", type=int,
        help="Total shards. Each worker gets its own output file and its own "
             "slice of the attempt cap, so concurrent workers cannot together "
             "exceed the frozen total.")
    args = parser.parse_args()

    roster = (build_confirmation_roster() if args.stage == "confirmation"
              else PILOT_8_ROSTER)
    plan = build_plan(roster, args.reasoning_model, args.judge_model)

    output_name = "paid_result.json"
    if args.shard_count:
        if args.shard_index is None:
            parser.error("--shard-index is required with --shard-count")
        # The plan hash covers the FULL roster, so every shard shares one
        # experiment identity while writing to its own file. The cap is divided
        # rather than replicated: giving each worker the full cap would authorise
        # shard_count times the approved spend.
        shard = shard_roster(roster, args.shard_index, args.shard_count)
        plan["selected_ids"] = list(shard)
        plan["sample_count"] = len(shard)
        plan["shard"] = {"index": args.shard_index, "count": args.shard_count,
                         "full_roster_sha256": plan["roster_sha256"]}
        plan["family_counts"] = _family_counts(shard)
        for key, value in call_caps(len(shard)).items():
            plan[key] = value
        output_name = f"paid_result.shard{args.shard_index:02d}of{args.shard_count:02d}.json"
    review = review_plan(plan)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Each worker writes its own plan and review; a shared filename would have
    # concurrent workers overwrite each other's provenance.
    suffix = "" if not args.shard_count else f".shard{args.shard_index:02d}of{args.shard_count:02d}"
    plan_path = args.output_dir / f"frozen_plan{suffix}.json"
    review_path = args.output_dir / f"prepaid_review{suffix}.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"conditions        : {', '.join(plan['conditions'])}")
    print(f"samples           : {plan['sample_count']}  {plan['family_counts']}")
    print(f"reasoning calls   : {plan['reasoning_llm_call_cap']}")
    print(f"judge calls       : {plan['judge_llm_call_cap']}")
    print(f"HTTP attempt cap  : {plan['llm_http_attempt_cap']}")
    print(f"cost estimate     : {plan['cost_estimate']}")
    print(f"paid_run_approved : {plan['paid_run_approved']}")
    print(f"stop_reason       : {plan['dry_run']['stop_reason']}")
    print(f"review            : {review['status']}")
    for finding in review["blocking_findings"]:
        print(f"  - {finding}")
    print(f"\nwrote {plan_path}")
    print(f"wrote {review_path}")

    if not args.run_paid:
        print("\nDRY RUN: no paid calls executed. Re-run with --run-paid to spend.")
        return 0 if review["status"] == "no_blocking_findings" else 1

    if review["status"] != "no_blocking_findings" or not plan["paid_run_approved"]:
        print("\nREFUSING to spend: plan is not clean.", file=sys.stderr)
        return 1

    result = run_paid(plan, args.output_dir / output_name,
                      max_new_jobs=args.max_new_jobs)
    print(f"\nstage_status      : {result['stage_status']}")
    print(f"paid calls        : {result['paid_calls_executed']} / cap "
          f"{result['llm_http_attempt_cap']}")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    return 0 if result["stage_status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
