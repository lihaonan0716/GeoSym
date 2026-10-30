#!/usr/bin/env python
"""PlotQA numeric-answer attribution on the CURRENT M0' extractor.

The experiment plan is frozen before any paid call.

Why this run exists
-------------------
Six candidate mechanisms for the PlotQA numeric penalty were tested offline and
**all six were falsified** -- duplicate facts (p=0.68), payload truncation
(p=0.19), tolerance near-misses (74/162 beyond 10x), series-label validity
(0.9563), measured-value accuracy (p=0.67), and entity addressability (a
difficulty confound: the baseline shows the same direction). What survived is a
stratification built from official answer-independent fields, not a mechanism:
every ``text`` stratum is net >= 0 while every ``numeric`` stratum is net <= 0,
and the same template flips sign with answer type (min_max/text +2 against
min_max/numeric -27).

The rows that established that live on disk, but they were produced **before**
the M0' repairs took value accuracy 0.116 -> 0.831 and coverage 0.522 -> 0.765.
Under invariant 12 they characterise an extractor that no longer exists, so the
current extractor's answer accuracy cannot be computed offline at all. That is
the single thing this run measures.

**This is an attribution run, not a winning run.** No hypothesis here states that
GeoSym beats the baseline, and no output of it may be reported as such.

Cost control
------------
The default performs no network call; ``--execute`` is required. The cap is
checked *before* each request by wrapping ``_call_with_retry``, so SDK-internal
retries consume it rather than escaping it. OCR attempts are **counted, not
assumed**: the Mathpix transport is patched to raise, because this path must read
frozen PaddleOCR boxes from disk and any real OCR call is a protocol violation.

:hypothesis: H_N1 the numeric penalty reproduces on the current extractor;
    H_N2 evidence alone is not sufficient for numeric answers; H_N3 label
    completeness is associated with correctness.
:falsification: Each is falsified by a paired test that is significant in the
    opposite direction, at alpha=0.05. A non-significant result supports nothing.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluator.adapters.base as adapter_base  # noqa: E402
from app.helpers import _create_llm_adapter, _get_saved_key  # noqa: E402
from scripts.prepare_plotqa_public_benchmark_pilot import e0_prompt  # noqa: E402
from scripts.prepare_public_benchmark_pilot import (  # noqa: E402
    canonical_json_bytes,
    sha256_hex_bytes,
)
from scripts.run_plotqa_locate import answer_prompt, measured_prompt_facts  # noqa: E402
from scripts.run_public_benchmark_pilot import (  # noqa: E402
    extract_prediction,
    is_year_answer,
    score_prediction,
)

# ── Frozen experiment identity ──────────────────────────────────────────

BASELINE_CONDITION = "E0_plotqa_direct"
MEASURED_CONDITION = "G_measured_raw"
EVIDENCE_ONLY_CONDITION = "Measured_flat_answer_baseline"

#: Order matters only for reporting. All three identifiers already name arms in
#: 1500 paid rows on disk and are deliberately reused rather than renamed: a
#: condition label is part of the data, not a display string.
CONDITIONS = (BASELINE_CONDITION, MEASURED_CONDITION, EVIDENCE_ONLY_CONDITION)

#: The two strata holding -44 of the -52 net loss. Official `qa_template` values
#: crossed with `answer_type`, both answer-independent fields, so this is a
#: pre-registered stratum and not the post-hoc subset invariant 7 forbids.
TARGET_STRATA = (("arithmetic", "numeric"), ("min_max", "numeric"))

REASONING_MODEL = "GPT-5.5"
TEMPERATURE = 0.0

PILOT_SAMPLE_COUNT = 8
CONFIRMATION_SAMPLE_COUNT = 80

#: One transport retry and no more. Retries must be symmetric across arms:
#: retrying only the measured arm would bias the contrast (invariant 6).
TRANSPORT_RETRY_CAP = 1

#: Below this, a response body is treated as an empty call rather than a wrong
#: answer. An empty body is a failed call, not a failed inference, and folding it
#: in as incorrect would bias whichever arm truncates more often.
EMPTY_OUTPUT_CHARS = 8

PACKAGE = ROOT / "outputs" / "paper_level_evidence_package"
DEFAULT_PLAN_SOURCE = PACKAGE / "plotqa_public_500" / "plotqa_500_zero_cost_plan.json"
#: Post-repair records. The pre-repair file sits beside it under
#: `plotqa_public_500/`; pointing here is the whole reason the run is worth money.
DEFAULT_RECORDS = PACKAGE / "plotqa_public_500_bar_repair" / "m0_measurement_records.json"
DEFAULT_OCR = PACKAGE / "plotqa_public_500" / "openocr_survey_paddleocr" / "raw_ocr.json"
DEFAULT_OUTPUT_DIR = PACKAGE / "plotqa_numeric_attribution"

#: Hashed into the plan so a later prompt edit invalidates the run instead of
#: silently changing what was tested.
PINNED_PROMPT_SOURCES = (
    "scripts/run_plotqa_locate.py",
    "scripts/prepare_plotqa_public_benchmark_pilot.py",
)

#: Fields that define *what experiment this is*. `created_at_utc` is excluded on
#: purpose: a resume guard that hashes its own timestamp rejects its own
#: checkpoint and re-bills every completed row.
EXPERIMENT_IDENTITY_FIELDS = (
    "stage",
    "conditions",
    "target_strata",
    "roster_rule",
    "selected_ids",
    "sample_count",
    "model_provenance",
    "records_sha256",
    "ocr_sha256",
    "prompt_source_hashes",
    "sample_condition_job_cap",
    "reasoning_llm_call_cap",
    "judge_llm_call_cap",
    "llm_http_attempt_cap",
    "ocr_http_attempt_cap",
    "hypotheses",
    "falsification_signals",
)


class LLMHttpAttemptCapExhausted(RuntimeError):
    """Raised before a request that would exceed the frozen cap."""


class OcrCallAttempted(RuntimeError):
    """Raised if anything on this path tries to bill an OCR call.

    This path must read frozen PaddleOCR boxes from disk. An earlier attribution runner
    recorded ``actual_ocr_http_attempts: 0`` as a hardcoded constant while really
    billing one Mathpix call per sample; the field was a claim, not a
    measurement. Here the number is measured and a non-zero count aborts.
    """


class AttemptCounter:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.http_attempts = 0
        self.cap_exhausted = False

    def consume(self) -> None:
        if self.http_attempts >= self.cap:
            self.cap_exhausted = True
            raise LLMHttpAttemptCapExhausted(
                f"frozen LLM HTTP attempt cap {self.cap} reached")
        self.http_attempts += 1


@contextlib.contextmanager
def count_llm_attempts(counter: AttemptCounter) -> Iterator[None]:
    """Charge the cap before the request, not after.

    Wrapping ``_call_with_retry`` rather than ``_call_api`` would let the
    adapter's own retry loop spend several HTTP attempts against one charge.
    Wrapping it and calling ``_call_api`` directly makes every transport attempt
    visible to the counter.
    """
    original = adapter_base.BaseLLMAdapter._call_with_retry

    def counted(self, image_b64: str, prompt: str) -> str:
        counter.consume()
        return self._call_api(image_b64, prompt)

    adapter_base.BaseLLMAdapter._call_with_retry = counted
    try:
        yield
    finally:
        adapter_base.BaseLLMAdapter._call_with_retry = original


@contextlib.contextmanager
def forbid_paid_ocr() -> Iterator[list[int]]:
    """Make a paid OCR call impossible, and prove it by counting.

    Yields a one-element list holding the attempt count so the caller can record
    a measured zero rather than asserting one.
    """
    import recognizer.mathpix.mathpix_image_ocr as mathpix_module

    attempts: list[int] = [0]
    original = mathpix_module.requests.post

    def blocked(*args: Any, **kwargs: Any):
        attempts[0] += 1
        raise OcrCallAttempted(
            "this path must use frozen PaddleOCR boxes; a Mathpix call is a "
            "protocol violation")

    mathpix_module.requests.post = blocked
    try:
        yield attempts
    finally:
        mathpix_module.requests.post = original


# ── Plan construction ───────────────────────────────────────────────────


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_of(payload: Any) -> list[dict[str, Any]]:
    """Read the two shapes the frozen artifacts use, without rewriting either."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("records", "results", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def roster_order_key(sample_id: str) -> str:
    """Deterministic, answer-independent ordering key.

    ``sha256(sample_id)`` rather than a seeded shuffle: a seed can be re-rolled
    until the roster looks favourable, a hash cannot. Nothing about the reference
    answer enters here.
    """
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()


def build_roster(plan_source: Path, sample_count: int) -> list[dict[str, Any]]:
    plan = load_json(plan_source)
    pool = [
        sample
        for sample in plan["selected_samples"]
        if (sample["metadata"].get("qa_template"), sample.get("answer_type"))
        in TARGET_STRATA
    ]
    ordered = sorted(pool, key=lambda s: roster_order_key(s["sample_id"]))
    return ordered[:sample_count]


def call_caps(sample_count: int) -> dict[str, int]:
    """Derive caps from the condition list.

    A literal over-authorises the moment an arm is added or removed.
    """
    reasoning = sample_count * len(CONDITIONS)
    return {
        "sample_condition_job_cap": sample_count * len(CONDITIONS),
        "reasoning_llm_call_cap": reasoning,
        # No judge: scoring is deterministic, so invariant 13's
        # judge-must-differ rule is satisfied by there being no judge at all.
        "judge_llm_call_cap": 0,
        "llm_http_attempt_cap": reasoning * (1 + TRANSPORT_RETRY_CAP),
        "ocr_http_attempt_cap": 0,
    }


def prompt_source_hashes() -> dict[str, str]:
    return {
        name: sha256_of_file(ROOT / name) for name in PINNED_PROMPT_SOURCES
    }


def build_plan(stage: str, roster: list[dict[str, Any]], *,
               records_path: Path, ocr_path: Path) -> dict[str, Any]:
    caps = call_caps(len(roster))
    selected_ids = [s["sample_id"] for s in roster]
    year_rows = [
        s["sample_id"] for s in roster
        if is_year_answer(s.get("question"), s.get("answer"))
    ]

    plan: dict[str, Any] = {
        "schema_version": "plotqa_numeric_attribution_plan_v1",
        "created_at_utc": utc_now(),
        "stage": stage,
        "frozen_plan_doc": "PLOTQA_NUMERIC_ATTRIBUTION_FROZEN_PLAN_2026-09-20",
        "conditions": list(CONDITIONS),
        "target_strata": [list(s) for s in TARGET_STRATA],
        "roster_rule": (
            "pool = plotqa_500 frozen plan restricted to "
            "qa_template in {arithmetic, min_max} AND answer_type == numeric "
            "(pool size 205); ordered by ascending sha256(sample_id); first N. "
            "Reads qa_template, answer_type and sample_id only -- never the "
            "reference answer or any prior outcome."
        ),
        "selected_ids": selected_ids,
        "sample_count": len(roster),
        "stratum_counts": _stratum_counts(roster),
        "model_provenance": {
            "reasoning_model": REASONING_MODEL,
            "model_id_source": "config/default_config.yaml models.OpenAI",
            "temperature": TEMPERATURE,
            "judge_model": None,
            "scoring": "deterministic scorer; no LLM judge",
        },
        "records_path": str(records_path),
        "records_sha256": sha256_of_file(records_path),
        "ocr_path": str(ocr_path),
        "ocr_sha256": sha256_of_file(ocr_path),
        "prompt_source_hashes": prompt_source_hashes(),
        **caps,
        "year_rule": {
            "applies": bool(year_rows),
            "year_answer_sample_ids": year_rows,
            "note": (
                "score_prediction takes `question` as an OPTIONAL argument, so "
                "omitting it would silently score these rows with a numeric "
                "tolerance instead of exact equality. The runner always passes "
                "it. A probe found 37 of the 205-sample pool are year answers "
                "despite answer_type == numeric."
            ),
        },
        "hypotheses": {
            "H_N1_numeric_penalty_reproduces": (
                f"On the current extractor, {MEASURED_CONDITION} scores below "
                f"{BASELINE_CONDITION} on this numeric roster."
            ),
            "H_N2_evidence_not_independently_sufficient": (
                f"{EVIDENCE_ONLY_CONDITION} scores below {MEASURED_CONDITION}."
            ),
            "H_N3_label_completeness_associates_with_correctness": (
                "Within the measured arm, a higher share of facts carrying BOTH "
                "series_label and category_label associates with correctness."
            ),
        },
        "falsification_signals": {
            "shared_rules": (
                "Paired over samples answered in both arms; paired t and "
                "Wilcoxon at alpha=0.05; Cohen's d and 95% CI reported beside "
                "p. A claim needs significance AND the stated direction; a "
                "significant result in the opposite direction is reported as "
                "such and never renamed. Empty or unparseable bodies are "
                "reported separately and excluded, because truncation rate is "
                "not guaranteed equal across arms (invariant 6)."
            ),
            "H_N1": (
                "Falsified if the paired test is significant with "
                f"{MEASURED_CONDITION} ahead of {BASELINE_CONDITION}."
            ),
            "H_N2": (
                "Falsified if "
                f"{EVIDENCE_ONLY_CONDITION} matches or exceeds {MEASURED_CONDITION}."
            ),
            "H_N3": "Falsified if the association is absent or inverted at alpha=0.05.",
        },
        "attribution_not_victory": (
            "No hypothesis states that GeoSym beats the baseline on PlotQA "
            "numeric answers, and no output of this run may be reported as such."
        ),
        "interpretation_limits": [
            "Two templates x numeric answers only; not representative of PlotQA "
            "overall. Every number carries its stratum.",
            "Six mechanisms were falsified offline. The surviving object is a "
            "stratification, not an explanation: this run measures position, "
            "not cause.",
            "N=80 puts the standard error of an arm mean near 5.5 points at "
            "p~0.5; smaller differences are not resolvable.",
            "Baselines are re-run rather than reused, because the cached rows "
            "predate the M0' repairs and would confound extractor version with "
            "the manipulation.",
        ],
        "paid_calls_executed": 0,
        "dry_run": {"paid_calls_executed": 0},
    }

    plan["plan_sha256"] = sha256_hex_bytes(
        canonical_json_bytes(
            {k: plan[k] for k in EXPERIMENT_IDENTITY_FIELDS if k in plan}))
    return plan


def _stratum_counts(roster: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in roster:
        key = f"{sample['metadata'].get('qa_template')}/{sample.get('answer_type')}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def review_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Re-derive every cost-gating fact from the roster rather than trusting it."""
    findings: list[str] = []
    caps = call_caps(plan["sample_count"])

    if plan["sample_count"] != len(plan["selected_ids"]):
        findings.append("sample_count_does_not_match_roster")
    if len(set(plan["selected_ids"])) != len(plan["selected_ids"]):
        findings.append("roster_contains_duplicates")
    for key, expected in caps.items():
        if plan.get(key) != expected:
            findings.append(f"{key}_not_{expected}")
    if plan.get("ocr_http_attempt_cap") != 0:
        findings.append("ocr_cap_not_zero")
    if set(plan.get("conditions") or []) != set(CONDITIONS):
        findings.append("conditions_not_frozen")
    if plan["model_provenance"]["temperature"] != 0.0:
        findings.append("temperature_not_zero")
    if plan.get("dry_run", {}).get("paid_calls_executed") != 0:
        findings.append("dry_run_executed_paid_calls")
    if not plan.get("plan_sha256"):
        findings.append("plan_identity_not_hashed")
    if "created_at_utc" in EXPERIMENT_IDENTITY_FIELDS:
        findings.append("identity_hash_includes_timestamp")

    return {
        "status": "no_blocking_findings" if not findings else "blocking_findings_present",
        "blocking_findings": findings,
        "derived_caps": caps,
    }


# ── Paid execution ──────────────────────────────────────────────────────


def build_prompt(condition: str, question: str, facts: dict[str, Any]) -> str:
    if condition == BASELINE_CONDITION:
        return e0_prompt(question)
    return answer_prompt(question, facts, condition)


def label_completeness(facts: dict[str, Any]) -> dict[str, Any]:
    """Share of injected facts carrying BOTH labels -- the H_N3 covariate."""
    items = list(facts.get("measured_facts") or [])
    both = sum(
        1 for f in items if f.get("series_label") and f.get("category_label"))
    return {
        "facts": len(items),
        "facts_with_both_labels": both,
        "both_label_share": (both / len(items)) if items else None,
    }


def is_empty_body(raw_text: str | None) -> bool:
    return len((raw_text or "").strip()) < EMPTY_OUTPUT_CHARS


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def run_paid(plan: dict[str, Any], output_path: Path, *,
             records_path: Path, ocr_path: Path) -> dict[str, Any]:
    plan_source = DEFAULT_PLAN_SOURCE
    samples = {s["sample_id"]: s
               for s in load_json(plan_source)["selected_samples"]}
    records = {r["sample_id"]: r for r in rows_of(load_json(records_path))}
    ocr_rows = {r["sample_id"]: r for r in rows_of(load_json(ocr_path))}

    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str]] = set()
    if output_path.is_file():
        previous = load_json(output_path)
        if previous.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError(
                "existing result file belongs to a different experiment "
                f"({previous.get('plan_sha256', '')[:12]} vs "
                f"{plan['plan_sha256'][:12]}); refusing to merge")
        rows = list(previous.get("results") or [])
        completed = {(r["sample_id"], r["condition"]) for r in rows
                     if r.get("status") == "success"}

    counter = AttemptCounter(plan["llm_http_attempt_cap"])
    counter.http_attempts = sum(int(r.get("llm_http_attempts") or 0) for r in rows)

    api_key = _get_saved_key(REASONING_MODEL)
    if not api_key:
        raise RuntimeError(f"no API key configured for {REASONING_MODEL}")

    adapter_base._MAX_API_RETRIES = 1 + TRANSPORT_RETRY_CAP

    ocr_attempts = [0]
    try:
        with forbid_paid_ocr() as ocr_attempts, count_llm_attempts(counter):
            adapter = _create_llm_adapter(REASONING_MODEL, api_key)
            if adapter is None:
                raise RuntimeError(f"no adapter for {REASONING_MODEL}")

            for sample_id in plan["selected_ids"]:
                sample = samples[sample_id]
                record = records.get(sample_id)
                facts = (
                    measured_prompt_facts(record, None, ocr_rows.get(sample_id))
                    if record is not None else {}
                )
                completeness = label_completeness(facts)
                image_b64 = encode_image(Path(sample["image_path"]))

                for condition in CONDITIONS:
                    if (sample_id, condition) in completed:
                        continue
                    before = counter.http_attempts
                    prompt = build_prompt(condition, sample["question"], facts)
                    try:
                        raw_text = adapter._call_with_retry(image_b64, prompt)
                        status, error = "success", ""
                    except LLMHttpAttemptCapExhausted:
                        write_result(output_path, plan, rows, counter,
                                     ocr_attempts[0], "stopped_at_cap")
                        return summarize_and_write(
                            output_path, plan, rows, counter, ocr_attempts[0],
                            "stopped_at_cap")
                    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                        raw_text, status = None, "error"
                        error = f"{type(exc).__name__}: {exc}"

                    row = score_row(sample, condition, raw_text, status, error,
                                    completeness,
                                    counter.http_attempts - before)
                    rows.append(row)
                    if row["status"] == "success":
                        completed.add((sample_id, condition))
                    write_result(output_path, plan, rows, counter,
                                 ocr_attempts[0], "running_paid_run")
                    print(
                        f"row={len(rows)}/{plan['sample_condition_job_cap']} "
                        f"{sample_id} {condition} status={row['status']} "
                        f"correct={row.get('correct')} "
                        f"attempts={counter.http_attempts}/{counter.cap}",
                        flush=True)
    finally:
        pass

    return summarize_and_write(output_path, plan, rows, counter,
                               ocr_attempts[0], "complete")


def score_row(sample: dict[str, Any], condition: str, raw_text: str | None,
              status: str, error: str, completeness: dict[str, Any],
              attempts: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "condition": condition,
        "qa_template": sample["metadata"].get("qa_template"),
        "qa_type": sample["metadata"].get("qa_type"),
        "answer_type": sample.get("answer_type"),
        "reference": sample.get("answer"),
        "status": status,
        "error": error,
        "raw_text": raw_text,
        "llm_http_attempts": attempts,
        **completeness,
    }
    if status != "success":
        row.update({"correct": None, "normalized_prediction": None})
        return row
    if is_empty_body(raw_text):
        # Recorded as an error, not as a wrong answer: an empty body is a failed
        # call and scoring it 0 would bias whichever arm truncates more.
        row.update({"status": "error", "error": "empty_model_output",
                    "correct": None, "normalized_prediction": None})
        return row

    # `extract_prediction` documents a None return for an unparseable body, but
    # it reaches `normalize_answer`, which *raises* when a numeric question is
    # answered with a bare category name ("Japan" to a "minimum number of
    # refugees" question). That killed the first confirmation attempt at row 124.
    # An unanswerable body is a wrong answer, not a crash and not a failed call:
    # the model did respond, it just did not produce a number. Scoring it
    # incorrect also matches the 70 unparseable rows in the historical paid
    # artifact, so this run stays comparable with them.
    try:
        prediction = extract_prediction(raw_text, sample.get("answer_type"))
        parseable = True
    except ValueError:
        prediction, parseable = None, False

    # `question` is passed on every call: it is optional in score_prediction, and
    # omitting it silently scores year answers with a numeric tolerance.
    scored = score_prediction(prediction, sample.get("answer"),
                              sample.get("answer_type"),
                              sample.get("question"))
    row.update({"normalized_prediction": prediction, "parseable": parseable,
                **scored})
    return row


def write_result(output_path: Path, plan: dict[str, Any],
                 rows: list[dict[str, Any]], counter: AttemptCounter,
                 ocr_attempts: int, stage_status: str) -> dict[str, Any]:
    payload = {
        "schema_version": "plotqa_numeric_attribution_result_v1",
        "created_at_utc": utc_now(),
        "stage": plan["stage"],
        "stage_status": stage_status,
        "plan_sha256": plan["plan_sha256"],
        "conditions": list(CONDITIONS),
        "model_provenance": plan["model_provenance"],
        "records_sha256": plan["records_sha256"],
        "ocr_sha256": plan["ocr_sha256"],
        "prompt_source_hashes": plan["prompt_source_hashes"],
        "paid_calls_executed": counter.http_attempts,
        "actual_llm_http_attempts": counter.http_attempts,
        # Measured by the patched transport, not asserted as a constant.
        "actual_ocr_http_attempts": ocr_attempts,
        "llm_http_attempt_cap": counter.cap,
        "cap_exhausted": counter.cap_exhausted,
        "hypotheses": plan["hypotheses"],
        "falsification_signals": plan["falsification_signals"],
        "attribution_not_victory": plan["attribution_not_victory"],
        "interpretation_limits": plan["interpretation_limits"],
        "results": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def summarize_and_write(output_path: Path, plan: dict[str, Any],
                        rows: list[dict[str, Any]], counter: AttemptCounter,
                        ocr_attempts: int, stage_status: str) -> dict[str, Any]:
    payload = write_result(output_path, plan, rows, counter, ocr_attempts,
                           stage_status)
    payload["summary"] = summarize(rows)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        scored = [r for r in subset if r.get("status") == "success"]
        by_condition[condition] = {
            "n": len(subset),
            "scored": len(scored),
            "errors": len(subset) - len(scored),
            "empty_output_rows": sum(
                1 for r in subset if str(r.get("error", "")).startswith("empty_model_output")),
            "correct": sum(1 for r in scored if r.get("correct")),
            "accuracy": (sum(1 for r in scored if r.get("correct")) / len(scored))
            if scored else None,
        }
    return {
        "by_condition": by_condition,
        "path_check": {
            "every_arm_produced_a_scored_row": all(
                v["scored"] > 0 for v in by_condition.values()),
            "reading": (
                "A pilot reports whether the path runs. Accuracies over eight "
                "samples must not be read as an effect: the standard error of "
                "an arm mean is about 18 points at n=8."
            ),
        },
    }


# ── CLI ─────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "confirmation"),
                        default="pilot")
    parser.add_argument("--plan-source", type=Path, default=DEFAULT_PLAN_SOURCE)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--ocr", type=Path, default=DEFAULT_OCR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true",
                        help="spend paid calls; default performs none")
    args = parser.parse_args(argv)

    sample_count = (PILOT_SAMPLE_COUNT if args.stage == "pilot"
                    else CONFIRMATION_SAMPLE_COUNT)
    roster = build_roster(args.plan_source, sample_count)
    plan = build_plan(args.stage, roster, records_path=args.records,
                      ocr_path=args.ocr)
    review = review_plan(plan)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"frozen_plan.{args.stage}.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / f"prepaid_review.{args.stage}.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"stage={args.stage} samples={plan['sample_count']} "
          f"conditions={list(CONDITIONS)}")
    print(f"strata={plan['stratum_counts']}")
    print(f"job_cap={plan['sample_condition_job_cap']} "
          f"llm_http_cap={plan['llm_http_attempt_cap']} "
          f"ocr_cap={plan['ocr_http_attempt_cap']}")
    print(f"year_answer_rows={len(plan['year_rule']['year_answer_sample_ids'])}")
    print(f"plan_sha256={plan['plan_sha256'][:20]}")
    print(f"review={review['status']} {review['blocking_findings']}")

    if review["status"] != "no_blocking_findings":
        print("BLOCKED: refusing to spend with blocking findings")
        return 2
    if not args.execute:
        print("dry run: no paid call performed. Pass --execute to spend.")
        return 0

    result = run_paid(plan, args.output_dir / f"paid_result.{args.stage}.json",
                      records_path=args.records, ocr_path=args.ocr)
    print(json.dumps(result.get("summary"), indent=2, ensure_ascii=False))
    print(f"ocr_http_attempts_measured={result['actual_ocr_http_attempts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
