"""M3 layered reasoning for PlotQA: L1 reading, L2 analysis, L3 answer.

Why this module exists
----------------------
GeoSym is M0' evidence plus M3 layered reasoning, but on PlotQA only M0' was
ever wired up: the answering path asked one question in one turn and took
whatever came back. The 2026-09-20 pilot showed what that costs. Asked for a
ratio between two countries' exports, the model replied that the evidence did
not name them and declined; asked for a total, it summed all twenty values it
had been handed. Both failures are of the same kind -- the model was given the
question and the evidence simultaneously and had no stage at which to work out
*which* values the question needed before computing with them.

Repairing M0' measurement quality did not fix this and could not: measurement
accuracy rose from 0.116 to 0.903 while the pilot margin stayed inside the
neutral band. Selecting operands is the reasoning stage's job, so the reasoning
stage has to exist.

The layer discipline, and why it is enforced rather than suggested
-----------------------------------------------------------------
Each layer sees only what the previous one concluded:

- **L1 reads the chart and never sees the question.** This is the load-bearing
  constraint. A reader who knows the question will find the answer in the
  pixels, and the reading stops being independent of it. Withholding the
  question is what makes L1's output usable as evidence rather than as a guess
  shaped by the target.
- **L2 receives L1's readings and the question**, names the entities the
  question refers to, and computes. It is told to quote L1 rather than re-read
  the image, so a disagreement between layers surfaces instead of being
  silently resolved.
- **L3 receives both and states the answer** in the form asked for.

``run_layered_answer`` takes an ``ask`` callable rather than an adapter so the
layer logic can be tested without a network call, and so the caller keeps
ownership of attempt counting -- this module must never be able to spend.

Nothing here scores, ranks or filters evidence. M1/M2 retired on 2026-09-19
(invariants 8-9) and must not re-enter through a reasoning layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT / "prompts"

L1_PROMPT_PATH = PROMPT_DIR / "plotqa_L1_prompt.txt"
L1_BASELINE_PROMPT_PATH = PROMPT_DIR / "plotqa_L1_baseline_prompt.txt"
L2_PROMPT_PATH = PROMPT_DIR / "plotqa_L2_prompt.txt"
L3_PROMPT_PATH = PROMPT_DIR / "plotqa_L3_prompt.txt"

LAYER_ORDER = ("L1", "L2", "L3")


def _load(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a model response.

    Models wrap JSON in prose or fences; a parse failure here is ordinary and
    must not abort the chain, because a later layer can still work from the raw
    text of an earlier one.
    """
    if not text:
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_l1(raw: str) -> Dict[str, Any]:
    """Parse L1 into chart readings.

    Missing keys are filled with empty containers rather than dropped, so L2
    always receives the same shape and cannot mistake "not reported" for a
    structurally different response.
    """
    parsed = _first_json_object(raw) or {}
    readings = parsed.get("readings")
    return {
        "value_axis": parsed.get("value_axis") or {},
        "series": parsed.get("series") or [],
        "categories": parsed.get("categories") or [],
        "readings": readings if isinstance(readings, list) else [],
        "unreadable": parsed.get("unreadable") or [],
    }


def parse_l2(raw: str) -> Dict[str, Any]:
    """Parse L2 into the selected operands and the computed result."""
    parsed = _first_json_object(raw) or {}
    operands = parsed.get("operands")
    return {
        "entities": parsed.get("entities") or [],
        "operands": operands if isinstance(operands, list) else [],
        "operation": parsed.get("operation"),
        "computation": parsed.get("computation"),
        "result": parsed.get("result"),
        # Absent means unstated, which is not the same as insufficient. Only an
        # explicit False records that the model found an operand missing.
        "operands_sufficient": parsed.get("operands_sufficient"),
    }


def parse_l3(raw: str) -> Dict[str, Any]:
    """Parse L3 into the final answer payload.

    Falls back to the raw text when no JSON is present, so a well-formed answer
    in prose is not discarded by a formatting slip.
    """
    parsed = _first_json_object(raw)
    if parsed is None:
        return {"raw_text": raw, "normalized_prediction": None}
    return {
        "raw_text": parsed.get("raw_text", raw),
        "normalized_prediction": parsed.get("normalized_prediction"),
    }


def build_l1_prompt(evidence: Optional[Dict[str, Any]]) -> str:
    """Build L1's prompt. ``evidence=None`` selects the baseline template.

    The two templates share their task wording and output schema; they differ
    only in how they talk about supplied measurements, because the baseline has
    none. Any task guidance beyond that would be an unmatched hint in the
    GeoSym arm (invariant 6).
    """
    if evidence is None:
        return _load(L1_BASELINE_PROMPT_PATH)
    return _load(L1_PROMPT_PATH).replace(
        "{measured_evidence}", json.dumps(evidence, ensure_ascii=False)
    )


def build_l2_prompt(l1_parsed: Dict[str, Any], question: str) -> str:
    return (
        _load(L2_PROMPT_PATH)
        .replace("{L1_results}", json.dumps(l1_parsed, ensure_ascii=False))
        .replace("{question}", question)
    )


def build_l3_prompt(l1_parsed: Dict[str, Any], l2_parsed: Dict[str, Any], question: str) -> str:
    return (
        _load(L3_PROMPT_PATH)
        .replace("{L1_results}", json.dumps(l1_parsed, ensure_ascii=False))
        .replace("{L2_results}", json.dumps(l2_parsed, ensure_ascii=False))
        .replace("{question}", question)
    )


def run_layered_answer(
    ask: Callable[[str, str], str],
    question: str,
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run L1 -> L2 -> L3 and return every step plus the final answer.

    ``ask(layer, prompt) -> raw_text`` performs one model call. Raising from it
    aborts the chain at that layer; the steps completed so far are returned with
    the failure recorded, because a partial chain is evidence about where the
    reasoning broke and discarding it would hide that.

    The question is deliberately absent from L1's arguments. It is not withheld
    by convention but by construction: :func:`build_l1_prompt` has no parameter
    to pass it through.
    """
    steps: Dict[str, Dict[str, Any]] = {}

    l1_prompt = build_l1_prompt(evidence)
    try:
        l1_raw = ask("L1", l1_prompt)
    except Exception as exc:
        return _partial(steps, "L1", l1_prompt, exc)
    l1_parsed = parse_l1(l1_raw)
    steps["L1"] = {"prompt": l1_prompt, "raw": l1_raw, "parsed": l1_parsed}

    l2_prompt = build_l2_prompt(l1_parsed, question)
    try:
        l2_raw = ask("L2", l2_prompt)
    except Exception as exc:
        return _partial(steps, "L2", l2_prompt, exc)
    l2_parsed = parse_l2(l2_raw)
    steps["L2"] = {"prompt": l2_prompt, "raw": l2_raw, "parsed": l2_parsed}

    l3_prompt = build_l3_prompt(l1_parsed, l2_parsed, question)
    try:
        l3_raw = ask("L3", l3_prompt)
    except Exception as exc:
        return _partial(steps, "L3", l3_prompt, exc)
    l3_parsed = parse_l3(l3_raw)
    steps["L3"] = {"prompt": l3_prompt, "raw": l3_raw, "parsed": l3_parsed}

    return {
        "steps": steps,
        "final_parsed": l3_parsed,
        "prediction_text": l3_parsed.get("normalized_prediction"),
        "layers_completed": list(steps),
        "failed_layer": None,
    }


def _partial(
    steps: Dict[str, Dict[str, Any]],
    layer: str,
    prompt: str,
    exc: Exception,
) -> Dict[str, Any]:
    steps[layer] = {"prompt": prompt, "raw": "", "parsed": None,
                    "error": f"{type(exc).__name__}: {exc}"}
    return {
        "steps": steps,
        "final_parsed": {},
        "prediction_text": None,
        "layers_completed": [name for name in LAYER_ORDER if name in steps and name != layer],
        "failed_layer": layer,
    }


def render_chain_text(result: Dict[str, Any]) -> str:
    """Render every layer's raw output in execution order.

    The GeoSym-Bench judge was once fed layered arms' *conclusions* while
    single-turn arms sent their full output, which understated the layered arms
    by construction. Rendering raw text keeps this path from repeating that.
    """
    sections: List[str] = []
    for layer in LAYER_ORDER:
        step = (result.get("steps") or {}).get(layer)
        raw = (step or {}).get("raw")
        if isinstance(raw, str) and raw.strip():
            sections.append(f"=== {layer} ===\n{raw.strip()}")
    return "\n\n".join(sections)


def prompt_hashes() -> Dict[str, str]:
    """Hash each layer template, for freezing into an experiment plan."""
    from hashlib import sha256

    return {
        name: sha256(_load(path).encode("utf-8")).hexdigest()
        for name, path in (
            ("plotqa_L1_measured", L1_PROMPT_PATH),
            ("plotqa_L1_baseline", L1_BASELINE_PROMPT_PATH),
            ("plotqa_L2", L2_PROMPT_PATH),
            ("plotqa_L3", L3_PROMPT_PATH),
        )
    }
