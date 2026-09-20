"""
Parse LLM reasoning outputs: extract final answers and convert to comparable objects.
Supports comprehensive (6 sub-question) format and Judge LLM responses.
"""

import json
import re
from typing import Any, Dict, List, Optional


def extract_answer(raw_response: str) -> str:
    """
    Extract the final answer from LLM output.
    Looks for patterns like '**Answer:**', 'Answer:', 'ANSWER:', etc.
    Returns the text after the last occurrence of such a pattern.
    """
    if not raw_response:
        return ""

    patterns = [
        r'\*\*Answer[:\s]*\*\*[:\s]*(.*)',
        r'Answer[:\s]+(.*)',
        r'ANSWER[:\s]+(.*)',
        r'答案[：:\s]+(.*)',
        r'最终答案[：:\s]+(.*)',
    ]

    last_match = ""
    last_pos = -1

    for pattern in patterns:
        for m in re.finditer(pattern, raw_response, re.IGNORECASE):
            if m.start() > last_pos:
                last_pos = m.start()
                last_match = m.group(1).strip()

    if last_match:
        last_match = re.sub(r'^\$+|\$+$', '', last_match).strip()
        last_match = re.sub(r'^\*+|\*+$', '', last_match).strip()
        last_match = re.sub(r'^["`\']+|["`\']+$', '', last_match).strip()
        last_match = last_match.rstrip('.。')
        return last_match

    lines = raw_response.strip().split('\n')
    return lines[-1].strip() if lines else ""


def parse_number_set(answer_str: str) -> List[float]:
    """
    Parse a set of numbers from answer string.
    Handles: '-2, 0, 3', '{-2, 0, 3}', 'x = -2, 0, 3', 'No real zeros'.
    """
    if not answer_str:
        return []

    lower = answer_str.lower()
    if any(kw in lower for kw in ["no real", "no zero", "none", "does not exist",
                                   "不存在", "无实数", "无零点", "无解"]):
        return []

    cleaned = answer_str.strip()
    cleaned = re.sub(r'[{}()\[\]]', '', cleaned)
    cleaned = re.sub(r'x\s*=\s*', '', cleaned)
    cleaned = re.sub(r'\\{[^}]*\\}', '', cleaned)

    numbers = []
    for token in re.findall(r'[−\-]?\s*\d+\.?\d*(?:/\d+)?', cleaned):
        token = token.replace('−', '-').replace(' ', '')
        try:
            if '/' in token:
                parts = token.split('/')
                numbers.append(float(parts[0]) / float(parts[1]))
            else:
                numbers.append(float(token))
        except ValueError:
            continue

    return sorted(set(numbers))


def normalize_interval_str(s: str) -> str:
    """
    Normalize an interval string for comparison.
    Converts various notations to canonical form: (-oo,+oo), [-3,+oo), etc.
    """
    if not s:
        return ""

    s = s.strip()
    s = re.sub(r'^\$+|\$+$', '', s)
    s = re.sub(r'^["`\']+|["`\']+$', '', s)
    s = re.sub(r'\\left|\\right', '', s)
    s = s.replace('\\;', '')
    s = s.replace('\\,', '')
    s = s.replace('−', '-')

    s = s.replace('\\infty', 'INF_PLACEHOLDER')
    s = s.replace('∞', 'INF_PLACEHOLDER')
    s = s.replace('infinity', 'INF_PLACEHOLDER')
    s = s.replace('inf', 'INF_PLACEHOLDER')

    s = s.replace('+INF_PLACEHOLDER', '+oo')
    s = s.replace('-INF_PLACEHOLDER', '-oo')
    s = s.replace('INF_PLACEHOLDER', '+oo')

    s = s.replace('\\cup', 'U')
    s = s.replace('∪', 'U')
    s = re.sub(r'\bunion\b', 'U', s, flags=re.IGNORECASE)
    s = re.sub(r'(?<=[\)\]])\s*[uU]\s*(?=[\(\[])', 'U', s)
    s = s.replace(' ', '')

    s = re.sub(r'[^\d\w\(\)\[\],\.+\-U]', '', s)

    return s


def parse_expression_answer(answer_str: str) -> str:
    """
    Clean an expression answer for comparison.
    Strips 'y=' prefix and normalizes.
    """
    if not answer_str:
        return ""
    s = answer_str.strip()
    s = re.sub(r'^\$+|\$+$', '', s)
    s = re.sub(r'^[yf]\s*\(?\s*x?\s*\)?\s*=\s*', '', s, flags=re.IGNORECASE)
    return s.strip()


def _loads_json_with_numeric_fraction_repair(candidate: str) -> Optional[Dict]:
    """Load JSON, repairing common LLM JSON slips only after strict parsing fails."""
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    def repair_non_json_infinity(text: str) -> str:
        return re.sub(
            r'(?P<prefix>[\[:,]\s*)(?P<value>[+-]?(?:inf|infinity|Infinity|INF))(?=\s*[\],}])',
            lambda m: (
                m.group("prefix")
                + json.dumps(
                    {
                        "inf": "Infinity",
                        "+inf": "Infinity",
                        "infinity": "Infinity",
                        "+infinity": "Infinity",
                        "-inf": "-Infinity",
                        "-infinity": "-Infinity",
                    }.get(m.group("value").lower(), m.group("value"))
                )
            ),
            text,
        )

    def repair_invalid_json_escapes(text: str) -> str:
        # LLMs often emit LaTeX in JSON strings, e.g. "\infty".  JSON only
        # allows a small escape alphabet; preserve the literal backslash.
        return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

    def repl(match: re.Match) -> str:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator == 0:
            return match.group(0)
        return str(numerator / denominator)

    repaired = candidate
    for repair in (repair_non_json_infinity, repair_invalid_json_escapes):
        repaired = repair(repaired)
    repaired = re.sub(
        r'(?<!["\w])([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)(?!["\w])',
        repl,
        repaired,
    )
    if repaired != candidate:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    return None


def _extract_json_block(text: str) -> Optional[Dict]:
    """Extract the first valid JSON object from text (between { and })."""
    # Try ```json ... ``` blocks first
    for m in re.finditer(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL):
        loaded = _loads_json_with_numeric_fraction_repair(m.group(1).strip())
        if loaded is not None:
            return loaded

    # Try raw { ... } blocks
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                loaded = _loads_json_with_numeric_fraction_repair(text[start:i + 1])
                if loaded is not None:
                    return loaded
                else:
                    start = None

    # Truncated output: the model began a JSON object and the response ended
    # mid-value, so neither path above can close it. Observed in real runs where
    # a fully correct answer was already written (`{"expression": "|x| - 1",`)
    # and scored zero because nothing recovered it. Salvage the complete
    # key/value pairs and discard the partial tail.
    return _salvage_truncated_json(text)


# One complete `"key": <value>` pair, where the value is a finished string,
# number, or literal. A value still being written has no closing quote or
# delimiter and therefore does not match -- which is the point.
_JSON_PAIR = re.compile(
    r'"(?P<key>[A-Za-z_][A-Za-z0-9_]*)"\s*:\s*'
    r'(?P<value>"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|true|false|null)\s*(?=[,}\n])'
)


def _salvage_truncated_json(text: str) -> Optional[Dict]:
    """Recover complete key/value pairs from an unterminated JSON object.

    Returns None unless an opening brace was seen and at least one pair was
    fully written, so ordinary prose cannot be mistaken for JSON.
    """
    if not text:
        return None
    brace = text.rfind('{')
    if brace == -1:
        return None
    fragment = text[brace:]
    if '}' in fragment and fragment.count('{') == fragment.count('}'):
        return None  # Well-formed and already handled above.

    recovered: Dict[str, Any] = {}
    for match in _JSON_PAIR.finditer(fragment):
        try:
            recovered[match.group("key")] = json.loads(match.group("value"))
        except json.JSONDecodeError:
            continue
    return recovered or None


# A captured "expression" that is really narrative text scores near zero even
# when the model derived the right formula, so a prose capture is worse than no
# capture at all: it silently converts a correct answer into a wrong one. These
# two checks reject a candidate that reads like a sentence rather than a formula.
_PROSE_LEAD = re.compile(
    r'^(?:the|this|these|that|there|it|we|i|so|thus|then|here|now|let|let\'s|'
    r'looking|using|based|since|from|because|therefore|hence|first|next|'
    r'finally|to\s|for\s|given|note|observe|notice|substituting|checking|'
    r'verify|verifying|solving|plugging)\b',
    re.IGNORECASE,
)

# Prose markers that never appear in a bare mathematical expression. Piecewise
# answers legitimately contain "for"/"if", so those are matched only as whole
# sentences elsewhere -- not here.
_PROSE_PHRASE = re.compile(
    r'\b(?:can be written|is given by|appears to be|suggests?|means that|'
    r'which gives|we get|so that|in other words|as follows|shows? that|'
    r'indicates?|implies|form where|vertex is at|value equals)\b',
    re.IGNORECASE,
)


def _looks_like_prose(value: str) -> bool:
    """True when a captured field reads as narration rather than an answer.

    Deliberately conservative: a piecewise expression with ``for``/``if``
    clauses, or anything carrying an operator with few English words, is kept.
    """
    if not value:
        return True
    stripped = value.strip()
    # A leading list marker ("- The vertex is at ...") is narration, but a
    # negative expression ("-x + 4") is not.
    if re.match(r'^[-*•]\s+', stripped):
        return True
    if _PROSE_LEAD.match(stripped) or _PROSE_PHRASE.search(stripped):
        return True
    # Count standalone English words; formulas have few, sentences have many.
    words = re.findall(r'\b[A-Za-z]{3,}\b', stripped)
    allowed = {"sqrt", "abs", "log", "ln", "exp", "sin", "cos", "tan", "inf",
               "for", "if", "else", "and", "or", "piecewise", "otherwise", "where"}
    prose_words = [w for w in words if w.lower() not in allowed]
    if len(prose_words) >= 3:
        return True
    # A sentence ending in a colon is an introduction to the answer, not the
    # answer -- the formula follows on the next line.
    return stripped.endswith(":")


def _extract_labeled_fields(text: str) -> Dict[str, str]:
    """
    Fallback: extract answers from labeled markers like:
    **Expression:** ..., **Domain:** ..., (a) ..., etc.

    Used only when no JSON object could be recovered. Each candidate is
    screened by :func:`_looks_like_prose`, because the L3 prompt numbers its
    tasks ``(h)/(i)/(j)`` and models freely emit ``(a)``-style list markers in
    their narration -- an unscreened positional pattern captured that narration
    as the answer.
    """
    result = {}

    patterns = {
        "expression": [
            r'\*\*(?:Expression|表达式)[:\s]*\*\*[:\s]*(.*?)(?:\n|$)',
            r'(?:Expression|表达式|解析式)[：:\s]+(.*?)(?:\n|$)',
            # Anchored to line start: a bare "(a)" mid-sentence is prose, and
            # the letter must be followed by an answer, not more narration.
            r'(?m)^\s*\(a\)\s*(.*?)(?:\n|$)',
        ],
        "domain": [
            r'\*\*(?:Domain|定义域)[:\s]*\*\*[:\s]*(.*?)(?:\n|$)',
            r'(?:Domain|定义域)[：:\s]+(.*?)(?:\n|$)',
            r'(?m)^\s*\(b\)\s*(.*?)(?:\n|$)',
        ],
        "range": [
            r'\*\*(?:Range|值域)[:\s]*\*\*[:\s]*(.*?)(?:\n|$)',
            r'(?:Range|值域)[：:\s]+(.*?)(?:\n|$)',
            r'(?m)^\s*\(c\)\s*(.*?)(?:\n|$)',
        ],
        "coordinates": [
            r'\*\*(?:Coordinates?|坐标|交点)[:\s]*\*\*[:\s]*(.*?)(?:\n|$)',
            r'(?:Coordinates?|坐标|交点|Key\s*points?)[：:\s]+(.*?)(?:\n|$)',
            r'(?m)^\s*\(d\)\s*(.*?)(?:\n|$)',
        ],
    }

    for key, pats in patterns.items():
        for pat in pats:
            matched = False
            for m in re.finditer(pat, text, re.IGNORECASE):
                val = m.group(1).strip()
                val = re.sub(r'^\$+|\$+$', '', val).strip()
                val = re.sub(r'^\*+|\*+$', '', val).strip()
                if not val:
                    continue
                # Only `expression` is screened. domain/range are interval
                # notation that the prose heuristic would not judge reliably,
                # and they are not where the observed failures occurred.
                if key == "expression" and _looks_like_prose(val):
                    continue
                result[key] = val
                matched = True
                break
            if matched:
                break

    return result


def parse_comprehensive_answer(raw_response: str) -> Dict[str, Any]:
    """
    Extract structured answers for all sub-questions from LLM output.

    Returns dict with keys:
        expression, domain, range, coordinates, predictions, rate_of_change,
        intercepts, extrema, monotonicity, symmetry, inference_explanation
    """
    result: Dict[str, Any] = {
        "expression": "", "domain": "", "range": "", "coordinates": "",
        "predictions": {}, "rate_of_change": None,
        "intercepts": {}, "extrema": [], "monotonicity": [], "symmetry": "",
        "inference_explanation": "",
    }

    if not raw_response:
        return result

    json_data = _extract_json_block(raw_response)
    if json_data and isinstance(json_data, dict):
        _text_keys = ["expression", "domain", "range", "coordinates", "inference_explanation"]
        for key in _text_keys:
            val = json_data.get(key, "")
            if isinstance(val, list):
                if key == "coordinates" and val and isinstance(val[0], (list, tuple)):
                    val = ", ".join(f"({v[0]},{v[1]})" for v in val)
                else:
                    val = ", ".join(str(v) for v in val)
            elif not isinstance(val, str):
                val = str(val) if val is not None else ""
            result[key] = val

        # predictions dict
        preds_raw = json_data.get("predictions", {})
        if isinstance(preds_raw, dict):
            cleaned = {}
            for k, v in preds_raw.items():
                try:
                    cleaned[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass
            result["predictions"] = cleaned

        # rate_of_change
        roc_raw = json_data.get("rate_of_change")
        if roc_raw is not None:
            try:
                result["rate_of_change"] = float(roc_raw)
            except (TypeError, ValueError):
                pass

        # --- New L1/L2 fields ---
        result["intercepts"] = _parse_intercepts(json_data.get("intercepts"))
        result["extrema"] = _parse_extrema(json_data.get("extrema"))
        result["monotonicity"] = _parse_monotonicity(json_data.get("monotonicity"))
        result["symmetry"] = str(json_data.get("symmetry", "")).strip().lower()

        if any(v for k, v in result.items() if k in _text_keys and k != "coordinates"):
            return result

    labeled = _extract_labeled_fields(raw_response)
    for key in ["expression", "domain", "range", "coordinates"]:
        if not result[key] and key in labeled:
            result[key] = labeled[key]

    return result


def _parse_intercepts(raw) -> Dict[str, Any]:
    """Parse intercepts from LLM JSON: {x_intercepts: [...], y_intercept: number}."""
    out: Dict[str, Any] = {"x_intercepts": [], "y_intercept": None}
    if not isinstance(raw, dict):
        return out
    xi = raw.get("x_intercepts", [])
    if isinstance(xi, list):
        for v in xi:
            try:
                out["x_intercepts"].append(float(v))
            except (TypeError, ValueError):
                pass
    yi = raw.get("y_intercept")
    if yi is not None:
        try:
            out["y_intercept"] = float(yi)
        except (TypeError, ValueError):
            pass
    return out


def _parse_extrema(raw) -> List[Dict[str, Any]]:
    """Parse extrema list: [{x, y, type}, ...]."""
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            result.append({
                "x": float(item.get("x", 0)),
                "y": float(item.get("y", 0)),
                "type": str(item.get("type", "")).lower(),
            })
        except (TypeError, ValueError):
            pass
    return result


def _parse_monotonicity(raw) -> List[Dict[str, Any]]:
    """Parse monotonicity: [{interval: [a,b], direction}, ...]."""
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        interval = item.get("interval", [])
        direction = str(item.get("direction", "")).lower()
        if isinstance(interval, str):
            bounds = [part.strip() for part in re.sub(r'^[\(\[]|[\)\]]$', '', interval).split(",", 1)]
            if len(bounds) == 2:
                interval = bounds
        if isinstance(interval, list) and len(interval) == 2 and direction:
            try:
                result.append({
                    "interval": [float(interval[0]), float(interval[1])],
                    "direction": direction,
                })
            except (TypeError, ValueError):
                pass
    return result


# ==================== Per-Step Agent Parsers ====================

def parse_L1_answer(raw_response: str) -> Dict[str, Any]:
    """Parse Step 1 (L1 Reading) output: coordinates, intercepts, extrema."""
    result: Dict[str, Any] = {
        "coordinates": "",
        "intercepts": {"x_intercepts": [], "y_intercept": None},
        "extrema": [],
    }
    if not raw_response:
        return result

    json_data = _extract_json_block(raw_response)
    if isinstance(json_data, dict):
        coords = json_data.get("coordinates", "")
        if isinstance(coords, list):
            if coords and isinstance(coords[0], (list, tuple)):
                coords = ", ".join(f"({v[0]},{v[1]})" for v in coords)
            else:
                coords = ", ".join(str(v) for v in coords)
        result["coordinates"] = str(coords) if coords else ""
        result["intercepts"] = _parse_intercepts(json_data.get("intercepts"))
        result["extrema"] = _parse_extrema(json_data.get("extrema"))
    return result


def parse_L2_answer(raw_response: str) -> Dict[str, Any]:
    """Parse Step 2 (L2 Analysis) output: predictions, rate_of_change, monotonicity, symmetry."""
    result: Dict[str, Any] = {
        "predictions": {},
        "rate_of_change": None,
        "monotonicity": [],
        "symmetry": "",
    }
    if not raw_response:
        return result

    json_data = _extract_json_block(raw_response)
    if isinstance(json_data, dict):
        preds_raw = json_data.get("predictions", {})
        if isinstance(preds_raw, dict):
            cleaned = {}
            for k, v in preds_raw.items():
                try:
                    cleaned[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass
            result["predictions"] = cleaned

        roc_raw = json_data.get("rate_of_change")
        if roc_raw is not None:
            try:
                result["rate_of_change"] = float(roc_raw)
            except (TypeError, ValueError):
                pass

        result["monotonicity"] = _parse_monotonicity(json_data.get("monotonicity"))
        result["symmetry"] = str(json_data.get("symmetry", "")).strip().lower()
    return result


def parse_L3_answer(raw_response: str) -> Dict[str, Any]:
    """Parse Step 3 (L3 Reasoning) output: expression, domain, range, inference explanation."""
    result: Dict[str, Any] = {
        "expression": "",
        "domain": "",
        "range": "",
        "inference_explanation": "",
    }
    if not raw_response:
        return result

    json_data = _extract_json_block(raw_response)
    if isinstance(json_data, dict):
        for key in ("expression", "domain", "range", "inference_explanation"):
            val = json_data.get(key, "")
            if not isinstance(val, str):
                val = str(val) if val is not None else ""
            result[key] = val
        if any(result[k] for k in result):
            return result

    labeled = _extract_labeled_fields(raw_response)
    for key in ("expression", "domain", "range", "inference_explanation"):
        if not result[key] and key in labeled:
            result[key] = labeled[key]

    # Last resort: the model narrated its way to a formula without emitting JSON
    # or a label. Observed repeatedly in real runs -- the derivation ends with a
    # standalone "f(x) = |x| - 1" line and every earlier extractor returns
    # nothing, scoring a fully correct answer as zero.
    if not result["expression"]:
        recovered = _recover_expression_from_prose(raw_response)
        if recovered:
            result["expression"] = recovered
    return result


# A standalone declaration line: optional "y ="/"f(x) =" then the formula. The
# tail must contain a digit or variable so headings like "Expression:" are
# excluded, and the line is screened for prose before being accepted.
_DECLARATION_LINE = re.compile(
    r'^\s*(?:so|thus|therefore|hence)?[,:\s]*'
    r'(?:the\s+)?(?:function|expression|answer|equation)?[,:\s]*'
    r'(?:y|f\s*\(\s*x\s*\))\s*=\s*(?P<body>[^\n]+?)\s*$',
    re.IGNORECASE,
)


def _recover_expression_from_prose(text: str) -> str:
    """Recover a formula stated in narration, or "" when none is stated.

    Scans from the end because the final declaration is the model's conclusion;
    earlier ones are usually candidates it went on to reject. Returns "" rather
    than guessing when nothing passes the prose screen -- a wrong expression is
    scored, whereas an empty one is at least honestly empty.
    """
    if not text:
        return ""
    for line in reversed(text.splitlines()):
        match = _DECLARATION_LINE.match(line)
        if not match:
            continue
        body = match.group("body").strip()
        body = re.sub(r'^\$+|\$+$', '', body).strip()
        body = body.rstrip('.,;')
        # Drop a trailing verification comment: "|x| - 1 ✓" or "2^x  (checks out)".
        body = re.sub(r'\s*[✓✔].*$', '', body).strip()
        if not body or _looks_like_prose(body):
            continue
        if not re.search(r'[\dxX]', body):
            continue
        return body
    return ""


# ==================== Judge Parser ====================

_JUDGE_KEYS = (
    "factual_accuracy",
    "inference_quality",
    "reasoning_coherence",
    "evidence_support",
)


def parse_judge_response(raw_response: str) -> Dict[str, Any]:
    """
    Parse the Judge LLM's rubric-based evaluation response.

    Expected JSON fields (each 0-5):
        factual_accuracy, inference_quality,
        reasoning_coherence, evidence_support

    Returns a dict with those scores (float) and ``comments`` (str).
    Missing or unparseable fields default to 0.
    """
    result: Dict[str, Any] = {k: 0.0 for k in _JUDGE_KEYS}
    result["comments"] = ""

    if not raw_response:
        return result

    json_data = _extract_json_block(raw_response)
    if not isinstance(json_data, dict):
        return result

    for key in _JUDGE_KEYS:
        raw_val = json_data.get(key)
        if raw_val is not None:
            try:
                val = float(raw_val)
                result[key] = max(0.0, min(5.0, val))
            except (TypeError, ValueError):
                pass

    result["comments"] = str(json_data.get("comments", ""))
    return result
