"""
Base adapter for LLM intersection recognition.
Subclasses implement the actual API calls.
"""

import base64
import functools
import json
import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=32)
def _cached_encode_file(path: str, mtime_ns: int) -> str:
    """Read a file and return its base64 encoding. Cached by (path, mtime)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

_MAX_API_RETRIES = 5
_BASE_RETRY_DELAY = 4.0   # seconds
_MAX_RETRY_DELAY = 90.0   # cap (generous for rate-limit recovery)

_RATE_LIMIT_PATTERNS = [
    "rate limit", "rate_limit", "too many requests",
    "429", "quota exceeded", "resource exhausted",
]

_TRANSIENT_PATTERNS = [
    *_RATE_LIMIT_PATTERNS,
    "timeout", "timed out",
    "500", "502", "503", "504", "529",
    "connection", "overloaded", "server error", "internal error",
    "service unavailable", "temporarily unavailable",
    "capacity", "retry",
]


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if *exc* specifically indicates a rate-limit / quota error."""
    msg = str(exc).lower()
    cls = type(exc).__name__.lower()
    for p in _RATE_LIMIT_PATTERNS:
        if p in msg or p in cls:
            return True
    code = getattr(exc, "status_code", None)
    return code == 429


def is_transient_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a transient / retriable failure."""
    msg = str(exc).lower()
    cls = type(exc).__name__.lower()
    for p in _TRANSIENT_PATTERNS:
        if p in msg or p in cls:
            return True
    code = getattr(exc, "status_code", None)
    if code in (429, 500, 502, 503, 504, 529):
        return True
    return False


class BaseLLMAdapter(ABC):
    """
    Base class for LLM adapters.
    Implement `_call_api` in subclasses with your specific LLM API.
    """

    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
        self.last_finish_reason = None
        self.last_response_metadata = {}

    def reset_client(self) -> None:
        """Drop a cached transport client before a long-run sample boundary."""
        if hasattr(self, "_client"):
            setattr(self, "_client", None)

    @abstractmethod
    def _call_api(self, image_base64: str, prompt: str) -> str:
        """
        Send image and prompt to LLM, return raw text response.
        Subclasses must implement this with actual API call.

        Args:
            image_base64: Base64-encoded image (PNG/JPEG)
            prompt: The prompt text asking for intersection coordinates

        Returns:
            Raw text response from the LLM
        """
        raise NotImplementedError("Subclasses must implement _call_api")

    def _encode_image(self, image: np.ndarray | str | Path) -> str:
        """Convert image to base64 string. File paths are LRU-cached."""
        if isinstance(image, (str, Path)):
            p = str(image)
            try:
                mtime_ns = os.stat(p).st_mtime_ns
            except OSError:
                mtime_ns = 0
            return _cached_encode_file(p, mtime_ns)
        # numpy array (BGR)
        _, buffer = cv2.imencode(".png", image)
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def _parse_response(self, response: str) -> List[Tuple[int, int]]:
        """
        Parse LLM response to extract integer (x, y) coordinate pairs.

        Strategy:
          1. Try JSON parsing first (preferred, structured output)
          2. Fall back to regex extraction of (x, y) pairs
        """
        # --- Strategy 1: JSON parsing ---
        coords = self._try_parse_json(response)
        if coords is not None:
            return coords

        # --- Strategy 2: Regex fallback ---
        pattern = r"\((-?\d+)\s*,\s*(-?\d+)\)"
        matches = re.findall(pattern, response)
        return [(int(x), int(y)) for x, y in matches]

    @staticmethod
    def _try_parse_json(response: str) -> List[Tuple[int, int]] | None:
        """
        Try to parse JSON response.
        Accepts formats:
          - {"points": [[x1,y1], [x2,y2], ...]}
          - [[x1,y1], [x2,y2], ...]
          - [{"x":1,"y":2}, ...]
        Returns None if JSON parsing fails.
        """
        # Extract JSON from possible markdown code fences
        text = response.strip()
        # Remove ```json ... ``` wrapper if present
        json_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Try to find a JSON object or array in the text
        # Look for the first { or [ and match to end
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start_idx = text.find(start_char)
            if start_idx == -1:
                continue
            # Find the matching closing bracket
            end_idx = text.rfind(end_char)
            if end_idx == -1 or end_idx <= start_idx:
                continue
            candidate = text[start_idx:end_idx + 1]
            try:
                data = json.loads(candidate)
                return BaseLLMAdapter._extract_points_from_json(data)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        return None

    @staticmethod
    def _extract_points_from_json(data) -> List[Tuple[int, int]]:
        """Extract (x, y) tuples from parsed JSON data."""
        points_list = None

        if isinstance(data, dict):
            # {"points": [...]} or {"coordinates": [...]} etc.
            for key in ("points", "coordinates", "intersections", "data"):
                if key in data:
                    points_list = data[key]
                    break
            if points_list is None:
                # Try the first list value in the dict
                for v in data.values():
                    if isinstance(v, list):
                        points_list = v
                        break
        elif isinstance(data, list):
            points_list = data

        if not points_list or not isinstance(points_list, list):
            return []

        result = []
        for item in points_list:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    result.append((int(item[0]), int(item[1])))
                except (ValueError, TypeError):
                    continue
            elif isinstance(item, dict):
                x = item.get("x", item.get("X"))
                y = item.get("y", item.get("Y"))
                if x is not None and y is not None:
                    try:
                        result.append((int(x), int(y)))
                    except (ValueError, TypeError):
                        continue
        return result

    def _call_with_retry(self, image_b64: str, prompt: str) -> str:
        """Wrap ``_call_api`` with retry + exponential backoff."""
        last_exc = None
        for attempt in range(1, _MAX_API_RETRIES + 1):
            try:
                return self._call_api(image_b64, prompt)
            except Exception as exc:
                last_exc = exc
                if attempt >= _MAX_API_RETRIES or not is_transient_error(exc):
                    raise
                if _is_rate_limit_error(exc):
                    delay = min(
                        _BASE_RETRY_DELAY * (3 ** (attempt - 1)),
                        _MAX_RETRY_DELAY,
                    )
                else:
                    delay = min(
                        _BASE_RETRY_DELAY * (2 ** (attempt - 1)),
                        _MAX_RETRY_DELAY,
                    )
                delay *= 0.5 + random.random()
                logger.warning(
                    "[%s] API attempt %d/%d failed (%s: %s), retry in %.1fs",
                    self.model_name, attempt, _MAX_API_RETRIES,
                    type(exc).__name__, str(exc)[:200], delay,
                )
                time.sleep(delay)
        raise last_exc

    def recognize_intersections(
        self,
        image: np.ndarray | str | Path,
        prompt: str,
    ) -> Tuple[List[Tuple[int, int]], str]:
        """
        Send image to LLM and get recognized intersection coordinates.

        Args:
            image: Image as numpy array (BGR), file path, or Path object
            prompt: Prompt text (use llm_prompt_template.txt with filled placeholders)

        Returns:
            (list of (x, y) integer tuples, raw response text)
        """
        image_b64 = self._encode_image(image)
        response = self._call_with_retry(image_b64, prompt)
        coords = self._parse_response(response)
        return coords, response

    def call_with_prompt(
        self,
        image: np.ndarray | str | Path | None,
        prompt: str,
    ) -> str:
        """
        Generic method: send image + prompt to LLM, return raw response text.
        If image is None, falls back to call_text_only for text-only prompts.

        Args:
            image: Image as numpy array (BGR), file path, Path object, or None
            prompt: The prompt text

        Returns:
            Raw text response from the LLM
        """
        if image is None:
            return self.call_text_only(prompt)
        image_b64 = self._encode_image(image)
        return self._call_with_retry(image_b64, prompt)

    def call_text_only(self, prompt: str) -> str:
        """
        Send text-only prompt to LLM (no image). Used for reasoning from
        numerical analysis reports. Default: sends a minimal 1x1 white image.
        Subclasses may override for native text-only API calls.
        """
        dummy = np.ones((1, 1, 3), dtype=np.uint8) * 255
        image_b64 = self._encode_image(dummy)
        return self._call_with_retry(image_b64, prompt)

    @staticmethod
    def _parse_expr_response(response: str) -> List[str]:
        """
        Parse LLM response to extract function expression strings.

        Expects JSON format: {"functions": ["expr1", "expr2", ...]}
        Falls back to heuristic extraction.

        Returns:
            List of expression strings.
        """
        text = response.strip()

        # Remove markdown code fences if present
        json_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Try to find JSON object
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start_idx = text.find(start_char)
            if start_idx == -1:
                continue
            end_idx = text.rfind(end_char)
            if end_idx == -1 or end_idx <= start_idx:
                continue
            candidate = text[start_idx:end_idx + 1]
            try:
                data = json.loads(candidate)
                return BaseLLMAdapter._extract_functions_from_json(data)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # Fallback: try to extract y=... or f(x)=... patterns
        patterns = [
            r"y\s*=\s*(.+?)(?:\n|$)",
            r"f\s*\(x\)\s*=\s*(.+?)(?:\n|$)",
            r"(?:^|\n)\s*(\d+)\.\s*(.+?)(?:\n|$)",
        ]
        results = []
        for pattern in patterns[:2]:
            matches = re.findall(pattern, text)
            if matches:
                results.extend(m.strip().rstrip(",;") for m in matches)
        if results:
            return results

        return []

    @staticmethod
    def _extract_functions_from_json(data) -> List[str]:
        """Extract function expression strings from parsed JSON data."""
        func_list = None

        if isinstance(data, dict):
            # {"functions": [...]} or {"expressions": [...]} etc.
            for key in ("functions", "expressions", "curves", "formulas", "equations"):
                if key in data:
                    func_list = data[key]
                    break
            if func_list is None:
                # Try the first list value in the dict
                for v in data.values():
                    if isinstance(v, list):
                        func_list = v
                        break
        elif isinstance(data, list):
            func_list = data

        if not func_list or not isinstance(func_list, list):
            return []

        results = []
        for item in func_list:
            if isinstance(item, str):
                results.append(item.strip())
            elif isinstance(item, dict):
                # {"expression": "x**2"} or {"formula": "x**2"}
                for key in ("expression", "formula", "expr", "function", "f"):
                    if key in item:
                        results.append(str(item[key]).strip())
                        break
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
