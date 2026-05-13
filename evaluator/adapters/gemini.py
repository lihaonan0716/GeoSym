"""
Google Gemini adapter for intersection recognition.
Uses the google-genai SDK (Gen AI SDK).

API docs: https://ai.google.dev/gemini-api/docs
Get API key: https://aistudio.google.com/apikey

Usage:
    adapter = GeminiAdapter(api_key="xxx", model="gemini-3-pro-preview")
    coords, raw_response = adapter.recognize_intersections(image_path, prompt)
"""

import base64
import logging
import warnings

# Suppress Gemini SDK "non-text parts" logger warnings globally.
# The SDK uses logger.warning() (not warnings.warn()), so we must
# raise the log level on the specific google.genai.types logger.
logging.getLogger("google.genai.types").setLevel(logging.ERROR)

# Suppress Gemini SDK thought_signature warnings globally
# These occur when response contains thinking parts alongside text parts
warnings.filterwarnings("ignore", message=".*non-text parts.*thought_signature.*")

from google import genai
from google.genai import types

from .base import BaseLLMAdapter
from i18n import t, get_lang


class GeminiAdapter(BaseLLMAdapter):
    """
    Google Gemini Vision adapter.
    Uses google-genai SDK for multimodal image understanding.

    Supported models (latest first):
        - gemini-3-pro-preview     : Gemini 3 Pro (latest, recommended)
        - gemini-3-flash-preview   : Gemini 3 Flash (fast)
        - gemini-2.5-pro-preview-06-05 : Gemini 2.5 Pro
        - gemini-2.5-flash-preview-05-20 : Gemini 2.5 Flash
        - gemini-2.0-flash        : Gemini 2.0 Flash
    """

    def __init__(self, api_key: str = "", model: str = "gemini-3-pro-preview"):
        super().__init__(model_name=model)
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise ValueError(t("eval_gemini_key_missing", get_lang()))
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _call_api(self, image_base64: str, prompt: str) -> str:
        """
        Call Gemini API with image and prompt.
        Transient failures are retried automatically by BaseLLMAdapter.
        """
        client = self._get_client()

        image_bytes = base64.b64decode(image_base64)

        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(
                thinking_budget=128,
            ),
        )

        response = client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/png",
                        ),
                    ],
                )
            ],
            config=config,
        )

        result_text = self._extract_text(response)
        if not result_text:
            debug_info = self._debug_response(response)
            return t("eval_gemini_empty_response", get_lang()).format(detail=debug_info)

        return result_text

    def call_text_only(self, prompt: str) -> str:
        """Text-only call to Gemini (no image).
        Retries transient failures with the same backoff as image calls."""
        from .base import is_transient_error, _MAX_API_RETRIES, _BASE_RETRY_DELAY, _MAX_RETRY_DELAY
        import random, time

        client = self._get_client()
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        )

        last_exc = None
        for attempt in range(1, _MAX_API_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=prompt)],
                        )
                    ],
                    config=config,
                )
                result = self._extract_text(response)
                if result:
                    return result
                return ""
            except Exception as exc:
                last_exc = exc
                if attempt >= _MAX_API_RETRIES or not is_transient_error(exc):
                    raise
                delay = min(_BASE_RETRY_DELAY * (2 ** (attempt - 1)), _MAX_RETRY_DELAY)
                delay *= 0.5 + random.random()
                logging.getLogger(__name__).warning(
                    "[%s] text-only attempt %d/%d failed (%s), retry in %.1fs",
                    self.model, attempt, _MAX_API_RETRIES, exc, delay,
                )
                time.sleep(delay)
        raise last_exc

    @staticmethod
    def _extract_text(response) -> str:
        """Extract text from Gemini response, handling various response structures.

        Iterates over parts directly instead of using ``response.text``
        to avoid the SDK warning about non-text parts (thought_signature).
        """
        try:
            if response.candidates:
                text_parts = []
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, "thought") and part.thought:
                                continue
                            if hasattr(part, "text") and part.text:
                                text_parts.append(part.text)
                if text_parts:
                    return "\n".join(text_parts)
        except (AttributeError, IndexError):
            pass

        # Last resort: try response.text (may emit SDK warning)
        try:
            if response.text:
                return response.text
        except (AttributeError, ValueError):
            pass

        return ""

    @staticmethod
    def _debug_response(response) -> str:
        """Generate debug info from response for troubleshooting."""
        parts = []
        try:
            if hasattr(response, "candidates") and response.candidates:
                for i, c in enumerate(response.candidates):
                    parts.append(f"candidate[{i}].finish_reason={getattr(c, 'finish_reason', '?')}")
                    if hasattr(c, "safety_ratings") and c.safety_ratings:
                        for sr in c.safety_ratings:
                            parts.append(f"  safety: {sr.category}={sr.probability}")
            else:
                parts.append("No candidates in response")

            if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                parts.append(f"prompt_feedback: {response.prompt_feedback}")
        except Exception as e:
            parts.append(f"Debug error: {e}")

        return " | ".join(parts) if parts else "No debug info available"
