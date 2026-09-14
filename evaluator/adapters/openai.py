"""
OpenAI-compatible GPT Vision adapter for intersection recognition.

Usage:
    adapter = OpenAIAdapter(api_key="...", model="gpt-5.5")
    coords, raw_response = adapter.recognize_intersections(image_path, prompt)
"""

import os

from openai import OpenAI

from .base import BaseLLMAdapter

class OpenAIAdapter(BaseLLMAdapter):
    """
    OpenAI-compatible GPT Vision adapter routed through 9router by default.

    Supported models (latest first):
        - gpt-5.5          : GPT-5.5 via 9router
    """

    def __init__(self, api_key: str = "", model: str = "gpt-5.5", base_url: str = ""):
        super().__init__(model_name=model)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError('OpenAI API key not provided. Please set api_key.')
            base_url = self.base_url or os.environ.get("GEOSYM_OPENAI_BASE_URL", "")
            if not base_url:
                raise ValueError("OpenAI-compatible base URL not configured for GPT model")
            timeout = float(os.environ.get("GEOSYM_OPENAI_TIMEOUT_SECONDS", "180.0"))
            max_retries = int(os.environ.get("GEOSYM_OPENAI_MAX_RETRIES", "2"))
            self._client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        return self._client

    def _call_api(self, image_base64: str, prompt: str) -> str:
        """Call OpenAI API with image and prompt."""
        client = self._get_client()

        max_completion_tokens = int(os.environ.get("GEOSYM_OPENAI_MAX_COMPLETION_TOKENS", "4096"))
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=max_completion_tokens,
            temperature=0.0,
            seed=42,
        )

        return response.choices[0].message.content or ""
