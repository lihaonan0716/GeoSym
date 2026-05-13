"""
OpenAI GPT Vision adapter for intersection recognition.

API docs: https://platform.openai.com/docs/quickstart
Get API key: https://platform.openai.com/api-keys

Usage:
    adapter = OpenAIAdapter(api_key="sk-xxx", model="gpt-5.2")
    coords, raw_response = adapter.recognize_intersections(image_path, prompt)
"""

from openai import OpenAI

from .base import BaseLLMAdapter
from i18n import t, get_lang


class OpenAIAdapter(BaseLLMAdapter):
    """
    OpenAI GPT Vision adapter.

    Supported models (latest first):
        - gpt-5.2          : GPT-5.2 (latest, recommended)
        - gpt-4.1           : GPT-4.1
        - gpt-4o           : GPT-4o
    """

    def __init__(self, api_key: str = "", model: str = "gpt-5.2"):
        super().__init__(model_name=model)
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError(t("eval_openai_key_missing", get_lang()))
            self._client = OpenAI(api_key=self.api_key, timeout=180.0)
        return self._client

    def _call_api(self, image_base64: str, prompt: str) -> str:
        """Call OpenAI API with image and prompt."""
        client = self._get_client()

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
            max_completion_tokens=4096,
            temperature=0.0,
            seed=42,
        )

        return response.choices[0].message.content or ""
