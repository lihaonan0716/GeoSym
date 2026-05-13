"""
Moonshot Kimi Vision adapter for intersection recognition.
Uses OpenAI-compatible API via https://api.moonshot.cn/v1.

API docs: https://platform.moonshot.cn/docs/api/chat
Vision guide: https://platform.moonshot.cn/docs/guide/use-kimi-vision-model
Get API key: https://platform.moonshot.cn/console/api-keys

Usage:
    adapter = KimiAdapter(api_key="sk-xxx")
    coords, raw_response = adapter.recognize_intersections(image_path, prompt)
"""

from openai import OpenAI

from .base import BaseLLMAdapter

# Moonshot Kimi API endpoint
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"


class KimiAdapter(BaseLLMAdapter):
    """
    Moonshot Kimi Vision adapter.
    Uses OpenAI SDK with Moonshot base_url (fully OpenAI-compatible).

    Supported vision models:
        - moonshot-v1-32k-vision-preview  : 32k context vision model (recommended)
        - moonshot-v1-8k-vision-preview   : 8k context vision model
        - moonshot-v1-128k-vision-preview : 128k context vision model

    Note: Image input only supports base64 encoding (not URL).
    """

    def __init__(self, api_key: str = "", model: str = "moonshot-v1-128k-vision-preview"):
        # Default model kept in sync with MODEL_REGISTRY in app.py
        super().__init__(model_name=model)
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Moonshot API key not provided. "
                    "Get one at https://platform.moonshot.cn/console/api-keys"
                )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=MOONSHOT_BASE_URL,
                timeout=180.0,
            )
        return self._client

    def _call_api(self, image_base64: str, prompt: str) -> str:
        """Call Kimi Vision API with image and prompt."""
        client = self._get_client()

        # Kimi-K2.5 only allows temperature=1; older moonshot models accept 0.0
        temp = 1.0 if "k2" in self.model.lower() else 0.0

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
            max_tokens=4096,
            temperature=temp,
        )

        return response.choices[0].message.content or ""
