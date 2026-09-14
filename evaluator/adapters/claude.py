"""
Anthropic-compatible Claude adapter for intersection recognition.
Routes through 9router by default.

Usage:
    adapter = ClaudeAdapter(api_key="...", model="cc/claude-opus-4-8")
"""

import os

import anthropic

from .base import BaseLLMAdapter


class ClaudeAdapter(BaseLLMAdapter):
    """
    Anthropic-compatible Claude Vision adapter routed through 9router by default.

    Supported models:
        - cc/claude-opus-4-8       : Claude Opus 4.8 via 9router
        - cc/claude-opus-5         : Claude Opus 5 via 9router
    """

    def __init__(self, api_key: str = "", model: str = "cc/claude-opus-4-8", base_url: str = ""):
        super().__init__(model_name=model)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise ValueError("9router API key not provided for Claude model")

        base_url = self.base_url or os.environ.get("GEOSYM_CLAUDE_BASE_URL", "")
        if not base_url:
            raise ValueError("OpenAI-compatible base URL not configured for Claude model")
        self._client = anthropic.Anthropic(
            api_key=self.api_key.strip(),
            base_url=base_url,
            timeout=180.0,
        )
        return self._client

    def _call_api(self, image_base64: str, prompt: str) -> str:
        """
        Call Claude Messages API with image and prompt.
        """
        client = self._get_client()

        message = client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            temperature=0.0,
        )

        # Extract text from response
        if message.content:
            for block in message.content:
                if hasattr(block, "text") and block.text:
                    return block.text
        return ""
