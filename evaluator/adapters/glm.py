"""
ZhipuAI GLM-4.6V Vision adapter for intersection recognition.
Uses OpenAI-compatible API via https://open.bigmodel.cn/api/paas/v4/.

API docs: https://docs.bigmodel.cn/cn/guide/develop/openai/introduction
Vision guide: https://docs.bigmodel.cn/cn/guide/models/vlm/glm-4.6v
Get API key: https://bigmodel.cn/usercenter/proj-mgmt/apikeys

Usage:
    adapter = GLMAdapter(api_key="xxx")
    coords, raw_response = adapter.recognize_intersections(image_path, prompt)
"""

import os

from openai import OpenAI

from .base import BaseLLMAdapter

# ZhipuAI OpenAI-compatible endpoint
ZHIPUAI_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


class GLMAdapter(BaseLLMAdapter):
    """
    ZhipuAI GLM Vision adapter.
    Uses OpenAI SDK with ZhipuAI base_url (fully OpenAI-compatible).

    Supported vision models:
        - glm-4.6v          : Flagship vision model (106B), 128K context (recommended)
        - glm-4.6v-flashx   : Lightweight fast version (9B)
        - glm-4.6v-flash    : Free version
        - glm-4.5v          : Previous generation MOE vision model
    """

    def __init__(self, api_key: str = "", model: str = "glm-4.6v"):
        super().__init__(model_name=model)
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError('ZhipuAI API key not provided. Get one at https://bigmodel.cn/usercenter/proj-mgmt/apikeys')
            timeout = float(os.environ.get("GEOSYM_GLM_TIMEOUT_SECONDS", "180.0"))
            max_retries = int(os.environ.get("GEOSYM_GLM_MAX_RETRIES", "2"))
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=ZHIPUAI_BASE_URL,
                timeout=timeout,
                max_retries=max_retries,
            )
        return self._client

    def _call_api(self, image_base64: str, prompt: str) -> str:
        """Call GLM Vision API with image and prompt."""
        client = self._get_client()

        max_tokens = int(os.environ.get("GEOSYM_GLM_MAX_TOKENS", "4096"))
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
            max_tokens=max_tokens,
            temperature=0.0,
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        # Debug: log finish reason if not 'stop'
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason and finish_reason != "stop":
            print(f"[GLM] Warning: finish_reason={finish_reason}, content length={len(content)}")

        if not content:
            print(f"[GLM] Warning: empty response. finish_reason={finish_reason}")

        return content
