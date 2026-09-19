"""
Alibaba Cloud Bailian (百炼) Qwen-VL adapter for intersection recognition.
Uses OpenAI-compatible API via DashScope endpoint.

API docs: https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions
Get API key: https://bailian.console.aliyun.com/?tab=model#/api-key

Usage:
    adapter = QwenVLAdapter(api_key="sk-xxx")
    coords, raw_response = adapter.recognize_intersections(image_path, prompt)
"""

import os

from openai import OpenAI

from .base import BaseLLMAdapter

# DashScope OpenAI-compatible endpoint (Beijing region)
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenVLAdapter(BaseLLMAdapter):
    """
    Alibaba Cloud Bailian Qwen-VL adapter.
    Uses OpenAI SDK with DashScope base_url (fully OpenAI-compatible).

    Supported models (latest to older):
        - qwen3-vl-plus         : Qwen3 flagship vision model (recommended)
        - qwen3-vl-flash        : Qwen3 fast vision model
        - qwen-vl-max-latest    : Qwen2.5-VL max
        - qwen-vl-plus-latest   : Qwen2.5-VL plus
    """

    def __init__(self, api_key: str = "", model: str = "qwen3-vl-plus"):
        super().__init__(model_name=model)
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError('Bailian API key not provided. Get one at https://bailian.console.aliyun.com/?tab=model#/api-key')
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=DASHSCOPE_BASE_URL,
                timeout=float(os.environ.get("GEOSYM_QWEN_TIMEOUT_SECONDS", "180.0")),
                max_retries=int(os.environ.get("GEOSYM_QWEN_MAX_RETRIES", "2")),
            )
        return self._client

    def _call_api(self, image_base64: str, prompt: str) -> str:
        """Call Qwen-VL via DashScope OpenAI-compatible API."""
        client = self._get_client()

        max_tokens = int(os.environ.get("GEOSYM_QWEN_MAX_TOKENS", "4096"))
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
            extra_body={
                "enable_thinking": False,
            },
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)
        self.last_finish_reason = finish_reason
        self.last_response_metadata = {
            "finish_reason": finish_reason,
            "model": getattr(response, "model", self.model),
            "response_id": getattr(response, "id", None),
        }
        return content
