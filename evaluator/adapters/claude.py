"""
Anthropic Claude adapter for intersection recognition.
Supports TWO authentication modes:

  1. Direct Anthropic API Key
     -> Uses anthropic.Anthropic client

  2. Google Cloud Vertex AI Service Account JSON
     -> Place JSON file at: config/claude_credentials.json
     -> Uses anthropic.AnthropicVertex client (auto-detected)

Usage:
    # Mode 1: Direct API key
    adapter = ClaudeAdapter(api_key="YOUR_ANTHROPIC_API_KEY")

    # Mode 2: Vertex AI (auto-detects config/claude_credentials.json)
    adapter = ClaudeAdapter(api_key="")
"""

import json
import os
from pathlib import Path

import anthropic

from .base import BaseLLMAdapter

# Path to Vertex AI service account JSON (relative to Experiment/)
_CREDENTIALS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "claude_credentials.json"

# Vertex AI model name mapping (Vertex AI uses different model IDs)
_VERTEX_MODEL_MAP = {
    "claude-opus-4-6": "claude-opus-4",         # Vertex may lag behind direct API
    "claude-sonnet-4-5": "claude-sonnet-4-5",
    "claude-sonnet-4-20250514": "claude-sonnet-4",
}


class ClaudeAdapter(BaseLLMAdapter):
    """
    Anthropic Claude Vision adapter.
    Auto-detects Vertex AI credentials if config/claude_credentials.json exists.

    Supported models:
        - claude-opus-4-6          : Claude Opus 4.6 (latest, direct API)
        - claude-opus-4            : Claude Opus 4 (Vertex AI)
        - claude-sonnet-4-5        : Claude Sonnet 4.5
    """

    def __init__(self, api_key: str = "", model: str = "claude-opus-4-6"):
        super().__init__(model_name=model)
        self.api_key = api_key
        self.model = model
        self._client = None
        self._use_vertex = False

    def _get_client(self):
        if self._client is not None:
            return self._client

        # Mode 1: Direct Anthropic API key
        if self.api_key and self.api_key.strip().startswith("sk-ant-"):
            self._client = anthropic.Anthropic(api_key=self.api_key.strip(), timeout=180.0)
            self._use_vertex = False
            return self._client

        # Mode 2: Vertex AI via service account JSON
        if _CREDENTIALS_PATH.exists():
            try:
                creds_data = json.loads(_CREDENTIALS_PATH.read_text(encoding="utf-8"))
                project_id = creds_data.get("project_id", "")
                if not project_id:
                    raise ValueError("project_id not found in claude_credentials.json")

                # Set Google credentials environment variable
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_CREDENTIALS_PATH)

                from anthropic import AnthropicVertex
                self._client = AnthropicVertex(
                    project_id=project_id,
                    region="us-east5",
                    timeout=180.0,
                )
                self._use_vertex = True
                return self._client

            except Exception as e:
                raise RuntimeError(
                    f"Failed to init Vertex AI client from {_CREDENTIALS_PATH}: {e}"
                ) from e

        # Neither mode available
        raise ValueError(
            "Claude API key not provided and no Vertex AI credentials found.\n"
            "Option 1: Enter your Anthropic API key in the UI.\n"
            f"Option 2: Place your GCP service account JSON at:\n  {_CREDENTIALS_PATH}"
        )

    def _call_api(self, image_base64: str, prompt: str) -> str:
        """
        Call Claude Messages API with image and prompt.
        Works with both direct API and Vertex AI.
        """
        client = self._get_client()

        # On Vertex AI, map model name if needed
        model = self.model
        if self._use_vertex and model in _VERTEX_MODEL_MAP:
            model = _VERTEX_MODEL_MAP[model]

        message = client.messages.create(
            model=model,
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
