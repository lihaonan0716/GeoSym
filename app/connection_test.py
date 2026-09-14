"""Backend logic for the Model Connection Test tab.

Tests each LLM provider's API connectivity by sending a minimal
request (tiny white image + short prompt) and measuring latency.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import cv2
import numpy as np

from app.helpers import (
    MODEL_REGISTRY, PROVIDER_MAP, _get_saved_key, _create_llm_adapter,
)
from i18n import t


_TEST_PROMPT = "Reply OK if you can see this image."


def _make_test_image_b64() -> str:
    """Create a tiny 64x64 white PNG and return its base64 encoding."""
    import base64
    img = np.ones((64, 64, 3), dtype=np.uint8) * 255
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _has_key(display_name: str) -> bool:
    """Check whether an API key is configured for a model."""
    return bool(_get_saved_key(display_name))


def build_provider_overview(lang: str = "中文") -> str:
    """Build a Markdown overview of all configured providers and models."""
    is_zh = lang == "中文"
    header = (
        "| 供应商 | 模型 | API Key |\n|---|---|---|"
        if is_zh else
        "| Provider | Models | API Key |\n|---|---|---|"
    )
    lines = [header]
    for provider, models in PROVIDER_MAP.items():
        if not models:
            continue
        key_ok = _has_key(models[0])
        key_icon = "✅" if key_ok else "❌"
        model_list = ", ".join(f"`{m}`" for m in models)
        lines.append(f"| **{provider}** | {model_list} | {key_icon} |")
    return "\n".join(lines)


def check_model_connection(display_name: str) -> Dict[str, Any]:
    """Test a single model's API connectivity.

    Returns dict with keys: provider, model, status, latency_ms, error.
    """
    provider = ""
    for pname, models in PROVIDER_MAP.items():
        if display_name in models:
            provider = pname
            break

    result: Dict[str, Any] = {
        "provider": provider,
        "model": display_name,
        "status": "fail",
        "latency_ms": 0,
        "error": "",
    }

    entry = MODEL_REGISTRY.get(display_name)
    if entry is None:
        result["error"] = f"Model '{display_name}' not found in registry"
        return result

    if not _has_key(display_name):
        result["error"] = "API key not configured"
        return result

    try:
        key = _get_saved_key(display_name)
        adapter = _create_llm_adapter(display_name, key)
        if adapter is None:
            result["error"] = "Failed to create adapter"
            return result

        img_b64 = _make_test_image_b64()
        t0 = time.perf_counter()
        response = adapter._call_api(img_b64, _TEST_PROMPT)
        elapsed = (time.perf_counter() - t0) * 1000

        result["latency_ms"] = round(elapsed)
        if response and len(response.strip()) > 0:
            result["status"] = "pass"
        else:
            result["error"] = "Empty response"
    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def run_single_connection_test(model_name: str, lang: str = "中文") -> str:
    """Test a single model and return a formatted Markdown result.

    Called directly from the Gradio button click.
    """
    is_zh = lang == "中文"
    if not model_name:
        return "⚠️ " + ("请先选择一个模型" if is_zh else "Please select a model first")

    r = check_model_connection(model_name)

    if r["status"] == "pass":
        status_line = "✅ " + ("连接成功" if is_zh else "Connection successful")
        latency = f"{r['latency_ms']} ms"
        parts = [
            f"### {r['provider']} / `{r['model']}`",
            "",
            f"**{'状态' if is_zh else 'Status'}**: {status_line}",
            f"**{'延迟' if is_zh else 'Latency'}**: {latency}",
        ]
    else:
        status_line = "❌ " + ("连接失败" if is_zh else "Connection failed")
        parts = [
            f"### {r['provider']} / `{r['model']}`",
            "",
            f"**{'状态' if is_zh else 'Status'}**: {status_line}",
            f"**{'错误信息' if is_zh else 'Error'}**: {r['error']}",
        ]

    return "\n".join(parts)
