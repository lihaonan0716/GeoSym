import copy
import logging
import random
import time

import requests
import json
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_API_KEYS_PATH = _CONFIG_DIR / "api_keys.yaml"

_MAX_RETRIES = 3
_BASE_DELAY = 3.0
_MAX_DELAY = 30.0
_TIMEOUT = 60

MATHPIX_ENDPOINT = "https://api.mathpix.com/v3/text"

# Request options for every M0 Stage 0 OCR call. Exposed as a module constant,
# with `build_request_body` as the single place the body is assembled, so a
# diagnostic that needs to record "what do we actually send" can import it
# instead of keeping a transcribed second copy that drifts when this is edited.
MATHPIX_REQUEST_OPTIONS = {
    "formats": ["text", "data", "html"],
    "data_options": {
        "include_asciimath": True,
        "include_latex": True,
    },
    "include_line_data": True,
    "include_smiles": False,
    "include_inchi": False,
    "include_diagram_text": True,
    "include_geometry_data": False,
}


def build_request_body(src: str) -> dict:
    """Return the exact JSON body posted to Mathpix for `src`.

    Deep-copied so a caller inspecting or mutating the result cannot alter the
    options every subsequent request sends.
    """
    return {"src": src, **copy.deepcopy(MATHPIX_REQUEST_OPTIONS)}


def _load_mathpix_credentials() -> tuple:
    """Load Mathpix app_id and app_key from config/api_keys.yaml."""
    if not _API_KEYS_PATH.exists():
        raise FileNotFoundError(
            f"[Mathpix] api_keys.yaml not found at {_API_KEYS_PATH}\n"
            f"  Please copy config/api_keys.yaml.example to config/api_keys.yaml\n"
            f"  and fill in your Mathpix credentials (mathpix_app_id, mathpix_app_key)."
        )
    with open(_API_KEYS_PATH, "r", encoding="utf-8") as f:
        keys = yaml.safe_load(f) or {}
    app_id = keys.get("mathpix_app_id", "")
    app_key = keys.get("mathpix_app_key", "")
    if not app_id or not app_key:
        raise ValueError(
            f"[Mathpix] mathpix_app_id or mathpix_app_key is missing in {_API_KEYS_PATH}\n"
            f"  Please add your Mathpix credentials. Get them at https://accounts.mathpix.com"
        )
    return app_id, app_key


def _is_retriable(status_code: int) -> bool:
    return status_code in (429, 500, 502, 503, 504)


class MathpixImageOCRTool:
    def ocr_image(self, image_path: str):
        """
        Invoke Mathpix Image OCR tool to process images and extract text
        and mathematical expressions.  Retries transient failures with
        exponential backoff.
        """
        mathpix_app_id, mathpix_app_key = _load_mathpix_credentials()

        src = image_path
        if not src:
            return "请提供图像URL或base64编码的图像数据"

        base_url = MATHPIX_ENDPOINT
        headers = {
            'app_id': mathpix_app_id,
            'app_key': mathpix_app_key,
            'Content-Type': 'application/json'
        }

        request_data = build_request_body(src)

        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = requests.post(
                    base_url, headers=headers,
                    json=request_data, timeout=_TIMEOUT,
                )

                if response.status_code == 200:
                    return response.json()

                if attempt < _MAX_RETRIES and _is_retriable(response.status_code):
                    delay = min(_BASE_DELAY * (2 ** (attempt - 1)), _MAX_DELAY)
                    delay *= 0.5 + random.random()
                    logger.warning(
                        "[Mathpix] HTTP %d on attempt %d/%d, retry in %.1fs",
                        response.status_code, attempt, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue

                response.raise_for_status()

            except requests.exceptions.RequestException as e:
                last_err = e
                is_transient = isinstance(
                    e, (requests.exceptions.Timeout,
                        requests.exceptions.ConnectionError),
                )
                if attempt < _MAX_RETRIES and is_transient:
                    delay = min(_BASE_DELAY * (2 ** (attempt - 1)), _MAX_DELAY)
                    delay *= 0.5 + random.random()
                    logger.warning(
                        "[Mathpix] %s on attempt %d/%d, retry in %.1fs",
                        type(e).__name__, attempt, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                return f"请求Mathpix API时发生错误: {e}"

            except Exception as e:
                return f"处理请求时发生错误: {e}"

        return f"请求Mathpix API时发生错误 (重试{_MAX_RETRIES}次后): {last_err}"
