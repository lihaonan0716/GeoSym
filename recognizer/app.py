"""
Standalone entry point for the Coordinate Recognizer Gradio UI.
For embedded use in Experiment, import from ui_builder instead.
"""

import os
import sys
from pathlib import Path

# Ensure recognizer is importable
_cr_dir = Path(__file__).resolve().parent
_parent_dir = _cr_dir.parent
if str(_cr_dir) not in sys.path:
    sys.path.insert(0, str(_cr_dir))
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import gradio as gr
from recognizer.ui_builder import build_recognizer_ui
from generator.config_loader import load_config


def create_gradio_interface():
    """Create a standalone Gradio interface for the coordinate recognizer."""
    with gr.Blocks(title="GeoSym Coordinate Recognition System") as demo:
        result = build_recognizer_ui(lang="zh")
    return demo


if __name__ == "__main__":
    _cfg = load_config()
    _server_cfg = _cfg.get("server", {}).get("recognizer_ui", {})
    _host = os.environ.get("GEOSYM_RECOGNIZER_UI_HOST", _server_cfg.get("host", "127.0.0.1"))
    _port = int(os.environ.get("GEOSYM_RECOGNIZER_UI_PORT", _server_cfg.get("port", 7860)))
    _display_host = "localhost" if _host == "0.0.0.0" else _host

    print(f"\n{'=' * 50}")
    print(f"  GeoSym Coordinate Recognizer UI")
    print(f"  http://{_display_host}:{_port}")
    print(f"{'=' * 50}\n")

    demo = create_gradio_interface()
    demo.launch(
        share=False,
        server_name=_host,
        server_port=_port,
        debug=os.environ.get("GEOSYM_DEBUG", "false").lower() == "true"
    )
