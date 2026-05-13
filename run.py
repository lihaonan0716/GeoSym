#!/usr/bin/env python3
"""Launch the GeoSym Gradio UI.``   

Usage:
    python run.py
"""

import os

import gradio as gr

from app import create_app
from app.helpers import _config, _reports_dir, _images_dir, custom_css

if __name__ == "__main__":
    _server_cfg = _config.get("server", {}).get("app", {})
    _host = os.environ.get("GEOSYM_APP_HOST", _server_cfg.get("host", "127.0.0.1"))
    _port = int(os.environ.get("GEOSYM_APP_PORT", _server_cfg.get("port", 7870)))
    _display_host = "localhost" if _host == "0.0.0.0" else _host

    print(f"\n{'=' * 50}")
    print(f"            GeoSym UI")
    print(f"            http://{_display_host}:{_port}")
    print(f"{'=' * 50}\n")

    app = create_app()
    app.launch(
        server_name=_host,
        server_port=_port,
        share=False,
        quiet=True,
        css=custom_css,
        theme=gr.themes.Base(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        allowed_paths=[str(_reports_dir), str(_images_dir)],
    )
 