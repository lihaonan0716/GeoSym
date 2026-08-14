import re
from pathlib import Path

import yaml

from generator.config_loader import load_config


ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERN = re.compile(
    r"(AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9_-]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|"
    r"-----BEGIN (?:RSA|OPENSSH|PRIVATE) KEY-----)"
)


def test_default_config_loads_public_settings():
    config = load_config()

    assert config["server"]["app"]["host"] == "127.0.0.1"
    assert config["server"]["recognizer_ui"]["host"] == "127.0.0.1"
    assert "models" in config


def test_api_key_template_uses_safe_placeholders():
    template = ROOT / "config" / "api_keys.yaml.example"
    text = template.read_text(encoding="utf-8")
    data = yaml.safe_load(text)

    assert data["openai"] == "YOUR_OPENAI_API_KEY"
    assert data["claude"] == "YOUR_ANTHROPIC_API_KEY"
    assert SECRET_PATTERN.search(text) is None


def test_public_generation_entrypoint_exists():
    assert (ROOT / "scripts" / "run_generate.py").is_file()
