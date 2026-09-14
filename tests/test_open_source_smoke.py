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

    assert data["ninerouter"] == "YOUR_NINEROUTER_API_KEY"
    assert data["ninerouter_base_url"] == "https://your-openai-compatible-gateway.example/v1"
    assert "openai" not in data
    assert "claude" not in data
    assert SECRET_PATTERN.search(text) is None


def test_public_files_do_not_hardcode_private_gateway_domain():
    forbidden = ".".join(["9router", "yqbqnn", "com"])
    public_files = [
        ROOT / "config" / "default_config.yaml",
        ROOT / "config" / "api_keys.yaml.example",
        ROOT / "evaluator" / "adapters" / "openai.py",
        ROOT / "evaluator" / "adapters" / "claude.py",
    ]

    for path in public_files:
        assert forbidden not in path.read_text(encoding="utf-8")


def test_readme_uses_repaired_public_claims():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lower_text = text.lower()

    assert "GPT-5.5" in text
    assert "Claude-Opus-4.8" in text
    assert "480 samples" in text
    assert "460-sample complete-case subset" in text
    assert "verified anchors" not in lower_text
    assert "oracle baseline" not in lower_text
    assert "sample-wise oracle" not in lower_text
    assert "conditional robustness" not in lower_text
    assert "post-hoc robustness" not in lower_text
    assert "confirmatory robustness evidence" in lower_text
    assert "Claude-Opus-4.6" not in text


def test_public_generation_entrypoint_exists():
    assert (ROOT / "scripts" / "run_generate.py").is_file()
