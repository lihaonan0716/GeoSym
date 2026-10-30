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

    # The README describes the executed method and the reported numbers.
    assert "GPT-5.5" in text
    assert "GLM-4.6V" in text
    assert "extract_clean_cartesian_grid" in text
    assert "benchmark/roster_240.json" in text
    assert "+10.66" in text
    # Null results are reported alongside the headline (layering, PlotQA accuracy).
    assert "+0.12" in text
    assert "p = 0.388" in text
    # Retired framing and descriptions of code paths the method never runs.
    for stale in ("460-sample", "hough", "stage 2.5", "oracle baseline",
                  "sample-wise oracle", "verified anchors",
                  "conditional robustness", "post-hoc robustness"):
        assert stale not in lower_text, stale
    # The weighting is unequal; the README may only mention equal weighting to deny it.
    assert "**not** equally weighted" in text
    assert "Claude-Opus-4.6" not in text


def test_public_generation_entrypoint_exists():
    assert (ROOT / "scripts" / "run_generate.py").is_file()
