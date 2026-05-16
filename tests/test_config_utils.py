from pathlib import Path

from core import config_utils


def test_update_key_uses_valid_nested_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "api:\n  key: ''\n  base_url: https://example.test\n  model: old-model\nllm:\n  provider: openai_compatible\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_utils, "CONFIG_PATH", str(config_path))

    config_utils.update_key("api.model", "new-model")
    config_utils.update_key("llm.provider", "openai_compatible")

    assert config_utils.load_key("api.base_url") == "https://example.test"
    assert config_utils.load_key("api.model") == "new-model"
    assert config_utils.load_key("llm.provider") == "openai_compatible"
    assert not list(Path(tmp_path).glob(".config.*.tmp"))
