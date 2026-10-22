from pathlib import Path
import os

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


def test_unchanged_yaml_is_parsed_once_and_results_are_private(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  names: [one, two]\n")
    monkeypatch.setattr(config_utils, "CONFIG_PATH", str(config_path))
    original = config_utils.yaml.load
    calls = []

    def counted_load(stream):
        calls.append(1)
        return original(stream)

    monkeypatch.setattr(config_utils.yaml, "load", counted_load)
    config_utils.load_key("model.names").append("caller-only")
    config_utils.load_config()["model"]["names"].clear()
    assert config_utils.load_key("model.names") == ["one", "two"]
    assert len(calls) == 1


def test_external_same_size_same_mtime_replacement_invalidates_cache(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: old\n")
    monkeypatch.setattr(config_utils, "CONFIG_PATH", str(config_path))
    assert config_utils.load_key("name") == "old"
    before = config_path.stat()
    replacement = tmp_path / "replacement.yaml"
    replacement.write_text("name: new\n")
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(config_path)
    assert config_utils.load_key("name") == "new"
    config_utils.update_key("name", "saved")
    assert config_utils.load_key("name") == "saved"


def test_cache_tracks_config_path_and_live_environment_overrides(tmp_path, monkeypatch):
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text("api:\n  model: first\n")
    second.write_text("api:\n  model: second\n")
    monkeypatch.delenv("VIDEOLINGO_API_MODEL", raising=False)
    monkeypatch.setattr(config_utils, "CONFIG_PATH", str(first))
    assert config_utils.load_key("api.model") == "first"
    monkeypatch.setattr(config_utils, "CONFIG_PATH", str(second))
    assert config_utils.load_key("api.model") == "second"
    monkeypatch.setenv("VIDEOLINGO_API_MODEL", "live-override")
    assert config_utils.load_key("api.model") == "live-override"
    monkeypatch.delenv("VIDEOLINGO_API_MODEL")
    assert config_utils.load_key("api.model") == "second"
