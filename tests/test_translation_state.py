import json
from pathlib import Path

from core.llm_provider import LLMProviderConfig
from core import translation_state


def _fake_provider(model: str = "deepseek-ai/DeepSeek-V3.2") -> LLMProviderConfig:
    return LLMProviderConfig(
        name="openai_compatible",
        model=model,
        base_url="https://api.siliconflow.cn/v1",
        api_key="secret",
        supports_json_object=False,
    )


def test_translation_status_flags_old_log_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(translation_state, "get_llm_provider_config", lambda: _fake_provider())

    Path("output/log").mkdir(parents=True)
    Path("output/gpt_log").mkdir(parents=True)
    Path("output/log/translation_results.xlsx").write_text("placeholder", encoding="utf-8")
    Path("output/gpt_log/translate_faithfulness.json").write_text(
        json.dumps([{"model": "old-omlx-model", "prompt": "p", "response": {"ok": True}}]),
        encoding="utf-8",
    )

    status = translation_state.build_translation_status()

    assert status["stale_stage_count"] >= 1
    assert "log_model_mismatch" in status["stages"]["translate"]["reasons"]
    assert "artifact_exists_without_manifest" in status["stages"]["translate"]["warnings"]


def test_record_llm_stage_writes_current_provider_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(translation_state, "get_llm_provider_config", lambda: _fake_provider())

    Path("output/log").mkdir(parents=True)
    Path("output/gpt_log").mkdir(parents=True)
    Path("output/log/translation_results.xlsx").write_text("placeholder", encoding="utf-8")
    Path("output/gpt_log/translate_faithfulness.json").write_text(
        json.dumps([{"model": "deepseek-ai/DeepSeek-V3.2", "prompt": "p", "response": {"ok": True}}]),
        encoding="utf-8",
    )

    translation_state.record_llm_stage("translate")
    status = translation_state.build_translation_status()

    assert status["manifest_exists"]
    assert status["stages"]["translate"]["manifest_provider"]["model"] == "deepseek-ai/DeepSeek-V3.2"
    assert "manifest_provider_mismatch" not in status["stages"]["translate"]["reasons"]
    assert status["stages"]["translate"]["stale"] is False


def test_archive_plan_is_dry_run_until_applied(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("output/log").mkdir(parents=True)
    Path("output/log/translation_results.xlsx").write_text("placeholder", encoding="utf-8")

    plan = translation_state.build_translation_archive_plan(reason="test")

    assert plan["item_count"] == 1
    assert Path("output/log/translation_results.xlsx").exists()


def test_adopt_current_writes_manifest_when_logs_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(translation_state, "get_llm_provider_config", lambda: _fake_provider())

    Path("output/log").mkdir(parents=True)
    Path("output/gpt_log").mkdir(parents=True)
    Path("output/log/translation_results.xlsx").write_text("placeholder", encoding="utf-8")
    Path("output/gpt_log/translate_faithfulness.json").write_text(
        json.dumps([{"model": "deepseek-ai/DeepSeek-V3.2", "prompt": "p", "response": {"ok": True}}]),
        encoding="utf-8",
    )

    payload = translation_state.adopt_current_translation_artifacts(dry_run=False)
    status = translation_state.build_translation_status()

    assert payload["ok"] is True
    assert "translate" in payload["adopted_stages"]
    assert status["stages"]["translate"]["warning"] is False


def test_guard_blocks_stale_translation_for_dependent_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(translation_state, "get_llm_provider_config", lambda: _fake_provider())

    Path("output/log").mkdir(parents=True)
    Path("output/gpt_log").mkdir(parents=True)
    Path("output/log/translation_results.xlsx").write_text("placeholder", encoding="utf-8")
    Path("output/gpt_log/translate_faithfulness.json").write_text(
        json.dumps([{"model": "old-omlx-model", "prompt": "p", "response": {"ok": True}}]),
        encoding="utf-8",
    )

    guard = translation_state.guard_translation_artifacts_for_steps(["translate", "gen_audio_task"])

    assert guard["ok"] is False
    assert guard["action"] == "blocked"
    assert "translate" in guard["stale_stages"]


def test_guard_ignores_non_translation_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(translation_state, "get_llm_provider_config", lambda: _fake_provider())

    guard = translation_state.guard_translation_artifacts_for_steps(["download", "import_video"])

    assert guard["ok"] is True
    assert guard["action"] == "not_applicable"
