import json

from core import ask_gpt as ask_gpt_module


def test_ask_gpt_history_is_model_scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(ask_gpt_module, "LOG_FOLDER", str(tmp_path))
    log_path = tmp_path / "translate.json"
    log_path.write_text(
        json.dumps(
            [
                {
                    "model": "old-omlx-model",
                    "prompt": "translate this",
                    "response": {"text": "old"},
                },
                {
                    "model": "deepseek-ai/DeepSeek-V3.2",
                    "prompt": "translate this",
                    "response": {"text": "new"},
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert ask_gpt_module.check_ask_gpt_history("translate this", "deepseek-ai/DeepSeek-V3.2", "translate") == {"text": "new"}
    assert ask_gpt_module.check_ask_gpt_history("translate this", "different-model", "translate") is False
