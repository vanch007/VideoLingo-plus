import json

from core import ask_gpt as ask_gpt_module


class _StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_account_and_client_errors_are_not_retryable():
    assert ask_gpt_module.is_non_retryable_api_error(_StatusError(402))
    assert ask_gpt_module.is_non_retryable_api_error(_StatusError(401))
    assert not ask_gpt_module.is_non_retryable_api_error(_StatusError(429))
    assert not ask_gpt_module.is_non_retryable_api_error(_StatusError(500))


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
