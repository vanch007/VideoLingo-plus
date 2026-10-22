import json
from types import SimpleNamespace
import pytest

from core import ask_gpt as ask_gpt_module
from core.llm_provider import LLMProviderConfig


class _StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_account_and_client_errors_are_not_retryable():
    assert ask_gpt_module.is_non_retryable_api_error(_StatusError(402))
    assert ask_gpt_module.is_non_retryable_api_error(_StatusError(401))
    assert not ask_gpt_module.is_non_retryable_api_error(_StatusError(429))
    assert not ask_gpt_module.is_non_retryable_api_error(_StatusError(500))


def test_cached_validator_runs_outside_log_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(ask_gpt_module, 'LOG_FOLDER', str(tmp_path))
    provider = LLMProviderConfig(name='omlx', model='test', base_url='http://localhost/v1', api_key='')
    monkeypatch.setattr(ask_gpt_module, 'get_llm_provider_config', lambda: provider)
    ask_gpt_module.save_log('test', 'cached', {'1': 'text'},
        request_fingerprint=ask_gpt_module.completion_fingerprint(provider, True))
    def validate(value):
        assert ask_gpt_module.LOCK.acquire(blocking=False), 'Nested semantic validation would deadlock'
        ask_gpt_module.LOCK.release()
        return {'status': 'success'}
    assert ask_gpt_module.ask_gpt('cached', valid_def=validate) == {'1': 'text'}


def test_invalid_cached_response_is_regenerated_with_validation_feedback(tmp_path, monkeypatch):
    monkeypatch.setattr(ask_gpt_module, 'LOG_FOLDER', str(tmp_path))
    provider = LLMProviderConfig(name='omlx', model='test', base_url='http://localhost/v1', api_key='')
    monkeypatch.setattr(ask_gpt_module, 'get_llm_provider_config', lambda: provider)
    ask_gpt_module.save_log('test', 'cached', {'1': 'wrong'},
        request_fingerprint=ask_gpt_module.completion_fingerprint(provider, True))
    calls = []
    def create(**kwargs):
        calls.append(kwargs['messages'])
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='{"1":"correct"}'))])
    monkeypatch.setattr(ask_gpt_module, 'create_chat_client', lambda _: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    validate = lambda d: {'status': 'success' if d['1'] == 'correct' else 'error', 'message': 'wrong meaning'}
    assert ask_gpt_module.ask_gpt('cached', valid_def=validate) == {'1': 'correct'}
    assert 'wrong meaning' in calls[0][-1]['content']
    assert ask_gpt_module.ask_gpt('cached', valid_def=validate) == {'1': 'correct'}
    assert len(calls) == 1


@pytest.mark.parametrize('native', [False, True])
def test_plain_text_validation_retry_contains_rejected_text_and_reason(tmp_path, monkeypatch, native):
    monkeypatch.setattr(ask_gpt_module, 'LOG_FOLDER', str(tmp_path))
    provider = LLMProviderConfig(name='omlx', model='test', base_url='http://localhost/v1', api_key='')
    monkeypatch.setattr(ask_gpt_module, 'get_llm_provider_config', lambda: provider)
    monkeypatch.setattr(ask_gpt_module, 'build_completion_args', lambda p, messages, response_json: {'messages': messages})
    calls = []
    def create(**kwargs):
        calls.append([dict(m) for m in kwargs['messages']])
        reply = '明天audition。' if len(calls) == 1 else '明天试镜。'
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=reply))])
    monkeypatch.setattr(ask_gpt_module, 'create_chat_client', lambda _: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    def native_retry(prompt, draft, error):
        return prompt + '\nRejected: ' + draft + '\nReason: ' + error
    result = ask_gpt_module.ask_gpt('Original source', response_json=False, max_retries=2, retry_interval=0,
        validation_retry_prompt=native_retry if native else None,
        valid_def=lambda text: {'status': 'success' if text == '明天试镜。' else 'error', 'message': 'Translate audition'})
    assert result == '明天试镜。'
    assert len(calls) == 2
    correction = str(calls[1])
    assert '明天audition。' in correction and 'Translate audition' in correction
    assert 'JSON' not in correction
    assert [m['role'] for m in calls[1]] == (['user'] if native else ['user', 'assistant', 'user'])
    assert json.loads((tmp_path / 'error.json').read_text())[0]['response'] == '明天audition。'


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


def test_validation_retry_includes_rejected_draft_before_corrective_request(tmp_path, monkeypatch):
    monkeypatch.setattr(ask_gpt_module, "LOG_FOLDER", str(tmp_path))
    provider = LLMProviderConfig(name="local", model="test", base_url="http://localhost/v1", api_key="")
    monkeypatch.setattr(ask_gpt_module, "get_llm_provider_config", lambda: provider)
    monkeypatch.setattr(ask_gpt_module, "build_completion_args", lambda p, messages, response_json: {"messages": messages})
    calls = []
    def create(**kwargs):
        calls.append([dict(m) for m in kwargs["messages"]])
        reply = '{"1":"Too many spoken words"}' if len(calls) == 1 else '{"1":"Fine"}'
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=reply))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(ask_gpt_module, "create_chat_client", lambda p: client)
    result = ask_gpt_module.ask_gpt("Original input", max_retries=2, retry_interval=0,
        valid_def=lambda d: {"status": "success" if d['1'] == 'Fine' else "error", "message": "Use one word"})
    assert result == {"1": "Fine"}
    assert [m["role"] for m in calls[1]] == ["user", "assistant", "user"]
    assert 'Too many spoken words' in calls[1][1]['content']


def test_native_translation_retry_keeps_one_source_message_and_has_a_separate_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ask_gpt_module, "LOG_FOLDER", str(tmp_path))
    provider = LLMProviderConfig(name="omlx", model="test", base_url="http://localhost/v1", api_key="")
    monkeypatch.setattr(ask_gpt_module, "get_llm_provider_config", lambda: provider)
    monkeypatch.setattr(ask_gpt_module, "build_completion_args", lambda p, messages, response_json: {"messages":messages})
    original = '【源文】\n{"1":"你好"}\n【约束要求】\n短句。'
    ask_gpt_module.save_log(provider.model, original, {"1":"Old cached answer"},
        request_fingerprint=ask_gpt_module.completion_fingerprint(provider, True))
    calls = []
    def create(**kwargs):
        calls.append([dict(m) for m in kwargs["messages"]])
        text = '{"1":"Long answer"}' if len(calls)==1 else '{"1":"Hi"}'
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",message=SimpleNamespace(content=text))])
    monkeypatch.setattr(ask_gpt_module, "create_chat_client", lambda _:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    def native_retry(prompt, rejected, error):
        return prompt + '\n失败候选数据：' + json.dumps({"draft":rejected,"reason":error}) + '\n只输出译文JSON。'
    result = ask_gpt_module.ask_gpt(original,max_retries=2,retry_interval=0,
        validation_retry_prompt=native_retry,
        valid_def=lambda d:{"status":"success" if d['1']=='Hi' else "error","message":"Choose a brief complete reply"})
    assert result == {"1":"Hi"}
    assert len(calls)==2
    assert [m["role"] for m in calls[1]] == ["user"]
    assert calls[1][0]["content"].count("【源文】")==1
    assert "Long answer" in calls[1][0]["content"]
