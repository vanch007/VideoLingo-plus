import httpx
import openai
import pytest

from core import llm_key_pool
from core.llm_provider import LLMProviderConfig, completion_fingerprint


def _status_error(status):
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(status, request=request)
    cls = {429: openai.RateLimitError, 400: openai.BadRequestError, 401: openai.AuthenticationError}.get(
        status, openai.InternalServerError)
    return cls(f"status {status}", response=response, body=None)


class FakeOpenAI:
    script = {}
    calls = []

    def __init__(self, api_key, base_url, timeout, max_retries):
        self.api_key = api_key
        outer = self

        class Completions:
            def create(self, **kwargs):
                FakeOpenAI.calls.append(outer.api_key)
                outcome = FakeOpenAI.script.get(outer.api_key, "ok")
                if isinstance(outcome, Exception):
                    raise outcome
                return f"answer from {outer.api_key}"

        self.chat = type("Chat", (), {"completions": Completions()})()


@pytest.fixture
def pool(tmp_path, monkeypatch):
    llm_key_pool.reset_state()
    monkeypatch.setattr(llm_key_pool, "OpenAI", FakeOpenAI)
    FakeOpenAI.script, FakeOpenAI.calls = {}, []
    keys = tmp_path / "keys.txt"
    keys.write_text("# comment\nk1\nk2  # trailing comment\n\nk3\nk1\n", encoding="utf-8")
    yield lambda: llm_key_pool.KeyPoolClient(str(keys), "https://example.test/v1", 10)
    llm_key_pool.reset_state()


def test_reads_keys_ignoring_comments_blanks_and_duplicates(pool):
    assert pool().size == 3


def test_requests_rotate_round_robin(pool):
    client = pool()
    answers = [client.chat.completions.create(model="m", messages=[]) for _ in range(4)]
    assert answers == ["answer from k1", "answer from k2", "answer from k3", "answer from k1"]


def test_rate_limited_key_moves_request_to_next_key_and_cools_down(pool):
    FakeOpenAI.script = {"k1": _status_error(429)}
    client = pool()
    assert client.chat.completions.create(model="m", messages=[]) == "answer from k2"
    # k1 is cooling down: later requests skip it while others are ready.
    for _ in range(3):
        client.chat.completions.create(model="m", messages=[])
    assert FakeOpenAI.calls.count("k1") == 1


def test_server_error_and_rejected_key_fail_over(pool):
    FakeOpenAI.script = {"k1": _status_error(503), "k2": _status_error(401)}
    assert pool().chat.completions.create(model="m", messages=[]) == "answer from k3"


def test_bad_request_is_not_retried_on_other_keys(pool):
    FakeOpenAI.script = {"k1": _status_error(400)}
    with pytest.raises(openai.BadRequestError):
        pool().chat.completions.create(model="m", messages=[])
    assert FakeOpenAI.calls == ["k1"]


def test_all_keys_failing_raises_last_error(pool):
    FakeOpenAI.script = {k: _status_error(429) for k in ("k1", "k2", "k3")}
    with pytest.raises(openai.RateLimitError):
        pool().chat.completions.create(model="m", messages=[])
    assert FakeOpenAI.calls == ["k1", "k2", "k3"]


def test_missing_or_empty_key_file_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        llm_key_pool.read_keys(str(tmp_path / "absent.txt"))
    empty = tmp_path / "empty.txt"
    empty.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no keys"):
        llm_key_pool.read_keys(str(empty))


def test_key_pool_path_does_not_change_cached_response_fingerprint():
    base = dict(name="g", model="gemini", base_url="https://x/v1", api_key="")
    assert completion_fingerprint(LLMProviderConfig(**base, key_pool="/a"), True) == \
        completion_fingerprint(LLMProviderConfig(**base, key_pool="/b"), True)


def test_reads_opencodex_pool_skipping_disabled_masked_and_duplicate_keys(tmp_path, monkeypatch):
    import json
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"providers": {"google": {
        "apiKey": "k1",
        "apiKeyPool": [{"id": "a", "key": "k1"}, {"id": "b", "key": "k2", "disabled": True},
                       {"id": "c", "key": "AIza****abcd"}, {"id": "d", "key": "k3"}]}}}), encoding="utf-8")
    monkeypatch.setattr(llm_key_pool, "OPENCODEX_CONFIG", str(cfg))
    assert llm_key_pool.read_keys("opencodex:google") == ["k1", "k3"]
    with pytest.raises(ValueError, match="no provider"):
        llm_key_pool.read_keys("opencodex:absent")
