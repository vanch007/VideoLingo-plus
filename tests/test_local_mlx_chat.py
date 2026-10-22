import io
import json
from types import SimpleNamespace

import pytest

from core.llm_provider import LLMProviderConfig,create_chat_client
from core.local_mlx_chat import LocalMLXChatClient
from core.duration_candidate_audit import validate_verdict


def config(tmp_path):
    return LLMProviderConfig(name="local_editor",model="test",base_url="local://mlx",api_key="",
        provider_type="mlx_local",model_path=str(tmp_path/"not-installed"),runtime_python="/missing/python")


def test_local_editor_is_lazy_and_cannot_download_missing_weights(tmp_path):
    client=create_chat_client(config(tmp_path))
    assert client.process is None
    with pytest.raises(FileNotFoundError,match="not installed"):
        client.chat.completions.create(messages=[{"role":"user","content":"test"}])


def test_local_completion_preserves_truncation_and_usage_for_existing_api_gates(tmp_path,monkeypatch):
    client=LocalMLXChatClient(config(tmp_path))
    pipe=io.StringIO()
    client.process=SimpleNamespace(poll=lambda:0,stdin=pipe)
    monkeypatch.setattr(client,"_start",lambda timeout:setattr(client,"process",SimpleNamespace(poll=lambda:None,stdin=pipe)))
    monkeypatch.setattr(client,"_receive",lambda timeout:{"content":"{\"1\":\"cut","finish_reason":"length","usage":{"completion_tokens":10}})
    result=client.create(messages=[{"role":"user","content":"data"}],max_tokens=10)
    assert result.choices[0].finish_reason=="length"
    assert result.usage.model_dump()["completion_tokens"]==10
    assert json.loads(pipe.getvalue())["messages"][0]["content"]=="data"
    client.process=None


def test_source_semantic_gate_rejects_ambiguous_or_contradictory_verdicts():
    assert validate_verdict({"equivalent":True,"issues":[]})["status"]=="success"
    assert validate_verdict({"equivalent":False,"issues":["Lost negation"]})["status"]=="success"
    assert validate_verdict({"equivalent":"yes","issues":[]})["status"]=="error"
    assert validate_verdict({"equivalent":True,"issues":["Changed payer"]})["status"]=="error"
