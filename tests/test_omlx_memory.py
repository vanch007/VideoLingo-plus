from contextlib import contextmanager
import copy
import json
import threading
from types import SimpleNamespace

import pytest
import requests

from core.llm_provider import LLMProviderConfig
from core.providers import omlx_memory as memory

_REAL_SELECTED_PROVIDERS = memory._selected_providers


def provider(model="Index-Translate", **kwargs):
    return LLMProviderConfig(name=kwargs.pop("name", "omlx"), model=model,
                             base_url=kwargs.pop("base_url", "http://127.0.0.1:8000/v1"),
                             api_key=kwargs.pop("api_key", ""), **kwargs)


class Server:
    def __init__(self):
        self.headers = {}
        self.trust_env = True
        self.calls = []
        self.models = [{"id": "Index-Translate", "model_alias": "translator",
                        "loaded": True, "is_loading": False, "pinned": False}]
        self.activity = {"id": "Index-Translate", "active_requests": 0,
                         "waiting_requests": 0, "is_loading": False, "idle_seconds": 4.0}
        self.paths = {path: {method: {}} for path, method in memory._API_PATHS.items()}
        self.post_status = "ok"
        self.failure = None

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        assert kwargs["allow_redirects"] is False
        assert self.trust_env is False
        if self.failure:
            raise self.failure
        if url.endswith("/openapi.json"):
            payload = {"paths": self.paths}
        elif url.endswith("/models/status"):
            payload = {"models": copy.deepcopy(self.models),
                       "current_model_memory": sum(30_000_000_000 for m in self.models if m["loaded"])}
        elif url.endswith("/api/activity"):
            payload = {"active_models": {"models": [self.activity] if self.activity else []}}
        elif method == "POST" and url.endswith("/unload"):
            assert self.models[0]["id"] in url or "%23" in url
            if self.post_status == "ok":
                self.models[0]["loaded"] = False
            payload = {"status": self.post_status}
        else:
            raise AssertionError((method, url))
        return SimpleNamespace(status_code=202 if payload.get("status") == "unloading" else 200,
                               json=lambda: payload, raise_for_status=lambda: None)


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    s = Server()
    monkeypatch.setattr(memory.requests, "Session", lambda: s)
    monkeypatch.setattr(memory, "_selected_providers", lambda: iter([provider()]))
    monkeypatch.setattr(memory, "load_key", lambda key, default=None: default)
    return s


def test_releases_only_selected_idle_model_and_confirms_memory(server, tmp_path):
    server.models.append({"id": "unrelated", "loaded": True, "is_loading": False})
    event = memory.release_omlx_for_tts("indextts2")
    item = event["models"][0]
    assert item["status"] == "unloaded"
    assert item["model_memory_before_bytes"] == 60_000_000_000
    assert item["model_memory_after_bytes"] == 30_000_000_000
    assert server.models[1]["loaded"] is True
    assert len([c for c in server.calls if c[0] == "POST"]) == 1
    saved = json.loads((tmp_path / "output/log/omlx_memory_events.jsonl").read_text())
    assert saved == event


@pytest.mark.parametrize("change", [
    {"active_requests": 1}, {"waiting_requests": 1}, {"idle_seconds": 0},
    {"idle_seconds": None}, {"idle_seconds": True}, {"is_loading": True},
    {"active_requests": None}, {"waiting_requests": None},
    {"active_requests": False}, {"idle_seconds": float("nan")},
])
def test_busy_or_unverified_activity_never_calls_unload(server, change):
    server.activity.update(change)
    assert memory.release_omlx_for_tts("indextts2")["models"][0]["status"] == "busy_or_unknown"
    assert not any(c[0] == "POST" for c in server.calls)


def test_missing_activity_is_not_idle(server):
    server.activity = None
    assert memory.release_omlx_for_tts("indextts2")["models"][0]["status"] == "busy_or_unknown"
    assert not any(c[0] == "POST" for c in server.calls)


@pytest.mark.parametrize("change", [{"pinned": True}, {"distributed": True}, {"is_loading": True}])
def test_does_not_override_protected_models(server, change):
    server.models[0].update(change)
    assert memory.release_omlx_for_tts("indextts2")["models"][0]["status"] == "protected_or_loading"
    assert not any(c[0] == "POST" for c in server.calls)


def test_already_unloaded_model_is_idempotent(server):
    server.models[0]["loaded"] = False
    assert memory.release_omlx_for_tts("indextts2")["models"][0]["status"] == "already_unloaded"
    assert not any(c[0] == "POST" for c in server.calls)


@pytest.mark.parametrize("cfg", [
    provider(base_url="https://example.com/v1"), provider(name="other_local"),
    provider(base_url="http://localhost:8000/v1?key=secret"),
    provider(base_url="http://user:secret@localhost:8000/v1"),
])
def test_remote_or_unrecognized_routes_are_untouched(server, monkeypatch, cfg):
    monkeypatch.setattr(memory, "_selected_providers", lambda: iter([cfg]))
    assert memory.release_omlx_for_tts("indextts2")["models"] == []
    assert server.calls == []


def test_alias_resolves_exactly_and_multiple_routes_unload_once(server, monkeypatch):
    monkeypatch.setattr(memory, "_selected_providers",
                        lambda: iter([provider("translator"), provider(name="omlx_duration")]))
    assert len(memory.release_omlx_for_tts("indextts2")["models"]) == 1
    assert len([c for c in server.calls if c[0] == "POST"]) == 1


def test_ambiguous_alias_never_unloads(server, monkeypatch):
    server.models.append({**server.models[0], "id": "different-model"})
    monkeypatch.setattr(memory, "_selected_providers", lambda: iter([provider("translator")]))
    assert memory.release_omlx_for_tts("indextts2")["models"][0]["status"] == "model_not_resolved"
    assert not any(c[0] == "POST" for c in server.calls)


def test_pending_unload_is_not_reported_as_success_or_retried(server):
    server.post_status = "unloading"
    assert memory.release_omlx_for_tts("indextts2")["models"][0]["status"] == "unload_unconfirmed"
    assert len([c for c in server.calls if c[0] == "POST"]) == 1


def test_missing_management_capability_never_attempts_unload(server):
    server.paths.pop("/admin/api/activity")
    assert memory.release_omlx_for_tts("indextts2")["models"][0]["status"] == "unsupported_api"
    assert not any(c[0] == "POST" for c in server.calls)


def test_api_failure_is_visible_without_stopping_tts_or_leaking_secrets(server, monkeypatch, tmp_path):
    server.failure = requests.ConnectionError("private-token")
    monkeypatch.setattr(memory, "_selected_providers", lambda: iter([provider(api_key="private-token")]))
    with memory.local_tts_memory_phase("indextts2"):
        pass  # Synthesis remains permitted; text/audio gates are independent.
    saved = (tmp_path / "output/log/omlx_memory_events.jsonl").read_text()
    assert "unavailable" in saved and "private-token" not in saved


def test_disabled_policy_has_no_side_effects(server, monkeypatch):
    monkeypatch.setattr(memory, "load_key", lambda key, default=None: "false")
    with memory.local_tts_memory_phase("indextts2"):
        pass
    assert server.calls == []


def test_only_current_local_routes_are_resolved_before_management(monkeypatch):
    settings = {"llm.provider": "omlx", "translation.speech_budget.candidate_provider": "omlx_homura",
                "translation.speech_budget.semantic_candidate_provider": "omlx_remote",
                "llm.providers.omlx.base_url": "http://localhost:8000/v1",
                "llm.providers.omlx_homura.base_url": "http://localhost:8000/v1",
                "llm.providers.omlx_remote.base_url": "https://remote.example/v1"}
    called = []
    def resolve(name):
        called.append(name)
        return provider(name=name)
    monkeypatch.setattr(memory, "load_key", lambda key, default=None: settings.get(key, default))
    monkeypatch.setattr("core.llm_provider.get_llm_provider_config", resolve)
    assert [p.name for p in _REAL_SELECTED_PROVIDERS()] == ["omlx", "omlx_homura"]
    assert called == ["omlx", "omlx_homura"]


def test_model_reloaded_by_later_inference_is_released_before_next_batch(server):
    assert memory.release_omlx_for_tts("indextts2")["models"][0]["status"] == "unloaded"
    server.models[0]["loaded"] = True  # Represents server on-demand loading.
    assert memory.release_omlx_for_tts("indextts2")["models"][0]["status"] == "unloaded"
    assert len([c for c in server.calls if c[0] == "POST"]) == 2


def test_local_inference_waits_for_tts_and_error_releases_guard(server):
    entered, completed = threading.Event(), threading.Event()
    def infer():
        entered.set()
        with memory.omlx_inference_phase(provider()): completed.set()
    with pytest.raises(RuntimeError, match="tts failure"):
        with memory.local_tts_memory_phase("indextts2"):
            thread = threading.Thread(target=infer)
            thread.start()
            assert entered.wait(1)
            assert not completed.wait(.05)
            raise RuntimeError("tts failure")
    thread.join(1)
    assert completed.is_set()


@pytest.mark.parametrize("batch", [False, True])
def test_actual_tts_subprocess_is_inside_memory_phase(monkeypatch, tmp_path, batch):
    import wave
    from core.providers import mlx_tts
    from core.providers.contracts import TTSRequest
    order = []
    @contextmanager
    def phase(backend):
        order.append("release")
        yield
        order.append("end")
    def run(cmd, **kwargs):
        assert order == ["release"]
        order.append("tts")
        path = (tmp_path / "out.wav") if not batch else __import__("pathlib").Path(cmd[cmd.index("--output-dir") + 1]) / "0001_take.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "w") as handle:
            handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000)
            handle.writeframes(b"\0\0" * 1600)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(mlx_tts, "local_tts_memory_phase", phase)
    monkeypatch.setattr(mlx_tts.subprocess, "run", run)
    monkeypatch.setattr(mlx_tts, "get_audio_duration", lambda p: .1)
    monkeypatch.setattr(mlx_tts, "write_conditioning_receipt", lambda *a, **kw: None)
    req = TTSRequest(text="Hello", ref_audio=str(tmp_path / "ref.wav"), output_path=str(tmp_path / "out.wav"))
    backend = mlx_tts.IndexTTS2Backend({"root": str(tmp_path)})
    if batch: backend.synthesize_batch([req])
    else: backend.synthesize(req)
    assert order == ["release", "tts", "end"]
