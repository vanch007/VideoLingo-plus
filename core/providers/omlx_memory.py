"""Release idle, configured local oMLX models at the actual TTS boundary.

The server stays running and loads a model again on its next inference request.
This is resource management, not a text/audio acceptance gate. An unavailable or
busy management API is reported and never bypasses a quality check.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import threading
import time
from urllib.parse import quote, urlsplit

import requests

from core.config_utils import load_key

LOG = logging.getLogger(__name__)
_PHASE_LOCK = threading.RLock()
_API_PATHS = {
    "/v1/models/status": "get",
    "/admin/api/activity": "get",
    "/v1/models/{model_id}/unload": "post",
}


def _enabled():
    value = load_key("mlx_tts.release_omlx_before_synthesis", True)
    return value is True or str(value).lower() in {"1", "true", "yes", "on"}


def _origin(provider):
    url = urlsplit(provider.base_url)
    if (not provider.name.startswith("omlx")
            or provider.provider_type != "openai_compatible"
            or url.scheme not in {"http", "https"}
            or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username or url.password or url.query or url.fragment
            or url.path.rstrip("/") not in {"", "/v1"}):
        return None
    return f"{url.scheme}://{url.netloc}"


@contextmanager
def omlx_inference_phase(provider):
    """Keep this process's LLM requests out of a resident TTS batch."""
    if _enabled() and _origin(provider):
        with _PHASE_LOCK:
            yield
    else:
        yield


def _request(session, method, url):
    response = session.request(method, url, timeout=(2, 20 if method == "POST" else 5),
                               allow_redirects=False)
    response.raise_for_status()
    if response.status_code not in {200, 202}:
        raise ValueError("Unexpected oMLX management response")
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("Invalid oMLX management payload")
    return value


def _selected_providers():
    from core.llm_provider import LLMProviderConfig, get_llm_provider_config
    names = [load_key("llm.provider", "openai_compatible"),
             load_key("translation.speech_budget.candidate_provider", ""),
             load_key("translation.speech_budget.semantic_candidate_provider", "")]
    for name in dict.fromkeys(names):
        if name and str(name).startswith("omlx"):
            # Reject non-local routes before resolving an "auto" model, which
            # may otherwise contact that remote provider during TTS.
            prefix = f"llm.providers.{name}"
            route = LLMProviderConfig(name=str(name), model="", api_key="",
                                     base_url=load_key(prefix + ".base_url", ""),
                                     provider_type=load_key(prefix + ".type", "openai_compatible"))
            if _origin(route):
                yield get_llm_provider_config(str(name))


def _resolve_model(models, requested):
    if (not isinstance(models, list) or any(not isinstance(m, dict)
            or not isinstance(m.get("id"), str) for m in models)):
        raise ValueError("Malformed oMLX model status")
    exact = [m for m in models if m.get("id") == requested]
    matches = exact or [m for m in models if m.get("model_alias") == requested]
    if len(matches) != 1:
        return None
    model = matches[0]
    source = model.get("source_model_id")
    if source and source != model.get("id"):
        matches = [m for m in models if m.get("id") == source]
        return matches[0] if len(matches) == 1 else None
    return model


def _release_server(origin, providers):
    results = []
    with requests.Session() as session:
        # Loopback management must not be sent through a user's HTTP proxy.
        session.trust_env = False
        session.headers.update({"Accept": "application/json"})
        if providers[0].api_key:
            session.headers["Authorization"] = f"Bearer {providers[0].api_key}"
        spec = _request(session, "GET", origin + "/openapi.json")
        if any(method not in spec.get("paths", {}).get(path, {})
               for path, method in _API_PATHS.items()):
            return [{"status": "unsupported_api"}]
        state = _request(session, "GET", origin + "/v1/models/status")
        models = state.get("models")
        if not isinstance(models, list):
            raise ValueError("Missing oMLX model status")
        handled = set()
        for provider in providers:
            model = _resolve_model(models, provider.model)
            item = {"provider": provider.name, "requested_model": provider.model}
            if model is None:
                results.append({**item, "status": "model_not_resolved"})
                continue
            model_id = model["id"]
            if model_id in handled:
                continue
            handled.add(model_id)
            item["model"] = model_id
            if model.get("loaded") is False and model.get("is_loading") is False:
                results.append({**item, "status": "already_unloaded"})
                continue
            if (model.get("loaded") is not True or model.get("is_loading") is not False
                    or model.get("pinned") or model.get("distributed")):
                results.append({**item, "status": "protected_or_loading"})
                continue
            # The unload API can abort active requests. Missing activity is not
            # proof of idleness; recheck immediately before each unload.
            activity = _request(session, "GET", origin + "/admin/api/activity")
            active_data = activity.get("active_models")
            if not isinstance(active_data, dict):
                raise ValueError("Missing oMLX activity status")
            active_models = active_data.get("models", [])
            if not isinstance(active_models, list) or any(not isinstance(m, dict) for m in active_models):
                raise ValueError("Malformed oMLX activity status")
            matches = [m for m in active_models if m.get("id") == model_id]
            active = matches[0] if len(matches) == 1 else {}
            idle = active.get("idle_seconds")
            if (type(active.get("active_requests")) is not int or active["active_requests"] != 0
                    or type(active.get("waiting_requests")) is not int or active["waiting_requests"] != 0
                    or active.get("is_loading") is not False
                    or not isinstance(idle, (float, int)) or isinstance(idle, bool)
                    or not math.isfinite(idle) or idle <= 0):
                results.append({**item, "status": "busy_or_unknown"})
                continue
            before = state.get("current_model_memory")
            reply = _request(session, "POST", origin + "/v1/models/" + quote(model_id, safe="") + "/unload")
            after = _request(session, "GET", origin + "/v1/models/status")
            observed = _resolve_model(after.get("models", []), model_id)
            verified = bool(observed and observed.get("loaded") is False
                            and observed.get("is_loading") is False)
            results.append({**item, "status": "unloaded" if verified else "unload_unconfirmed",
                            "server_status": reply.get("status"),
                            "model_memory_before_bytes": before,
                            "model_memory_after_bytes": after.get("current_model_memory")})
            state = after
    return results


def release_omlx_for_tts(backend):
    """Best effort; only selected local routes, never all server models."""
    started = time.monotonic()
    groups, results = {}, []
    try:
        for provider in _selected_providers():
            origin = _origin(provider)
            if origin:
                groups.setdefault((origin, provider.api_key), []).append(provider)
        for (origin, _), providers in groups.items():
            try:
                results.extend({"server": origin, **item}
                               for item in _release_server(origin, providers))
            except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
                # Do not log HTTP bodies/headers or credentials from exceptions.
                results.append({"server": origin, "status": "unavailable",
                                "error_type": type(exc).__name__})
    except (KeyError, ValueError, TypeError, RuntimeError, OSError, requests.RequestException) as exc:
        results.append({"status": "configuration_unavailable", "error_type": type(exc).__name__})
    event = {"at": datetime.now(timezone.utc).isoformat(), "stage": "before_local_tts",
             "backend": backend, "models": results,
             "elapsed_seconds": round(time.monotonic() - started, 3)}
    if results:
        try:
            path = Path("output/log/omlx_memory_events.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            LOG.warning("Could not write oMLX memory handoff evidence")
        for item in results:
            status = item["status"]
            message = "oMLX memory handoff before %s: %s (%s)"
            args = (backend, status, item.get("model", item.get("requested_model", "server")))
            if status not in {"unloaded", "already_unloaded"}:
                LOG.warning(message, *args)
            else:
                LOG.info(message, *args)
    return event


@contextmanager
def local_tts_memory_phase(backend):
    if not _enabled():
        yield
        return
    # The entire TTS child lifetime is protected, including error teardown.
    with _PHASE_LOCK:
        release_omlx_for_tts(backend)
        yield
