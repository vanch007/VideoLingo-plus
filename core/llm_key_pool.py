"""Rotate requests across several API keys for one provider.

A provider with ``key_pool`` reads its keys at call time, from either
``opencodex:<provider>`` (that provider's ``apiKeyPool`` in
~/.opencodex/config.json, so opencodex stays the one place keys are managed)
or a local file (one key per line, ``#`` comments allowed). Keys never enter
config.yaml, the provider config, cached-response fingerprints or logs. Each request starts
from the next key; a key that answers 429, 5xx or a connection error cools
down and the same request moves on to the next key. A key that is rejected
(401/403) is parked for the rest of the process. When every key fails the last
error is raised and the caller's normal retry/backoff applies.

Used to call Google AI Studio directly: opencodex silently moves failing
AI Studio Gemini requests to Antigravity, which the user does not want.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

RATE_LIMIT_COOLDOWN_S = 60.0
SERVER_ERROR_COOLDOWN_S = 15.0
REJECTED_COOLDOWN_S = 24 * 3600.0

_LOCK = threading.Lock()
_CURSOR: dict[str, int] = {}
_COOLDOWN: dict[tuple[str, int], float] = {}


OPENCODEX_CONFIG = "~/.opencodex/config.json"


def _opencodex_keys(provider: str) -> list[str]:
    file = Path(OPENCODEX_CONFIG).expanduser()
    if not file.is_file():
        raise ValueError(f"opencodex config not found: {file}")
    entry = (json.loads(file.read_text(encoding="utf-8")).get("providers") or {}).get(provider)
    if not entry:
        raise ValueError(f"opencodex has no provider '{provider}'")
    keys = [str(item.get("key", "")).strip() for item in entry.get("apiKeyPool") or []
            if isinstance(item, dict) and not item.get("disabled")]
    keys.append(str(entry.get("apiKey") or "").strip())
    keys = [key for i, key in enumerate(keys) if key and "*" not in key and key not in keys[:i]]
    if not keys:
        raise ValueError(f"opencodex provider '{provider}' has no usable API keys")
    return keys


def read_keys(source: str) -> list[str]:
    if source.startswith("opencodex:"):
        return _opencodex_keys(source.split(":", 1)[1].strip())
    file = Path(source).expanduser()
    if not file.is_file():
        raise ValueError(f"Key pool file not found: {file}. Put one API key per line.")
    keys = []
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and line not in keys:
            keys.append(line)
    if not keys:
        raise ValueError(f"Key pool file has no keys: {file}")
    return keys


def _cooldown_for(error: Exception) -> float | None:
    """Seconds to park the key, or None when another key would not help."""
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return SERVER_ERROR_COOLDOWN_S
    status = getattr(error, "status_code", None)
    if status == 429:
        return RATE_LIMIT_COOLDOWN_S
    if status in {401, 403}:
        return REJECTED_COOLDOWN_S
    if isinstance(error, APIStatusError) and status is not None and status >= 500:
        return SERVER_ERROR_COOLDOWN_S
    return None


def _order(pool_id: str, size: int, now: float) -> list[int]:
    """Key indexes to try for one request: round-robin start, cooled-down keys last."""
    with _LOCK:
        start = _CURSOR.get(pool_id, 0) % size
        _CURSOR[pool_id] = start + 1
        ring = [(start + i) % size for i in range(size)]
        ready = [i for i in ring if _COOLDOWN.get((pool_id, i), 0) <= now]
        return ready or ring


class _Completions:
    def __init__(self, pool: "KeyPoolClient"):
        self._pool = pool

    def create(self, **kwargs: Any) -> Any:
        return self._pool._create(**kwargs)


class _Chat:
    def __init__(self, pool: "KeyPoolClient"):
        self.completions = _Completions(pool)


class KeyPoolClient:
    """Duck-types the ``client.chat.completions.create`` surface ask_gpt uses."""

    def __init__(self, source: str, base_url: str, timeout: float | None):
        self._keys_file = source if source.startswith("opencodex:") else str(Path(source).expanduser())
        self._keys = read_keys(self._keys_file)
        self._clients = [OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)
                         for key in self._keys]
        self.chat = _Chat(self)

    @property
    def size(self) -> int:
        return len(self._clients)

    def _create(self, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for index in _order(self._keys_file, self.size, time.monotonic()):
            try:
                return self._clients[index].chat.completions.create(**kwargs)
            except Exception as error:
                cooldown = _cooldown_for(error)
                if cooldown is None:
                    raise
                with _LOCK:
                    _COOLDOWN[(self._keys_file, index)] = time.monotonic() + cooldown
                last_error = error
        raise last_error  # type: ignore[misc]


def reset_state() -> None:
    with _LOCK:
        _CURSOR.clear()
        _COOLDOWN.clear()
