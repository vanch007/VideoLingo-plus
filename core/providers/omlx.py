from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from core.config_utils import load_key


@dataclass(frozen=True)
class OmlxModel:
    id: str
    owned_by: str = "omlx"


def _auth_headers() -> dict[str, str]:
    api_key = load_key("llm.providers.omlx.api_key", load_key("llm.providers.omlx.default_api_key", "1234"))
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def get_omlx_base_url() -> str:
    return str(load_key("llm.providers.omlx.base_url", "http://127.0.0.1:8000/v1")).rstrip("/")


def list_omlx_models(timeout: float = 5.0) -> list[OmlxModel]:
    url = f"{get_omlx_base_url()}/models"
    response = requests.get(url, headers=_auth_headers(), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    models = []
    for item in payload.get("data", []):
        model_id = item.get("id")
        if model_id:
            models.append(OmlxModel(id=model_id, owned_by=item.get("owned_by", "omlx")))
    return models


def resolve_omlx_model(configured_model: str | None = None, timeout: float = 5.0) -> str:
    configured_model = configured_model or load_key("llm.providers.omlx.model", "auto")
    if configured_model and configured_model != "auto":
        return configured_model
    models = list_omlx_models(timeout=timeout)
    if not models:
        raise RuntimeError("oMLX returned no models from /v1/models")
    model_ids = {model.id for model in models}
    settings_path = Path.home() / ".omlx" / "model_settings.json"
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            for model_id, model_settings in settings.get("models", {}).items():
                if model_id in model_ids and model_settings.get("is_default"):
                    return model_id
        except Exception:
            pass
    return models[0].id


def smoke_chat(prompt: str = "Translate to Vietnamese only: hello", timeout: float = 20.0) -> dict[str, Any]:
    model = resolve_omlx_model(timeout=5.0)
    response = requests.post(
        f"{get_omlx_base_url()}/chat/completions",
        headers=_auth_headers(),
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "model": model,
        "content": payload.get("choices", [{}])[0].get("message", {}).get("content", ""),
        "usage": payload.get("usage", {}),
    }
