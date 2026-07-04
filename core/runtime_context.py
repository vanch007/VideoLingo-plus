from __future__ import annotations

import json
from pathlib import Path

from core.config_utils import load_key
from core.constants import OUTPUT_DIR


def _pipeline_state() -> dict:
    state_path = Path(OUTPUT_DIR) / "pipeline_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return state if isinstance(state, dict) else {}


def _config_value(key: str, default: str) -> str:
    try:
        value = load_key(key, default)
    except (FileNotFoundError, KeyError):
        return default
    return str(value or default)


def effective_target_language(default: str = "auto") -> str:
    state = _pipeline_state()
    target = state.get("target") or state.get("target_language") or _config_value("target_language", default)
    return str(target or default)


def effective_source_language(default: str = "auto") -> str:
    state = _pipeline_state()
    source = state.get("source") or state.get("source_language") or _config_value("source_language", default)
    return str(source or default)
