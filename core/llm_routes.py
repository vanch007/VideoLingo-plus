"""Per-task model routing: translation and analysis need different models.

A translation-tuned model (Index-Translate) writes good subtitles but is
unstable on analysis: asked three times to find recurring wordplay it found 腿
once, missed it once and returned invalid JSON once. `llm.routes` sends each
kind of request to its own provider, chosen by the request's log title:

    llm:
      routes:
        analysis: deepseek_v4_flash   # summary, roles, names, wordplay, STT fixes
        review: deepseek_v4_flash     # translation constraint and semantic review

An empty or missing route keeps the default provider (llm.provider), so
configs without routes behave exactly as before.
"""
from __future__ import annotations

from fnmatch import fnmatch

from core.config_utils import load_key

ROUTES: dict[str, tuple[str, ...]] = {
    "analysis": ("summary", "dialogue_roles_b*", "stt_correction_batch_*", "name_consistency", "wordplay"),
    "review": ("constraint_review", "translation_semantic_review_*"),
}


def route_for(log_title: str | None) -> str | None:
    """The route a request with this log title belongs to, if any."""
    title = str(log_title or "")
    for route, patterns in ROUTES.items():
        if any(fnmatch(title, pattern) for pattern in patterns):
            return route
    return None


def routed_provider_name(log_title: str | None) -> str | None:
    """The configured provider for this request's route, or None for the default provider."""
    route = route_for(log_title)
    if not route:
        return None
    name = str(load_key(f"llm.routes.{route}", "") or "").strip()
    if name and (load_key(f"llm.providers.{name}", {}) or {}).get("purpose") == "speech_duration":
        raise ValueError(f"llm.routes.{route} points at the speech-duration translator '{name}'; "
                         "it cannot run analysis or review")
    return name or None


def configured_routes() -> dict[str, str]:
    return {route: name for route in ROUTES if (name := str(load_key(f"llm.routes.{route}", "") or "").strip())}
