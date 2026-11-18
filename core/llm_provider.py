import os
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from openai import OpenAI

from core.config_utils import load_key


@dataclass(frozen=True)
class LLMProviderConfig:
    name: str
    model: str
    base_url: str
    api_key: str
    supports_json_object: bool = False
    provider_type: str = "openai_compatible"
    timeout_seconds: float | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    runtime_python: str | None = None
    model_path: str | None = None
    model_revision: str | None = None


def fix_base_url(base_url: str) -> str:
    if "ark" in base_url:
        return "https://ark.cn-beijing.volces.com/api/v3"
    if "v1" not in base_url:
        return base_url.strip("/") + "/v1"
    return base_url


def _provider_from_api_block() -> LLMProviderConfig:
    support_json = load_key("llm_support_json", [])
    model = load_key("api.model")
    return LLMProviderConfig(
        name="openai_compatible",
        model=model,
        base_url=fix_base_url(load_key("api.base_url")),
        api_key=load_key("api.key", ""),
        supports_json_object=model in support_json,
        timeout_seconds=float(load_key("llm.timeout_seconds", 180)),
    )


def get_llm_provider_config(provider_name: str | None = None) -> LLMProviderConfig:
    default_route = provider_name is None
    provider_name = provider_name or load_key("llm.provider", "openai_compatible")
    if provider_name == "openai_compatible":
        return _provider_from_api_block()

    provider = load_key(f"llm.providers.{provider_name}", None)
    if not provider:
        raise KeyError(f"LLM provider '{provider_name}' is not defined")
    if default_route and provider.get("purpose") == "speech_duration":
        raise ValueError("The speech-duration translator cannot run general pipeline tasks; "
                         "configure translation.speech_budget.candidate_provider instead")

    env_key = provider.get("api_key_env")
    api_key = os.environ.get(env_key, "") if env_key else provider.get("api_key", "")
    if not api_key:
        api_key = provider.get("default_api_key", "")
    # Read leaf keys so local environment overrides work for presets too.
    api_key = load_key(f"llm.providers.{provider_name}.api_key", api_key or "")
    model = load_key(f"llm.providers.{provider_name}.model", provider["model"])
    if provider_name == "omlx" and model == "auto":
        from core.providers.omlx import resolve_omlx_model

        model = resolve_omlx_model(model)
    model_revision = provider.get("model_revision")
    if "index-homura" in model.lower() and provider.get("model_path"):
        from core.model_artifacts import local_model_fingerprint
        try:
            model_revision = f"{model_revision or 'local'}#sha256={local_model_fingerprint(provider['model_path'])}"
        except Exception:
            model_revision = f"{model_revision or 'local'}#unresolved"
    return LLMProviderConfig(
        name=provider_name,
        model=model,
        base_url=(provider["base_url"] if provider.get("type")=="mlx_local" else
                  fix_base_url(load_key(f"llm.providers.{provider_name}.base_url", provider["base_url"]))),
        api_key=api_key,
        supports_json_object=bool(provider.get("supports_json_object", False)),
        provider_type=provider.get("type", "openai_compatible"),
        timeout_seconds=float(provider.get("timeout_seconds", load_key("llm.timeout_seconds", 180))),
        temperature=provider.get("temperature"),
        max_tokens=provider.get("max_tokens"),
        chat_template_kwargs=dict(provider.get("chat_template_kwargs") or {}),
        runtime_python=provider.get("runtime_python"),model_path=provider.get("model_path"),
        model_revision=model_revision,
    )


def create_chat_client(config: LLMProviderConfig | None = None) -> Any:
    config = config or get_llm_provider_config()
    if config.provider_type == "mlx_local":
        from core.local_mlx_chat import get_local_mlx_client
        return get_local_mlx_client(config)
    from urllib.parse import urlparse
    local_omlx = config.name.startswith("omlx") and urlparse(config.base_url).hostname in {"127.0.0.1", "localhost", "::1"}
    if not config.api_key and not local_omlx:
        raise ValueError(
            f"API key is missing for LLM provider '{config.name}'. "
            "Set the matching environment variable or configure api.key."
        )
    return OpenAI(api_key=config.api_key or "local", base_url=config.base_url, timeout=config.timeout_seconds)


def build_completion_args(config: LLMProviderConfig, messages: list[dict[str, Any]], response_json: bool) -> dict[str, Any]:
    args = {"model": config.model, "messages": messages}
    if response_json and config.supports_json_object:
        args["response_format"] = {"type": "json_object"}
    if config.temperature is not None:
        args["temperature"] = config.temperature
    if config.max_tokens is not None:
        args["max_tokens"] = config.max_tokens
    if config.chat_template_kwargs:
        args["extra_body"] = {"chat_template_kwargs": dict(config.chat_template_kwargs)}
    return args


def completion_fingerprint(config: LLMProviderConfig, response_json: bool) -> str:
    """Scope cached responses to the endpoint and generation policy, never secrets."""
    data = asdict(config)
    data.pop("api_key", None)
    data.pop("timeout_seconds", None)
    data["response_json"] = response_json
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def get_llm_max_workers() -> int:
    global_limit = max(1, int(load_key("max_workers", 1)))
    name = load_key("llm.provider", "openai_compatible")
    limit = load_key(f"llm.providers.{name}.max_workers", global_limit)
    return min(global_limit, max(1, int(limit)))


def get_translation_strategy(config: LLMProviderConfig | None = None) -> str:
    strategy = str(load_key("translation.strategy", "auto"))
    if strategy not in {"auto", "localized", "legacy"}:
        raise ValueError("translation.strategy must be auto, localized, or legacy")
    if strategy == "auto":
        config = config or get_llm_provider_config()
        return "localized" if "index-translate" in config.model.lower() else "legacy"
    return strategy
