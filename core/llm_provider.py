import os
from dataclasses import dataclass
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


def get_llm_provider_config() -> LLMProviderConfig:
    provider_name = load_key("llm.provider", "openai_compatible")
    if provider_name == "openai_compatible":
        return _provider_from_api_block()

    provider = load_key(f"llm.providers.{provider_name}", None)
    if not provider:
        raise KeyError(f"LLM provider '{provider_name}' is not defined")

    env_key = provider.get("api_key_env")
    api_key = os.environ.get(env_key, "") if env_key else provider.get("api_key", "")
    if not api_key:
        api_key = provider.get("default_api_key", "")
    model = provider["model"]
    if provider_name == "omlx" and model == "auto":
        from core.providers.omlx import resolve_omlx_model

        model = resolve_omlx_model(model)
    return LLMProviderConfig(
        name=provider_name,
        model=model,
        base_url=fix_base_url(provider["base_url"]),
        api_key=api_key,
        supports_json_object=bool(provider.get("supports_json_object", False)),
        provider_type=provider.get("type", "openai_compatible"),
        timeout_seconds=float(provider.get("timeout_seconds", load_key("llm.timeout_seconds", 180))),
    )


def create_chat_client(config: LLMProviderConfig | None = None) -> OpenAI:
    config = config or get_llm_provider_config()
    if not config.api_key and config.name != "omlx":
        raise ValueError(
            f"API key is missing for LLM provider '{config.name}'. "
            "Set the matching environment variable or configure api.key."
        )
    return OpenAI(api_key=config.api_key or "local", base_url=config.base_url, timeout=config.timeout_seconds)


def build_completion_args(config: LLMProviderConfig, messages: list[dict[str, Any]], response_json: bool) -> dict[str, Any]:
    args = {"model": config.model, "messages": messages}
    if response_json and config.supports_json_object:
        args["response_format"] = {"type": "json_object"}
    return args
