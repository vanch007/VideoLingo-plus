from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from core.all_tts_functions.tts_registry import TTS_PROVIDERS, list_tts_provider_metadata
from core.config_utils import get_env_names, load_key
from core.llm_provider import fix_base_url


@dataclass(frozen=True)
class ProviderRecord:
    name: str
    kind: str
    status: str
    implemented: bool
    recommended: bool = False
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    supports_json_object: bool = False
    requires_service: bool = False
    requires_api_key: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VerifyFinding:
    provider: str
    kind: str
    status: str
    severity: str
    message: str


def _llm_status(provider_name: str, provider: dict[str, Any]) -> tuple[str, list[str]]:
    notes: list[str] = []
    model = str(provider.get("model", "") or "")
    status = str(provider.get("status", "") or "").strip()
    if status:
        return status, notes

    lowered_name = provider_name.lower()
    lowered_model = model.lower()
    if provider_name == "gemini_3_pro" or "gemini-3-pro-preview" in lowered_model:
        notes.append("Gemini 3 Pro Preview preset is stale; replace with a currently verified Gemini model before use.")
        return "deprecated", notes
    if provider_name == "omlx":
        return "experimental", notes
    if provider_name == "lm_studio":
        return "experimental", notes
    if lowered_name.startswith("openai_") or lowered_name.startswith("deepseek_") or lowered_name.startswith("gemini_"):
        return "candidate", notes
    return "candidate", notes


def list_llm_provider_records() -> list[ProviderRecord]:
    records = [
        ProviderRecord(
            name="openai_compatible",
            kind="llm",
            status="stable",
            implemented=True,
            recommended=True,
            model=str(load_key("api.model", "")),
            base_url=fix_base_url(str(load_key("api.base_url", ""))),
            api_key_env="/".join(get_env_names("api.key")) or None,
            supports_json_object=str(load_key("api.model", "")) in (load_key("llm_support_json", []) or []),
            requires_api_key=True,
            notes=["Default route; reads api.base_url and api.model."],
        )
    ]
    providers = load_key("llm.providers", {}) or {}
    for provider_name, provider in providers.items():
        provider = dict(provider or {})
        status, notes = _llm_status(provider_name, provider)
        env_key = provider.get("api_key_env")
        requires_key = bool(env_key) and provider_name not in {"omlx", "lm_studio"}
        records.append(
            ProviderRecord(
                name=provider_name,
                kind="llm",
                status=status,
                implemented=provider.get("type", "openai_compatible") == "openai_compatible",
                recommended=False,
                model=str(provider.get("model", "")),
                base_url=fix_base_url(str(provider.get("base_url", ""))),
                api_key_env=str(env_key) if env_key else None,
                supports_json_object=bool(provider.get("supports_json_object", False)),
                requires_api_key=requires_key,
                notes=notes,
            )
        )
    return records


def list_tts_provider_records() -> list[ProviderRecord]:
    records: list[ProviderRecord] = []
    for item in list_tts_provider_metadata():
        provider = TTS_PROVIDERS[item["name"]]
        records.append(
            ProviderRecord(
                name=provider.name,
                kind="tts",
                status=provider.lifecycle,
                implemented=provider.implemented,
                recommended=provider.recommended,
                requires_service=provider.requires_service,
                requires_api_key=provider.requires_api_key,
                notes=[provider.notes] if provider.notes else [],
            )
        )
    return records


def list_provider_records() -> dict[str, list[dict[str, Any]]]:
    return {
        "llm": [asdict(record) for record in list_llm_provider_records()],
        "tts": [asdict(record) for record in list_tts_provider_records()],
    }


def _credential_available(record: ProviderRecord) -> bool:
    if record.api_key_env:
        env_names = [item for item in record.api_key_env.split("/") if item]
        if any(os.environ.get(env_name) for env_name in env_names):
            return True
    if record.kind == "llm" and record.name == "openai_compatible":
        return bool(load_key("api.key", ""))
    return False


def verify_provider_records(*, include_experimental: bool = False) -> dict[str, Any]:
    findings: list[VerifyFinding] = []
    records = list_llm_provider_records() + list_tts_provider_records()

    for record in records:
        if record.status in {"deprecated", "unavailable"}:
            findings.append(
                VerifyFinding(
                    provider=record.name,
                    kind=record.kind,
                    status=record.status,
                    severity="error",
                    message="Provider is not eligible for normal selection or promotion.",
                )
            )
        if not record.implemented:
            findings.append(
                VerifyFinding(
                    provider=record.name,
                    kind=record.kind,
                    status=record.status,
                    severity="error",
                    message="Provider is registered but its adapter is not implemented.",
                )
            )
        if record.requires_api_key and record.status not in {"deprecated", "unavailable"}:
            if not _credential_available(record):
                severity = "warning" if record.status in {"candidate", "experimental"} else "error"
                credential_hint = record.api_key_env or "configured secret"
                findings.append(
                    VerifyFinding(
                        provider=record.name,
                        kind=record.kind,
                        status=record.status,
                        severity=severity,
                        message=f"API key is not configured: {credential_hint}",
                    )
                )
        if record.status == "experimental" and not include_experimental:
            findings.append(
                VerifyFinding(
                    provider=record.name,
                    kind=record.kind,
                    status=record.status,
                    severity="warning",
                    message="Experimental provider is excluded from recommended/default routes.",
                )
            )

    errors = [finding for finding in findings if finding.severity == "error"]
    warnings = [finding for finding in findings if finding.severity == "warning"]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "findings": [asdict(finding) for finding in findings],
        "providers": list_provider_records(),
    }
