from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMRequest:
    messages: list[dict[str, Any]]
    response_json: bool = True
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    content: Any
    model: str
    provider: str
    raw: Any | None = None


class LLMClient(Protocol):
    name: str

    def complete(self, request: LLMRequest) -> LLMResponse:
        ...


@dataclass(frozen=True)
class TTSRequest:
    text: str
    output_path: str
    number: int | None = None
    language: str = "auto"
    source_language: str = "auto"
    target_language: str = "auto"
    ref_audio: str | None = None
    ref_text: str | None = None
    speaker_id: str | None = None
    emotion_ref: str | None = None
    target_duration: float | None = None
    task_row: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    emo_alpha: float | None = None


@dataclass(frozen=True)
class TTSResult:
    output_path: str
    backend: str
    duration: float = 0.0
    rtf: float | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TTSBackend(Protocol):
    name: str

    def synthesize(self, request: TTSRequest) -> TTSResult:
        ...


@dataclass(frozen=True)
class ASRVerificationResult:
    status: str
    content_score: float | None = None
    leakage_score: float | None = None
    transcript: str | None = None
    warnings: list[str] = field(default_factory=list)


class ASRVerifier(Protocol):
    name: str

    def verify(self, audio_path: str, expected_text: str, ref_text: str | None = None) -> ASRVerificationResult:
        ...
