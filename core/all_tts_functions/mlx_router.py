from __future__ import annotations

from typing import Any

from rich import print as rprint

from core.config_utils import load_key
from core.all_tts_functions.tts_utils import get_prompt_text_from_df, get_reference_audio_path
from core.providers.contracts import TTSRequest
from core.providers.mlx_tts import synthesize_with_mlx_router


def _clean_optional(value):
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>", "none"}:
        return None
    return text


def mlx_router_tts(
    text: str,
    save_as: str,
    number: int,
    task_df,
    task_row: dict[str, Any] | None = None,
    target_duration: float | None = None,
) -> bool:
    task_row = task_row or {}
    speaker = task_row.get("speaker")
    ref_audio = task_row.get("ref_audio") or None
    fallback_number = None
    if not ref_audio:
        ref_audio, fallback_number = get_reference_audio_path(number, task_df=task_df, speaker=speaker)
    ref_number = fallback_number if fallback_number is not None else number
    ref_text = task_row.get("ref_text") or get_prompt_text_from_df(task_df, ref_number, fallback_text="")
    language = task_row.get("language") or load_key("target_language", "auto")
    source_language = load_key("source_language", "auto")
    forced_backend = _clean_optional(task_row.get("tts_backend")) or _clean_optional(task_row.get("backend"))

    result = synthesize_with_mlx_router(
        TTSRequest(
            text=text,
            output_path=save_as,
            number=number,
            language=language,
            source_language=source_language,
            target_language=load_key("target_language", "auto"),
            ref_audio=ref_audio,
            ref_text=ref_text,
            speaker_id=str(speaker) if speaker else None,
            emotion_ref=task_row.get("emotion_ref") or task_row.get("emotion_ref_audio"),
            target_duration=target_duration,
            task_row=task_row,
            metadata={"backend": forced_backend} if forced_backend else {},
        )
    )
    rprint(f"[green]✅ MLX router saved {save_as} via {result.backend} ({result.duration:.2f}s)[/green]")
    return True
