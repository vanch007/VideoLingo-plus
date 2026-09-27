from __future__ import annotations

from typing import Any

from rich import print as rprint

from core.config_utils import load_key
from core.all_tts_functions.tts_utils import get_prompt_text_from_df
from core.providers.contracts import TTSRequest
from core.providers.mlx_tts import synthesize_with_mlx_router
from core.runtime_context import effective_source_language, effective_target_language
from core.tts_reference_plan import resolve_reference_plan, write_conditioning_receipt


def _clean_optional(value):
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>", "none"}:
        return None
    return text


def _clean_bool(value) -> bool:
    if value is None or (isinstance(value, float) and value != value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_mlx_tts_request(
    text: str,
    save_as: str,
    number: int,
    task_df,
    task_row: dict[str, Any] | None = None,
    target_duration: float | None = None,
) -> TTSRequest:
    task_row = task_row or {}
    speaker = _clean_optional(task_row.get("speaker"))
    reference_plan = resolve_reference_plan({**task_row, "number": number}, task_df)
    ref_audio = reference_plan["speaker_reference"]["path"]
    ref_number = reference_plan["speaker_reference_number"]
    ref_text = _clean_optional(task_row.get("ref_text")) or get_prompt_text_from_df(task_df, ref_number, fallback_text="")
    target_language = effective_target_language(load_key("target_language", "auto"))
    language = task_row.get("language") or target_language
    source_language = effective_source_language(load_key("source_language", "auto"))
    forced_backend = _clean_optional(task_row.get("tts_backend")) or _clean_optional(task_row.get("backend"))

    return TTSRequest(
        text=text,
        output_path=save_as,
        number=number,
        language=language,
        source_language=source_language,
        target_language=target_language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        speaker_id=str(speaker) if speaker else None,
        emotion_ref=(reference_plan["emotion_reference"] or {}).get("path"),
        emo_alpha=reference_plan["emo_alpha"],
        target_duration=target_duration,
        task_row=task_row,
        metadata={
            "reference_plan": reference_plan,
            "requires_separate_emotion": reference_plan["requires_separate_emotion"],
            **({"backend": forced_backend} if forced_backend else {}),
            **(
                {"disable_native_fit": True}
                if _clean_bool(task_row.get("disable_native_fit", False))
                else {}
            ),
        },
    )


def mlx_router_tts(
    text: str,
    save_as: str,
    number: int,
    task_df,
    task_row: dict[str, Any] | None = None,
    target_duration: float | None = None,
) -> bool:
    request = build_mlx_tts_request(
        text, save_as, number, task_df,
        task_row=task_row,
        target_duration=target_duration,
    )
    result = synthesize_with_mlx_router(request)
    write_conditioning_receipt(request, backend=result.backend)
    rprint(f"[green]✅ MLX router saved {save_as} via {result.backend} ({result.duration:.2f}s)[/green]")
    return True
