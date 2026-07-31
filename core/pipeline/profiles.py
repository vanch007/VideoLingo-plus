from __future__ import annotations

from core.config_utils import update_key


def apply_cinematic_profile() -> None:
    """Apply conservative defaults for high-quality subtitle + dubbing runs."""
    update_key("quality_mode", "high_sync")
    update_key("dubbing_quality.mode", "high_sync")
    update_key("dubbing_quality.allow_silence_fallback", False)
    update_key("dubbing_quality.asr_readback", True)
    update_key("dubbing_quality.content_score_min", 0.88)
    update_key("dubbing_quality.leak_score_max", 0.12)
    update_key("dubbing_quality.loudness_target_lufs", -20.0)
    update_key("demucs", True)
    update_key("whisper.runtime", "stable-ts")
    update_key("dubbing_quality.asr_readback_backend", "moss-mlx")
    update_key("tts_method", "mlx_router")
    update_key("mlx_tts.default_backend", "auto")
    update_key("rewrite_text_for_dubbing", True)


def apply_profile(name: str | None) -> None:
    if not name:
        return
    if name == "cinematic":
        apply_cinematic_profile()
        return
    if name in {"subtitle", "subtitle_only"}:
        update_key("quality_mode", "subtitle")
        update_key("dubbing_quality.mode", "subtitle")
        return
    raise ValueError(f"Unknown pipeline profile: {name}")
