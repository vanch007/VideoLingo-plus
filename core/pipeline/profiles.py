from __future__ import annotations

from core.config_utils import update_key, load_key

FAST_POLICY_VERSION = "fast-v2"

# Persist the user's strict choices before applying a fast overlay.
_STRICT_KEYS = {
    "translation.strict_review_gate": True, "translation.review_mode": "audit_only",
    "mlx_tts.backends.indextts2.profile": "auto",
    "mlx_tts.backends.indextts2.chinese_profile": "standard",
    "source_quality.enabled": True,
}

def _strict_baseline():
    return load_key("pipeline_profiles.strict_baseline", {})



def apply_fast_profile() -> None:
    """Apply fast defaults: streamlined batch translation & cloning dubbing."""
    if not _strict_baseline():
        previous_fast = load_key("quality_mode", "high_sync") == "fast" or load_key("fast_mode.enabled", False)
        baseline = {key: (default if previous_fast else load_key(key, default)) for key, default in _STRICT_KEYS.items()}
        update_key("pipeline_profiles.strict_baseline", baseline)
    update_key("pipeline_policy_version", FAST_POLICY_VERSION)
    update_key("source_quality.enabled", False)
    update_key("quality_mode", "fast")
    update_key("dubbing_quality.mode", "fast")
    update_key("dubbing_quality.allow_silence_fallback", False)
    update_key("dubbing_quality.asr_readback", False)
    update_key("dubbing_quality.max_speed_factor", 1.25)
    update_key("translation.speech_budget.max_speed_factor", 1.25)
    update_key("translation.strict_review_gate", False)
    update_key("translation.review_mode", "none")
    update_key("demucs", True)
    update_key("speaker_diarization.enabled", True)
    update_key("tts_method", "mlx_indextts2")
    update_key("mlx_tts.backends.indextts2.chinese_profile", "v25")
    update_key("mlx_tts.backends.indextts2.profile", "v25")
    update_key("mlx_tts.release_omlx_before_synthesis", True)
    update_key("fast_mode.enabled", True)
    update_key("fast_mode.max_speed_factor", 1.25)
    update_key("fast_mode.max_gap_extension_seconds", 0.35)
    update_key("fast_mode.video_extension_enabled", True)


def apply_cinematic_profile() -> None:
    """Apply conservative defaults for high-quality subtitle + dubbing runs."""
    baseline = _strict_baseline()
    for key, default in _STRICT_KEYS.items():
        update_key(key, baseline.get(key, default if load_key("fast_mode.enabled", False) else load_key(key, default)))
    update_key("pipeline_policy_version", "cinematic-v1")
    update_key("quality_mode", "high_sync")
    update_key("dubbing_quality.mode", "high_sync")
    update_key("dubbing_quality.allow_silence_fallback", False)
    update_key("dubbing_quality.asr_readback", True)
    update_key("dubbing_quality.content_score_min", 0.88)
    update_key("dubbing_quality.leak_score_max", 0.12)
    update_key("dubbing_quality.loudness_target_lufs", -20.0)
    update_key("dubbing_quality.max_speed_factor", 1.08)
    update_key("translation.speech_budget.max_speed_factor", 1.08)
    update_key("fast_mode.enabled", False)
    update_key("demucs", True)
    # Preserve the source recognizer selected in configuration. A dubbing
    # profile must not undo independently calibrated source-ASR settings.
    update_key("speaker_diarization.enabled", True)
    update_key("speaker_diarization.backend", "nemotron-mlx")
    update_key("speaker_diarization.required", True)
    update_key("speaker_diarization.min_word_coverage", 0.95)
    update_key("speaker_diarization.source_coverage.required", True)
    update_key("translation_context.require_speakers", True)
    # The local Qwen3 command is the verified readback route for generated
    # English WAVs; MOSS can fail to load those files on this runtime.
    update_key("dubbing_quality.asr_readback_backend", "command")
    update_key("tts_method", "mlx_router")
    update_key("mlx_tts.default_backend", "auto")
    update_key("rewrite_text_for_dubbing", True)


def apply_profile(name: str | None) -> None:
    if not name:
        return
    if name == "fast":
        apply_fast_profile()
        return
    if name == "cinematic":
        apply_cinematic_profile()
        return
    if name in {"subtitle", "subtitle_only"}:
        update_key("quality_mode", "subtitle")
        update_key("dubbing_quality.mode", "subtitle")
        return
    raise ValueError(f"Unknown pipeline profile: {name}")
