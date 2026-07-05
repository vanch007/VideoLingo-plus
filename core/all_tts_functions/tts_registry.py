from dataclasses import dataclass


@dataclass(frozen=True)
class TTSProvider:
    name: str
    label: str
    requires_service: bool = False
    requires_api_key: bool = False
    deprecated: bool = False
    implemented: bool = True
    recommended: bool = False
    lifecycle: str = "stable"
    notes: str = ""

    @property
    def selectable(self) -> bool:
        return self.implemented and not self.deprecated and self.lifecycle != "unavailable"


TTS_PROVIDERS = {
    "edge_tts": TTSProvider("edge_tts", "Cloudflare Edge TTS / Microsoft Edge", requires_service=True, recommended=True),
    "voxcpm_tts": TTSProvider("voxcpm_tts", "VoxCPM local REST API", requires_service=True),
    "gpt_sovits": TTSProvider("gpt_sovits", "GPT-SoVITS local server", requires_service=True),
    "index_tts2": TTSProvider("index_tts2", "IndexTTS2 local server", requires_service=True),
    "piper_tts": TTSProvider("piper_tts", "Piper local TTS"),
    "indonesian_tts": TTSProvider("indonesian_tts", "Indonesian VITS local TTS"),
    "custom_tts": TTSProvider("custom_tts", "Custom TTS adapter"),
    "cosyvoice3_tts": TTSProvider(
        "cosyvoice3_tts",
        "CosyVoice 3 local/API adapter",
        requires_service=True,
        implemented=False,
        lifecycle="unavailable",
        notes="Registered placeholder; adapter is not implemented in tts_main.",
    ),
    "elevenlabs_tts": TTSProvider(
        "elevenlabs_tts",
        "ElevenLabs TTS",
        requires_api_key=True,
        implemented=False,
        lifecycle="unavailable",
        notes="Registered placeholder; adapter is not implemented in tts_main.",
    ),
    "openai_tts": TTSProvider(
        "openai_tts",
        "OpenAI TTS",
        requires_api_key=True,
        implemented=False,
        lifecycle="unavailable",
        notes="Registered placeholder; adapter is not implemented in tts_main.",
    ),
    "mlx_router": TTSProvider("mlx_router", "Local MLX TTS router", requires_service=True, recommended=True),
    "mlx_indextts2": TTSProvider("mlx_indextts2", "MLX IndexTTS2 via router", requires_service=True, recommended=True),
    "mlx_omnivoice": TTSProvider("mlx_omnivoice", "MLX OmniVoice via router", requires_service=True),
    "mlx_qwen3_tts": TTSProvider("mlx_qwen3_tts", "MLX Qwen3-TTS via router", requires_service=True),
    "mlx_voxcpm2": TTSProvider("mlx_voxcpm2", "MLX VoxCPM2 via router", requires_service=True, lifecycle="experimental"),
}

LOCAL_TTS_METHODS = {
    "custom_tts",
    "index_tts2",
    "gpt_sovits",
    "voxcpm_tts",
    "mlx_router",
    "mlx_indextts2",
    "mlx_omnivoice",
    "mlx_qwen3_tts",
    "mlx_voxcpm2",
}

MLX_ROUTER_BACKENDS = {
    "mlx_indextts2": "indextts2",
    "mlx_omnivoice": "omnivoice",
    "mlx_qwen3_tts": "qwen3_tts",
    "mlx_voxcpm2": "voxcpm2",
}


def get_tts_provider(name: str) -> TTSProvider:
    if name not in TTS_PROVIDERS:
        raise ValueError(f"Unknown TTS method: {name}")
    return TTS_PROVIDERS[name]


def list_tts_methods(*, include_unavailable: bool = True) -> list[str]:
    if include_unavailable:
        return list(TTS_PROVIDERS.keys())
    return [name for name, provider in TTS_PROVIDERS.items() if provider.selectable]


def list_selectable_tts_methods() -> list[str]:
    """Methods that should be shown as normal choices in UI/CLI controls."""
    return list_tts_methods(include_unavailable=False)


def list_tts_provider_metadata() -> list[dict]:
    return [
        {
            "name": provider.name,
            "label": provider.label,
            "requires_service": provider.requires_service,
            "requires_api_key": provider.requires_api_key,
            "deprecated": provider.deprecated,
            "implemented": provider.implemented,
            "recommended": provider.recommended,
            "lifecycle": provider.lifecycle,
            "selectable": provider.selectable,
            "notes": provider.notes,
        }
        for provider in TTS_PROVIDERS.values()
    ]


def list_implemented_tts_methods() -> list[str]:
    """Methods that have code paths in tts_main, including experimental routes."""
    return [name for name, provider in TTS_PROVIDERS.items() if provider.implemented]


def is_local_tts_method(name: str) -> bool:
    return name in LOCAL_TTS_METHODS
