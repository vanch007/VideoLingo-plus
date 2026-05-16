from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from core.all_whisper_methods.audio_preprocess import get_audio_duration
from core.config_utils import load_key
from core.providers.contracts import TTSRequest, TTSResult


VIETNAMESE_RE = re.compile(
    r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
    r"óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶ"
    r"ÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ]"
)
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class BackendHealth:
    name: str
    ok: bool
    detail: str


def looks_vietnamese(text: str) -> bool:
    return bool(VIETNAMESE_RE.search(text or ""))


def looks_chinese(text: str) -> bool:
    return bool(CHINESE_RE.search(text or ""))


def _load_backend_config(name: str) -> dict[str, Any]:
    return dict(load_key(f"mlx_tts.backends.{name}", {}) or {})


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _clean_backend_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>", "none"}:
        return None
    return text


def _cmd_prefix(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return shlex.split(value)
    return []


def _project_path(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((Path.cwd() / path).resolve())


class CliBackend:
    name = "base"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.root = Path(cfg.get("root", ".")).expanduser()
        self.timeout = int(cfg.get("timeout_seconds", 300))

    def _run(self, cmd: list[str], request: TTSRequest) -> TTSResult:
        start = time.perf_counter()
        output_path_text = _project_path(request.output_path) or request.output_path
        cmd = [output_path_text if item == request.output_path else item for item in cmd]
        output_path = Path(output_path_text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        pythonpath = self.cfg.get("pythonpath")
        if pythonpath:
            env["PYTHONPATH"] = str(Path(pythonpath).expanduser())
        elif self.root.exists():
            env["PYTHONPATH"] = f"{self.root}{os.pathsep}{env.get('PYTHONPATH', '')}"

        result = subprocess.run(
            cmd,
            cwd=str(self.root) if self.root.exists() else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"{self.name} failed with code {result.returncode}: {stderr[-1200:]}")
        if not output_path.exists() or output_path.stat().st_size < int(load_key("dubbing_quality.min_audio_size", 1000)):
            raise RuntimeError(f"{self.name} did not create a valid audio file: {output_path}")
        duration = get_audio_duration(str(output_path))
        elapsed = max(time.perf_counter() - start, 0.001)
        return TTSResult(
            output_path=output_path_text,
            backend=self.name,
            duration=duration,
            rtf=elapsed / duration if duration > 0 else None,
            metadata={"stdout": (result.stdout or "").strip()[-1200:]},
        )

    def health(self) -> BackendHealth:
        if not self.root.exists():
            return BackendHealth(self.name, False, f"missing root: {self.root}")
        return BackendHealth(self.name, True, f"root ok: {self.root}")


class IndexTTS2Backend(CliBackend):
    name = "indextts2"

    def synthesize(self, request: TTSRequest) -> TTSResult:
        if not request.ref_audio:
            raise ValueError("indextts2 requires ref_audio for VideoLingo voice cloning")
        cfg = self.cfg
        cmd = _cmd_prefix(cfg.get("command_prefix", ["uv", "run", "mlx-indextts"]))
        cmd += [
            "generate",
            "--ref-audio",
            _project_path(request.ref_audio) or request.ref_audio,
            "--text",
            request.text,
            "--output",
            _project_path(request.output_path) or request.output_path,
            "--profile",
            str(cfg.get("profile", "auto")),
        ]
        if cfg.get("model"):
            cmd += ["--model", str(cfg["model"])]
        if request.target_duration:
            cmd += ["--target-duration", f"{request.target_duration:.3f}"]
            if _as_bool(cfg.get("fit_duration"), False):
                cmd += ["--fit-duration"]
        if not _as_bool(cfg.get("denoise_ref", True), True):
            cmd += ["--no-denoise-ref"]
        if request.emotion_ref:
            cmd += ["--emotion-ref-audio", _project_path(request.emotion_ref) or request.emotion_ref]
        return self._run(cmd, request)


class QwenTTSBackend(CliBackend):
    name = "qwen3_tts"

    def synthesize(self, request: TTSRequest) -> TTSResult:
        cfg = self.cfg
        cmd = _cmd_prefix(cfg.get("command_prefix", ["uv", "run", "mlx-qwen-tts"]))
        cmd += [
            "generate",
            "--model",
            str(cfg.get("model", "qwen3_tts_8bit")),
            "--text",
            request.text,
            "--output",
            _project_path(request.output_path) or request.output_path,
            "--lang-code",
            request.language if request.language != "auto" else str(cfg.get("lang_code", "auto")),
        ]
        if request.ref_audio:
            cmd += ["--ref-audio", _project_path(request.ref_audio) or request.ref_audio]
        if request.ref_text:
            cmd += ["--ref-text", request.ref_text]
        if _as_bool(cfg.get("require_ref_text"), True) and request.ref_audio:
            cmd += ["--require-ref-text"]
        if cfg.get("voice"):
            cmd += ["--voice", str(cfg["voice"])]
        if cfg.get("instruct"):
            cmd += ["--instruct", str(cfg["instruct"])]
        if cfg.get("require_official"):
            cmd += ["--require-official"]
        return self._run(cmd, request)


class OmniVoiceBackend(CliBackend):
    name = "omnivoice"

    def synthesize(self, request: TTSRequest) -> TTSResult:
        cfg = self.cfg
        cmd = _cmd_prefix(cfg.get("command_prefix", ["uv", "run", "mlx-omnivoice"]))
        cmd += [
            "generate",
            "--model",
            str(cfg.get("model", "omnivoice_8bit")),
            "--text",
            request.text,
            "--output",
            _project_path(request.output_path) or request.output_path,
        ]
        if request.ref_audio:
            cmd += ["--ref-audio", _project_path(request.ref_audio) or request.ref_audio]
        if request.ref_text:
            cmd += ["--ref-text", request.ref_text]
        if cfg.get("instruct"):
            cmd += ["--instruct", str(cfg["instruct"])]
        if request.target_duration:
            cmd += ["--duration", f"{request.target_duration:.3f}"]
        if _as_bool(cfg.get("require_ref_text"), True) and request.ref_audio:
            cmd += ["--require-ref-text"]
        return self._run(cmd, request)


class VoxCPM2Backend(CliBackend):
    name = "voxcpm2"

    def synthesize(self, request: TTSRequest) -> TTSResult:
        cfg = self.cfg
        cmd = _cmd_prefix(cfg.get("command_prefix", ["python3", "-m", "mlx_voxcpm2.cli"]))
        max_len = int(cfg.get("max_len", max(32, min(256, len(request.text) * 2))))
        cmd += [
            "generate",
            "--model-dir",
            str(cfg.get("model_dir", "models/VoxCPM2-official-mlx-int8-components")),
            "--text",
            request.text,
            "--output",
            _project_path(request.output_path) or request.output_path,
            "--max-len",
            str(max_len),
            "--inference-timesteps",
            str(cfg.get("inference_timesteps", 4)),
            "--cfg-value",
            str(cfg.get("cfg_value", 2.0)),
        ]
        if request.ref_audio:
            cmd += ["--reference-wav-path", _project_path(request.ref_audio) or request.ref_audio]
        if request.ref_text:
            cmd += ["--prompt-text", request.ref_text]
        if _as_bool(cfg.get("retry_badcase"), True):
            cmd += ["--retry-badcase"]
        if _as_bool(cfg.get("denoise_ref"), False):
            cmd += ["--denoise-ref"]
        return self._run(cmd, request)


class MlxTTSRouter:
    def __init__(self):
        self.backends = {
            "indextts2": IndexTTS2Backend(_load_backend_config("indextts2")),
            "omnivoice": OmniVoiceBackend(_load_backend_config("omnivoice")),
            "qwen3_tts": QwenTTSBackend(_load_backend_config("qwen3_tts")),
            "voxcpm2": VoxCPM2Backend(_load_backend_config("voxcpm2")),
        }

    def select_backend(self, request: TTSRequest) -> str:
        forced = _clean_backend_name(request.metadata.get("backend")) or _clean_backend_name(load_key("mlx_tts.default_backend", "auto")) or "auto"
        if forced and forced != "auto":
            if forced not in self.backends:
                raise ValueError(f"Unknown MLX TTS backend: {forced}")
            return forced

        text = request.text or ""
        language = request.language if request.language != "auto" else request.target_language
        if language in {"vi", "vi-VN", "vietnamese"} or looks_vietnamese(text):
            return "indextts2"
        if request.emotion_ref:
            return "indextts2"
        if request.ref_audio and request.ref_text and _as_bool(load_key("mlx_tts.router.prefer_qwen_with_clean_ref", False), False):
            return "qwen3_tts"
        if language in {"zh", "zh-CN", "chinese"} or looks_chinese(text):
            return "omnivoice"
        if _as_bool(load_key("mlx_tts.router.allow_voxcpm2_auto", False), False):
            return "voxcpm2"
        return "indextts2"

    def synthesize(self, request: TTSRequest) -> TTSResult:
        backend_name = self.select_backend(request)
        backend = self.backends[backend_name]
        result = backend.synthesize(request)
        metadata = dict(result.metadata)
        metadata["selected_by"] = "mlx_router"
        return TTSResult(
            output_path=result.output_path,
            backend=result.backend,
            duration=result.duration,
            rtf=result.rtf,
            warnings=result.warnings,
            metadata=metadata,
        )

    def health(self) -> list[BackendHealth]:
        results = []
        for name, backend in self.backends.items():
            enabled = _as_bool(load_key(f"mlx_tts.backends.{name}.enabled", True), True)
            if not enabled:
                results.append(BackendHealth(name, True, "disabled by config"))
                continue
            results.append(backend.health())
        return results


def list_backend_status() -> list[dict[str, Any]]:
    router = MlxTTSRouter()
    return [health.__dict__ for health in router.health()]


def synthesize_with_mlx_router(request: TTSRequest) -> TTSResult:
    return MlxTTSRouter().synthesize(request)


def write_router_plan(path: str = "output/audio/mlx_tts_router_plan.json") -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "default_backend": load_key("mlx_tts.default_backend", "auto"),
        "backends": list_backend_status(),
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
