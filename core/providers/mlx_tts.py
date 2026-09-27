from __future__ import annotations

import json
import csv
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from core.all_whisper_methods.audio_preprocess import get_audio_duration
from core.config_utils import load_key
from core.providers.contracts import TTSRequest, TTSResult
from core.tts_reference_plan import validate_emotion_alpha, write_conditioning_receipt


VIETNAMESE_RE = re.compile(
    r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
    r"óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶ"
    r"ÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ]"
)
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
BACKEND_ALIASES = {
    "indextts": "indextts2",
    "mlx_indextts2": "indextts2",
    "qwen_tts": "qwen3_tts",
    "mlx_qwen3_tts": "qwen3_tts",
    "mlx_omnivoice": "omnivoice",
    "mlx_voxcpm2": "voxcpm2",
    "mlx_higgs_audio": "higgs",
    "mlx_dots_tts": "dots",
    "mlx_zonos2": "zonos2",
    "mlx_moss_tts": "moss",
    "mlx_ming_omni_tts": "ming",
}

LANGUAGE_NAMES = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
}


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
    return BACKEND_ALIASES.get(text, text)


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


def _audio_as_wav_bytes(path: str) -> bytes:
    """Normalize arbitrary project reference formats to mono WAV for clone APIs."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        output = Path(handle.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ac", "1", str(output)],
            check=True,
            capture_output=True,
        )
        return output.read_bytes()
    finally:
        output.unlink(missing_ok=True)


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
        inherited_pythonpath = env.get("PYTHONPATH", "") if _as_bool(self.cfg.get("inherit_pythonpath"), False) else ""
        if pythonpath:
            entries = pythonpath if isinstance(pythonpath, list) else str(pythonpath).split(os.pathsep)
            resolved = [str(Path(str(item)).expanduser()) for item in entries if str(item).strip()]
            env["PYTHONPATH"] = os.pathsep.join(resolved + ([inherited_pythonpath] if inherited_pythonpath else []))
        elif self.root.exists():
            env["PYTHONPATH"] = os.pathsep.join([str(self.root)] + ([inherited_pythonpath] if inherited_pythonpath else []))

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
        effective_backend = request.metadata.get("backend") or self.name
        write_conditioning_receipt(request, backend=effective_backend)
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
        prefix = _cmd_prefix(self.cfg.get("command_prefix"))
        if prefix and not (Path(prefix[0]).expanduser().exists() or shutil.which(prefix[0])):
            return BackendHealth(self.name, False, f"missing executable: {prefix[0]}")
        for required in self.cfg.get("required_files", []) or []:
            required_path = Path(str(required)).expanduser()
            if not required_path.is_absolute():
                required_path = self.root / required_path
            if not required_path.exists():
                return BackendHealth(self.name, False, f"missing required file: {required_path}")
        return BackendHealth(self.name, True, f"runtime ready: {self.root}")


class IndexTTS2Backend(CliBackend):
    name = "indextts2"

    def _emotion_alpha(self, request):
        return validate_emotion_alpha(
            request.emo_alpha if request.emo_alpha is not None else self.cfg.get("emo_alpha", 1.0)
        )

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
        language = str(cfg.get("language", "auto") or "auto").strip().lower()
        if language == "auto":
            language = str(request.language or request.target_language or "auto").lower()
        if language and language != "auto":
            cmd += ["--language", language]
        if cfg.get("seed") is not None:
            cmd += ["--seed", str(cfg["seed"])]
        fit_duration = _as_bool(cfg.get("fit_duration"), True)
        estimated_duration = request.task_row.get("est_dur") if request.task_row else None
        try:
            estimated_ratio = float(estimated_duration) / float(request.target_duration)
        except (TypeError, ValueError, ZeroDivisionError):
            estimated_ratio = 1.0
        disable_native_fit = _as_bool(
            request.metadata.get("disable_native_fit")
            or request.task_row.get("disable_native_fit"),
            False,
        )
        native_fit_safe = (
            not disable_native_fit
            and estimated_ratio <= float(cfg.get("native_fit_max_estimated_ratio", 1.08))
        )
        # The patched local IndexTTS2 treats target_duration as a safe token
        # budget with a 320-token content floor. Always pass the window so
        # short retries do not fall back to an excessively loose 1500-token
        # sampling cap; only the actual time stretch remains safety-gated.
        if request.target_duration:
            cmd += ["--target-duration", f"{request.target_duration:.3f}"]
            # IndexTTS2 can synthesize directly to a subtitle slot. Prefer its
            # native fit mode to generic post-generation time stretching,
            # which degrades cloned speech and leaves subtitle drift.
            if fit_duration and native_fit_safe:
                cmd += ["--fit-duration"]
        if not _as_bool(cfg.get("denoise_ref", True), True):
            cmd += ["--no-denoise-ref"]
        if not _as_bool(
            cfg.get("denoise_emotion_ref", cfg.get("denoise_ref", True)), True
        ):
            cmd += ["--no-denoise-emotion-ref"]
        if request.emotion_ref:
            cmd += ["--emotion-ref-audio", _project_path(request.emotion_ref) or request.emotion_ref]
            cmd += ["--emo-alpha", str(self._emotion_alpha(request))]
        return self._run(cmd, request)

    def synthesize_batch(self, requests: list[TTSRequest]) -> list[TTSResult]:
        """Generate a subtitle batch in one model-resident IndexTTS2 process."""
        if not requests:
            return []
        if any(not request.ref_audio for request in requests):
            raise ValueError("indextts2 requires ref_audio for every batch item")

        cfg = self.cfg
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="videolingo_indextts2_") as temp_dir_text:
            temp_dir = Path(temp_dir_text)
            csv_path = temp_dir / "batch.csv"
            generated_dir = temp_dir / "generated"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                fieldnames = [
                    "id", "text", "ref_audio", "emotion_ref_audio",
                    "target_duration_s", "fit_duration", "emo_alpha", "language", "speaker",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for index, request in enumerate(requests, start=1):
                    words_count = len(re.findall(r"[\wÀ-ỹ]+", str(request.text or "")))
                    # Estimated duration for English is ~0.40-0.45s per word
                    fresh_est = max(words_count * 0.42, 0.3) if words_count > 0 else 0.5
                    estimated_duration = fresh_est
                    try:
                        estimated_ratio = float(estimated_duration) / float(request.target_duration)
                    except (TypeError, ValueError, ZeroDivisionError):
                        estimated_ratio = 1.0
                    disable_native_fit = _as_bool(
                        request.metadata.get("disable_native_fit")
                        or request.task_row.get("disable_native_fit"),
                        False,
                    )
                    native_fit_safe = (
                        not disable_native_fit
                        and estimated_ratio <= float(
                            cfg.get("native_fit_max_estimated_ratio", 1.08)
                        )
                    )
                    use_native_fit = bool(
                        request.target_duration
                        and _as_bool(cfg.get("fit_duration"), True)
                        and native_fit_safe
                    )
                    writer.writerow({
                        "id": f"{index:04d}",
                        "text": request.text,
                        "ref_audio": _project_path(request.ref_audio) or request.ref_audio,
                        "emotion_ref_audio": (
                            _project_path(request.emotion_ref) or request.emotion_ref or ""
                        ),
                        "emo_alpha": self._emotion_alpha(request),
                        "language": request.language,
                        "speaker": request.speaker_id or "",
                        "target_duration_s": (
                            f"{request.target_duration:.3f}" if request.target_duration else ""
                        ),
                        "fit_duration": "true" if use_native_fit else "false",
                    })

            cmd = _cmd_prefix(cfg.get("command_prefix", ["uv", "run", "mlx-indextts"]))
            cmd += [
                "batch", "--input", str(csv_path), "--output-dir", str(generated_dir),
                "--profile", str(cfg.get("profile", "auto")),
            ]
            if cfg.get("model"):
                cmd += ["--model", str(cfg["model"])]
            language = str(cfg.get("language", "auto") or "auto").strip().lower()
            if language and language != "auto":
                cmd += ["--language", language]
            if cfg.get("seed") is not None:
                cmd += ["--seed", str(cfg["seed"])]
            if not _as_bool(cfg.get("denoise_ref", True), True):
                cmd += ["--no-denoise-ref"]
            if not _as_bool(
                cfg.get("denoise_emotion_ref", cfg.get("denoise_ref", True)), True
            ):
                cmd += ["--no-denoise-emotion-ref"]

            env = os.environ.copy()
            pythonpath = cfg.get("pythonpath")
            if pythonpath:
                entries = pythonpath if isinstance(pythonpath, list) else str(pythonpath).split(os.pathsep)
                env["PYTHONPATH"] = os.pathsep.join(
                    str(Path(str(item)).expanduser()) for item in entries if str(item).strip()
                )
            elif self.root.exists():
                env["PYTHONPATH"] = str(self.root)
            completed = subprocess.run(
                cmd,
                cwd=str(self.root) if self.root.exists() else None,
                env=env,
                capture_output=True,
                text=True,
                timeout=int(cfg.get("batch_timeout_seconds", max(self.timeout, 3600))),
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(f"indextts2 batch failed with code {completed.returncode}: {detail[-2000:]}")

            results: list[TTSResult] = []
            elapsed = max(time.perf_counter() - started, 0.001)
            for index, request in enumerate(requests, start=1):
                matches = sorted(generated_dir.glob(f"{index:04d}_*.wav"))
                if len(matches) != 1:
                    raise RuntimeError(
                        f"indextts2 batch output mismatch for item {index}: found {len(matches)} files"
                    )
                destination_text = _project_path(request.output_path) or request.output_path
                destination = Path(destination_text)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(matches[0], destination)
                if destination.stat().st_size < int(load_key("dubbing_quality.min_audio_size", 1000)):
                    raise RuntimeError(f"indextts2 produced invalid batch audio: {destination}")
                duration = get_audio_duration(str(destination))
                effective_backend = request.metadata.get("backend") or self.name
                write_conditioning_receipt(request, backend=effective_backend)
                results.append(TTSResult(
                    output_path=destination_text,
                    backend=self.name,
                    duration=duration,
                    rtf=(elapsed / len(requests)) / duration if duration > 0 else None,
                    metadata={"mode": "model_resident_batch"},
                ))
            return results


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
        clone_mode = str(cfg.get("clone_mode", "prompt") or "prompt").strip().lower()
        reference_only = clone_mode in {"reference", "reference_only", "clone"}
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
            reference_flag = "--reference-wav-path" if reference_only else "--prompt-wav-path"
            cmd += [reference_flag, _project_path(request.ref_audio) or request.ref_audio]
        if request.ref_text and not reference_only:
            cmd += ["--prompt-text", request.ref_text]
        if reference_only and cfg.get("control"):
            cmd += ["--control", str(cfg["control"])]
        if _as_bool(cfg.get("retry_badcase"), True):
            cmd += ["--retry-badcase"]
        if _as_bool(cfg.get("denoise_ref"), False):
            cmd += ["--denoise-ref"]
        return self._run(cmd, request)


class HiggsBackend(CliBackend):
    name = "higgs"

    def synthesize(self, request: TTSRequest) -> TTSResult:
        if not request.ref_audio:
            raise ValueError("higgs requires ref_audio for voice cloning")
        cmd = _cmd_prefix(self.cfg.get("command_prefix", [self.root / ".venv/bin/python", "scripts/run_tts.py"]))
        cmd += [
            "--text", request.text,
            "--output", _project_path(request.output_path) or request.output_path,
            "--reference-audio", _project_path(request.ref_audio) or request.ref_audio,
        ]
        if request.ref_text:
            cmd += ["--reference-text", request.ref_text]
        max_new_tokens = int(self.cfg.get("max_new_tokens", 512))
        if request.target_duration and _as_bool(
            self.cfg.get("use_target_duration_token_cap", True), True
        ):
            tokens_per_second = float(self.cfg.get("tokens_per_second", 36.0))
            token_buffer = int(self.cfg.get("token_buffer", 48))
            min_new_tokens = int(self.cfg.get("min_new_tokens", 64))
            dynamic_cap = max(
                min_new_tokens,
                int(math.ceil(request.target_duration * tokens_per_second + token_buffer)),
            )
            max_new_tokens = min(max_new_tokens, dynamic_cap)
        cmd += ["--max-new-tokens", str(max_new_tokens)]
        return self._run([str(item) for item in cmd], request)


class DotsBackend(CliBackend):
    name = "dots"

    def synthesize(self, request: TTSRequest) -> TTSResult:
        if not request.ref_audio:
            raise ValueError("dots requires ref_audio for voice cloning")
        runner = Path(__file__).with_name("dots_clone_runner.py")
        cmd = _cmd_prefix(self.cfg.get("command_prefix", [self.root / ".venv/bin/python", runner]))
        cmd += [
            "--model", str(self.cfg.get("model", "weights")),
            "--text", request.text,
            "--ref-audio", _project_path(request.ref_audio) or request.ref_audio,
            "--output", _project_path(request.output_path) or request.output_path,
            "--language", str(request.language if request.language != "auto" else request.target_language).upper(),
            "--max-audio-tokens", str(self.cfg.get("max_audio_tokens", 30)),
            "--num-steps", str(self.cfg.get("num_steps", 3)),
            "--seed", str(self.cfg.get("seed", 42)),
        ]
        # dots.tts does not implement native duration fitting. Its runner only
        # converts this value into a token ceiling; short subtitle windows can
        # therefore cut or hallucinate the final words. Generate to natural EOS
        # and let VideoLingo's measured post-fit enforce the timeline instead.
        if request.target_duration and bool(self.cfg.get("use_target_duration_token_cap", False)):
            cmd += ["--target-duration", f"{request.target_duration:.3f}"]
        return self._run([str(item) for item in cmd], request)


class MossTTSBackend(CliBackend):
    name = "moss"

    def synthesize(self, request: TTSRequest) -> TTSResult:
        if not request.ref_audio:
            raise ValueError("moss requires ref_audio for voice cloning")
        language = request.language if request.language != "auto" else request.target_language
        language = LANGUAGE_NAMES.get(str(language).lower(), language if language != "auto" else "English")
        cmd = _cmd_prefix(self.cfg.get("command_prefix", ["python3", "-m", "mlx_moss_tts_local.cli"]))
        max_tokens = int(self.cfg.get("max_tokens", 2048))
        if request.target_duration and _as_bool(
            self.cfg.get("use_target_duration_token_cap", True), True
        ):
            tokens_per_second = float(self.cfg.get("tokens_per_second", 24.0))
            token_buffer = int(self.cfg.get("token_buffer", 32))
            min_tokens = int(self.cfg.get("min_tokens", 32))
            dynamic_cap = max(
                min_tokens,
                int(math.ceil(request.target_duration * tokens_per_second + token_buffer)),
            )
            max_tokens = min(max_tokens, dynamic_cap)
        cmd += [
            "--model", str(self.cfg.get("model")),
            "--text", request.text,
            "--language", str(language),
            "--ref-audio", _project_path(request.ref_audio) or request.ref_audio,
            "--output", _project_path(request.output_path) or request.output_path,
            "--max-tokens", str(max_tokens),
        ]
        if request.ref_text:
            cmd += ["--ref-text", request.ref_text]
        return self._run(cmd, request)


class MingTTSBackend(CliBackend):
    name = "ming"

    def synthesize(self, request: TTSRequest) -> TTSResult:
        if not request.ref_audio:
            raise ValueError("ming requires ref_audio for speaker cloning")
        cmd = _cmd_prefix(
            self.cfg.get(
                "command_prefix",
                ["python3", str(Path(__file__).with_name("ming_tts_runner.py"))],
            )
        )
        max_steps = int(self.cfg.get("max_steps", 200))
        if request.target_duration and _as_bool(
            self.cfg.get("use_target_duration_step_cap", True), True
        ):
            steps_per_second = float(self.cfg.get("steps_per_second", 3.125))
            step_buffer = int(self.cfg.get("step_buffer", 8))
            max_steps = min(
                max_steps,
                max(8, int(math.ceil(request.target_duration * steps_per_second + step_buffer))),
            )
        cmd += [
            "--model-dir",
            str(self.cfg.get("model_dir", "mlx_models/Ming-omni-tts-16.8B-A3B-bf16")),
            "--text",
            request.text,
            "--ref-audio",
            _project_path(request.ref_audio) or request.ref_audio,
            "--output",
            _project_path(request.output_path) or request.output_path,
            "--prompt",
            str(
                self.cfg.get(
                    "prompt", "Please generate speech based on the following description.\n"
                )
            ),
            "--max-steps",
            str(max_steps),
            "--cfg",
            str(self.cfg.get("cfg", 2.0)),
            "--seed",
            str(self.cfg.get("seed", 42)),
            "--dtype",
            str(self.cfg.get("dtype", "auto")),
        ]
        if request.ref_text:
            cmd += ["--ref-text", request.ref_text]
        return self._run([str(item) for item in cmd], request)


class Zonos2Backend:
    name = "zonos2"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.url = str(cfg.get("api_url", "http://127.0.0.1:1920")).rstrip("/")
        self.timeout = int(cfg.get("timeout_seconds", 600))

    def synthesize(self, request: TTSRequest) -> TTSResult:
        if not request.ref_audio:
            raise ValueError("zonos2 requires ref_audio for voice cloning")
        import numpy as np
        import soundfile as sf

        started = time.perf_counter()
        ref = _audio_as_wav_bytes(_project_path(request.ref_audio) or request.ref_audio)
        language = request.language if request.language != "auto" else request.target_language
        payload = {
            "text": request.text,
            "language": language if language != "auto" else "en_us",
            "text_normalization": str(language).lower().startswith("en"),
            "max_tokens": int(self.cfg.get("max_tokens", 1024)),
            "speaker_audio_base64": b64encode(ref).decode("ascii"),
        }
        response = requests.post(f"{self.url}/tts/generate", json=payload, timeout=self.timeout)
        response.raise_for_status()
        sample_rate = int(response.headers.get("X-Audio-Sample-Rate", 44100))
        audio = np.frombuffer(response.content, dtype=np.float32)
        output = Path(_project_path(request.output_path) or request.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, audio, sample_rate)
        duration = len(audio) / sample_rate if sample_rate else 0.0
        elapsed = max(time.perf_counter() - started, 0.001)
        return TTSResult(str(output), self.name, duration, elapsed / duration if duration else None)

    def health(self) -> BackendHealth:
        try:
            response = requests.get(f"{self.url}/health", timeout=3)
            return BackendHealth(self.name, response.ok, f"HTTP {response.status_code}: {self.url}")
        except requests.RequestException as exc:
            return BackendHealth(self.name, False, f"service unavailable: {exc}")


class MlxTTSRouter:
    def __init__(self):
        self.backends = {
            "indextts2": IndexTTS2Backend(_load_backend_config("indextts2")),
            "omnivoice": OmniVoiceBackend(_load_backend_config("omnivoice")),
            "qwen3_tts": QwenTTSBackend(_load_backend_config("qwen3_tts")),
            "voxcpm2": VoxCPM2Backend(_load_backend_config("voxcpm2")),
            "higgs": HiggsBackend(_load_backend_config("higgs")),
            "dots": DotsBackend(_load_backend_config("dots")),
            "zonos2": Zonos2Backend(_load_backend_config("zonos2")),
            "moss": MossTTSBackend(_load_backend_config("moss")),
            "ming": MingTTSBackend(_load_backend_config("ming")),
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
        if language in {"zh", "zh-CN", "chinese"} or looks_chinese(text):
            return "omnivoice"
        if _as_bool(load_key("mlx_tts.router.allow_voxcpm2_auto", False), False):
            return "voxcpm2"
        return "indextts2"

    def synthesize(self, request: TTSRequest) -> TTSResult:
        backend_name = self.select_backend(request)
        if request.metadata.get("requires_separate_emotion") and backend_name != "indextts2":
            raise ValueError(
                f"Backend {backend_name} cannot preserve separate emotion conditioning; "
                "refusing a silent speaker-only fallback"
            )
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


def synthesize_indextts2_batch(requests: list[TTSRequest]) -> list[TTSResult]:
    return IndexTTS2Backend(_load_backend_config("indextts2")).synthesize_batch(requests)


def write_router_plan(path: str = "output/audio/mlx_tts_router_plan.json") -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "default_backend": load_key("mlx_tts.default_backend", "auto"),
        "backends": list_backend_status(),
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
