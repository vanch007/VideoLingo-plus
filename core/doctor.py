import importlib
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_utils import get_env_names, load_key
from core.step_checker import STEP_CHECKPOINTS, is_step_completed


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _run(cmd: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        output = (result.stdout or result.stderr).strip().splitlines()
        return result.returncode == 0, output[0] if output else "no output"
    except Exception as exc:
        return False, str(exc)


def _check_import(module_name: str) -> CheckResult:
    try:
        importlib.import_module(module_name)
        return CheckResult(f"import:{module_name}", True, "ok")
    except Exception as exc:
        return CheckResult(f"import:{module_name}", False, f"{type(exc).__name__}: {exc}")


def _check_port(name: str, host: str, port: int) -> CheckResult:
    try:
        with socket.create_connection((host, port), timeout=2):
            return CheckResult(name, True, f"{host}:{port} reachable")
    except OSError as exc:
        return CheckResult(name, False, f"{host}:{port} unreachable: {exc}")


def _check_omlx() -> list[CheckResult]:
    try:
        from core.providers.omlx import list_omlx_models, smoke_chat

        models = list_omlx_models(timeout=5)
        checks = [CheckResult("omlx-models", bool(models), f"{len(models)} models discovered")]
        if models:
            smoke = smoke_chat(timeout=20)
            checks.append(CheckResult("omlx-chat", bool(smoke.get("content")), f"{smoke['model']}: {smoke.get('content', '')[:80]}"))
        return checks
    except Exception as exc:
        return [CheckResult("omlx", False, f"{type(exc).__name__}: {exc}")]


def _check_mlx_tts() -> list[CheckResult]:
    try:
        from core.providers.mlx_tts import list_backend_status

        return [
            CheckResult(f"mlx-tts:{item['name']}", bool(item["ok"]), str(item["detail"]))
            for item in list_backend_status()
        ]
    except Exception as exc:
        return [CheckResult("mlx-tts", False, f"{type(exc).__name__}: {exc}")]


def _check_moss_asr() -> CheckResult:
    try:
        from core.providers.moss_asr import moss_asr_health

        health = moss_asr_health()
        return CheckResult("mlx-asr:moss", health.ok, health.detail)
    except Exception as exc:
        return CheckResult("mlx-asr:moss", False, f"{type(exc).__name__}: {exc}")


def _check_nemotron_diarization() -> CheckResult:
    try:
        from core.providers.nemotron_diarization import nemotron_diarization_health

        health = nemotron_diarization_health()
        return CheckResult("diarization:nemotron-mlx", health.ok, health.detail)
    except Exception as exc:
        return CheckResult("diarization:nemotron-mlx", False, f"{type(exc).__name__}: {exc}")


def run_checks(include_services: bool = True) -> list[CheckResult]:
    checks: list[CheckResult] = []

    ok, detail = _run(["ffmpeg", "-version"])
    checks.append(CheckResult("ffmpeg", ok, detail))

    ok, detail = _run([sys.executable, "-m", "pip", "check"])
    if not ok and "demucs" in detail and "torchaudio" in detail:
        ok = True
        detail = f"known non-blocking Demucs/Torchaudio metadata conflict: {detail}"
    checks.append(CheckResult("pip-check", ok, detail))

    checks.append(CheckResult("python", True, sys.version.split()[0]))
    asr_runtime = load_key("whisper.runtime", "stable-ts")
    if asr_runtime == "stable-ts":
        checks.append(_check_import("stable_whisper"))
    elif asr_runtime == "local":
        checks.append(_check_import("whisperx"))
    else:
        checks.append(CheckResult("source-asr", False, f"ineligible runtime: {asr_runtime}"))
    if load_key("dubbing_quality.asr_readback_backend", "moss-mlx") == "moss-mlx":
        checks.append(_check_moss_asr())
    diar_backend = str(load_key("speaker_diarization.backend", "moss-mlx")).strip().lower()
    shadow_backend = str(load_key("speaker_diarization.shadow_backend", "") or "").strip().lower()
    if "nemotron-mlx" in (diar_backend, shadow_backend):
        checks.append(_check_nemotron_diarization())
    if "moss-mlx" in (diar_backend, shadow_backend) and load_key("dubbing_quality.asr_readback_backend", "moss-mlx") != "moss-mlx":
        checks.append(_check_moss_asr())
    for module in ("openai", "streamlit"):
        checks.append(_check_import(module))

    output_ok = os.path.isdir("output") or os.access(".", os.W_OK)
    checks.append(CheckResult("output-writable", output_ok, "output exists or project root is writable"))

    tts_method = load_key("tts_method", "mlx_router")
    llm_provider = load_key("llm.provider", "openai_compatible")
    secret_requirements = {
        "api.key": llm_provider == "openai_compatible",
        "llm.providers.omlx.api_key": False,
        "hf_token": False,
    }
    for key, required in secret_requirements.items():
        envs = get_env_names(key)
        env_hit = any(os.environ.get(name) for name in envs)
        try:
            configured = bool(load_key(key, ""))
        except Exception:
            configured = False
        ok = env_hit or configured or not required
        label = "required" if required else "optional"
        checks.append(CheckResult(f"secret:{key}", ok, f"{label}; env={','.join(envs) or 'n/a'}"))

    if include_services:
        if llm_provider == "omlx":
            checks.extend(_check_omlx())
        if tts_method == "mlx_router" or tts_method.startswith("mlx_"):
            checks.extend(_check_mlx_tts())

    completed = [name for name in STEP_CHECKPOINTS if is_step_completed(name)]
    checks.append(CheckResult("step-checkpoints", True, f"{len(completed)}/{len(STEP_CHECKPOINTS)} completed"))
    checks.append(CheckResult("ffmpeg-path", bool(shutil.which("ffmpeg")), shutil.which("ffmpeg") or "missing"))
    return checks


def print_report() -> int:
    failed = 0
    for check in run_checks():
        status = "OK" if check.ok else "FAIL"
        if not check.ok:
            failed += 1
        print(f"[{status}] {check.name}: {check.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(print_report())
