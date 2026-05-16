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
    for module in ("stable_whisper", "whisperx", "funasr", "edge_tts", "openai", "streamlit"):
        checks.append(_check_import(module))

    output_ok = os.path.isdir("output") or os.access(".", os.W_OK)
    checks.append(CheckResult("output-writable", output_ok, "output exists or project root is writable"))

    tts_method = load_key("tts_method", "edge_tts")
    llm_provider = load_key("llm.provider", "openai_compatible")
    secret_requirements = {
        "api.key": llm_provider == "openai_compatible",
        "llm.providers.omlx.api_key": False,
        "hf_token": False,
        "sf_indextts2.api_key": tts_method == "sf_indextts2",
        "openai_tts.api_key": tts_method == "openai_tts",
        "elevenlabs_tts.api_key": tts_method == "elevenlabs_tts",
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
        if tts_method == "edge_tts":
            service_check = _check_port("cloudflare-edge-tts", "127.0.0.1", 5566)
            if not service_check.ok:
                service_check = CheckResult(
                    "cloudflare-edge-tts",
                    True,
                    "worker unavailable; native edge-tts fallback will be used",
                )
            checks.append(service_check)
        elif tts_method == "voxcpm_tts":
            api_url = load_key("voxcpm_tts.api_url", "http://127.0.0.1:8809")
            port = int(api_url.rstrip("/").split(":")[-1])
            checks.append(_check_port("voxcpm", "127.0.0.1", port))
        elif tts_method in {"mlx_router", "mlx_indextts2", "mlx_omnivoice", "mlx_qwen3_tts", "mlx_voxcpm2"}:
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
