"""
VoxCPM TTS Module for VideoLingo-plus
======================================
Calls the VoxCPM REST API (OpenAI-compatible /v1/audio/speech endpoint)
to synthesize speech with dynamic voice cloning per segment.

Auto-start: if the API server is not running, this module will launch it
automatically using: uv run python api_server.py --port 8809 --preload
(inside /Users/vanch/VoxCPM)
"""
import os
import sys
import base64
import subprocess
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.config_utils import load_key
from core.all_tts_functions.tts_utils import (
    get_reference_audio_path,
    get_prompt_text_from_df,
    ensure_output_dir,
)
from rich import print as rprint

# Default VoxCPM REST API endpoint (api_server.py default port)
DEFAULT_BASE_URL = "http://127.0.0.1:8809"
API_PATH = "/v1/audio/speech"

# VoxCPM project directory (where api_server.py and uv.lock live)
VOXCPM_DIR = os.path.expanduser("~/VoxCPM")

# Module-level flag: True once we've confirmed the server is up this session
_server_ready: bool = False
_server_proc: subprocess.Popen | None = None  # reference to auto-launched process


def _is_server_up(base_url: str, timeout: float = 3.0) -> bool:
    """Return True if the VoxCPM health endpoint responds OK."""
    import requests
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def ensure_server_running() -> None:
    """
    Check whether the VoxCPM API server is up.
    If not, launch it automatically and wait up to 90 s for model load.
    Safe to call multiple times — skips the check after first success.
    """
    global _server_ready, _server_proc
    if _server_ready:
        return

    base_url = get_base_url()

    # ── Fast path: already up ─────────────────────────────────────────────────
    if _is_server_up(base_url):
        rprint("[green]✅ VoxCPM API server is already running.[/green]")
        _server_ready = True
        return

    # ── Launch the server ─────────────────────────────────────────────────────
    if not os.path.isdir(VOXCPM_DIR):
        raise FileNotFoundError(
            f"VoxCPM project directory not found: {VOXCPM_DIR}\n"
            "Please clone the repo or adjust VOXCPM_DIR in voxcpm_tts.py."
        )

    rprint(f"[yellow]🚀 VoxCPM API server not running — launching automatically...[/yellow]")
    rprint(f"[dim]   cwd: {VOXCPM_DIR}[/dim]")

    # Extract port from base_url (default 8809)
    try:
        port = int(base_url.rsplit(":", 1)[-1])
    except ValueError:
        port = 8809

    log_path = os.path.join(VOXCPM_DIR, "api_server.log")
    log_file = open(log_path, "a")  # append mode so history is preserved

    _server_proc = subprocess.Popen(
        ["uv", "run", "python", "api_server.py", "--port", str(port), "--preload"],
        cwd=VOXCPM_DIR,
        stdout=log_file,
        stderr=log_file,
        # Detach from current process group so it survives if st.py reruns
        start_new_session=True,
    )
    rprint(f"[dim]   PID: {_server_proc.pid}  |  log: {log_path}[/dim]")

    # ── Wait for model to load (up to 90 s) ──────────────────────────────────
    max_wait = 90
    poll_interval = 3
    elapsed = 0
    rprint(f"[cyan]⏳ Waiting for VoxCPM model to load (up to {max_wait}s)...[/cyan]")

    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval

        # Check if process crashed immediately
        if _server_proc.poll() is not None:
            raise RuntimeError(
                f"VoxCPM API server exited unexpectedly (code {_server_proc.returncode}).\n"
                f"Check the log for details: {log_path}"
            )

        if _is_server_up(base_url, timeout=2.0):
            rprint(f"[bold green]✅ VoxCPM API server is ready! ({elapsed}s)[/bold green]")
            _server_ready = True
            return

        rprint(f"[dim]   ... still loading ({elapsed}/{max_wait}s)[/dim]")

    # Timeout — leave the process running, but warn
    rprint(
        f"[red]❌ VoxCPM server did not become ready within {max_wait}s.\n"
        f"   Check the log: {log_path}[/red]"
    )
    raise TimeoutError(
        f"VoxCPM API server did not respond within {max_wait}s. "
        f"See {log_path} for details."
    )


def get_base_url() -> str:
    """Get VoxCPM API base URL from config or use default."""
    try:
        url = load_key("voxcpm_tts.api_url")
        if url:
            return url.rstrip("/")
    except (KeyError, TypeError):
        pass
    return DEFAULT_BASE_URL


def _load_cfg(key: str, default):
    """Safe config loader with fallback default."""
    try:
        v = load_key(key)
        if v is not None:
            return v
    except (KeyError, TypeError):
        pass
    return default


def voxcpm_tts(text: str, save_as: str, number: int, task_df, attempt: int = 0):
    """
    Generate TTS audio via VoxCPM REST API with voice cloning.

    Uses the per-segment reference audio extracted in step9 for voice cloning.
    Falls back to a valid reference audio if the primary one is unavailable.

    Args:
        text:     Translated text to synthesize.
        save_as:  Output WAV file path.
        number:   Segment index (used to locate reference audio).
        task_df:  Task DataFrame with 'origin' column (original transcript).
        attempt:  Retry attempt number.
    """
    import requests
    import shutil

    # ── 0. Auto-start API server if needed ────────────────────────────────────
    ensure_server_running()

    # ── 1. Load configuration ──────────────────────────────────────────────────
    voice        = _load_cfg("voxcpm_tts.voice", "alloy")
    control      = _load_cfg("voxcpm_tts.control", "")          # custom control instruction
    cfg_value    = _load_cfg("voxcpm_tts.cfg_value", 2.0)
    timesteps    = _load_cfg("voxcpm_tts.inference_timesteps", 10)
    denoise      = _load_cfg("voxcpm_tts.denoise", False)
    clone_mode   = _load_cfg("voxcpm_tts.clone_mode", "dynamic")  # "dynamic" | "none"

    # ── 2. Resolve reference audio for voice cloning ──────────────────────────
    ref_audio_b64 = None
    prompt_text   = None

    if clone_mode == "dynamic":
        try:
            spk_audio_path, fallback_number = get_reference_audio_path(number)
            ref_number = fallback_number if fallback_number is not None else number
            prompt_text = get_prompt_text_from_df(task_df, ref_number, fallback_text="")

            with open(spk_audio_path, "rb") as f:
                ref_audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            rprint(
                f"[cyan]🎙️ VoxCPM: '{text[:45]}' "
                f"[ref:{os.path.basename(spk_audio_path)}][/cyan]"
            )
        except FileNotFoundError as e:
            rprint(f"[yellow]⚠️ VoxCPM: no reference audio found ({e}), using preset voice '{voice}'[/yellow]")

    else:
        rprint(f"[cyan]🎙️ VoxCPM: '{text[:55]}' [voice:{voice}][/cyan]")

    # ── 3. Build request payload ───────────────────────────────────────────────
    payload: dict = {
        "model": "voxcpm2",
        "input": text,
        "voice": voice,
        "speed": 1.0,
        "response_format": "wav",
        "cfg_value": float(cfg_value),
        "inference_timesteps": int(timesteps),
        "denoise": bool(denoise),
    }

    # Custom control instruction overrides voice preset
    if control and control.strip():
        payload["control"] = control.strip()

    # Attach reference audio for voice cloning
    if ref_audio_b64:
        payload["reference_audio"] = ref_audio_b64
        if prompt_text and prompt_text.strip():
            payload["prompt_text"] = prompt_text.strip()

    # ── 4. Call API ────────────────────────────────────────────────────────────
    ensure_output_dir(save_as)
    base_url = get_base_url()
    url = f"{base_url}{API_PATH}"

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()

        # Write raw WAV bytes directly
        with open(save_as, "wb") as out:
            out.write(resp.content)

        # Quick sanity check
        if os.path.getsize(save_as) < 1000:
            raise Exception(f"VoxCPM returned suspiciously small file ({os.path.getsize(save_as)} bytes)")

        gen_time = resp.headers.get("X-Generation-Time", "?")
        audio_dur = resp.headers.get("X-Audio-Duration", "?")
        rprint(f"[green]✅ VoxCPM saved: {save_as}  [{gen_time}s gen / {audio_dur}s audio][/green]")

    except requests.exceptions.ConnectionError:
        raise Exception(
            f"Cannot connect to VoxCPM API at {url}. "
            "Start the server with: "
            "cd /Users/vanch/VoxCPM && python api_server.py --port 8809"
        )
    except requests.exceptions.HTTPError as e:
        # Try to extract error detail from JSON response
        try:
            detail = resp.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        raise Exception(f"VoxCPM API error {resp.status_code}: {detail}")
    except Exception as e:
        rprint(f"[red]❌ VoxCPM error: {e}[/red]")
        raise


if __name__ == "__main__":
    # Quick smoke test
    print("Running VoxCPM REST API test...")
    test_text = "你好，这是通过VoxCPM REST API生成的语音测试。"
    test_save = "/tmp/test_voxcpm_rest.wav"
    try:
        voxcpm_tts(test_text, test_save, 0, None)
        print(f"✅ Success: {test_save}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("Ensure VoxCPM API server is running: python /Users/vanch/VoxCPM/api_server.py --port 8809")
