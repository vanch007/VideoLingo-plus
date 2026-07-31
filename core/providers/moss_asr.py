from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config_utils import load_key


DEFAULT_ROOT = "/Users/vanch/mlx-MOSS-Transcribe-Diarize"
DEFAULT_MODEL = "vanch007/mlx-MOSS-Transcribe-Diarize-8bit"
DEFAULT_LOCAL_MODEL = (
    "/Users/vanch/mlx-MOSS-Transcribe-Diarize/pretrained/"
    "mlx-moss-transcribe-diarize-8bit"
)
DEFAULT_PYTHON = "/Users/vanch/.codex/envs/bluestone-mlx-audio/bin/python"
_TOLERANT_SEGMENT_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]\[+\s*(?P<speaker>S\d+)\]\s*"
    r"(?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]",
    re.DOTALL,
)


@dataclass(frozen=True)
class MossASRHealth:
    ok: bool
    detail: str
    root: str
    model: str
    python: str


@dataclass(frozen=True)
class MossASRRun:
    segments: list[dict[str, Any]]
    output_dir: str
    summary: dict[str, Any]


def _config() -> dict[str, Any]:
    return dict(load_key("moss_asr", {}) or {})


def _path(value: Any, default: str) -> str:
    return str(Path(str(value or default)).expanduser())


def _python_command(cfg: dict[str, Any]) -> list[str]:
    raw = cfg.get("python", DEFAULT_PYTHON)
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return shlex.split(str(raw))


def resolve_model(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or _config()
    configured = str(cfg.get("model", DEFAULT_LOCAL_MODEL) or DEFAULT_LOCAL_MODEL)
    local = Path(configured).expanduser()
    if local.exists():
        return str(local)
    fallback = Path(DEFAULT_LOCAL_MODEL)
    if fallback.exists():
        return str(fallback)
    return str(cfg.get("model_repo", DEFAULT_MODEL) or DEFAULT_MODEL)


def moss_asr_health() -> MossASRHealth:
    cfg = _config()
    root = _path(cfg.get("root"), DEFAULT_ROOT)
    python_cmd = _python_command(cfg)
    python = python_cmd[0] if python_cmd else ""
    model = resolve_model(cfg)
    missing: list[str] = []
    if not Path(root).is_dir():
        missing.append(f"root:{root}")
    if not python or not Path(python).is_file():
        missing.append(f"python:{python or 'empty'}")
    model_path = Path(model).expanduser()
    if model_path.is_absolute() and not (model_path / "model.safetensors").is_file():
        missing.append(f"weights:{model}")
    detail = "ready" if not missing else "missing " + ", ".join(missing)
    return MossASRHealth(not missing, detail, root, model, python)


def recover_segments(raw_transcript: str) -> list[dict[str, Any]]:
    """Recover common near-valid MOSS timestamp output rejected by strict export.

    Some short clips produce ``[0.0][[S01] text[1.2]``. The speech and timing
    are usable, so VideoLingo accepts repeated left brackets while keeping the
    original transcript and a separate recovery artifact for audit.
    """
    recovered: list[dict[str, Any]] = []
    for index, match in enumerate(_TOLERANT_SEGMENT_RE.finditer(raw_transcript), start=1):
        start = float(match.group("start"))
        end = float(match.group("end"))
        text = " ".join(match.group("text").split())
        if not text or end < start:
            continue
        recovered.append(
            {
                "id": f"recovered_{index:04d}",
                "start": start,
                "end": end,
                "speaker": match.group("speaker"),
                "text": text,
            }
        )
    return recovered


def run_moss_asr(
    audio_path: str,
    *,
    output_root: str | None = None,
    purpose: str = "transcribe",
) -> MossASRRun:
    cfg = _config()
    health = moss_asr_health()
    if not health.ok:
        raise RuntimeError(f"MOSS ASR is not ready: {health.detail}")

    audio = str(Path(audio_path).expanduser().resolve())
    base = Path(output_root or cfg.get("output_root", "output/log/moss_asr")).expanduser()
    run_dir = base / f"{purpose}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)

    max_new_tokens = int(
        cfg.get("diarization_max_new_tokens", 8192)
        if purpose == "speaker_diarization"
        else cfg.get("max_new_tokens", 2048)
    )
    cmd = _python_command(cfg) + [
        "-m",
        "moss_transcribe_diarize.mlx.cli",
        audio,
        "--model",
        health.model,
        "--out-dir",
        str(run_dir.resolve()),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    env = os.environ.copy()
    configured_pythonpath = str(cfg.get("pythonpath", health.root) or health.root)
    env["PYTHONPATH"] = configured_pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        cmd,
        cwd=health.root,
        env=env,
        capture_output=True,
        text=True,
        timeout=int(cfg.get("timeout_seconds", 1800)),
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-1600:]
        raise RuntimeError(f"MOSS ASR failed with code {result.returncode}: {detail}")

    segments_path = run_dir / "segments.json"
    if not segments_path.is_file():
        raise RuntimeError(f"MOSS ASR did not create segments.json: {run_dir}")
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    if not isinstance(segments, list):
        raise RuntimeError(f"MOSS ASR segments.json is not a list: {segments_path}")
    recovered = False
    if not segments:
        raw_path = run_dir / "raw_transcript.txt"
        raw_transcript = raw_path.read_text(encoding="utf-8") if raw_path.is_file() else ""
        segments = recover_segments(raw_transcript)
        if segments:
            recovered = True
            (run_dir / "videolingo_recovered_segments.json").write_text(
                json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    try:
        summary = json.loads(result.stdout)
    except json.JSONDecodeError:
        summary = {"stdout": result.stdout.strip()[-1600:]}
    summary["command"] = cmd
    summary["output_dir"] = str(run_dir.resolve())
    summary["videolingo_segments"] = len(segments)
    summary["tolerant_recovery"] = recovered
    (run_dir / "videolingo_run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return MossASRRun(segments=segments, output_dir=str(run_dir.resolve()), summary=summary)


def transcript_from_segments(segments: list[dict[str, Any]]) -> str:
    return " ".join(
        str(segment.get("text", "")).strip()
        for segment in segments
        if str(segment.get("text", "")).strip()
    )
