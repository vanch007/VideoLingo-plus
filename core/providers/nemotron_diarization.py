from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config_utils import load_key

DEFAULT_PYTHON = "/Users/vanch/.codex/envs/nemotron-diarization/bin/python"
DEFAULT_MODEL = "mlx-community/Nemotron-3-Diarization-8bit"
DEFAULT_REVISION = "dd8b8ce3d69a80540a7e50150e2beed1c34fee37"
DEFAULT_RUNNER = str(Path(__file__).parent / "nemotron_diarization_runner.py")
DEFAULT_OUTPUT_ROOT = "output/log/nemotron_diarization"


@dataclass(frozen=True)
class NemotronDiarizationHealth:
    ok: bool
    detail: str
    python: str
    model: str
    revision: str


@dataclass(frozen=True)
class NemotronDiarizationRun:
    segments: list[dict[str, Any]]
    output_dir: str
    summary: dict[str, Any]
    probabilities_path: str | None


def _config() -> dict[str, Any]:
    return dict(load_key("nemotron_diarization", {}) or {})


def nemotron_diarization_health() -> NemotronDiarizationHealth:
    cfg = _config()
    python_bin = str(cfg.get("python", DEFAULT_PYTHON))
    model = str(cfg.get("model", DEFAULT_MODEL))
    revision = str(cfg.get("revision", DEFAULT_REVISION))
    runner = str(cfg.get("runner", DEFAULT_RUNNER))

    missing: list[str] = []
    if not Path(python_bin).is_file():
        missing.append(f"python:{python_bin}")
    if not Path(runner).is_file():
        missing.append(f"runner:{runner}")

    if not missing:
        # Check MLX vad availability via quick dry invocation
        try:
            probe = subprocess.run(
                [python_bin, "-c", "import mlx_audio.vad; import mlx.core; import numpy"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if probe.returncode != 0:
                missing.append(f"mlx_audio_import_failed:{probe.stderr.strip()[:100]}")
        except Exception as exc:
            missing.append(f"python_probe_error:{exc}")

    detail = "ready" if not missing else "missing " + ", ".join(missing)
    return NemotronDiarizationHealth(
        ok=not missing,
        detail=detail,
        python=python_bin,
        model=model,
        revision=revision,
    )


def run_nemotron_diarization(
    audio_path: str,
    *,
    output_root: str | None = None,
    preset: str | None = None,
    threshold: float | None = None,
    min_duration: float | None = None,
    merge_gap: float | None = None,
    save_probs: bool = True,
    timeout_seconds: int | None = None,
) -> NemotronDiarizationRun:
    cfg = _config()
    health = nemotron_diarization_health()
    if not health.ok:
        raise RuntimeError(f"Nemotron Diarization is not ready: {health.detail}")

    audio = str(Path(audio_path).expanduser().resolve())
    base = Path(output_root or cfg.get("output_root", DEFAULT_OUTPUT_ROOT)).expanduser()
    run_dir = base / f"diarization_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)

    runner = str(cfg.get("runner", DEFAULT_RUNNER))
    model = str(cfg.get("model", health.model))
    effective_preset = str(preset or cfg.get("preset", "offline"))
    effective_threshold = float(
        threshold if threshold is not None else cfg.get("threshold", 0.5)
    )
    effective_min_duration = float(
        min_duration if min_duration is not None else cfg.get("min_duration", 0.0)
    )
    effective_merge_gap = float(
        merge_gap if merge_gap is not None else cfg.get("merge_gap", 0.0)
    )
    timeout = int(timeout_seconds or cfg.get("timeout_seconds", 1800))

    cmd = [
        health.python,
        runner,
        "--audio",
        audio,
        "--out-dir",
        str(run_dir.resolve()),
        "--model",
        model,
        "--preset",
        effective_preset,
        "--threshold",
        str(effective_threshold),
        "--min-duration",
        str(effective_min_duration),
        "--merge-gap",
        str(effective_merge_gap),
    ]
    if save_probs:
        cmd.append("--save-probs")

    env = os.environ.copy()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-1600:]
        raise RuntimeError(
            f"Nemotron Diarization runner failed with code {result.returncode}: {detail}"
        )

    segments_file = run_dir / "segments.json"
    if not segments_file.is_file():
        raise RuntimeError(
            f"Nemotron Diarization did not create segments.json in {run_dir}"
        )
    segments = json.loads(segments_file.read_text(encoding="utf-8"))
    if not isinstance(segments, list):
        raise RuntimeError(f"Nemotron Diarization segments.json is not a list: {segments_file}")

    summary_file = run_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_file.is_file():
        try:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    if not summary:
        try:
            summary = json.loads(result.stdout.strip().splitlines()[-1])
        except Exception:
            summary = {"stdout": result.stdout.strip()[-1000:]}

    summary["command"] = cmd
    summary["output_dir"] = str(run_dir.resolve())
    summary["backend"] = "nemotron-mlx"
    (run_dir / "run_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    probs_path = str(run_dir / "probabilities.npz") if (run_dir / "probabilities.npz").is_file() else None
    return NemotronDiarizationRun(
        segments=segments,
        output_dir=str(run_dir.resolve()),
        summary=summary,
        probabilities_path=probs_path,
    )
