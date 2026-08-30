from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path
from typing import Any

import librosa
import pandas as pd
import stable_whisper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.providers.quality import content_similarity


BACKENDS = (
    "indextts2",
    "indextts2_v25",
    "omnivoice",
    "qwen3_tts",
    "voxcpm2",
    "higgs",
    "dots",
    "zonos2",
    "moss",
    "ming",
)


def _expected_text(value: Any) -> str:
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return " ".join(str(item).strip() for item in parsed if str(item).strip())
        except (SyntaxError, ValueError):
            pass
    return str(value or "").strip()


def _transcribe(model: Any, path: Path) -> str:
    audio, _ = librosa.load(path, sr=16000, mono=True)
    result = model.transcribe(audio, language="en", verbose=None)
    text = str(getattr(result, "text", "") or "").strip()
    if not text and getattr(result, "segments", None):
        text = " ".join(str(segment.text).strip() for segment in result.segments).strip()
    return text


def _locations(run_dir: Path, backend: str) -> tuple[Path, Path, Path | None]:
    root = run_dir / "output/comparisons/mlx_tts_8_models"
    if backend == "indextts2":
        return (
            root / "baseline/tts_tasks_indextts2.xlsx",
            run_dir / "output/audio/segs",
            run_dir / "output/audio/temp",
        )
    # Ming was first generated with a non-official prompt; audit the
    # recommended official-prompt render when it is present.
    workspace_root = root / ("ming_official_prompt" if backend == "ming" else backend)
    workspace = workspace_root / "workspace/output/audio"
    return workspace / "tts_tasks.xlsx", workspace / "segs", workspace / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all frozen-script MLX TTS comparison segments")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--threshold", type=float, default=0.82)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    model = stable_whisper.load_mlx_whisper(args.model)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "model": args.model,
        "threshold": args.threshold,
        "run_dir": str(run_dir),
        "backends": {},
    }

    for backend in BACKENDS:
        tasks_path, fitted_dir, raw_dir = _locations(run_dir, backend)
        tasks = pd.read_excel(tasks_path)
        rows: list[dict[str, Any]] = []
        for _, row in tasks.iterrows():
            number = int(row["number"])
            expected = _expected_text(row.get("lines", row.get("text", "")))
            fitted_path = fitted_dir / f"{number}_0.wav"
            fitted_text = _transcribe(model, fitted_path)
            fitted_score = content_similarity(expected, fitted_text)
            item: dict[str, Any] = {
                "number": number,
                "expected": expected,
                "fitted_audio": str(fitted_path),
                "fitted_transcript": fitted_text,
                "fitted_score": round(fitted_score, 6),
                "flagged": fitted_score < args.threshold,
            }
            if item["flagged"] and raw_dir is not None:
                raw_path = (
                    raw_dir / f"{number}_0_temp.wav"
                    if backend == "indextts2"
                    else raw_dir / f"{number}_0.wav"
                )
                if raw_path.is_file():
                    raw_text = _transcribe(model, raw_path)
                    raw_score = content_similarity(expected, raw_text)
                    item.update(
                        {
                            "raw_audio": str(raw_path),
                            "raw_transcript": raw_text,
                            "raw_score": round(raw_score, 6),
                            "failure_stage": (
                                "post_fit_or_short_window"
                                if raw_score >= args.threshold and fitted_score < args.threshold
                                else "generation_or_asr_uncertain"
                            ),
                        }
                    )
            rows.append(item)
        scores = [float(item["fitted_score"]) for item in rows]
        flagged = [item for item in rows if item["flagged"]]
        report["backends"][backend] = {
            "rows": len(rows),
            "flagged": len(flagged),
            "minimum_score": min(scores) if scores else None,
            "average_score": sum(scores) / len(scores) if scores else None,
            "items": rows,
        }
        print(f"{backend}: {len(flagged)}/{len(rows)} flagged", flush=True)

    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
