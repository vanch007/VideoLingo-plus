from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from pydub import AudioSegment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.audio_speed import adjust_audio_speed, trim_edge_silence
from core.dubbing_quality import normalize_lines, parse_list
from core.providers.asr_readback import verify_tasks_df
from core.providers.contracts import TTSRequest
from core.providers.mlx_tts import synthesize_with_mlx_router


METHODS = {
    "omnivoice": "mlx_omnivoice",
    "qwen3_tts": "mlx_qwen3_tts",
    "voxcpm2": "mlx_voxcpm2",
    "higgs": "mlx_higgs_audio",
    "dots": "mlx_dots_tts",
    "zonos2": "mlx_zonos2",
    "moss": "mlx_moss_tts",
}


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def _prepare_workspace(run_dir: Path, output_dir: Path) -> Path:
    workspace = output_dir / "workspace"
    output = workspace / "output"
    (output / "audio" / "raw").mkdir(parents=True, exist_ok=True)
    (output / "audio" / "temp").mkdir(parents=True, exist_ok=True)
    (output / "audio" / "segs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "config.yaml", workspace / "config.yaml")
    for relative in (
        "output/source.mp4",
        "output/audio/background.mp3",
        "output/audio/vocals.mp3",
    ):
        source = run_dir / relative
        if source.exists():
            _link_or_copy(source, workspace / relative)
    refers = run_dir / "output/audio/refers"
    _link_or_copy(refers, output / "audio" / "refers")
    return workspace


def _target_rows(tasks: pd.DataFrame, numbers: set[int] | None) -> pd.DataFrame:
    selected = tasks.copy()
    if numbers:
        selected = selected[selected["number"].astype(int).isin(numbers)].copy()
    return selected.reset_index(drop=True)


def _clear_quality_columns(tasks: pd.DataFrame) -> None:
    for column in (
        "asr_transcript",
        "asr_content_score",
        "asr_leakage_score",
        "asr_status",
        "asr_line_results",
        "asr_fingerprint",
        "asr_language",
        "asr_backend",
        "tts_generation_fingerprint",
    ):
        if column in tasks.columns:
            tasks[column] = None


def _request(
    backend: str,
    row: pd.Series,
    text: str,
    output: Path,
    ref_audio: Path,
    target_duration: float,
) -> TTSRequest:
    return TTSRequest(
        text=text,
        output_path=str(output.resolve()),
        number=int(row["number"]),
        language="en",
        source_language="zh",
        target_language="en",
        ref_audio=str(ref_audio.resolve()),
        ref_text=str(row.get("origin", "") or "").strip() or None,
        speaker_id=str(row.get("speaker", "") or "") or None,
        target_duration=target_duration,
        task_row=row.to_dict(),
        metadata={"backend": backend, "comparison_run": True},
    )


def _fit_to_window(raw_path: Path, temp_path: Path, segment_path: Path, window: float) -> dict:
    audio = trim_edge_silence(AudioSegment.from_file(raw_path))
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(temp_path, format="wav")
    natural_duration = len(audio) / 1000.0
    if natural_duration <= 0:
        raise RuntimeError(f"empty audio after edge trim: {raw_path}")
    # Leave a small guard band for atempo's codec-frame rounding. The guard is
    # filled with silence afterwards, so spoken content is preserved intact.
    fit_target = max(min(window * 0.98, window - 0.04), 0.001) if window > 0 else window
    speed_factor = max(1.0, natural_duration / fit_target) if window > 0 else 1.0
    segment_path.parent.mkdir(parents=True, exist_ok=True)
    if speed_factor > 1.001:
        adjust_audio_speed(str(temp_path), str(segment_path), speed_factor)
        fitted = AudioSegment.from_file(segment_path)
    else:
        fitted = audio
    window_ms = max(1, int(round(window * 1000)))
    if len(fitted) < window_ms:
        fitted += AudioSegment.silent(duration=window_ms - len(fitted), frame_rate=fitted.frame_rate)
    elif len(fitted) > window_ms:
        # ffmpeg's atempo output may be a few milliseconds longer because of
        # codec/frame rounding. Correct that residual with another speed pass;
        # never truncate spoken audio at the subtitle boundary.
        correction_path = segment_path.with_name(f"{segment_path.stem}_corrected.wav")
        for _ in range(3):
            if len(fitted) <= window_ms:
                break
            correction_target_ms = max(
                1, int(round(min(window_ms * 0.98, window_ms - 20)))
            )
            correction = len(fitted) / correction_target_ms
            fitted.export(segment_path, format="wav")
            adjust_audio_speed(str(segment_path), str(correction_path), correction)
            fitted = AudioSegment.from_file(correction_path)
            correction_path.unlink(missing_ok=True)
            speed_factor *= correction
        if len(fitted) > window_ms:
            raise RuntimeError(
                f"unable to fit audio without clipping: {len(fitted)}ms > {window_ms}ms"
            )
    fitted.export(segment_path, format="wav")
    return {
        "natural_duration": round(natural_duration, 4),
        "window_duration": round(window, 4),
        "speed_factor": round(speed_factor, 4),
        "fitted_duration": round(len(fitted) / 1000.0, 4),
    }


def generate_backend(
    backend: str,
    run_dir: Path,
    output_dir: Path,
    baseline_tasks: Path,
    numbers: set[int] | None,
    resume: bool,
) -> tuple[Path, dict]:
    workspace = _prepare_workspace(run_dir, output_dir)
    tasks = _target_rows(pd.read_excel(baseline_tasks), numbers)
    _clear_quality_columns(tasks)
    tasks["tts_method"] = METHODS[backend]
    tasks["tts_backend"] = backend
    report_rows: list[dict] = []
    started = time.perf_counter()
    old_cwd = Path.cwd()
    os.chdir(workspace)
    try:
        for idx, row in tasks.iterrows():
            number = int(row["number"])
            lines = normalize_lines(row.get("lines", row.get("text", "")))
            times = parse_list(row.get("new_sub_times"))
            if len(lines) != len(times):
                raise RuntimeError(
                    f"row {number} line/timestamp mismatch: {len(lines)} != {len(times)}"
                )
            ref_audio = run_dir / "output/audio/refers" / f"{number}.wav"
            if not ref_audio.is_file():
                raise FileNotFoundError(ref_audio)
            natural_total = 0.0
            row_speed = 1.0
            for line_index, (text, window_pair) in enumerate(zip(lines, times)):
                start, end = map(float, window_pair)
                window = end - start
                raw_path = workspace / "output/audio/raw" / f"{number}_{line_index}.wav"
                temp_path = workspace / "output/audio/temp" / f"{number}_{line_index}_temp.wav"
                segment_path = workspace / "output/audio/segs" / f"{number}_{line_index}.wav"
                generation_elapsed = 0.0
                generated_backend = backend
                if not (resume and raw_path.is_file() and raw_path.stat().st_size >= 1000):
                    request = _request(backend, row, text, raw_path, ref_audio, window)
                    call_started = time.perf_counter()
                    result = synthesize_with_mlx_router(request)
                    generation_elapsed = time.perf_counter() - call_started
                    generated_backend = result.backend
                    if generated_backend != backend:
                        raise RuntimeError(
                            f"backend substitution is forbidden: requested={backend}, got={generated_backend}"
                        )
                fit = _fit_to_window(raw_path, temp_path, segment_path, window)
                natural_total += fit["natural_duration"]
                row_speed = max(row_speed, fit["speed_factor"])
                report_rows.append({
                    "number": number,
                    "line_index": line_index,
                    "speaker": str(row.get("speaker", "")),
                    "text": text,
                    "ref_audio": str(ref_audio),
                    "ref_text": str(row.get("origin", "")),
                    "raw_audio": str(raw_path),
                    "segment_audio": str(segment_path),
                    "backend": generated_backend,
                    "generation_elapsed": round(generation_elapsed, 4),
                    **fit,
                })
            tasks.at[idx, "real_dur"] = natural_total
            tasks.at[idx, "speed_factor"] = row_speed
            tasks.to_excel(workspace / "output/audio/tts_tasks.xlsx", index=False)
            (output_dir / "generation_rows.json").write_text(
                json.dumps(report_rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    finally:
        os.chdir(old_cwd)
    elapsed = time.perf_counter() - started
    generated_duration = sum(float(row["natural_duration"]) for row in report_rows)
    report = {
        "backend": backend,
        "status": "generated",
        "rows": len(tasks),
        "segments": len(report_rows),
        "elapsed_seconds": round(elapsed, 3),
        "generated_duration_seconds": round(generated_duration, 3),
        "wall_rtf": round(elapsed / generated_duration, 4) if generated_duration else None,
        "max_speed_factor": max((row["speed_factor"] for row in report_rows), default=0),
        "workspace": str(workspace),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return workspace, report


def verify_backend(workspace: Path, report_path: Path) -> dict:
    old_cwd = Path.cwd()
    os.chdir(workspace)
    try:
        tasks = pd.read_excel("output/audio/tts_tasks.xlsx")
        verified, summary = verify_tasks_df(tasks, force=True)
        verified.to_excel("output/audio/tts_tasks.xlsx", index=False)
    finally:
        os.chdir(old_cwd)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["asr_readback"] = asdict(summary)
    report["status"] = "verified"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def render_backend(workspace: Path, output_dir: Path, backend: str) -> Path:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    python = "/Users/vanch/pinokio/bin/miniconda/envs/videolingo/bin/python"
    subprocess.run([python, "-m", "core.step11_merge_full_audio"], cwd=workspace, env=env, check=True)
    subprocess.run([python, "-m", "core.step12_merge_dub_to_vid"], cwd=workspace, env=env, check=True)
    source = workspace / "output/AI配音.mp4"
    destination = output_dir / f"AI配音_{backend}.mp4"
    shutil.copy2(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a frozen-script MLX TTS video comparison")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=sorted(METHODS), required=True)
    parser.add_argument("--baseline-tasks", type=Path)
    parser.add_argument("--numbers", help="Comma-separated row numbers for a pilot")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_tasks = (
        args.baseline_tasks.expanduser().resolve()
        if args.baseline_tasks
        else run_dir / "output/comparisons/mlx_tts_8_models/baseline/tts_tasks_indextts2.xlsx"
    )
    numbers = {int(item) for item in args.numbers.split(",")} if args.numbers else None
    workspace, report = generate_backend(
        args.backend, run_dir, output_dir, baseline_tasks, numbers, args.resume
    )
    if args.verify:
        report = verify_backend(workspace, output_dir / "report.json")
    if args.render:
        video = render_backend(workspace, output_dir, args.backend)
        report["video"] = str(video)
        report["status"] = "rendered"
        (output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
