import os
import sys
import hashlib
import json
import logging
import re
import tempfile
import warnings
from typing import Tuple

import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config_utils import load_key
from core.all_whisper_methods.audio_preprocess import get_audio_duration
from core.all_tts_functions.tts_main import tts_main
from core.all_tts_functions.mlx_router import build_mlx_tts_request
from core.all_tts_functions.tts_registry import MLX_ROUTER_BACKENDS, is_local_tts_method
from core.providers.mlx_tts import synthesize_indextts2_batch
from core.audio_speed import adjust_audio_speed as _adjust_audio_speed
from core.audio_speed import build_atempo_filter as _build_atempo_filter
from core.audio_speed import trim_edge_silence as _trim_edge_silence
from core.dubbing_quality import (
    actual_expand_needed,
    actual_rewrite_needed,
    apply_dubbing_budget_columns,
    get_quality_config,
    normalize_lines,
    row_available_duration,
    temp_audio_file_for,
    write_dubbing_eval,
)
from core.dubbing_rewrite import rewrite_task_lines
from core.timing_utils import srt_time_to_seconds
from core.constants import SEGS_DIR, TEMP_DIR, TTS_TASKS_FILE, DEFAULT_WARMUP_SIZE

# Suppress Streamlit warnings
logging.getLogger('streamlit').setLevel(logging.ERROR)

os.environ['TORCHAUDIO_USE_BACKEND_DISPATCHER'] = '1'
warnings.filterwarnings("ignore", message=".*TorchAudio's global backend is now deprecated.*")

console = Console()

TEMP_FILE_TEMPLATE = f"{TEMP_DIR}/{{}}_temp.wav"
OUTPUT_FILE_TEMPLATE = f"{SEGS_DIR}/{{}}.wav"


def _generation_fingerprint(row: dict) -> str:
    number = int(row["number"])
    lines = normalize_lines(row.get("lines", row.get("text", "")))
    audio = []
    for line_index in range(len(lines)):
        path = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
        try:
            stat = os.stat(path)
            audio.append({"path": path, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        except OSError:
            audio.append({"path": path, "missing": True})
    payload = {"number": number, "lines": lines, "audio": audio}
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def mark_tts_generation_checkpoint(tasks_df: pd.DataFrame) -> pd.DataFrame:
    tasks_df["tts_generation_fingerprint"] = [
        _generation_fingerprint(row.to_dict()) for _, row in tasks_df.iterrows()
    ]
    return tasks_df


def tts_generation_checkpoint_valid(tasks_df: pd.DataFrame) -> bool:
    if "tts_generation_fingerprint" not in tasks_df.columns or "real_dur" not in tasks_df.columns:
        return False
    return all(_tts_generation_row_valid(row) for _, row in tasks_df.iterrows())


def _tts_generation_row_valid(row: pd.Series) -> bool:
    stored = row.get("tts_generation_fingerprint")
    return (
        isinstance(stored, str)
        and stored == _generation_fingerprint(row.to_dict())
        and float(row.get("real_dur", 0) or 0) > 0
    )


def build_atempo_filter(speed_factor: float) -> str:
    return _build_atempo_filter(speed_factor)


def adjust_audio_speed(input_file: str, output_file: str, speed_factor: float) -> None:
    _adjust_audio_speed(input_file, output_file, speed_factor)

def remove_silence_from_file(file_path: str, min_silence_len: int = 200, silence_thresh: int = -45) -> None:
    """Remove silence from audio file to optimize duration"""
    if not os.path.exists(file_path):
        return
    try:
        audio = AudioSegment.from_wav(file_path)
        # split_on_silence returns a list of audio chunks (non-silent parts)
        # min_silence_len: Minimum length of silence to be considered (ms)
        # silence_thresh: Threshold for silence (dBFS)
        # keep_silence: Amount of silence to leave at beginning/end of chunks (ms)
        chunks = split_on_silence(
            audio, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh,
            keep_silence=50 
        )
        
        if chunks:
            # Recombine chunks
            output = chunks[0]
            for chunk in chunks[1:]:
                output += chunk
            output.export(file_path, format="wav")
            
            # Log if significant reduction
            original_len = len(audio) / 1000
            new_len = len(output) / 1000
            if original_len - new_len > 0.2:
                 rprint(f"[dim]  ✂️ Silence removed: {original_len:.2f}s -> {new_len:.2f}s[/dim]")
    except Exception as e:
        rprint(f"[yellow]⚠️ Silence removal failed for {file_path}: {e}[/yellow]")

def check_audio(file_path: str) -> None:
    """Check and fix audio file: apply fade-out to prevent popping"""
    if not os.path.exists(file_path):
        return
    try:
        audio = AudioSegment.from_wav(file_path)
        
        # Only apply fade-out to prevent popping
        # Silence trimming is too aggressive and can cause issues
        if len(audio) > 20:
            audio = audio.fade_out(10)
            audio.export(file_path, format="wav")
    except Exception as e:
        rprint(f"[yellow]⚠️ Audio check failed for {file_path}: {e}[/yellow]")


def _export_wav_if_audio_changed(audio: AudioSegment, output_file: str) -> None:
    """Preserve mtime/ASR cache when native fitting produces identical PCM."""
    if os.path.exists(output_file):
        try:
            existing = AudioSegment.from_wav(output_file)
            if (
                existing.frame_rate == audio.frame_rate
                and existing.channels == audio.channels
                and existing.sample_width == audio.sample_width
                and existing.raw_data == audio.raw_data
            ):
                return
        except Exception:
            pass
    audio.export(output_file, format="wav")


def process_row(row: dict, tasks_df: pd.DataFrame) -> Tuple[int, float]:
    """Helper function for processing single row data"""
    number = row['number']
    lines = normalize_lines(row.get('lines', row.get('text', '')))
    target_duration = row_available_duration(row)
    real_dur = 0
    for line_index, line in enumerate(lines):
        temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
        tts_main(
            line,
            temp_file,
            number,
            tasks_df,
            task_row=row,
            line_index=line_index,
            target_duration=target_duration / max(len(lines), 1),
        )
        if bool(load_key("dubbing_quality.trim_generated_silence", False)):
            remove_silence_from_file(temp_file)
        check_audio(temp_file)
        real_dur += get_audio_duration(temp_file)
    return number, real_dur


def process_indextts2_batch(
    tasks_df: pd.DataFrame,
    rows_df: pd.DataFrame | None = None,
) -> list[tuple[int, float]]:
    """Generate all IndexTTS2 subtitle lines while keeping one model resident."""
    requests = []
    row_outputs: list[tuple[int, list[str]]] = []
    generation_rows = tasks_df if rows_df is None else rows_df
    for _, series in generation_rows.iterrows():
        row = series.to_dict()
        number = int(row["number"])
        lines = normalize_lines(row.get("lines", row.get("text", "")))
        outputs = []
        per_line_duration = row_available_duration(row) / max(len(lines), 1)
        for line_index, line in enumerate(lines):
            output = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
            outputs.append(output)
            requests.append(build_mlx_tts_request(
                line,
                output,
                number,
                tasks_df,
                task_row=row,
                target_duration=per_line_duration,
            ))
        row_outputs.append((number, outputs))

    synthesize_indextts2_batch(requests)
    durations = []
    for number, outputs in row_outputs:
        real_duration = 0.0
        for output in outputs:
            if bool(load_key("dubbing_quality.trim_generated_silence", False)):
                remove_silence_from_file(output)
            check_audio(output)
            real_duration += get_audio_duration(output)
        durations.append((number, real_duration))
    return durations


def _batch_regenerate_rewritten_rows(
    tasks_df: pd.DataFrame,
    row_indices: list[int],
) -> None:
    if not row_indices:
        return
    rows_df = tasks_df.loc[row_indices].copy()
    for number, real_dur in process_indextts2_batch(tasks_df, rows_df):
        tasks_df.loc[tasks_df["number"] == number, "real_dur"] = real_dur


def _delete_temp_files_for_row(row: dict) -> None:
    number = int(row["number"])
    lines = normalize_lines(row.get("lines", row.get("text", "")))
    for line_index in range(len(lines)):
        for file_path in (
            TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}"),
            temp_audio_file_for(number, line_index),
            OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}"),
        ):
            if os.path.exists(file_path):
                os.remove(file_path)


def _ensure_rewrite_columns(tasks_df: pd.DataFrame) -> None:
    defaults = {
        "dubbing_rewrite_rounds": 0,
        "rewritten_for_dubbing": False,
        "rewrite_reason": "",
    }
    for column, default in defaults.items():
        if column not in tasks_df.columns:
            tasks_df[column] = default
    tasks_df["rewrite_reason"] = tasks_df["rewrite_reason"].astype(object)


def _write_row_values(tasks_df: pd.DataFrame, idx: int, row: dict) -> None:
    for key, value in row.items():
        if key in tasks_df.columns:
            tasks_df.at[idx, key] = value


def _regenerate_rewritten_row(
    tasks_df: pd.DataFrame,
    idx: int,
    row: dict,
    lines: list[str],
    *,
    rewrite_reason: str,
) -> int:
    _delete_temp_files_for_row(row)
    row["lines"] = lines
    row["text"] = " ".join(lines)
    row["rewritten_for_dubbing"] = True
    row["rewrite_reason"] = rewrite_reason
    row["dubbing_rewrite_rounds"] = int(row.get("dubbing_rewrite_rounds", 0) or 0) + 1
    number, real_dur = process_row(row, tasks_df)
    row["real_dur"] = real_dur
    _write_row_values(tasks_df, idx, row)
    tasks_df.loc[tasks_df["number"] == number, "real_dur"] = real_dur
    return number


def retry_overlong_rows(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Regenerate rows whose measured TTS duration is too long for the target slot."""
    quality = get_quality_config()
    if not quality.enabled or not load_key("rewrite_text_for_dubbing", True):
        return tasks_df

    _ensure_rewrite_columns(tasks_df)
    if load_key("tts_method") == "mlx_indextts2":
        for _ in range(quality.max_rewrite_rounds):
            rewritten_indices = []
            for idx, row in tasks_df.iterrows():
                row_dict = row.to_dict()
                rounds = int(row_dict.get("dubbing_rewrite_rounds", 0) or 0)
                if not actual_rewrite_needed(row_dict) or rounds >= quality.max_rewrite_rounds:
                    continue
                try:
                    reason = (
                        f"Measured TTS duration {float(row_dict.get('real_dur', 0) or 0):.2f}s exceeds "
                        f"available window {float(row_dict.get('available_duration', 0) or 0):.2f}s."
                    )
                    rewritten = rewrite_task_lines(row_dict, reason=reason)
                    if not rewritten:
                        continue
                    _delete_temp_files_for_row(row_dict)
                    row_dict["lines"] = rewritten
                    row_dict["text"] = " ".join(rewritten)
                    row_dict["rewritten_for_dubbing"] = True
                    row_dict["rewrite_reason"] = "measured_over_duration"
                    row_dict["dubbing_rewrite_rounds"] = rounds + 1
                    _write_row_values(tasks_df, idx, row_dict)
                    rewritten_indices.append(idx)
                except Exception as exc:
                    rprint(f"[yellow]⚠️ Measured-duration rewrite skipped for row {row_dict.get('number', idx)}: {exc}[/yellow]")
            if not rewritten_indices:
                break
            rprint(
                f"[yellow]📦 Batch-regenerating {len(rewritten_indices)} measured-overlong "
                "IndexTTS2 rows[/yellow]"
            )
            _batch_regenerate_rewritten_rows(tasks_df, rewritten_indices)
        return tasks_df

    for idx, row in tasks_df.iterrows():
        row_dict = row.to_dict()
        rounds = int(row_dict.get("dubbing_rewrite_rounds", 0) or 0)
        while actual_rewrite_needed(row_dict) and rounds < quality.max_rewrite_rounds:
            try:
                reason = (
                    f"Measured TTS duration {float(row_dict.get('real_dur', 0) or 0):.2f}s exceeds "
                    f"available window {float(row_dict.get('available_duration', 0) or 0):.2f}s."
                )
                rewritten = rewrite_task_lines(row_dict, reason=reason)
                if not rewritten:
                    break
                number = _regenerate_rewritten_row(
                    tasks_df,
                    idx,
                    row_dict,
                    rewritten,
                    rewrite_reason="measured_over_duration",
                )
                rounds += 1
                rprint(f"[green]🔁 Rewrote and regenerated row {number} ({rounds}/{quality.max_rewrite_rounds})[/green]")
            except Exception as exc:
                rprint(f"[yellow]⚠️ Measured-duration rewrite skipped for row {row_dict.get('number', idx)}: {exc}[/yellow]")
                break
    return tasks_df


def retry_underlong_rows(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Regenerate rows whose measured TTS duration is too short for the target slot."""
    quality = get_quality_config()
    if not quality.enabled or not load_key("rewrite_text_for_dubbing", True):
        return tasks_df

    _ensure_rewrite_columns(tasks_df)
    min_speed = max(float(load_key("speed_factor.min", 0.8)), 0.1)
    if load_key("tts_method") == "mlx_indextts2":
        for _ in range(quality.max_rewrite_rounds):
            rewritten_indices = []
            for idx, row in tasks_df.iterrows():
                row_dict = row.to_dict()
                rounds = int(row_dict.get("dubbing_rewrite_rounds", 0) or 0)
                if not actual_expand_needed(row_dict) or rounds >= quality.max_rewrite_rounds:
                    continue
                try:
                    real_dur = float(row_dict.get("real_dur", 0) or 0)
                    projected = real_dur / min_speed if real_dur > 0 else 0
                    reason = (
                        f"Measured TTS duration {real_dur:.2f}s is too short; even at min speed "
                        f"{min_speed:.2f}, projected duration is {projected:.2f}s for "
                        f"available window {float(row_dict.get('available_duration', 0) or 0):.2f}s."
                    )
                    rewritten = rewrite_task_lines(row_dict, reason=reason, direction="expand")
                    if not rewritten:
                        continue
                    _delete_temp_files_for_row(row_dict)
                    row_dict["lines"] = rewritten
                    row_dict["text"] = " ".join(rewritten)
                    row_dict["rewritten_for_dubbing"] = True
                    row_dict["rewrite_reason"] = "measured_under_duration"
                    row_dict["dubbing_rewrite_rounds"] = rounds + 1
                    _write_row_values(tasks_df, idx, row_dict)
                    rewritten_indices.append(idx)
                except Exception as exc:
                    rprint(f"[yellow]⚠️ Measured-duration expansion skipped for row {row_dict.get('number', idx)}: {exc}[/yellow]")
            if not rewritten_indices:
                break
            rprint(
                f"[yellow]📦 Batch-regenerating {len(rewritten_indices)} measured-underlong "
                "IndexTTS2 rows[/yellow]"
            )
            _batch_regenerate_rewritten_rows(tasks_df, rewritten_indices)
        return tasks_df

    for idx, row in tasks_df.iterrows():
        row_dict = row.to_dict()
        rounds = int(row_dict.get("dubbing_rewrite_rounds", 0) or 0)
        while actual_expand_needed(row_dict) and rounds < quality.max_rewrite_rounds:
            try:
                real_dur = float(row_dict.get("real_dur", 0) or 0)
                projected = real_dur / min_speed if real_dur > 0 else 0
                reason = (
                    f"Measured TTS duration {real_dur:.2f}s is too short; even at min speed "
                    f"{min_speed:.2f}, projected duration is {projected:.2f}s for "
                    f"available window {float(row_dict.get('available_duration', 0) or 0):.2f}s."
                )
                rewritten = rewrite_task_lines(row_dict, reason=reason, direction="expand")
                if not rewritten:
                    break
                number = _regenerate_rewritten_row(
                    tasks_df,
                    idx,
                    row_dict,
                    rewritten,
                    rewrite_reason="measured_under_duration",
                )
                rounds += 1
                rprint(f"[green]🔁 Expanded and regenerated row {number} ({rounds}/{quality.max_rewrite_rounds})[/green]")
            except Exception as exc:
                rprint(f"[yellow]⚠️ Measured-duration expansion skipped for row {row_dict.get('number', idx)}: {exc}[/yellow]")
                break
    return tasks_df


def retry_fast_speech_rows(tasks_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Shorten and regenerate rows that would require noticeably accelerated speech."""
    quality = get_quality_config()
    if not quality.enabled or not load_key("rewrite_text_for_dubbing", True):
        return tasks_df, False

    _ensure_rewrite_columns(tasks_df)
    changed = False
    if load_key("tts_method") == "mlx_indextts2":
        rewritten_indices = []
        for idx, row in tasks_df.iterrows():
            row_dict = row.to_dict()
            speed_factor = float(row_dict.get("speed_factor", 1.0) or 1.0)
            rounds = int(row_dict.get("dubbing_rewrite_rounds", 0) or 0)
            if speed_factor <= quality.max_natural_speed_factor or rounds >= quality.max_rewrite_rounds:
                continue
            try:
                reason = (
                    f"Generated audio requires speed factor {speed_factor:.2f}, above the natural speech "
                    f"limit {quality.max_natural_speed_factor:.2f}. Shorten the target-language text so the "
                    "dub can stay close to the original speaking speed."
                )
                rewritten = rewrite_task_lines(row_dict, reason=reason, direction="shorten")
                if not rewritten:
                    continue
                _delete_temp_files_for_row(row_dict)
                row_dict["lines"] = rewritten
                row_dict["text"] = " ".join(rewritten)
                row_dict["rewritten_for_dubbing"] = True
                row_dict["rewrite_reason"] = "speech_rate_fast"
                row_dict["dubbing_rewrite_rounds"] = rounds + 1
                row_dict["speed_factor"] = 1.0
                row_dict["new_sub_times"] = None
                _write_row_values(tasks_df, idx, row_dict)
                rewritten_indices.append(idx)
            except Exception as exc:
                rprint(f"[yellow]⚠️ Natural-speed rewrite skipped for row {row_dict.get('number', idx)}: {exc}[/yellow]")
        if rewritten_indices:
            rprint(
                f"[yellow]📦 Batch-regenerating {len(rewritten_indices)} fast-speech "
                "IndexTTS2 rows[/yellow]"
            )
            _batch_regenerate_rewritten_rows(tasks_df, rewritten_indices)
            changed = True
        return tasks_df, changed

    for idx, row in tasks_df.iterrows():
        row_dict = row.to_dict()
        speed_factor = float(row_dict.get("speed_factor", 1.0) or 1.0)
        rounds = int(row_dict.get("dubbing_rewrite_rounds", 0) or 0)
        while speed_factor > quality.max_natural_speed_factor and rounds < quality.max_rewrite_rounds:
            try:
                reason = (
                    f"Generated audio requires speed factor {speed_factor:.2f}, above the natural speech "
                    f"limit {quality.max_natural_speed_factor:.2f}. Shorten the target-language text so the "
                    "dub can stay close to the original speaking speed."
                )
                rewritten = rewrite_task_lines(row_dict, reason=reason, direction="shorten")
                if not rewritten:
                    break
                number = _regenerate_rewritten_row(
                    tasks_df,
                    idx,
                    row_dict,
                    rewritten,
                    rewrite_reason="speech_rate_fast",
                )
                row_dict["speed_factor"] = 1.0
                row_dict["new_sub_times"] = None
                _write_row_values(tasks_df, idx, row_dict)
                rounds += 1
                changed = True
                rprint(f"[green]🎚️ Rewrote row {number} to preserve natural speech rate ({rounds}/{quality.max_rewrite_rounds})[/green]")
                break
            except Exception as exc:
                rprint(f"[yellow]⚠️ Natural-speed rewrite skipped for row {row_dict.get('number', idx)}: {exc}[/yellow]")
                break
    return tasks_df, changed

def generate_tts_audio(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Generate TTS audio sequentially and calculate actual duration"""
    tasks_df = apply_dubbing_budget_columns(tasks_df)
    if "real_dur" not in tasks_df.columns:
        tasks_df["real_dur"] = 0.0
    tasks_df["real_dur"] = pd.to_numeric(tasks_df["real_dur"], errors="coerce").fillna(0.0)
    rprint("[bold green]🎯 Starting TTS audio generation...[/bold green]")

    tts_method = load_key("tts_method")

    with Progress() as progress:
        task = progress.add_task("[cyan]🔄 Generating TTS audio...", total=len(tasks_df))

        if tts_method == "mlx_indextts2":
            rprint("[yellow]📌 Using model-resident IndexTTS2 batch generation[/yellow]")
            stale_mask = tasks_df.apply(lambda row: not _tts_generation_row_valid(row), axis=1)
            stale_rows = tasks_df.loc[stale_mask].copy()
            progress.advance(task, len(tasks_df) - len(stale_rows))
            if len(stale_rows) < len(tasks_df):
                rprint(
                    f"[yellow]♻️ Reusing {len(tasks_df) - len(stale_rows)} valid rows; "
                    f"generating {len(stale_rows)} stale rows[/yellow]"
                )
            for number, real_dur in process_indextts2_batch(tasks_df, stale_rows):
                tasks_df.loc[tasks_df["number"] == number, "real_dur"] = real_dur
                progress.advance(task)
        else:

            # warm up for first 5 rows
            warmup_size = min(DEFAULT_WARMUP_SIZE, len(tasks_df))
            for _, row in tasks_df.head(warmup_size).iterrows():
                try:
                    number, real_dur = process_row(row.to_dict(), tasks_df)
                    tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                    progress.advance(task)
                except Exception as e:
                    rprint(f"[red]❌ Error in warmup: {str(e)}[/red]")
                    raise e

            if is_local_tts_method(tts_method):
                if len(tasks_df) > warmup_size:
                    rprint(f"[yellow]📌 Using sequential processing for {tts_method} (local API does not support multi-threading)[/yellow]")
                    for _, row in tasks_df.iloc[warmup_size:].iterrows():
                        try:
                            number, real_dur = process_row(row.to_dict(), tasks_df)
                            tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                            progress.advance(task)
                        except Exception as e:
                            rprint(f"[red]❌ Error: {str(e)}[/red]")
                            raise e
            else:
                max_workers = load_key("max_workers")
                Executor = ThreadPoolExecutor

                if len(tasks_df) > warmup_size:
                    remaining_tasks = tasks_df.iloc[warmup_size:].copy()
                    with Executor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(process_row, row.to_dict(), tasks_df)
                            for _, row in remaining_tasks.iterrows()
                        ]

                        for future in as_completed(futures):
                            try:
                                number, real_dur = future.result()
                                tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                                progress.advance(task)
                            except Exception as e:
                                rprint(f"[red]❌ Error: {str(e)}[/red]")
                                raise e

    rprint("[bold green]✨ TTS audio generation completed![/bold green]")
    tasks_df = retry_overlong_rows(tasks_df)
    # Native IndexTTS2 duration fit can leave a short interjection shorter than
    # its slot. Preserve its natural speech and pad the slot during merge;
    # expanding/re-synthesizing a one-word response often makes it less natural.
    if tts_method != "mlx_indextts2":
        tasks_df = retry_underlong_rows(tasks_df)
    tasks_df = retry_overlong_rows(tasks_df)
    return tasks_df


def _row_natural_speed_limit(row: pd.Series, quality) -> float:
    """Allow slightly faster fitting only for ASR-unreliable short replies."""
    base_limit = float(quality.max_natural_speed_factor)
    tokens = re.findall(r"[\w\u00c0-\u024f\u1e00-\u1eff\u4e00-\u9fff]+", str(row.get("text", "")))
    if len(tokens) <= 2 and row_available_duration(row) <= 0.8:
        short_limit = float(
            load_key("dubbing_quality.short_utterance_max_speed_factor", 1.35)
        )
        return max(base_limit, short_limit)
    return base_limit


def merge_indextts2_native_fit(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Fit complete IndexTTS2 speech without ever clipping spoken content."""
    tasks_df["new_sub_times"] = None
    tasks_df["speed_factor"] = 1.0
    tasks_df["keep_gaps"] = True
    rounding_tolerance_ms = int(
        load_key("dubbing_quality.timeline_rounding_tolerance_ms", 30)
    )
    quality = get_quality_config()

    for idx, row in tasks_df.iterrows():
        number = int(row["number"])
        lines = normalize_lines(row.get("lines", row.get("text", "")))
        available_duration = row_available_duration(row)
        max_natural_speed = _row_natural_speed_limit(row, quality)
        row_window_ms = int(round(available_duration * 1000))
        current_ms = int(round(srt_time_to_seconds(row["start_time"]) * 1000))
        new_sub_times = []
        source_files = []
        audios = []
        for line_index, _line in enumerate(lines):
            temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
            source_files.append(temp_file)
            audios.append(_trim_edge_silence(AudioSegment.from_wav(temp_file)))

        total_audio_ms = sum(len(audio) for audio in audios)
        fit_margin_ms = min(max(rounding_tolerance_ms, 0) * max(len(audios), 1), 50)
        row_speed_factor = 1.0
        if total_audio_ms > row_window_ms:
            natural_speed_factor = total_audio_ms / max(row_window_ms, 1)
            row_speed_factor = total_audio_ms / max(row_window_ms - fit_margin_ms, 1)
            if natural_speed_factor > max_natural_speed:
                raise RuntimeError(
                    f"IndexTTS2 row {number} needs {natural_speed_factor:.3f}x speed to fit, "
                    f"above the natural limit {max_natural_speed:.3f}x. Refusing to "
                    "truncate spoken content; rewrite/regenerate first."
                )
            adjusted = []
            for line_index, (source_file, audio) in enumerate(zip(source_files, audios)):
                output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", dir=os.path.dirname(output_file), delete=False
                ) as handle:
                    trimmed_file = handle.name
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", dir=os.path.dirname(output_file), delete=False
                ) as handle:
                    adjusted_file = handle.name
                try:
                    audio.export(trimmed_file, format="wav")
                    adjust_audio_speed(trimmed_file, adjusted_file, row_speed_factor)
                    adjusted.append(AudioSegment.from_wav(adjusted_file))
                finally:
                    if os.path.exists(trimmed_file):
                        os.remove(trimmed_file)
                    if os.path.exists(adjusted_file):
                        os.remove(adjusted_file)
            audios = adjusted

        overflow_ms = sum(len(audio) for audio in audios) - row_window_ms
        if overflow_ms > 0:
            silence_dbfs = float(
                load_key("dubbing_quality.timeline_rounding_silence_dbfs", -35.0)
            )
            remaining = overflow_ms
            for audio_index in range(len(audios) - 1, -1, -1):
                if remaining <= 0:
                    break
                removable = min(remaining, len(audios[audio_index]))
                tail = audios[audio_index][-removable:]
                if tail.dBFS > silence_dbfs:
                    break
                audios[audio_index] = audios[audio_index][:-removable]
                remaining -= removable
            if overflow_ms > rounding_tolerance_ms or remaining > 0:
                raise RuntimeError(
                    f"IndexTTS2 row {number} still exceeds its slot by {overflow_ms}ms "
                    "after natural speed fitting; the overflow contains audible samples, "
                    "so spoken content will not be truncated."
                )

        # Allocate the row window in proportion to measured line duration. Equal
        # partitions make a short interjection steal time from a long sentence.
        measured_total = max(sum(len(audio) for audio in audios), 1)
        line_windows = []
        allocated = 0
        for line_index, audio in enumerate(audios):
            if line_index == len(audios) - 1:
                window_ms = row_window_ms - allocated
            else:
                window_ms = int(round(row_window_ms * len(audio) / measured_total))
                allocated += window_ms
            line_windows.append(max(window_ms, len(audio)))
        # Rounding can add a millisecond. Take it back from lines with padding.
        excess = sum(line_windows) - row_window_ms
        for line_index in range(len(line_windows) - 1, -1, -1):
            if excess <= 0:
                break
            padding = line_windows[line_index] - len(audios[line_index])
            reduction = min(excess, max(padding, 0))
            line_windows[line_index] -= reduction
            excess -= reduction
        if excess > 0:
            raise RuntimeError(
                f"IndexTTS2 row {number} allocation exceeds its hard window by {excess}ms."
            )

        for line_index, (audio, line_window_ms) in enumerate(zip(audios, line_windows)):
            output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
            if len(audio) < line_window_ms:
                audio += AudioSegment.silent(
                    duration=line_window_ms - len(audio),
                    frame_rate=audio.frame_rate,
                )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            _export_wav_if_audio_changed(audio, output_file)
            end_ms = current_ms + line_window_ms
            new_sub_times.append([float(current_ms / 1000), float(end_ms / 1000)])
            current_ms = end_ms
        tasks_df.at[idx, "speed_factor"] = row_speed_factor
        tasks_df.at[idx, "new_sub_times"] = new_sub_times
    return tasks_df


def _clear_asr_results(tasks_df: pd.DataFrame, indices: list[int]) -> None:
    for column in (
        "asr_transcript", "asr_content_score", "asr_leakage_score", "asr_status",
        "asr_line_results", "asr_fingerprint", "asr_language", "asr_backend",
    ):
        if column in tasks_df.columns:
            tasks_df.loc[indices, column] = None


def _trim_generated_edges(tasks_df: pd.DataFrame, indices: list[int]) -> None:
    for idx in indices:
        row = tasks_df.loc[idx].to_dict()
        number = int(row["number"])
        lines = normalize_lines(row.get("lines", row.get("text", "")))
        total_duration = 0.0
        for line_index in range(len(lines)):
            temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
            audio = _trim_edge_silence(AudioSegment.from_wav(temp_file))
            audio.export(temp_file, format="wav")
            total_duration += len(audio) / 1000
        tasks_df.at[idx, "real_dur"] = total_duration


def _rewrite_content_rows(tasks_df: pd.DataFrame, indices: list[int]) -> list[int]:
    rewritten: list[int] = []
    for idx in indices:
        row = tasks_df.loc[idx].to_dict()
        transcript = str(row.get("asr_transcript", ""))
        try:
            lines = rewrite_task_lines(
                row,
                reason=(
                    f"ASR readback was incomplete or severely unclear as {transcript!r}. "
                    "Shorten the translation enough to "
                    "fit its hard subtitle window while preserving the sentence-final meaning, numbers, "
                    "names, and speaker intent."
                ),
                direction="shorten",
            )
        except Exception as exc:
            rprint(
                f"[yellow]⚠️ Content rewrite unavailable for row "
                f"{int(row.get('number', idx))}: {exc}[/yellow]"
            )
            continue
        if not lines or lines == normalize_lines(row.get("lines", row.get("text", ""))):
            continue
        tasks_df.at[idx, "lines"] = lines
        tasks_df.at[idx, "text"] = " ".join(lines)
        tasks_df.at[idx, "rewritten_for_dubbing"] = True
        tasks_df.at[idx, "rewrite_reason"] = "asr_tail_incomplete"
        tasks_df.at[idx, "dubbing_rewrite_rounds"] = int(
            row.get("dubbing_rewrite_rounds", 0) or 0
        ) + 1
        rewritten.append(int(row["number"]))
    return rewritten


def _split_content_rows_for_retry(tasks_df: pd.DataFrame, indices: list[int]) -> list[int]:
    """Split persistent incomplete speech without deleting or rewriting words."""
    changed: list[int] = []
    for idx in indices:
        row = tasks_df.loc[idx]
        original = normalize_lines(row.get("lines", row.get("text", "")))
        split_lines: list[str] = []
        for line in original:
            clauses = [
                clause.strip()
                for clause in re.findall(r"[^.!?]+[.!?]?", str(line))
                if clause.strip()
            ] or [str(line).strip()]
            for clause in clauses:
                words = clause.split()
                if len(words) <= 4:
                    split_lines.append(clause)
                    continue
                midpoint = max(2, len(words) // 2)
                split_lines.extend((" ".join(words[:midpoint]), " ".join(words[midpoint:])))
        if split_lines == original or not all(split_lines):
            continue
        tasks_df.at[idx, "lines"] = split_lines
        tasks_df.at[idx, "text"] = " ".join(split_lines)
        tasks_df.at[idx, "rewritten_for_dubbing"] = True
        tasks_df.at[idx, "rewrite_reason"] = "asr_incomplete_split_retry"
        tasks_df.at[idx, "dubbing_rewrite_rounds"] = int(
            row.get("dubbing_rewrite_rounds", 0) or 0
        ) + 1
        changed.append(int(row["number"]))
    return changed


def _content_fallback_method() -> tuple[str, str]:
    """Resolve the configured MLX backend to its row-level TTS method."""
    backend = str(
        load_key("dubbing_repair.low_content_fallback_backend", "dots")
        or ""
    ).strip()
    if backend.lower() in {"", "none", "off", "disabled"}:
        return "", ""
    if backend in MLX_ROUTER_BACKENDS:
        return backend, MLX_ROUTER_BACKENDS[backend]
    for method, routed_backend in MLX_ROUTER_BACKENDS.items():
        if backend == routed_backend:
            return method, routed_backend
    raise ValueError(
        f"Unsupported low-content fallback backend {backend!r}; expected one of "
        f"{sorted(set(MLX_ROUTER_BACKENDS) | set(MLX_ROUTER_BACKENDS.values()))}."
    )


def _regenerate_content_rows(tasks_df: pd.DataFrame, indices: list[int]) -> None:
    """Regenerate rows with their row-level backend overrides when present."""
    indextts2_indices: list[int] = []
    for idx in indices:
        method = str(tasks_df.at[idx, "tts_method"]).strip() if "tts_method" in tasks_df else ""
        if method.lower() in {"", "nan", "<na>", "none", "mlx_indextts2"}:
            indextts2_indices.append(idx)
            continue
        row = tasks_df.loc[idx].to_dict()
        _delete_temp_files_for_row(row)
        number, real_dur = process_row(row, tasks_df)
        tasks_df.at[idx, "real_dur"] = real_dur
        rprint(f"[yellow]🔁 Row {number} regenerated with persistent fallback {method}[/yellow]")
    # Content repair is intentionally isolated per row. A model-resident batch
    # can retain sampling/reference state across heterogeneous clips even with
    # the same seed; retrying each failed row alone gives IndexTTS2 one clean,
    # deterministic attempt before the workflow changes model.
    for idx in indextts2_indices:
        process_indextts2_batch(tasks_df, tasks_df.loc[[idx]].copy())


def repair_indextts2_content_rows(
    tasks_df: pd.DataFrame,
    indices: list[int],
    rewrite: bool,
) -> tuple[pd.DataFrame, list[int]]:
    """Regenerate selected rows uncapped, then safely refit the complete speech."""
    if not indices:
        return tasks_df, []
    if "disable_native_fit" not in tasks_df.columns:
        tasks_df["disable_native_fit"] = False
    tasks_df.loc[indices, "disable_native_fit"] = True
    rewritten: list[int] = []
    if rewrite:
        # A persistently unreadable take needs a different synthesizer, not a
        # shorter translation. Only shorten when the ASR heard a matching
        # prefix and specifically proved that the sentence tail is missing.
        from core.dubbing_content_gate import probable_tail_truncation

        tail_indices = [
            idx
            for idx in indices
            if probable_tail_truncation(
                str(tasks_df.at[idx, "text"]),
                str(tasks_df.loc[idx].get("asr_transcript", "")),
            )
        ]
        # Splitting preserves every word, so it is safe for both explicit
        # boundary loss and severely unreadable/near-empty takes.
        split_retries = _split_content_rows_for_retry(tasks_df, indices)
        rewritten.extend(split_retries)
        split_numbers = set(split_retries)
        unsplit_indices = [
            idx for idx in tail_indices if int(tasks_df.at[idx, "number"]) not in split_numbers
        ]
        rewritten.extend(_rewrite_content_rows(tasks_df, unsplit_indices))
    if rewrite:
        fallback_method, fallback_backend = _content_fallback_method()
        if fallback_method:
            if "tts_method" not in tasks_df.columns:
                tasks_df["tts_method"] = ""
            if "tts_backend" not in tasks_df.columns:
                tasks_df["tts_backend"] = ""
            tasks_df.loc[indices, "tts_method"] = fallback_method
            tasks_df.loc[indices, "tts_backend"] = fallback_backend
    _clear_asr_results(tasks_df, indices)

    _regenerate_content_rows(tasks_df, indices)
    _trim_generated_edges(tasks_df, indices)

    quality = get_quality_config()
    overlong: list[int] = []
    for idx in indices:
        row = tasks_df.loc[idx]
        if float(row.get("real_dur", 0) or 0) > row_available_duration(row) * _row_natural_speed_limit(row, quality):
            overlong.append(idx)
    if overlong:
        duration_rewritten = _rewrite_content_rows(tasks_df, overlong)
        rewritten.extend(duration_rewritten)
        unresolved = [idx for idx in overlong if int(tasks_df.at[idx, "number"]) not in duration_rewritten]
        if unresolved:
            numbers = [int(tasks_df.at[idx, "number"]) for idx in unresolved]
            raise RuntimeError(
                f"IndexTTS2 complete speech is too long for rows {numbers}, and compact rewriting failed."
            )
        # The first repair pass is intentionally uncapped so the content gate
        # can learn whether the model is capable of speaking the full tail. Once
        # that complete take proves too long and the script is shortened, turn
        # native duration control back on for the replacement take.
        tasks_df.loc[overlong, "disable_native_fit"] = False
        _regenerate_content_rows(tasks_df, overlong)
        _trim_generated_edges(tasks_df, overlong)
        still_overlong = [
            idx for idx in overlong
            if float(tasks_df.at[idx, "real_dur"] or 0)
            > row_available_duration(tasks_df.loc[idx])
            * _row_natural_speed_limit(tasks_df.loc[idx], quality)
        ]
        if still_overlong:
            numbers = [int(tasks_df.at[idx, "number"]) for idx in still_overlong]
            raise RuntimeError(
                f"IndexTTS2 duration-aware repair is still too long for rows {numbers}; "
                "refusing to accelerate or truncate beyond the natural limit."
            )

    tasks_df = merge_indextts2_native_fit(tasks_df)
    return tasks_df, sorted(set(rewritten))


def run_indextts2_content_gate(tasks_df: pd.DataFrame) -> pd.DataFrame:
    from core.dubbing_content_gate import enforce_content_completion

    def checkpoint(verified: pd.DataFrame) -> None:
        # Content repair may still fail closed after replacing a few rows. Save
        # the exact text/backend/audio fingerprints after every round so resume
        # rechecks only those rows instead of regenerating the whole soundtrack.
        mark_tts_generation_checkpoint(verified)
        verified.to_excel(TTS_TASKS_FILE, index=False)

    verified, summary = enforce_content_completion(
        tasks_df,
        repair_rows=repair_indextts2_content_rows,
        checkpoint=checkpoint,
    )
    rprint(
        f"[bold green]🔎 Content gate: {summary.status}, rounds={summary.rounds}, "
        f"remaining_tail_rows={summary.remaining_tail_truncation_rows}[/bold green]"
    )
    return verified

def process_chunk(chunk_df: pd.DataFrame, accept: float, min_speed: float) -> tuple[float, bool]:
    """Process audio chunk and calculate speed factor"""
    chunk_durs = chunk_df['real_dur'].sum()
    tol_durs = chunk_df['tol_dur'].sum()
    durations = tol_durs - chunk_df.iloc[-1]['tolerance']
    all_gaps = chunk_df['gap'].sum() - chunk_df.iloc[-1]['gap']

    # Check the gap of the last line in the chunk
    last_gap = chunk_df.iloc[-1]['gap']
    
    # Dynamic speed_var_error: if gap is small (< 0.1s), it means continuous speech
    # In this case, we don't want to leave any safety gap (speed_var_error = 0)
    # Otherwise, we keep the 0.1s safety margin
    speed_var_error = 0.1 if last_gap >= 0.1 else 0
    
    rprint(f"[dim]  └─ Last gap: {last_gap:.3f}s, speed_var_error: {speed_var_error:.3f}s[/dim]")
    
    # Only check for division by zero - don't skip small durations
    # Small duration chunks still need speed adjustment, possibly with high speed factors
    if (tol_durs - speed_var_error) <= 0.01 or (durations - speed_var_error) <= 0.01:
        rprint(f"[yellow]⚠️ Invalid duration after speed_var_error adjustment (tol_durs={tol_durs:.3f}s), using speed factor 1.0[/yellow]")
        return 1.0, True

    keep_gaps = True

    if (chunk_durs + all_gaps) / accept < durations:
        speed_factor = max(min_speed, (chunk_durs + all_gaps) / (durations-speed_var_error))
    elif chunk_durs / accept < durations:
        speed_factor = max(min_speed, chunk_durs / (durations-speed_var_error))
        keep_gaps = False
    elif (chunk_durs + all_gaps) / accept < tol_durs:
        speed_factor = max(min_speed, (chunk_durs + all_gaps) / (tol_durs-speed_var_error))
    else:
        speed_factor = chunk_durs / (tol_durs-speed_var_error)
        keep_gaps = False

    return round(speed_factor, 3), keep_gaps


def merge_chunks(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Merge audio chunks and adjust timeline"""
    rprint("[bold blue]🔄 Starting audio chunks processing...[/bold blue]")
    accept = load_key("speed_factor.accept")
    min_speed = load_key("speed_factor.min")
    chunk_start = 0

    tasks_df['new_sub_times'] = None
    tasks_df['speed_factor'] = 1.0
    tasks_df['keep_gaps'] = True

    for index, row in tasks_df.iterrows():
        if row['cut_off'] == 1:
            chunk_df = tasks_df.iloc[chunk_start:index+1].reset_index(drop=True)
            speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)

            # 🎯 Step1: Start processing new timeline
            chunk_start_time = srt_time_to_seconds(chunk_df.iloc[0]['start_time'])
            chunk_end_time = srt_time_to_seconds(chunk_df.iloc[-1]['end_time']) + chunk_df.iloc[-1]['tolerance']
            cur_time = chunk_start_time
            for i, row in chunk_df.iterrows():
                # If i is not 0, which is not the first row of the chunk, cur_time needs to be added with the gap of the previous row, remember to divide by speed_factor
                if i != 0 and keep_gaps:
                    cur_time += chunk_df.iloc[i-1]['gap']/speed_factor
                new_sub_times = []
                number = row['number']
                lines = normalize_lines(row.get('lines', row.get('text', '')))
                for line_index, line in enumerate(lines):
                    # 🔄 Step2: Start speed change and save as OUTPUT_FILE_TEMPLATE
                    temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
                    output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
                    adjust_audio_speed(temp_file, output_file, speed_factor)
                    ad_dur = get_audio_duration(output_file)
                    new_sub_times.append([float(cur_time), float(cur_time + ad_dur)])
                    cur_time += ad_dur
                # 🔄 Step3: Find corresponding main DataFrame index and update new_sub_times
                main_df_idx = tasks_df[tasks_df['number'] == row['number']].index[0]
                tasks_df.at[main_df_idx, 'new_sub_times'] = new_sub_times
                tasks_df.at[main_df_idx, 'speed_factor'] = speed_factor
                tasks_df.at[main_df_idx, 'keep_gaps'] = keep_gaps
                # 🎯 Step4: Choose emoji based on speed_factor and accept comparison
                emoji = "⚡" if speed_factor <= accept else "⚠️"
                rprint(f"[cyan]{emoji} Processed chunk {chunk_start} to {index} with speed factor {speed_factor}[/cyan]")
            # 🔄 Step5: Check if the last row exceeds the range
            if cur_time > chunk_end_time + 0.05:
                time_diff = cur_time - chunk_end_time
                fit_ratio = (cur_time - chunk_start_time) / max(0.1, chunk_end_time - chunk_start_time)
                rprint(f"[yellow]⚠️ Chunk {chunk_start} to {index} delta {time_diff:.3f}s, micro-refitting with factor {fit_ratio:.3f}[/yellow]")
                cur_time = chunk_start_time
                for i, r_row in chunk_df.iterrows():
                    if i != 0 and keep_gaps:
                        cur_time += chunk_df.iloc[i-1]['gap'] / (speed_factor * fit_ratio)
                    r_num = r_row['number']
                    r_lines = normalize_lines(r_row.get('lines', r_row.get('text', '')))
                    new_sub_times = []
                    for line_index, _ in enumerate(r_lines):
                        temp_file = TEMP_FILE_TEMPLATE.format(f"{r_num}_{line_index}")
                        output_file = OUTPUT_FILE_TEMPLATE.format(f"{r_num}_{line_index}")
                        adjust_audio_speed(temp_file, output_file, speed_factor * fit_ratio)
                        ad_dur = get_audio_duration(output_file)
                        new_sub_times.append([float(cur_time), float(cur_time + ad_dur)])
                        cur_time += ad_dur
                    m_idx = tasks_df[tasks_df['number'] == r_num].index[0]
                    tasks_df.at[m_idx, 'new_sub_times'] = new_sub_times
                    tasks_df.at[m_idx, 'speed_factor'] = speed_factor * fit_ratio
            chunk_start = index+1

    rprint("[bold green]✅ Audio chunks processing completed![/bold green]")
    return tasks_df

def gen_audio() -> None:
    """Main function: Generate audio and process timeline"""
    rprint("[bold magenta]🚀 Starting audio generation process...[/bold magenta]")
    tts_method = load_key("tts_method")

    # 🎯 Step1: Create necessary directories
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(SEGS_DIR, exist_ok=True)

    # 📝 Step2: Load task file
    tasks_df = pd.read_excel(TTS_TASKS_FILE)
    # Note: Filtering is done in step8_1, so tasks_df is already clean
    
    rprint("[green]📊 Loaded task file successfully[/green]")

    # 🔊 Step3: Generate TTS audio, or resume an exact script/audio checkpoint.
    if tts_method == "mlx_indextts2" and tts_generation_checkpoint_valid(tasks_df):
        rprint("[yellow]♻️ Reusing verified IndexTTS2 generation checkpoint[/yellow]")
        # A previous attempt may have stopped at the native-fit gate after all
        # audio was generated. Resume the measured rewrite loop instead of
        # retrying the same merge forever or regenerating every row.
        tasks_df = retry_overlong_rows(tasks_df)
        tasks_df = mark_tts_generation_checkpoint(tasks_df)
        tasks_df.to_excel(TTS_TASKS_FILE, index=False)
    else:
        tasks_df = generate_tts_audio(tasks_df)
        # Persist measured durations and any LLM rewrites before native fitting
        # or ASR validation. Fingerprints prevent edited text/stale audio reuse.
        tasks_df = mark_tts_generation_checkpoint(tasks_df)
        tasks_df.to_excel(TTS_TASKS_FILE, index=False)

    # 🔄 Step4: Merge audio chunks
    if (
        tts_method == "mlx_indextts2"
        and bool(load_key("mlx_tts.backends.indextts2.fit_duration", True))
    ):
        tasks_df = merge_indextts2_native_fit(tasks_df)
    else:
        tasks_df = merge_chunks(tasks_df)
        tasks_df, natural_speed_changed = retry_fast_speech_rows(tasks_df)
        if natural_speed_changed:
            tasks_df = merge_chunks(tasks_df)

    if (
        tts_method == "mlx_indextts2"
        and bool(load_key("dubbing_quality.content_completion_gate", True))
        and bool(load_key("dubbing_quality.asr_readback", True))
    ):
        tasks_df = run_indextts2_content_gate(tasks_df)

    # 💾 Step5: Save results
    tasks_df.to_excel(TTS_TASKS_FILE, index=False)
    summary = write_dubbing_eval(tasks_df)
    rprint(f"[bold green]📊 Dubbing eval written: {summary}[/bold green]")
    rprint("[bold green]🎉 Audio generation completed successfully![/bold green]")

if __name__ == "__main__":
    gen_audio()
