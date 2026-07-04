import os
import sys
import logging
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
from core.all_tts_functions.tts_registry import is_local_tts_method
from core.audio_speed import adjust_audio_speed as _adjust_audio_speed
from core.audio_speed import build_atempo_filter as _build_atempo_filter
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
    tasks_df['real_dur'] = 0
    rprint("[bold green]🎯 Starting TTS audio generation...[/bold green]")

    with Progress() as progress:
        task = progress.add_task("[cyan]🔄 Generating TTS audio...", total=len(tasks_df))

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

        tts_method = load_key("tts_method")

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
    tasks_df = retry_underlong_rows(tasks_df)
    tasks_df = retry_overlong_rows(tasks_df)
    return tasks_df

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
                    new_sub_times.append([cur_time, cur_time+ad_dur])
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
            if cur_time > chunk_end_time:
                time_diff = cur_time - chunk_end_time
                # Increase tolerance to handle accumulated speed variation errors
                # especially when short chunks follow long-gap chunks
                if time_diff <= 5.0:
                    rprint(f"[yellow]⚠️ Chunk {chunk_start} to {index} exceeds by {time_diff:.3f}s, truncating last audio[/yellow]")
                    # Get the last audio file
                    last_number = tasks_df.iloc[index]['number']
                    last_lines = normalize_lines(tasks_df.iloc[index].get('lines', tasks_df.iloc[index].get('text', '')))
                    last_line_index = len(last_lines) - 1
                    last_file = OUTPUT_FILE_TEMPLATE.format(f"{last_number}_{last_line_index}")

                    # Calculate the duration to keep
                    audio = AudioSegment.from_wav(last_file)
                    original_duration = len(audio) / 1000  # Convert to seconds
                    new_duration = original_duration - time_diff
                    if new_duration > 0:
                        trimmed_audio = audio[:(new_duration * 1000)].fade_out(10)  # pydub uses milliseconds
                        trimmed_audio.export(last_file, format="wav")
                    else:
                        rprint(f"[red]⚠️ Cannot trim audio: new_duration={new_duration:.3f}s <= 0[/red]")

                    # Update the last timestamp
                    last_times = tasks_df.at[index, 'new_sub_times']
                    last_times[-1][1] = chunk_end_time
                    tasks_df.at[index, 'new_sub_times'] = last_times
                else:
                    raise Exception(f"Chunk {chunk_start} to {index} exceeds the chunk end time {chunk_end_time:.2f} seconds with current time {cur_time:.2f} seconds")
            chunk_start = index+1

    rprint("[bold green]✅ Audio chunks processing completed![/bold green]")
    return tasks_df

def gen_audio() -> None:
    """Main function: Generate audio and process timeline"""
    rprint("[bold magenta]🚀 Starting audio generation process...[/bold magenta]")

    # 🎯 Step1: Create necessary directories
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(SEGS_DIR, exist_ok=True)

    # 📝 Step2: Load task file
    tasks_df = pd.read_excel(TTS_TASKS_FILE)
    # Note: Filtering is done in step8_1, so tasks_df is already clean
    
    rprint("[green]📊 Loaded task file successfully[/green]")

    # 🔊 Step3: Generate TTS audio
    tasks_df = generate_tts_audio(tasks_df)

    # 🔄 Step4: Merge audio chunks
    tasks_df = merge_chunks(tasks_df)
    tasks_df, natural_speed_changed = retry_fast_speech_rows(tasks_df)
    if natural_speed_changed:
        tasks_df = merge_chunks(tasks_df)

    # 💾 Step5: Save results
    tasks_df.to_excel(TTS_TASKS_FILE, index=False)
    summary = write_dubbing_eval(tasks_df)
    rprint(f"[bold green]📊 Dubbing eval written: {summary}[/bold green]")
    rprint("[bold green]🎉 Audio generation completed successfully![/bold green]")

if __name__ == "__main__":
    gen_audio()
