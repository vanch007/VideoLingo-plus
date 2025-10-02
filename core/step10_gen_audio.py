import os
import sys
import time
import shutil
import subprocess
from typing import Tuple

import pandas as pd
from pydub import AudioSegment
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config_utils import load_key
from core.all_whisper_methods.audio_preprocess import get_audio_duration
from core.all_tts_functions.tts_main import tts_main

console = Console()

TEMP_DIR = 'output/audio/tmp'
SEGS_DIR = 'output/audio/segs'
TASKS_FILE = "output/audio/tts_tasks.xlsx"
OUTPUT_FILE = "output/audio/tts_tasks.xlsx"
TEMP_FILE_TEMPLATE = f"{TEMP_DIR}/{{}}_temp.{{}}"
OUTPUT_FILE_TEMPLATE = f"{SEGS_DIR}/{{}}.wav"
WARMUP_SIZE = 5
MIN_SPEED_DENOMINATOR = 0.01 # Small positive value to prevent division by zero or negative denominators for speed factor calculation

def parse_df_srt_time(time_str: str) -> float:
    """Convert SRT time format to seconds"""
    hours, minutes, seconds = time_str.strip().split(':')
    if '.' in seconds:
        seconds, microseconds = seconds.split('.')
        # Convert microseconds to fraction of second
        # If the microseconds part has 6 digits, divide by 1000000
        # If it has 3 digits, divide by 1000
        microseconds_value = float('0.' + microseconds)
    else:
        microseconds_value = 0
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + microseconds_value

def adjust_audio_speed(input_file: str, output_file: str, speed_factor: float) -> None:
    """Adjust audio speed and handle edge cases"""
    # Validate speed factor to prevent negative or invalid values
    if speed_factor <= 0:
        raise ValueError(f"Invalid speed factor: {speed_factor}. Speed factor must be positive.")
    
    # If the speed factor is close to 1, directly copy the file
    if abs(speed_factor - 1.0) < 0.001:
        shutil.copy2(input_file, output_file)
        return

    # Determine input format
    input_format = os.path.splitext(input_file)[1][1:].lower()  # Get extension without dot
    
    atempo = max(0.5, min(speed_factor, 100.0))
    cmd = ['ffmpeg', '-i', input_file, '-filter:a', f'atempo={atempo}', '-y', output_file]
    input_duration = get_audio_duration(input_file)
    max_retries = 2
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
            output_duration = get_audio_duration(output_file)
            expected_duration = input_duration / speed_factor
            diff = output_duration - expected_duration
            # If the output duration exceeds the expected duration, but the input audio is less than 3 seconds, and the error is within 0.1 seconds, truncate to the expected length
            if output_duration >= expected_duration * 1.02 and input_duration < 3 and diff <= 0.1:
                audio = AudioSegment.from_file(output_file)
                trimmed_audio = audio[:(expected_duration * 1000)]  # pydub uses milliseconds
                trimmed_audio.export(output_file, format="wav")
                print(f"✂️ Trimmed to expected duration: {expected_duration:.2f} seconds")
                return
            elif output_duration >= expected_duration * 1.02:
                raise Exception(f"Audio duration abnormal: input file={input_file}, output file={output_file}, speed factor={speed_factor}, input duration={input_duration:.2f}s, output duration={output_duration:.2f}s")
            return
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                rprint(f"[yellow]⚠️ Audio speed adjustment failed, retrying in 1s ({attempt + 1}/{max_retries})[/yellow]")
                time.sleep(1)
            else:
                rprint(f"[red]❌ Audio speed adjustment failed, max retries reached ({max_retries})[/red]")
                raise e

def detect_silence(audio_file: str, silence_threshold: float = -40.0, min_silence_duration: float = 0.5) -> None:
    """Detect and trim silence from audio file"""
    audio = AudioSegment.from_file(audio_file)
    trimmed_audio = audio.strip_silence(
        silence_thresh=silence_threshold,
        silence_len=int(min_silence_duration * 1000)  # Convert to integer explicitly
    )
    trimmed_audio.export(audio_file, format="wav")

def create_silent_fallback(temp_file):
    """Creates a 1-second silent WAV file as a fallback."""
    silence = AudioSegment.silent(duration=1000, frame_rate=16000)
    silence = silence.set_channels(1)
    silence.export(temp_file, format="wav", parameters=["-ar", "16000", "-ac", "1"])
    return 1.0

def process_row(row: pd.Series, tasks_df: pd.DataFrame) -> Tuple[int, float]:
    """Helper function for processing single row data"""
    number = row['number']
    text = row['text']
    temp_file_mp3 = TEMP_FILE_TEMPLATE.format(f"{number}", "mp3")
    temp_file_wav = TEMP_FILE_TEMPLATE.format(f"{number}", "wav")
    real_dur = 0
    max_retries = 3

    # Check for problematic numbers (based on the error report)
    problematic_numbers = [693, 699, 700, 701, 702, 703, 704, 705, 698, 858, 653, 1069]
    if number in problematic_numbers:
        rprint(f"[yellow]⚠️ 检测到问题音频文件编号: {number}, 将强制重新生成[/yellow]")
        for temp_file in [temp_file_mp3, temp_file_wav]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # Check if file already exists and is valid
    temp_file = temp_file_mp3 if os.path.exists(temp_file_mp3) else temp_file_wav
    if os.path.exists(temp_file):
        file_size = os.path.getsize(temp_file)
        if file_size < 10000:  # File exists but is too small
            rprint(f"[yellow]⚠️ 已存在的音频文件过小 ({file_size} 字节)，将重新生成: {temp_file}[/yellow]")
            os.remove(temp_file)
        else:
            # Try to get duration to validate file
            try:
                # Try to load with pydub to verify it's a valid audio file
                audio = AudioSegment.from_file(temp_file)
                duration = len(audio) / 1000  # Convert to seconds

                if duration > 0.2 and duration < 10:  # Valid duration
                    rprint(f"[green]✅ 使用已存在的有效音频文件: {temp_file} (时长: {duration:.2f}秒)[/green]")
                    return number, duration
                else:
                    rprint(f"[yellow]⚠️ 已存在的音频文件时长异常 ({duration:.2f}秒)，将重新生成: {temp_file}[/yellow]")
                    os.remove(temp_file)
            except Exception as e:
                rprint(f"[red]❌ 已存在的音频文件损坏: {temp_file} - {str(e)}[/red]")
                os.remove(temp_file)

    for attempt in range(max_retries):
        try:
            # Generate TTS audio with a timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(tts_main, text, temp_file_mp3, number, tasks_df)
                try:
                    # Set a 60-second timeout for the TTS task
                    future.result(timeout=60)
                except Exception as e:
                    # Re-raise the exception to be caught by the outer loop
                    raise e

            # Verify the file exists after TTS generation
            if not os.path.exists(temp_file_mp3):
                raise Exception(f"TTS生成后文件不存在: {temp_file_mp3}")

            file_size = os.path.getsize(temp_file_mp3)

            # Check if file is too small (likely invalid)
            if file_size < 10000:
                os.remove(temp_file_mp3)
                if attempt < max_retries - 1:
                    rprint(f"[yellow]⚠️ 音频文件过小 (大小:{file_size}字节), 重试中({attempt + 1}/{max_retries})[/yellow]")
                    time.sleep(1)  # 增加重试间隔
                    continue
                else:
                    raise Exception(f"音频文件过小且达到最大重试次数(大小:{file_size}字节)")

            # Verify the file is a valid audio file
            try:
                # Try to load with pydub to verify it's a valid audio file
                audio = AudioSegment.from_file(temp_file_mp3)
                duration = len(audio) / 1000  # Convert to seconds
            except Exception as e:
                os.remove(temp_file_mp3)
                if attempt < max_retries - 1:
                    rprint(f"[red]❌ 音频文件损坏: {temp_file_mp3} - {str(e)}[/red]")
                    time.sleep(1)
                    continue
                else:
                    raise Exception(f"音频文件损坏且达到最大重试次数: {str(e)}")

            # 检查音频文件是否异常(大于10秒或小于0.2秒)
            if duration > 10 or duration < 0.2:
                os.remove(temp_file_mp3)
                if attempt < max_retries - 1:
                    rprint(f"[yellow]⚠️ 音频文件时长异常(时长:{duration:.2f}秒), 重试中({attempt + 1}/{max_retries})[/yellow]")
                    time.sleep(1)  # 增加重试间隔
                    continue
                else:
                    raise Exception(f"音频文件时长异常且达到最大重试次数(时长:{duration:.2f}秒)")

            # 只对1.5秒以上的音频检测并剪切静音片段
            if duration >= 1.5:
                try:
                    detect_silence(temp_file_mp3)
                    # 重新检查剪切后的时长
                    audio = AudioSegment.from_file(temp_file_mp3)
                    new_duration = len(audio) / 1000  # Convert to seconds

                    # 如果剪切后时长小于0.4秒，使用原始音频
                    if new_duration < 0.4:
                        rprint(f"[yellow]⚠️ 剪切静音后时长过短 ({new_duration:.2f}秒)，使用原始音频[/yellow]")
                        tts_main(text, temp_file_mp3, number, tasks_df)  # 重新生成
                        audio = AudioSegment.from_file(temp_file_mp3)
                        duration = len(audio) / 1000  # Convert to seconds
                    else:
                        duration = new_duration
                except Exception as e:
                    rprint(f"[yellow]⚠️ 剪切静音失败: {str(e)}[/yellow]")
                    # Continue with the original audio

            # Final validation
            if os.path.exists(temp_file_mp3):
                file_size = os.path.getsize(temp_file_mp3)
                if file_size < 10000 or duration <= 0:
                    rprint(f"[red]❌ 最终验证失败: 文件大小={file_size}字节, 时长={duration:.2f}秒[/red]")
                    duration = create_silent_fallback(temp_file_wav)

            real_dur = duration
            break

        except Exception as e:
            error_message = str(e)
            if "TimeoutError" in str(type(e)):
                 error_message = "TTS generation timed out after 30 seconds."

            if attempt < max_retries - 1:
                rprint(f"[yellow]⚠️ TTS generation failed for {number}, retrying ({attempt + 1}/{max_retries}): {error_message}[/yellow]")
                time.sleep(1)
            else:
                rprint(f"[red]❌ TTS for number {number} failed after {max_retries} retries: {error_message}.[/red]")
                real_dur = 0  # Mark as failed
                break  # Exit the retry loop

    return number, real_dur

def generate_tts_audio(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Generate TTS audio sequentially and calculate actual duration"""
    tasks_df['real_dur'] = 0
    rprint("[bold green]🎯 Starting TTS audio generation...[/bold green]")

    with Progress() as progress:
        task = progress.add_task("[cyan]🔄 Generating TTS audio...", total=len(tasks_df))

        # warm up for first 5 rows
        warmup_size = min(WARMUP_SIZE, len(tasks_df))
        for _, row in tasks_df.head(warmup_size).iterrows():
            try:
                number, real_dur = process_row(row, tasks_df)
                tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
            except Exception as e:
                rprint(f"[red]❌ Error in warmup for task {row['number']}: {str(e)}[/red]")
                tasks_df.loc[tasks_df['number'] == row['number'], 'real_dur'] = 0
            finally:
                progress.advance(task)

        # for gpt_sovits, do not use parallel to avoid mistakes
        tts_method = load_key("tts_method")
        max_workers = load_key("max_workers") if tts_method not in ["gpt_sovits", "custom_tts"] else 1
        
        Executor = ProcessPoolExecutor if tts_method == "sf_indextts2" else ThreadPoolExecutor
        
        # parallel processing for remaining tasks
        if len(tasks_df) > warmup_size:
            remaining_tasks = tasks_df.iloc[warmup_size:].copy()
            with Executor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_row, row, tasks_df.copy()): row for _, row in remaining_tasks.iterrows()}

                for future in as_completed(futures):
                    row = futures[future]
                    number = row['number']
                    try:
                        returned_number, real_dur = future.result()
                        # Make sure the returned number matches
                        if returned_number == number:
                            tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                        else:
                            rprint(f"[red]❌ Mismatched number in future result! Expected {number}, got {returned_number}.[/red]")
                            tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = 0
                    except Exception as e:
                        rprint(f"[red]❌ Error for task {number}: {str(e)}[/red]")
                        tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = 0
                    finally:
                        progress.advance(task)

    rprint("[bold green]✨ Initial TTS audio generation completed![/bold green]")

    # Retry failed tasks
    failed_tasks = tasks_df[tasks_df['real_dur'] == 0]
    if not failed_tasks.empty:
        rprint(f"[bold yellow]⚠️ Found {len(failed_tasks)} failed tasks. Retrying them now...[/bold yellow]")
        
        with Progress() as progress:
            retry_task_progress = progress.add_task("[cyan]🔄 Retrying failed tasks...", total=len(failed_tasks))
            
            for _, row in failed_tasks.iterrows():
                number = row['number']
                try:
                    # Retry logic is inside process_row, so just call it again.
                    _, real_dur = process_row(row, tasks_df)
                    tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                    
                    if real_dur == 0:
                        # If it still fails, try to use reference audio as a fallback
                        rprint(f"[red]❌ Retry failed for task {number}. Trying to use reference audio as fallback.[/red]")
                        temp_file_mp3 = TEMP_FILE_TEMPLATE.format(f"{number}", "mp3")
                        temp_file_wav = TEMP_FILE_TEMPLATE.format(f"{number}", "wav")
                        ref_audio_path = f"output/audio/refers/{number}.wav"

                        if os.path.exists(ref_audio_path) and os.path.getsize(ref_audio_path) > 20000:
                            try:
                                shutil.copy2(ref_audio_path, temp_file_wav)
                                duration = get_audio_duration(temp_file_wav)
                                tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = duration
                                rprint(f"[green]✅ Used reference audio for task {number}.[/green]")
                            except Exception as e:
                                rprint(f"[red]❌ Failed to use reference audio for task {number}: {e}. Generating 1s silent audio.[/red]")
                                tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = create_silent_fallback(temp_file_wav)
                        else:
                            rprint(f"[red]❌ Reference audio for task {number} is missing or invalid. Generating 1s silent audio.[/red]")
                            tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = create_silent_fallback(temp_file_wav)
                        
                except Exception as e:
                    rprint(f"[red]❌ An unexpected error occurred during retry for task {number}: {str(e)}. Generating 1s silent audio.[/red]")
                    temp_file_wav = TEMP_FILE_TEMPLATE.format(f"{number}", "wav")
                    tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = create_silent_fallback(temp_file_wav)
                
                progress.advance(retry_task_progress)

    rprint("[bold green]✅ All TTS audio tasks completed (with retries if necessary).[/bold green]")
    return tasks_df

def process_chunk(chunk_df: pd.DataFrame, accept: float, min_speed: float) -> tuple[float, bool]:
    """Process audio chunk and calculate speed factor"""
    chunk_durs = chunk_df['real_dur'].sum()
    tol_durs = chunk_df['tol_dur'].sum()
    durations = tol_durs - chunk_df.iloc[-1]['tolerance']
    all_gaps = chunk_df['gap'].sum() - chunk_df.iloc[-1]['gap']

    keep_gaps = True
    speed_var_error = 0.1

    # Calculate base speed factor with more tolerance
    # Ensure denominators are always positive to prevent negative speed factors
    if (chunk_durs + all_gaps) / accept < durations:
        denominator = max(MIN_SPEED_DENOMINATOR, durations-speed_var_error*2)
        base_factor = max(min_speed, (chunk_durs + all_gaps) / denominator)
    elif chunk_durs / accept < durations:
        denominator = max(MIN_SPEED_DENOMINATOR, durations-speed_var_error*2)
        base_factor = max(min_speed, chunk_durs / denominator)
        keep_gaps = False
    elif (chunk_durs + all_gaps) / accept < tol_durs:
        denominator = max(MIN_SPEED_DENOMINATOR, tol_durs-speed_var_error*2)
        base_factor = max(min_speed, (chunk_durs + all_gaps) / denominator)
    else:
        denominator = max(MIN_SPEED_DENOMINATOR, tol_durs-speed_var_error*2)
        base_factor = max(min_speed, chunk_durs / denominator)
        keep_gaps = False

    # Apply safety margin when approaching limits
    safety_margin = 1.0
    if base_factor > accept * 1.5:  # If we're way over
        safety_margin = 0.95  # Less aggressive
    elif base_factor > accept * 1.2:  # If we're moderately over
        safety_margin = 0.98

    speed_factor = max(min_speed, base_factor * safety_margin)

    return round(speed_factor, 3), keep_gaps

def merge_chunks(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Merge audio chunks and adjust timeline"""
    rprint("[bold blue]🔄 Starting audio chunks processing...[/bold blue]")

    if 'cut_off' not in tasks_df.columns:
        rprint("[yellow]⚠️ 'cut_off' column not found in task file. Treating all tasks as a single chunk.[/yellow]")
        tasks_df['cut_off'] = 0
        if not tasks_df.empty:
            tasks_df.iloc[-1, tasks_df.columns.get_loc('cut_off')] = 1

    accept = load_key("speed_factor.accept")
    min_speed = load_key("speed_factor.min")
    chunk_start = 0

    tasks_df['new_sub_times'] = None

    # First, verify all audio files exist and are valid
    missing_files = []
    for _, row in tasks_df.iterrows():
        number = row['number']
        temp_file_mp3 = TEMP_FILE_TEMPLATE.format(f"{number}", "mp3")
        temp_file_wav = TEMP_FILE_TEMPLATE.format(f"{number}", "wav")
        temp_file = temp_file_mp3 if os.path.exists(temp_file_mp3) else temp_file_wav
        if not os.path.exists(temp_file) or os.path.getsize(temp_file) < 10000:
            missing_files.append(number)

    if missing_files:
        rprint(f"[red]❌ Found {len(missing_files)} missing or invalid audio files. Please regenerate them first.[/red]")
        rprint(f"Missing files: {missing_files[:10]}{'...' if len(missing_files) > 10 else ''}")
        raise Exception("Missing audio files detected. Run clean_audio_files.py first.")

    for index, row in tasks_df.iterrows():
        if row['cut_off'] == 1:
            chunk_df = tasks_df.iloc[chunk_start:index+1].reset_index(drop=True)

            # Skip empty chunks (can happen if there are consecutive cut_off=1)
            if len(chunk_df) == 0:
                chunk_start = index+1
                continue

            # Special handling for problematic chunks (1058-1060)
            if chunk_start == 1058 and index == 1060:
                rprint(f"[yellow]⚠️ Special handling for problematic chunk {chunk_start} to {index}[/yellow]")
                # Force a more conservative speed factor for this chunk
                speed_factor = 0.95
                keep_gaps = True
            else:
                speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)

            # 🎯 Step1: Start processing new timeline
            chunk_start_time = parse_df_srt_time(chunk_df.iloc[0]['start_time'])
            chunk_end_time = parse_df_srt_time(chunk_df.iloc[-1]['end_time']) + chunk_df.iloc[-1]['tolerance'] * 1.5 # 增加50%的tolerance容差
            cur_time = chunk_start_time

            # Pre-calculate total duration to check if we need to adjust speed factor
            total_duration = 0
            for i, row in chunk_df.iterrows():
                number = row['number']
                temp_file_mp3 = TEMP_FILE_TEMPLATE.format(f"{number}", "mp3")
                temp_file_wav = TEMP_FILE_TEMPLATE.format(f"{number}", "wav")
                temp_file = temp_file_mp3 if os.path.exists(temp_file_mp3) else temp_file_wav
                duration = get_audio_duration(temp_file) / speed_factor
                total_duration += duration
                if i != 0 and keep_gaps:
                    total_duration += chunk_df.iloc[i-1]['gap']/speed_factor

            # If total duration exceeds chunk time, adjust speed factor
            available_time = chunk_end_time - chunk_start_time
            if total_duration > available_time and available_time > 0:
                adjusted_factor = speed_factor * (total_duration / available_time)
                # Ensure adjusted factor is positive and within reasonable limits
                adjusted_factor = max(min_speed, adjusted_factor)
                if adjusted_factor <= accept * 1.1:  # Only adjust if within reasonable limits
                    rprint(f"[yellow]⚠️ Adjusted speed factor from {speed_factor:.3f} to {adjusted_factor:.3f} for chunk {chunk_start} to {index}[/yellow]")
                    speed_factor = round(adjusted_factor, 3)

            # Process each row in the chunk
            for i, row in chunk_df.iterrows():
                # If i is not 0, which is not the first row of the chunk, cur_time needs to be added with the gap of the previous row, remember to divide by speed_factor
                if i != 0 and keep_gaps:
                    cur_time += chunk_df.iloc[i-1]['gap']/speed_factor
                new_sub_times = []
                number = row['number']
                # 🔄 Step2: Start speed change and save as OUTPUT_FILE_TEMPLATE
                temp_file_mp3 = TEMP_FILE_TEMPLATE.format(f"{number}", "mp3")
                temp_file_wav = TEMP_FILE_TEMPLATE.format(f"{number}", "wav")
                temp_file = temp_file_mp3 if os.path.exists(temp_file_mp3) else temp_file_wav
                output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}")

                # Verify the temp file exists and is valid
                if not os.path.exists(temp_file) or os.path.getsize(temp_file) < 10000:
                    rprint(f"[red]❌ Missing or invalid audio file: {temp_file}[/red]")
                    # Create a silent audio file as a fallback
                    silence = AudioSegment.silent(duration=1000)  # 1 second silence
                    silence.export(temp_file_wav, format="wav")

                adjust_audio_speed(temp_file, output_file, speed_factor)
                ad_dur = get_audio_duration(output_file)

                # Verify the output file is valid
                if ad_dur <= 0:
                    rprint(f"[red]❌ Invalid output audio duration: {output_file}[/red]")
                    # Use a minimum duration as fallback
                    ad_dur = 0.5

                new_sub_times.append([cur_time, cur_time+ad_dur])
                cur_time += ad_dur
                # 🔄 Step3: Find corresponding main DataFrame index and update new_sub_times
                main_df_idx = tasks_df[tasks_df['number'] == row['number']].index[0]
                tasks_df.at[main_df_idx, 'new_sub_times'] = new_sub_times

            # 🎯 Step4: Choose emoji based on speed_factor and accept comparison
            emoji = "⚡" if speed_factor <= accept else "⚠️"
            rprint(f"[cyan]{emoji} Processed chunk {chunk_start} to {index} with speed factor {speed_factor}[/cyan]")

            # 🔄 Step5: Check if the last row exceeds the range
            if cur_time > chunk_end_time:
                time_diff = cur_time - chunk_end_time
                if time_diff <= 3.0:  # Increased tolerance to 3.0s
                    rprint(f"[yellow]⚠️ Chunk {chunk_start} to {index} exceeds by {time_diff:.3f}s (tolerance: 3.0s), truncating last audio[/yellow]")
                    # Get the last audio file (use same format as process_row)
                    last_number = tasks_df.iloc[index]['number']
                    last_file = OUTPUT_FILE_TEMPLATE.format(f"{last_number}")

                    # Calculate the duration to keep
                    try:
                        audio = AudioSegment.from_wav(last_file)
                        original_duration = len(audio) / 1000  # Convert to seconds
                        new_duration = max(0.5, original_duration - time_diff)  # Ensure at least 0.5s
                        trimmed_audio = audio[:(new_duration * 1000)]  # pydub uses milliseconds
                        trimmed_audio.export(last_file, format="wav")

                        # Update the last timestamp
                        last_times = tasks_df.at[index, 'new_sub_times']
                        last_times[-1][1] = chunk_end_time
                        tasks_df.at[index, 'new_sub_times'] = last_times
                    except Exception as e:
                        rprint(f"[red]❌ Failed to trim audio: {str(e)}[/red]")
                        # Create a fallback solution - just adjust the timestamps
                        last_times = tasks_df.at[index, 'new_sub_times']
                        last_times[-1][1] = chunk_end_time
                        tasks_df.at[index, 'new_sub_times'] = last_times
                else:
                    # Try to fix by increasing speed factor
                    required_speed = cur_time / chunk_end_time
                    current_speed = load_key("speed_factor.accept")

                    # If required speed is within 20% of accept, try to fix automatically
                    if required_speed <= current_speed * 1.2:
                        rprint(f"[yellow]⚠️ Chunk {chunk_start} to {index} exceeds by {time_diff:.3f}s, attempting automatic fix[/yellow]")
                        # Reprocess the chunk with the required speed
                        new_speed_factor = min(required_speed * 1.05, current_speed * 1.2)  # Add 5% safety margin, cap at 20% over accept

                        # Reset timeline
                        cur_time = chunk_start_time

                        # Reprocess each row with new speed factor
                        for i, row in chunk_df.iterrows():
                            if i != 0 and keep_gaps:
                                cur_time += chunk_df.iloc[i-1]['gap']/new_speed_factor
                            new_sub_times = []
                            number = row['number']
                            temp_file_mp3 = TEMP_FILE_TEMPLATE.format(f"{number}", "mp3")
                            temp_file_wav = TEMP_FILE_TEMPLATE.format(f"{number}", "wav")
                            temp_file = temp_file_mp3 if os.path.exists(temp_file_mp3) else temp_file_wav
                            output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}")
                            adjust_audio_speed(temp_file, output_file, new_speed_factor)
                            ad_dur = get_audio_duration(output_file)
                            new_sub_times.append([cur_time, cur_time+ad_dur])
                            cur_time += ad_dur
                            main_df_idx = tasks_df[tasks_df['number'] == row['number']].index[0]
                            tasks_df.at[main_df_idx, 'new_sub_times'] = new_sub_times

                        rprint(f"[green]✅ Successfully fixed chunk {chunk_start} to {index} with speed factor {new_speed_factor:.3f}[/green]")
                    else:
                        # If we can't fix automatically, raise an exception with detailed information
                        raise Exception(
                            f"Chunk {chunk_start} to {index} exceeds chunk end time {chunk_end_time:.2f}s (current: {cur_time:.2f}s)\n"
                            f"Required speed factor: {required_speed:.2f}x (current accept: {current_speed:.2f}x)\n"
                            f"Possible solutions:\n"
                            f"1. Increase 'speed_factor.accept' in config to at least {required_speed:.2f}\n"
                            f"2. Reduce TTS audio duration by editing text or using different TTS method\n"
                            f"3. Adjust chunk boundaries in the task file to allow more time"
                        )
            chunk_start = index+1

    rprint("[bold green]✅ Audio chunks processing completed![/bold green]")
    return tasks_df

def clean_invalid_audio_files() -> None:
    """Clean up invalid audio files in both temp and segs directories"""
    rprint("[bold blue]🧹 Checking for invalid audio files...[/bold blue]")
    invalid_count = 0
    small_files = []

    # Check directories
    for directory in [TEMP_DIR, SEGS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            continue

        # Check all files in the directory
        for filename in os.listdir(directory):
            if not (filename.endswith('.wav') or filename.endswith('.mp3')):
                continue

            file_path = os.path.join(directory, filename)
            file_size = os.path.getsize(file_path)

            # Check if file is too small (likely invalid)
            if file_size < 10000:
                small_files.append((file_path, file_size))
                invalid_count += 1
            else:
                # Verify the file is a valid audio file
                try:
                    # Try to load the file with pydub to verify it's a valid audio file
                    audio = AudioSegment.from_file(file_path)
                    if len(audio) < 100:  # Less than 100ms is likely invalid
                        small_files.append((file_path, file_size))
                        invalid_count += 1
                        rprint(f"[yellow]⚠️ File has valid size but invalid duration: {file_path}[/yellow]")
                except Exception as e:
                    # File is corrupted or not a valid audio file
                    small_files.append((file_path, file_size))
                    invalid_count += 1
                    rprint(f"[red]❌ Corrupted audio file: {file_path} - {str(e)}[/red]")

    # Report findings
    if invalid_count > 0:
        rprint(f"[yellow]⚠️ Found {invalid_count} invalid audio files[/yellow]")
        for file_path, file_size in small_files:
            rprint(f"  - {file_path} ({file_size} bytes)")
            os.remove(file_path)
        rprint(f"[green]✅ Removed {invalid_count} invalid audio files[/green]")
    else:
        rprint("[green]✅ No invalid audio files found[/green]")

def gen_audio() -> None:
    """Main function: Generate audio and process timeline"""
    rprint("[bold magenta]🚀 Starting audio generation process...[/bold magenta]")

    # 🎯 Step1: Create necessary directories
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(SEGS_DIR, exist_ok=True)

    # 🧹 Step1.5: Clean up invalid audio files
    clean_invalid_audio_files()

    # 📝 Step2: Load task file
    tasks_df = pd.read_excel(TASKS_FILE)
    rprint("[green]📊 Loaded task file successfully[/green]")

    # 🔊 Step3: Generate TTS audio
    tasks_df = generate_tts_audio(tasks_df)

    # 🔄 Step4: Merge audio chunks
    tasks_df = merge_chunks(tasks_df)

    # 💾 Step5: Save results
    tasks_df.to_excel(OUTPUT_FILE, index=False)
    rprint("[bold green]🎉 Audio generation completed successfully![/bold green]")

if __name__ == "__main__":
    gen_audio()
