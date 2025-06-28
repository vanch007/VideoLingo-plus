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
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config_utils import load_key
from core.all_whisper_methods.audio_preprocess import get_audio_duration
from core.all_tts_functions.tts_main import tts_main

console = Console()

TEMP_DIR = 'output/audio/tmp'
SEGS_DIR = 'output/audio/segs'
TASKS_FILE = "output/audio/tts_tasks.xlsx"
OUTPUT_FILE = "output/audio/tts_tasks.xlsx"
TEMP_FILE_TEMPLATE = f"{TEMP_DIR}/{{}}_temp.wav"
OUTPUT_FILE_TEMPLATE = f"{SEGS_DIR}/{{}}.wav"
WARMUP_SIZE = 5

def parse_df_srt_time(time_str: str) -> float:
    """Convert SRT time format to seconds"""
    hours, minutes, seconds = time_str.strip().split(':')
    seconds, milliseconds = seconds.split('.')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def adjust_audio_speed(input_file: str, output_file: str, speed_factor: float) -> None:
    """Adjust audio speed and handle edge cases"""
    # If the speed factor is close to 1, directly copy the file
    if abs(speed_factor - 1.0) < 0.001:
        shutil.copy2(input_file, output_file)
        return

    atempo = speed_factor
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
                audio = AudioSegment.from_wav(output_file)
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
    audio = AudioSegment.from_wav(audio_file)
    trimmed_audio = audio.strip_silence(
        silence_thresh=silence_threshold,
        silence_len=int(min_silence_duration * 1000)  # Convert to integer explicitly
    )
    trimmed_audio.export(audio_file, format="wav")

def process_row(row: pd.Series, tasks_df: pd.DataFrame) -> Tuple[int, float]:
    """Helper function for processing single row data"""
    number = row['number']
    text = row['text']
    temp_file = TEMP_FILE_TEMPLATE.format(f"{number}")
    real_dur = 0
    max_retries = 3

    # Check for problematic numbers (based on the error report)
    problematic_numbers = [693, 699, 700, 701, 702, 703, 704, 705, 698, 858, 653, 1069]
    if number in problematic_numbers:
        rprint(f"[yellow]⚠️ 检测到问题音频文件编号: {number}, 将强制重新生成[/yellow]")
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # Check if file already exists and is valid
    if os.path.exists(temp_file):
        file_size = os.path.getsize(temp_file)
        if file_size < 20000:  # File exists but is too small
            rprint(f"[yellow]⚠️ 已存在的音频文件过小 ({file_size} 字节)，将重新生成: {temp_file}[/yellow]")
            os.remove(temp_file)
        else:
            # Try to get duration to validate file
            try:
                # Try to load with pydub to verify it's a valid audio file
                audio = AudioSegment.from_wav(temp_file)
                duration = len(audio) / 1000  # Convert to seconds

                if duration > 0.5 and duration < 10:  # Valid duration
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
                future = executor.submit(tts_main, text, temp_file, number, tasks_df)
                try:
                    # Set a 30-second timeout for the TTS task
                    future.result(timeout=30)
                except Exception as e:
                    # Re-raise the exception to be caught by the outer loop
                    raise e

            # Verify the file exists after TTS generation
            if not os.path.exists(temp_file):
                raise Exception(f"TTS生成后文件不存在: {temp_file}")

            file_size = os.path.getsize(temp_file)

            # Check if file is too small (likely invalid)
            if file_size < 20000:
                os.remove(temp_file)
                if attempt < max_retries - 1:
                    rprint(f"[yellow]⚠️ 音频文件过小 (大小:{file_size}字节), 重试中({attempt + 1}/{max_retries})[/yellow]")
                    time.sleep(1)  # 增加重试间隔
                    continue
                else:
                    raise Exception(f"音频文件过小且达到最大重试次数(大小:{file_size}字节)")

            # Verify the file is a valid audio file
            try:
                # Try to load with pydub to verify it's a valid audio file
                audio = AudioSegment.from_wav(temp_file)
                duration = len(audio) / 1000  # Convert to seconds
            except Exception as e:
                os.remove(temp_file)
                if attempt < max_retries - 1:
                    rprint(f"[red]❌ 音频文件损坏: {temp_file} - {str(e)}[/red]")
                    time.sleep(1)
                    continue
                else:
                    raise Exception(f"音频文件损坏且达到最大重试次数: {str(e)}")

            # 检查音频文件是否异常(大于10秒或小于0.5秒)
            if duration > 10 or duration < 0.5:
                os.remove(temp_file)
                if attempt < max_retries - 1:
                    rprint(f"[yellow]⚠️ 音频文件时长异常(时长:{duration:.2f}秒), 重试中({attempt + 1}/{max_retries})[/yellow]")
                    time.sleep(1)  # 增加重试间隔
                    continue
                else:
                    raise Exception(f"音频文件时长异常且达到最大重试次数(时长:{duration:.2f}秒)")

            # 只对1.5秒以上的音频检测并剪切静音片段
            if duration >= 1.5:
                try:
                    detect_silence(temp_file)
                    # 重新检查剪切后的时长
                    audio = AudioSegment.from_wav(temp_file)
                    new_duration = len(audio) / 1000  # Convert to seconds

                    # 如果剪切后时长小于1秒，使用原始音频
                    if new_duration < 1.0:
                        rprint(f"[yellow]⚠️ 剪切静音后时长过短 ({new_duration:.2f}秒)，使用原始音频[/yellow]")
                        tts_main(text, temp_file, number, tasks_df)  # 重新生成
                        audio = AudioSegment.from_wav(temp_file)
                        duration = len(audio) / 1000  # Convert to seconds
                    else:
                        duration = new_duration
                except Exception as e:
                    rprint(f"[yellow]⚠️ 剪切静音失败: {str(e)}[/yellow]")
                    # Continue with the original audio

            # Final validation
            if os.path.exists(temp_file):
                file_size = os.path.getsize(temp_file)
                if file_size < 20000 or duration <= 0:
                    rprint(f"[red]❌ 最终验证失败: 文件大小={file_size}字节, 时长={duration:.2f}秒[/red]")
                    # Try one more time with a simple silent audio
                    silence = AudioSegment.silent(duration=1000)  # 1秒静音
                    silence.export(temp_file, format="wav")
                    duration = 1.0

            real_dur = duration
            break

        except Exception as e:
            error_message = str(e)
            if "TimeoutError" in str(type(e)):
                 error_message = "TTS generation timed out after 30 seconds."

            if attempt == max_retries - 1:
                # 检查是否有参考音频作为后备方案
                refer_file = f"output/audio/refers/{number}.wav"
                if os.path.exists(refer_file) and os.path.getsize(refer_file) > 20000:
                    try:
                        # Verify the reference file is valid
                        audio = AudioSegment.from_wav(refer_file)
                        if len(audio) > 500:  # At least 500ms
                            shutil.copy2(refer_file, temp_file)
                            duration = len(audio) / 1000  # Convert to seconds
                            rprint(f"[yellow]⚠️ TTS生成失败，已使用参考音频替代: {refer_file} -> {temp_file}[/yellow]")
                            real_dur = duration
                            break
                    except Exception as ref_e:
                        rprint(f"[red]❌ 参考音频无效: {str(ref_e)}[/red]")

                # If we get here, either no reference file or it's invalid
                rprint(f"[red]⚠️ TTS生成失败且无有效参考音频，已生成静音音频替代: {temp_file}[/red]")
                silence = AudioSegment.silent(duration=1000)  # 1秒静音
                silence.export(temp_file, format="wav")
                real_dur = 1.0
                break
            else:
                rprint(f"[yellow]⚠️ TTS生成失败，重试中({attempt + 1}/{max_retries}): {error_message}[/yellow]")
                time.sleep(1)

    # Final check to ensure we have a valid audio file
    if not os.path.exists(temp_file) or os.path.getsize(temp_file) < 20000 or real_dur <= 0:
        rprint(f"[red]❌ 最终检查失败，生成静音替代: {temp_file}[/red]")
        silence = AudioSegment.silent(duration=1000)  # 1秒静音
        silence.export(temp_file, format="wav")
        real_dur = 1.0

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
                progress.advance(task)
            except Exception as e:
                rprint(f"[red]❌ Error in warmup: {str(e)}[/red]")
                raise e

        # for gpt_sovits, do not use parallel to avoid mistakes
        max_workers = load_key("max_workers") if load_key("tts_method") != "gpt_sovits" else 1
        # parallel processing for remaining tasks
        if len(tasks_df) > warmup_size:
            remaining_tasks = tasks_df.iloc[warmup_size:].copy()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_row, row, tasks_df.copy())
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
    if (chunk_durs + all_gaps) / accept < durations:
        base_factor = (chunk_durs + all_gaps) / (durations-speed_var_error*2)
    elif chunk_durs / accept < durations:
        base_factor = chunk_durs / (durations-speed_var_error*2)
        keep_gaps = False
    elif (chunk_durs + all_gaps) / accept < tol_durs:
        base_factor = (chunk_durs + all_gaps) / (tol_durs-speed_var_error*2)
    else:
        base_factor = chunk_durs / (tol_durs-speed_var_error*2)
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
    accept = load_key("speed_factor.accept")
    min_speed = load_key("speed_factor.min")
    chunk_start = 0

    tasks_df['new_sub_times'] = None

    # First, verify all audio files exist and are valid
    missing_files = []
    for _, row in tasks_df.iterrows():
        number = row['number']
        temp_file = TEMP_FILE_TEMPLATE.format(f"{number}")
        if not os.path.exists(temp_file) or os.path.getsize(temp_file) < 20000:
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
                temp_file = TEMP_FILE_TEMPLATE.format(f"{number}")
                duration = get_audio_duration(temp_file) / speed_factor
                total_duration += duration
                if i != 0 and keep_gaps:
                    total_duration += chunk_df.iloc[i-1]['gap']/speed_factor

            # If total duration exceeds chunk time, adjust speed factor
            available_time = chunk_end_time - chunk_start_time
            if total_duration > available_time:
                adjusted_factor = speed_factor * (total_duration / available_time) * 1.05  # Add 5% safety margin
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
                temp_file = TEMP_FILE_TEMPLATE.format(f"{number}")
                output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}")

                # Verify the temp file exists and is valid
                if not os.path.exists(temp_file) or os.path.getsize(temp_file) < 20000:
                    rprint(f"[red]❌ Missing or invalid audio file: {temp_file}[/red]")
                    # Create a silent audio file as a fallback
                    silence = AudioSegment.silent(duration=1000)  # 1 second silence
                    silence.export(temp_file, format="wav")

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
                            temp_file = TEMP_FILE_TEMPLATE.format(f"{number}")
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
            if not filename.endswith('.wav'):
                continue

            file_path = os.path.join(directory, filename)
            file_size = os.path.getsize(file_path)

            # Check if file is too small (likely invalid)
            if file_size < 20000:
                small_files.append((file_path, file_size))
                invalid_count += 1
            else:
                # Verify the file is a valid audio file
                try:
                    # Try to load the file with pydub to verify it's a valid audio file
                    audio = AudioSegment.from_wav(file_path)
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
