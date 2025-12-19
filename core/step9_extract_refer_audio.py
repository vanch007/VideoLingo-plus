import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import pandas as pd
import soundfile as sf

from core.timing_utils import srt_time_to_seconds, seconds_to_srt_time, time_to_samples
from core.constants import REFERS_DIR, TTS_TASKS_FILE
from core.all_whisper_methods.demucs_vl import demucs_main, VOCAL_AUDIO_FILE

console = Console()


def extract_audio(audio_data, sr, start_time, end_time, out_file):
    """从音频数据中提取指定时间范围的片段"""
    start = time_to_samples(start_time, sr)
    end = time_to_samples(end_time, sr)
    # 防止切片超出音频长度或创建空切片
    if start >= end:
        rprint(f"[bold red]Warning: Invalid time slice for {out_file}. Start: {start_time}, End: {end_time}. Skipping.[/bold red]")
        return
    end = min(end, len(audio_data))
    sf.write(out_file, audio_data[start:end], sr)


def extract_refer_audio_main():
    """提取参考音频的主函数"""
    demucs_main()  # 确保 demucs 已运行

    os.makedirs(REFERS_DIR, exist_ok=True)
    
    df = pd.read_excel(TTS_TASKS_FILE)
    # Note: Filtering is done in step8_1, so df is already clean
    
    if not os.path.exists(VOCAL_AUDIO_FILE):
        rprint(Panel(f"[bold red]Error: Vocal audio file not found at {VOCAL_AUDIO_FILE}[/bold red]", title="Error"))
        return
        
    data, sr = sf.read(VOCAL_AUDIO_FILE)
    total_audio_duration_sec = len(data) / sr
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("[green]Extracting audio segments with optimized logic...", total=len(df))
        
        for i in range(len(df)):
            current_row = df.iloc[i]
            current_start_sec = srt_time_to_seconds(str(current_row['start_time']))
            current_end_sec = srt_time_to_seconds(str(current_row['end_time']))

            # --- 计算 new_start_sec ---
            new_start_sec = current_start_sec - 1.0

            if i > 0:
                prev_row = df.iloc[i-1]
                prev_end_sec = srt_time_to_seconds(str(prev_row['end_time']))
                if new_start_sec < prev_end_sec:
                    # 中点计算
                    new_start_sec = prev_end_sec + (current_start_sec - prev_end_sec) / 2
            
            # 确保开始时间不为负
            new_start_sec = max(0, new_start_sec)

            # --- 计算 new_end_sec ---
            new_end_sec = current_end_sec + 1.0

            if i < len(df) - 1:
                next_row = df.iloc[i+1]
                next_start_sec = srt_time_to_seconds(str(next_row['start_time']))
                if new_end_sec > next_start_sec:
                    # 中点计算
                    new_end_sec = current_end_sec + (next_start_sec - current_end_sec) / 2
            
            # 确保结束时间不超过总时长
            new_end_sec = min(new_end_sec, total_audio_duration_sec)

            # 转换回时间字符串
            final_start_time_str = seconds_to_srt_time(new_start_sec)
            final_end_time_str = seconds_to_srt_time(new_end_sec)

            out_file = os.path.join(REFERS_DIR, f"{current_row['number']}.wav")
            extract_audio(data, sr, final_start_time_str, final_end_time_str, out_file)
            progress.update(task, advance=1)
            
    rprint(Panel(f"Audio segments saved to {REFERS_DIR} with optimized logic", title="Success", border_style="green"))


if __name__ == "__main__":
    extract_refer_audio_main()