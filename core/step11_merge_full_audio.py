import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pandas as pd
import numpy as np
import subprocess
from pydub import AudioSegment
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from core.config_utils import load_key
from core.dubbing_quality import DUBBING_MERGE_ISSUES_JSON, get_quality_config, normalize_lines, parse_list, write_dubbing_eval
console = Console()

INPUT_EXCEL = 'output/audio/tts_tasks.xlsx'
DUB_VOCAL_FILE = 'output/dub.mp3'

DUB_SUB_FILE = 'output/dub.srt'
SEGS_DIR = 'output/audio/segs'
OUTPUT_FILE_TEMPLATE = f"{SEGS_DIR}/{{}}.wav"

def load_and_flatten_data(excel_file):
    """Load Excel data and flatten timestamps"""
    df = pd.read_excel(excel_file)
    
    # Flatten lines (translated text)
    if 'lines' in df.columns:
        lines = [normalize_lines(line) for line in df['lines'].tolist()]
        lines = [item for sublist in lines for item in sublist]
    else:
        # Fallback to 'text' if 'lines' is missing (should not happen with new step8_2)
        lines = df['text'].tolist()

    # Flatten src_lines (original text)
    if 'src_lines' in df.columns:
        src_lines = [parse_list(line) for line in df['src_lines'].tolist()]
        # Handle potential None values or empty lists if any
        src_lines = [item for sublist in src_lines for item in (sublist if sublist is not None else [])]
    else:
        # Fallback to 'origin' if 'src_lines' is missing
        src_lines = df['origin'].tolist()

    # Flatten timestamps
    if 'new_sub_times' in df.columns:
        times = [parse_list(t) for t in df['new_sub_times'].tolist()]
        # Flatten if it's list of lists (step10 generates list of lists)
        if times and isinstance(times[0], list):
             times = [item for sublist in times for item in sublist]
    else:
         times = [parse_list(t) for t in df['sub_times'].tolist()]
    
    return df, lines, src_lines, times

def get_audio_files(df):
    """Generate a list of audio file paths"""
    audios = []
    for index, row in df.iterrows():
        number = row['number']
        if 'lines' in df.columns:
            lines_data = normalize_lines(row.get('lines', row.get('text', '')))
            line_count = len(lines_data)
            for line_index in range(line_count):
                audio_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
                audios.append(audio_file)
        else:
            audio_file = OUTPUT_FILE_TEMPLATE.format(f"{number}")
            audios.append(audio_file)
    return audios

def process_audio_segment(audio_file, allow_silence_fallback=False):
    """Process a single audio segment with MP3 compression"""
    # 检查文件是否存在且有效
    if not os.path.exists(audio_file):
        console.print(f"[bold red]❌ Audio file does not exist: {audio_file}[/bold red]")
        if not allow_silence_fallback:
            raise FileNotFoundError(audio_file)
        # 创建一个100ms的静音文件作为替代
        silence = AudioSegment.silent(duration=100)
        temp_file = f"{audio_file}_temp_silence.wav"
        silence.export(temp_file, format="wav")
        audio_file = temp_file
    elif os.path.getsize(audio_file) < 1000:  # 文件太小，可能已损坏
        console.print(f"[bold yellow]⚠️ Audio file too small ({os.path.getsize(audio_file)} bytes): {audio_file}[/bold yellow]")
        if not allow_silence_fallback:
            raise ValueError(f"Audio file too small: {audio_file}")
        # 创建一个100ms的静音文件作为替代
        silence = AudioSegment.silent(duration=100)
        temp_file = f"{audio_file}_temp_silence.wav"
        silence.export(temp_file, format="wav")
        audio_file = temp_file
    
    temp_file = f"{audio_file}_temp.mp3"
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', audio_file,
        '-ar', '16000',  # 固定采样率为16kHz
        '-ac', '1',      # 单声道
        '-b:a', '64k',   # 比特率64kbps
        '-f', 'mp3',
        temp_file
    ]
    try:
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ FFmpeg处理失败: {e.stderr.decode('utf-8') if e.stderr else str(e)}[/bold red]")
        # 如果FFmpeg处理失败，尝试直接使用pydub加载
        try:
            audio_segment = AudioSegment.from_file(audio_file)
            audio_segment.export(temp_file, format="mp3", parameters=["-ar", "16000", "-ac", "1", "-b:a", "64k"])
        except Exception as pydub_error:
            console.print(f"[bold red]❌ Pydub处理也失败了: {str(pydub_error)}[/bold red]")
            if not allow_silence_fallback:
                raise
            # 创建一个100ms的静音文件作为最终替代
            silence = AudioSegment.silent(duration=100)
            silence.export(temp_file, format="mp3", parameters=["-ar", "16000", "-ac", "1", "-b:a", "64k"])
    audio_segment = AudioSegment.from_mp3(temp_file)
    os.remove(temp_file)
    
    # 如果是临时静音文件，也删除它
    if "_temp_silence.wav" in audio_file:
        os.remove(audio_file)
        
    return audio_segment


def _write_merge_issues(merge_issues):
    if not merge_issues:
        if os.path.exists(DUBBING_MERGE_ISSUES_JSON):
            os.remove(DUBBING_MERGE_ISSUES_JSON)
        return
    os.makedirs(os.path.dirname(DUBBING_MERGE_ISSUES_JSON), exist_ok=True)
    with open(DUBBING_MERGE_ISSUES_JSON, "w", encoding="utf-8") as f:
        json.dump(merge_issues, f, ensure_ascii=False, indent=2)

def merge_audio_segments(audios, new_sub_times, sample_rate):
    quality = get_quality_config()
    total_duration_ms = int(max(end for _, end in new_sub_times) * 1000) if new_sub_times else 0
    merged_audio = AudioSegment.silent(duration=total_duration_ms, frame_rate=sample_rate)
    merge_issues = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        merge_task = progress.add_task("🎵 Merging audio segments...", total=len(audios))
        
        for i, (audio_file, time_range) in enumerate(zip(audios, new_sub_times)):
            try:
                audio_segment = process_audio_segment(audio_file, allow_silence_fallback=quality.allow_silence_fallback)
                start_time, end_time = time_range
                merged_audio = merged_audio.overlay(audio_segment, position=max(0, int(start_time * 1000)))
                
            except Exception as e:
                console.print(f"[bold red]❌ Error processing {audio_file}: {str(e)}[/bold red]")
                merge_issues.append({
                    "audio_file": audio_file,
                    "time_range": time_range,
                    "error": str(e),
                    "silence_fallback": quality.allow_silence_fallback,
                })
                if quality.allow_silence_fallback:
                    silent_segment = AudioSegment.silent(duration=100, frame_rate=sample_rate)
                    merged_audio = merged_audio.overlay(silent_segment, position=max(0, int(time_range[0] * 1000)))
                else:
                    _write_merge_issues(merge_issues)
                    raise

            progress.advance(merge_task)

    _write_merge_issues(merge_issues)
    return merged_audio

def create_srt_subtitle():
    # Correctly unpack all four returned values
    df, lines, _, new_sub_times = load_and_flatten_data(INPUT_EXCEL) 
    
    with open(DUB_SUB_FILE, 'w', encoding='utf-8') as f:
        for i, ((start_time, end_time), line) in enumerate(zip(new_sub_times, lines), 1):
            start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},{int((start_time*1000)%1000):03d}"
            end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},{int((end_time*1000)%1000):03d}"
            
            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{line}\n\n")
    
    rprint(f"[bold green]✅ Dub subtitle file created: {DUB_SUB_FILE}[/bold green]")

def create_orig_srt_subtitle():
    """Create SRT subtitle file using original text"""
    DUB_ORIG_SRT_FILE = 'output/dub_orig.srt'
    df, _, origins, new_sub_times = load_and_flatten_data(INPUT_EXCEL) # Get origins text
    
    with open(DUB_ORIG_SRT_FILE, 'w', encoding='utf-8') as f:
        for i, ((start_time, end_time), line) in enumerate(zip(new_sub_times, origins), 1): # Use origins text
            start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},{int((start_time*1000)%1000):03d}"
            end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},{int((end_time*1000)%1000):03d}"
            
            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{line}\n\n") # Write origin line
    
    rprint(f"[bold green]✅ Original subtitle file created: {DUB_ORIG_SRT_FILE}[/bold green]")


def merge_full_audio():
    """Main function: Process the complete audio merging process"""
    console.print("\n[bold cyan]🎬 Starting audio merging process...[/bold cyan]")
    
    with console.status("[bold cyan]📊 Loading data from Excel...[/bold cyan]"):
        df, lines, origins, new_sub_times = load_and_flatten_data(INPUT_EXCEL) # Load origins too
    console.print("[bold green]✅ Data loaded successfully[/bold green]")
    
    with console.status("[bold cyan]🔍 Getting audio file list...[/bold cyan]"):
        audios = get_audio_files(df)
    console.print(f"[bold green]✅ Found {len(audios)} audio segments[/bold green]")
    
    with console.status("[bold cyan]📝 Generating subtitle files...[/bold cyan]"):
        create_srt_subtitle()
        create_orig_srt_subtitle() # Generate original subtitle file
    
    if not os.path.exists(audios[0]):
        console.print(f"[bold red]❌ Error: First audio file {audios[0]} does not exist![/bold red]")
        raise FileNotFoundError(audios[0])
    
    sample_rate = 16000
    console.print(f"[bold green]✅ Sample rate: {sample_rate}Hz[/bold green]")

    console.print("[bold cyan]🔄 Starting audio merge process...[/bold cyan]")
    try:
        merged_audio = merge_audio_segments(audios, new_sub_times, sample_rate)
    except Exception as e:
        console.print(f"[bold red]❌ Error during audio merging: {str(e)}[/bold red]")
        if not get_quality_config().allow_silence_fallback:
            raise
        console.print("[bold yellow]⚠️  Creating a merged audio with silence as fallback...[/bold yellow]")
        # 计算总时长并创建静音音频作为后备方案
        total_duration_ms = int(new_sub_times[-1][1] * 1000)
        merged_audio = AudioSegment.silent(duration=total_duration_ms, frame_rate=sample_rate)
    
    with console.status("[bold cyan]💾 Exporting final audio file...[/bold cyan]"):
        merged_audio = merged_audio.set_frame_rate(16000).set_channels(1)
        merged_audio.export(
            DUB_VOCAL_FILE, 
            format="mp3",
            parameters=["-b:a", "64k"]
        )
        summary = write_dubbing_eval(df)
    console.print(f"[bold green]✅ Audio file successfully merged![/bold green]")
    console.print(f"[bold green]📁 Output file: {DUB_VOCAL_FILE}[/bold green]")
    console.print(f"[bold green]📊 Dubbing eval: {summary}[/bold green]")

if __name__ == "__main__":
    merge_full_audio()
