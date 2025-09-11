import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import pandas as pd
import soundfile as sf
console = Console()
from core.all_whisper_methods.demucs_vl import demucs_main, VOCAL_AUDIO_FILE

# Simplified path definitions
REF_DIR = 'output/audio/refers'
SEG_DIR = 'output/audio/segs'
TASKS_FILE = 'output/audio/tts_tasks.xlsx'

# --- Robust Time Helper Functions ---
def time_str_to_seconds(time_str):
    """Converts HH:MM:SS.ms or HH:MM:SS,ms string to seconds."""
    if not isinstance(time_str, str):
        return 0.0
    time_str = time_str.replace(',', '.') # Standardize to dot
    try:
        main_part, ms_part = time_str.split('.')
        h, m, s = main_part.split(':')
        # Pad ms_part to 3 digits for consistent calculation
        ms = int(ms_part.ljust(3, '0')[:3])
        return int(h) * 3600 + int(m) * 60 + int(s) + ms / 1000.0
    except ValueError:
        try:
            # Case without milliseconds
            h, m, s = time_str.split(':')
            return float(int(h) * 3600 + int(m) * 60 + int(s))
        except ValueError:
            return 0.0

def seconds_to_time_str(seconds):
    """Converts seconds to HH:MM:SS,ms string for FFmpeg compatibility."""
    seconds = max(0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds * 1000) % 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def time_to_samples(time_str, sr):
    """Unified time conversion function using the robust helper."""
    seconds = time_str_to_seconds(time_str)
    return int(seconds * sr)
# --- End of Helper Functions ---

def extract_audio(audio_data, sr, start_time, end_time, out_file):
    """Simplified audio extraction function"""
    start = time_to_samples(start_time, sr)
    end = time_to_samples(end_time, sr)
    # Prevent slicing beyond the audio length or creating an empty slice
    if start >= end:
        rprint(f"[bold red]Warning: Invalid time slice for {out_file}. Start: {start_time}, End: {end_time}. Skipping.[/bold red]")
        return
    end = min(end, len(audio_data))
    sf.write(out_file, audio_data[start:end], sr)

def extract_refer_audio_main():
    demucs_main() #!!! in case demucs is not run

    os.makedirs(REF_DIR, exist_ok=True)
    
    df = pd.read_excel(TASKS_FILE)
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
        task = progress.add_task("[green]Extracting audio segments (fixed)...", total=len(df))
        
        for i in range(len(df)):
            current_row = df.iloc[i]
            start_time_str = str(current_row['start_time'])
            
            current_end_sec = time_str_to_seconds(str(current_row['end_time']))
            new_end_sec = current_end_sec + 1.0

            if i < len(df) - 1:
                next_row = df.iloc[i+1]
                next_start_sec = time_str_to_seconds(str(next_row['start_time']))
                if new_end_sec > next_start_sec:
                    new_end_sec = next_start_sec
            
            new_end_sec = min(new_end_sec, total_audio_duration_sec)

            final_end_time_str = seconds_to_time_str(new_end_sec)

            out_file = os.path.join(REF_DIR, f"{current_row['number']}.wav")
            extract_audio(data, sr, start_time_str, final_end_time_str, out_file)
            progress.update(task, advance=1)
            
    rprint(Panel(f"Audio segments saved to {REF_DIR} with fixed logic", title="Success", border_style="green"))

if __name__ == "__main__":
    extract_refer_audio_main()