import pandas as pd
import datetime
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re

from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from core.config_utils import load_key
from core.constants import TRANS_SRT, SRC_SRT, TTS_TASKS_FILE, TRANS_SUBS_FOR_AUDIO_FILE, SRC_SUBS_FOR_AUDIO_FILE

console = Console()


def time_diff_seconds(t1: datetime.time, t2: datetime.time, base_date: datetime.date) -> float:
    """Calculate the difference in seconds between two time objects"""
    dt1 = datetime.datetime.combine(base_date, t1)
    dt2 = datetime.datetime.combine(base_date, t2)
    return (dt2 - dt1).total_seconds()

def process_srt():
    """Process srt file or excel timeline, generate audio tasks"""
    
    excel_path = 'output/audio/final_timeline.xlsx'
    if os.path.exists(excel_path):
        rprint(Panel(f"Found {excel_path}, loading tasks from Excel...", title="Info", border_style="cyan"))
        df_excel = pd.read_excel(excel_path)
        
        subtitles = []
        for i, row in df_excel.iterrows():
            # Parse timestamp "00:00:00,000 --> 00:00:00,000"
            time_str = row['timestamp']
            start_str, end_str = time_str.split(' --> ')
            start_time = datetime.datetime.strptime(start_str, '%H:%M:%S,%f').time()
            end_time = datetime.datetime.strptime(end_str, '%H:%M:%S,%f').time()
            
            # Final timeline timestamps are the source of truth for dubbing budgets.
            duration = time_diff_seconds(start_time, end_time, datetime.date.today())
            
            # Handle potential NaN/float values in Translation column
            raw_text = row['Translation']
            if pd.isna(raw_text):
                text = ""
            else:
                text = str(raw_text)
                
            # Remove content within parentheses
            text = re.sub(r'\([^)]*\)', '', text).strip()
            text = re.sub(r'（[^）]*）', '', text).strip()
            text = text.replace('-', '')
            
            # Filter out pure punctuation/whitespace (e.g., "、", "。", ", ", etc.)
            # Keep only text that contains at least one alphanumeric or CJK character
            if text and not re.search(r'[\w\u4e00-\u9fff]', text):
                text = ""
            
            subtitles.append({
                'number': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'text': text,
                'origin': str(row['Source']),
                'speaker': row.get('speaker', None)
            })
        
        df = pd.DataFrame(subtitles)
        
    else:
        rprint(Panel("Excel timeline not found, falling back to SRT parsing...", title="Info", border_style="yellow"))
        with open(TRANS_SUBS_FOR_AUDIO_FILE, 'r', encoding='utf-8') as file:
            content = file.read()

        with open(SRC_SUBS_FOR_AUDIO_FILE, 'r', encoding='utf-8') as src_file:
            src_content = src_file.read()

        subtitles = []
        src_subtitles = {}

        for block in src_content.strip().split('\n\n'):
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if len(lines) < 3:
                continue

            number = int(lines[0])
            src_text = ' '.join(lines[2:])
            src_subtitles[number] = src_text

        for block in content.strip().split('\n\n'):
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if len(lines) < 3:
                continue

            try:
                number = int(lines[0])
                start_time, end_time = lines[1].split(' --> ')
                start_time = datetime.datetime.strptime(start_time, '%H:%M:%S,%f').time()
                end_time = datetime.datetime.strptime(end_time, '%H:%M:%S,%f').time()
                duration = time_diff_seconds(start_time, end_time, datetime.date.today())
                text = ' '.join(lines[2:])
                # Remove content within parentheses (including English and Chinese parentheses)
                text = re.sub(r'\([^)]*\)', '', text).strip()
                text = re.sub(r'（[^）]*）', '', text).strip()
                # Remove only '-' character, keep other punctuation
                text = text.replace('-', '')
                
                # Filter out pure punctuation/whitespace (e.g., "、", "。", ", ", etc.)
                # Keep only text that contains at least one alphanumeric or CJK character
                if text and not re.search(r'[\w\u4e00-\u9fff]', text):
                    text = ""

                # Add the original text from src_subs_for_audio.srt
                origin = src_subtitles.get(number, '')

            except ValueError as e:
                rprint(Panel(f"Unable to parse subtitle block '{block}', error: {str(e)}, skipping this subtitle block.", title="Error", border_style="red"))
                continue

            subtitles.append({'number': number, 'start_time': start_time, 'end_time': end_time, 'duration': duration, 'text': text, 'origin': origin, 'speaker': None})

        df = pd.DataFrame(subtitles)

    # Add sub_times column
    df['sub_times'] = df.apply(lambda row: [
        (row['start_time'].hour * 3600 + row['start_time'].minute * 60 + row['start_time'].second + row['start_time'].microsecond / 1_000_000),
        (row['end_time'].hour * 3600 + row['end_time'].minute * 60 + row['end_time'].second + row['end_time'].microsecond / 1_000_000)
    ], axis=1)

    df['start_time'] = df['start_time'].apply(lambda x: x.strftime('%H:%M:%S.%f')[:-3])
    df['end_time'] = df['end_time'].apply(lambda x: x.strftime('%H:%M:%S.%f')[:-3])

    return df

def save_to_srt(df, srt_path):
    """Save DataFrame to SRT file"""
    with open(srt_path, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            # Convert time format from 00:00:00.000 to 00:00:00,000 for SRT
            start = row['start_time'].replace('.', ',')
            end = row['end_time'].replace('.', ',')
            
            # Ensure text is string and handle NaN
            text = row['text']
            if pd.isna(text):
                text = " "
            else:
                text = str(text)
                if not text.strip():
                    text = " "
            
            f.write(f"{row['number']}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")
    rprint(f"[bold green]✅ Synced rewritten subtitles to {srt_path}[/bold green]")

def gen_audio_task_main():
    if os.path.exists(TTS_TASKS_FILE):
        rprint(Panel(f"{TTS_TASKS_FILE} already exists, skip.", title="Info", border_style="blue"))
    else:
        df = process_srt()
        
        # Filter out invalid rows before saving
        # This ensures tts_tasks.xlsx and trans.srt are consistent with each other
        initial_count = len(df)
        
        # Filter empty text rows
        df = df[df['text'].notna() & (df['text'].astype(str).str.strip() != '')]
        empty_filtered = initial_count - len(df)
        
        # Filter rows with duration <= 0 (invalid timestamps)
        df = df[df['duration'] > 0]
        df = df.reset_index(drop=True)
        duration_filtered = initial_count - empty_filtered - len(df)
        
        if empty_filtered > 0:
            rprint(f"[yellow]🗑️ Filtered out {empty_filtered} empty text rows[/yellow]")
        if duration_filtered > 0:
            rprint(f"[yellow]🗑️ Filtered out {duration_filtered} rows with duration <= 0[/yellow]")
        
        if len(df) == 0:
            rprint("[red]❌ No valid rows to generate TTS tasks![/red]")
            return
        
        console.print(df)
        df.to_excel(TTS_TASKS_FILE, index=False)
        rprint(Panel(f"Successfully generated {TTS_TASKS_FILE}", title="Success", border_style="green"))
        
        # Note: Disabled auto-sync to trans.srt when using pre-merged translations (Plan C)
        # save_to_srt(df, TRANS_SRT)

if __name__ == '__main__':
    gen_audio_task_main()
