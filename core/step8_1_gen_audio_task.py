import pandas as pd
import datetime
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
from core.ask_gpt import ask_gpt
from core.prompts_storage import get_subtitle_trim_prompt
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from core.config_utils import load_key
from core.all_tts_functions.estimate_duration import init_estimator, estimate_duration

console = Console()
speed_factor = load_key("speed_factor")

TRANS_SUBS_FOR_AUDIO_FILE = 'output/audio/trans_subs_for_audio.srt'
SRC_SUBS_FOR_AUDIO_FILE = 'output/audio/src_subs_for_audio.srt'
SOVITS_TASKS_FILE = 'output/audio/tts_tasks.xlsx'
SRC_SRT = 'output/src.srt'
TRANS_SRT = 'output/trans.srt'
ESTIMATOR = None

def check_len_then_trim(text, duration):
    global ESTIMATOR
    if ESTIMATOR is None:
        ESTIMATOR = init_estimator()
    estimated_duration = estimate_duration(text, ESTIMATOR) / speed_factor['max']

    console.print(f"Subtitle text: {text}, "
                  f"[bold green]Estimated reading duration: {estimated_duration:.2f} seconds[/bold green]")

    if estimated_duration > duration:
        rprint(Panel(f"Estimated reading duration {estimated_duration:.2f} seconds exceeds given duration {duration:.2f} seconds, shortening...", title="Processing", border_style="yellow"))
        original_text = text
        prompt = get_subtitle_trim_prompt(text, duration)
        def valid_trim(response):
            if 'result' not in response:
                return {'status': 'error', 'message': 'No result in response'}
            return {'status': 'success', 'message': ''}
        try:
            response = ask_gpt(prompt, response_json=True, log_title='subtitle_trim', valid_def=valid_trim)
            shortened_text = response['result']
        except Exception:
            rprint("[bold red]🚫 AI refused to answer due to sensitivity, so manually remove punctuation[/bold red]")
            shortened_text = re.sub(r'[,.!?;:，。！？；：]', ' ', text).strip()
        rprint(Panel(f"Subtitle before shortening: {original_text}\nSubtitle after shortening: {shortened_text}", title="Subtitle Shortening Result", border_style="green"))
        return shortened_text
    else:
        return text

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
            
            # Calculate duration if not present or valid
            duration = row.get('duration', 0)
            if duration <= 0:
                duration = time_diff_seconds(start_time, end_time, datetime.date.today())
            
            text = str(row['Translation'])
            # Remove content within parentheses
            text = re.sub(r'\([^)]*\)', '', text).strip()
            text = re.sub(r'（[^）]*）', '', text).strip()
            text = text.replace('-', '')
            
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
        with open(TRANS_SRT, 'r', encoding='utf-8') as file:
            content = file.read()

        with open(SRC_SRT, 'r', encoding='utf-8') as src_file:
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

    # check and trim subtitle length, for twice to ensure the subtitle length is within the limit
    # df['text'] = df.apply(lambda x: check_len_then_trim(x['text'], x['duration']), axis=1)

    return df

def gen_audio_task_main():
    if os.path.exists(SOVITS_TASKS_FILE):
        rprint(Panel(f"{SOVITS_TASKS_FILE} already exists, skip.", title="Info", border_style="blue"))
    else:
        df = process_srt()
        console.print(df)
        df.to_excel(SOVITS_TASKS_FILE, index=False)
        rprint(Panel(f"Successfully generated {SOVITS_TASKS_FILE}", title="Success", border_style="green"))

if __name__ == '__main__':
    gen_audio_task_main()
