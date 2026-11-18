import os
import sys
import gc
import time
import shutil
import json
import argparse
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.panel import Panel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from batch.utils.settings_check import check_settings
from batch.utils.video_processor import process_video
from core.config_utils import load_key, update_key

console = Console()

def record_and_update_config(source_language, target_language):
    original_source_lang = load_key('whisper.language', 'auto')
    original_target_lang = load_key('target_language', 'zh')
    if source_language and not pd.isna(source_language):
        update_key('whisper.language', str(source_language).strip())
        update_key('source_language', str(source_language).strip())
    if target_language and not pd.isna(target_language):
        update_key('target_language', str(target_language).strip())
    return original_source_lang, original_target_lang

def clean_workspace():
    output_dir = Path('output')
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
        except Exception:
            for item in output_dir.iterdir():
                try:
                    if item.is_dir(): shutil.rmtree(item)
                    else: item.unlink()
                except Exception: pass
    output_dir.mkdir(parents=True, exist_ok=True)

def find_task_error_checkpoint(video_target):
    error_base = Path('batch/output/ERROR')
    if not error_base.exists():
        return None
    raw_name = os.path.splitext(os.path.basename(str(video_target)))[0]
    direct = error_base / raw_name
    if direct.is_dir() and (direct / 'pipeline_state.json').is_file():
        return str(direct)
    for folder in error_base.iterdir():
        if folder.is_dir():
            st_file = folder / 'pipeline_state.json'
            if st_file.is_file():
                try:
                    data = json.loads(st_file.read_text(encoding='utf-8'))
                    if str(data.get('input', '')).strip() == str(video_target).strip():
                        return str(folder)
                except Exception:
                    pass
    return None

def move_error_checkpoint_to_workspace(error_folder):
    clean_workspace()
    # Move files and folders from error directory into output workspace
    for item in os.listdir(error_folder):
        src = os.path.join(error_folder, item)
        dst = os.path.join('output', item)
        try:
            shutil.move(src, dst)
        except Exception:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
                shutil.rmtree(src, ignore_errors=True)
            else:
                shutil.copy2(src, dst)
                try: os.unlink(src)
                except Exception: pass
    # Clean up empty error folder
    try:
        if os.path.exists(error_folder) and not os.listdir(error_folder):
            shutil.rmtree(error_folder)
    except Exception:
        pass
    for p in Path('output').glob('*.lock'):
        try: p.unlink()
        except Exception: pass

def process_batch(profile='fast', task_sheet='batch/tasks_setting.xlsx'):
    if not check_settings():
        raise RuntimeError('Settings check failed. Please check batch/tasks_setting.xlsx and batch/input.')
    df = pd.read_excel(task_sheet)
    total_tasks = len(df)
    for index, row in df.iterrows():
        status_val = str(row['Status']) if not pd.isna(row.get('Status')) else ''
        if not status_val or 'Error' in status_val or status_val.lower() == 'pending':
            video_file = str(row['Video File']).strip()
            checkpoint_folder = find_task_error_checkpoint(video_file)
            if checkpoint_folder and os.path.isdir(checkpoint_folder):
                console.print(Panel(f'Resuming task from checkpoint: {video_file}\nTask {index + 1}/{total_tasks}\nRestoring: {os.path.basename(checkpoint_folder)}', title='[bold yellow]Resume Task[/bold yellow]', expand=False))
                move_error_checkpoint_to_workspace(checkpoint_folder)
                is_resume = True
            else:
                console.print(Panel(f'Now processing task: {video_file}\nTask {index + 1}/{total_tasks}', title='[bold blue]Current Task[/bold blue]', expand=False))
                clean_workspace()
                is_resume = False
            source_language = row.get('Source Language')
            target_language = row.get('Target Language')
            orig_src, orig_tgt = record_and_update_config(source_language, target_language)
            try:
                dubbing = 0 if pd.isna(row.get('Dubbing')) else int(row['Dubbing'])
                status, error_step, error_message = process_video(video_file, dubbing=dubbing, is_retry=is_resume, source_language=source_language, target_language=target_language, profile=profile)
                status_msg = 'Done' if status else f'Error: {error_step} - {error_message}'
            except Exception as e:
                status_msg = f'Error: Unhandled exception - {str(e)}'
                console.print(f'[bold red]Error processing {video_file}: {status_msg}[/bold red]')
            finally:
                update_key('whisper.language', orig_src)
                update_key('source_language', orig_src)
                update_key('target_language', orig_tgt)
                df.at[index, 'Status'] = status_msg
                df.to_excel(task_sheet, index=False)
                clean_workspace()
                gc.collect()
                time.sleep(1)
        else:
            console.print(f'[dim]Skipping task: {row["Video File"]} - Status: {row["Status"]}[/dim]')
    console.print(Panel('All tasks processed!\nCheck out in `batch/output`!', title='[bold green]Batch Processing Complete[/bold green]', expand=False))

def main():
    parser = argparse.ArgumentParser(description='VideoLingo Batch Processor')
    parser.add_argument('--profile', choices=['fast', 'cinematic', 'subtitle', 'subtitle_only'], default='fast', help='Execution profile (default: fast)')
    parser.add_argument('--sheet', default='batch/tasks_setting.xlsx', help='Path to tasks Excel file')
    args = parser.parse_args()
    process_batch(profile=args.profile, task_sheet=args.sheet)

if __name__ == '__main__':
    main()

