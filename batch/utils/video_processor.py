import os
import sys
import shutil
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config_utils import load_key, update_key
from core.onekeycleanup import cleanup
from core.pipeline.runner import build_steps_for_input, run_pipeline, PipelineRun
from core.pipeline.profiles import apply_profile
from core.pipeline.job import workspace_lock
from core.step1_ytdlp import download_video_ytdlp, find_video_files

console = Console()
if not (os.environ.get('all_proxy') or os.environ.get('https_proxy') or os.environ.get('http_proxy')):
    import socket
    for p in (7897, 7890, 10808, 1087):
        try:
            s = socket.socket(); s.settimeout(0.1)
            if s.connect_ex(('127.0.0.1', p)) == 0:
                os.environ['http_proxy'] = f'http://127.0.0.1:{p}'
                os.environ['https_proxy'] = f'http://127.0.0.1:{p}'
                os.environ['all_proxy'] = f'http://127.0.0.1:{p}'
                s.close(); break
            s.close()
        except Exception: pass

INPUT_DIR = 'batch/input'
OUTPUT_DIR = 'output'
SAVE_DIR = 'batch/output'
ERROR_OUTPUT_DIR = 'batch/output/ERROR'
YTB_RESOLUTION_KEY = 'ytb_resolution'

LANGUAGE_CODE_MAP = {
    '简体中文': 'zh', '中文': 'zh', '汉语': 'zh', '普通话': 'zh', 'zh-cn': 'zh', 'zh-hans': 'zh',
    '繁體中文': 'zh-tw', '繁体中文': 'zh-tw', 'zh-hant': 'zh-tw',
    '英语': 'en', '英文': 'en', 'english': 'en',
    '越南语': 'vi', '越语': 'vi', 'tiếng việt': 'vi', 'vietnamese': 'vi',
    '日语': 'ja', '日文': 'ja', 'japanese': 'ja',
    '韩语': 'ko', '韩文': 'ko', 'korean': 'ko',
    '泰语': 'th', '泰文': 'th', 'thai': 'th',
    '法语': 'fr', '法文': 'fr', 'french': 'fr',
    '德语': 'de', '德文': 'de', 'german': 'de',
    '西班牙语': 'es', 'spanish': 'es',
    '俄语': 'ru', 'russian': 'ru',
}

def normalize_language_code(lang_str, default='auto'):
    if not lang_str or str(lang_str).strip() == '' or str(lang_str).lower() == 'nan':
        return default
    s = str(lang_str).strip()
    return LANGUAGE_CODE_MAP.get(s.lower(), LANGUAGE_CODE_MAP.get(s, s))

def prepare_output_folder(output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

def process_video(file, dubbing=False, is_retry=False, source_language=None, target_language=None, profile='fast'):
    src_lang = normalize_language_code(source_language, default=load_key('whisper.language', 'auto'))
    tgt_lang = normalize_language_code(target_language, default=load_key('target_language', 'zh'))
    if src_lang and src_lang != 'auto':
        update_key('whisper.language', src_lang)
        update_key('source_language', src_lang)
    if tgt_lang and tgt_lang != 'auto':
        update_key('target_language', tgt_lang)
    apply_profile(profile)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if file.startswith(('http://', 'https://')):
        input_target = file
    else:
        local_src = Path(INPUT_DIR) / file
        if not local_src.is_file():
            raise FileNotFoundError(f'Local input file not found: {local_src}')
        input_target = str(local_src.resolve())
    run_id = f'batch-{time.strftime("%Y%m%d-%H%M%S")}'
    steps = build_steps_for_input(
        input_target,
        dubbing=bool(dubbing),
        subtitles=True,
        profile=profile,
    )
    run = PipelineRun(
        run_id=run_id,
        profile=profile,
        source=src_lang,
        target=tgt_lang,
        steps=[s.key for s in steps],
        input=input_target,
        subtitle_only=not bool(dubbing),
        tts='mlx_router' if profile == 'fast' else load_key('tts_method', 'auto'),
        quality_status='fast_mode' if profile == 'fast' else None,
    )
    try:
        with workspace_lock():
            result = run_pipeline(run, steps, resume=is_retry)
            if result.status == 'failed':
                raise RuntimeError(result.error or f'Failed at step: {result.failed_step}')
    except Exception as e:
        error_step = run.failed_step or run.current_step or 'unknown'
        error_msg = str(e)
        error_panel = Panel(f'[bold red]Error in step \'{error_step}\':[/]\n{error_msg}', border_style='red')
        console.print(error_panel)
        cleanup(ERROR_OUTPUT_DIR)
        return False, error_step, error_msg
    console.print(Panel('[bold green]Video processed successfully! 🎉[/]', border_style='green'))
    cleanup(SAVE_DIR)
    return True, '', ''

