#!/usr/bin/env python3
import os
import sys
import time
import json
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
console = Console()

TASK_SHEET = Path('batch/tasks_setting.xlsx')
STATE_FILE = Path('output/pipeline_state.json')
PROGRESS_FILE = Path('output/log/fast_progress.json')

ALL_STEPS = [
    'download', 'transcribe', 'split_spacy', 'split_meaning',
    'summarize', 'translate', 'split_subtitle', 'timeline',
    'gen_audio_task', 'extract_refer', 'gen_audio', 'merge_video'
]
STEP_CN = {
    'download': '1. 视频下载',
    'transcribe': '2. 人声与ASR转录',
    'split_spacy': '3. 标点语义分句',
    'split_meaning': '4. 意群分割',
    'summarize': '5. 摘要与术语提取',
    'translate': '6. 大模型批量翻译',
    'split_subtitle': '7. 字幕长短句切分',
    'timeline': '8. 字幕时间轴对齐',
    'gen_audio_task': '9. 配音任务规划',
    'extract_refer': '10. 参考音频抽取',
    'gen_audio': '11. 极速声音克隆合成',
    'merge_video': '12. 1080p高清混音压制'
}

def get_tasks_summary():
    if not TASK_SHEET.is_file():
        return []
    try:
        df = pd.read_excel(TASK_SHEET)
        tasks = []
        for idx, row in df.iterrows():
            st = str(row.get('Status', '')).strip()
            if pd.isna(row.get('Status')) or st.lower() in ('', 'nan', 'none'):
                badge = '[dim]⏳ 队列排队中[/dim]'
            elif st == 'Done':
                badge = '[bold green]✅ 已完成 (Done)[/bold green]'
            elif 'Error' in st:
                badge = f'[bold red]❌ 失败: {st[:45]}[/bold red]'
            else:
                badge = f'[cyan]{st}[/cyan]'
            tasks.append((idx + 1, str(row.get('Video File', '')), badge))
        return tasks
    except Exception:
        return []

def get_current_pipeline_info():
    if not STATE_FILE.is_file():
        return None
    try:
        return json.loads(STATE_FILE.read_text(encoding='utf-8'))
    except Exception:
        return None

def get_audio_progress():
    if not PROGRESS_FILE.is_file():
        return None
    try:
        return json.loads(PROGRESS_FILE.read_text(encoding='utf-8'))
    except Exception:
        return None

def render():
    layout = Layout()
    layout.split_column(
        Layout(name='header', size=3),
        Layout(name='current_task', size=9),
        Layout(name='table', ratio=1)
    )
    now_str = time.strftime('%Y-%m-%d %H:%M:%S')
    layout['header'].update(Panel(f'[bold cyan]🎬 VideoLingo-plus 批量任务实时监控[/bold cyan]  |  更新时间: {now_str}', border_style='cyan'))
    pipe = get_current_pipeline_info()
    if pipe:
        inp = pipe.get('input', '未知')
        curr_step = pipe.get('current_step', '准备中')
        status = pipe.get('status', '未知')
        completed = pipe.get('completed_steps', [])
        total_steps = len(ALL_STEPS)
        comp_count = len(completed)
        step_display = STEP_CN.get(curr_step, curr_step)
        audio_prog = get_audio_progress()
        audio_info = ''
        if curr_step == 'gen_audio' and audio_prog:
            tot = audio_prog.get('total', 0)
            gen = audio_prog.get('generated', 0)
            reu = audio_prog.get('reused', 0)
            fai = audio_prog.get('failed', 0)
            audio_info = f'\n🎙️ 语音合成详情: 已生成 {gen} / 现存复用 {reu} / 失败 {fai} (总计 {tot} 句)'
        desc = (
            f'[bold yellow]目标链接:[/] {inp}\n'
            f'[bold green]当前步骤:[/] [bold underline cyan]{step_display}[/] ({comp_count}/{total_steps})\n'
            f'[bold blue]运行状态:[/] {status.upper()}{audio_info}'
        )
        layout['current_task'].update(Panel(desc, title='[bold]📌 正在处理的任务状态[/bold]', border_style='green'))
    else:
        layout['current_task'].update(Panel('[dim]暂无运行中的流水线或正在切换下一个视频...[/dim]', title='📌 正在处理的任务状态', border_style='dim'))
    table = Table(title='📋 批量任务队列明细', expand=True, border_style='blue')
    table.add_column('序号', width=6, justify='center')
    table.add_column('视频源链接 / 文件名', ratio=3)
    table.add_column('处理状态', ratio=2)
    for item in get_tasks_summary():
        table.add_row(str(item[0]), item[1], item[2])
    layout['table'].update(table)
    return layout

def main():
    with Live(render(), refresh_per_second=2, screen=True) as live:
        try:
            while True:
                live.update(render())
                time.sleep(1)
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()

