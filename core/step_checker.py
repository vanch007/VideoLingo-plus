"""
Step Checker - 检查工作流程各步骤是否已完成
通过检查输出文件来判断每个步骤是否需要执行
"""
import os
from typing import Callable

import pandas as pd

# 定义每个步骤的输出文件检查点
STEP_CHECKPOINTS = {
    # 文本处理流程
    "transcribe": "output/log/cleaned_chunks.xlsx",
    "split_spacy": "output/log/sentence_splitbynlp.txt", 
    "split_meaning": "output/log/sentence_splitbymeaning.txt",
    "summarize": "output/log/terminology.json",
    "translate": "output/log/translation_results.xlsx",
    "split_subtitle": "output/log/translation_results_for_subtitles.xlsx",
    "timeline": "output/audio/trans_subs_for_audio.srt",
    "merge_subtitle": "output/AI字幕.mp4",
    
    # 音频处理流程
    "gen_audio_task": "output/audio/tts_tasks.xlsx",
    "gen_dub_chunks": "output/audio/tts_tasks.xlsx",
    "extract_refer": "output/audio/refers",  # 目录
    "gen_audio": "output/audio/segs",
    "dubbing_eval": "output/audio/dubbing_eval.json",
    "merge_audio": "output/dub.mp3",
    "merge_video": "output/AI配音.mp4",
}

STEP_REQUIRED_COLUMNS = {
    "gen_audio_task": {"number", "start_time", "end_time", "duration", "text"},
    "gen_dub_chunks": {"number", "start_time", "end_time", "duration", "text", "gap", "tolerance", "tol_dur", "cut_off", "lines"},
    "gen_audio": {"number", "real_dur", "new_sub_times"},
}


def _file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _dir_has_files(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


def _excel_has_columns(path: str, columns: set[str]) -> bool:
    if not _file_exists(path):
        return False
    try:
        df = pd.read_excel(path, nrows=5)
    except Exception:
        return False
    return columns.issubset(set(df.columns))


def _step_has_required_columns(step_name: str) -> bool:
    columns = STEP_REQUIRED_COLUMNS.get(step_name)
    checkpoint = STEP_CHECKPOINTS.get(step_name)
    if not columns or not checkpoint:
        return False
    return _excel_has_columns(checkpoint, columns)


STEP_VALIDATORS: dict[str, Callable[[], bool]] = {
    "gen_audio_task": lambda: _step_has_required_columns("gen_audio_task"),
    "gen_dub_chunks": lambda: _step_has_required_columns("gen_dub_chunks"),
    "gen_audio": lambda: (
        _dir_has_files(STEP_CHECKPOINTS["gen_audio"]) and _step_has_required_columns("gen_audio")
    ) or (_file_exists("output/dub.mp3") and _file_exists("output/dub.srt")),
    "merge_audio": lambda: _file_exists("output/dub.mp3") and _file_exists("output/dub.srt"),
}

def is_step_completed(step_name: str) -> bool:
    """检查指定步骤是否已完成"""
    validator = STEP_VALIDATORS.get(step_name)
    if validator:
        return validator()

    checkpoint = STEP_CHECKPOINTS.get(step_name)
    if not checkpoint:
        return False
    
    if os.path.isdir(checkpoint):
        # 对于目录，检查是否存在且非空
        return _dir_has_files(checkpoint)
    else:
        # 对于文件，检查是否存在
        return _file_exists(checkpoint)

def get_completed_steps() -> list:
    """获取所有已完成的步骤列表"""
    return [step for step in STEP_CHECKPOINTS if is_step_completed(step)]

def get_pending_steps() -> list:
    """获取所有待执行的步骤列表"""
    return [step for step in STEP_CHECKPOINTS if not is_step_completed(step)]

def get_next_step(step_order: list) -> str:
    """根据步骤顺序，返回下一个需要执行的步骤"""
    for step in step_order:
        if not is_step_completed(step):
            return step
    return None

def print_step_status():
    """打印所有步骤的状态"""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    table = Table(title="Step Status")
    table.add_column("Step", style="cyan")
    table.add_column("Checkpoint File", style="blue")
    table.add_column("Status", style="green")
    
    for step, checkpoint in STEP_CHECKPOINTS.items():
        status = "✅ Completed" if is_step_completed(step) else "⏳ Pending"
        table.add_row(step, checkpoint, status)
    
    console.print(table)


if __name__ == "__main__":
    print_step_status()
