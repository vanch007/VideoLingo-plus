"""
Step Checker - 检查工作流程各步骤是否已完成
通过检查输出文件来判断每个步骤是否需要执行
"""
import os

# 定义每个步骤的输出文件检查点
STEP_CHECKPOINTS = {
    # 文本处理流程
    "transcribe": "output/log/cleaned_chunks.xlsx",
    "split_spacy": "output/log/sentence_splitbynlp.txt", 
    "split_meaning": "output/log/sentence_splitbymeaning.txt",
    "summarize": "output/log/terminology.json",
    "translate": "output/log/translation_results.xlsx",
    "split_subtitle": "output/log/translation_results_for_subtitles.xlsx",
    "timeline": "output/trans_subs_for_audio.srt",
    "merge_subtitle": "output/AI字幕.mp4",
    
    # 音频处理流程
    "gen_audio_task": "output/audio/audio_task.xlsx",
    "gen_dub_chunks": "output/audio/dub_chunks.xlsx",
    "extract_refer": "output/audio/refers",  # 目录
    "gen_audio": "output/audio/segs",  # 目录
    "merge_audio": "output/audio/full_audio.wav",
    "merge_video": "output/AI配音.mp4",
}

def is_step_completed(step_name: str) -> bool:
    """检查指定步骤是否已完成"""
    checkpoint = STEP_CHECKPOINTS.get(step_name)
    if not checkpoint:
        return False
    
    if os.path.isdir(checkpoint):
        # 对于目录，检查是否存在且非空
        return os.path.exists(checkpoint) and len(os.listdir(checkpoint)) > 0
    else:
        # 对于文件，检查是否存在
        return os.path.exists(checkpoint)

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
