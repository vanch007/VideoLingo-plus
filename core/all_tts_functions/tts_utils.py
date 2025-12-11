"""
TTS 共享工具函数

提供各 TTS 实现共用的功能：
- 参考音频获取与验证
- 原始文本提取
- 输出目录确保

使用方法：
    from core.all_tts_functions.tts_utils import (
        get_reference_audio_path,
        get_prompt_text_from_df,
        ensure_output_dir
    )
"""
import os
from pathlib import Path
from rich import print as rprint
from pydub import AudioSegment

# 常量
MIN_AUDIO_SIZE = 20000  # 最小有效音频文件大小 (20KB)
MIN_AUDIO_DURATION_MS = 500  # 最小有效音频时长 (500ms)
DEFAULT_REFERS_DIR = "output/audio/refers"


def get_audio_duration_ms(file_path: str) -> int:
    """获取音频文件时长（毫秒）"""
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio)
    except Exception:
        return 0


def is_valid_reference_audio(file_path: str, min_size: int = MIN_AUDIO_SIZE, min_duration_ms: int = MIN_AUDIO_DURATION_MS) -> bool:
    """
    检查参考音频是否有效（文件大小和时长都满足要求）
    
    Args:
        file_path: 音频文件路径
        min_size: 最小文件大小（字节）
        min_duration_ms: 最小时长（毫秒）
    """
    if not os.path.exists(file_path):
        return False
    if os.path.getsize(file_path) < min_size:
        return False
    duration = get_audio_duration_ms(file_path)
    if duration < min_duration_ms:
        rprint(f"[yellow]⚠️ 音频 {os.path.basename(file_path)} 时长过短: {duration}ms < {min_duration_ms}ms[/yellow]")
        return False
    return True


def get_reference_audio_path(
    number: int, 
    refers_dir: str = None,
    task_df = None,
    speaker: str = None
) -> tuple[str, int | None]:
    """
    获取参考音频路径，如果指定编号不可用则自动 fallback。
    
    优先按以下顺序查找：
    1. 原始编号的音频
    2. 同一 speaker 的其他有效音频（如果提供了 task_df 和 speaker）
    3. 任意有效的参考音频
    
    Args:
        number: 要获取的音频片段编号
        refers_dir: 参考音频目录，默认为 output/audio/refers
        task_df: 包含 'number' 和 'speaker' 列的 DataFrame（可选）
        speaker: 当前片段的说话人标识（可选）
        
    Returns:
        tuple: (音频文件路径, fallback编号或None)
        - 如果使用了 fallback，返回 (fallback路径, fallback编号)
        - 如果使用原始编号，返回 (原始路径, None)
        
    Raises:
        FileNotFoundError: 如果找不到任何有效的参考音频
    """
    if refers_dir is None:
        refers_dir = os.path.join(os.getcwd(), DEFAULT_REFERS_DIR)
    
    primary_path = os.path.join(refers_dir, f'{number}.wav')
    
    # 检查主音频是否有效
    if is_valid_reference_audio(primary_path):
        return primary_path, None
    
    rprint(f"[yellow]⚠️ 参考音频 {number} 无效，搜索备选...[/yellow]")
    
    # 优先查找同 speaker 的参考音频
    if task_df is not None and speaker is not None:
        try:
            # 获取同 speaker 的所有编号
            same_speaker_numbers = task_df[task_df['speaker'] == speaker]['number'].tolist()
            # 排除当前编号，按距离当前编号的远近排序
            same_speaker_numbers = [n for n in same_speaker_numbers if n != number]
            same_speaker_numbers.sort(key=lambda x: abs(x - number))
            
            for fallback_num in same_speaker_numbers:
                fallback_path = os.path.join(refers_dir, f'{fallback_num}.wav')
                if is_valid_reference_audio(fallback_path):
                    rprint(f"[green]📌 找到同说话人 ({speaker}) 的备选音频: {fallback_num}.wav[/green]")
                    return fallback_path, fallback_num
            
            rprint(f"[yellow]未找到同说话人 ({speaker}) 的有效音频，尝试其他音频...[/yellow]")
        except (KeyError, TypeError, AttributeError) as e:
            rprint(f"[yellow]无法按说话人查找: {e}[/yellow]")
    
    # Fallback: 查找任意有效的参考音频
    for i in range(1, 200):
        fallback_path = os.path.join(refers_dir, f'{i}.wav')
        if is_valid_reference_audio(fallback_path):
            rprint(f"[yellow]📌 使用备选参考音频: {i}.wav[/yellow]")
            return fallback_path, i
    
    raise FileNotFoundError(f"在 {refers_dir} 中未找到有效的参考音频")


def get_prompt_text_from_df(
    task_df, 
    number: int, 
    fallback_text: str = ""
) -> str:
    """
    从 task_df 获取原始文本（作为参考音频的文本提示）。
    
    用于语音克隆 TTS 方法，需要知道参考音频对应的原始文本。
    
    Args:
        task_df: 包含 'number' 和 'origin' 列的 DataFrame
        number: 要查找的编号
        fallback_text: 如果找不到则使用的备选文本
        
    Returns:
        str: 原始文本或备选文本
    """
    if task_df is None:
        return fallback_text
        
    try:
        prompt_text = task_df.loc[task_df['number'] == number, 'origin'].values[0]
        if prompt_text and len(str(prompt_text).strip()) > 0:
            return str(prompt_text)
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    
    if fallback_text:
        rprint(f"[yellow]⚠️ 无法获取编号 {number} 的原始文本，使用备选[/yellow]")
    return fallback_text


def ensure_output_dir(file_path: str):
    """
    确保输出文件的目录存在。
    
    Args:
        file_path: 输出文件的路径
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def validate_audio_file(file_path: str, min_size: int = MIN_AUDIO_SIZE) -> bool:
    """
    验证音频文件是否有效。
    
    Args:
        file_path: 音频文件路径
        min_size: 最小文件大小（字节）
        
    Returns:
        bool: 文件是否有效
    """
    if not os.path.exists(file_path):
        return False
    return os.path.getsize(file_path) >= min_size


def extract_refer_audio_if_needed(number: int = 1):
    """
    如果参考音频不存在，尝试提取。
    
    Args:
        number: 要检查的音频编号
    """
    refers_dir = os.path.join(os.getcwd(), DEFAULT_REFERS_DIR)
    audio_path = os.path.join(refers_dir, f'{number}.wav')
    
    if not validate_audio_file(audio_path):
        try:
            from core.step9_extract_refer_audio import extract_refer_audio_main
            rprint(f"[yellow]参考音频文件不存在，尝试提取: {audio_path}[/yellow]")
            extract_refer_audio_main()
        except Exception as e:
            rprint(f"[bold red]提取参考音频失败: {str(e)}[/bold red]")
            raise
