import os
import sys
import platform
import subprocess

import numpy as np
import cv2
from rich import print as rprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.all_whisper_methods.demucs_vl import BACKGROUND_AUDIO_FILE
from core.step7_merge_sub_to_vid import check_gpu_available
from core.config_utils import load_key
from core.step1_ytdlp import find_video_files
from pydub import AudioSegment

DUB_VIDEO = "output/AI配音.mp4"
DUB_SUB_FILE = 'output/dub.srt'
SRC_SRT = 'output/dub_orig.srt'  # 使用新生成的原语言字幕路径
DUB_AUDIO = 'output/dub.mp3'

SRC_FONT_SIZE = 15  # 原语言字幕字号
TRANS_FONT_SIZE = 17
SRC_FONT_NAME = 'Arial'
TRANS_FONT_NAME = 'Arial'
if platform.system() == 'Linux':
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
if platform.system() == 'Darwin':
    TRANS_FONT_NAME = 'Arial Unicode MS'

TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'

def normalize_audio_volume(audio_path: str, output_path: str, target_db: float = -20.0):
    audio = AudioSegment.from_file(audio_path)
    change_in_dBFS = target_db - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)
    normalized_audio.export(output_path, format="wav")
    rprint(f"[green]✅ Audio normalized from {audio.dBFS:.1f}dB to {target_db:.1f}dB[/green]")
    return output_path

def merge_video_audio():
    """Merge video and audio, and reduce video volume"""
    VIDEO_FILE = find_video_files()
    background_file = BACKGROUND_AUDIO_FILE
    
    if not load_key("burn_subtitles"):
        rprint("[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")

        # Create a black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(DUB_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()

        rprint("[bold green]Placeholder video has been generated.[/bold green]")
        return

    # Normalize dub audio
    normalized_dub_audio = 'output/normalized_dub.wav'
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)
    
    # Merge video and audio with translated subtitles
    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")
    
    # 原语言字幕样式 (默认位置，白色)
    src_sub_filter = (
        f"subtitles={SRC_SRT}:force_style='FontSize={SRC_FONT_SIZE},"
        f"FontName={SRC_FONT_NAME},PrimaryColour=&HFFFFFF,"
        f"OutlineColour=&H000000,OutlineWidth=1,"
        f"Alignment=2,BorderStyle=1'"
    )
    
    # 翻译字幕样式 (底部，黄色)
    trans_sub_filter = (
        f"subtitles={DUB_SUB_FILE}:force_style='FontSize={TRANS_FONT_SIZE},"
        f"FontName={TRANS_FONT_NAME},PrimaryColour={TRANS_FONT_COLOR},"
        f"OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
        f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle=4'"
    )
    
    # 组合两个字幕filter (先处理原文字幕在上，再处理翻译字幕在下)
    subtitle_filter = f"{src_sub_filter},{trans_sub_filter}"
    
    cmd = [
        'ffmpeg', '-y', 
        '-threads', '0',  # 自动使用所有可用线程
        '-i', VIDEO_FILE, 
        '-i', background_file, 
        '-i', normalized_dub_audio,
        '-filter_complex',
        f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,'
        f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,'
        f'{subtitle_filter}[v];'
        f'[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=3[a]'
    ]

    if check_gpu_available():
        rprint("[bold green]Using NVIDIA GPU acceleration...[/bold green]")
        cmd.extend([
            '-map', '[v]', 
            '-map', '[a]', 
            '-c:v', 'h264_nvenc',
            '-preset', 'fast'  # 更快的编码预设
        ])
    elif platform.system() == 'Darwin':
        rprint("[bold green]Using Apple Silicon VideoToolbox acceleration...[/bold green]")
        cmd.extend([
            '-map', '[v]', 
            '-map', '[a]', 
            '-c:v', 'h264_videotoolbox',
            '-q:v', '70',  # 质量参数(0-100)
            '-profile:v', 'high',
            '-allow_sw', '1'  # 允许软件回退
        ])
    else:
        cmd.extend([
            '-map', '[v]', 
            '-map', '[a]',
            '-c:v', 'libx264',
            '-preset', 'fast'
        ])
    
    cmd.extend([
        '-c:a', 'aac', 
        '-b:a', '192k',
        '-movflags', '+faststart',  # 优化网络播放
        DUB_VIDEO
    ])
    
    subprocess.run(cmd)
    rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")

if __name__ == '__main__':
    merge_video_audio()
