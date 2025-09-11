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
DUB_SUB_FILE = "output/trans.srt"
SRC_SRT = "output/src.srt"
DUB_AUDIO = 'output/dub.mp3'

# --- Bilingual Subtitle Styles (from step7) ---
SRC_FONT_SIZE = 15
TRANS_FONT_SIZE = 17
SRC_FONT_NAME = 'Arial'
TRANS_FONT_NAME = 'Arial'

if platform.system() == 'Linux':
    SRC_FONT_NAME = 'NotoSansCJK-Regular'
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
elif platform.system() == 'Darwin':
    SRC_FONT_NAME = 'Arial Unicode MS'
    TRANS_FONT_NAME = 'Arial Unicode MS'

SRC_FONT_COLOR = '&HFFFFFF'
SRC_OUTLINE_COLOR = '&H000000'
SRC_OUTLINE_WIDTH = 1
SRC_SHADOW_COLOR = '&H80000000'
TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'
# --- End of Styles ---

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
    
    normalized_dub_audio = 'output/normalized_dub.wav'
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)
    
    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")
    
    video_filter_parts = [
        f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease',
        f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2'
    ]

    if load_key("burn_subtitles"):
        if not os.path.exists(SRC_SRT) or not os.path.exists(DUB_SUB_FILE):
            rprint(f"[bold red]Subtitle Error: One or both subtitle files not found.[/bold red]")
            rprint(f"Searched for: {SRC_SRT} and {DUB_SUB_FILE}")
            rprint("[bold yellow]Skipping subtitle burning.[/bold yellow]")
        else:
            rprint("[bold green]Both subtitle files found, burning bilingual subtitles...[/bold green]")
            escaped_dub_file = DUB_SUB_FILE.replace('\\', '/')
            escaped_src_file = SRC_SRT.replace('\\', '/')

            src_style = (
                f"subtitles={escaped_src_file}:force_style='FontSize={SRC_FONT_SIZE},FontName={SRC_FONT_NAME},"
                f"PrimaryColour={SRC_FONT_COLOR},OutlineColour={SRC_OUTLINE_COLOR},OutlineWidth={SRC_OUTLINE_WIDTH},"
                f"ShadowColour={SRC_SHADOW_COLOR},BorderStyle=1'"
            )

            dub_style = (
                f"subtitles={escaped_dub_file}:force_style='FontSize={TRANS_FONT_SIZE},FontName={TRANS_FONT_NAME},"
                f"PrimaryColour={TRANS_FONT_COLOR},OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
                f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=40,BorderStyle=4'"
            )

            video_filter_parts.append(src_style)
            video_filter_parts.append(dub_style)

    video_filter = ",".join(video_filter_parts) + "[v]"

    cmd = [
        'ffmpeg', '-y',
        '-threads', '0',
        '-i', VIDEO_FILE,
        '-i', background_file,
        '-i', normalized_dub_audio,
        '-filter_complex',
        f'{video_filter};[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=3[a]'
    ]

    if check_gpu_available():
        rprint("[bold green]Using NVIDIA GPU acceleration...[/bold green]")
        cmd.extend([
            '-map', '[v]', 
            '-map', '[a]', 
            '-c:v', 'h264_nvenc',
            '-preset', 'fast'
        ])
    elif platform.system() == 'Darwin':
        rprint("[bold green]Using Apple Silicon VideoToolbox acceleration...[/bold green]")
        cmd.extend([
            '-map', '[v]', 
            '-map', '[a]', 
            '-c:v', 'h264_videotoolbox',
            '-q:v', '70',
            '-profile:v', 'high',
            '-allow_sw', '1'
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
        '-movflags', '+faststart',
        DUB_VIDEO
    ])
    
    subprocess.run(cmd)
    rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")

if __name__ == '__main__':
    merge_video_audio()