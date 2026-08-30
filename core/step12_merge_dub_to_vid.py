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
from core.constants import SOURCE_VIDEO
from pydub import AudioSegment

DUB_VIDEO = "output/AI配音.mp4"
DUB_SUB_FILE = "output/dub.srt"
SRC_SRT = "output/dub_orig.srt"
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
    target_db = float(load_key("dubbing_quality.loudness_target_lufs", -14.0))
    audio = AudioSegment.from_file(audio_path)
    change_in_dBFS = target_db - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)
    normalized_audio.export(output_path, format="wav")
    rprint(f"[green]✅ Audio normalized from {audio.dBFS:.1f}dB to {target_db:.1f}dB[/green]")
    return output_path

def merge_video_audio():
    """Merge video and audio with subtitle burning (two-pass for FFmpeg 8.x compatibility)."""
    if not os.path.isfile(SOURCE_VIDEO):
        raise FileNotFoundError(
            f"Canonical source video is missing: {SOURCE_VIDEO}. Refusing to use an "
            "already-subtitled or previously dubbed derivative as merge input."
        )
    VIDEO_FILE = SOURCE_VIDEO
    background_file = BACKGROUND_AUDIO_FILE

    normalized_dub_audio = 'output/normalized_dub.wav'
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)

    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")

    # ── Pass 1: Burn subtitles into video using -vf (avoids filter_complex escaping hell) ──
    burn_subs = load_key("burn_subtitles") and os.path.exists(SRC_SRT) and os.path.exists(DUB_SUB_FILE)

    if burn_subs:
        rprint("[bold green]Pass 1: Burning bilingual subtitles into video...[/bold green]")

        # Adaptive subtitle sizing based on video orientation and LibASS default PlayResY=288
        is_portrait = TARGET_HEIGHT > TARGET_WIDTH
        aspect_ratio = TARGET_WIDTH / TARGET_HEIGHT

        if is_portrait:
            # Target 22 Chinese chars and 18 Vietnamese chars per line to fit narrow screens
            chars_per_line_src = 22
            chars_per_line_trans = 18
            bottom_margin_percent = 0.10  # 10% from bottom
            wrap_lines = 2.5  # Expect up to 2.5 lines of translation
            src_outline = 1
            trans_outline = 2
            rprint("[cyan]📱 Portrait mode: Applying exact aspect-ratio font scaling[/cyan]")
        else:
            # Target standard chars per line for landscape
            chars_per_line_src = 36
            chars_per_line_trans = 30
            bottom_margin_percent = 0.08
            wrap_lines = 1.5
            src_outline = SRC_OUTLINE_WIDTH
            trans_outline = TRANS_OUTLINE_WIDTH
            rprint("[cyan]🖥️ Landscape mode: Applying exact aspect-ratio font scaling[/cyan]")

        # Libass defaults to PlayResY=288, PlayResX=384 for standard SRT styling
        # Formula: FontSize = (TARGET_WIDTH / TARGET_HEIGHT) * (288 / chars_per_line)
        src_size = int(aspect_ratio * (288 / max(chars_per_line_src, 1)))
        trans_size = int(aspect_ratio * (288 / max(chars_per_line_trans, 1)))

        margin_v = int(bottom_margin_percent * 288)
        src_margin_v = margin_v + int(trans_size * wrap_lines) + 5
        margin_lr = 20 # 20/384 ≈ 5% horizontal margin to force wrapping

        src_style = (
            f"FontSize={src_size},FontName={SRC_FONT_NAME},"
            f"PrimaryColour={SRC_FONT_COLOR},OutlineColour={SRC_OUTLINE_COLOR},OutlineWidth={src_outline},"
            f"ShadowColour={SRC_SHADOW_COLOR},BorderStyle=1,"
            f"MarginL={margin_lr},MarginR={margin_lr},MarginV={src_margin_v}"
        )
        dub_style = (
            f"FontSize=11,FontName={TRANS_FONT_NAME},"
            f"PrimaryColour=&H00FFFF&,OutlineColour=&H000000&,OutlineWidth=1.2,"
            f"ShadowColour=&H80000000&,Alignment=2,BorderStyle=1,"
            f"MarginL=15,MarginR=15,MarginV=4"
        )

        # Use absolute paths to avoid escaping issues
        abs_src_srt = os.path.abspath(SRC_SRT).replace("'", "'\\''")
        abs_dub_srt = os.path.abspath(DUB_SUB_FILE).replace("'", "'\\''")

        filters = [
            f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease",
            f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
        ]
        if bool(load_key("burn_source_subtitles", True)):
            filters.append(f"subtitles='{abs_src_srt}':force_style='{src_style}'")
        filters.append(f"subtitles='{abs_dub_srt}':force_style='{dub_style}'")
        vf = ",".join(filters)

        temp_subbed = 'output/_temp_subbed.mp4'

        # Since we fixed the path quote bug, Homebrew ffmpeg 8.1 natively supports libass.
        # We can safely use system ffmpeg and hardware acceleration.
        sub_cmd = ['ffmpeg', '-y', '-i', VIDEO_FILE, '-vf', vf, '-an']

        gpu_available = check_gpu_available()
        if gpu_available:
            rprint("[bold green]NVIDIA GPU encoder detected, will use GPU acceleration.[/bold green]")
            sub_cmd.extend(['-c:v', 'h264_nvenc'])
        elif platform.system() == 'Darwin':
            rprint("[bold green]Apple Silicon detected, will use VideoToolbox acceleration.[/bold green]")
            sub_cmd.extend(['-c:v', 'h264_videotoolbox', '-b:v', '5M'])
        else:
            rprint("[bold yellow]No GPU encoder detected, will use CPU instead.[/bold yellow]")
            sub_cmd.extend(['-c:v', 'libx264', '-preset', 'fast'])

        sub_cmd.append(temp_subbed)

        rprint(f"[dim]Running: {' '.join(sub_cmd[:6])} ...[/dim]")
        result = subprocess.run(sub_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = result.stderr[-800:] if result.stderr else "No stderr output"
            rprint(f"[bold red]Pass 1 failed! stderr:\n{error_msg}[/bold red]")
            raise RuntimeError(
                f"FFmpeg subtitle burn failed (exit code {result.returncode}). "
                f"This means the dubbed video would have NO subtitles. "
                f"FFmpeg error: {error_msg[-200:]}"
            )
        else:
            rprint("[green]✅ Pass 1: Subtitles burned successfully[/green]")
            VIDEO_FILE = temp_subbed

    # ── Pass 2: Mix audio streams with filter_complex (no subtitle filters here) ──
    rprint("[bold green]Pass 2: Mixing audio and merging...[/bold green]")

    background_volume = load_key("background_volume", 0.35)

    audio_filter = (
        f"[1:a]volume={background_volume}[bg];"
        f"[2:a]aformat=channel_layouts=stereo[dub];"
        f"[dub]asplit[dub_sc][dub_mix];"
        f"[bg][dub_sc]sidechaincompress=threshold=0.015:ratio=8:attack=20:release=250[docked_bg];"
        f"[docked_bg][dub_mix]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[a]"
    )

    cmd = [
        'ffmpeg', '-y',
        '-threads', '0',
        '-i', VIDEO_FILE,
        '-i', background_file,
        '-i', normalized_dub_audio,
        '-filter_complex', audio_filter,
        '-map', '0:v',
        '-map', '[a]',
        '-c:v', 'copy',  # Video already encoded in pass 1
        '-c:a', 'aac',
        '-b:a', '192k',
        '-movflags', '+faststart',
        DUB_VIDEO
    ]

    subprocess.run(cmd)
    rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")

    # Cleanup temp file
    if burn_subs and os.path.exists('output/_temp_subbed.mp4'):
        os.remove('output/_temp_subbed.mp4')

if __name__ == '__main__':
    merge_video_audio()
