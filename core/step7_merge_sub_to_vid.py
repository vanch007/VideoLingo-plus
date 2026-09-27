import os, subprocess, time, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config_utils import load_key
from core.step1_ytdlp import find_video_files
from rich import print as rprint
import cv2
import numpy as np
import platform

SRC_FONT_SIZE = 15
TRANS_FONT_SIZE = 17
FONT_NAME = 'Arial'
TRANS_FONT_NAME = 'Arial'

# Linux need to install google noto fonts: apt-get install fonts-noto
if platform.system() == 'Linux':
    FONT_NAME = 'NotoSansCJK-Regular'
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
# Mac OS has different font names
elif platform.system() == 'Darwin':
    FONT_NAME = 'Arial Unicode MS'
    TRANS_FONT_NAME = 'Arial Unicode MS'

SRC_FONT_COLOR = '&HFFFFFF'
SRC_OUTLINE_COLOR = '&H000000'
SRC_OUTLINE_WIDTH = 1
SRC_SHADOW_COLOR = '&H80000000'
TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1
TRANS_BACK_COLOR = '&H33000000'

OUTPUT_DIR = "output"
OUTPUT_VIDEO = f"{OUTPUT_DIR}/AI字幕.mp4"
SRC_SRT = f"{OUTPUT_DIR}/src.srt"
TRANS_SRT = f"{OUTPUT_DIR}/trans.srt"

def check_gpu_available():
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
        return 'h264_nvenc' in result.stdout
    except:
        return False

def merge_subtitles_to_video():
    video_file = find_video_files()
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    # Check resolution
    if not load_key("burn_subtitles"):
        rprint("[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")

        # Create a black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()

        rprint("[bold green]Placeholder video has been generated.[/bold green]")
        return

    if not os.path.exists(SRC_SRT) or not os.path.exists(TRANS_SRT):
        print("Subtitle files not found in the 'output' directory.")
        exit(1)

    video = cv2.VideoCapture(video_file)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")
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
        f"FontSize={src_size},FontName={FONT_NAME},"
        f"PrimaryColour={SRC_FONT_COLOR},OutlineColour={SRC_OUTLINE_COLOR},OutlineWidth={src_outline},"
        f"ShadowColour={SRC_SHADOW_COLOR},BorderStyle=1,"
        f"MarginL={margin_lr},MarginR={margin_lr},MarginV={src_margin_v}"
    )
    dub_style = (
        f"FontSize={trans_size},FontName={TRANS_FONT_NAME},"
        f"PrimaryColour={TRANS_FONT_COLOR},OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={trans_outline},"
        f"BackColour={TRANS_BACK_COLOR},Alignment=2,BorderStyle=4,"
        f"MarginL={margin_lr},MarginR={margin_lr},MarginV={margin_v}"
    )

    # Use absolute paths to avoid escaping issues
    abs_src_srt = os.path.abspath(SRC_SRT).replace("'", "'\\''")
    abs_trans_srt = os.path.abspath(TRANS_SRT).replace("'", "'\\''")

    filters = [
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease",
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
    ]
    if bool(load_key("burn_source_subtitles", True)):
        filters.append(f"subtitles='{abs_src_srt}':force_style='{src_style}'")
    filters.append(f"subtitles='{abs_trans_srt}':force_style='{dub_style}'")
    vf_str = ",".join(filters)

    from core.step12_merge_dub_to_vid import get_ffmpeg_binary
    ffmpeg_cmd = [
        get_ffmpeg_binary(), '-i', video_file,
        '-vf', vf_str,
    ]

    gpu_available = check_gpu_available()
    if gpu_available:
        rprint("[bold green]NVIDIA GPU encoder detected, will use GPU acceleration.[/bold green]")
        ffmpeg_cmd.extend(['-c:v', 'h264_nvenc'])
    elif platform.system() == 'Darwin':
        rprint("[bold green]Apple Silicon detected, will use VideoToolbox acceleration.[/bold green]")
        ffmpeg_cmd.extend(['-c:v', 'h264_videotoolbox', '-b:v', '5M'])
    else:
        rprint("[bold yellow]No GPU encoder detected, will use CPU instead.[/bold yellow]")

    ffmpeg_cmd.extend(['-y', OUTPUT_VIDEO])

    print("🎬 Start merging subtitles to video...")
    start_time = time.time()
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"\n✅ Done! Time taken: {time.time() - start_time:.2f} seconds")
    else:
        error_msg = result.stderr[-800:] if result.stderr else "No stderr output"
        rprint(f"[bold red]❌ FFmpeg subtitle merge failed! stderr:\n{error_msg}[/bold red]")
        raise RuntimeError(
            f"FFmpeg subtitle merge failed (exit code {result.returncode}). "
            f"Error: {error_msg[-200:]}"
        )

if __name__ == "__main__":
    merge_subtitles_to_video()
