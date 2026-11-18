import os, subprocess, time, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config_utils import load_key
from core.step1_ytdlp import find_video_files
from rich import print as rprint
import cv2
import numpy as np
import platform
import json
from pathlib import Path

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
    from core.utterance_plan import plan_active, load_plan
    is_fast = (load_key("quality_mode", "high_sync") == "fast") and bool(load_key("fast_mode.enabled", False))
    if plan_active() and not is_fast and load_key('burn_subtitles', False):
        from core.bilingual_alignment import verify_bilingual_display
        verify_bilingual_display(load_plan(), dubbed=False)
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
    from core.subtitle_layout import compute_subtitle_burn_style
    layout_style = compute_subtitle_burn_style(TARGET_WIDTH, TARGET_HEIGHT)
    is_portrait = bool(layout_style["is_portrait"])
    aspect_ratio = TARGET_WIDTH / TARGET_HEIGHT

    if is_portrait:
        src_size = int(layout_style["src_font_size"])
        trans_size = int(layout_style["font_size"])
        margin_v = int(layout_style["margin_v"])
        margin_lr = int(layout_style["margin_lr"])
        wrap_lines = 2.2
        src_outline = 1
        trans_outline = 1
        rprint("[cyan]📱 Portrait mode: Applying exact aspect-ratio font scaling[/cyan]")
    else:
        # Target standard chars per line for landscape
        chars_per_line_src = 36
        chars_per_line_trans = 30
        bottom_margin_percent = 0.08
        wrap_lines = 1.5
        src_outline = SRC_OUTLINE_WIDTH
        trans_outline = TRANS_OUTLINE_WIDTH
        src_size = int(aspect_ratio * (288 / max(chars_per_line_src, 1)))
        trans_size = int(aspect_ratio * (288 / max(chars_per_line_trans, 1)))
        margin_v = int(bottom_margin_percent * 288)
        margin_lr = 20
        rprint("[cyan]🖥️ Landscape mode: Applying exact aspect-ratio font scaling[/cyan]")

    src_margin_v = margin_v + int(trans_size * wrap_lines) + 6

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
    if str(load_key("subtitle_engine", "ass")) == "ass":
        # One ASS script: the source row is stacked above the translation rows actually shown.
        from core.subtitle_ass import TrackStyle, burn_filter
        burn_src = bool(load_key("burn_source_subtitles", True))
        filters.append(burn_filter(
            TARGET_WIDTH, TARGET_HEIGHT, SRC_SRT if burn_src else None, TRANS_SRT,
            TrackStyle(FONT_NAME, src_size, SRC_FONT_COLOR, SRC_OUTLINE_COLOR, src_outline, 0, SRC_SHADOW_COLOR, 1, margin_v, margin_lr) if burn_src else None,
            TrackStyle(TRANS_FONT_NAME, trans_size, TRANS_FONT_COLOR, TRANS_OUTLINE_COLOR, trans_outline, 0, TRANS_BACK_COLOR, 1, margin_v, margin_lr),
            "output/_subtitle_burn.ass"))
    else:
        if bool(load_key("burn_source_subtitles", True)):
            filters.append(f"subtitles='{abs_src_srt}':force_style='{src_style}'")
        filters.append(f"subtitles='{abs_trans_srt}':force_style='{dub_style}'")
    from core.subtitle_quality import subtitle_quality_report
    quality = subtitle_quality_report()
    Path("output/log/subtitle_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    if not quality["accepted"]:
        filters.append("drawtext=text='DRAFT - NEEDS REVIEW':fontsize=14:fontcolor=yellow:box=1:boxcolor=black@0.65:x=10:y=10")
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
