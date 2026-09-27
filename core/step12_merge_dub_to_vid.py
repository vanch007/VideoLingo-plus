import os
import re
import sys
import json
import math
import shutil
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
from core.step11_merge_full_audio import INPUT_EXCEL, load_and_flatten_data
from pydub import AudioSegment

DUB_VIDEO = "output/AI配音.mp4"
AUDIO_DIR = "output/audio"
DUB_SUB_FILE = "output/dub.srt"
SRC_SRT = "output/dub_orig.srt"
DUB_AUDIO = 'output/dub.wav' if os.path.exists('output/dub.wav') else 'output/dub.mp3'
ORIGINAL_CONTEXT_AUDIO = 'output/audio/original_non_dub_context.wav'
BACKGROUND_CONTEXT_AUDIO = 'output/audio/background_dub_context.wav'

def get_ffmpeg_binary() -> str:
    """Return an ffmpeg binary with libass subtitle support."""
    try:
        res = subprocess.run(['ffmpeg', '-filters'], capture_output=True, text=True, timeout=5)
        if 'subtitles' in res.stdout:
            return 'ffmpeg'
    except Exception:
        pass
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    return 'ffmpeg'

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
    target_lufs = float(load_key("dubbing_quality.loudness_target_lufs", -18.0))
    cmd = [
        get_ffmpeg_binary(), "-y", "-i", audio_path,
        "-af", f"loudnorm=I={target_lufs}:TP=-1.0:LRA=11",
        "-c:a", "pcm_s16le", "-ar", "48000", output_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        rprint(f"[green]✅ Audio normalized with ffmpeg loudnorm to {target_lufs:.1f} LUFS[/green]")
        return output_path

    audio = AudioSegment.from_file(audio_path)
    change_in_dBFS = target_lufs - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)
    normalized_audio.export(output_path, format="wav")
    rprint(f"[green]✅ Audio normalized from {audio.dBFS:.1f}dB to {target_lufs:.1f}dB (dBFS fallback)[/green]")
    return output_path


def _normalise_dub_windows(times, source_duration_ms: int):
    """Return sorted, merged TTS windows clamped to the source duration."""
    windows = []
    for value in times:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        try:
            start_ms = max(0, int(round(float(value[0]) * 1000)))
            end_ms = min(source_duration_ms, int(round(float(value[1]) * 1000)))
        except (TypeError, ValueError):
            continue
        if end_ms > start_ms:
            windows.append((start_ms, end_ms))

    merged = []
    for start_ms, end_ms in sorted(windows):
        if merged and start_ms <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_ms))
        else:
            merged.append((start_ms, end_ms))
    return merged


def _build_audio_context_tracks(source_audio: AudioSegment, background_audio: AudioSegment, windows):
    """Keep source audio outside dub windows and Demucs background inside them.

    Replacing the full source with a Demucs background stem makes intro/outro and
    other no-dialogue moments sound unlike the original.  These complementary
    context tracks keep the untouched source where no translated speech is
    scheduled, then use only the separated background under the cloned voice.
    """
    source_audio = source_audio.set_channels(2)
    background_audio = (
        background_audio
        .set_frame_rate(source_audio.frame_rate)
        .set_channels(source_audio.channels)
    )
    duration_ms = len(source_audio)
    silence = AudioSegment.silent(duration=duration_ms, frame_rate=source_audio.frame_rate)
    silence = silence.set_channels(source_audio.channels)

    original_context = source_audio
    background_context = silence
    for start_ms, end_ms in windows:
        muted = AudioSegment.silent(duration=end_ms - start_ms, frame_rate=source_audio.frame_rate)
        muted = muted.set_channels(source_audio.channels)
        original_context = original_context[:start_ms] + muted + original_context[end_ms:]
        background_context = background_context.overlay(background_audio[start_ms:end_ms], position=start_ms)
    return original_context, background_context


def _collect_speech_mask_windows(input_excel: str, source_duration_ms: int) -> list[tuple[int, int]]:
    """Collect all regions containing dialogue/speech to mute in original context."""
    raw_windows = []
    # 1. Dubbed windows from tts_tasks.xlsx
    if os.path.isfile(input_excel):
        _, _, _, times = load_and_flatten_data(input_excel)
        for val in times:
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                raw_windows.append(val)

    # 2. Source speech segments from cleaned_chunks.xlsx if present
    cleaned_chunks_file = "output/log/cleaned_chunks.xlsx"
    if os.path.isfile(cleaned_chunks_file):
        try:
            import pandas as pd
            df_chunks = pd.read_excel(cleaned_chunks_file)
            for _, r in df_chunks.iterrows():
                s = r.get("start")
                e = r.get("end")
                if pd.notna(s) and pd.notna(e) and float(e) > float(s):
                    raw_windows.append((float(s), float(e)))
        except Exception:
            pass

    # 3. Acoustic vocal activity from vocal.wav:
    # Distinguish dialogue speech from isolated non-verbal events (laughter, sighs, gasps).
    # Dialogue speech is masked so original speech does not leak; isolated non-verbal events
    # outside dialogue are preserved in original context to retain movie atmospheric fidelity.
    vocal_wav = "output/audio/vocal.wav"
    events_report = []
    if os.path.isfile(vocal_wav):
        try:
            from pydub import AudioSegment
            from pydub.silence import detect_nonsilent
            v_seg = AudioSegment.from_file(vocal_wav)
            thresh = min(-35, math.floor(v_seg.dBFS - 12)) if math.isfinite(v_seg.dBFS) else -35
            v_spans = detect_nonsilent(v_seg, min_silence_len=150, silence_thresh=thresh, seek_step=20)

            # Known dialogue spans
            dialogue_spans = list(raw_windows)
            for s_ms, e_ms in v_spans:
                s_sec = s_ms / 1000.0
                e_sec = e_ms / 1000.0
                # Check if this vocal span is close to or overlaps any dialogue chunk
                near_dialogue = any(
                    max(s_sec, d_s - 0.2) <= min(e_sec, d_e + 0.2)
                    for d_s, d_e in dialogue_spans
                ) if dialogue_spans else True

                if near_dialogue or not dialogue_spans:
                    raw_windows.append((s_sec, e_sec))
                else:
                    events_report.append({
                        "event_type": "uncertain",
                        "start": round(s_sec, 3),
                        "end": round(e_sec, 3),
                        "duration": round(e_sec - s_sec, 3),
                        "preservation": "retained_in_original_context",
                    })
            if events_report:
                os.makedirs(AUDIO_DIR, exist_ok=True)
                with open(os.path.join(AUDIO_DIR, "nonverbal_events.json"), "w", encoding="utf-8") as f:
                    json.dump(events_report, f, ensure_ascii=False, indent=2)
                rprint(f"[green]🎭 Preserved {len(events_report)} non-verbal vocal event(s) in original context track.[/green]")
        except Exception as exc:
            rprint(f"[yellow]⚠️ Could not process vocal activity from {vocal_wav}: {exc}[/yellow]")

    return _normalise_dub_windows(raw_windows, source_duration_ms)


def prepare_audio_context_tracks(source_video: str, background_file: str):
    """Write complementary original/background tracks from the current TTS timeline."""
    if not os.path.isfile(background_file):
        raise FileNotFoundError(f"Separated background audio is missing: {background_file}")
    if not os.path.isfile(INPUT_EXCEL):
        raise FileNotFoundError(f"TTS timeline is missing: {INPUT_EXCEL}")

    source_audio = AudioSegment.from_file(source_video)
    background_audio = AudioSegment.from_file(background_file)
    windows = _collect_speech_mask_windows(INPUT_EXCEL, len(source_audio))
    if not windows:
        raise RuntimeError("No valid dubbing windows were found; refusing to mute the original audio.")

    original_context, background_context = _build_audio_context_tracks(
        source_audio, background_audio, windows
    )
    os.makedirs(os.path.dirname(ORIGINAL_CONTEXT_AUDIO), exist_ok=True)
    original_context.export(ORIGINAL_CONTEXT_AUDIO, format='wav')
    background_context.export(BACKGROUND_CONTEXT_AUDIO, format='wav')
    rprint(
        f"[green]✅ Preserved original audio outside {len(windows)} dubbed windows; "
        "using separated background only beneath cloned speech.[/green]"
    )
    return ORIGINAL_CONTEXT_AUDIO, BACKGROUND_CONTEXT_AUDIO

def measure_final_audio_loudness(media_path: str) -> dict:
    """Measure True Peak (dBTP), Integrated Loudness (LUFS), and LRA (LU) on decoded audio stream."""
    cmd = [
        get_ffmpeg_binary(), "-hide_banner", "-i", media_path,
        "-af", "ebur128=peak=true", "-f", "null", "-"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    last = res.stderr[res.stderr.rfind("Summary:"):] if "Summary:" in res.stderr else res.stderr

    true_peak = None
    lufs = None
    lra = None
    for line in last.splitlines():
        line = line.strip()
        m_peak = re.search(r"Peak:\s+(-?[\d.]+)\s+dBFS", line)
        if m_peak:
            try:
                true_peak = float(m_peak.group(1))
            except ValueError:
                pass
        m_i = re.search(r"I:\s+(-?[\d.]+)\s+LUFS", line)
        if m_i:
            try:
                lufs = float(m_i.group(1))
            except ValueError:
                pass
        m_lra = re.search(r"LRA:\s+(-?[\d.]+)\s+LU", line)
        if m_lra:
            try:
                lra = float(m_lra.group(1))
            except ValueError:
                pass
    return {
        "true_peak_dBTP": true_peak,
        "integrated_lufs": lufs,
        "loudness_range_lu": lra,
    }

def validate_merged_video(video_path: str, source_video_path: str | None = None, tolerance_seconds: float | None = None) -> dict:
    """Validate that the rendered video is non-empty, decodable, and contains audio/video streams."""
    if not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
        raise RuntimeError(f"Merged video file {video_path} does not exist or is empty.")

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,start_time:stream=codec_type,start_time,duration,sample_rate,channels,avg_frame_rate",
        "-of", "json", video_path
    ]
    probe = subprocess.run(cmd, capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed to read merged video {video_path}: {probe.stderr}")

    info = json.loads(probe.stdout)
    streams = info.get("streams", [])
    format_info = info.get("format", {})
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    if not has_video:
       raise RuntimeError(f"Merged file {video_path} is missing a video stream.")
    if not has_audio:
        raise RuntimeError(f"Merged file {video_path} is missing an audio stream.")

    dur_str = format_info.get("duration")
    if not dur_str or float(dur_str) <= 0:
        raise RuntimeError(f"Merged file {video_path} has invalid duration: {dur_str}")
    format_dur = float(dur_str)

    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    v_dur = float(v_stream.get("duration") or format_dur) if v_stream else format_dur
    a_dur = float(a_stream.get("duration") or format_dur) if a_stream else format_dur
    v_start = float(v_stream.get("start_time") or format_info.get("start_time") or 0.0) if v_stream else 0.0
    a_start = float(a_stream.get("start_time") or format_info.get("start_time") or 0.0) if a_stream else 0.0

    if a_dur <= 0:
        raise RuntimeError(f"Merged file {video_path} has invalid audio stream duration: {a_dur}")
    if v_dur <= 0:
        raise RuntimeError(f"Merged file {video_path} has invalid video stream duration: {v_dur}")

    # Calculate frame-accurate tolerance: max(1 video frame, 1 audio frame)
    v_fps_str = str(v_stream.get("avg_frame_rate", "")).strip() if v_stream else ""
    v_frame_dur = None
    if v_fps_str and v_fps_str not in {"0/0", "N/A"}:
        try:
            if "/" in v_fps_str:
                num, den = v_fps_str.split("/")
                v_fps = float(num) / float(den) if float(den) > 0 else 0.0
            else:
                v_fps = float(v_fps_str)
            if v_fps > 0 and math.isfinite(v_fps):
                v_frame_dur = 1.0 / v_fps
        except Exception:
            pass

    a_sr_str = a_stream.get("sample_rate", "48000") if a_stream else "48000"
    try:
        a_sr = float(a_sr_str)
    except Exception:
        a_sr = 48000.0
    a_frame_dur = 1024.0 / max(a_sr, 8000.0)

    if v_frame_dur is not None:
        computed_tolerance = max(v_frame_dur, a_frame_dur)
    else:
        # Without verified video frame rate, tolerance cannot assume video frames
        computed_tolerance = a_frame_dur

    eff_tolerance = float(tolerance_seconds) if tolerance_seconds is not None else computed_tolerance

    diff_start = abs(v_start - a_start)
    if diff_start > eff_tolerance:
        raise RuntimeError(
            f"Merged file {video_path} audio stream start ({a_start:.3f}s) deviates from "
            f"video stream start ({v_start:.3f}s) by {diff_start:.3f}s (tolerance: {eff_tolerance:.3f}s). "
            f"Audio stream is shifted!"
        )

    # For stream duration within output: allow up to 1 audio encoding packet / video frame
    diff_stream = abs(v_dur - a_dur)
    stream_internal_tol = max(eff_tolerance, 0.065)
    if diff_stream > stream_internal_tol:
        raise RuntimeError(
            f"Merged file {video_path} audio stream duration ({a_dur:.3f}s) deviates from "
            f"video stream duration ({v_dur:.3f}s) by {diff_stream:.3f}s (tolerance: {stream_internal_tol:.3f}s). "
            f"Audio tail is truncated or out of sync!"
        )

    if source_video_path:
       if not os.path.isfile(source_video_path):
           raise RuntimeError(f"Source video file {source_video_path} not found for validation.")
       src_cmd = [
           "ffprobe", "-v", "error",
           "-show_entries", "format=duration,start_time:stream=codec_type,start_time,duration,avg_frame_rate",
           "-of", "json", source_video_path
       ]
       src_probe = subprocess.run(src_cmd, capture_output=True, text=True)
       if src_probe.returncode != 0:
           raise RuntimeError(f"ffprobe failed to read source video {source_video_path}: {src_probe.stderr}")
       if True:
           try:
               src_info = json.loads(src_probe.stdout)
               src_fmt_dur = float(src_info.get("format", {}).get("duration", 0) or 0)
               src_v_stream = next((s for s in src_info.get("streams", []) if s.get("codec_type") == "video"), None)
               src_a_stream = next((s for s in src_info.get("streams", []) if s.get("codec_type") == "audio"), None)
               src_v_dur = float(src_v_stream.get("duration") or src_fmt_dur) if src_v_stream else src_fmt_dur
               src_a_dur = float(src_a_stream.get("duration") or src_fmt_dur) if src_a_stream else src_fmt_dur

               if src_v_dur > 0:
                   diff_v = abs(v_dur - src_v_dur)
                   if diff_v > eff_tolerance:
                       raise RuntimeError(
                           f"Merged video duration {v_dur:.3f}s differs from source video duration {src_v_dur:.3f}s "
                           f"by {diff_v:.3f}s (tolerance: {eff_tolerance:.3f}s)!"
                       )
               if src_a_dur > 0:
                   diff_a = abs(a_dur - src_a_dur)
                   if diff_a > eff_tolerance:
                       raise RuntimeError(
                           f"Merged audio duration {a_dur:.3f}s differs from source audio duration {src_a_dur:.3f}s "
                           f"by {diff_a:.3f}s (tolerance: {eff_tolerance:.3f}s)!"
                       )
           except (KeyError, ValueError, TypeError) as exc:
               rprint(f"[yellow]⚠️ Could not parse source video duration: {exc}[/yellow]")

    return info

def _align_wav_duration_to_video(wav_path: str, video_path: str):
    """Ensure WAV file matches exact video duration if audio filtergraph ended slightly early."""
    if not os.path.isfile(wav_path) or not os.path.isfile(video_path):
        return
    try:
        import soundfile as sf, numpy as np
        cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,duration", "-of", "json", video_path]
        p_dur = subprocess.run(cmd_dur, capture_output=True, text=True)
        if p_dur.returncode != 0:
            return
        info_dur = json.loads(p_dur.stdout)
        v_st = next((s for s in info_dur.get("streams", []) if s.get("codec_type") == "video"), None)
        v_dur = float(v_st.get("duration") or info_dur.get("format", {}).get("duration", 0) or 0) if v_st else float(info_dur.get("format", {}).get("duration", 0) or 0)
        if v_dur <= 0:
            return
        with sf.SoundFile(wav_path) as h:
            master_frames = h.frames
            master_sr = h.samplerate
            target_frames = int(round(v_dur * master_sr))
        if master_frames < target_frames:
            data, sr_in = sf.read(wav_path, dtype="int16", always_2d=True)
            pad_frames = target_frames - len(data)
            padded_data = np.pad(data, ((0, pad_frames), (0, 0)), mode="constant")
            sf.write(wav_path, padded_data, sr_in, subtype="PCM_16")
    except Exception:
        pass

def merge_video_audio():
    """Merge video and audio with subtitle burning (two-pass for FFmpeg 8.x compatibility)."""
    if not os.path.isfile(SOURCE_VIDEO):
        raise FileNotFoundError(
            f"Canonical source video is missing: {SOURCE_VIDEO}. Refusing to use an "
            "already-subtitled or previously dubbed derivative as merge input."
        )
    VIDEO_FILE = SOURCE_VIDEO
    background_wav = "output/audio/background.wav"
    background_file = background_wav if os.path.isfile(background_wav) else BACKGROUND_AUDIO_FILE

    normalized_dub_audio = 'output/normalized_dub.wav'
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)
    preserve_original_non_speech = bool(
        load_key("audio_mix.preserve_original_non_speech", True)
    )
    temp_dub_video = 'output/_temp_final_dubbed.mp4'
    if preserve_original_non_speech:
        original_context_file, dubbed_background_file = prepare_audio_context_tracks(
            VIDEO_FILE, background_file
        )
    else:
        original_context_file, dubbed_background_file = None, background_file

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
        sub_cmd = [get_ffmpeg_binary(), '-y', '-i', VIDEO_FILE, '-vf', vf, '-an']

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

    background_volume = float(load_key("background_volume", 1.0))
    enable_ducking = bool(load_key("audio_mix.ducking", False))
    ducking_ratio = float(load_key("audio_mix.ducking_ratio", 2.0))
    ducking_threshold = float(load_key("audio_mix.ducking_threshold", 0.08))

    if preserve_original_non_speech and original_context_file and dubbed_background_file:
        audio_inputs = ['-i', original_context_file, '-i', dubbed_background_file, '-i', normalized_dub_audio]
        if enable_ducking:
            audio_filter = (
                f"[0:a]aformat=channel_layouts=stereo,aresample=48000[orig_ctx];"
                f"[1:a]volume={background_volume},aformat=channel_layouts=stereo,aresample=48000[dub_bg];"
                f"[orig_ctx][dub_bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[full_bg];"
                f"[2:a]aformat=channel_layouts=stereo,aresample=48000,apad[dub];"
                f"[dub]asplit[dub_sc][dub_mix];"
                f"[full_bg][dub_sc]sidechaincompress=threshold={ducking_threshold}:ratio={ducking_ratio}:attack=20:release=250[ducked_bg];"
                f"[ducked_bg][dub_mix]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.75:attack=5:release=50:asc=true:level=false:latency=true[a]"
            )
        else:
            audio_filter = (
                f"[0:a]aformat=channel_layouts=stereo,aresample=48000[orig_ctx];"
                f"[1:a]volume={background_volume},aformat=channel_layouts=stereo,aresample=48000[dub_bg];"
                f"[orig_ctx][dub_bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[full_bg];"
                f"[2:a]aformat=channel_layouts=stereo,aresample=48000,apad[dub];"
                f"[full_bg][dub]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.75:attack=5:release=50:asc=true:level=false:latency=true[a]"
            )
    else:
        audio_inputs = ['-i', background_file, '-i', normalized_dub_audio]
        if enable_ducking:
            audio_filter = (
                f"[0:a]volume={background_volume},aformat=channel_layouts=stereo,aresample=48000[bg];"
                f"[1:a]aformat=channel_layouts=stereo,aresample=48000,apad[dub];"
                f"[dub]asplit[dub_sc][dub_mix];"
                f"[bg][dub_sc]sidechaincompress=threshold={ducking_threshold}:ratio={ducking_ratio}:attack=20:release=250[docked_bg];"
                f"[docked_bg][dub_mix]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.75:attack=5:release=50:asc=true:level=false:latency=true[a]"
            )
        else:
            audio_filter = (
                f"[0:a]volume={background_volume},aformat=channel_layouts=stereo,aresample=48000[bg];"
                f"[1:a]aformat=channel_layouts=stereo,aresample=48000,apad[dub];"
                f"[bg][dub]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.75:attack=5:release=50:asc=true:level=false:latency=true[a]"
            )

    # Step A: Render the exact pre-encode lossless mixing master directly from filter_complex
    lossless_master_wav = os.path.join(AUDIO_DIR, "final_mix_master.wav")
    render_master_cmd = [
        get_ffmpeg_binary(), '-y', '-threads', '0',
        *audio_inputs,
        '-filter_complex', audio_filter,
        '-map', '[a]',
        '-c:a', 'pcm_s16le', '-ar', '48000',
        lossless_master_wav,
    ]
    res_master = subprocess.run(render_master_cmd, capture_output=True, text=True)
    if res_master.returncode != 0:
        raise RuntimeError(f"FFmpeg render lossless master failed: {res_master.stderr[-800:]}")
    rprint(f"[green]✅ Pre-encode lossless mixing master saved → {lossless_master_wav}[/green]")

    _align_wav_duration_to_video(lossless_master_wav, VIDEO_FILE)

    # Step B: Mux video with the exact pre-encode master audio
    cmd = [
        get_ffmpeg_binary(), '-y', '-threads', '0',
        '-i', VIDEO_FILE,
        '-i', lossless_master_wav,
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'copy',

        '-c:a', 'aac', '-b:a', '256k', '-ar', '48000',
        '-movflags', '+faststart',
        temp_dub_video,
    ]
    run_res = subprocess.run(cmd, capture_output=True, text=True)
    if run_res.returncode != 0:
       if os.path.exists(temp_dub_video):
           os.remove(temp_dub_video)
       raise RuntimeError(f"FFmpeg final merge failed with code {run_res.returncode}: {run_res.stderr[-800:]}")

    # Step C: Audit Decoded AAC True Peak / LUFS / LRA and perform limited bus correction if needed
    stats = measure_final_audio_loudness(temp_dub_video)
    max_tp = stats.get("true_peak_dBTP")
    stats_file = os.path.join(AUDIO_DIR, "final_mix_stats.json")
    if max_tp is not None and max_tp > -1.0:
        rprint(f"[yellow]⚠️ Decoded AAC True Peak {max_tp:.2f} dBTP exceeds -1.0 dBTP. Applying limited bus correction...[/yellow]")
        reduction_db = min(3.0, max(0.2, max_tp - (-1.5)))
        corrected_master_wav = os.path.join(AUDIO_DIR, "final_mix_master_corrected.wav")
        corr_cmd = [
            get_ffmpeg_binary(), "-y", "-i", lossless_master_wav,
            "-af", f"volume=-{reduction_db:.2f}dB",
            "-c:a", "pcm_s16le", "-ar", "48000",
            corrected_master_wav
        ]
        res_corr = subprocess.run(corr_cmd, capture_output=True, text=True)
        if res_corr.returncode == 0 and os.path.isfile(corrected_master_wav):
            _align_wav_duration_to_video(corrected_master_wav, VIDEO_FILE)
            remux_cmd = [
                get_ffmpeg_binary(), '-y', '-threads', '0',
                '-i', VIDEO_FILE,
                '-i', corrected_master_wav,
                '-map', '0:v', '-map', '1:a',
                '-c:v', 'copy',

                '-c:a', 'aac', '-b:a', '256k', '-ar', '48000',
                '-movflags', '+faststart',
                temp_dub_video,
            ]
            subprocess.run(remux_cmd, capture_output=True, text=True)
            stats = measure_final_audio_loudness(temp_dub_video)
            max_tp = stats.get("true_peak_dBTP")
            stats["bus_correction_applied_dB"] = reduction_db

    stats["gate_passed"] = bool(max_tp is not None and max_tp <= -1.0)
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if max_tp is None:
        raise RuntimeError("Decoded audio loudness/True Peak measurement failed or is unavailable!")
    if max_tp > -1.0:
        raise RuntimeError(f"Decoded AAC True Peak {max_tp:.2f} dBTP exceeds limit -1.0 dBTP!")

    validate_merged_video(temp_dub_video, source_video_path=SOURCE_VIDEO)
    shutil.move(temp_dub_video, DUB_VIDEO)
    rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")

    # Cleanup temp file
    if burn_subs and os.path.exists('output/_temp_subbed.mp4'):
        os.remove('output/_temp_subbed.mp4')

if __name__ == '__main__':
    merge_video_audio()
