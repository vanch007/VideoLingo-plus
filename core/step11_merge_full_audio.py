import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pandas as pd
import numpy as np
import subprocess
from pydub import AudioSegment
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from core.config_utils import load_key
from core.dubbing_quality import DUBBING_MERGE_ISSUES_JSON, get_quality_config, normalize_lines, parse_list, write_dubbing_eval
console = Console()

INPUT_EXCEL = 'output/audio/tts_tasks.xlsx'
DUB_VOCAL_FILE = 'output/dub.mp3'

DUB_SUB_FILE = 'output/dub.srt'
SEGS_DIR = 'output/audio/segs'
OUTPUT_FILE_TEMPLATE = f"{SEGS_DIR}/{{}}.wav"

def load_and_flatten_data(excel_file):
    """Load Excel data and flatten timestamps"""
    df = pd.read_excel(excel_file)
    from core.utterance_plan import plan_active, load_plan, validate_tasks
    if plan_active():
        plan = load_plan()
        validate_tasks(df, plan, generated=True)
        from core.atomic_dubbing import verify_preflight_identity
        verify_preflight_identity(pd.DataFrame([{"LineID": u["unit_id"], "Source": u["source"],
                                                 "Translation": u["target"]} for u in plan["units"]]))
        return df, [u["target"] for u in plan["units"]], [u["source"] for u in plan["units"]], [[u["start"], u["end"]] for u in plan["units"]]

    # Flatten lines (translated text)
    if 'lines' in df.columns:
        lines = [normalize_lines(line) for line in df['lines'].tolist()]
        lines = [item for sublist in lines for item in sublist]
    else:
        # Fallback to 'text' if 'lines' is missing (should not happen with new step8_2)
        lines = df['text'].tolist()

    # Flatten src_lines (original text), matching line count per task row
    src_lines = []
    for _, row in df.iterrows():
        row_lines = normalize_lines(row.get('lines', row.get('text', '')))
        n_lines = len(row_lines)
        origin = str(row.get('origin', '')).strip()
        raw_src = parse_list(row.get('src_lines')) if 'src_lines' in row else []
        if raw_src and len(raw_src) == n_lines:
            src_lines.extend(raw_src)
        elif n_lines == 1:
            src_lines.append(origin)
        else:
            raise ValueError('Subtitle/source/audio timeline cardinality mismatch; regenerate paired subtitles')

    # Flatten timestamps
    if 'new_sub_times' in df.columns:
        times = [parse_list(t) for t in df['new_sub_times'].tolist()]
        # Flatten if it's list of lists (step10 generates list of lists)
        if times and isinstance(times[0], list):
             times = [item for sublist in times for item in sublist]
    else:
         times = [parse_list(t) for t in df['sub_times'].tolist()]

    if not (len(lines) == len(src_lines) == len(times)):
        raise ValueError("Subtitle/source/audio timeline cardinality mismatch; refusing zip truncation")
    return df, lines, src_lines, times

def get_audio_files(df):
    """Generate a list of audio file paths"""
    audios = []
    for index, row in df.iterrows():
        number = row['number']
        if 'lines' in df.columns:
            lines_data = normalize_lines(row.get('lines', row.get('text', '')))
            line_count = len(lines_data)
            for line_index in range(line_count):
                audio_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
                audios.append(audio_file)
        else:
            audio_file = OUTPUT_FILE_TEMPLATE.format(f"{number}")
            audios.append(audio_file)
    return audios

def process_audio_segment(audio_file, allow_silence_fallback=False):
    """Process a single audio segment with MP3 compression"""
    # 检查文件是否存在且有效
    if not os.path.exists(audio_file):
        console.print(f"[bold red]❌ Audio file does not exist: {audio_file}[/bold red]")
        if not allow_silence_fallback:
            raise FileNotFoundError(audio_file)
        # 创建一个100ms的静音文件作为替代
        silence = AudioSegment.silent(duration=100)
        temp_file = f"{audio_file}_temp_silence.wav"
        silence.export(temp_file, format="wav")
        audio_file = temp_file
    elif os.path.getsize(audio_file) < 1000:  # 文件太小，可能已损坏
        console.print(f"[bold yellow]⚠️ Audio file too small ({os.path.getsize(audio_file)} bytes): {audio_file}[/bold yellow]")
        if not allow_silence_fallback:
            raise ValueError(f"Audio file too small: {audio_file}")
        # 创建一个100ms的静音文件作为替代
        silence = AudioSegment.silent(duration=100)
        temp_file = f"{audio_file}_temp_silence.wav"
        silence.export(temp_file, format="wav")
        audio_file = temp_file

    audio_segment = AudioSegment.from_file(audio_file)
    audio_segment = audio_segment.set_frame_rate(24000).set_channels(1)
    # Apply smooth micro-fade to eliminate digital DC-offset click and buzzing on entry/exit
    if len(audio_segment) > 40:
        audio_segment = audio_segment.fade_in(15).fade_out(15)

    # 如果是临时静音文件，也删除它
    if "_temp_silence.wav" in audio_file:
        os.remove(audio_file)

    return audio_segment


def _write_merge_issues(merge_issues):
    if not merge_issues:
        if os.path.exists(DUBBING_MERGE_ISSUES_JSON):
            os.remove(DUBBING_MERGE_ISSUES_JSON)
        return
    os.makedirs(os.path.dirname(DUBBING_MERGE_ISSUES_JSON), exist_ok=True)
    with open(DUBBING_MERGE_ISSUES_JSON, "w", encoding="utf-8") as f:
        json.dump(merge_issues, f, ensure_ascii=False, indent=2)

def merge_audio_segments(audios, new_sub_times, sample_rate, tasks_df=None):
    quality = get_quality_config()
    if len(audios) != len(new_sub_times):
        raise ValueError(
            f"Audio/timeline cardinality mismatch: {len(audios)} audio files "
            f"but {len(new_sub_times)} timestamp ranges"
        )
    total_duration_ms = int(max(end for _, end in new_sub_times) * 1000) if new_sub_times else 0
    merged_audio = AudioSegment.silent(duration=total_duration_ms, frame_rate=sample_rate)
    merge_issues = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        merge_task = progress.add_task("🎵 Merging audio segments...", total=len(audios))

        for i, (audio_file, time_range) in enumerate(zip(audios, new_sub_times)):
            try:
                audio_segment = process_audio_segment(audio_file, allow_silence_fallback=quality.allow_silence_fallback)
                # Per-sentence dynamic energy alignment
                energy_cfg = load_key("dubbing_quality.dynamic_energy_alignment", {}) or {}
                if bool(energy_cfg.get("enabled", True)) and tasks_df is not None and i < len(tasks_df):
                    row = tasks_df.iloc[i]
                    ref_audio = row.get("resolved_emotion_ref") or row.get("resolved_speaker_ref")
                    if ref_audio and os.path.isfile(str(ref_audio)):
                        try:
                            import math
                            ref_seg = AudioSegment.from_file(str(ref_audio))
                            if math.isfinite(ref_seg.dBFS) and math.isfinite(audio_segment.dBFS) and ref_seg.rms > 0 and audio_segment.rms > 0:
                                ratio = float(energy_cfg.get("ratio", 0.8))
                                max_boost = float(energy_cfg.get("max_gain_db", 5.0))
                                max_cut = float(energy_cfg.get("min_gain_db", -6.0))
                                delta = ref_seg.dBFS - audio_segment.dBFS
                                gain = max(max_cut, min(max_boost, delta * ratio))
                                if abs(gain) >= 0.2:
                                    audio_segment = audio_segment.apply_gain(gain)
                        except Exception:
                            pass
                # Peak headroom limiter to prevent digital clipping and buzzing
                if audio_segment.max_possible_amplitude > 0:
                    peak_ratio = audio_segment.max / float(audio_segment.max_possible_amplitude)
                    if peak_ratio > 0.88:
                        import math
                        headroom_reduction_db = 20.0 * math.log10(peak_ratio / 0.88)
                        audio_segment = audio_segment.apply_gain(-headroom_reduction_db)
                start_time, end_time = time_range
                next_start = new_sub_times[i + 1][0] if i + 1 < len(new_sub_times) else (float(total_duration_ms) / 1000.0 if total_duration_ms else end_time)
                hard_end = float(next_start) if float(next_start) > float(start_time) else float(end_time)
                window_ms = max(0, int(round((hard_end - float(start_time)) * 1000)))
                if window_ms <= 0:
                    raise ValueError(f"Invalid or overlapping timestamp window: {time_range}")
                if len(audio_segment) > window_ms:
                    overflow_ms = len(audio_segment) - window_ms
                    rounding_tolerance_ms = int(
                        load_key("dubbing_quality.timeline_rounding_tolerance_ms", 10)
                    )
                    if overflow_ms <= rounding_tolerance_ms:
                        audio_segment = audio_segment[:window_ms]
                        if len(audio_segment) > 10:
                            audio_segment = audio_segment.fade_out(10)
                        merge_issues.append({
                            "audio_file": audio_file,
                            "time_range": [float(start_time), float(end_time)],
                            "error": "trimmed_encoding_rounding_tail",
                            "overflow_ms": overflow_ms,
                            "silence_fallback": False,
                        })
                    elif os.path.isfile(audio_file) and bool(load_key("dubbing_quality.speed_fit_overflow", True)):
                        from core.audio_speed import adjust_audio_speed
                        import tempfile
                        speed = len(audio_segment) / max(1, window_ms)
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_fit:
                            tmp_fit_path = tmp_fit.name
                        try:
                            adjust_audio_speed(audio_file, tmp_fit_path, speed)
                            audio_segment = AudioSegment.from_file(tmp_fit_path).set_frame_rate(sample_rate).set_channels(1)
                            if len(audio_segment) > window_ms:
                                audio_segment = audio_segment[:window_ms]
                            merge_issues.append({
                                "audio_file": audio_file,
                                "time_range": [float(start_time), float(end_time)],
                                "error": "fitted_overflow_speed",
                                "overflow_ms": overflow_ms,
                                "speed_factor": speed,
                                "silence_fallback": False,
                            })
                        finally:
                            if os.path.exists(tmp_fit_path):
                                os.remove(tmp_fit_path)
                    else:
                        raise RuntimeError(
                            f"Audio exceeds its hard timeline boundary by {overflow_ms}ms. "
                            "Refusing to truncate spoken content; rewrite/regenerate before merge."
                        )
                # Smooth entry/exit crossfade on the positioned clip to avoid DC click / edge buzzing
                fade_dur = min(15, len(audio_segment) // 4)
                if fade_dur > 2:
                    audio_segment = audio_segment.fade_in(fade_dur).fade_out(fade_dur)
                merged_audio = merged_audio.overlay(audio_segment, position=max(0, int(start_time * 1000)))

            except Exception as e:
                console.print(f"[bold red]❌ Error processing {audio_file}: {str(e)}[/bold red]")
                merge_issues.append({
                    "audio_file": audio_file,
                    "time_range": time_range,
                    "error": str(e),
                    "silence_fallback": quality.allow_silence_fallback,
                })
                if quality.allow_silence_fallback:
                    silent_segment = AudioSegment.silent(duration=100, frame_rate=sample_rate)
                    merged_audio = merged_audio.overlay(silent_segment, position=max(0, int(time_range[0] * 1000)))
                else:
                    _write_merge_issues(merge_issues)
                    raise

            progress.advance(merge_task)

    _write_merge_issues(merge_issues)
    return merged_audio

def create_srt_subtitle():
    from core.utterance_plan import plan_active
    if plan_active():
        from core.plan_subtitles import render_plan_subtitles
        render_plan_subtitles(dubbed=True)
        return
    # Correctly unpack all four returned values
    df, lines, _, new_sub_times = load_and_flatten_data(INPUT_EXCEL)

    with open(DUB_SUB_FILE, 'w', encoding='utf-8') as f:
        for i, ((start_time, end_time), line) in enumerate(zip(new_sub_times, lines), 1):
            start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},{int((start_time*1000)%1000):03d}"
            end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},{int((end_time*1000)%1000):03d}"

            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            from core.subtitle_layout import compute_subtitle_line_limits, probe_video_dimensions, split_display_text
            cjk_cap, latin_cap = compute_subtitle_line_limits(*(probe_video_dimensions() or (None, None)))
            is_cjk = bool(re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", str(line)))
            cap = cjk_cap if is_cjk else latin_cap
            wrapped = "\n".join(split_display_text(str(line), cap))
            f.write(f"{wrapped}\n\n")

    rprint(f"[bold green]✅ Dub subtitle file created: {DUB_SUB_FILE}[/bold green]")

def create_orig_srt_subtitle():
    """Create synchronized SRT subtitle files (original & bilingual) directly matching new_sub_times"""
    from core.utterance_plan import plan_active
    if plan_active():
        from core.plan_subtitles import render_plan_subtitles
        render_plan_subtitles(dubbed=True)
        return
    DUB_ORIG_SRT_FILE = 'output/dub_orig.srt'
    bilingual_srt_path = 'output/dub_bilingual.srt'
    df, lines_data, origins, new_sub_times = load_and_flatten_data(INPUT_EXCEL)

    with open(DUB_ORIG_SRT_FILE, 'w', encoding='utf-8') as f_orig, open(bilingual_srt_path, 'w', encoding='utf-8') as f_bi:
        for idx, ((start_time, end_time), orig_line, trans_line) in enumerate(zip(new_sub_times, origins, lines_data), 1):
            start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},{int((start_time*1000)%1000):03d}"
            end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},{int((end_time*1000)%1000):03d}"
            time_header = f"{start_str} --> {end_str}\n"
            from core.subtitle_layout import compute_subtitle_line_limits, probe_video_dimensions, split_display_text
            cjk_cap, latin_cap = compute_subtitle_line_limits(*(probe_video_dimensions() or (None, None)))
            orig_cjk = bool(re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", str(orig_line)))
            trans_cjk = bool(re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", str(trans_line)))
            wrapped_orig = "\n".join(split_display_text(str(orig_line), cjk_cap if orig_cjk else latin_cap))
            wrapped_trans = "\n".join(split_display_text(str(trans_line), cjk_cap if trans_cjk else latin_cap))
            f_orig.write(f"{idx}\n{time_header}{wrapped_orig}\n\n")
            f_bi.write(f"{idx}\n{time_header}{wrapped_orig}\n{wrapped_trans}\n\n")

    rprint(f"[bold green]✅ Synchronized dub subtitles created: {DUB_ORIG_SRT_FILE} & {bilingual_srt_path}[/bold green]")


def merge_full_audio():
    from core.config_utils import load_key
    if (load_key("quality_mode", "high_sync") == "fast") or bool(load_key("fast_mode.enabled", False)):
        from pathlib import Path
        if Path("output/audio/dub.wav").is_file():
            console.print("[green]✅ Fast Mode: Master dubbing track already composited.[/green]")
            return
        from core.fast_video_merge import build_full_dubbing_track, MANIFEST_PATH
        import json
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.is_file() else {}
        tasks = manifest.get("tasks", [])
        build_full_dubbing_track(tasks, float(manifest["source_duration"]) + float(manifest["total_video_extension_seconds"]))
        console.print("[green]✅ Fast Mode: Master dubbing track composited.[/green]")
        return
    """Main function: Process the complete audio merging process"""
    console.print("\n[bold cyan]🎬 Starting audio merging process...[/bold cyan]")

    with console.status("[bold cyan]📊 Loading data from Excel...[/bold cyan]"):
        df, lines, origins, new_sub_times = load_and_flatten_data(INPUT_EXCEL) # Load origins too
    console.print("[bold green]✅ Data loaded successfully[/bold green]")

    with console.status("[bold cyan]🔍 Getting audio file list...[/bold cyan]"):
        audios = get_audio_files(df)
    console.print(f"[bold green]✅ Found {len(audios)} audio segments[/bold green]")

    with console.status("[bold cyan]📝 Generating subtitle files...[/bold cyan]"):
        create_srt_subtitle()
        create_orig_srt_subtitle() # Generate original subtitle file

    if not os.path.exists(audios[0]):
        console.print(f"[bold red]❌ Error: First audio file {audios[0]} does not exist![/bold red]")
        raise FileNotFoundError(audios[0])

    sample_rate = 24000
    console.print(f"[bold green]✅ Sample rate: {sample_rate}Hz[/bold green]")

    console.print("[bold cyan]🔄 Starting audio merge process...[/bold cyan]")
    try:
        merged_audio = merge_audio_segments(audios, new_sub_times, sample_rate, tasks_df=df)
    except Exception as e:
        console.print(f"[bold red]❌ Error during audio merging: {str(e)}[/bold red]")
        if not get_quality_config().allow_silence_fallback:
            raise
        console.print("[bold yellow]⚠️  Creating a merged audio with silence as fallback...[/bold yellow]")
        # 计算总时长并创建静音音频作为后备方案
        total_duration_ms = int(new_sub_times[-1][1] * 1000)
        merged_audio = AudioSegment.silent(duration=total_duration_ms, frame_rate=sample_rate)

    with console.status("[bold cyan]💾 Exporting final audio file...[/bold cyan]"):
        dub_wav_path = os.path.join(os.path.dirname(DUB_VOCAL_FILE), "dub.wav")
        merged_audio.export(dub_wav_path, format="wav")
        merged_audio.export(
            DUB_VOCAL_FILE,
            format="mp3",
            parameters=["-b:a", "192k"]
        )
        summary = write_dubbing_eval(df)
    console.print(f"[bold green]✅ Audio file successfully merged![/bold green]")
    console.print(f"[bold green]📁 Output file: {DUB_VOCAL_FILE}[/bold green]")
    console.print(f"[bold green]📊 Dubbing eval: {summary}[/bold green]")

if __name__ == "__main__":
    merge_full_audio()
