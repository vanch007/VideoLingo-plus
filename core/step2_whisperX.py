import os,sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich import print as rprint
import subprocess

from core.config_utils import load_key
from core.all_whisper_methods.demucs_vl import demucs_main, RAW_AUDIO_FILE, VOCAL_AUDIO_FILE
from core.all_whisper_methods.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results, compress_audio, CLEANED_CHUNKS_EXCEL_PATH
from core.step1_ytdlp import find_video_files
from core.asr_schema import MAIN_ASR_RUNTIMES, sanitize_word_timestamps, validate_word_timestamps
from core.providers.speaker_diarization import apply_speaker_diarization, validate_cached_speaker_rows
import json

WHISPER_FILE = "output/audio/for_whisper.mp3"
ENHANCED_VOCAL_PATH = "output/audio/enhanced_vocals.mp3"
ASR_FINGERPRINT_PATH = "output/log/asr_fingerprint.json"
ASR_PROCESSING_VERSION = 2

def source_fingerprint():
    from core.pipeline.artifact_manifest import hash_path
    video = find_video_files()
    settings = dict(load_key("whisper"))
    settings.pop("detected_language", None)
    return {
        "processing_version": ASR_PROCESSING_VERSION,
        "source_sha256": hash_path(video),
        "runtime": load_key("whisper.runtime"),
        "model": load_key("whisper.model"),
        "audio_source": load_key("whisper.audio_source", "original"),
        "stable_ts_mlx": load_key("whisper.stable_ts_mlx", True),
        "language": load_key("whisper.language"),
        "whisper_settings": settings,
        "diarization_settings": load_key("speaker_diarization", {}),
        "demucs": load_key("demucs", False),
        "demucs_model": load_key("demucs_model", "htdemucs"),
        "qwen3_source": load_key("qwen3_source", {}),
        "source_asr": load_key("source_asr", {}),
        "source_review_sha256": hash_path(load_key("source_asr.review_file", "")) if load_key("source_asr.review_file", "") else None,
    }

def prepare_audio_and_vocals():
    """Prepare audio files and vocals - this is needed for both transcription and embedded subtitle workflows"""
    # step0 Convert video to audio
    video_file = find_video_files()
    convert_video_to_audio(video_file)

    # step1 Demucs vocal separation:
    if load_key("demucs"):
        demucs_model = load_key("demucs_model", "htdemucs")
        demucs_main(demucs_model)

    audio_source = load_key("whisper.audio_source", "original")
    if audio_source == "original":
        whisper_audio = "output/audio/raw_master.wav"
        if not os.path.isfile(whisper_audio):
            raise RuntimeError("Original lossless audio is missing")
    elif audio_source == "vocals":
        choose_audio = enhance_vocals() if load_key("demucs") else RAW_AUDIO_FILE
        whisper_audio = compress_audio(choose_audio, WHISPER_FILE)
    else:
        raise ValueError("whisper.audio_source must be original or vocals")

    # 确保输出目录存在
    os.makedirs('output/log', exist_ok=True)

    return whisper_audio


def transcribe_recovery_audio(ts, preferred_audio, start, end):
    from core.pipeline.artifact_manifest import hash_path
    candidates = list(dict.fromkeys(["output/audio/raw_master.wav", preferred_audio, "output/audio/vocal.wav"]))
    result = {"segments": []}
    for audio in candidates:
        if not os.path.isfile(audio):
            continue
        result = ts(audio, start, end)
        result["recovery_audio"] = {"path": audio, "sha256": hash_path(audio)}
        if any(s.get("words") for s in result.get("segments", [])):
            return result
    return result

def prepare_audio_only():
    """Prepare only audio files without vocal separation - for embedded subtitle workflow"""
    # step0 Convert video to audio
    video_file = find_video_files()
    convert_video_to_audio(video_file)

    # step1 Compress audio for whisper
    whisper_audio = compress_audio(RAW_AUDIO_FILE, WHISPER_FILE)

    # 确保输出目录存在
    os.makedirs('output/log', exist_ok=True)

    return whisper_audio

def enhance_vocals(vocals_ratio=2.50):
    """Enhance vocals audio volume"""
    # Check if vocal separation is enabled
    if not load_key("demucs"):
        return RAW_AUDIO_FILE

    try:
        print(f"[cyan]🎙️ Enhancing vocals with volume ratio: {vocals_ratio}[/cyan]")
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{_vocal_source()}" '
            f'-filter:a "volume={vocals_ratio}" '
            f'"{ENHANCED_VOCAL_PATH}"'
        )
        subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)

        return ENHANCED_VOCAL_PATH
    except subprocess.CalledProcessError as e:
        print(f"[red]Error enhancing vocals: {str(e)}[/red]")
        return VOCAL_AUDIO_FILE  # Fallback to original vocals if enhancement fails

def _vocal_source() -> str:
    """Prefer the Demucs WAV: its MP3 twin carries encoder priming (~25 ms late)."""
    wav = os.path.join(os.path.dirname(VOCAL_AUDIO_FILE), "vocal.wav")
    return wav if os.path.isfile(wav) else VOCAL_AUDIO_FILE

GUARD_MIN_WINDOW_SECONDS = 60.0


def _result_text(result: dict) -> str:
    return "".join(str(seg.get("text", "")) for seg in (result or {}).get("segments", []))


def guarded_transcribe(ts, audio: str, start: float, end: float, report: list, depth: int = 0) -> list:
    """Recognize [start, end]; a looping, sparse or wrong-script chunk is recognized again in
    smaller windows. A window that stays suspicious at the minimum size is kept (never dropped)
    and recorded as needs_review in ``report``."""
    from core.providers.asr_guards import assess
    from core.runtime_context import effective_source_language
    from core.all_whisper_methods.audio_preprocess import active_seconds, find_cut_point

    result = ts(audio, start, end)
    seconds = end - start
    try:
        language = effective_source_language("auto")
    except Exception:
        language = "auto"
    text = _result_text(result)
    speech = active_seconds(audio, start, end) if seconds >= 20 else None
    reasons = assess(text, seconds, language, speech)
    if not reasons:
        return [result]
    event = {"start": start, "end": end, "reasons": reasons, "depth": depth, "final": False}
    if seconds / 2 < GUARD_MIN_WINDOW_SECONDS or depth >= 4:
        event["final"] = True
        report.append(event)
        rprint(f"[yellow]⚠️ ASR chunk {start:.1f}-{end:.1f}s suspicious ({', '.join(reasons)}); kept, needs review[/yellow]")
        return [result]
    mid = (start + end) / 2
    cut = find_cut_point(audio, max(start, mid - 15), min(end, mid + 15)) or mid
    event["retry_at"] = cut
    report.append(event)
    rprint(f"[yellow]⚠️ ASR chunk {start:.1f}-{end:.1f}s suspicious ({', '.join(reasons)}); retrying in two windows[/yellow]")
    return (guarded_transcribe(ts, audio, start, cut, report, depth + 1)
            + guarded_transcribe(ts, audio, cut, end, report, depth + 1))


def transcribe():
    current_fp = source_fingerprint()

    if os.path.exists(CLEANED_CHUNKS_EXCEL_PATH):
        cached_fp = None
        if os.path.exists(ASR_FINGERPRINT_PATH):
            try:
                with open(ASR_FINGERPRINT_PATH, "r", encoding="utf-8") as f:
                    cached_fp = json.load(f)
            except Exception:
                cached_fp = None

        # If resume from valid checkpoint, allow matching on runtime & model & language
        fp_match = (cached_fp == current_fp)
        if not fp_match and cached_fp and isinstance(cached_fp, dict):
            # Same core ASR configuration and source length
            keys_to_match = ["model", "runtime", "language", "stable_ts_mlx"]
            if all(cached_fp.get(k) == current_fp.get(k) for k in keys_to_match):
                fp_match = True

        if fp_match:
            import pandas as pd
            validate_cached_speaker_rows(pd.read_excel(CLEANED_CHUNKS_EXCEL_PATH).to_dict("records"))
            rprint("[yellow]⚠️ Transcription results matching current model already exist, skipping transcription step.[/yellow]")
            prepare_audio_and_vocals()
            return
        else:
            raise RuntimeError(
                "ASR source/settings fingerprint changed or is missing. "
                "Use a fresh run directory; existing transcripts and deliveries were not altered."
            )

    # Prepare audio files (this is also needed for embedded subtitle workflow)
    whisper_audio = prepare_audio_and_vocals()

    # Only ASR engines with native word timestamps may feed subtitle alignment.
    runtime = load_key("whisper.runtime")
    if runtime not in MAIN_ASR_RUNTIMES:
        raise ValueError(
            f"ASR runtime {runtime!r} cannot feed subtitle alignment. "
            f"Choose one of: {', '.join(MAIN_ASR_RUNTIMES)}"
        )

    # Whisper already windows the decoder internally. External 25s hard cuts
    # lose clause/name context and force recognition to start mid-utterance.
    chunk_seconds = int(load_key("whisper.chunk_seconds", 0))
    if chunk_seconds > 0:
        segments = split_audio(whisper_audio, target_len=chunk_seconds)
    else:
        from core.all_whisper_methods.audio_preprocess import get_audio_duration
        segments = [(0.0, get_audio_duration(whisper_audio))]
    all_results = []
    if runtime == "local-aligned":
        from core.all_whisper_methods.local_aligned import transcribe_audio as ts
        whisper_audio = "output/audio/raw_master.wav"
        segments = split_audio(whisper_audio, target_len=int(load_key("source_asr.chunk_seconds", 180)))
    elif runtime == "qwen3-aligned":
        from core.all_whisper_methods.qwen3_aligned import transcribe_audio as ts
        # Recognition uses original audio; Demucs remains available for dubbing.
        whisper_audio = "output/audio/raw_master.wav"
        segments = split_audio(whisper_audio)
    elif runtime == "local":
        from core.all_whisper_methods.whisperX_local import transcribe_audio as ts
        rprint("[cyan]🎤 Transcribing audio with local WhisperX model...[/cyan]")
    elif runtime == "stable-ts":
        try:
            # Check if stable_whisper is installed
            import stable_whisper
            from core.all_whisper_methods.stable_ts_local import transcribe_audio as ts
            rprint("[cyan]🎤 Transcribing audio with local stable-ts model...[/cyan]")
            rprint(f"[cyan]📊 Total segments to process: {len(segments)}[/cyan]")
        except ImportError:
            rprint("[bold red]❌ Error: stable-whisper is not installed![/bold red]")
            rprint("[yellow]Please run 'python install_stable_ts.py' to install stable-ts and its dependencies.[/yellow]")
            rprint("[yellow]Alternatively, change whisper.runtime to 'local' (WhisperX) in config.yaml.[/yellow]")
            raise ImportError("stable-whisper is not installed. Please run 'python install_stable_ts.py' to install it.")
    else:
        raise ValueError(f"Unknown whisper runtime: {runtime}. Supported: 'local', 'stable-ts'")

    raw_results = []
    from copy import deepcopy
    guard_report = []
    for i, (start, end) in enumerate(segments):
        rprint(f"[cyan]📊 Processing segment {i+1}/{len(segments)}: {start:.2f}s to {end:.2f}s[/cyan]")
        for result in guarded_transcribe(ts, whisper_audio, start, end, guard_report):
            raw_results.append(deepcopy(result))
            # Write recognition before normalization, including when timing fails.
            with open("output/log/asr_raw_results.json", "w", encoding="utf-8") as handle:
                json.dump(raw_results, handle, ensure_ascii=False, indent=2)
            if runtime not in {"qwen3-aligned", "local-aligned"}:
                sanitize_word_timestamps(result)
            validate_word_timestamps(result, backend=runtime, allow_empty=True)
            all_results.append(result)

        # 如果使用 stable-ts，显示当前处理进度
        if runtime == "stable-ts":
            rprint(f"[green]✅ Completed segment {i+1}/{len(segments)} ({(i+1)/len(segments)*100:.1f}%)[/green]")

    with open("output/log/asr_guard_report.json", "w", encoding="utf-8") as handle:
        json.dump({"status": "needs_review" if any(g["final"] for g in guard_report) else "pass",
                   "events": guard_report}, handle, ensure_ascii=False, indent=2)

    with open("output/log/asr_raw_results.json", "w", encoding="utf-8") as handle:
        json.dump(raw_results, handle, ensure_ascii=False, indent=2)
    with open("output/log/asr_timing_review.json", "w", encoding="utf-8") as handle:
        json.dump([r.get("word_timestamp_sanitizer", {}) for r in all_results], handle, ensure_ascii=False, indent=2)

    # step6 Combine results
    combined_result = {'segments': [], "timing_source": (
        "qwen3_forced_alignment_80ms" if runtime in {"qwen3-aligned", "local-aligned"} else "native_asr")}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])
    validate_word_timestamps(combined_result, backend=runtime)

    # Keep native word timings from stable-ts/WhisperX and add only MOSS speaker IDs.
    def _robust_transcribe_range(start: float, end: float):
        if runtime in {"qwen3-aligned", "local-aligned"}:
            try:
                return ts(whisper_audio, start, end)
            except Exception:
                return {"segments": []}
        return transcribe_recovery_audio(ts, whisper_audio, start, end)

    diarization_audio = "output/audio/vocal.wav" if os.path.exists("output/audio/vocal.wav") else whisper_audio
    diarization = apply_speaker_diarization(
        combined_result,
        diarization_audio,
        transcribe_range=_robust_transcribe_range,
        source_backend=runtime,
    )
    if diarization.get("status") == "pass":
        rprint(
            f"[green]✅ MOSS speaker alignment: {diarization['speaker_count']} speakers, "
            f"{diarization['coverage']:.1%} word coverage, "
            f"{diarization['speaker_transitions']} turn boundaries[/green]"
        )

    # step7 Process df
    with open("output/log/asr_final_results.json", "w", encoding="utf-8") as handle:
        json.dump(combined_result, handle, ensure_ascii=False, indent=2)
    df = process_transcription(combined_result)
    save_results(df)
    reference_file = load_key("source_asr.reference_checks_file", "")
    if reference_file:
        from pathlib import Path
        from core.source_accuracy import evaluate_source_checkpoints
        from core.pipeline.artifact_manifest import hash_path
        reference = json.loads(Path(reference_file).read_text(encoding="utf-8"))
        assessment = evaluate_source_checkpoints(combined_result, reference)
        assessment.update(transcript_sha256=hash_path(CLEANED_CHUNKS_EXCEL_PATH),
                          original_audio_sha256=hash_path("output/audio/raw_master.wav"),
                          reference_file=str(reference_file), reference_sha256=hash_path(reference_file))
        Path("output/log/source_reference_assessment.json").write_text(json.dumps(assessment, ensure_ascii=False, indent=2))
        if assessment["status"] == "fail" and load_key("source_asr.reference_required", True):
            raise RuntimeError("Source ASR failed external reference checks; translation blocked. Inspect source_reference_assessment.json")
    os.makedirs(os.path.dirname(ASR_FINGERPRINT_PATH), exist_ok=True)
    with open(ASR_FINGERPRINT_PATH, "w", encoding="utf-8") as f:
        json.dump(current_fp, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    transcribe()
