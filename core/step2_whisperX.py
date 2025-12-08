import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich import print as rprint
import subprocess

from core.config_utils import load_key
from core.all_whisper_methods.demucs_vl import demucs_main, RAW_AUDIO_FILE, VOCAL_AUDIO_FILE
from core.all_whisper_methods.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results, compress_audio, CLEANED_CHUNKS_EXCEL_PATH
from core.step1_ytdlp import find_video_files

WHISPER_FILE = "output/audio/for_whisper.mp3"
ENHANCED_VOCAL_PATH = "output/audio/enhanced_vocals.mp3"

def prepare_audio_and_vocals():
    """Prepare audio files and vocals - this is needed for both transcription and embedded subtitle workflows"""
    # step0 Convert video to audio
    video_file = find_video_files()
    convert_video_to_audio(video_file)

    # step1 Demucs vocal separation:
    if load_key("demucs"):
        demucs_main()

    # step2 Enhance vocals if needed
    choose_audio = enhance_vocals() if load_key("demucs") else RAW_AUDIO_FILE
    
    # step3 Compress audio for whisper
    whisper_audio = compress_audio(choose_audio, WHISPER_FILE)
    
    # 确保输出目录存在
    os.makedirs('output/log', exist_ok=True)
    
    return whisper_audio

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
    if not load_key("demucs"):
        return RAW_AUDIO_FILE

    try:
        print(f"[cyan]🎙️ Enhancing vocals with volume ratio: {vocals_ratio}[/cyan]")
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{VOCAL_AUDIO_FILE}" '
            f'-filter:a "volume={vocals_ratio}" '
            f'"{ENHANCED_VOCAL_PATH}"'
        )
        subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)

        return ENHANCED_VOCAL_PATH
    except subprocess.CalledProcessError as e:
        print(f"[red]Error enhancing vocals: {str(e)}[/red]")
        return VOCAL_AUDIO_FILE  # Fallback to original vocals if enhancement fails

def transcribe():
    if os.path.exists(CLEANED_CHUNKS_EXCEL_PATH):
        rprint("[yellow]⚠️ Transcription results already exist, skipping transcription step.[/yellow]")
        # 但仍然需要确保音频文件存在
        prepare_audio_and_vocals()
        return

    # Prepare audio files (this is also needed for embedded subtitle workflow)
    whisper_audio = prepare_audio_and_vocals()

    # step4 Extract audio
    segments = split_audio(whisper_audio)

    # step5 Transcribe audio
    all_results = []
    runtime = load_key("whisper.runtime")
    if runtime == "local":
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
            rprint("[yellow]Alternatively, you can change the whisper.runtime to 'local' or 'cloud' in config.yaml.[/yellow]")
            raise ImportError("stable-whisper is not installed. Please run 'python install_stable_ts.py' to install it.")
    else:
        raise ValueError(f"Unknown whisper runtime: {runtime}. Supported: 'local', 'stable-ts'")

    for i, (start, end) in enumerate(segments):
        rprint(f"[cyan]📊 Processing segment {i+1}/{len(segments)}: {start:.2f}s to {end:.2f}s[/cyan]")
        result = ts(whisper_audio, start, end)
        all_results.append(result)

        # 如果使用 stable-ts，显示当前处理进度
        if runtime == "stable-ts":
            rprint(f"[green]✅ Completed segment {i+1}/{len(segments)} ({(i+1)/len(segments)*100:.1f}%)[/green]")

    # step6 Combine results
    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])

    # step7 Process df
    df = process_transcription(combined_result)
    save_results(df)

if __name__ == "__main__":
    transcribe()