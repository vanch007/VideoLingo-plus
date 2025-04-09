import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch
import time
import subprocess
from typing import Dict
from rich import print as rprint
import librosa
import tempfile
from core.config_utils import load_key
from core.all_whisper_methods.audio_preprocess import save_language

MODEL_DIR = load_key("model_dir")

def check_device():
    """Check and return the available device"""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    rprint(f"🚀 Using device: {device}")

    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        rprint(f"[cyan]🎮 GPU memory:[/cyan] {gpu_mem:.2f} GB")
    elif device == "mps":
        rprint(f"[cyan]🍎 Using Apple Silicon acceleration[/cyan]")
    return device

def transcribe_audio(audio_file: str, start: float, end: float) -> Dict:
    """
    Transcribe audio segment using stable-ts

    Args:
        audio_file: Path to the audio file
        start: Start time of the segment in seconds
        end: End time of the segment in seconds

    Returns:
        Dict containing transcription results in WhisperX compatible format
    """
    try:
        # Import stable_whisper here to avoid loading it unnecessarily
        import stable_whisper

        device = check_device()
        WHISPER_LANGUAGE = load_key("whisper.language")
        WHISPER_MODEL = load_key("whisper.model")

        rprint(f"[green]▶️ Starting stable-ts for segment {start:.2f}s to {end:.2f}s...[/green]")

        # Create temp file with wav format for better compatibility
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        # Extract audio segment using ffmpeg
        ffmpeg_cmd = f'ffmpeg -y -i "{audio_file}" -ss {start} -t {end-start} -vn -ar 32000 -ac 1 "{temp_audio_path}"'
        subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)

        try:
            # Load audio segment with librosa
            audio_segment, sample_rate = librosa.load(temp_audio_path, sr=16000)
        finally:
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

        # Determine model path or name
        if WHISPER_LANGUAGE == 'zh':
            model_name = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
            local_model = os.path.join(MODEL_DIR, "Belle-whisper-large-v3-zh-punct-fasterwhisper")
        else:
            model_name = WHISPER_MODEL
            local_model = os.path.join(MODEL_DIR, model_name)

        if os.path.exists(local_model):
            rprint(f"[green]📥 Loading local WHISPER model:[/green] {local_model} ...")
            model_name = local_model
        else:
            rprint(f"[green]📥 Using WHISPER model from HuggingFace:[/green] {model_name} ...")

        # On Apple Silicon, always use MLX with large-v3-turbo for better performance
        if device == "mps":
            rprint("[green]📥 Using MLX Whisper large-v3-turbo model for Apple Silicon...[/green]")
            model = stable_whisper.load_mlx_whisper('large-v3-turbo')
        else:
            # Load the model with stable_whisper on other devices
            rprint("[green]📥 Loading stable-ts model...[/green]")
            model = stable_whisper.load_model(
                model_name,
                device=device,
                download_root=MODEL_DIR
            )

        rprint("[bold green]note: You will see Progress if working correctly[/bold green]")

        # Transcribe with stable-ts
        transcribe_options = {
            'word_timestamps': True,    # Enable word-level timestamps
            'vad': True,               # Use Voice Activity Detection for better timestamps
            'suppress_silence': True,   # Suppress silent parts
            'suppress_word_ts': True,   # Adjust word timestamps
            'verbose': True            # Show progress
        }

        # Add language parameter if not auto
        if WHISPER_LANGUAGE != 'auto':
            transcribe_options['language'] = WHISPER_LANGUAGE

        result = model.transcribe(audio_segment, **transcribe_options)

        # Free GPU resources
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        # Save language
        save_language(result.language)
        if result.language == 'zh' and WHISPER_LANGUAGE != 'zh':
            raise ValueError("Please specify the transcription language as zh and try again!")

        # Convert stable-ts result to WhisperX compatible format
        whisperx_result = {
            'segments': [],
            'language': result.language
        }

        # Process segments
        for segment in result.segments:
            whisperx_segment = {
                'start': segment.start + start,
                'end': segment.end + start,
                'text': ' '.join(segment.text.split()),
                'words': []
            }

            # Process words if available
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    whisperx_word = {
                        'start': word.start + start,
                        'end': word.end + start,
                        'word': word.word.strip()
                    }
                    whisperx_segment['words'].append(whisperx_word)

            whisperx_result['segments'].append(whisperx_segment)

        return whisperx_result

    except Exception as e:
        rprint(f"[red]stable-ts processing error:[/red] {e}")
        raise
