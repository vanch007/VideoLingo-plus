import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch
import time
import subprocess
from typing import Dict, List
from rich import print as rprint
import librosa
import tempfile
import platform
from core.config_utils import load_key, get_joiner
from core.all_whisper_methods.audio_preprocess import save_language
from core.asr_schema import normalize_asr_result
from huggingface_hub import snapshot_download

# 过滤torchaudio相关警告
warnings.filterwarnings("ignore", message=".*torchaudio.*backend.*")

MODEL_DIR = load_key("model_dir")

# MLX Whisper 模型映射
MLX_MODELS = {
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v1": "mlx-community/whisper-large-v1-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo"
}

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

        WHISPER_LANGUAGE = load_key("whisper.language")
        WHISPER_MODEL = load_key("whisper.model")

        # Determine if we should use MLX
        use_mlx = False
        is_apple_silicon = (platform.system() == "Darwin" and "arm" in platform.machine())
        if is_apple_silicon and load_key("whisper.stable_ts_mlx", default=True):
            use_mlx = True
            device = "mps"
            rprint(f"🚀 Using device: {device} (MLX enabled by user)")
        else:
            if is_apple_silicon:
                rprint("[yellow]MLX disabled by user. Using CPU instead. This will be slower but allows for more features.[/yellow]")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            rprint(f"🚀 Using device: {device}")
            if device == "cuda":
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                rprint(f"[cyan]🎮 GPU memory:[/cyan] {gpu_mem:.2f} GB")

        rprint(f"[green]▶️ Starting stable-ts for segment {start:.2f}s to {end:.2f}s...[/green]")

        # Create temp file with wav format for better compatibility
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        # Extract audio segment using ffmpeg
        ffmpeg_cmd = f'ffmpeg -y -i "{audio_file}" -ss {start} -t {end-start} -vn -ar 16000 -ac 1 "{temp_audio_path}"'
        subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)

        try:
            # Load audio segment with librosa
            audio_segment, sample_rate = librosa.load(temp_audio_path, sr=16000)
        finally:
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

        model_name = WHISPER_MODEL

        # On Apple Silicon, use MLX with appropriate model for better performance
        if use_mlx:
            mlx_model_name = WHISPER_MODEL
            rprint(f"[cyan]User selected model: {mlx_model_name}[/cyan]")
            if mlx_model_name in MLX_MODELS:
                rprint(f"[green]📥 Using MLX Whisper {mlx_model_name} model for Apple Silicon...[/green]")
                model = stable_whisper.load_mlx_whisper(mlx_model_name)
            else:
                rprint(f"[yellow]Warning: Model {mlx_model_name} not in MLX support list, defaulting to large-v3-turbo[/yellow]")
                model = stable_whisper.load_mlx_whisper('large-v3-turbo')
            using_mlx_whisper = True
        else:
            # For non-MLX devices, handle standard vs. faster-whisper models
            if model_name == "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper":
                # This is our special faster-whisper model. Ensure it's downloaded.
                local_model_path = os.path.abspath(os.path.join(MODEL_DIR, model_name))
                if not os.path.exists(local_model_path):
                    rprint(f"[yellow]Local Chinese model not found at {local_model_path}.[/yellow]")
                    rprint(f"[cyan]Downloading from Hugging Face: {model_name}...[/cyan]")
                    try:
                        snapshot_download(repo_id=model_name, local_dir=local_model_path, local_dir_use_symlinks=False)
                        rprint(f"[green]✓ Successfully downloaded model to {local_model_path}[/green]")
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to download model. Please manually download from 'https://huggingface.co/{model_name}' "
                            f"and place it in {os.path.abspath(MODEL_DIR)}. Error: {e}"
                        )

                # Bypass `load_model` and instantiate directly for robustness
                from stable_whisper.whisper_word_level import FasterWhisper
                rprint("[green]📥 Directly instantiating faster-whisper model for Chinese...[/green]")
                model = FasterWhisper(local_model_path, device=device)

            else:
                # For other standard models, use the regular load_model function
                rprint("[green]📥 Loading stable-ts model...[/green]")
                download_root_path = MODEL_DIR if not os.path.isabs(model_name) else None
                model = stable_whisper.load_model(
                    model_name,
                    device=device,
                    download_root=download_root_path,
                    dq=False
                )
            using_mlx_whisper = False

        rprint("[bold green]note: You will see Progress if working correctly[/bold green]")

        # Transcribe with stable-ts
        transcribe_options = {
            'word_timestamps': True,
            'vad': True,
            'vad_threshold': load_key("whisper.vad_threshold", 0.3),  # Load from config
            'min_word_dur': load_key("whisper.min_word_dur", 0.1),    # Load from config
            'condition_on_previous_text': False,
            'regroup': False,      # Disable default regrouping
            'suppress_silence': True, # Enable silence suppression
            'suppress_word_ts': True, # Enable word timestamp suppression based on silence
            'use_word_position': True,
            'verbose': True,
        }

        if not using_mlx_whisper:
            rprint("[green]Applying advanced options for non-MLX models[/green]")
            # dynamic_heads optimization
            # transcribe_options['dynamic_heads'] = True
            # transcribe_options['aligner'] = 'new'
            # transcribe_options['resume'] = True
        else:
            rprint("[yellow]MLX Whisper does not support some advanced features.[/yellow]")
            # For MLX, we might want to enable regrouping if the manual logic is removed/changed
            # transcribe_options['regroup'] = True

            # Also ensure MLX uses the configured VAD threshold if possible (MLX support varies)
            # transcribe_options['vad_threshold'] = load_key("whisper.vad_threshold", 0.3)

        if WHISPER_LANGUAGE != 'auto':
            transcribe_options['language'] = WHISPER_LANGUAGE

        # For non-MLX, use the high-level `transcribe_stable` function which handles refine/regroup internally.
        if not using_mlx_whisper:
            rprint("[green]Transcribing with `transcribe_stable` for improved accuracy...[/green]")
            transcribe_options["regroup"] = True
            result = model.transcribe(audio_segment, **transcribe_options)

            # Post-refinement for even better timestamp accuracy
            rprint("[cyan]Applying additional refinement for precise timestamps...[/cyan]")
            # result = result.refine(audio_segment, precision=0.05, verbose=False)

            # Gap adjustment for optimal segment boundaries
            rprint("[cyan]Adjusting gaps for better segment boundaries...[/cyan]")
            result = result.adjust_gaps(duration_threshold=0.75, one_section=False)
        else:
            # MLX backend does not support `transcribe_stable`, use the basic `transcribe`.
            result = model.transcribe(audio_segment, **transcribe_options)
            # Preserve native word timings. The former manual sentence split
            # replaced them with segment estimates and broke Step 6 alignment.
            rprint("[cyan]Adjusting gaps for better segment boundaries...[/cyan]")
            result = result.adjust_gaps(duration_threshold=0.75, one_section=False)


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



        return normalize_asr_result(whisperx_result, "stable-ts-mlx" if using_mlx_whisper else "stable-ts")

    except Exception as e:
        rprint(f"[red]stable-ts processing error:[/red] {e}")
        raise
