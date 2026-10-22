import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
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

def adjust_nonempty_gaps(result):
    # stable-ts adjust_gaps indexes segments[-1] on a silent short probe.
    # An empty recognition remains empty for coverage checking and recovery.
    return result.adjust_gaps(duration_threshold=0.75, one_section=False) if result.segments else result


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

        if not using_mlx_whisper:
            audio_segment, sample_rate = librosa.load(temp_audio_path, sr=16000)
            if os.path.exists(temp_audio_path):
                try: os.unlink(temp_audio_path)
                except Exception: pass
            rprint("[green]Transcribing with `transcribe_stable` for improved accuracy...[/green]")
            transcribe_options["regroup"] = True
            result = model.transcribe(audio_segment, **transcribe_options)

            # Post-refinement for even better timestamp accuracy
            rprint("[cyan]Applying additional refinement for precise timestamps...[/cyan]")
            # result = result.refine(audio_segment, precision=0.05, verbose=False)

            # Gap adjustment for optimal segment boundaries
            rprint("[cyan]Adjusting gaps for better segment boundaries...[/cyan]")
            result = adjust_nonempty_gaps(result)
        else:
            # Subprocess-isolated MLX Whisper runner: completely eliminates libomp/PyTorch C++ SIGSEGV conflicts on macOS
            import json
            from pathlib import Path
            
            worker_script = str(Path(__file__).resolve().parent.parent.parent / "scripts" / "mlx_whisper_isolated_worker.py")
            with tempfile.TemporaryDirectory(prefix="mlx_whisper_") as tmp_dir:
                tmp_json = str(Path(tmp_dir) / "result.json")
                cmd = [
                    sys.executable, worker_script,
                    "--audio", temp_audio_path,
                    "--model", mlx_model_name,
                    "--output", tmp_json
                ]
                if WHISPER_LANGUAGE and WHISPER_LANGUAGE != 'auto':
                    cmd.extend(["--language", WHISPER_LANGUAGE])
                    
                rprint(f"[cyan]🚀 Isolated MLX Whisper transcribing ({mlx_model_name})...[/cyan]")
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"MLX Whisper isolated worker failed: {proc.stderr[-1000:]}")
                    
                raw_mlx = json.loads(Path(tmp_json).read_text(encoding="utf-8"))
                
            detected_lang = raw_mlx.get("language", WHISPER_LANGUAGE if WHISPER_LANGUAGE != "auto" else "en")
            save_language(detected_lang)
            
            whisperx_result = {
                'segments': [],
                'language': detected_lang
            }
            for seg in raw_mlx.get('segments', []):
                whisperx_segment = {
                    'start': float(seg['start']) + start,
                    'end': float(seg['end']) + start,
                    'text': ' '.join(str(seg['text']).split()),
                    'words': []
                }
                for word in seg.get('words', []):
                    w_text = str(word['word']).strip()
                    # Filter out isolated punctuations that normalize to empty text and break utterance planning
                    from core.source_verification import normalize_transcript
                    if not normalize_transcript(w_text):
                        continue
                    whisperx_segment['words'].append({
                        'start': float(word['start']) + start,
                        'end': float(word['end']) + start,
                        'word': w_text
                    })
                whisperx_result['segments'].append(whisperx_segment)
                
            if os.path.exists(temp_audio_path):
                try: os.unlink(temp_audio_path)
                except Exception: pass
            return normalize_asr_result(whisperx_result, "stable-ts-mlx")

    except Exception as e:
        rprint(f"[red]stable-ts processing error:[/red] {e}")
        raise
