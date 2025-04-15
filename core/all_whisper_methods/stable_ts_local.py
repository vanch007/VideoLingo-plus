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

def generate_split_files(result: Dict, language: str, is_first_segment: bool = False) -> None:
    """
    此函数已被禁用，不再生成 sentence_splitbynlp.txt 文件
    现在将由 step3_1_spacy_split.py 负责生成该文件

    Args:
        result: stable-ts 转换为 WhisperX 格式的结果
        language: 检测到的语言代码
        is_first_segment: 是否是第一个音频段（已不再使用）
    """
    # 不执行任何操作，让 step3_1_spacy_split.py 来处理
    rprint(f"[cyan]ℹ️ 不生成 sentence_splitbynlp.txt，将由 step3_1_spacy_split.py 处理[/cyan]")
    return

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

        # On Apple Silicon, use MLX with appropriate model for better performance
        if device == "mps":
            # 强制将 MPS 设备视为 Apple Silicon
            # 因为 MPS 只在 Apple Silicon 上可用
            is_apple_silicon = True
            rprint(f"[cyan]检测到 MPS 设备，将其视为 Apple Silicon[/cyan]")

            # 使用用户选择的模型
            mlx_model_name = WHISPER_MODEL

            # 打印调试信息
            rprint(f"[cyan]用户选择的模型: {mlx_model_name}[/cyan]")
            rprint(f"[cyan]模型是否在 MLX_MODELS 中: {mlx_model_name in MLX_MODELS}[/cyan]")

            # 如果用户选择的模型在 MLX_MODELS 中，使用该模型
            if mlx_model_name in MLX_MODELS:
                rprint(f"[green]📥 Using MLX Whisper {mlx_model_name} model for Apple Silicon...[/green]")
                model = stable_whisper.load_mlx_whisper(mlx_model_name)
            # 否则默认使用 large-v3-turbo
            else:
                rprint(f"[yellow]警告: 模型 {mlx_model_name} 不在 MLX 支持列表中，使用 large-v3-turbo 代替[/yellow]")
                model = stable_whisper.load_mlx_whisper('large-v3-turbo')

            # 记录使用的是 MLX Whisper
            using_mlx_whisper = True
        else:
            # Load the model with stable_whisper on other devices
            rprint("[green]📥 Loading stable-ts model...[/green]")
            model = stable_whisper.load_model(
                model_name,
                device=device,
                download_root=MODEL_DIR
            )
            # 记录使用的不是 MLX Whisper
            using_mlx_whisper = False

        rprint("[bold green]note: You will see Progress if working correctly[/bold green]")

        # Transcribe with stable-ts
        # 基本参数，所有模型都支持
        transcribe_options = {
            'word_timestamps': True,    # Enable word-level timestamps
            'vad': True,               # Use Voice Activity Detection for better timestamps
            'vad_threshold': 0.5,      # Higher threshold for more accurate speech detection
            'verbose': True            # Show progress
        }

        # 只有非 MLX Whisper 模型支持的高级参数
        if not using_mlx_whisper:
            rprint("[green]添加高级参数以提高转录质量[/green]")
            transcribe_options.update({
                'suppress_silence': True,   # Suppress silent parts
                'suppress_word_ts': True,   # Adjust word timestamps
                'nonspeech_skip': 1.0,     # Skip non-speech sections longer than 1 second
                'max_instant_words': 0.3,  # Remove segments with too many instantaneous words
            })
        else:
            rprint("[yellow]MLX Whisper不支持某些高级参数，使用基本配置[/yellow]")

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

        # 不再生成 sentence_splitbynlp.txt 文件
        # 注释掉相关代码，让 step3_1_spacy_split.py 来处理
        # is_first_segment = start == 0 or start < 1.0
        # generate_split_files(whisperx_result, result.language, is_first_segment)

        # 只调用一次，记录日志
        if start == 0 or start < 1.0:  # 只在第一个段时显示信息
            rprint(f"[cyan]ℹ️ stable-ts 不再生成 sentence_splitbynlp.txt，将由 step3_1_spacy_split.py 处理[/cyan]")

        return whisperx_result

    except Exception as e:
        rprint(f"[red]stable-ts processing error:[/red] {e}")
        raise
