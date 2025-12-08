"""Piper TTS - Fast local neural text-to-speech engine.

Piper supports 40+ languages. Voice models are automatically downloaded on first use.
Available voices: https://huggingface.co/rhasspy/piper-voices/tree/main

Example voices:
- en_US-lessac-medium (English US)
- zh_CN-huayan-medium (Chinese)
- id_ID-news_tts-medium (Indonesian)
- de_DE-thorsten-medium (German)
- fr_FR-siwis-medium (French)
- ja_JP-takumi-medium (Japanese)
- ko_KR-kss-medium (Korean)
"""

import subprocess
from pathlib import Path
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from core.config_utils import load_key

# Directory to store piper voice models
PIPER_MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "piper_voices"


def get_model_path(voice: str) -> Path:
    """Get the path to the model file for a voice."""
    return PIPER_MODELS_DIR / f"{voice}.onnx"


def ensure_voice_downloaded(voice: str) -> Path:
    """Ensure the voice model is downloaded. Returns the model path."""
    model_path = get_model_path(voice)
    
    if model_path.exists():
        return model_path
    
    # Create models directory
    PIPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"🔽 Downloading Piper voice: {voice}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "piper.download_voices", 
             "--download_dir", str(PIPER_MODELS_DIR), voice],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout for download
        )
        if result.returncode == 0 and model_path.exists():
            print(f"✅ Voice {voice} downloaded successfully")
            return model_path
        else:
            print(f"⚠️ Voice download warning: {result.stderr}")
            # Try to use existing model if any
            if model_path.exists():
                return model_path
            raise Exception(f"Failed to download voice: {voice}")
    except subprocess.TimeoutExpired:
        raise Exception(f"Voice download timed out after 5 minutes")
    except Exception as e:
        raise Exception(f"Voice download error: {e}")


def piper_tts(text: str, save_path: str):
    """Generate TTS audio using Piper.
    
    Args:
        text: Text to convert to speech
        save_path: Path to save the generated audio file
    """
    # Load settings from config
    piper_config = load_key("piper_tts", {})
    voice = piper_config.get("voice", "en_US-lessac-medium")
    
    # Ensure voice is downloaded and get model path
    model_path = ensure_voice_downloaded(voice)
    
    # Create output directory if needed
    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use piper CLI with full model path
    cmd = [
        sys.executable, "-m", "piper",
        "-m", str(model_path),
        "-f", str(speech_file_path),
        "--", text
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        print(f"Audio saved to {speech_file_path}")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Piper TTS failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise Exception(f"Piper TTS timed out after 60 seconds")


if __name__ == "__main__":
    # Test with a sample sentence
    piper_tts("Welcome to VideoLingo with Piper TTS!", "piper_test.wav")

