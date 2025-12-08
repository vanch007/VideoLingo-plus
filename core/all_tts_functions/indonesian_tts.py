"""Indonesian TTS using Coqui TTS with native Indonesian VITS model.

This module uses the Wikidepia/indonesian-tts VITS model specifically trained
for Indonesian speech synthesis. It requires g2p-id for grapheme-to-phoneme
conversion.

Model source: https://github.com/Wikidepia/indonesian-tts/releases/tag/v1.2

Requirements:
    pip install TTS
    pip install git+https://github.com/Wikidepia/g2p-id

Usage:
    from core.all_tts_functions.indonesian_tts import indonesian_tts_for_videolingo
    indonesian_tts_for_videolingo(text, save_as, number, task_df)
"""

import os
import sys
from pathlib import Path
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from core.config_utils import load_key

# Cache for the TTS model and g2p converter
_SYNTHESIZER = None
_G2P = None

VOICE_DIR = Path(__file__).parent / "voice"
AUDIO_REFERS_DIR = Path.cwd() / "output" / "audio" / "refers"
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "indonesian_tts"

# Available Indonesian speakers (main voices)
INDONESIAN_SPEAKERS = ["wibowo", "ardi", "gadis"]


def _get_g2p():
    """Load and cache the g2p-id converter."""
    global _G2P
    if _G2P is None:
        try:
            from g2p_id import G2P
            _G2P = G2P()
            print("✅ Loaded g2p-id for Indonesian phoneme conversion")
        except ImportError:
            raise ImportError(
                "g2p-id is not installed. Please install it with: "
                "pip install git+https://github.com/Wikidepia/g2p-id"
            )
    return _G2P


def _get_synthesizer():
    """Load and cache the Indonesian VITS TTS synthesizer."""
    global _SYNTHESIZER
    if _SYNTHESIZER is None:
        os.environ["COQUI_TOS_AGREED"] = "1"
        
        try:
            from TTS.utils.synthesizer import Synthesizer
        except ImportError:
            raise ImportError(
                "Coqui TTS is not installed. Please install it with: pip install TTS"
            )
        
        # Check model files exist
        model_path = MODEL_DIR / "checkpoint.pth"
        config_path = MODEL_DIR / "config.json"
        
        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"Indonesian TTS model files not found at {MODEL_DIR}. "
                "Please download from https://github.com/Wikidepia/indonesian-tts/releases/tag/v1.2"
            )
        
        print(f"🔊 Loading Indonesian VITS model...")
        
        _SYNTHESIZER = Synthesizer(
            tts_checkpoint=str(model_path),
            tts_config_path=str(config_path),
            use_cuda=False  # Use CPU for better compatibility
        )
        
        # Get available speakers
        speakers = list(_SYNTHESIZER.tts_model.speaker_manager.name_to_id.keys())
        print(f"✅ Indonesian VITS model loaded with {len(speakers)} speakers")
        print(f"   Main Indonesian voices: {[s for s in INDONESIAN_SPEAKERS if s in speakers]}")
    
    return _SYNTHESIZER


def text_to_phonemes(text: str) -> str:
    """Convert Indonesian text to IPA phonemes using g2p-id."""
    g2p = _get_g2p()
    phonemes = g2p(text)
    return phonemes


def get_available_speakers():
    """Get list of available speakers."""
    synthesizer = _get_synthesizer()
    return list(synthesizer.tts_model.speaker_manager.name_to_id.keys())


def indonesian_tts(
    text: str, 
    save_path: str, 
    speaker_name: str = "wibowo",
    use_phonemes: bool = True
):
    """Generate Indonesian TTS audio.
    
    Args:
        text: Text to synthesize (in Indonesian)
        save_path: Path to save the generated audio file
        speaker_name: Speaker name (e.g., 'wibowo', 'ardi', 'gadis')
        use_phonemes: Whether to convert text to phonemes first (recommended)
    """
    synthesizer = _get_synthesizer()
    
    # Create output directory if needed
    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to phonemes if requested (recommended for Indonesian)
    if use_phonemes:
        synthesis_text = text_to_phonemes(text)
        print(f"📝 Text: {text}")
        print(f"📝 Phonemes: {synthesis_text}")
    else:
        synthesis_text = text
    
    # Validate speaker
    available_speakers = synthesizer.tts_model.speaker_manager.name_to_id.keys()
    if speaker_name not in available_speakers:
        print(f"⚠️ Speaker '{speaker_name}' not found, using 'wibowo'")
        speaker_name = "wibowo"
    
    print(f"🎙️ Generating Indonesian TTS with speaker: {speaker_name}")
    
    # Generate audio
    wav = synthesizer.tts(synthesis_text, speaker_name=speaker_name)
    synthesizer.save_wav(wav, str(speech_file_path))
    
    print(f"✅ Audio saved to {speech_file_path}")


def indonesian_tts_for_videolingo(
    text: str, 
    save_as: str, 
    number: int, 
    task_df, 
    clone_mode: str = None,  # Kept for API compatibility, not used
    fixed_voice_name: str = None
):
    """Generate Indonesian TTS for VideoLingo workflow.
    
    Note: The Indonesian VITS model uses pre-trained speaker embeddings,
    not voice cloning from reference audio.
    
    Args:
        text: Text to synthesize (in Indonesian)
        save_as: Path to save the generated audio file
        number: Segment number (not used)
        task_df: DataFrame with task information (not used)
        clone_mode: Kept for API compatibility (not used)
        fixed_voice_name: Speaker name (e.g., 'wibowo', 'ardi', 'gadis'). 
                          Falls back to config if not provided.
    """
    # Use provided speaker or fallback to config
    if fixed_voice_name:
        speaker_name = fixed_voice_name
    else:
        speaker_name = load_key("indonesian_tts.speaker", "wibowo")
    
    print(f"🎙️ Using Indonesian TTS speaker: {speaker_name}")
    
    # Generate TTS with Indonesian VITS model
    indonesian_tts(
        text=text, 
        save_path=save_as, 
        speaker_name=speaker_name,
        use_phonemes=True
    )


if __name__ == "__main__":
    # Test basic functionality
    print("Testing Indonesian TTS with VITS model...")
    print("=" * 50)
    
    # Test g2p-id
    from g2p_id import G2P
    g2p = G2P()
    test_text = "Halo, selamat datang di VideoLingo."
    phonemes = g2p(test_text)
    print(f"Text: {test_text}")
    print(f"Phonemes: {phonemes}")
    print()
    
    # Generate test audio
    test_output = "test_indonesian_vits.wav"
    indonesian_tts(
        text="Halo, selamat datang di VideoLingo.",
        save_path=test_output,
        speaker_name="wibowo",
        use_phonemes=True
    )
    
    if os.path.exists(test_output):
        print(f"\n✅ Test completed! Output: {test_output}")
        print(f"   File size: {os.path.getsize(test_output)} bytes")
    else:
        print("❌ Test failed: output file not created")
