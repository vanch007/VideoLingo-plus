import os
import re
import sys
import time

from rich import print as rprint
from pydub import AudioSegment

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from core.config_utils import load_key
from core.all_whisper_methods.audio_preprocess import get_audio_duration
from core.all_tts_functions.tts_registry import MLX_ROUTER_BACKENDS, get_tts_provider

def clean_text_for_tts(text):
    """Normalize text while preserving punctuation that changes spoken meaning."""
    text = str(text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    return text.strip()

def tts_main(text, save_as, number, task_df, task_row=None, line_index=0, target_duration=None):
    text = clean_text_for_tts(text)
    # Check if text is empty or single character, single character voiceovers are prone to bugs
    cleaned_text = re.sub(r'[^\w\s]', '', text).strip()
    if not cleaned_text or len(cleaned_text) <= 1:
        silence = AudioSegment.silent(duration=100)  # 100ms = 0.1s
        silence.export(save_as, format="wav")
        rprint(f"Created silent audio for empty/single-char text: {save_as}")
        return
    
    # Skip if file exists
    if os.path.exists(save_as):
        return
    
    print(f"Generating <{text}...>")
    task_row = task_row or {}
    row_tts_method = task_row.get("tts_method")
    row_tts_method = str(row_tts_method).strip() if row_tts_method is not None else ""
    if row_tts_method.lower() in {"", "nan", "<na>", "none"}:
        row_tts_method = ""
    TTS_METHOD = row_tts_method or load_key("tts_method")
    get_tts_provider(TTS_METHOD)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            
            if TTS_METHOD == 'mlx_router' or TTS_METHOD in MLX_ROUTER_BACKENDS:
                from core.all_tts_functions.mlx_router import mlx_router_tts
                row_payload = dict(task_row)
                if TTS_METHOD in MLX_ROUTER_BACKENDS:
                    row_payload['tts_backend'] = MLX_ROUTER_BACKENDS[TTS_METHOD]
                mlx_router_tts(
                    text,
                    save_as,
                    number,
                    task_df,
                    task_row=row_payload,
                    target_duration=target_duration,
                )
            else:
                raise ValueError(f"Unknown TTS method: {TTS_METHOD}")
                
            duration = get_audio_duration(save_as)
            if duration > 0:
                break
            else:
                if os.path.exists(save_as):
                    os.remove(save_as)
                if attempt == max_retries - 1:
                    print(f"Warning: Generated audio duration is 0 for text: {text}")
                    silence = AudioSegment.silent(duration=100)  # 100ms silence
                    silence.export(save_as, format="wav")
                    return
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(3)
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to generate audio after {max_retries} attempts: {str(e)}")
            print(f"Attempt {attempt + 1} failed, retrying in 5s...")
            time.sleep(5)
