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
    TTS_METHOD = load_key("tts_method")
    get_tts_provider(TTS_METHOD)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            
            # Conditional imports based on TTS method to avoid import errors
            if TTS_METHOD == 'gpt_sovits':
                from core.all_tts_functions.gpt_sovits_tts import gpt_sovits_tts_for_videolingo
                gpt_sovits_tts_for_videolingo(text, save_as, number, task_df)
            elif TTS_METHOD == 'edge_tts':
                from core.all_tts_functions.edge_tts import edge_tts
                edge_tts(text, save_as)
            elif TTS_METHOD == 'piper_tts':
                from core.all_tts_functions.piper_tts import piper_tts
                piper_tts(text, save_as)
            elif TTS_METHOD == 'custom_tts':
                from core.all_tts_functions.custom_tts import custom_tts
                custom_tts(text, save_as, number, task_df, attempt)
            elif TTS_METHOD == 'index_tts2':
                from core.all_tts_functions.index_tts2 import custom_tts
                custom_tts(text, save_as, number, task_df, attempt)
            elif TTS_METHOD == 'sf_indextts2':
                from core.all_tts_functions.sf_indextts2 import indextts2_tts_for_videolingo
                try:
                    clone_mode = load_key("sf_indextts2.clone_mode")
                except KeyError:
                    clone_mode = "dynamic"
                
                fixed_voice_name = None
                if clone_mode == "fixed":
                    try:
                        fixed_voice_name = load_key("sf_indextts2.fixed_voice")
                    except KeyError:
                        pass

                indextts2_tts_for_videolingo(
                    text,
                    save_as,
                    number,
                    task_df,
                    clone_mode=clone_mode,
                    fixed_voice_name=fixed_voice_name,
                    target_duration=target_duration,
                )
            elif TTS_METHOD == 'indonesian_tts':
                from core.all_tts_functions.indonesian_tts import indonesian_tts_for_videolingo
                speaker = load_key("indonesian_tts.speaker", "wibowo")
                indonesian_tts_for_videolingo(text, save_as, number, task_df, fixed_voice_name=speaker)
            elif TTS_METHOD == 'voxcpm_tts':
                from core.all_tts_functions.voxcpm_tts import voxcpm_tts
                voxcpm_tts(text, save_as, number, task_df, attempt)
            elif TTS_METHOD == 'mlx_router' or TTS_METHOD in MLX_ROUTER_BACKENDS:
                from core.all_tts_functions.mlx_router import mlx_router_tts
                row_payload = dict(task_row or {})
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
            elif TTS_METHOD == 'cosyvoice3_tts':
                raise NotImplementedError("cosyvoice3_tts provider is registered but not implemented yet. Configure its local/API adapter before use.")
            elif TTS_METHOD == 'elevenlabs_tts':
                raise NotImplementedError("elevenlabs_tts provider is registered but not implemented yet. Add ELEVENLABS_API_KEY and adapter before use.")
            elif TTS_METHOD == 'openai_tts':
                raise NotImplementedError("openai_tts provider is registered but not implemented yet. Add OPENAI_API_KEY and adapter before use.")
            else:
                raise ValueError(f"Unknown TTS method: {TTS_METHOD}")
                
            # For custom_tts, skip the duration check as per user request.
            # For all other methods, validate the generated audio.
            if TTS_METHOD == 'custom_tts':
                break  # Assume success and exit the retry loop

            # Check generated audio duration for other TTS methods
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
