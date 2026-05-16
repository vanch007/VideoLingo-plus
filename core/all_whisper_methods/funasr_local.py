import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

# Fix SSL certificate issues on macOS
import ssl
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Disable SSL verification if still having issues
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import torch
import subprocess
from typing import Dict
from rich import print as rprint
import tempfile
from core.config_utils import load_key
from core.all_whisper_methods.audio_preprocess import save_language
from core.asr_schema import normalize_asr_result

MODEL_DIR = load_key("model_dir")

def _download_hf_model(repo_id: str) -> str:
    """
    Download a model from HuggingFace Hub and return local path.
    Uses huggingface_hub.snapshot_download which caches models locally.
    
    Args:
        repo_id: HuggingFace repo ID, e.g. 'funasr/paraformer-zh'
    
    Returns:
        Local path to the downloaded model
    """
    try:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(repo_id=repo_id)
        return local_path
    except Exception as e:
        rprint(f"[yellow]⚠️ Failed to download {repo_id}: {e}[/yellow]")
        raise

def transcribe_audio(audio_file: str, start: float, end: float) -> Dict:
    """
    Transcribe audio segment using FunASR
    
    Args:
        audio_file: Path to the audio file
        start: Start time of the segment in seconds
        end: End time of the segment in seconds
    
    Returns:
        Dict containing transcription results in WhisperX compatible format
    """
    try:
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
    except ImportError:
        rprint("[bold red]❌ Error: funasr is not installed![/bold red]")
        rprint("[yellow]Please run 'pip install funasr' to install FunASR.[/yellow]")
        raise ImportError("funasr is not installed. Please run 'pip install funasr' to install it.")
    
    # Load configuration
    WHISPER_LANGUAGE = load_key("whisper.language")
    FUNASR_MODEL = load_key("funasr.model", default="sensevoice")
    ENABLE_SPK = load_key("funasr.enable_spk", default=True)
    MAX_SEGMENT_TIME = load_key("funasr.max_single_segment_time", default=30000)
    
    # Determine device - support CUDA, MPS (Apple Silicon), and CPU
    import platform
    if torch.cuda.is_available():
        device = "cuda:0"
        rprint(f"[cyan]🚀 Using device: CUDA[/cyan]")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        rprint(f"[cyan]🎮 GPU memory:[/cyan] {gpu_mem:.2f} GB")
    elif platform.system() == "Darwin" and "arm" in platform.machine():
        # Apple Silicon Mac - use MPS
        device = "mps"
        rprint(f"[cyan]🚀 Using device: MPS (Apple Silicon)[/cyan]")
    else:
        device = "cpu"
        rprint(f"[cyan]🚀 Using device: CPU[/cyan]")
    
    rprint(f"[green]▶️ Starting FunASR for segment {start:.2f}s to {end:.2f}s...[/green]")
    
    # Create temp file with wav format
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
    
    try:
        # Extract audio segment using ffmpeg
        ffmpeg_cmd = f'ffmpeg -y -i "{audio_file}" -ss {start} -t {end-start} -vn -ar 16000 -ac 1 "{temp_audio_path}"'
        subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)
        
        # Configure model based on selection
        if FUNASR_MODEL == "sensevoice":
            model_name = "FunAudioLLM/SenseVoiceSmall"  # HuggingFace model name
            rprint(f"[green]📥 Loading FunASR SenseVoice model from HuggingFace: {model_name}...[/green]")
            
            # SenseVoice has built-in VAD, no need for external VAD model
            model_kwargs = {
                "model": model_name,
                "hub": "hf",  # Use HuggingFace hub
                "device": device,
                "disable_update": True,  # Disable version check to avoid SSL issues
            }
            
            # SenseVoice doesn't support spk_model directly, but we can add it for future
            model = AutoModel(**model_kwargs)
            
            # Determine language parameter
            language = "auto" if WHISPER_LANGUAGE == "auto" else WHISPER_LANGUAGE
            
            rprint("[cyan]🎤 Transcribing with SenseVoice...[/cyan]")
            res = model.generate(
                input=temp_audio_path,
                cache={},
                language=language,
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
            )
            
            # Process SenseVoice output with segment duration for timestamp calculation
            result = _process_sensevoice_result(res, start, end - start)
            
            if ENABLE_SPK:
                rprint("[yellow]⚠️ Speaker diarization not yet implemented for SenseVoice model.[/yellow]")
                rprint("[yellow]   Use Paraformer model or WhisperX for speaker identification.[/yellow]")

        elif FUNASR_MODEL == "nano":
            model_name = "FunAudioLLM/Fun-ASR-Nano-2512"  # HuggingFace model ID
            rprint(f"[green]📥 Loading FunASR Nano model from HuggingFace: {model_name}...[/green]")
            
            # Get the local path of the downloaded model using huggingface_hub
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(repo_id=model_name)
            
            rprint(f"[cyan]  📦 Model path: {model_path}[/cyan]")
            
            # Add model path to sys.path to allow importing model.py
            if model_path not in sys.path:
                sys.path.insert(0, model_path)
            
            # Import FunASRNano directly from model.py (as per official docs)
            from model import FunASRNano
            
            # MPS doesn't support mixed precision well, use fp32 for stability
            # CUDA can use fp16 for acceleration
            if device.startswith("cuda"):
                model_dtype = "fp16"  # Use float16 for CUDA acceleration
                rprint("[cyan]  🚀 Using float16 for CUDA acceleration[/cyan]")
            else:
                model_dtype = "fp32"  # Use float32 for MPS/CPU (MPS doesn't support mixed precision)
                if device == "mps":
                    rprint("[cyan]  📝 Using float32 for MPS (mixed precision not supported)[/cyan]")
            
            # Use direct inference with FunASRNano.from_pretrained
            nano_model, kwargs = FunASRNano.from_pretrained(
                model=model_path, 
                device=device,
                llm_dtype=model_dtype,  # Pass dtype to model
            )
            nano_model.eval()
            
            rprint("[cyan]🎤 Transcribing with Fun-ASR-Nano...[/cyan]")
            res = nano_model.inference(data_in=[temp_audio_path], **kwargs)
            
            # Process Nano output - res[0][0] contains {'text': '...'}
            text = res[0][0]["text"] if res and res[0] else ""
            
            # Format result to match expected structure
            result = {'segments': [], 'language': 'auto'}
            if text:
                # Distribute timestamps evenly across characters/words
                is_cjk = _is_cjk_text(text)
                words_list = list(text.replace(' ', '')) if is_cjk else text.split()
                segment_duration = end - start
                time_per_word = segment_duration / len(words_list) if words_list else 0.1
                
                words = []
                current_time = start
                for word in words_list:
                    if word.strip():
                        words.append({
                            'word': word.strip(),
                            'start': current_time,
                            'end': current_time + time_per_word
                        })
                        current_time += time_per_word
                
                result['segments'].append({
                    'start': start,
                    'end': end,
                    'text': text,
                    'words': words
                })
            
            # Clean up
            del nano_model
            
            if ENABLE_SPK:
                rprint("[yellow]⚠️ Speaker diarization not yet implemented for Fun-ASR-Nano model.[/yellow]")
                rprint("[yellow]   Use Paraformer model or WhisperX for speaker identification.[/yellow]")
            
        else:  # paraformer
            # Use FunASR combined ASR + VAD + SPK pipeline
            # Download models from HuggingFace to local paths to bypass hub propagation issue
            rprint("[green]📥 Setting up FunASR Paraformer + VAD + Speaker pipeline...[/green]")
            
            # Paraformer uses float64 which MPS doesn't support - fallback to CPU
            paraformer_device = device
            if device == "mps":
                rprint("[yellow]⚠️ Paraformer uses float64, falling back to CPU (MPS not supported).[/yellow]")
                rprint("[yellow]   Tip: Use SenseVoice model for MPS acceleration.[/yellow]")
                paraformer_device = "cpu"
            
            try:
                # Use the bundled ASR+VAD+PUNC model from ModelScope that supports timestamps + speaker diarization
                # This model is only available on ModelScope, not HuggingFace
                bundled_model = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
                rprint(f"[cyan]  📦 Loading bundled ASR+VAD+PUNC model from ModelScope...[/cyan]")
                rprint(f"[cyan]  📦 Model ID: {bundled_model}[/cyan]")
                
                # Setup model with ModelScope (default hub)
                model_kwargs = {
                    "model": bundled_model,
                    "vad_kwargs": {"max_single_segment_time": MAX_SEGMENT_TIME},
                    "device": paraformer_device,
                    "disable_update": True,
                }
                
                # Add speaker model if enabled (using ModelScope ID)
                if ENABLE_SPK:
                    spk_model = "iic/speech_campplus_sv_zh-cn_16k-common"
                    model_kwargs["spk_model"] = spk_model
                    rprint(f"[cyan]👥 Speaker diarization enabled with CAM++ ({spk_model})[/cyan]")
                
                model = AutoModel(**model_kwargs)
                rprint("[green]✅ Model loaded from ModelScope![/green]")
                
                rprint("[cyan]🎤 Transcribing with Paraformer + VAD + SPK pipeline...[/cyan]")
                res = model.generate(
                    input=temp_audio_path,
                    batch_size_s=300,
                )
                
                # Process output with speaker info
                result = _process_paraformer_with_spk(res, start, ENABLE_SPK, end - start)
                
            except Exception as e:
                rprint(f"[yellow]⚠️ Combined pipeline failed: {e}[/yellow]")
                rprint("[yellow]⚠️ Falling back to ASR-only mode...[/yellow]")
                
                # Fallback to simple ASR without VAD/SPK
                asr_repo = "funasr/paraformer-zh" if WHISPER_LANGUAGE == "zh" else "funasr/paraformer"
                model = AutoModel(model=asr_repo, hub="hf", device=paraformer_device, disable_update=True)
                
                res = model.generate(input=temp_audio_path, batch_size_s=300, sentence_timestamp=True)
                result = _process_paraformer_result(res, start, False, end - start)
        
        # Detect and save language
        detected_lang = _detect_language(result)
        save_language(detected_lang)
        result['language'] = detected_lang
        
        rprint(f"[green]✅ FunASR transcription complete! Detected language: {detected_lang}[/green]")
        
        # Free resources
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return normalize_asr_result(result, f"funasr:{FUNASR_MODEL}")
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)


def _process_sensevoice_result(res: list, time_offset: float, segment_duration: float = 1.0) -> Dict:
    """
    Process SenseVoice output to WhisperX compatible format
    
    SenseVoice output format:
    [{'key': 'audio_path', 'text': '<|zh|><|ANGRY|><|Speech|><|withitn|>transcribed text'}]
    
    Args:
        res: FunASR result list
        time_offset: Start time of the segment
        segment_duration: Duration of the audio segment in seconds
    """
    result = {'segments': [], 'language': 'auto'}
    
    if not res or len(res) == 0:
        return result
    
    for item in res:
        if 'text' not in item:
            continue
        
        raw_text = item.get('text', '')
        
        # SenseVoice returns rich text with emotion/language tags, clean it
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        text = rich_transcription_postprocess(raw_text)
        
        if not text or not text.strip():
            continue
        
        text = text.strip()
        
        # Detect if it's CJK text for proper word splitting
        is_cjk = _is_cjk_text(text)
        
        # Split text into words/characters
        if is_cjk:
            # For CJK, split into individual characters
            words_list = list(text.replace(' ', ''))
        else:
            # For other languages, split by whitespace
            words_list = text.split()
        
        if not words_list:
            continue
        
        # Calculate time per word based on segment duration
        num_words = len(words_list)
        time_per_word = segment_duration / num_words if num_words > 0 else 0.1
        
        # Create word entries with distributed timestamps
        words = []
        current_time = time_offset
        for word in words_list:
            word_clean = word.strip()
            if word_clean:
                words.append({
                    'word': word_clean,
                    'start': current_time,
                    'end': current_time + time_per_word
                })
                current_time += time_per_word
        
        if words:
            # Create segment with the full text and words
            segment = {
                'start': time_offset,
                'end': time_offset + segment_duration,
                'text': text,
                'words': words
            }
            result['segments'].append(segment)
    
    return result


def _process_paraformer_result(res: list, time_offset: float, has_speaker: bool, segment_duration: float = 1.0) -> Dict:
    """
    Process Paraformer output to WhisperX compatible format
    
    Paraformer output format:
    [{'key': 'audio_path', 'text': 'transcribed text'}]
    
    Note: In current FunASR/HuggingFace version, Paraformer returns only text without timestamps.
    We distribute timestamps evenly across characters/words similar to SenseVoice.
    
    Args:
        res: FunASR result list
        time_offset: Start time of the segment
        has_speaker: Whether speaker info is expected (not used - handled separately)
        segment_duration: Duration of the audio segment in seconds
    """
    result = {'segments': [], 'language': 'auto'}
    
    if not res or len(res) == 0:
        return result
    
    for item in res:
        if 'text' not in item:
            continue
        
        text = item.get('text', '').strip()
        if not text:
            continue
        
        # Detect if it's CJK text for proper word splitting
        is_cjk = _is_cjk_text(text)
        
        # Split text into words/characters
        if is_cjk:
            words_list = list(text.replace(' ', ''))
        else:
            words_list = text.split()
        
        if not words_list:
            continue
        
        # Check if we have native timestamps
        if 'timestamp' in item and item['timestamp']:
            # Use native timestamps
            timestamps = item['timestamp']
            words = []
            for i, ts in enumerate(timestamps):
                if i < len(words_list) and len(ts) >= 2:
                    words.append({
                        'start': ts[0] / 1000.0 + time_offset,
                        'end': ts[1] / 1000.0 + time_offset,
                        'word': words_list[i]
                    })
            
            if words:
                segment = {
                    'start': words[0]['start'],
                    'end': words[-1]['end'],
                    'text': text,
                    'words': words
                }
                result['segments'].append(segment)
        else:
            # No native timestamps - distribute evenly based on segment duration
            num_words = len(words_list)
            time_per_word = segment_duration / num_words if num_words > 0 else 0.1
            
            words = []
            current_time = time_offset
            for word in words_list:
                word_clean = word.strip()
                if word_clean:
                    words.append({
                        'word': word_clean,
                        'start': current_time,
                        'end': current_time + time_per_word
                    })
                    current_time += time_per_word
            
            if words:
                segment = {
                    'start': time_offset,
                    'end': time_offset + segment_duration,
                    'text': text,
                    'words': words
                }
                result['segments'].append(segment)
    
    return result


def _process_paraformer_with_spk(res: list, time_offset: float, has_speaker: bool, segment_duration: float) -> Dict:
    """
    Process Paraformer + VAD + SPK combined pipeline output to WhisperX compatible format
    
    Combined pipeline output format (with spk_model enabled):
    [{'key': 'audio_path', 'text': 'full text', 'sentence_info': [
        {'text': 'sentence', 'start': ms, 'end': ms, 'spk': 'SPEAKER_XX'},
        ...
    ]}]
    
    Args:
        res: FunASR result list from combined pipeline
        time_offset: Start time of the audio segment
        has_speaker: Whether speaker model was enabled
        segment_duration: Duration of the audio segment
    """
    result = {'segments': [], 'language': 'auto'}
    
    if not res or len(res) == 0:
        return result
    
    for item in res:
        # Check for sentence_info (from combined pipeline with VAD)
        if 'sentence_info' in item and item['sentence_info']:
            for sent in item['sentence_info']:
                text = sent.get('text', '').strip()
                if not text:
                    continue
                
                # Get timing from sentence info (in milliseconds)
                start_ms = sent.get('start', 0)
                end_ms = sent.get('end', start_ms + 1000)
                start_time = start_ms / 1000.0 + time_offset
                end_time = end_ms / 1000.0 + time_offset
                sentence_duration = (end_ms - start_ms) / 1000.0
                
                # Get speaker if available
                speaker = sent.get('spk', None)
                
                # Split text into words/characters
                is_cjk = _is_cjk_text(text)
                if is_cjk:
                    words_list = list(text.replace(' ', ''))
                else:
                    words_list = text.split()
                
                if not words_list:
                    continue
                
                # Distribute timestamps across words
                num_words = len(words_list)
                time_per_word = sentence_duration / num_words if num_words > 0 else 0.1
                
                words = []
                current_time = start_time
                for word in words_list:
                    word_clean = word.strip()
                    if word_clean:
                        word_entry = {
                            'word': word_clean,
                            'start': current_time,
                            'end': current_time + time_per_word
                        }
                        if has_speaker and speaker:
                            word_entry['speaker'] = speaker
                        words.append(word_entry)
                        current_time += time_per_word
                
                if words:
                    segment = {
                        'start': start_time,
                        'end': end_time,
                        'text': text,
                        'words': words
                    }
                    if has_speaker and speaker:
                        segment['speaker'] = speaker
                    result['segments'].append(segment)
        
        elif 'text' in item:
            # Handle output with 'text' and 'timestamp' array (bundled model output)
            text = item.get('text', '').strip()
            if text:
                # Text is space-separated, each word corresponds to a timestamp
                words_list = text.split()
                timestamps = item.get('timestamp', [])
                
                # Create word entries with precise timestamps
                words = []
                for i, word in enumerate(words_list):
                    word_clean = word.strip()
                    if word_clean:
                        if i < len(timestamps) and len(timestamps[i]) >= 2:
                            # Use precise timestamps from array (in milliseconds)
                            start_time = timestamps[i][0] / 1000.0 + time_offset
                            end_time = timestamps[i][1] / 1000.0 + time_offset
                        else:
                            # Fallback: estimate based on position
                            start_time = time_offset + (i * segment_duration / len(words_list))
                            end_time = start_time + (segment_duration / len(words_list))
                        
                        words.append({
                            'word': word_clean,
                            'start': start_time,
                            'end': end_time
                        })
                
                if words:
                    # Create single segment with all words
                    segment = {
                        'start': words[0]['start'] if words else time_offset,
                        'end': words[-1]['end'] if words else time_offset + segment_duration,
                        'text': text.replace(' ', ''),  # Remove spaces for display
                        'words': words
                    }
                    result['segments'].append(segment)
    
    return result


def _is_cjk_text(text: str) -> bool:
    """Check if text contains CJK characters"""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # Chinese
            return True
        if '\u3040' <= char <= '\u309f':  # Hiragana
            return True
        if '\u30a0' <= char <= '\u30ff':  # Katakana
            return True
        if '\uac00' <= char <= '\ud7af':  # Korean
            return True
    return False


def _detect_language(result: Dict) -> str:
    """Detect language from transcription result"""
    # Try to detect from text content
    all_text = ' '.join([seg.get('text', '') for seg in result.get('segments', [])])
    
    if not all_text:
        return 'en'
    
    # Simple heuristic: check for CJK characters
    cjk_count = sum(1 for c in all_text if _is_cjk_text(c))
    total_chars = len(all_text.replace(' ', ''))
    
    if total_chars > 0 and cjk_count / total_chars > 0.3:
        return 'zh'
    
    return 'en'


def _apply_speaker_diarization(result: Dict, audio_path: str, device: str, time_offset: float) -> Dict:
    """
    Apply speaker diarization to transcription result using FunASR's CAM++ model
    
    Args:
        result: Transcription result with segments
        audio_path: Path to the audio file
        device: Device to run inference on (cuda, mps, cpu)
        time_offset: Time offset for the audio segment
    
    Returns:
        Updated result with speaker information
    """
    from funasr import AutoModel
    from rich import print as rprint
    
    # Load speaker model from HuggingFace
    # The correct HuggingFace model name is funasr/campplus
    try:
        spk_model = AutoModel(
            model="funasr/campplus",
            hub="hf",
            device=device,
            disable_update=True,
        )
    except Exception as e:
        rprint(f"[yellow]⚠️ Could not load CAM++ speaker model from HuggingFace: {e}[/yellow]")
        return result
    
    # Run speaker segmentation on audio
    try:
        spk_result = spk_model.generate(input=audio_path)
        
        if not spk_result or len(spk_result) == 0:
            return result
        
        # Parse speaker segments from result
        # Format depends on model output - typically contains speaker labels
        speaker_segments = []
        for item in spk_result:
            if 'spk' in item:
                speaker_segments.append({
                    'speaker': item['spk'],
                    'start': item.get('start', 0) / 1000.0 + time_offset,
                    'end': item.get('end', 0) / 1000.0 + time_offset
                })
        
        # Assign speakers to words based on timestamp overlap
        for segment in result.get('segments', []):
            for word in segment.get('words', []):
                word_start = word.get('start', 0)
                word_end = word.get('end', word_start + 0.1)
                word_mid = (word_start + word_end) / 2
                
                # Find the speaker segment that contains this word
                for spk_seg in speaker_segments:
                    if spk_seg['start'] <= word_mid <= spk_seg['end']:
                        word['speaker'] = spk_seg['speaker']
                        break
            
            # Also assign speaker to segment if any word has speaker
            speakers_in_segment = [w.get('speaker') for w in segment.get('words', []) if w.get('speaker')]
            if speakers_in_segment:
                # Use the most common speaker
                from collections import Counter
                segment['speaker'] = Counter(speakers_in_segment).most_common(1)[0][0]
        
        # Clean up
        del spk_model
        
    except Exception as e:
        rprint(f"[yellow]⚠️ Speaker segmentation inference failed: {e}[/yellow]")
    
    return result
