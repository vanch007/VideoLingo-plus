import json
import hashlib, time

def _hash_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
from rich.console import Console
from rich import print as rprint
from torch.cuda import is_available as is_cuda_available
import gc

AUDIO_DIR = "output/audio"
RAW_AUDIO_FILE = os.path.join(AUDIO_DIR, "raw.mp3")
RAW_MASTER_AUDIO_FILE = os.path.join(AUDIO_DIR, "raw_master.wav")
BACKGROUND_AUDIO_FILE = os.path.join(AUDIO_DIR, "background.mp3")
VOCAL_AUDIO_FILE = os.path.join(AUDIO_DIR, "vocal.mp3")

def demucs_main(model_name: str = "htdemucs"):
    vocal_wav = os.path.join(AUDIO_DIR, "vocal.wav")
    background_wav = os.path.join(AUDIO_DIR, "background.wav")
    receipt_file = os.path.join(AUDIO_DIR, "demucs_receipt.json")
    input_file = RAW_MASTER_AUDIO_FILE if os.path.exists(RAW_MASTER_AUDIO_FILE) else RAW_AUDIO_FILE
    if os.path.exists(vocal_wav) and os.path.exists(background_wav) and os.path.exists(receipt_file):
        try:
            with open(receipt_file, 'r', encoding='utf-8') as rf:
                rc = json.load(rf)
            input_valid = (rc.get("input_sha256") == _hash_file(input_file))
            model_valid = (rc.get("model") == model_name)
            vocal_valid = (rc.get("vocal_sha256") == _hash_file(vocal_wav))
            bg_valid = (rc.get("background_sha256") == _hash_file(background_wav))
            if input_valid and model_valid and vocal_valid and bg_valid:
                rprint(f"[yellow]⚠️ {vocal_wav} and {background_wav} are current with receipt, skip Demucs processing.[/yellow]")
                return
            else:
                rprint(f"[yellow]⚠️ Demucs cache invalid (input={input_valid}, model={model_valid}, vocal={vocal_valid}, bg={bg_valid}). Recomputing separation...[/yellow]")
        except Exception:
            pass

    # Keep Demucs optional for stable-ts-only runs. Demucs 4.0.x does not
    # expose `demucs.api`, so importing it at module load previously broke
    # ASR even when `demucs: false`.
    try:
        from demucs.pretrained import get_model
        from demucs.audio import save_audio
        from demucs.api import Separator
    except ImportError as exc:
        raise RuntimeError(
            "Demucs is enabled but its API runtime is unavailable. "
            "Install the supported Demucs API build or set demucs: false."
        ) from exc

    class PreloadedSeparator(Separator):
        def __init__(self, model, shifts: int = 1, overlap: float = 0.25,
                     split: bool = True, segment=None, jobs: int = 0):
            self._model, self._audio_channels, self._samplerate = model, model.audio_channels, model.samplerate
            device = "cuda" if is_cuda_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            self.update_parameter(device=device, shifts=shifts, overlap=overlap, split=split,
                                  segment=segment, jobs=jobs, progress=True, callback=None, callback_arg=None)

    console = Console()
    os.makedirs(AUDIO_DIR, exist_ok=True)

    console.print(f"🤖 Loading <{model_name}> model...")
    model = get_model(model_name)
    separator = PreloadedSeparator(model=model, shifts=1, overlap=0.25)

    console.print("🎵 Separating audio...")
    input_file = RAW_MASTER_AUDIO_FILE if os.path.exists(RAW_MASTER_AUDIO_FILE) else RAW_AUDIO_FILE
    _, outputs = separator.separate_audio_file(input_file)

    kwargs = {"samplerate": model.samplerate, "bitrate": 64, "preset": 2,
             "clip": "rescale", "as_float": False, "bits_per_sample": 16}

    vocal_wav = os.path.join(AUDIO_DIR, "vocal.wav")
    background_wav = os.path.join(AUDIO_DIR, "background.wav")

    console.print("🎤 Saving vocals track (WAV + MP3)...")
    save_audio(outputs['vocals'].cpu(), vocal_wav, samplerate=model.samplerate, clip="rescale", bits_per_sample=16)
    save_audio(outputs['vocals'].cpu(), VOCAL_AUDIO_FILE, **kwargs)

    console.print("🎹 Saving background music (WAV + MP3)...")
    background = sum(audio for source, audio in outputs.items() if source != 'vocals')
    save_audio(background.cpu(), background_wav, samplerate=model.samplerate, clip="rescale", bits_per_sample=16)
    save_audio(background.cpu(), BACKGROUND_AUDIO_FILE, **kwargs)

    vocal_tensor = outputs['vocals'].cpu()
    bg_tensor = background.cpu()
    vocal_peak = float(vocal_tensor.abs().max().item()) if hasattr(vocal_tensor, 'abs') else 1.0
    bg_peak = float(bg_tensor.abs().max().item()) if hasattr(bg_tensor, 'abs') else 1.0
    vocal_denom = max(1.01 * vocal_peak, 1.0)
    bg_denom = max(1.01 * bg_peak, 1.0)
    vocal_scale = float(1.0 / vocal_denom)
    bg_scale = float(1.0 / bg_denom)

    model_sr = model.samplerate
    model_ch = getattr(model, "audio_channels", 2)
    receipt = {
        "model": model_name,
        "samplerate": model_sr,
        "channels": model_ch,
        "clip": "rescale",
        "input_file": input_file,
        "input_sha256": _hash_file(input_file),
        "vocal_wav": vocal_wav,
       "vocal_sha256": _hash_file(vocal_wav),
       "background_wav": background_wav,
       "background_sha256": _hash_file(background_wav),
        "delay_ms": 0.0,
        "delay_status": "zero_delay_aligned",
       "gain_applied": {
           "clip_strategy": "rescale",
            "vocals_raw_peak": round(vocal_peak, 6),
            "vocals_scale_applied": round(vocal_scale, 6),
            "background_raw_peak": round(bg_peak, 6),
            "background_scale_applied": round(bg_scale, 6),
            "is_unity_gain": bool(vocal_denom <= 1.0 and bg_denom <= 1.0),
        },
        "timestamp": time.time(),
    }
    with open(receipt_file, 'w', encoding='utf-8') as rf:
        json.dump(receipt, rf, ensure_ascii=False, indent=2)
    console.print(f"[green]✨ Audio separation completed with receipt → {receipt_file}[/green]")

    # Clean up memory
    del outputs, background, model, separator
    gc.collect()

if __name__ == "__main__":
    demucs_main()
