import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
from rich.console import Console
from rich import print as rprint
from torch.cuda import is_available as is_cuda_available
import gc

AUDIO_DIR = "output/audio"
RAW_AUDIO_FILE = os.path.join(AUDIO_DIR, "raw.mp3")
BACKGROUND_AUDIO_FILE = os.path.join(AUDIO_DIR, "background.mp3")
VOCAL_AUDIO_FILE = os.path.join(AUDIO_DIR, "vocal.mp3")

def demucs_main(model_name: str = "htdemucs"):
    if os.path.exists(VOCAL_AUDIO_FILE) and os.path.exists(BACKGROUND_AUDIO_FILE):
        rprint(f"[yellow]⚠️ {VOCAL_AUDIO_FILE} and {BACKGROUND_AUDIO_FILE} already exist, skip Demucs processing.[/yellow]")
        return
    
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
    _, outputs = separator.separate_audio_file(RAW_AUDIO_FILE)
    
    kwargs = {"samplerate": model.samplerate, "bitrate": 64, "preset": 2, 
             "clip": "rescale", "as_float": False, "bits_per_sample": 16}
    
    console.print("🎤 Saving vocals track...")
    save_audio(outputs['vocals'].cpu(), VOCAL_AUDIO_FILE, **kwargs)
    
    console.print("🎹 Saving background music...")
    background = sum(audio for source, audio in outputs.items() if source != 'vocals')
    save_audio(background.cpu(), BACKGROUND_AUDIO_FILE, **kwargs)
    
    # Clean up memory
    del outputs, background, model, separator
    gc.collect()
    
    console.print("[green]✨ Audio separation completed![/green]")

if __name__ == "__main__":
    demucs_main()
