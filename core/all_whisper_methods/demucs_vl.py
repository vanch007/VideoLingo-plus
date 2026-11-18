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


def _audio_duration(path: str) -> float:
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def _wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "64k") -> None:
    """Encode through FFmpeg's libmp3lame: its LAME header lets decoders trim encoder priming
    (lameenc writes none, which left the MP3 twins ~25 ms late)."""
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav_path, "-c:a", "libmp3lame", "-b:a", bitrate, mp3_path], check=True)


def _decode_float(path: str, start: float, end: float, samplerate: int):
    """Stereo float32 tensor (2, n) of [start, end] at ``samplerate``."""
    import subprocess
    import numpy as np
    proc = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.6f}", "-t", f"{end - start:.6f}", "-i", path,
                           "-vn", "-ac", "2", "-ar", str(samplerate), "-f", "f32le", "-"], capture_output=True, check=True)
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    return torch.from_numpy(data[: len(data) // 2 * 2].reshape(-1, 2).T.copy())


def separate_in_chunks(separator, input_file: str, samplerate: int, vocal_wav: str, background_wav: str,
                       chunk_seconds: int, context_seconds: float = 5.0):
    """Separate a long file chunk by chunk: bounded memory, streamed to disk.

    Every chunk is heard with ``context_seconds`` of audio on both sides so the model sees no
    hard edge; only the central part is kept. Chunk boundaries sit on an integer sample grid so
    the stems have exactly the input's length. Returns (vocal_peak, background_peak) before
    clipping; samples are clamped to 16-bit (a global rescale needs the whole file).
    """
    import numpy as np
    import soundfile as sf
    total = round(_audio_duration(input_file) * samplerate)
    step = int(chunk_seconds * samplerate)
    ctx = int(context_seconds * samplerate)
    peaks = [0.0, 0.0]
    with sf.SoundFile(vocal_wav, "w", samplerate=samplerate, channels=2, subtype="PCM_16") as vocal_out, \
         sf.SoundFile(background_wav, "w", samplerate=samplerate, channels=2, subtype="PCM_16") as bg_out:
        for first in range(0, total, step):
            last = min(first + step, total)
            lo, hi = max(0, first - ctx), min(total, last + ctx)
            wav = _decode_float(input_file, lo / samplerate, hi / samplerate, samplerate)
            _, stems = separator.separate_tensor(wav, samplerate)
            keep = slice(first - lo, first - lo + (last - first))
            vocals = stems["vocals"][:, keep].cpu()
            background = sum(a for k, a in stems.items() if k != "vocals")[:, keep].cpu()
            peaks[0] = max(peaks[0], float(vocals.abs().max()))
            peaks[1] = max(peaks[1], float(background.abs().max()))
            vocal_out.write(np.clip(vocals.numpy().T, -1.0, 1.0))
            bg_out.write(np.clip(background.numpy().T, -1.0, 1.0))
            del wav, stems, vocals, background
            gc.collect()
    return peaks[0], peaks[1]


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

    from core import asset_cache
    from core.config_utils import load_key
    chunk_seconds = int(load_key("demucs_chunk_seconds", 0) or 0)
    cache_files = {"vocal.wav": vocal_wav, "background.wav": background_wav, "vocal.mp3": VOCAL_AUDIO_FILE,
                   "background.mp3": BACKGROUND_AUDIO_FILE, "demucs_receipt.json": receipt_file}
    cache_key = asset_cache.make_key(kind="demucs", input_sha256=_hash_file(input_file), model=model_name,
                                     chunk_seconds=chunk_seconds, shifts=1, overlap=0.25) if os.path.isfile(input_file) else None
    hit = asset_cache.lookup("demucs", cache_key, list(cache_files)) if cache_key else None
    if hit:
        os.makedirs(AUDIO_DIR, exist_ok=True)
        asset_cache.restore(hit, cache_files)
        rprint(f"[green]♻️ Demucs stems restored from the cross-task cache ({cache_key[:10]}); separation skipped.[/green]")
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
    input_file = RAW_MASTER_AUDIO_FILE if os.path.exists(RAW_MASTER_AUDIO_FILE) else RAW_AUDIO_FILE
    vocal_wav = os.path.join(AUDIO_DIR, "vocal.wav")
    background_wav = os.path.join(AUDIO_DIR, "background.wav")
    chunked = chunk_seconds > 0 and _audio_duration(input_file) > chunk_seconds * 1.5
    outputs = background = None
    if chunked:
        console.print(f"🧩 Chunked separation ({chunk_seconds}s chunks, 5s context)")
        vocal_peak, bg_peak = separate_in_chunks(separator, input_file, model.samplerate, vocal_wav, background_wav, chunk_seconds)
        _wav_to_mp3(vocal_wav, VOCAL_AUDIO_FILE)
        _wav_to_mp3(background_wav, BACKGROUND_AUDIO_FILE)
    else:
        _, outputs = separator.separate_audio_file(input_file)

        kwargs = {"samplerate": model.samplerate, "bitrate": 64, "preset": 2,
                 "clip": "rescale", "as_float": False, "bits_per_sample": 16}


        console.print("🎤 Saving vocals track (WAV + MP3)...")
        save_audio(outputs['vocals'].cpu(), vocal_wav, samplerate=model.samplerate, clip="rescale", bits_per_sample=16)
        _wav_to_mp3(vocal_wav, VOCAL_AUDIO_FILE)

        console.print("🎹 Saving background music (WAV + MP3)...")
        background = sum(audio for source, audio in outputs.items() if source != 'vocals')
        save_audio(background.cpu(), background_wav, samplerate=model.samplerate, clip="rescale", bits_per_sample=16)
        _wav_to_mp3(background_wav, BACKGROUND_AUDIO_FILE)

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
        "clip": "clamp" if chunked else "rescale",
        "chunk_seconds": chunk_seconds if chunked else 0,
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
    if cache_key:
        asset_cache.store("demucs", cache_key, cache_files, meta={"model": model_name, "chunk_seconds": chunk_seconds})

    # Clean up memory
    del outputs, background, model, separator
    gc.collect()

if __name__ == "__main__":
    demucs_main()
