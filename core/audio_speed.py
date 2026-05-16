from __future__ import annotations

import shutil
import subprocess
import time

from pydub import AudioSegment
from rich import print as rprint

from core.all_whisper_methods.audio_preprocess import get_audio_duration


def build_atempo_filter(speed_factor: float) -> str:
    factors = []
    remaining = float(speed_factor)
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6g}" for factor in factors)


def adjust_audio_speed(input_file: str, output_file: str, speed_factor: float) -> None:
    if abs(speed_factor - 1.0) < 0.001:
        shutil.copy2(input_file, output_file)
        return

    input_duration = get_audio_duration(input_file)
    expected_duration = input_duration / speed_factor
    command = ["ffmpeg", "-i", input_file, "-filter:a", build_atempo_filter(speed_factor), "-y", output_file]
    for attempt in range(2):
        try:
            subprocess.run(command, check=True, stderr=subprocess.PIPE)
            _validate_adjusted_duration(output_file, input_duration, expected_duration)
            return
        except subprocess.CalledProcessError:
            if attempt == 1:
                rprint("[red]❌ Audio speed adjustment failed, max retries reached (2)[/red]")
                raise
            rprint(f"[yellow]⚠️ Audio speed adjustment failed, retrying in 1s ({attempt + 1}/2)[/yellow]")
            time.sleep(1)


def _validate_adjusted_duration(output_file: str, input_duration: float, expected_duration: float) -> None:
    output_duration = get_audio_duration(output_file)
    diff = output_duration - expected_duration
    if abs(diff) < 0.05:
        return
    if output_duration <= expected_duration:
        return
    if input_duration < 3 and diff <= 0.1:
        _trim_audio(output_file, expected_duration)
        return
    if output_duration <= expected_duration * 1.05:
        rprint(f"[yellow]⚠️ Duration mismatch accepted: {output_duration:.2f}s (expected {expected_duration:.2f}s)[/yellow]")
        return
    raise RuntimeError(
        f"Audio duration abnormal: input={input_duration:.2f}s, "
        f"output={output_duration:.2f}s, expected={expected_duration:.2f}s"
    )


def _trim_audio(output_file: str, expected_duration: float) -> None:
    audio = AudioSegment.from_wav(output_file)
    trimmed_audio = audio[: expected_duration * 1000].fade_out(10)
    trimmed_audio.export(output_file, format="wav")
    rprint(f"[yellow]✂️ Trimmed to expected duration: {expected_duration:.2f}s[/yellow]")
