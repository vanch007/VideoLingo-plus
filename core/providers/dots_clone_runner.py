from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_dots_tts.audio_preprocess import trim_waveform_edges
from mlx_dots_tts.fbank import extract_speaker_fbank_np
from mlx_dots_tts.runtime import MlxDotsTtsRuntime


def _build_prompt_cache(runtime, ref_audio: Path, speaker_scale: float, seed: int):
    """Decode with ffmpeg so cloning does not require dots.tts' optional librosa extra."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        decoded_path = Path(handle.name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(ref_audio),
                "-ar", str(runtime.sample_rate), "-ac", "1", str(decoded_path),
            ],
            check=True,
        )
        audio, sample_rate = sf.read(decoded_path, dtype="float32", always_2d=False)
    finally:
        decoded_path.unlink(missing_ok=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio, metrics = trim_waveform_edges(
        audio,
        int(sample_rate),
        enabled=True,
        top_db=18.0,
        pad_seconds=0.08,
        metric_prefix="prompt_audio",
    )
    prompt_audio = audio[None, :]
    fbank = extract_speaker_fbank_np(audio, sample_rate=int(sample_rate))[None, :, :]
    cache = runtime.prepare_prompt_cache(
        prompt_audio=mx.array(prompt_audio),
        fbank=mx.array(fbank.astype(np.float32, copy=False)),
        fbank_lengths=mx.array(np.array([fbank.shape[1]], dtype=np.int32)),
        speaker_scale=speaker_scale,
        seed=seed,
        include_prompt_history=False,
    )
    metrics["prompt_audio_sample_rate"] = int(sample_rate)
    return cache, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoLingo single-clip dots.tts clone runner")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="ZH")
    parser.add_argument("--target-duration", type=float)
    parser.add_argument("--max-audio-tokens", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--speaker-scale", type=float, default=1.5)
    parser.add_argument("--qwen-variant", default="int8_g64")
    parser.add_argument("--component-variant", default="int8_g64")
    args = parser.parse_args()

    runtime = MlxDotsTtsRuntime.from_pretrained(
        args.model,
        qwen_variant=args.qwen_variant,
        component_variant=args.component_variant,
    )
    prompt_cache, prompt_metrics = _build_prompt_cache(runtime, args.ref_audio, args.speaker_scale, args.seed)
    max_tokens = args.max_audio_tokens
    if args.target_duration:
        max_tokens = min(max_tokens, max(8, int(math.ceil(args.target_duration / 0.16 + 6))))
    result = runtime.generate(
        text=args.text,
        prompt_cache=prompt_cache,
        prompt_text="",
        max_audio_tokens=max_tokens,
        num_steps=args.num_steps,
        eos_threshold=0.8,
        seed=args.seed,
        language=args.language,
        use_fm_buckets=True,
        compile_meanflow=True,
        compile_vocoder=True,
        speaker_preroll_patches=2,
        profile=False,
    )
    audio = np.array(result["audio"].astype(mx.float32))[0, 0].astype(np.float32, copy=False)
    audio, trim_metrics = trim_waveform_edges(
        audio,
        int(runtime.sample_rate),
        enabled=True,
        top_db=18.0,
        pad_seconds=0.08,
        metric_prefix="output_audio",
    )
    if audio.size == 0:
        raise RuntimeError("dots.tts produced empty audio")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, int(runtime.sample_rate))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sample_rate": int(runtime.sample_rate),
                "duration_seconds": audio.size / int(runtime.sample_rate),
                "time_used": float(result["time_used"]),
                "rtf": float(result["rtf"]),
                "prompt_audio_metrics": prompt_metrics,
                "trim_metrics": trim_metrics,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
