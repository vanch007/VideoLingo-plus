"""VideoLingo adapter runner for the local MLX Ming Omni TTS checkout.

The upstream CLI exposes text generation but not the speaker-reference bridge.
This thin runner calls the upstream ``MingMLX`` API directly and keeps all
Ming-specific imports isolated in the Ming project.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


MING_ROOT = Path(os.environ.get("MING_ROOT", "/Users/vanch/mlx-Ming-omni-tts")).expanduser()
if str(MING_ROOT) not in sys.path:
    sys.path.insert(0, str(MING_ROOT))

from mlx_ming.pipeline import MingMLX  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cloned speech with local MLX Ming Omni TTS")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="Please generate speech based on the following description.\n",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    args = parser.parse_args()

    reference, sample_rate = sf.read(args.ref_audio, always_2d=False, dtype="float32")
    reference = np.asarray(reference, dtype=np.float32)
    if reference.ndim == 2:
        # soundfile returns (samples, channels); Ming's torch bridge expects a
        # mono waveform or channel-first tensor.
        reference = reference.mean(axis=1)
    ming = MingMLX(str(args.model_dir), dtype=args.dtype)
    waveform = ming.generate(
        args.prompt,
        args.text,
        prompt_waveform=reference,
        prompt_sample_rate=int(sample_rate),
        # This integration uses the stable CampPlus speaker-embedding path.
        # The upstream prompt-latent branch can emit near-silent output for
        # very short references, so keep the source transcript as audit
        # metadata only and do not enable that branch by default.
        prompt_text=None,
        max_decode_steps=args.max_steps,
        cfg=args.cfg,
        noise_seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, waveform, ming.sample_rate)
    print(
        f"wrote {len(waveform)} samples @ {ming.sample_rate}Hz -> {args.output}; "
        f"ref_text_supplied={bool(args.ref_text)}"
    )


if __name__ == "__main__":
    main()
