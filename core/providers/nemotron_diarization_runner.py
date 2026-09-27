"""Nemotron 3 Diarization runner executed in the dedicated MLX environment.

Outputs:
  - segments.json: list of detected speaker segments
  - rttm.txt: RTTM formatted speaker intervals
  - summary.json: runtime and audio metrics
  - probabilities.npz (optional): frame-level speaker probabilities
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nemotron Diarization via MLX")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--model",
        default="mlx-community/Nemotron-3-Diarization-8bit",
        help="HuggingFace model ID or local directory",
    )
    parser.add_argument(
        "--preset",
        default="offline",
        choices=["offline", "low", "very_low", "ultra_low"],
        help="Streaming/processing preset",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Speaker activity threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Minimum speech segment duration in seconds",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.0,
        help="Maximum gap in seconds between segments of same speaker to merge",
    )
    parser.add_argument(
        "--save-probs",
        action="store_true",
        help="Save raw frame-level probability tensor to probabilities.npz",
    )
    return parser.parse_args()


def format_speaker_id(speaker_idx: int) -> str:
    """Format arrival-order speaker index 0 -> S01, 1 -> S02, etc."""
    return f"S{speaker_idx + 1:02d}"


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        sys.stderr.write(f"Error: audio file not found: {audio_path}\n")
        return 1

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.perf_counter()

    try:
        from mlx_audio.vad import load
    except ImportError as exc:
        sys.stderr.write(f"Error importing mlx_audio: {exc}\n")
        return 2

    # Load model
    load_start = time.perf_counter()
    try:
        model = load(args.model, strict=True)
    except Exception as exc:
        sys.stderr.write(f"Error loading model {args.model}: {exc}\n")
        return 3
    load_time = time.perf_counter() - load_start

    # Configure preset if needed
    if hasattr(model, "set_streaming_config") and args.preset:
        try:
            model.set_streaming_config(args.preset)
        except Exception as exc:
            sys.stderr.write(f"Warning setting streaming preset {args.preset}: {exc}\n")

    # Run inference
    infer_start = time.perf_counter()
    try:
        result = model.generate(
            str(audio_path),
            threshold=args.threshold,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
        )
    except Exception as exc:
        sys.stderr.write(f"Error during diarization inference: {exc}\n")
        return 4
    infer_time = time.perf_counter() - infer_start

    # Process segments
    segments_data: list[dict[str, Any]] = []
    seen_speakers: set[int] = set()

    for idx, seg in enumerate(result.segments):
        spk_num = int(seg.speaker)
        seen_speakers.add(spk_num)
        segments_data.append(
            {
                "index": idx,
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "speaker": format_speaker_id(spk_num),
                "speaker_index": spk_num,
            }
        )

    # Calculate frame stride
    hop_length = getattr(model._processor_config, "hop_length", 160)
    sample_rate = getattr(model, "sample_rate", 16000)
    subsampling = getattr(model.config, "output_subsampling_factor", 1)
    frame_stride = (hop_length * subsampling) / sample_rate

    # Convert probs
    probs_np: np.ndarray | None = None
    total_frames = 0
    if result.speaker_probs is not None:
        probs_np = np.array(result.speaker_probs)
        total_frames = int(probs_np.shape[0])

    if args.save_probs and probs_np is not None:
        probs_file = out_dir / "probabilities.npz"
        np.savez_compressed(
            probs_file,
            probs=probs_np.astype(np.float32),
            frame_stride=frame_stride,
            num_speakers=len(seen_speakers),
        )

    # Save segments.json
    segments_file = out_dir / "segments.json"
    segments_file.write_text(
        json.dumps(segments_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Save rttm.txt
    rttm_file = out_dir / "rttm.txt"
    rttm_file.write_text(result.text, encoding="utf-8")

    # Save summary.json
    total_time = time.perf_counter() - start_total
    summary_data = {
        "status": "pass",
        "audio_path": str(audio_path),
        "model": args.model,
        "preset": args.preset,
        "threshold": args.threshold,
        "min_duration": args.min_duration,
        "merge_gap": args.merge_gap,
        "load_time_seconds": round(load_time, 3),
        "infer_time_seconds": round(infer_time, 3),
        "total_time_seconds": round(total_time, 3),
        "num_speakers": len(seen_speakers),
        "speaker_labels": sorted(format_speaker_id(s) for s in seen_speakers),
        "speaker_indices": sorted(list(seen_speakers)),
        "segments_count": len(segments_data),
        "total_frames": total_frames,
        "frame_stride_seconds": round(frame_stride, 4),
        "audio_duration_seconds": round(total_frames * frame_stride, 3) if total_frames else 0.0,
    }

    summary_file = out_dir / "summary.json"
    summary_file.write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Output json summary to stdout for calling parent process
    print(json.dumps(summary_data, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
