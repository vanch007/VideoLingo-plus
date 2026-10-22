"""Local Qwen forced alignment adapter; stdout is reserved for JSON."""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path


def group_alignment_segments(segments, *, max_seconds=20., max_gap=1.5):
    groups = []
    for segment in segments:
        start, end = float(segment["start"]), float(segment["end"])
        if groups and end - groups[-1]["start"] <= max_seconds and start - groups[-1]["end"] <= max_gap:
            groups[-1]["segments"].append(segment)
            groups[-1]["end"] = max(end, groups[-1]["end"])
        else:
            groups.append({"start": start, "end": end, "segments": [segment]})
    return groups


def alignment_windows(groups, audio_seconds, padding=.25):
    """Bound context without presenting adjacent speech to two text windows.

    Seams only choose the audio to measure. No generated word timestamp is
    shifted, stretched, or copied from the recognizer's phrase estimate.
    """
    windows = [[max(0., float(g['start']) - padding), min(audio_seconds, float(g['end']) + padding)]
               for g in groups]
    for i in range(1, len(windows)):
        if windows[i][0] < windows[i - 1][1]:
            seam = (float(groups[i - 1]['end']) + float(groups[i]['start'])) / 2
            windows[i - 1][1] = seam
            windows[i][0] = seam
    if any(start < 0 or end <= start or end > audio_seconds for start, end in windows):
        raise ValueError('Recognized phrase windows cannot be aligned monotonically')
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--model", default="mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
    parser.add_argument("--segments-file", help="ASR phrase intervals used only to bound acoustic alignment")
    parser.add_argument("--individual-phrases", action="store_true", help="Retry collapsed phrase boundaries in separate audio crops")
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        from mlx_audio.stt.utils import load_model
        model = load_model(args.model)
        words, groups = [], []
        if args.segments_file:
            import subprocess
            import numpy as np
            pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", args.audio,
                                  "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"], capture_output=True, check=True)
            audio = np.frombuffer(pcm.stdout, dtype="<f4").copy()
            segments = json.loads(Path(args.segments_file).read_text())
            joiner = "" if args.language in {"Chinese", "Japanese", "Korean", "Cantonese"} else " "
            alignment_groups = ([{'start': s['start'], 'end': s['end'], 'segments': [s]} for s in segments]
                                if args.individual_phrases else group_alignment_segments(segments))
            windows = alignment_windows(alignment_groups, len(audio) / 16000)
            for group, (start, end) in zip(alignment_groups, windows):
                text = joiner.join(s["text"] for s in group["segments"])
                result = model.generate(audio=audio[int(start * 16000):int(end * 16000)], text=text, language=args.language)
                words.extend({"word": item.text, "start": item.start_time + start,
                              "end": item.end_time + start} for item in result)
                groups.append({"start": start, "end": end, "text": text})
        else:
            result = model.generate(audio=args.audio, text=Path(args.text_file).read_text(), language=args.language)
            words = [{"word": item.text, "start": item.start_time, "end": item.end_time} for item in result]
    print(json.dumps({"words": words, "alignment_groups": groups}, ensure_ascii=False))


if __name__ == "__main__":
    main()
