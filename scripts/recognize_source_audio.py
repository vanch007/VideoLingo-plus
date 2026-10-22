"""Offline source recognition worker. Stdout is reserved for the JSON receipt."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--backend", choices=["moss", "qwen3", "confucius"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--moss-root", default="/Users/vanch/mlx-MOSS-Transcribe-Diarize")
    parser.add_argument("--hotwords-file")
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    model_path = Path(args.model).expanduser()
    if not model_path.is_dir() or not (model_path / "config.json").is_file() or not list(model_path.glob("*.safetensors")):
        raise FileNotFoundError(f"Complete local source model required: {model_path}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    hotwords = json.loads(Path(args.hotwords_file).read_text()) if args.hotwords_file else []
    if hotwords and args.backend != "moss":
        raise ValueError("This worker supports acoustic vocabulary hints only with MOSS")
    with contextlib.redirect_stdout(sys.stderr):
        if args.backend == "moss":
            import numpy as np
            sys.path.insert(0, args.moss_root)
            from moss_transcribe_diarize.mlx.model import load_model
            from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
            from moss_transcribe_diarize.subtitle import subtitle_segments_from_transcript, export_json
            pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", args.audio,
                                  "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                                 capture_output=True, check=True)
            audio = np.frombuffer(pcm.stdout, dtype="<f4").copy()
            prompt = DEFAULT_PROMPT
            if hotwords:
                prompt += "\n参考词汇（仅在音频确实说出时使用）：" + ", ".join(hotwords)
            clip_duration = len(audio) / 16000
            effective_max_tokens = min(args.max_tokens, max(64, int(clip_duration * 35)))
            result = load_model(str(model_path)).generate(audio, prompt=prompt,
                       max_tokens=effective_max_tokens, temperature=0)
            segments = json.loads(export_json(subtitle_segments_from_transcript(result.text, postprocess=False)))
            text = "".join(s["text"] for s in segments) if args.language in {"zh", "ja", "ko", "yue"} else " ".join(s["text"] for s in segments)
        else:
            from mlx_audio.stt.utils import load_model, load_audio
            audio = load_audio(args.audio)
            language = {"zh": "Chinese", "en": "English", "ja": "Japanese"}.get(args.language, args.language)
            result = load_model(str(model_path)).generate(audio, language=language,
                        temperature=0, max_tokens=args.max_tokens, chunk_duration=30.)
            text = result.text
            segments = []
            effective_max_tokens = args.max_tokens
    tokens = getattr(result, "generation_tokens", None)
    if tokens is not None and tokens >= effective_max_tokens and text.strip():
        raise RuntimeError("Source recognition reached token budget; incomplete text rejected")
    print(json.dumps({"text": text, "segments": segments, "raw_text": result.text,
                      "backend": args.backend, "model": str(model_path.resolve()),
                      "generation_tokens": tokens, "token_budget_reached": False,
                      "hotwords": hotwords}, ensure_ascii=False))


if __name__ == "__main__":
    main()
