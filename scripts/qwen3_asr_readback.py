#!/usr/bin/env python3
"""Emit a compact Qwen3-ASR transcript for VideoLingo's local readback hook."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys

from mlx_audio.stt.utils import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio")
    parser.add_argument("--manifest", help="Batch JSON [{audio, language}] with one resident model")
    parser.add_argument("--language", default="auto")
    args = parser.parse_args()

    if not args.audio and not args.manifest:
        parser.error("--audio or --manifest is required")
    names = {
        "ms": "Malay",
        "zh": "Chinese",
        "en": "English",
    }
    # The model loader emits cache progress on stdout. Keep stdout reserved for
    # the JSON contract consumed by VideoLingo's command readback hook.
    with contextlib.redirect_stdout(sys.stderr):
        model = load_model("mlx-community/Qwen3-ASR-0.6B-8bit")
        if args.manifest:
            from pathlib import Path
            outputs = []
            for item in json.loads(Path(args.manifest).read_text(encoding="utf-8")):
                language = names.get(item["language"].lower(), item["language"])
                result = model.generate(item["audio"], language=language)
                outputs.append({"audio": item["audio"], "text": result.text})
        else:
            result = model.generate(args.audio, language=names.get(args.language.lower(), args.language))
            outputs = {"text": result.text}
    print(json.dumps(outputs, ensure_ascii=False))


if __name__ == "__main__":
    main()
