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
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language", default="auto")
    args = parser.parse_args()

    language = {
        "ms": "Malay",
        "zh": "Chinese",
        "en": "English",
    }.get(args.language.lower(), args.language)
    # The model loader emits cache progress on stdout. Keep stdout reserved for
    # the JSON contract consumed by VideoLingo's command readback hook.
    with contextlib.redirect_stdout(sys.stderr):
        model = load_model("mlx-community/Qwen3-ASR-0.6B-8bit")
        result = model.generate(args.audio, language=language)
    print(json.dumps({"text": result.text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
