"""Align locked target speech unit-by-unit with one resident local aligner."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import re
import sys


def alignment_text(text):
    """Keep adjoining words separate when the aligner removes punctuation."""
    def separate(match):
        numeric = (match.group() in '.,' and match.start() > 0
                   and text[match.start()-1].isdigit() and text[match.end()].isdigit())
        return match.group() if numeric else match.group() + ' '
    return re.sub(r'[.!?;,，。！？；:：—–/](?=\w)', separate, text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", required=True)
    args = parser.parse_args()
    units = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    result = []
    with contextlib.redirect_stdout(sys.stderr):
        from mlx_audio.stt.utils import load_model
        model = load_model(args.model)
        for unit in units:
            words = model.generate(audio=unit["audio"], text=alignment_text(unit["text"]), language=args.language)
            result.append({"unit_id": unit["unit_id"], "words": [
                {"word": w.text, "start": w.start_time, "end": w.end_time} for w in words]})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
