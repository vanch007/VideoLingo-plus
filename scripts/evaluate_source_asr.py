"""Evaluate timed source transcripts against explicit external reference checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.source_accuracy import evaluate_source_checkpoints


def load_result(path):
    if path.suffix == ".xlsx":
        import pandas as pd
        rows = pd.read_excel(path).to_dict("records")
        return {"segments": [{"start": r["start"], "end": r["end"], "text": str(r["text"]).strip('"')} for r in rows]}
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {"segments": [s for r in data for s in r["segments"]]} if data and "segments" in data[0] else {"segments": data}
    return data.get("result", data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", nargs=2, action="append", metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text())
    results = {name: evaluate_source_checkpoints(load_result(Path(path)), reference) for name, path in args.candidate}
    Path(args.output).write_text(json.dumps({"reference": args.reference, "results": results}, ensure_ascii=False, indent=2))
    print(json.dumps({name: {k: v for k, v in r.items() if k != "checkpoints"} for name, r in results.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
