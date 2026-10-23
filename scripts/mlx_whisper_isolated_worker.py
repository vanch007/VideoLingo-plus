import sys
import os
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Isolated MLX Whisper Worker")
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--language", type=str, default="auto")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    import mlx_whisper

    lang = args.language if args.language != "auto" else None
    
    # Resolve local snapshot if possible
    model_path = args.model
    clean_name = model_path.replace("mlx-community/", "").replace("-mlx", "")
    import glob
    snaps = glob.glob(f"/Users/vanch/.cache/huggingface/hub/models--mlx-community--whisper-{clean_name}*/snapshots/*")
    if snaps:
        model_path = snaps[0]

    res = mlx_whisper.transcribe(
        args.audio,
        path_or_hf_repo=model_path,
        word_timestamps=True,
        language=lang,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        verbose=False
    )

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"SUCCESS: {len(res.get('segments', []))} segments")

if __name__ == "__main__":
    main()

