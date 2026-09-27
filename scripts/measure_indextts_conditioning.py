"""Diagnostic CampPlus identity and native GPT emotion proxies for an A/B pilot.

Run with the IndexTTS environment. These are model-internal proxies, not a
calibrated perceptual acceptance gate or an independent emotion recognizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--backend-root", required=True)
    parser.add_argument("--model", default="models/mlx-IndexTTS-2.5-8bit")
    args = parser.parse_args()
    root = Path(args.backend_root).resolve()
    sys.path.insert(0, str(root))
    import mlx.core as mx
    import numpy as np
    import torch.nn.functional as F
    from mlx_indextts.generate_v25 import IndexTTSv25

    pilot = Path(args.pilot).resolve()
    cases = json.loads((pilot / "pilot_inputs.json").read_text())
    tts = IndexTTSv25(str(root / args.model))
    features = {}

    def extract(path):
        if path not in features:
            audio = tts._process_reference_audio(path)
            features[path] = (audio["style"], tts._mlx_reference_features(audio)["emotion_vec"])
        return features[path]

    def emotion_cos(a, b):
        value = mx.sum(a * b, axis=-1) / mx.maximum(
            mx.sqrt(mx.sum(a * a, axis=-1) * mx.sum(b * b, axis=-1)), mx.array(1e-8)
        )
        mx.eval(value)
        return float(np.asarray(value).mean())

    anchors = {case["speaker"]: extract(case["reference_plan"]["speaker_reference"]["path"])[0]
               for case in cases}
    results = []
    for case in cases:
        source_emo = extract(case["reference_plan"]["emotion_reference"]["path"])[1]
        result = {"number": case["number"], "speaker": case["speaker"]}
        for variant in ("anchor_only", "dual"):
            style, emotion = extract(str(pilot / f"{case['number']}_{variant}.wav"))
            scores = {speaker: float(F.cosine_similarity(style.float(), anchor.float(), dim=-1).mean())
                      for speaker, anchor in anchors.items()}
            result[variant] = {"speaker_cosines": scores,
                               "highest_scoring_speaker": max(scores, key=scores.get),
                               "emotion_cosine_to_source": emotion_cos(emotion, source_emo)}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    (pilot / "conditioning_metrics.json").write_text(json.dumps({
        "metrics": "CampPlus cosine and native GPT emotion-vector cosine",
        "limitations": "Internal model proxies; no universal threshold; short/cross-language clips are uncertain. Listening pending.",
        "model": str(root / args.model), "rows": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
