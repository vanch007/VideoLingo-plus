"""Non-destructive A/B pilot: identical text/voice with and without row emotion.

Run from the project root with the VideoLingo environment. Output must be a
new directory. This is a pilot, not a replacement for full-video acceptance.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from pydub import AudioSegment

from core.all_tts_functions.mlx_router import build_mlx_tts_request
from core.config_utils import load_key
from core.providers.mlx_tts import IndexTTS2Backend
from core.providers.asr_readback import _run_asr
from core.providers.quality import content_similarity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="Comma separated source task numbers")
    parser.add_argument("--output", required=True)
    parser.add_argument("--emotion-weight", type=float, help="Explicit candidate weight; never inferred from pitch")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    tasks = pd.read_excel("output/audio/tts_tasks.xlsx")
    requests, cases = [], []
    for number in map(int, args.rows.split(",")):
        row = tasks.loc[tasks.number.eq(number)].iloc[0].to_dict()
        if args.emotion_weight is not None:
            row["emo_alpha"] = args.emotion_weight
        request = build_mlx_tts_request(str(row["text"]), str(output / f"{number}_dual.wav"),
                                        number, tasks, task_row=row, target_duration=None)
        plan = request.metadata["reference_plan"]
        if not request.emotion_ref:
            raise ValueError("Pilot requires source-row emotion configuration")
        shutil.copy2(request.emotion_ref, output / f"{number}_source.wav")
        # Freeze the voice anchor and all generation settings across both arms.
        baseline_plan = {**plan, "emotion_reference": None, "emotion_source": "speaker_reference",
                         "requires_separate_emotion": False, "fingerprint": "pilot_baseline"}
        baseline = replace(request, output_path=str(output / f"{number}_anchor_only.wav"),
                           emotion_ref=None, metadata={**request.metadata,
                           "requires_separate_emotion": False, "reference_plan": baseline_plan})
        requests.extend([baseline, request])
        cases.append({"number": number, "speaker": request.speaker_id, "text": request.text,
                      "reference_plan": plan, "source_duration": float(row["duration"])})
    (output / "pilot_inputs.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    config = dict(load_key("mlx_tts.backends.indextts2", {}))
    config["fit_duration"] = False
    results = IndexTTS2Backend(config).synthesize_batch(requests)
    for request, result in zip(requests, results):
        print(f"generated {Path(request.output_path).name}: {result.duration:.3f}s", flush=True)
    for case in cases:
        number = case["number"]
        comparison = AudioSegment.empty()
        for variant in ("source", "anchor_only", "dual"):
            clip = AudioSegment.from_file(output / f"{number}_{variant}.wav")
            comparison += clip + AudioSegment.silent(duration=650)
            if variant != "source":
                readback = _run_asr(str(output / f"{number}_{variant}.wav"), "en")
                case[variant] = {
                    "duration": len(clip) / 1000, "asr_status": readback.status,
                    "transcript": readback.transcript,
                    "content_score": content_similarity(case["text"], readback.transcript or ""),
                    "speaker_similarity": "missing evidence", "emotion_fidelity": "pending listening",
                }
                print(f"readback {number} {variant}: {case[variant]}", flush=True)
        comparison.export(output / f"{number}_comparison.wav", format="wav")
        (output / "pilot_results.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
