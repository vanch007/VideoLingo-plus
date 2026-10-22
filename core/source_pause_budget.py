"""Measure quiet source gaps before assigning a bounded dubbing lead-in."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from core.pipeline.artifact_manifest import hash_path


def quiet_source_spans(audio_path: Path, report_path: Path) -> dict:
    identity = {"audio_sha256": hash_path(audio_path), "worker_sha256": hash_path(Path(__file__)),
                "window_seconds": .02, "threshold_floor_dbfs": -40., "relative_floor_db": 15.}
    if report_path.is_file():
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        if saved.get("identity") == identity:
            return saved
    samples, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Source pause measurement requires finite nonempty audio")
    global_rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    threshold = min(-40., 20 * math.log10(max(global_rms, 1e-12)) - 15.)
    block = max(1, round(sr * .02))
    count = len(samples) // block
    blocks = samples[:count * block].reshape(count, block, samples.shape[1])
    # Maximum channel RMS avoids treating a voice on only one channel as silence.
    rms = np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1)).max(axis=1)
    quiet = rms <= 10 ** (threshold / 20)
    spans, beginning = [], None
    for i, is_quiet in enumerate([*quiet, False]):
        if is_quiet and beginning is None:
            beginning = i
        elif not is_quiet and beginning is not None:
            if i - beginning >= 2:
                spans.append([round(beginning * block / sr, 6), round(i * block / sr, 6)])
            beginning = None
    report = {"status": "measured", "basis": "source_vocal_RMS_in_recognized_dialogue_gaps",
              "identity": identity, "threshold_dbfs": threshold, "quiet_spans": spans}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
