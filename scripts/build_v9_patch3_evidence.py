import json
import os
import sys
import math
import shutil
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import pandas as pd

ROOT = Path("/Users/vanch/VideoLingo-plus").resolve()
OUT = ROOT / "run_workspaces/gemini_fidelity_repair_v9_patch3_20260927"
OUT.mkdir(parents=True, exist_ok=True)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

print("1. Checking frozen baseline deliverables/e0c803_selected_v9_20260927...")
frozen_dir = ROOT / "deliverables/e0c803_selected_v9_20260927"
frozen_manifest = json.loads((frozen_dir / "SHA256SUMS.json").read_text(encoding="utf-8"))
frozen_checks = {}
for rel_path, expected_hash in frozen_manifest.items():
    actual_hash = sha256_file(frozen_dir / rel_path)
    frozen_checks[rel_path] = (actual_hash == expected_hash)
frozen_pass_count = sum(frozen_checks.values())
frozen_total = len(frozen_checks)
assert frozen_pass_count == frozen_total, f"Frozen baseline mismatch: {frozen_pass_count}/{frozen_total}"
print(f"✅ Frozen baseline intact: {frozen_pass_count}/{frozen_total} files match.")

print("2. Hashing touched production and test code...")
touched = [
    "core/dubbing_content_gate.py",
    "core/dubbing_quality.py",
    "core/all_whisper_methods/demucs_vl.py",
    "core/step12_merge_dub_to_vid.py",
    "core/step4_1_summarize.py",
    "core/translation_review.py",
    "tests/test_patch_audit.py",
    "tests/test_source_anomalies.py",
    "tests/test_dubbing_content_gate.py",
    "tests/test_translation_review.py",
]
code_hashes = {f: sha256_file(ROOT / f) for f in touched if (ROOT / f).exists()}

print("3. Running independent audit test suite tests/test_patch_audit.py...")
proc_audit = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_patch_audit.py", "-v"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    env={**os.environ, "PYTHONPATH": str(ROOT)},
)
(OUT / "pytest_audit_19_cases.txt").write_text(proc_audit.stdout + chr(10) + proc_audit.stderr, encoding="utf-8")
assert proc_audit.returncode == 0, f"Audit tests failed:\n{proc_audit.stdout}"
audit_summary_line = proc_audit.stdout.strip().splitlines()[-1] if proc_audit.stdout else ""
print(f"✅ tests/test_patch_audit.py passed: {audit_summary_line}")

print("4. Running full pytest suite tests/...")
proc_full = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    env={**os.environ, "PYTHONPATH": str(ROOT)},
)
(OUT / "pytest_full_suite.txt").write_text(proc_full.stdout + chr(10) + proc_full.stderr, encoding="utf-8")
assert proc_full.returncode == 0, f"Full test suite failed:\n{proc_full.stdout}"
full_summary_line = proc_full.stdout.strip().splitlines()[-1] if proc_full.stdout else ""
print(f"✅ Full test suite passed: {full_summary_line}")

print("5. Running production media verification using core/step12_merge_dub_to_vid.py...")
media_dir = OUT / "production_media_verification"
media_dir.mkdir(parents=True, exist_ok=True)

from core import step12_merge_dub_to_vid as merge

mix_results = {}
configs = [
    ("no_preserve_no_duck", False, False, 2),
    ("no_preserve_duck", False, True, 2),
    ("preserve_no_duck", True, False, 2),
    ("preserve_duck", True, True, 2),
    ("preserve_duck_short_dialogue_3s", True, True, 3),
]

for name, preserve, duck, total_sec in configs:
    case_dir = media_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    cwd_orig = os.getcwd()
    try:
        os.chdir(case_dir)
        Path("output/audio").mkdir(parents=True, exist_ok=True)
        sr = 48000
        t = np.arange(sr * total_sec) / sr
        stereo = lambda v: np.column_stack([v, v])
        sf.write("output/audio/background.wav", stereo(0.7 * np.sin(2 * np.pi * 440 * t)), sr)
        dub_sec = 1 if "short_dialogue" in name else total_sec
        t_dub = np.arange(sr * dub_sec) / sr
        sf.write("output/dub.wav", stereo(0.1 * np.sin(2 * np.pi * 660 * t_dub)), sr)
        sf.write("output/audio/ctx.wav", np.zeros((sr * total_sec, 2)), sr)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi",
            "-i", f"color=c=black:s=160x90:r=25:d={total_sec}",
            "-c:v", "libx264", "output/source.mp4"
        ], check=True)

        merge.DUB_AUDIO = "output/dub.wav"
        merge.load_key = lambda k, default=None: {
            "audio_mix.preserve_original_non_speech": preserve,
            "audio_mix.ducking": duck,
            "burn_subtitles": False
        }.get(k, default)
        merge.prepare_audio_context_tracks = lambda *a: ("output/audio/ctx.wav", "output/audio/background.wav")

        merge.merge_video_audio()

        master_wav = Path("output/audio/final_mix_master.wav")
        assert master_wav.exists(), f"final_mix_master.wav missing for {name}"
        info = sf.info(str(master_wav))
        expected_frames = sr * total_sec
        assert info.frames == expected_frames, f"{name}: expected {expected_frames} frames, got {info.frames}"

        stats_json = json.loads(Path("output/audio/final_mix_stats.json").read_text(encoding="utf-8"))
        assert stats_json.get("gate_passed") is True, f"{name}: gate_passed is False"
        peak = stats_json.get("true_peak_dBTP")
        assert peak is not None and peak <= -1.0, f"{name}: peak {peak} > -1.0"

        mix_results[name] = {
            "master_frames": info.frames,
            "expected_frames": expected_frames,
            "master_samplerate": info.samplerate,
            "decoded_aac_true_peak_dBTP": peak,
            "integrated_lufs": stats_json.get("integrated_lufs"),
            "loudness_range_lu": stats_json.get("loudness_range_lu"),
            "satisfies_true_peak_limit": (peak <= -1.0),
            "satisfies_duration_exact": (info.frames == expected_frames),
        }
        print(f"  Branch {name}: frames={info.frames} (exact match), peak={peak} dBTP, LUFS={stats_json.get('integrated_lufs')}")
    finally:
        os.chdir(cwd_orig)

evidence_data = {
    "date": "2026-09-27",
    "execution_agent": "google-antigravity/gemini-3.8-flash",
    "purpose": "Astra V9 patch 2 audit resolution, production code fixes, and thorough delivery verification",
    "frozen_baseline_verified": {
        "path": "deliverables/e0c803_selected_v9_20260927",
        "files_checked": frozen_total,
        "files_matching": frozen_pass_count,
        "status": "intact",
    },
    "code_hashes": code_hashes,
    "pytest_results": {
        "audit_tests_19_cases": audit_summary_line,
        "full_suite": full_summary_line,
    },
    "production_media_verification": mix_results,
    "goal_status": "active",
}
(OUT / "evidence.json").write_text(json.dumps(evidence_data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✅ Evidence saved to {OUT / 'evidence.json'}")
