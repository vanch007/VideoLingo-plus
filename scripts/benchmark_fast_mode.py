"""Benchmark & verification script for Fast Mode on 15s excerpt.
Runs pipeline under --profile fast in an isolated workspace,
measures timing, requests, dubbing synthesis, and subtitle generation.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import pandas as pd
from rich import print as rprint

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLE_VIDEO = ROOT / "run_workspaces/test_fast_smoke/sample15s.mp4"
WORKSPACE = ROOT / "run_workspaces/benchmark_fast_run_20261018"

if WORKSPACE.exists():
    shutil.rmtree(WORKSPACE)

# Copy config and mock necessary symlinks
from core.pipeline.job import prepare_workspace
prepare_workspace(WORKSPACE)

# Ensure sample video exists
if not SAMPLE_VIDEO.is_file():
    raise FileNotFoundError(f"Missing sample video: {SAMPLE_VIDEO}")

rprint(f"[bold cyan]🚀 Running Fast Mode pipeline benchmark on {SAMPLE_VIDEO}...[/bold cyan]")
start_time = time.time()

cmd = [
    "conda", "run", "--no-capture-output", "-n", "videolingo",
    "python", "-m", "core.cli", "run",
    "--input", str(SAMPLE_VIDEO.resolve()),
    "--profile", "fast",
    "--source", "en",
    "--target", "zh",
    "--run-id", "fast_smoke_001"
]

log_file = WORKSPACE / "output/fast_run.log"
with log_file.open("w", encoding="utf-8") as lf:
    proc = subprocess.run(cmd, cwd=WORKSPACE, stdout=lf, stderr=subprocess.STDOUT)

elapsed = time.time() - start_time
rprint(f"[bold cyan]⏱️ Pipeline process finished with exit code {proc.returncode} in {elapsed:.2f}s[/bold cyan]")

# Output inspections
out_dir = WORKSPACE / "output"
final_video = out_dir / "AI配音.mp4"
bilingual_srt = out_dir / "dub_bilingual.srt"
manifest_file = out_dir / "log/fast_dubbing_manifest.json"
timeline_file = out_dir / "final_timeline.xlsx"

success = (proc.returncode == 0) and final_video.is_file() and bilingual_srt.is_file()
report = {
    "status": "pass" if success else "failed",
    "exit_code": proc.returncode,
    "total_elapsed_seconds": round(elapsed, 2),
    "final_video_exists": final_video.is_file(),
    "final_video_size_bytes": final_video.stat().st_size if final_video.is_file() else 0,
    "bilingual_srt_exists": bilingual_srt.is_file(),
    "bilingual_srt_lines": len(bilingual_srt.read_text(encoding="utf-8").splitlines()) if bilingual_srt.is_file() else 0,
    "manifest_exists": manifest_file.is_file(),
    "timeline_exists": timeline_file.is_file(),
}

if manifest_file.is_file():
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    report["dubbed_units"] = manifest_data.get("units", 0)
    report["total_video_extension_seconds"] = manifest_data.get("total_video_extension_seconds", 0.0)
    report["extended_segments_count"] = manifest_data.get("extended_segments_count", 0)

# Check LLM request logs
request_events = out_dir / "gpt_log/request_events.jsonl"
if request_events.is_file():
    lines = [json.loads(l) for l in request_events.read_text(encoding="utf-8").splitlines() if l.strip()]
    report["llm_requests_count"] = len(lines)
else:
    report["llm_requests_count"] = 0

print(json.dumps(report, ensure_ascii=False, indent=2))

report_save = ROOT / "docs/reports/fast-mode-benchmark-report.json"
report_save.parent.mkdir(parents=True, exist_ok=True)
report_save.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
rprint(f"[bold green]Report saved to {report_save}[/bold green]")
