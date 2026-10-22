import os
import sys
import json
import subprocess
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

PYTHON_BIN = "/Users/vanch/.cache/ocr-recognition/venv-mlx-vlm-0.6.6/bin/python"
WORKER_SCRIPT = str(Path(__file__).resolve().parent.parent.parent / "scripts" / "hard_subtitle_ocr_worker.py")
DEFAULT_MODEL_DIR = "/Users/vanch/.cache/huggingface/hub/models--AutomatosX--AX-Unlimited-OCR-3B-MoE-MLX-MXFP8/snapshots/95689e561e6442f1db6d418b5b534f5425358900"

def get_local_model_dir() -> str:
    if os.path.exists(DEFAULT_MODEL_DIR):
        return DEFAULT_MODEL_DIR
    matches = glob.glob("/Users/vanch/.cache/huggingface/hub/models--AutomatosX--AX-Unlimited-OCR-3B-MoE-MLX-MXFP8/snapshots/*")
    if matches:
        return matches[0]
    return DEFAULT_MODEL_DIR

class HardSubtitleOCRClient:
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or get_local_model_dir()
        self.proc: Optional[subprocess.Popen] = None

    def start(self):
        if self.proc is not None:
            return
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        
        cmd = [PYTHON_BIN, WORKER_SCRIPT, "--model", self.model_dir]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )
        # Wait for ready signal
        while True:
            line = self.proc.stdout.readline()
            if not line:
                stderr = self.proc.stderr.read()
                raise RuntimeError(f"OCR Worker failed to start: {stderr}")
            try:
                msg = json.loads(line.strip())
                if msg.get("status") == "ready":
                    break
            except Exception:
                continue

    def recognize(self, image_path: str, task: str = "boxes", prompt: Optional[str] = None) -> Dict[str, Any]:
        if self.proc is None:
            self.start()
        
        req = {
            "image_path": str(Path(image_path).resolve()),
            "task": task,
            "prompt": prompt
        }
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        
        while True:
            line = self.proc.stdout.readline()
            if not line:
                stderr = self.proc.stderr.read()
                raise RuntimeError(f"OCR Worker terminated unexpectedly: {stderr}")
            try:
                res = json.loads(line.strip())
                if "status" in res:
                    return res
            except Exception:
                continue

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.write(json.dumps({"action": "stop"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            finally:
                self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

