"""Local, process-resident text inference without changing an oMLX server."""
from __future__ import annotations

import atexit
import json
from pathlib import Path
import queue
import subprocess
import threading
from types import SimpleNamespace

_CLIENTS = {}


class LocalMLXChatClient:
    def __init__(self, config):
        self.config = config
        self.chat = SimpleNamespace(completions=self)
        self.process = None
        self.inbox = queue.Queue()
        self.lock = threading.Lock()
        self.log = None
        atexit.register(self.close)

    def _read(self, process, inbox):
        for line in process.stdout:
            try:
                inbox.put(json.loads(line))
            except json.JSONDecodeError:
                inbox.put({"error":"Local MLX worker emitted an invalid protocol response"})
        inbox.put({"error":"Local MLX worker ended before completing its response"})

    def _receive(self, timeout):
        try:
            result = self.inbox.get(timeout=timeout)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError("Local MLX inference exceeded its configured timeout") from exc
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    def _start(self, timeout):
        model = Path(self.config.model_path or "").expanduser()
        python = Path(self.config.runtime_python or "").expanduser()
        if not model.is_dir() or not any(model.glob("*.safetensors")) or not (model/"config.json").is_file():
            raise FileNotFoundError(f"Local text editor weights are not installed: {model}")
        if not python.is_file():
            raise FileNotFoundError(f"Local MLX runtime is missing: {python}")
        worker = Path(__file__).resolve().parents[1]/"scripts/local_mlx_chat_worker.py"
        logs = Path("output/log"); logs.mkdir(parents=True,exist_ok=True)
        self.log = (logs/"local_mlx_editor_runtime.log").open("a",encoding="utf-8")
        self.inbox = queue.Queue()
        self.process = subprocess.Popen([str(python),"-u",str(worker),"--model-path",str(model)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        threading.Thread(target=self._read,args=(self.process,self.inbox),daemon=True).start()
        ready = self._receive(timeout)
        if ready.get("ready") is not True:
            self.close()
            raise RuntimeError("Local MLX worker did not confirm successful model loading")

    def create(self, *, timeout=None, **args):
        timeout = float(timeout or self.config.timeout_seconds or 300)
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self._start(timeout)
            self.process.stdin.write(json.dumps(args,ensure_ascii=False)+"\n")
            self.process.stdin.flush()
            result = self._receive(timeout)
        usage = result.get("usage",{})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=result["content"]),
            finish_reason=result["finish_reason"])],usage=SimpleNamespace(model_dump=lambda:usage))

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=3)
            self.process = None
        if self.log is not None:
            self.log.close();self.log = None


def get_local_mlx_client(config):
    key = (config.model,config.model_path,config.runtime_python)
    if key not in _CLIENTS:
        _CLIENTS[key] = LocalMLXChatClient(config)
    return _CLIENTS[key]
