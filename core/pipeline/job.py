"""Durable local execution shared by Streamlit and the public CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
import uuid
from contextlib import contextmanager

import fcntl

JOB_FILE = Path('output/pipeline_job.json')


def prepare_workspace(path: Path) -> Path:
    """Snapshot settings, reuse installed code/models, never reuse output."""
    from core.config_utils import CONFIG_PATH
    root = Path(__file__).resolve().parents[2]
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    shutil.copy2(CONFIG_PATH, path / 'config.yaml')
    for name in ('core', 'scripts', '_model_cache'):
        if (root / name).exists():
            (path / name).symlink_to(root / name, target_is_directory=True)
    if (root / 'custom_terms.xlsx').is_file():
        shutil.copy2(root / 'custom_terms.xlsx', path / 'custom_terms.xlsx')
    (path / 'output').mkdir()
    return path


@contextmanager
def workspace_lock(name='pipeline'):
    path = Path('output') / f'.{name}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('A pipeline is already running in this workspace') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def job_status():
    job = json.loads(JOB_FILE.read_text()) if JOB_FILE.is_file() else {}
    active = False
    if job.get('pid'):
        # Also reap a child started by this Streamlit process, so a zombie is
        # not mistaken for a still-running job.
        try:
            os.waitpid(int(job['pid']), os.WNOHANG)
        except ChildProcessError:
            pass
        try:
            os.kill(int(job['pid']), 0)
            active = True
        except ProcessLookupError:
            pass
    root_state = Path('output/pipeline_state.json')
    if root_state.is_file():
        state_path = root_state
    else:
        state_path = Path(job.get('workspace', '.')) / 'output/pipeline_state.json'
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    # Ensure workspace in job points to root if running in output
    if not job.get('workspace') or job.get('workspace') == '.':
        job['workspace'] = str(Path('.').resolve())
    return {**job, 'active': active, 'state': state}


def start_pipeline_job(input_value, *, profile=None, subtitle_only=False, no_subtitles=False, workspace=None):
    """Start exactly one normal CLI invocation; UI reruns cannot interrupt it."""
    from core.config_utils import load_key
    with workspace_lock('launch'):
        existing = job_status()
        if existing['active']:
            return existing
        root = Path(__file__).resolve().parents[2]
        run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
        if not str(input_value).startswith(('http://', 'https://')):
            input_value = str(Path(input_value).resolve())
        # Run directly in the root / output workspace unless the caller names another one.
        workspace = Path(workspace).resolve() if workspace else root.resolve()
        (workspace / 'output').mkdir(parents=True, exist_ok=True)
        w_state = workspace / 'output/pipeline_state.json'
        state = json.loads(w_state.read_text()) if w_state.is_file() else {}
        is_resume = workspace is not None and w_state.is_file() and state.get('status') in {'failed', 'interrupted', 'paused'}
        if is_resume and profile and state.get("profile") and profile != state["profile"]:
            raise ValueError(f"A resumable run cannot change its profile ({state['profile']} -> {profile}); start a fresh workspace")
        active_profile = profile or state.get("profile") or {"fast": "fast", "high_sync": "cinematic", "subtitle": "subtitle"}.get(
            load_key("quality_mode", "fast"), "fast")
        
        source, target = load_key('source_language', 'auto'), load_key('target_language', 'en')
        llm, tts = load_key('llm.provider', 'omlx'), load_key('tts_method', 'mlx_indextts2')

        command = [sys.executable, '-u', '-m', 'core.cli', 'run', '--input', str(input_value),
                   '--source', str(source),
                   '--target', str(target),
                   '--llm', str(llm),
                   '--tts', str(tts),
                   '--profile', active_profile, '--run-id', run_id]
        if subtitle_only:
            command.append('--subtitle-only')
        if no_subtitles:
            command.append('--no-subtitles')
        log = workspace / 'output' / f'pipeline-{run_id}.log'
        env = {**os.environ, 'PYTHONUNBUFFERED': '1',
               'PYTHONPATH': str(root) + os.pathsep + os.environ.get('PYTHONPATH', '')}
        with log.open('a') as handle:
            proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True, env=env, cwd=workspace)
        job = {'run_id': run_id, 'pid': proc.pid, 'input': str(input_value),
               'workspace': str(workspace), 'log': str(log), 'started_at': time.time(), 'command': command}
        temporary = JOB_FILE.with_suffix('.tmp')
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2))
        temporary.replace(JOB_FILE)
        return job
