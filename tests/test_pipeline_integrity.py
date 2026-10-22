import json
from pathlib import Path

import pytest

from core.pipeline.job import workspace_lock
from core.pipeline.runner import PipelineRun, PipelineStep, run_pipeline
from core.providers.quality import reference_leak_score


def test_normal_local_non_mp4_import_creates_the_canonical_export_source(tmp_path, monkeypatch):
    import subprocess
    from core.pipeline.runner import prepare_local_video
    from core.step1_ytdlp import find_video_files
    monkeypatch.chdir(tmp_path)
    original = tmp_path / 'input.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=s=64x64:d=1:r=25',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1',
                    '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(original)], check=True)
    result = prepare_local_video(str(original))
    assert result == 'output/source.mp4'
    assert Path(result).is_file() and original.is_file()
    assert Path('output/source.mkv').read_bytes() == original.read_bytes()
    assert find_video_files() == result
    streams = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', result]))['streams']
    assert {s['codec_type'] for s in streams} == {'video', 'audio'}


def test_canonical_import_does_not_choose_between_multiple_originals(tmp_path):
    from core.step1_ytdlp import ensure_canonical_source_video
    (tmp_path / 'source.webm').write_bytes(b'one')
    (tmp_path / 'another.mkv').write_bytes(b'two')
    with pytest.raises(ValueError, match='Ambiguous'):
        ensure_canonical_source_video(str(tmp_path))


def test_downloading_media_does_not_modify_the_python_environment(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from core import step1_ytdlp as download
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(download.subprocess, 'check_call', lambda *_: pytest.fail('pip ran inside a video job'))
    monkeypatch.setattr(download, 'ensure_canonical_source_video', lambda *_: 'output/source.mp4')
    class Downloader:
        def __init__(self, options):
            self.options = options
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def download(self, urls):
            assert urls == ['https://www.youtube.com/watch?v=fixture']
    monkeypatch.setitem(sys.modules, 'yt_dlp', SimpleNamespace(YoutubeDL=Downloader))
    download.download_video_ytdlp('https://www.youtube.com/watch?v=fixture')


def test_background_job_uses_normal_cli_and_an_isolated_workspace(tmp_path, monkeypatch):
    from core.pipeline import job
    from core import config_utils
    monkeypatch.chdir(tmp_path)
    config = tmp_path / 'config.yaml'
    config.write_text('source_language: en\ntarget_language: zh\n')
    monkeypatch.setattr(config_utils, 'CONFIG_PATH', str(config))
    monkeypatch.setattr(job, 'job_status', lambda: {'active': False})
    workspace = tmp_path / 'fresh'
    original = job.prepare_workspace
    monkeypatch.setattr(job, 'prepare_workspace', lambda _: original(workspace))
    calls = []
    def start(command, **kwargs):
        calls.append((command, kwargs))
        return type('Process', (), {'pid': 12345})()
    monkeypatch.setattr(job.subprocess, 'Popen', start)
    info = job.start_pipeline_job('https://www.youtube.com/watch?v=fixture')
    assert info['workspace'] == str(workspace)
    assert calls[0][0][1:5] == ['-u', '-m', 'core.cli', 'run']
    assert calls[0][1]['cwd'] == workspace
    assert (workspace / 'config.yaml').read_text() == config.read_text()
    assert not (workspace / 'output/pipeline_state.json').exists()
    with pytest.raises(FileExistsError):
        original(workspace)


def test_reference_leak_checks_extra_speech_not_approved_product_names():
    expected = '打开 Claude Code。'
    assert reference_leak_score('Open Claude Code.', expected, expected=expected) == 0
    assert reference_leak_score('secret instruction', '打开。secret instruction', expected='打开。') > .9
    assert reference_leak_score('打开', '打开打开', expected='打开') == 1


def test_workspace_excludes_duplicate_writers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with workspace_lock():
        with pytest.raises(RuntimeError, match='already running'):
            with workspace_lock():
                pytest.fail('two writers acquired the workspace')


def test_pipeline_failure_retains_exact_step_and_does_not_continue(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = []
    def fail():
        raise ValueError('source content mismatch')
    run = PipelineRun('run', 'cinematic', 'en', 'zh', [])
    with pytest.raises(ValueError, match='source content mismatch'):
        run_pipeline(run, [PipelineStep('audit', 'Audit', action=fail),
                           PipelineStep('export', 'Export', action=lambda: seen.append('export'))], resume=False)
    state = json.loads(Path('output/pipeline_state.json').read_text())
    assert state['failed_step'] == 'audit' and state['completed_steps'] == []
    assert state['events'][-1]['event'] == 'failed' and not seen


def test_interrupted_pipeline_cannot_leave_a_running_or_completed_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = []
    def interrupt():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_pipeline(PipelineRun('interrupted', 'cinematic', 'en', 'zh', []),
                     [PipelineStep('audit', 'Audit', action=interrupt),
                      PipelineStep('export', 'Export', action=lambda: seen.append('export'))], resume=False)
    state = json.loads(Path('output/pipeline_state.json').read_text())
    assert state['status'] == 'interrupted' and state['quality_gate_passed'] is False
    assert state['events'][-1]['event'] == 'interrupted' and state['completed_steps'] == []
    assert not seen


def test_pipeline_cannot_mark_manifest_failure_as_completed(tmp_path, monkeypatch):
    from core.pipeline import runner
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, '_record_step_manifest', lambda *_: (_ for _ in ()).throw(ValueError('missing evidence')))
    with pytest.raises(ValueError):
        run_pipeline(PipelineRun('run', 'cinematic', 'en', 'zh', []),
                     [PipelineStep('render', 'Render', action=lambda: None)], resume=False)
    assert json.loads(Path('output/pipeline_state.json').read_text())['completed_steps'] == []


def test_code_stale_text_checkpoint_is_archived_then_really_executed(tmp_path, monkeypatch):
    from core.pipeline import runner
    from core.pipeline.artifact_manifest import build_artifact_manifest, write_artifact_manifest
    monkeypatch.chdir(tmp_path)
    code = Path('summary-code.py')
    code.write_text('old policy')
    checkpoint = Path('output/log/terminology.json')
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('old terminology')
    manifest = build_artifact_manifest(artifact=str(checkpoint), step='summarize',
        run_context={'run_id': 'old'}, code_paths=[str(code)])
    write_artifact_manifest(manifest)
    code.write_text('current policy')
    calls = []
    def execute():
        assert not checkpoint.exists(), 'The legacy exists shortcut must not supply current acceptance'
        assert (Path('output/history/code_stale_checkpoints/new/summarize') / checkpoint.name).read_text() == 'old terminology'
        calls.append('executed')
        checkpoint.write_text('current terminology')
    monkeypatch.setattr(runner, '_record_step_manifest', lambda *_: None)
    run = run_pipeline(PipelineRun('new', 'cinematic', 'en', 'zh', []),
        [PipelineStep('summarize', 'Summarize', action=execute)])
    assert calls == ['executed'] and run.completed_steps == ['summarize']
    assert checkpoint.read_text() == 'current terminology'


@pytest.mark.parametrize('change', ['input', 'artifact', 'config', 'dependency'])
def test_code_stale_recovery_cannot_hide_changed_data_or_settings(tmp_path, monkeypatch, change):
    from core.pipeline import runner
    from core.pipeline.artifact_manifest import build_artifact_manifest, write_artifact_manifest, hash_path
    monkeypatch.chdir(tmp_path)
    files = {key: Path(key+'.txt') for key in ('input', 'config', 'dependency', 'code')}
    for path in files.values(): path.write_text('original')
    checkpoint = Path('output/log/terminology.json')
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('original')
    manifest = build_artifact_manifest(artifact=str(checkpoint), step='summarize',
        run_context={'input': str(files['input'])}, code_paths=[str(files['code'])],
        quality={'config_path': str(files['config']), 'config_hash': hash_path(files['config'])})
    manifest['dependencies'] = {str(files['dependency']): hash_path(files['dependency'])}
    write_artifact_manifest(manifest)
    files['code'].write_text('new policy')
    (checkpoint if change == 'artifact' else files[change]).write_text('changed')
    with pytest.raises(RuntimeError, match='Stale summarize checkpoint'):
        run_pipeline(PipelineRun('new', 'cinematic', 'en', 'zh', []),
            [PipelineStep('summarize', 'Summarize', action=lambda: pytest.fail('Must not execute'))])
    assert checkpoint.exists() and not Path('output/history').exists()
    state = json.loads(Path('output/pipeline_state.json').read_text())
    assert state['run_id'] == 'new' and state['status'] == 'failed' and state['failed_step'] == 'summarize'
