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
    assert not Path('output/source.mkv').exists()  # single video copy: the transcoded source.mp4 replaces it
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


def test_background_job_uses_normal_cli_in_the_root_output_workspace(tmp_path, monkeypatch):
    from core.pipeline import job
    from core import config_utils
    monkeypatch.chdir(tmp_path)
    config = tmp_path / 'config.yaml'
    config.write_text('source_language: en\ntarget_language: zh\n')
    monkeypatch.setattr(config_utils, 'CONFIG_PATH', str(config))
    monkeypatch.setattr(job, 'job_status', lambda: {'active': False})
    monkeypatch.setattr(job, 'JOB_FILE', tmp_path / 'job.json')
    monkeypatch.setattr(job, 'ROOT', tmp_path, raising=False)
    calls = []
    def start(command, **kwargs):
        calls.append((command, kwargs))
        return type('Process', (), {'pid': 12345})()
    monkeypatch.setattr(job.subprocess, 'Popen', start)
    info = job.start_pipeline_job('https://www.youtube.com/watch?v=fixture')
    assert calls[0][0][1:5] == ['-u', '-m', 'core.cli', 'run']
    assert Path(info['workspace']) == Path(calls[0][1]['cwd'])
    assert (Path(info['workspace']) / 'output').is_dir()


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


def test_code_stale_text_checkpoint_is_archived_for_a_real_rerun(tmp_path, monkeypatch):
    from core.pipeline import runner
    from core.pipeline.artifact_manifest import build_artifact_manifest, write_artifact_manifest
    monkeypatch.chdir(tmp_path)
    code = Path('summary-code.py')
    code.write_text('old policy')
    checkpoint = Path('output/log/terminology.json')
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('old terminology')
    write_artifact_manifest(build_artifact_manifest(artifact=str(checkpoint), step='summarize',
        run_context={'run_id': 'old'}, code_paths=[str(code)]))
    code.write_text('current policy')
    assert runner._recover_code_stale_text_checkpoint('summarize', str(checkpoint), PipelineRun('new', 'cinematic', 'en', 'zh', []))
    # The stale output is preserved in history and the producer is free to run again.
    assert not checkpoint.exists()
    assert (Path('output/history/code_stale_checkpoints/new/summarize') / checkpoint.name).read_text() == 'old terminology'


@pytest.mark.parametrize('change', ['input', 'artifact', 'config', 'dependency'])
def test_code_stale_recovery_refuses_changed_data_or_settings(tmp_path, monkeypatch, change):
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
    # Only a code-only change may be recovered; changed data or settings must never be hidden.
    assert not runner._recover_code_stale_text_checkpoint('summarize', str(checkpoint), PipelineRun('new', 'cinematic', 'en', 'zh', []))
    assert checkpoint.exists() and not Path('output/history').exists()


def test_display_alignment_not_expected_when_cues_are_preserved(monkeypatch):
    from core import bilingual_alignment as ba
    values = {"translation.preserve_cues": True}
    monkeypatch.setattr("core.config_utils.load_key", lambda key, default=None: values.get(key, default))
    monkeypatch.setattr("core.llm_provider.get_translation_strategy", lambda config=None: "localized")
    assert ba.display_alignment_expected() is False
    monkeypatch.setattr("core.llm_provider.get_translation_strategy", lambda config=None: "literal")
    assert ba.display_alignment_expected() is True


def test_download_retries_without_cookies_when_browser_store_is_missing(monkeypatch, tmp_path):
    import yt_dlp
    from core import step1_ytdlp

    calls = []

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def download(self, urls):
            calls.append("cookiesfrombrowser" in self.opts)
            if "cookiesfrombrowser" in self.opts:
                raise yt_dlp.utils.DownloadError('ERROR: could not find chrome cookies database in "/nowhere"')

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(step1_ytdlp, "ensure_canonical_source_video", lambda save_path="output": None)
    monkeypatch.setenv("VIDEOLINGO_YTDLP_COOKIES_FROM_BROWSER", "chrome")
    step1_ytdlp.download_video_ytdlp("https://www.youtube.com/watch?v=abc", save_path=str(tmp_path))
    assert calls == [True, False]


def test_resume_clears_only_the_partial_artifact_of_the_interrupted_step(tmp_path, monkeypatch):
    from core.pipeline import runner
    monkeypatch.chdir(tmp_path)
    artifact = Path('output/log/cleaned_chunks.xlsx')
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'partial')
    Path('output/pipeline_state.json').write_text(json.dumps(
        {'run_id': 'old', 'status': 'failed', 'failed_step': 'transcribe', 'steps': ['transcribe'], 'completed_steps': []}))
    seen = []
    monkeypatch.setattr(runner, '_record_step_manifest', lambda *_: None)
    run = run_pipeline(PipelineRun('new', 'cinematic', 'en', 'zh', []),
                       [PipelineStep('transcribe', 'ASR', action=lambda: seen.append('ran'))])
    assert seen == ['ran'] and run.completed_steps == ['transcribe']
    assert not artifact.exists()  # the unproven partial output was cleared, not reused


def test_resume_refuses_an_unproven_checkpoint_when_no_run_was_interrupted_there(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact = Path('output/log/cleaned_chunks.xlsx')
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'left over from another video')
    with pytest.raises(RuntimeError, match='Stale transcribe checkpoint'):
        run_pipeline(PipelineRun('new', 'cinematic', 'en', 'zh', []),
                     [PipelineStep('transcribe', 'ASR', action=lambda: pytest.fail('must not run'))])
    assert artifact.read_bytes() == b'left over from another video'


def test_a_step_rerun_upstream_rechecks_downstream_checkpoints(tmp_path, monkeypatch):
    from core.pipeline import runner
    import core.step_checker as checker
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, 'pending_steps', lambda keys: ['timeline'])
    monkeypatch.setattr(runner, '_record_step_manifest', lambda *_: None)
    # After timeline ran, merge_subtitle's manifest no longer matches its inputs; render_extra still does.
    monkeypatch.setattr(checker, 'is_step_completed', lambda key: key != 'merge_subtitle')
    ran = []
    steps = [PipelineStep(k, k, action=lambda k=k: ran.append(k)) for k in ('translate', 'timeline', 'merge_subtitle', 'render_extra')]
    run = run_pipeline(PipelineRun('r', 'subtitle', 'zh', 'en', []), steps)
    assert ran == ['timeline', 'merge_subtitle']
    assert run.completed_steps == ['translate', 'timeline', 'merge_subtitle', 'render_extra']
