from pathlib import Path

import pandas as pd
import pytest

from core import source_verification, runtime_context


def test_witness_context_does_not_transcribe_the_next_groups_words(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio = Path('output/audio/raw_master.wav')
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b'fixture audio')
    options = {'source_quality.enabled': True, 'source_quality.model': 'witness',
               'whisper.runtime': 'local-aligned'}
    monkeypatch.setattr(source_verification, 'load_key', lambda key, default=None: options.get(key, default))
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda _: 'en')
    windows = []
    def cut(command, **kwargs):
        start = float(command[command.index('-ss') + 1])
        windows.append((start, start + float(command[command.index('-t') + 1])))
        Path(command[-1]).write_bytes(b'witness fixture')
    monkeypatch.setattr(source_verification.subprocess, 'run', cut)
    texts = iter(['First phrase.', 'Next phrase.'])
    monkeypatch.setattr(source_verification, 'transcribe_witness', lambda *_: next(texts))
    rows = pd.DataFrame([
        {'speaker': 'S01', 'line_id': 'L1', 'text': 'First phrase.', 'start': 1., 'end': 19.},
        {'speaker': 'S01', 'line_id': 'L2', 'text': 'Next phrase.', 'start': 19.1, 'end': 37.1},
    ])
    report = source_verification.verify_source_rows(rows)
    assert report['status'] == 'pass'
    assert windows[0][1] == pytest.approx(19.05)
    assert windows[1][0] == pytest.approx(19.05)
    assert windows[0][0] == pytest.approx(.75)
    assert windows[1][1] == pytest.approx(37.35)
