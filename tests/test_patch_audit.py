"""Independent audit. All generated media/reports live in pytest tmp_path."""
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from core import dubbing_content_gate as gate
from core import dubbing_quality as quality
from core import step12_merge_dub_to_vid as merge
from core.all_whisper_methods import demucs_vl


def test_filler_wrong_lexical_readback_not_exempt():
    row = pd.Series(dict(text='Ah!', asr_transcript='Stop!', asr_content_score=0.0,
                         real_dur=0.3, available_duration=0.4))
    df = pd.DataFrame([row])
    assert gate._repair_indices(df, [0], []) == [0]


def test_filler_budget_is_not_measured_duration():
    row = pd.Series(dict(text='Ah!', asr_transcript='', asr_content_score=0.0,
                         available_duration=0.4))
    assert not gate._is_untranscribable_sub_500ms_utterance(row)


@pytest.mark.parametrize('native_exists', [False, True])
def test_native_layer_requires_native_measurement(monkeypatch, tmp_path, native_exists):
    sr = 16000
    def tone(seconds):
        return 0.1 * np.sin(2 * np.pi * 440 * np.arange(int(sr * seconds)) / sr)
    sf.write(tmp_path / '1_0.wav', np.r_[np.zeros(1600), tone(0.5)], sr)
    if native_exists:
        sf.write(tmp_path / '1_0.native.wav', np.r_[np.zeros(3200), tone(0.7)], sr)
    monkeypatch.setattr(quality, 'SEGS_DIR', str(tmp_path))
    tasks = pd.DataFrame([dict(number=1, start_time='00:00:00.000',
        end_time='00:00:01.000', text='Hello', lines=['Hello'],
        new_sub_times=[[0.0, 0.6]], duration=1., available_duration=1., real_dur=0.6)])
    result, _ = quality.evaluate_dubbing(tasks)
    layer = result.iloc[0]['four_layer_timing']['layer2_native_take']
    if native_exists:
        assert layer['duration'] == pytest.approx(0.9, abs=0.001)
    else:
        assert layer.get('onset') is None and layer.get('duration') is None


@pytest.mark.parametrize('peak', [0.995, 1.2])
def test_demucs_receipt_gain_matches_installed_transform(monkeypatch, tmp_path, peak):
    import demucs.audio
    import demucs.api
    import demucs.pretrained
    monkeypatch.chdir(tmp_path)
    Path('output/audio').mkdir(parents=True)
    Path('output/audio/raw_master.wav').write_bytes(b'fixture')
    model = SimpleNamespace(samplerate=16000, audio_channels=2)
    monkeypatch.setattr(demucs.pretrained, 'get_model', lambda _: model)
    monkeypatch.setattr(demucs.api.Separator, 'update_parameter', lambda *a, **kw: None)
    signal = torch.tensor([[peak, -peak] * 200] * 2, dtype=torch.float32)
    monkeypatch.setattr(demucs.api.Separator, 'separate_audio_file',
        lambda *a: (None, {'vocals': signal, 'other': signal / 2}))
    actual = {}
    def save(wav, path, **kw):
        out = demucs.audio.prevent_clip(wav, kw['clip'])
        actual[str(path)] = float(out.abs().max() / wav.abs().max())
        sf.write(path, out.numpy().T, kw['samplerate'], format='WAV')
    monkeypatch.setattr(demucs.audio, 'save_audio', save)
    demucs_vl.demucs_main()
    receipt = json.loads(Path('output/audio/demucs_receipt.json').read_text())
    assert receipt['gain_applied']['vocals_scale_applied'] == pytest.approx(
        actual['output/audio/vocal.wav'], abs=0.0001)


@pytest.mark.parametrize('preserve,duck', [(False,False),(False,True),(True,False),(True,True)])
def test_real_preencode_master_render(monkeypatch, tmp_path, preserve, duck):
    monkeypatch.chdir(tmp_path)
    Path('output/audio').mkdir(parents=True)
    sr = 48000
    t = np.arange(sr * 2) / sr
    stereo = lambda v: np.column_stack([v, v])
    sf.write('output/audio/background.wav', stereo(0.7*np.sin(2*np.pi*440*t)), sr)
    sf.write('output/dub.wav', stereo(0.1*np.sin(2*np.pi*660*t)), sr)
    sf.write('output/audio/ctx.wav', np.zeros((sr*2,2)), sr)
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',
        'color=c=black:s=160x90:r=25:d=2','-c:v','libx264','output/source.mp4'],check=True)
    monkeypatch.setattr(merge, 'DUB_AUDIO', 'output/dub.wav')
    monkeypatch.setattr(merge, 'load_key', lambda k, default=None: {
        'audio_mix.preserve_original_non_speech':preserve,
        'audio_mix.ducking':duck,'burn_subtitles':False}.get(k,default))
    monkeypatch.setattr(merge, 'prepare_audio_context_tracks',
        lambda *a: ('output/audio/ctx.wav','output/audio/background.wav'))
    merge.merge_video_audio()
    info = sf.info('output/audio/final_mix_master.wav')
    assert info.samplerate == 48000 and info.channels == 2 and info.frames == sr*2
    assert Path('output/AI配音.mp4').stat().st_size > 0


def test_source_vocal_activity_contributes_to_mask(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path('output/audio').mkdir(parents=True)
    sr = 16000
    voice = 0.2 * np.sin(2*np.pi*220*np.arange(sr)/sr)
    sf.write('output/audio/vocal.wav', np.r_[np.zeros(sr), voice, np.zeros(sr)], sr)
    windows = merge._collect_speech_mask_windows('absent.xlsx', 3000)
    assert any(start < 1500 < end for start, end in windows), windows


@pytest.mark.parametrize('preserve,duck', [(False,False),(False,True),(True,False),(True,True)])
def test_final_decoded_true_peak(monkeypatch, tmp_path, preserve, duck):
    test_real_preencode_master_render(monkeypatch, tmp_path, preserve, duck)
    result = subprocess.run(['ffmpeg','-hide_banner','-i','output/AI配音.mp4',
        '-af','ebur128=peak=true','-f','null','-'],capture_output=True,text=True,check=True)
    last = result.stderr[result.stderr.rfind('Summary:'):]
    peak = float(re.search(r'Peak:\s+(-?[\d.]+) dBFS',last).group(1))
    assert peak <= -1.0, last


@pytest.mark.parametrize('preserve', [False, True])
def test_ducking_preserves_background_after_short_dialogue(monkeypatch, tmp_path, preserve):
    monkeypatch.chdir(tmp_path)
    Path('output/audio').mkdir(parents=True)
    sr = 48000
    t = np.arange(sr * 3) / sr
    tone = lambda f: np.column_stack([0.1*np.sin(2*np.pi*f*t)] * 2)
    sf.write('output/audio/background.wav', tone(440), sr)
    sf.write('output/dub.wav', tone(660)[:sr], sr)
    sf.write('output/audio/ctx.wav', np.zeros((sr*3, 2)), sr)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'color=c=black:s=160x90:r=25:d=3', '-c:v', 'libx264',
                    'output/source.mp4'], check=True)
    monkeypatch.setattr(merge, 'DUB_AUDIO', 'output/dub.wav')
    monkeypatch.setattr(merge, 'load_key', lambda k, default=None: {
        'audio_mix.preserve_original_non_speech': preserve,
        'audio_mix.ducking': True, 'burn_subtitles': False}.get(k, default))
    monkeypatch.setattr(merge, 'prepare_audio_context_tracks',
                        lambda *a: ('output/audio/ctx.wav', 'output/audio/background.wav'))
    merge.merge_video_audio()
    info = sf.info('output/audio/final_mix_master.wav')
    assert info.frames == sr * 3
    stats = json.loads(Path('output/audio/final_mix_stats.json').read_text())
    assert stats['gate_passed'] is True
    assert stats['true_peak_dBTP'] <= -1.0


def test_validator_rejects_missing_audio_tail(tmp_path):
    video = tmp_path / 'truncated_audio.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
        'color=c=black:s=160x90:r=25:d=3', '-f', 'lavfi', '-i',
        'sine=frequency=440:sample_rate=48000:duration=1', '-c:v', 'libx264',
        '-c:a', 'aac', str(video)], check=True)
    with pytest.raises(RuntimeError):
        merge.validate_merged_video(str(video), str(video))


def test_demucs_cache_rejects_modified_output_and_model(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path('output/audio').mkdir(parents=True)
    source = Path('output/audio/raw_master.wav')
    source.write_bytes(b'unchanged source')
    Path('output/audio/vocal.wav').write_bytes(b'corrupted vocal')
    Path('output/audio/background.wav').write_bytes(b'corrupted background')
    Path('output/audio/demucs_receipt.json').write_text(json.dumps({
        'input_sha256': demucs_vl._hash_file(str(source)),
        'model': 'previous-model', 'vocal_sha256': 'old-hash',
        'background_sha256': 'old-hash', 'delay_ms': None,
        'delay_status': 'unknown',
    }))
    import demucs.pretrained
    class Recompute(Exception):
        pass
    def load_model(*args):
        raise Recompute()
    monkeypatch.setattr(demucs.pretrained, 'get_model', load_model)
    with pytest.raises(Recompute):
        demucs_vl.demucs_main('htdemucs')


def _make_dummy_media(path, duration, audio_offset=0):
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
           f'color=c=black:s=160x90:r=25:d={duration}']
    if audio_offset:
        cmd += ['-itsoffset', str(audio_offset)]
    cmd += ['-f', 'lavfi', '-i', f'sine=frequency=440:sample_rate=48000:duration={duration}',
            '-c:v', 'libx264', '-c:a', 'aac', str(path)]
    subprocess.run(cmd, capture_output=True, check=True)


def test_rejects_matching_streams_that_are_both_shorter_than_source(tmp_path):
    source, output = tmp_path/'source.mp4', tmp_path/'output.mp4'
    _make_dummy_media(source, 3)
    _make_dummy_media(output, 2)
    with pytest.raises(RuntimeError):
        merge.validate_merged_video(str(output), str(source))


def test_rejects_shifted_audio_even_when_stream_durations_match(tmp_path):
    source, output = tmp_path/'source.mp4', tmp_path/'output.mp4'
    _make_dummy_media(source, 3)
    _make_dummy_media(output, 3, audio_offset=0.6)
    with pytest.raises(RuntimeError):
        merge.validate_merged_video(str(output), str(source))


def test_render_rejects_unavailable_final_peak_measurement(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path('output/audio').mkdir(parents=True)
    _make_dummy_media(Path('output/source.mp4'), 1)
    for target in ['output/dub.wav', 'output/audio/background.wav']:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', 'output/source.mp4',
            '-vn', '-c:a', 'pcm_s16le', target], check=True, capture_output=True)
    monkeypatch.setattr(merge, 'DUB_AUDIO', 'output/dub.wav')
    monkeypatch.setattr(merge, 'load_key', lambda k, default=None: {
        'audio_mix.preserve_original_non_speech': False,
        'audio_mix.ducking': False, 'burn_subtitles': False}.get(k, default))
    monkeypatch.setattr(merge, 'measure_final_audio_loudness', lambda _: {
        'true_peak_dBTP': None, 'integrated_lufs': None, 'loudness_range_lu': None})
    with pytest.raises(RuntimeError):
        merge.merge_video_audio()
