"""Independent production-validator regressions, using real FFmpeg media."""
import subprocess
from pathlib import Path

import pytest

from core.step12_merge_dub_to_vid import validate_merged_video
from core import step12_merge_dub_to_vid as merge


def media(path, duration, audio_offset=0):
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
           f'color=c=black:s=160x90:r=25:d={duration}']
    if audio_offset:
        cmd += ['-itsoffset', str(audio_offset)]
    cmd += ['-f', 'lavfi', '-i', f'sine=frequency=440:sample_rate=48000:duration={duration}',
            '-c:v', 'libx264', '-c:a', 'aac', str(path)]
    subprocess.run(cmd, capture_output=True, check=True)


def test_rejects_matching_streams_that_are_both_shorter_than_source(tmp_path):
    source, output = tmp_path/'source.mp4', tmp_path/'output.mp4'
    media(source, 3)
    media(output, 2)
    with pytest.raises(RuntimeError):
        validate_merged_video(str(output), str(source))


def test_rejects_shifted_audio_even_when_stream_durations_match(tmp_path):
    source, output = tmp_path/'source.mp4', tmp_path/'output.mp4'
    media(source, 3)
    media(output, 3, audio_offset=0.6)
    with pytest.raises(RuntimeError):
        validate_merged_video(str(output), str(source))


def test_render_rejects_unavailable_final_peak_measurement(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path('output/audio').mkdir(parents=True)
    media(Path('output/source.mp4'), 1)
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
