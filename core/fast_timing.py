"""One continuous time mapping shared by Fast audio, captions and renderer."""
import json
import math
from pathlib import Path
import subprocess
from fractions import Fraction
import numpy as np
import soundfile as sf


def probe_media(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))


def source_clock(path):
    probe = probe_media(path)
    video = next(s for s in probe['streams'] if s['codec_type']=='video')
    fps = float(Fraction(video.get('avg_frame_rate') or video['r_frame_rate']))
    duration = max(float(probe['format']['duration']), *(float(s.get('duration', 0)) for s in probe['streams']))
    return duration, fps


def quiet_borrow(end, next_start, duration, stem, cap=.35):
    limit = min(.35, cap, max(0., next_start-end), max(0., duration-end))
    if limit <= 0 or not Path(stem).is_file(): return 0.
    with sf.SoundFile(stem) as f:
        sr = f.samplerate; f.seek(min(len(f), round(end*sr)))
        data = f.read(round(limit*sr), always_2d=True)
    # Conservative 10 ms windows: any vocal energy ends the usable pause.
    frame = max(1, round(sr*.01)); safe = 0
    for start in range(0,len(data),frame):
        block = data[start:start+frame]
        if np.sqrt(np.mean(block**2)) > 10**(-45/20): break
        safe += len(block)
    return min(limit, safe/sr)


def time_map(t, extensions):
    return t + sum(e['extend_by'] * min(1., max(0., (t-e['orig_start'])/(e['orig_end']-e['orig_start']))) for e in extensions)


def plan_audio_timeline(rows, source_duration, fps):
    extensions, cumulative, previous_end = [], 0., 0.
    for row in rows:
        s, e = row['orig_start'], row['orig_end'] + row['borrow_gap']
        if s < previous_end - .001: raise ValueError('Overlapping source dialogue windows require explicit handling')
        if e <= s or e > source_duration+.001: raise ValueError('Invalid source dialogue window')
        info = sf.info(row['audio_path']); actual = info.frames/info.samplerate
        extra = max(0., actual-(e-s))
        # One frame grid for all timeline consumers, based on measured WAV length.
        extra = math.ceil((extra-1e-9)*fps)/fps if extra > 1e-9 else 0.
        row['final_start'] = s + cumulative
        row['final_end'] = row['final_start'] + actual
        row['extend_by'] = extra
        row['actual_duration'] = actual
        if extra:
            extensions.append(dict(number=row['number'], orig_start=s, orig_end=e,
                                   extend_by=extra, extended_start=s+cumulative,
                                   extended_end=e+cumulative+extra))
            cumulative += extra
        previous_end = e
    return extensions
