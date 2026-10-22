"""Fast export: one source clock, complete audio, one final video encode."""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from core.config_utils import load_key
from core.step12_merge_dub_to_vid import get_ffmpeg_binary
from core.fast_audio_cache import atomic_json, file_hash, valid_audio
from core.fast_timing import probe_media, source_clock

MANIFEST_PATH = Path('output/log/fast_dubbing_manifest.json')
FINAL_MP4 = Path('output/AI配音.mp4')
TEMP_DIR = Path('output/_temp_fast_merge')


def _get_video_duration(path):
    return float(probe_media(path)['format']['duration'])


def build_full_dubbing_track(tasks, total_duration_seconds):
    if not tasks or total_duration_seconds <= 0:
        raise ValueError('Missing speech manifest or nonpositive master duration')
    total_ms = math.ceil(total_duration_seconds*1000)
    master = AudioSegment.silent(duration=total_ms, frame_rate=24000)
    for row in tasks:
        path = Path(row['audio_path'])
        if not valid_audio(path): raise ValueError(f'Missing/invalid dubbing segment: {path}')
        if row.get('audio_sha256') and file_hash(path)!=row['audio_sha256']:
            raise ValueError(f'Audio changed after timing plan: {path}')
        seg = AudioSegment.from_file(path)
        start_ms = round(row['final_start']*1000)
        if start_ms<0 or start_ms+len(seg)>total_ms+1:
            raise ValueError(f'Complete speech does not fit master: {path}')
        master=master.overlay(seg,position=start_ms)
    path=Path('output/audio/dub.wav');path.parent.mkdir(parents=True,exist_ok=True)
    master.export(path,format='wav')
    return path


def _background_source(tasks):
    bg=Path('output/audio/background.wav')
    if not bg.is_file(): raise FileNotFoundError('Separated lossless background.wav is required')
    # Default to pure background music to ensure 100% clean dubbing without original vocal bleed
    if not load_key('audio_mix.preserve_original_non_speech', False): return bg
    if not original.is_file(): raise FileNotFoundError('raw_master.wav is required to preserve non-speech audio')
    raw,sr=sf.read(original,always_2d=True,dtype='float32')
    back,bsr=sf.read(bg,always_2d=True,dtype='float32')
    if bsr!=sr:
        from scipy.signal import resample_poly
        from math import gcd
        g=gcd(sr,bsr);back=resample_poly(back,sr//g,bsr//g,axis=0)
    if back.shape[1]!=raw.shape[1]:
        if back.shape[1]==1: back=np.repeat(back,raw.shape[1],axis=1)
        else: back=back.mean(axis=1,keepdims=True);back=np.repeat(back,raw.shape[1],axis=1)
    if len(back)<len(raw): back=np.pad(back,((0,len(raw)-len(back)),(0,0)))
    back=back[:len(raw)]
    # Replace recognized dialogue windows only; laughter/ambience outside survives.
    mask=np.zeros(len(raw),dtype='float32')
    for r in tasks:
        a=max(0,round(r['orig_start']*sr));b=min(len(raw),math.ceil((r['orig_end']+r['borrow_gap'])*sr))
        mask[a:b]=1.
    mixed=raw*(1-mask[:,None])+back*mask[:,None]
    result=TEMP_DIR/'background_source.wav';sf.write(result,mixed,sr)
    return result


def merge_fast_dubbing_video():
    src=Path('output/source_canonical.mp4')
    if not src.is_file(): src=Path('output/source.mp4')
    source_duration,fps=source_clock(src)
    manifest=json.loads(MANIFEST_PATH.read_text())
    if manifest.get('policy_version')!='fast-v2': raise ValueError('Rebuild the Fast v2 audio/timing plan before rendering')
    tasks,extensions=manifest['tasks'],manifest['extensions']
    extension=sum(e['extend_by'] for e in extensions)
    frames=math.ceil((source_duration+extension)*fps-1e-7)
    target=frames/fps
    speech=build_full_dubbing_track(tasks,target)
    TEMP_DIR.mkdir(parents=True,exist_ok=True)
    bg=_background_source(tasks)
    from core.audio_speed import build_atempo_filter
    # Segment audio on the exact source clock, then fit each declared mapped interval.
    slices=[];cursor=0.
    for e in extensions:
        s,t=e['orig_start'],e['orig_end']
        if s<cursor-.0001: raise ValueError('Overlapping extension intervals')
        if s>cursor: slices.append((cursor,s,0.))
        slices.append((s,t,e['extend_by']));cursor=t
    if cursor<source_duration: slices.append((cursor,source_duration,0.))
    filters=[]
    for i,(s,e,extra) in enumerate(slices):
        span=e-s;out=span+extra
        filters.append(f'[2:a]atrim=start={s:.9f}:end={e:.9f},asetpts=PTS-STARTPTS,'
                       f'{build_atempo_filter(span/out)},apad,atrim=duration={out:.9f}[bg{i}]')
    filters.append(''.join(f'[bg{i}]' for i in range(len(slices)))+f'concat=n={len(slices)}:v=0:a=1[background]')
        # ── Render Audio Master purely from speech and background (100% no original voice bleed) ──
    speech_gain = float(load_key('audio_mix.speech_gain_db', 2.5))
    bg_vol = float(load_key('audio_mix.background_volume', 0.5))
    mixed_audio_file = TEMP_DIR / 'final_dubbed_audio.wav'
    audio_cmd = [
        get_ffmpeg_binary(), '-y', '-v', 'error',
        '-i', str(speech),
        '-i', str(bg),
        '-filter_complex', f'[0:a]volume={speech_gain}dB[sp];[1:a]volume={bg_vol}[bg_proc];[sp][bg_proc]amix=inputs=2:normalize=0[aout]',
        '-map', '[aout]',
        str(mixed_audio_file)
    ]
    subprocess.run(audio_cmd, check=True)
    expr='PTS-STARTPTS'
    for e in extensions:
        span=e['orig_end']-e['orig_start']
        expr+=f"+min(max(T-{e['orig_start']:.9f},0),{span:.9f})*{e['extend_by']/span:.12f}/TB"
    vf=f"setpts='{expr}',tpad=stop_mode=clone:stop_duration=1,fps={fps:.12f},trim=end_frame={frames},setpts=PTS-STARTPTS"
    if load_key('burn_subtitles',True):
        target_sub = 'output/dub.srt'
        if not Path(target_sub).is_file(): raise FileNotFoundError(f'Target subtitle missing: {target_sub}')
        vf+=f",subtitles=filename='{target_sub}':force_style='FontSize=20,FontName=Arial,Outline=1,BorderStyle=1,BackColour=&H00000000,Alignment=2,MarginV=18'"
    partial=FINAL_MP4.with_name('AI配音.partial.mp4')
    video_cmd = [
        get_ffmpeg_binary(), '-y', '-v', 'error',
        '-i', str(src),
        '-i', str(mixed_audio_file),
        '-filter_complex', f'[0:v]{vf}[v]',
        '-map', '[v]',
        '-map', '1:a:0',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '19',
        '-c:a', 'aac', '-b:a', '192k',
        str(partial)
    ]
    subprocess.run(video_cmd, check=True)
    report=_validate_media(partial,target,fps)
    partial.replace(FINAL_MP4)
    atomic_json('output/log/fast_render_clock.json',dict(report,source_duration=source_duration,
        extension_seconds=extension,frames=frames,target_seconds=target,fps=fps,
        video_sha256=file_hash(FINAL_MP4),manifest_sha256=file_hash(MANIFEST_PATH)))
    validate_fast_delivery()


def _validate_media(path,target,fps):
    probe=probe_media(path)
    video=next((s for s in probe['streams'] if s['codec_type']=='video'),None)
    audio=next((s for s in probe['streams'] if s['codec_type']=='audio'),None)
    if video is None or audio is None: raise ValueError('Missing media stream')
    vd,ad=float(video['duration']),float(audio['duration'])
    packet=1024/float(audio['sample_rate'])
    v_tol = max(2/fps, 0.12); a_tol = max(packet + 0.001, 0.05)
    if abs(vd-target)>v_tol or abs(ad-target)>a_tol:
        raise ValueError(f'Incomplete media clocks: expected {target}, video {vd}, audio {ad}')
    if abs(float(video.get('start_time',0)))>1/fps or abs(float(audio.get('start_time',0)))>packet:
        raise ValueError('Nonzero media stream onset')
    return dict(status='pass',video_seconds=vd,audio_seconds=ad)


def validate_fast_delivery():
    from core.fast_subtitles import build_cues
    manifest=json.loads(MANIFEST_PATH.read_text());tasks=manifest['tasks']
    if manifest.get('policy_version')!='fast-v2': raise ValueError('Old fast policy requires actual reprocessing')
    if len(tasks)!=manifest['units'] or not tasks: raise ValueError('Incomplete task manifest')
    expected=build_cues(tasks)
    coverage=json.loads(Path('output/log/fast_subtitle_coverage.json').read_text())
    if coverage.get('cues')!=expected: raise ValueError('Stale subtitle coverage report')
    for filename,key in [('src.srt','src'),('dub.srt','tgt')]:
        from core.timing_utils import seconds_to_srt_time
        wanted=''.join(f"{c['index']}\n{seconds_to_srt_time(c['start'])} --> {seconds_to_srt_time(c['end'])}\n{c[key]}\n\n" for c in expected)
        if Path('output',filename).read_text()!=wanted: raise ValueError(f'Subtitle content/timing changed: {filename}')
    for r in tasks:
        if not valid_audio(r['audio_path']) or file_hash(r['audio_path'])!=r['audio_sha256']:
            raise ValueError(f'Invalid/stale segment: {r["number"]}')
        if not 1.<=r['speed_factor']<=1.25: raise ValueError('Speed cap exceeded')
        if not 0.<=r['borrow_gap']<=.350001: raise ValueError('Pause borrowing cap exceeded')
        info=sf.info(r['audio_path'])
        if abs(info.duration-(r['final_end']-r['final_start']))>1/info.samplerate+1e-7:
            raise ValueError('Subtitle clock not based on actual speech samples')
    clock=json.loads(Path('output/log/fast_render_clock.json').read_text())
    if clock['manifest_sha256']!=file_hash(MANIFEST_PATH) or clock['video_sha256']!=file_hash(FINAL_MP4):
        raise ValueError('Stale final delivery')
    _validate_media(FINAL_MP4,clock['target_seconds'],clock['fps'])
    report=dict(status='fast_basic_pass',strict_quality_gate='not_run',units=len(tasks),subtitle_cues=len(expected),
        checks=['audio_files','subtitle_coverage','display_limits','actual_timing','media_streams'],
        not_run=['deep_semantics','asr_readback','perceptual_voice_emotion'],
        manifest_sha256=file_hash(MANIFEST_PATH),video_sha256=file_hash(FINAL_MP4))
    atomic_json('output/log/fast_delivery_validation.json',report)
    return report
