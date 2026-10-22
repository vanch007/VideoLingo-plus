"""Fast Mode: High-performance batch cloning dubbing, dynamic volume alignment & timing adaptation.

Performance optimization:
- Single-process resident batch synthesis: all pending speech chunks are synthesized
  in one single IndexTTS2 session, keeping the 2.5B model resident and GPU fully utilized.
- Bounded 1.25x speed limit with 0.35s quiet gap borrowing.
- Batch compact rewrite retry on overlong dialogue, then batch resynthesis.
- Local video extension (time stretching) if still overlong, preventing audio truncation.
- Spoken name transliteration with original English names preserved for subtitles.
"""
from __future__ import annotations
from core.step12_merge_dub_to_vid import get_ffmpeg_binary

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shutil

import numpy as np
import pandas as pd
from pydub import AudioSegment
from rich import print as rprint
import soundfile as sf

from core.config_utils import load_key
from core.timing_utils import seconds_to_srt_time, srt_time_to_seconds
from core.all_whisper_methods.audio_preprocess import get_audio_duration
from core.tts_pronunciation import spoken_plan


MANIFEST_PATH = Path("output/log/fast_dubbing_manifest.json")
SEGS_DIR = Path("output/audio/segs")
REFERS_DIR = Path("output/audio/refers")
TTS_TASKS_FILE = Path("output/audio/tts_tasks.xlsx")


def _audio_fingerprint(text: str, ref_hash: str, model: str) -> str:
    raw = f"{text}|{ref_hash}|{model}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_valid_wav(path: Path | str) -> bool:
    p = Path(path)
    if not p.is_file() or p.stat().st_size < 1000:
        return False
    try:
        data, sr = sf.read(str(p))
        if len(data) == 0 or np.all(np.abs(data) < 1e-4):
            return False
        return True
    except Exception:
        return False


def _adjust_volume_to_reference(audio_path: str | Path, ref_path: str | Path) -> None:
    """Align loudness/RMS to the reference voice to maintain volume consistency."""
    try:
        dub = AudioSegment.from_file(str(audio_path))
        ref = AudioSegment.from_file(str(ref_path))
        if len(dub) == 0 or len(ref) == 0:
            return
        dub_rms = dub.rms
        ref_rms = ref.rms
        if dub_rms <= 0 or ref_rms <= 0:
            return
        diff_db = 20 * math.log10(ref_rms / dub_rms)
        diff_db = max(-12.0, min(8.0, diff_db, -1.0 - dub.max_dBFS))
        adjusted = dub.apply_gain(diff_db)
        adjusted.export(str(audio_path), format="wav")
    except Exception as exc:
        rprint(f"[yellow]⚠️ Volume alignment skipped for {audio_path}: {exc}[/yellow]")


def _apply_atempo(in_wav: str | Path, out_wav: str | Path, speed_factor: float) -> None:
    """Speed up audio while preserving pitch using ffmpeg atempo."""
    p_in, p_out = Path(in_wav).resolve(), Path(out_wav).resolve()
    if abs(speed_factor - 1.0) < 0.005:
        if p_in != p_out:
            shutil.copy2(p_in, p_out)
        return
    factors = []
    curr = float(speed_factor)
    while curr > 2.0:
        factors.append(2.0)
        curr /= 2.0
    while curr < 0.5:
        factors.append(0.5)
        curr /= 0.5
    factors.append(curr)
    filter_chain = ",".join(f"atempo={f:.4f}" for f in factors)
    tmp_out = p_out.with_suffix(".tmp.wav")
    cmd = [
        get_ffmpeg_binary(), "-y", "-loglevel", "error",
        "-i", str(p_in),
        "-filter:a", filter_chain,
        "-c:a", "pcm_s16le",
        str(tmp_out)
    ]
    subprocess.run(cmd, check=True)
    tmp_out.replace(p_out)


def _compact_rewrite_single_line(original: str, current_translation: str, target_seconds: float, language=None, terms=None) -> str:
    from core.ask_gpt import ask_gpt
    from core.runtime_context import effective_target_language
    from core.fast_text import validate_fast_text
    language = language or effective_target_language("zh")
    prompt = f"""Shorten this complete dubbing sentence in target language {language} to fit {target_seconds:.2f}s.
Preserve the meaning, facts and exact original display names. Return only the shorter sentence.
Source: {original}
Current: {current_translation}"""
    try:
        text = ask_gpt(prompt, response_json=False, max_retries=1, log_title="fast_dubbing_compact_rewrite")
        text = validate_fast_text(original, text, language, terms)
        return text if len(text) < len(current_translation) else current_translation
    except Exception as exc:
        rprint(f"[yellow]Compact candidate not used: {exc}[/yellow]")
        return current_translation


def run_fast_dubbing() -> None:
    from dataclasses import replace
    import time
    from core.utterance_plan import plan_active, load_plan, tasks_from_plan
    from core.all_tts_functions.mlx_router import build_mlx_tts_request
    from core.providers.mlx_tts import synthesize_indextts2_batch
    from core.runtime_context import effective_target_language
    from core.fast_audio_cache import AudioLedger, atomic_json, file_hash, digest, model_identity, request_identity, valid_audio
    from core.fast_text import validate_fast_text
    from core.fast_timing import source_clock, quiet_borrow, plan_audio_timeline
    from core.tts_reference_plan import refresh_conditioning_receipt_audio
    started = time.monotonic()
    SEGS_DIR.mkdir(parents=True, exist_ok=True)
    tasks_df = tasks_from_plan(load_plan()) if plan_active() else pd.read_excel(TTS_TASKS_FILE)
    tasks = tasks_df.to_dict('records')
    if not tasks: raise ValueError('No dubbing units')
    language = effective_target_language(load_key('target_language', 'zh'))
    terms_file = Path('output/log/terminology.json')
    terms = json.loads(terms_file.read_text()).get('terms', []) if terms_file.is_file() else []
    source = Path('output/source_canonical.mp4')
    if not source.is_file(): source = Path('output/source.mp4')
    source_duration, fps = source_clock(source)
    cfg = load_key('mlx_tts.backends.indextts2', {})
    model = model_identity(cfg)
    ledger = AudioLedger()
    old = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.is_file() else {}
    previous = {r['unit_id']:r for r in old.get('tasks', [])}
    old_policy = old.get('policy_version')
    stats = dict(total=len(tasks), generated=0, reused=0, failed=0, batches=0, elapsed_seconds=0.)

    def progress():
        stats['elapsed_seconds'] = round(time.monotonic()-started, 3)
        atomic_json('output/log/fast_progress.json', dict(stats, stage='dubbing'))
        rprint(f"Fast voices: generated={stats['generated']} reused={stats['reused']} failed={stats['failed']} / {stats['total']}")

    items = []
    for i,row in enumerate(tasks):
        number = int(row['number']); uid = str(row.get('unit_id', f'U{number:04d}'))
        source_text = str(row.get('origin', '')); base_text = str(row['text'])
        prior = previous.get(uid, {})
        # A saved compact draft is usable only for the same original plan text.
        compact = ledger.data['rewrites'].get(uid, {})
        target = compact.get('text', base_text) if compact.get('base_text') == base_text else base_text
        legacy = False
        if old_policy is None and prior.get('origin') == source_text and prior.get('unit_id') == uid:
            target = prior.get('text', base_text); legacy = True
        validate_fast_text(source_text, target, language, terms)
        spoken = spoken_plan(target, language, terms)['spoken_text']
        start = srt_time_to_seconds(str(row['start_time'])); end = srt_time_to_seconds(str(row['end_time']))
        next_start = srt_time_to_seconds(str(tasks[i+1]['start_time'])) if i+1<len(tasks) else source_duration
        borrow = quiet_borrow(end, next_start, source_duration, 'output/audio/vocal.wav')
        budget = end-start+borrow
        if budget <= 0: raise ValueError(f'Invalid speech budget: {uid}')
        raw = SEGS_DIR/f'{number}_raw.wav'
        request = build_mlx_tts_request(spoken, str(raw.resolve()), number, tasks_df,
                    task_row={**row,'text':spoken,'lines':[spoken],'disable_native_fit':True}, target_duration=budget)
        identity = request_identity(request,model)
        item = dict(row=row, uid=uid, number=number, origin=source_text, base_text=base_text,
                    text=target, spoken=spoken, start=start, end=end, borrow=borrow, budget=budget,
                    request=request, identity=identity, raw=raw)
        # Legacy migration is explicit and evidence-bound, not a fabricated v2 synthesis receipt.
        # Native WAV must match its original conditioning receipt and the actual spoken draft.
        if legacy and not ledger.reusable(uid,identity):
            receipt_path = raw.with_suffix('.conditioning.json')
            if not receipt_path.is_file():
                raise RuntimeError(f'Legacy conditioning receipt missing: {uid}; no synthesis attempted')
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                native = raw.with_name(raw.stem+'.native.wav')
                rp = receipt.get('reference_plan', {}); desired = request.metadata['reference_plan']
                matches = (receipt.get('text') == spoken and receipt.get('language') == language
                    and rp.get('backend_config_fingerprint') == desired.get('backend_config_fingerprint')
                    and all((rp.get(k) or {}).get('sha256') == (desired.get(k) or {}).get('sha256') for k in ['speaker_reference','emotion_reference'])
                    and rp.get('emo_alpha') == desired.get('emo_alpha')
                    and valid_audio(native) and file_hash(native) == receipt.get('native_audio',{}).get('sha256'))
                if matches:
                    # Read immutable native media; do not overwrite original audio or its receipt.
                    request = replace(request, output_path=str(native.resolve()))
                    item['request'], item['raw'] = request, native
                    ledger.put(uid,request,identity, provenance='legacy_conditioning_and_native_hash',
                               historical_weight_hash='missing_evidence')
                    if target != base_text:
                        ledger.data['rewrites'][uid] = {'base_text':base_text,'text':target,'attempted':True}
                        ledger.save()
                else:
                    raise RuntimeError(f'Legacy audio identity cannot be verified: {uid}; retained audio was not regenerated')
        if ledger.reusable(uid,identity):
            item['raw'] = Path(ledger.data['entries'][uid]['path'])
            stats['reused'] += 1
        items.append(item)
    progress()

    def generate(pending):
        for attempt in range(2):
            pending = [x for x in pending if not ledger.reusable(x['uid'],x['identity'])]
            if not pending: return
            for x in pending: ledger.reserve(x['uid'],x['identity'])
            by_path = {x['request'].output_path:x for x in pending}
            def completed(request):
                x = by_path[request.output_path]
                ledger.put(x['uid'], request, x['identity'])
                stats['generated'] += 1
                progress()
            stats['batches'] += 1
            try:
                from core.providers.mlx_tts import synthesize_indextts2_batch
                synthesize_indextts2_batch([x['request'] for x in pending], on_result=completed)
            except Exception as exc:
                atomic_json('output/log/fast_batch_error.json', {'error':str(exc),'attempt':attempt+1})
            # The adapter may return success before callbacks in a test/alternate implementation.
            # Only its callback commits audio; a stale old file is never accepted here.
        unresolved = [x['uid'] for x in pending if not ledger.reusable(x['uid'],x['identity'])]
        if unresolved:
            stats['failed'] = len(unresolved); progress()
            raise RuntimeError(f'Fast synthesis failed after bounded repair: {unresolved}')

    generate([x for x in items if not ledger.reusable(x['uid'],x['identity'])])
    rewritten=[]
    for item in items:
        item['raw'] = Path(ledger.data['entries'][item['uid']]['path'])
        duration = sf.info(item['raw']).duration
        uid = item['uid']
        if duration/item['budget'] <= 1.25 or ledger.data['rewrites'].get(uid,{}).get('attempted'):
            continue
        ledger.data['rewrites'][uid] = dict(base_text=item['base_text'],text=item['text'],attempted=True)
        ledger.save()  # Never reset the rewrite budget on process restart.
        text = _compact_rewrite_single_line(item['origin'],item['text'],item['budget'],language,terms)
        if text == item['text']: continue
        spoken = spoken_plan(text,language,terms)['spoken_text']
        item['text'],item['spoken'] = text,spoken
        request = build_mlx_tts_request(spoken, str((SEGS_DIR/f"{item['number']}_compact.wav").resolve()),
            item['number'], tasks_df, task_row={**item['row'], 'text':spoken,'lines':[spoken],
            'disable_native_fit':True}, target_duration=item['budget'])
        item['request'],item['identity'] = request,request_identity(request,model)
        ledger.data['rewrites'][uid]['text'] = text; ledger.save()
        rewritten.append(item)
    generate(rewritten)
    from concurrent.futures import ThreadPoolExecutor

    def _process_single_item(item):
        entry = ledger.data['entries'][item['uid']]; raw = Path(entry['path'])
        processed = SEGS_DIR / f"{item['number']}_gain.wav"; final = SEGS_DIR / f"{item['number']}_0.wav"
        shutil.copy2(raw, processed)
        _adjust_volume_to_reference(processed, item['request'].ref_audio)
        duration = sf.info(processed).duration
        speed = min(1.25, max(1., duration / item['budget']))
        _apply_atempo(processed, final, speed)
        if not valid_audio(final):
            raise ValueError(f'Invalid processed audio: {final}')
        return dict(item['row'], number=item['number'], unit_id=item['uid'], origin=item['origin'],
            text=item['text'], spoken_text=item['spoken'], orig_start=item['start'], orig_end=item['end'],
            borrow_gap=item['borrow'], target_duration=item['budget'], raw_duration=duration, speed_factor=speed,
            audio_path=str(final), audio_sha256=file_hash(final), natural_sha256=entry['sha256'],
            fingerprint=entry['identity'], base_text=item['base_text'])

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
        results = list(executor.map(_process_single_item, items))
    for row in results:
        row['start_time']=seconds_to_srt_time(row['final_start']);row['end_time']=seconds_to_srt_time(row['final_end'])
    extension=sum(e['extend_by'] for e in extensions)
    manifest=dict(status='completed',mode='fast',policy_version='fast-v2',units=len(results),
        source_duration=source_duration,fps=fps,total_video_extension_seconds=extension,
        extended_segments_count=len(extensions),extensions=extensions,tasks=results,
        synthesis_statistics=stats,not_run=['deep_semantic_review','asr_readback','perceptual_quality'])
    atomic_json(MANIFEST_PATH,manifest)
    pd.DataFrame(results).to_excel(TTS_TASKS_FILE,index=False)
    from core.fast_subtitles import build_fast_subtitles
    build_fast_subtitles(results,extension)
    from core.fast_video_merge import build_full_dubbing_track
    build_full_dubbing_track(results,source_duration+extension)
    progress()
