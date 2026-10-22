"""Explicit full/sample ASR screening. Text discrepancies are not proof of wrong speech."""
from __future__ import annotations
import argparse
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
import soundfile as sf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def screen(item, transcript):
    rec = dict(item, transcript=transcript)
    if not isinstance(transcript, str) or not transcript.strip():
        return dict(rec, status='error', reason='missing_or_empty_readback', similarity=None)
    norm = lambda s: ''.join(c for c in s.casefold() if c.isalnum())
    expected, actual = norm(item['expected_text']), norm(transcript)
    if not expected:
        return dict(rec, status='error', reason='missing_expected_text', similarity=None)
    score = SequenceMatcher(None, expected, actual).ratio()
    # Report all nontrivial differences, even when the aggregate score is high.
    try:
        from pypinyin import lazy_pinyin, Style
        phonetic_equal = lazy_pinyin(expected, style=Style.TONE3) == lazy_pinyin(actual, style=Style.TONE3)
    except Exception:
        phonetic_equal = False
    reason = 'exact_text' if expected == actual else ('homophone_transcription' if phonetic_equal else 'text_discrepancy')
    return dict(rec, status='pass' if expected == actual else 'needs_review',
                reason=reason, similarity=score)


def inspect(workspace, transcript_rows, scope='full', sample_size=60):
    workspace = Path(workspace).resolve()
    tasks = pd.read_excel(workspace/'output/audio/tts_tasks.xlsx').to_dict('records')
    index = {str(Path(r.get('audio', '')).resolve()):r for r in transcript_rows}
    records, physical = [], []
    selected = set(range(len(tasks))) if scope == 'full' else set(np.linspace(0,len(tasks)-1,min(sample_size,len(tasks)),dtype=int))
    for i,row in enumerate(tasks):
        path = workspace/f"output/audio/segs/{int(row['number'])}_0.wav"
        item = dict(number=int(row['number']), audio=str(path), expected_text=str(row.get('spoken_text',row['text'])))
        try:
            data,sr = sf.read(path)
            if not len(data) or not np.isfinite(data).all() or np.max(np.abs(data))<1e-4:
                raise ValueError('silent_or_invalid_audio')
            item.update(audio_sha256=sha(path),duration=len(data)/sr)
        except Exception as exc:
            physical.append(dict(item,error=str(exc)));continue
        if i not in selected: continue
        witness=index.get(str(path),{})
        # Previously stored transcripts must carry the hash from recognition time.
        if witness.get('audio_sha256') != item['audio_sha256']:
            records.append(dict(item,status='error',reason='unbound_or_changed_audio',similarity=None));continue
        records.append(screen(item,witness.get('text')))
    errors=sum(r['status']=='error' for r in records)
    reviews=sum(r['status']=='needs_review' for r in records)
    scores=[r['similarity'] for r in records if r.get('similarity') is not None]
    return dict(status='fail' if physical or errors else ('needs_review' if reviews else ('pass' if scope=='full' else 'sample_pass')),
        scope=scope,total=len(tasks),readback_count=len(records),physical_failures=physical,
        readback_errors=errors,needs_review=reviews,records=records,
        average_similarity=sum(scores)/len(scores) if scores else None,
        limitations=['ASR transcription discrepancies require adjudication; voice/emotion not assessed'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',required=True,type=Path)
    p.add_argument('--scope',choices=['full','sample'],default='full')
    p.add_argument('--sample-size',type=int,default=60)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--language',default='zh')
    p.add_argument('--transcripts',type=Path,help='Existing hash-bound recognition records')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--python',default='/Users/vanch/.codex/envs/bluestone-mlx-audio/bin/python')
    args=p.parse_args()
    if args.batch_size<1 or args.sample_size<1:p.error('Batch and sample sizes must be positive')
    if args.output.exists():p.error('Output already exists; choose a new report path')
    if args.transcripts:
        rows=json.loads(args.transcripts.read_text())
    else:
        tasks=pd.read_excel(args.workspace/'output/audio/tts_tasks.xlsx').to_dict('records')
        if args.scope=='sample':tasks=[tasks[i] for i in sorted(set(np.linspace(0,len(tasks)-1,min(args.sample_size,len(tasks)),dtype=int)))]
        rows=[]
        with tempfile.TemporaryDirectory(prefix='videolingo-readback-') as temp:
            for offset in range(0,len(tasks),args.batch_size):
                batch=[{'audio':str((args.workspace/f"output/audio/segs/{int(t['number'])}_0.wav").resolve()),'language':args.language} for t in tasks[offset:offset+args.batch_size]]
                hashes={r['audio']:sha(r['audio']) for r in batch if Path(r['audio']).is_file()}
                manifest=Path(temp)/'batch.json';manifest.write_text(json.dumps(batch))
                try:
                    proc=subprocess.run([args.python,str(ROOT/'scripts/qwen3_asr_readback.py'),'--manifest',str(manifest)],capture_output=True,text=True,timeout=600,check=True)
                    payload=next(json.loads(line) for line in reversed(proc.stdout.splitlines()) if line.startswith('[') and line.endswith(']'))
                    rows.extend(dict(r,audio_sha256=hashes.get(r.get('audio'))) for r in payload)
                except Exception as exc:
                    rows.extend(dict(r,text=None,audio_sha256=hashes.get(r['audio']),error=str(exc)) for r in batch)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.with_suffix('.readbacks.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    report=inspect(args.workspace,rows,args.scope,args.sample_size)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in {'records','physical_failures'}},ensure_ascii=False,indent=2))
    return 0 if report['status'] in {'pass','sample_pass'} else 2

if __name__=='__main__':raise SystemExit(main())
