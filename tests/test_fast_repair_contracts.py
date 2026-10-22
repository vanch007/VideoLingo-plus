from pathlib import Path
from contextlib import nullcontext
import json
import re
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import soundfile as sf
import pytest
from core.fast_subtitles import build_cues
from core.fast_audio_cache import AudioLedger, atomic_json, digest
from core.fast_timing import quiet_borrow, plan_audio_timeline, time_map
from core.fast_text import validate_fast_text
from core.providers.contracts import TTSRequest


def wav(path, duration=1.):
    sf.write(path,np.sin(np.arange(round(duration*24000))*.1)*.1,24000)
    return path


def test_subtitles_all_text_and_latin_words_survive():
    source="Everything you're seeing in this video was edited by my Claude Code system while I was asleep."
    target='我想在这个系统里用hyperframes、FFmpeg、TELLA和Epidemic sounds。'
    cues=build_cues([dict(number=1,unit_id='u',origin=source,text=target,final_start=0.,final_end=4.)])
    norm=lambda s:re.sub(r'\s+','',s)
    assert norm(''.join(c['src'] for c in cues))==norm(source)
    assert norm(''.join(c['tgt'] for c in cues))==norm(target)
    assert all(len(l)<=42 for c in cues for l in c['src'].splitlines())
    assert all(len(l)<=22 for c in cues for l in c['tgt'].splitlines())
    assert any('FFmpeg' in c['tgt'] for c in cues)
    assert cues[-1]['end']==4.
    assert all(a['end']==b['start'] for a,b in zip(cues,cues[1:]))


def test_long_token_is_error_not_truncated():
    with pytest.raises(ValueError,match='Unbreakable'):
        build_cues([dict(origin='x'*50,text='测试',final_start=0,final_end=1)])


def test_ledger_reuse_identity_audio_and_persistent_retry_budget(tmp_path):
    p=wav(tmp_path/'a.wav');req=TTSRequest(text='hello',output_path=str(p))
    ledger=AudioLedger(tmp_path/'cache.json');identity={'text':'hello','reference':'one','model':'v25'}
    ledger.put('u',req,identity)
    assert AudioLedger(ledger.path).reusable('u',identity)
    assert not ledger.reusable('u',{**identity,'reference':'changed'})
    assert not ledger.reusable('u',{**identity,'model':'changed'})
    sf.write(p,np.zeros(24000),24000)
    assert not ledger.reusable('u',identity)
    ledger.reserve('u',identity);ledger.reserve('u',identity)
    with pytest.raises(RuntimeError,match='exhausted'): AudioLedger(ledger.path).reserve('u',identity)


def test_quiet_gap_requires_measured_silence(tmp_path):
    p=tmp_path/'stem.wav';data=np.zeros(48000);data[30000:]=.1
    sf.write(p,data,24000)
    assert quiet_borrow(1.,1.5,2.,p)==pytest.approx(.25)
    assert quiet_borrow(1.3,1.5,2.,p)==0
    assert quiet_borrow(1.,1.05,2.,p)<=.050001
    assert quiet_borrow(1.,1.5,2.,tmp_path/'missing')==0


def test_actual_duration_multiple_extensions_shift_all_clocks(tmp_path):
    p=wav(tmp_path/'a.wav',1.15);q=wav(tmp_path/'b.wav',2.8)
    rows=[dict(number=1,orig_start=1.,orig_end=2.,borrow_gap=0.,audio_path=str(p)),
          dict(number=2,orig_start=3.,orig_end=4.,borrow_gap=.1,audio_path=str(q))]
    ext=plan_audio_timeline(rows,5.,30.)
    assert rows[0]['final_end']==pytest.approx(2.15)
    assert rows[1]['final_start']==pytest.approx(time_map(3.,ext))
    assert len(ext)==2
    assert time_map(5.,ext)==pytest.approx(5+sum(e['extend_by'] for e in ext))
    assert all(abs(e['extend_by']*30-round(e['extend_by']*30))<1e-8 for e in ext)


def test_master_missing_audio_must_fail(tmp_path,monkeypatch):
    from core.fast_video_merge import build_full_dubbing_track
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError): build_full_dubbing_track([],1.)
    with pytest.raises(ValueError): build_full_dubbing_track([dict(audio_path='absent.wav',final_start=0.)],1.)


@pytest.mark.parametrize('text',['',None,float('nan'),'NaN','初稿が来たら、'])
def test_basic_invalid_text_never_falls_back_to_source(text):
    with pytest.raises(ValueError): validate_fast_text('When the draft arrives',text,'zh',[])


def test_web_fast_profile_and_resume_are_explicit(tmp_path,monkeypatch):
    import core.pipeline.job as job
    monkeypatch.setattr(job,'workspace_lock',lambda *a:nullcontext())
    monkeypatch.setattr(job,'job_status',lambda:{'active':False})
    w=tmp_path/'workspace';w.mkdir();(w/'output').mkdir();(w/'config.yaml').write_text('source_language: en\ntarget_language: zh\n')
    monkeypatch.setattr(job,'prepare_workspace',lambda p:w)
    monkeypatch.setattr(job,'JOB_FILE',tmp_path/'job.json')
    monkeypatch.setattr(job.subprocess,'Popen',lambda *a,**kw:SimpleNamespace(pid=99))
    cmd=job.start_pipeline_job('test.mp4',profile='fast')['command']
    assert cmd[cmd.index('--profile')+1]=='fast'
    atomic_json(w/'output/pipeline_state.json',{'profile':'fast','source':'en','target':'zh'})
    cmd=job.start_pipeline_job('test.mp4',workspace=w)['command']
    assert cmd[cmd.index('--profile')+1]=='fast'
    with pytest.raises(ValueError,match='cannot change'):
        job.start_pipeline_job('test.mp4',profile='cinematic',workspace=w)


def test_fast_strict_overlay_preserves_user_model(monkeypatch):
    from core.pipeline import profiles
    state={'quality_mode':'high_sync','mlx_tts.backends.indextts2.profile':'standard','translation.strict_review_gate':True,'translation.review_mode':'audit_only'}
    monkeypatch.setattr(profiles,'load_key',lambda k,d=None:state.get(k,d))
    monkeypatch.setattr(profiles,'update_key',lambda k,v:state.update({k:v}))
    profiles.apply_fast_profile();profiles.apply_cinematic_profile()
    assert state['translation.strict_review_gate'] is True
    assert state['translation.review_mode']=='audit_only'
    assert state['mlx_tts.backends.indextts2.profile']=='standard'
    profiles.apply_fast_profile();assert state['mlx_tts.backends.indextts2.profile']=='v25'


def test_fast_route_does_not_encode_original_subtitle_video():
    from core.pipeline.runner import build_steps_for_input
    keys=[s.key for s in build_steps_for_input('input.mp4',profile='fast')]
    assert 'merge_subtitle' not in keys
    assert 'gen_dub_chunks' not in keys
    assert keys[-1]=='merge_video'


def test_empty_asr_and_high_average_never_hide_failure(tmp_path):
    from scripts.verify_audio_asr_acceptance import screen,inspect,sha
    for value in (None,'',{},'   '):
        assert screen({'expected_text':'你好'},value)['status']=='error'
    p=tmp_path/'output/audio/segs';p.mkdir(parents=True)
    import pandas as pd
    rows=[];witnesses=[]
    for n in (1,2):
        audio=wav(p/f'{n}_0.wav');rows.append({'number':n,'text':'你好'})
        witnesses.append({'audio':str(audio),'audio_sha256':sha(audio),'text':'你好' if n==1 else None})
    pd.DataFrame(rows).to_excel(tmp_path/'output/audio/tts_tasks.xlsx',index=False)
    report=inspect(tmp_path,witnesses)
    assert report['status']=='fail'
    assert report['readback_errors']==1
    witnesses[1]['text']='你好';witnesses[1]['audio_sha256']='stale'
    assert inspect(tmp_path,witnesses)['status']=='fail'
