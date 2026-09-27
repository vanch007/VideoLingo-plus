"""Independent contract counterexamples, using invented dialogue and probabilities."""
import numpy as np
import pandas as pd
import pytest
import json
from types import SimpleNamespace

from core import translation_review
from core.providers.speaker_diarization import align_speakers_to_words


@pytest.mark.parametrize('final', [
    'You paid me seven dollars.',
    'I paid you seventy dollars.',
    'I did not pay you seven dollars.',
])
def test_final_semantic_audit_cannot_accept_changed_facts(tmp_path, monkeypatch, final):
    monkeypatch.setattr(translation_review, 'load_key', lambda key, default=None: 'en' if key == 'target_language' else default)
    tr = tmp_path / 'translation.xlsx'
    pd.DataFrame([{'LineID':'L00001','Translation':'I paid you seven dollars.'}]).to_excel(tr,index=False)
    tasks = pd.DataFrame([{'number':1,'line_id':'L00001','origin':'我付了你七美元。','text':final,'speaker':'S01'}])
    report = translation_review.audit_final_tts_tasks(tasks, tmp_path / 'audit.json', str(tr))
    assert report['status'] != 'pass', 'Language validity cannot prove preservation of actors, numbers or negation'


def test_missing_translation_evidence_is_not_semantic_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(translation_review, 'load_key', lambda key, default=None: 'en' if key == 'target_language' else default)
    tasks = pd.DataFrame([{'number':1,'line_id':'L00001','origin':'停下。','text':'Continue.','speaker':'S01'}])
    report = translation_review.audit_final_tts_tasks(tasks, tmp_path / 'audit.json', str(tmp_path/'missing.xlsx'))
    assert report['status'] != 'pass'


def diarize(tmp_path, first_probs):
    probs = np.tile([0.1, 0.95], (150, 1))
    probs[10:30] = first_probs
    p = tmp_path/'probs.npz'
    np.savez(p, probs=probs, frame_stride=0.01)
    data = {'segments':[{'words':[
        {'word':'No!','start':0.1,'end':0.3},
        {'word':'Listen.','start':0.3,'end':1.3},
    ]}]}
    align_speakers_to_words(data,[{'start':0.1,'end':0.3,'speaker':'S01'},{'start':0.3,'end':1.3,'speaker':'S02'}], probabilities_path=str(p),backend_name='nemotron-mlx')
    return data['segments'][0]['words'][0]


def test_zero_gap_must_not_move_confident_short_interjection_to_next_speaker(tmp_path):
    word = diarize(tmp_path, [0.95,0.41])
    assert word['speaker']=='S01', 'Temporal adjacency is not identity evidence; existing high-margin identity was overwritten'


def test_zero_gap_must_not_resolve_tied_identity_without_independent_evidence(tmp_path):
    word = diarize(tmp_path, [0.51,0.50])
    assert word['speaker'] is None, 'A next-speaker match cannot disambiguate a tied overlapping word'


def test_failed_source_probe_must_not_accept_media(tmp_path, monkeypatch):
    from core import step12_merge_dub_to_vid as merge
    out, src = tmp_path/'out.mp4', tmp_path/'source.mp4'
    out.write_bytes(b'fixture'); src.write_bytes(b'fixture')
    info = {'format':{'duration':'3','start_time':'0'}, 'streams':[
        {'codec_type':'video','duration':'3','start_time':'0','avg_frame_rate':'25/1'},
        {'codec_type':'audio','duration':'3','start_time':'0','sample_rate':'48000'},
    ]}
    def probe(cmd, **kwargs):
        if str(src) == cmd[-1]:
            return SimpleNamespace(returncode=1,stdout='',stderr='source probe failed')
        return SimpleNamespace(returncode=0,stdout=json.dumps(info),stderr='')
    monkeypatch.setattr(merge.subprocess,'run',probe)
    with pytest.raises(RuntimeError):
        merge.validate_merged_video(str(out),str(src))


def test_missing_frame_rate_is_not_verified_frame_tolerance(tmp_path, monkeypatch):
    from core import step12_merge_dub_to_vid as merge
    out=tmp_path/'out.mp4';out.write_bytes(b'fixture')
    info={'format':{'duration':'3','start_time':'0'},'streams':[
        {'codec_type':'video','duration':'3','start_time':'0'},
        {'codec_type':'audio','duration':'3','start_time':'0.035','sample_rate':'48000'},
    ]}
    monkeypatch.setattr(merge.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout=json.dumps(info),stderr=''))
    with pytest.raises(RuntimeError):
        merge.validate_merged_video(str(out))
