"""Independent V11 counterexamples; no external dialogue or character knowledge."""
import json
import subprocess
from pathlib import Path
import pandas as pd
import pytest
from pydub import AudioSegment
from pydub.generators import Sine
from core import dubbing_quality as quality
from core import step12_merge_dub_to_vid as merge
from core.dubbing_content_gate import _is_untranscribable_sub_500ms_utterance


def media(path, duration, offset=0):
    cmd = ['ffmpeg','-v','error','-y','-f','lavfi','-i',f'color=s=160x90:r=25:d={duration}']
    if offset:
        cmd += ['-itsoffset',str(offset)]
    cmd += ['-f','lavfi','-i',f'sine=frequency=440:sample_rate=48000:duration={duration}', '-c:v','libx264','-c:a','aac',str(path)]
    subprocess.run(cmd,capture_output=True,check=True)


def test_name_hu_with_empty_readback_must_not_be_filler():
    row = pd.Series({'text':'Hu','origin':'胡','event_type':'speech','real_dur':0.239,'asr_transcript':'{"text":""}'})
    assert not _is_untranscribable_sub_500ms_utterance(row)


def test_acoustic_proxies_do_not_complete_independent_perceptual_measurement(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    segs=tmp_path/'segs';segs.mkdir()
    Sine(200).to_audio_segment(duration=2000).export(segs/'1_0.wav',format='wav')
    monkeypatch.setattr(quality,'SEGS_DIR',str(segs))
    monkeypatch.setattr(quality,'measure_voice_similarity',lambda *_:0.99)
    monkeypatch.setattr(quality,'measure_emotion_fidelity',lambda *_:0.8)
    row={'number':1,'start_time':'00:00:00.000','end_time':'00:00:02.000','duration':2.,'available_duration':2.,'tolerance':0.,'text':'Hello there','lines':['Hello there'],'new_sub_times':[[0.,2.]],'real_dur':2.,'resolved_speaker_ref':'speaker.wav','resolved_emotion_ref':'emotion.wav'}
    _,summary=quality.evaluate_dubbing(pd.DataFrame([row]))
    assert summary['perceptual_quality']['status']=='perceptual_pending'


def test_unknown_isolated_sound_cannot_be_asserted_nonverbal_without_evidence(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    Path('output/log').mkdir(parents=True);Path('output/audio').mkdir()
    pd.DataFrame([{'start':0.,'end':0.5}]).to_excel('output/log/cleaned_chunks.xlsx',index=False)
    vocal=AudioSegment.silent(duration=3000).overlay(Sine(440).to_audio_segment(duration=400),position=2000)
    vocal.export('output/audio/vocal.wav',format='wav')
    merge._collect_speech_mask_windows('missing.xlsx',3000)
    events=json.loads(Path('output/audio/nonverbal_events.json').read_text())
    assert events and all(e['event_type']=='uncertain' for e in events), 'Acoustic isolation alone is no speech/nonverbal classifier'


def test_two_video_frames_shift_is_rejected(tmp_path):
    source,out=tmp_path/'source.mp4',tmp_path/'out.mp4'
    media(source,3);media(out,3,offset=0.1)
    with pytest.raises(RuntimeError):
        merge.validate_merged_video(str(out),str(source))


def test_three_video_frames_truncation_is_rejected(tmp_path):
    source,out=tmp_path/'source.mp4',tmp_path/'out.mp4'
    media(source,3);media(out,2.88)
    with pytest.raises(RuntimeError):
        merge.validate_merged_video(str(out),str(source))


def test_tied_overlap_must_not_commit_single_identity(tmp_path):
    import numpy as np
    from core.providers.speaker_diarization import align_speakers_to_words
    p=tmp_path/'probs.npz'
    np.savez(p,probs=np.tile([0.51,0.50],(100,1)),frame_stride=0.01)
    result={'segments':[{'words':[{'word':'hello','start':0.1,'end':0.3}]}]}
    turns=[{'start':0.,'end':1.,'speaker':'S01'},{'start':0.,'end':1.,'speaker':'S02'}]
    out=align_speakers_to_words(result,turns,probabilities_path=str(p),backend_name='nemotron-mlx')
    word=result['segments'][0]['words'][0]
    assert word.get('speaker') is None, 'A tied overlap cannot authorize a single-speaker clone reference'
