from pathlib import Path
import json, re, sys, hashlib, subprocess
import numpy as np
import soundfile as sf
import pandas as pd
ROOT=Path('/Users/vanch/VideoLingo-plus'); sys.path.insert(0,str(ROOT))
from core.fast_subtitles import _split_text_by_punctuation, _wrap_lines
W=ROOT/'run_workspaces/20261018-231604-5b8636'; D=ROOT/'docs/reports/fast-plan-audit-20261019'
m=json.loads((W/'output/log/fast_dubbing_manifest.json').read_text()); rows=m['tasks']
norm=lambda s: re.sub(r'\s+','',s)
loss=[]; caps=[]; physical=[]; duration_diffs=[]; fps=[]; manifests=[]; rendered_src=[]; rendered_tgt=[]
for r in rows:
    a=_split_text_by_punctuation(r['origin'],False); b=_split_text_by_punctuation(r['text'],True)
    pairs=list(zip(a,b)) if len(a)==len(b) and len(a)>1 else [(r['origin'],r['text'])]
    aa=[_wrap_lines(x,42,2,False) for x,y in pairs]; bb=[_wrap_lines(y,22,2,True) for x,y in pairs]
    rendered_src+=aa; rendered_tgt+=bb
    for lang,expected,actual,limit in [('en',r['origin'],' '.join(aa),42),('zh',r['text'],' '.join(bb),22)]:
        if norm(expected)!=norm(actual):loss.append(dict(number=r['number'],language=lang,expected=expected,actual=actual))
        for cue in (aa if lang=='en' else bb):
            if any(len(s)>limit for s in cue.splitlines()): caps.append(dict(number=r['number'],language=lang,text=cue))
    p=W/r['audio_path']; data,sr=sf.read(p); dur=len(data)/sr
    physical.append(dict(number=r['number'],path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),duration=dur,rms=float(np.sqrt(np.mean(data**2))),finite=bool(np.isfinite(data).all())))
    diff=dur-(r['final_end']-r['final_start'])
    if abs(diff)>0.02:duration_diffs.append(dict(number=r['number'],actual=dur,declared=r['final_end']-r['final_start'],difference=diff))
    ref=W/f"output/audio/refers/{r['number']}.wav"
    refhash=hashlib.sha256(ref.read_bytes()).hexdigest() if ref.is_file() else ''
    fp=hashlib.sha256(f"{r['spoken_text']}|{refhash}|indextts2_v25".encode()).hexdigest()
    if fp!=r.get('fingerprint'):fps.append(r['number'])
    manifests.append(dict(audio=str(p),language='zh'))
timeline=pd.read_excel(W/'output/final_timeline.xlsx').fillna('')
report=dict(units=len(rows),cues=len(timeline),max_speed=max(r['speed_factor'] for r in rows),extension=m['total_video_extension_seconds'],subtitle_losses=loss,line_cap_violations=caps,physical=physical,audio_duration_discrepancies=duration_diffs,fingerprint_mismatches=fps,subtitle_reproduction_matches_file=dict(en=norm(''.join(rendered_src))==norm(''.join(timeline.Source.astype(str))),zh=norm(''.join(rendered_tgt))==norm(''.join(timeline.Translation.astype(str)))))
report['media']={}
for name in ['AI配音.mp4','source.mp4']:
    p=W/'output'/name
    report['media'][name]=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name,codec_type,start_time,duration,width,height,r_frame_rate:format=duration','-of','json',str(p)]))
(D/'artifact-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(D/'asr-manifest.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['physical','subtitle_losses','line_cap_violations','audio_duration_discrepancies']},ensure_ascii=False,indent=2))
print('loss counts', {l:sum(x['language']==l for x in loss) for l in ['en','zh']},'line cap violations',len(caps),'duration mismatch >20ms',len(duration_diffs))
print(json.dumps(loss[:4],ensure_ascii=False,indent=2))
