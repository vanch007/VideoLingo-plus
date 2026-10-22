import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
w=ROOT/'run_workspaces/20261018-231604-5b8636';os.chdir(w)
from core.config_utils import load_key
from core.utterance_plan import plan_active,load_plan,tasks_from_plan
from core.all_tts_functions.mlx_router import build_mlx_tts_request
from core.tts_pronunciation import spoken_plan
from core.fast_audio_cache import file_hash,valid_audio,atomic_json
import pandas as pd
m=json.loads(Path('output/log/fast_dubbing_manifest.json').read_text())
terms=json.loads(Path('output/log/terminology.json').read_text()).get('terms',[])
df=tasks_from_plan(load_plan()) if plan_active() else pd.read_excel('output/audio/tts_tasks.xlsx')
rows=[]
for prior in m['tasks']:
 n=int(prior['number']);row=df[df.number==n].iloc[0].to_dict();spoken=spoken_plan(prior['text'],'zh',terms)['spoken_text']
 raw=Path(f'output/audio/segs/{n}_raw.wav');receipt=json.loads(raw.with_suffix('.conditioning.json').read_text());native=raw.with_name(raw.stem+'.native.wav')
 req=build_mlx_tts_request(spoken,str(raw.resolve()),n,df,task_row={**row,'text':spoken,'lines':[spoken],'disable_native_fit':True},target_duration=float(prior['target_duration']))
 rp=receipt.get('reference_plan',{});want=req.metadata['reference_plan']
 checks={'text':receipt.get('text')==spoken,'language':receipt.get('language')=='zh',
  'config':rp.get('backend_config_fingerprint')==want.get('backend_config_fingerprint'),
  'speaker':(rp.get('speaker_reference') or {}).get('sha256')==(want.get('speaker_reference') or {}).get('sha256'),
  'emotion':(rp.get('emotion_reference') or {}).get('sha256')==(want.get('emotion_reference') or {}).get('sha256'),
  'alpha':rp.get('emo_alpha')==want.get('emo_alpha'),
  'native':valid_audio(native) and file_hash(native)==receipt.get('native_audio',{}).get('sha256')}
 rows.append(dict(number=n,checks=checks))
report=dict(pass_count=sum(all(r['checks'].values()) for r in rows),total=len(rows),syntheses=0,records=rows)
atomic_json(ROOT/'docs/reports/fast-repair-20261019/reuse-check.json',report)
print({k:v for k,v in report.items() if k!='records'})
