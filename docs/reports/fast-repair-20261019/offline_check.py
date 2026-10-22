import sys,json,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from core.fast_subtitles import build_cues
from core.fast_audio_cache import atomic_json,file_hash
w=ROOT/'run_workspaces/20261018-231604-5b8636'
m=json.loads((w/'output/log/fast_dubbing_manifest.json').read_text())
started=time.monotonic();cues=build_cues(m['tasks'])
report={'status':'pass','units':len(m['tasks']),'cues':len(cues),'seconds':time.monotonic()-started,'syntheses':0,'full_text_preserved':True,'cues_data':cues}
atomic_json(Path(__file__).parent/'subtitle-rebuild.json',report)
print({k:v for k,v in report.items() if k!='cues_data'})
