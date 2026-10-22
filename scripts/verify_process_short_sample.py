"""Run real semantic/speech/subtitle gates on source-bound movie excerpts.

Reuses explicitly bound ASR evidence for this short regression only. The full
acceptance must separately start through core.cli from the original video.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.pipeline.job import prepare_workspace
from core.pipeline.artifact_manifest import hash_path
from core.utterance_plan import save_plan, signature
import pandas as pd
from pydub import AudioSegment

BASE = Path(os.environ.get('VLO_SHORT_BASE',
    str(ROOT/'run_workspaces/HzXD4GVqXwM_en-zh_20261011_v7'))).resolve()
workspace = (Path(sys.argv[1]).resolve() if len(sys.argv)>1 else
    prepare_workspace(ROOT/'run_workspaces'/('HzXD4GVqXwM_contract_short_'+time.strftime('%Y%m%d_%H%M%S'))))
print('SHORT_WORKSPACE', workspace, flush=True)
workspace_note = Path(os.environ.get('VLO_SHORT_WORKSPACE_NOTE',
    str(ROOT/'docs/reports/process-repair-20261012/short-workspace.txt')))
workspace_note.write_text(str(workspace))
plan = json.loads((BASE/'output/log/utterance_plan.json').read_text())
chosen = {'L00019.01','L00027.01','L00032.01','L00034.01','L00117.01','L00157.01','L00228.01','L00236.01','L00320.01'}
if os.environ.get('VLO_SHORT_UNITS'):
    chosen = set(os.environ['VLO_SHORT_UNITS'].split(','))
units = [copy.deepcopy(u) for u in plan['units'] if u['unit_id'] in chosen]
assert len(units) == len(chosen)
if os.environ.get('VLO_SHORT_FAILED_DRAFTS') == '1':
    # Reproduce rejected current drafts, never change the full workspace or
    # import their failed audio as an accepted take.
    candidates = json.loads((BASE/'output/log/translation_speech_candidate_state.json').read_text())
    drafts = {item['unit_id']:item['target'] for item in candidates['candidates']}
    for unit in units:
        unit['target'] = drafts[unit['unit_id']]
original_words = pd.read_excel(BASE/'output/log/cleaned_chunks.xlsx')
vocal = AudioSegment.from_file(BASE/'output/audio/vocal.wav')
combined = vocal[:0]
words, filters, streams, cursor = [], [], [], 0.
parents = {}
for i,u in enumerate(units):
    left, right = max(0., u['start']-.2), u['end']+.2
    duration = round((right-left)*1000)/1000
    shift = cursor-left
    filters.extend([f'[0:v]trim=start={left}:duration={duration},setpts=PTS-STARTPTS[v{i}]',
        f'[0:a]atrim=start={left}:duration={duration},asetpts=PTS-STARTPTS[a{i}]'])
    streams.append(f'[v{i}][a{i}]')
    combined += vocal[round(left*1000):round((left+duration)*1000)]
    selected = original_words.iloc[u['word_start']:u['word_end']+1].copy()
    selected['start'] += shift
    selected['end'] += shift
    u['word_start'], u['word_end'] = len(words), len(words)+len(selected)-1
    words.extend(selected.to_dict('records'))
    for key in ['start','source_start','source_end','end']:
        u[key] = round(u[key]+shift, 6)
    if u.get('lead_pause_span'):
        u['lead_pause_span'] = [round(v+shift,6) for v in u['lead_pause_span']]
    u.pop('target_words', None)
    u.pop('source_words', None)
    u.pop('pronunciation_plan', None)
    u['status'] = 'translated'
    for part in u.get('source_slices') or [{'line_id':u['parent_line_id'], 'source':u['source']}]:
        parents[part['line_id']] = parents.get(part['line_id'],'')+part['source']
    cursor += duration
filters.append(''.join(streams)+f'concat=n={len(units)}:v=1:a=1[v][a]')
output = workspace/'output'
(output/'log').mkdir(exist_ok=True)
(output/'audio').mkdir(exist_ok=True)
subprocess.run(['ffmpeg','-y','-v','error','-i',str(BASE/'output/source.mp4'),
    '-filter_complex',';'.join(filters),'-map','[v]','-map','[a]', '-c:v','libx264','-preset','veryfast',
    '-crf','20','-c:a','aac',str(output/'source.mp4')],check=True)
combined.export(output/'audio/vocal.wav',format='wav')
subprocess.run(['ffmpeg','-y','-v','error','-i',str(output/'source.mp4'),'-vn','-ar','24000',
    '-ac','1',str(output/'audio/raw_master.wav')],check=True)
pd.DataFrame(words).to_excel(output/'log/cleaned_chunks.xlsx',index=False)
terminology=json.loads((BASE/'output/log/terminology.json').read_text())
(output/'log/terminology.json').write_text(json.dumps(terminology,ensure_ascii=False))
plan.update(units=units,parents=[{'line_id':lid,'source':src} for lid,src in parents.items()],status='translated')
os.chdir(workspace)
if os.environ.get('VLO_SHORT_PROFILE'):
    from core.config_utils import update_key
    update_key('mlx_tts.backends.indextts2.profile', os.environ['VLO_SHORT_PROFILE'])
save_plan(plan)
frame=pd.DataFrame([{'LineID':u['unit_id'],'Source':u['source'],'Translation':u['target']} for u in units])
if os.environ.get('VLO_SHORT_TRANSLATE') == '1':
    # Exercise the normal local translator, including conflicting summary
    # spelling, before measuring synthesis. Do not reuse historical targets.
    from core.local_translation import translate_localized
    from core.speech_budget import budget_for
    context = (json.loads(Path(os.environ['VLO_SHORT_CONTEXT']).read_text())
        if os.environ.get('VLO_SHORT_CONTEXT') else
        json.dumps({'topic': terminology['topic'], 'terms': terminology['terms']}, ensure_ascii=False))
    translations, echoed = translate_localized('\n'.join(u['source'] for u in units), context,
        terminology['terms'], speech_budgets=[budget_for({**u, 'line_id': u['unit_id']}) for u in units])
    assert echoed.splitlines() == [u['source'] for u in units]
    frame['Translation'] = translations.splitlines()
    for u, target in zip(plan['units'], translations.splitlines()):
        u.update(target=target, status='translated')
    save_plan(plan)
    (output/'short_translation_input.json').write_text(json.dumps({
        'mode': 'normal_local_translator', 'units': units, 'terms': terminology['terms']},
        ensure_ascii=False, indent=2))
from core.atomic_dubbing import translation_speech_preflight, create_tasks, reuse_locked_speech
from core.plan_subtitles import render_plan_subtitles
from core.bilingual_alignment import verify_bilingual_display
from core.tts_pronunciation import original_name_conflicts
from core.dubbing_quality import evaluate_dubbing
from core.duration_candidate_audit import audit_duration_candidate
from core.speech_budget import duration_context
from core.utterance_plan import load_plan

accepted = translation_speech_preflight(frame,terminology)
# The preflight returns candidates; persist their exact approved wording and
# clock before invoking the stock locked-take consumer, as Step 4 does.
plan = load_plan()
for u,row in zip(plan['units'],accepted.to_dict('records')):
    u.update(target=row['Translation'],status='translated')
save_plan(plan)
accepted.to_excel(output/'log/translation_results.xlsx',index=False)
create_tasks()
reuse_locked_speech()
render_plan_subtitles(dubbed=False)
render_plan_subtitles(dubbed=True)
plan=load_plan()
verify_bilingual_display(plan,dubbed=False)
verify_bilingual_display(plan,dubbed=True)
assert all(not original_name_conflicts(u['source'],u['target'],'zh',terminology['terms']) for u in plan['units'])
tasks=pd.read_excel(output/'audio/tts_tasks.xlsx')
rows,summary=evaluate_dubbing(tasks)
assert summary['quality_gate']['passed'],summary
negative=audit_duration_candidate('I save the file.', '', '我删除文件。', json.dumps({'topic':'video editing'}))
assert not negative['equivalent'],negative
report={'status':'pass','workspace':str(workspace),'source_asr_basis':'retained_source_words_clipped_and_relocated',
    'source_evidence':hash_path(BASE/'output/log/cleaned_chunks.xlsx'),
    'semantic_negative_control':negative,'summary':summary,'units':plan['units']}
(output/'short_acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('SHORT_ACCEPTANCE_PASS',json.dumps(summary,ensure_ascii=False),flush=True)
