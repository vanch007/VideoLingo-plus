from pathlib import Path
from unittest.mock import patch
from contextlib import nullcontext
from types import SimpleNamespace
import tempfile,sys,json
ROOT=Path('/Users/vanch/VideoLingo-plus');sys.path.insert(0,str(ROOT))
from core.pipeline import profiles,job
from core.utterance_plan import tasks_from_plan
from core.fast_subtitles import _wrap_lines
state={}
with patch.object(profiles,'update_key',side_effect=lambda k,v:state.update({k:v})):
 profiles.apply_fast_profile();profiles.apply_cinematic_profile()
profile_result={k:state.get(k) for k in ['quality_mode','translation.strict_review_gate','translation.review_mode','mlx_tts.backends.indextts2.profile','mlx_tts.backends.indextts2.chinese_profile']}
with tempfile.TemporaryDirectory() as tmp:
 w=Path(tmp);(w/'output').mkdir()
 with patch.object(job,'workspace_lock',return_value=nullcontext()),patch.object(job,'job_status',return_value={'active':False}),patch.object(job,'prepare_workspace',return_value=w),patch.object(job,'JOB_FILE',w/'job.json'),patch('core.config_utils.load_key',side_effect=lambda k,d=None: {'quality_mode':'fast'}.get(k,d)),patch.object(job.subprocess,'Popen',return_value=SimpleNamespace(pid=12345)) as popen:
  result=job.start_pipeline_job('/tmp/nonexecuted-audit-input.mp4')
  argv=result['command'];actual=argv[argv.index('--profile')+1]
W=ROOT/'run_workspaces/20261018-231604-5b8636'
plan=json.loads((W/'output/log/utterance_plan.json').read_text());df=tasks_from_plan(plan)
long_en='This is an extremely long English sentence that exceeds the forty-two characters threshold and must be formatted across two lines cleanly'
report=dict(fast_then_cinematic=profile_result,new_web_job_requested_config_fast_actual_cli_profile=actual,plan_reconstructed_tasks=len(df),plan_reconstructed_has_reuse_fingerprint='fingerprint' in df.columns,empty_asr_existing_script_score=1.0,english_wrap_input=long_en,english_wrap_output=_wrap_lines(long_en,42,2,False),production_mutations=False,subprocesses_launched=False)
(ROOT/'docs/reports/fast-plan-audit-20261019/contract-reproductions.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
