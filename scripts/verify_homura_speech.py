"""Exercise local Homura and real speech gates on source units in a fresh workspace.

This preserves the supplied source clocks and initial Index drafts. It does not
replace historical artifacts or certify an entire video or perceptual quality.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from ruamel.yaml import YAML

from core import config_utils
from core.atomic_dubbing import translation_speech_preflight, verify_preflight_identity, unit_clock_signature
from core.llm_provider import get_llm_provider_config
from core.pipeline.artifact_manifest import hash_path
from core.plan_subtitles import render_plan_subtitles
from core.translation_review import review_and_repair_translation
from core.translation_state import record_llm_stage
from core.utterance_plan import bind_translation, save_plan, validate_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unit', action='append', required=True)
    args = parser.parse_args()
    source = args.source_workspace.resolve()
    output = args.output.resolve()
    original_path = source / 'output/log/utterance_plan.json'
    plan = json.loads(original_path.read_text(encoding='utf-8'))
    selected = set(args.unit)
    original_units = plan['units']
    if selected - {unit['unit_id'] for unit in original_units}:
        raise ValueError('Unknown requested source unit')
    plan = copy.deepcopy(plan)
    plan['units'] = [unit for unit in plan['units'] if unit['unit_id'] in selected]
    # Keep original context/clocks while making subset lineage explicit.
    slices = {}
    for unit in plan['units']:
        for part in unit.get('source_slices') or [{'line_id': unit['parent_line_id'], 'source': unit['source']}]:
            slices.setdefault(part['line_id'], []).append(part['source'])
    plan['parents'] = [{**parent, 'source': ''.join(slices[parent['line_id']])}
                       for parent in plan['parents'] if parent['line_id'] in slices]
    plan['evaluation_subset'] = {'original_plan_sha256': hash_path(original_path), 'units': args.unit}
    validate_plan(plan)
    checkpoint = json.loads((source / 'output/log/translation_speech_candidate_state.json').read_text(encoding='utf-8'))
    initial = {row['LineID']: row for row in checkpoint['input_rows']}
    frame = pd.DataFrame([{**initial[unit['unit_id']], 'Speaker': unit['speaker'], 'UnitID': unit['unit_id']}
                          for unit in plan['units']])
    if list(frame.Source) != [unit['source'] for unit in plan['units']]:
        raise ValueError('Initial Index drafts no longer match source units')
    terminology = json.loads((source / 'output/log/terminology.json').read_text(encoding='utf-8'))
    yaml = YAML()
    config = yaml.load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    if config['llm']['provider'] != 'omlx' or config['translation']['speech_budget']['candidate_provider'] != 'omlx_homura':
        raise ValueError('This check requires the configured local Index/Homura routes')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'output/audio').mkdir(parents=True)
    (output / 'output/log').mkdir(parents=True)
    config_path = output / 'config.yaml'
    with config_path.open('w', encoding='utf-8') as handle:
        yaml.dump(config, handle)
    shutil.copy2(source / 'output/audio/vocal.wav', output / 'output/audio/vocal.wav')
    config_utils.CONFIG_PATH = str(config_path)
    os.chdir(output)
    os.environ['HF_HUB_OFFLINE'] = '1'
    save_plan(plan)
    Path('output/log/terminology.json').write_text(json.dumps(terminology, ensure_ascii=False, indent=2), encoding='utf-8')
    frame.to_excel('initial_index_drafts.xlsx', index=False)
    clock = unit_clock_signature(plan)
    started = time.monotonic()
    result = {'status': 'fail', 'scope': 'selected_real_source_units; no_full_video_or_independent_human_acceptance',
              'source_workspace': str(source), 'original_plan_sha256': hash_path(original_path),
              'unit_clock_signature': clock, 'units': args.unit, 'stage': 'provider_check'}
    try:
        for name in ('omlx', 'omlx_homura'):
            provider = get_llm_provider_config(name)
            if urlsplit(provider.base_url).hostname not in {'localhost', '127.0.0.1', '::1'}:
                raise ValueError('Refusing non-local inference')
            result[name] = {'model': provider.model, 'model_revision': provider.model_revision,
                            'base_url': provider.base_url}
        result['stage'] = 'translation_speech_preflight'
        accepted = translation_speech_preflight(frame, terminology)
        result['stage'] = 'final_semantic_review'
        reviewed, review = review_and_repair_translation(accepted, terminology, audit_only=True)
        result['final_review'] = review
        if review['status'] != 'pass':
            raise ValueError('Final semantic review has unresolved findings')
        verify_preflight_identity(reviewed)
        bind_translation(reviewed)
        result['stage'] = 'target_alignment_and_subtitle_layout'
        render_plan_subtitles()
        updated = json.loads(Path('output/log/utterance_plan.json').read_text(encoding='utf-8'))
        if unit_clock_signature(updated) != clock:
            raise ValueError('Source/dubbing clocks changed')
        reviewed.to_excel('output/log/translation_results.xlsx', index=False)
        record_llm_stage('translate')
        result.update(status='pass', stage='selected_units_complete', rows=reviewed.to_dict('records'))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    provenance = Path('output/log/translation_duration_candidate_provenance.json')
    if provenance.is_file():
        result['homura_candidates'] = json.loads(provenance.read_text(encoding='utf-8'))['candidates']
    report_path = Path('output/log/translation_speech_preflight.json')
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        result['speech_status'] = report['status']
        result['speech_issues'] = report.get('issues', [])
        result['speech_changes'] = report.get('changes', [])
    Path('integration_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'stage', 'elapsed_seconds')}, ensure_ascii=False), flush=True)
    if result.get('error'):
        print(result['error'], flush=True)
    return int(result['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
