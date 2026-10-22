"""Rebuild aligned bilingual subtitles from bound accepted speech, in a new workspace."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.pipeline.artifact_manifest import hash_path

REQUIRED = ('source.mp4', 'audio/vocal.wav', 'audio/background.wav', 'audio/tts_tasks.xlsx',
            'log/utterance_plan.json', 'log/terminology.json', 'log/translation_results.xlsx',
            'log/translation_review.json', 'log/translation_speech_preflight.json')
OPTIONAL = ('audio/raw_master.wav', 'audio/demucs_receipt.json', 'log/target_word_alignment.json',
            'log/llm_artifacts_manifest.json', 'log/cleaned_chunks.xlsx', 'log/source_verification.json',
            'log/lexical_boundary_review.json', 'log/dialogue_role_review.json',
            'log/source_reference_assessment.json', 'log/source_segmentation.json',
            'log/sentence_splitbymeaning.txt', 'log/source_pause_budget.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    source = args.source_workspace.resolve() / 'output'
    output = args.output.resolve()
    if output.exists() != args.resume:
        raise ValueError('Use a new directory, or explicitly resume this verification workspace')
    missing = [p for p in REQUIRED if not (source / p).is_file()]
    if missing:
        raise ValueError(f'Accepted source and speech evidence is missing: {missing}')
    preflight = json.loads((source / 'log/translation_speech_preflight.json').read_text())
    review = json.loads((source / 'log/translation_review.json').read_text())
    if preflight.get('status') != 'pass' or review.get('status') != 'pass':
        raise ValueError('The retained speech and source semantic review must both pass')
    from core.pipeline.runner import PipelineRun, build_steps_for_input, run_pipeline
    selected = {'split_subtitle', 'timeline', 'merge_subtitle', 'merge_audio', 'merge_video'}
    steps = [s for s in build_steps_for_input(str(source / 'source.mp4')) if s.key in selected]
    speech_hashes = {p['unit_id']: {'fitted': hash_path(source.parent / p['path']),
                                    'natural': hash_path(source.parent / p['raw_path'])}
                     for p in preflight['audio']}
    if any(speech_hashes[p['unit_id']] != {'fitted': p['sha256'], 'natural': p['raw_sha256']}
           for p in preflight['audio']):
        raise ValueError('The retained accepted audio changed')
    lineage = {'scope': 'bilingual_subtitle_alignment_only; retained_locked_speech',
               'source_workspace': str(source.parent), 'source_video': hash_path(source / 'source.mp4'),
               'translation': hash_path(source / 'log/translation_results.xlsx'),
               'speech': speech_hashes, 'steps': [s.key for s in steps],
               'translation_rerun': False, 'voice_generation_rerun': False}
    if args.dry_run:
        print(json.dumps({**lineage, 'output': str(output)}, ensure_ascii=False, indent=2))
        return 0
    if args.resume:
        if json.loads((output / 'subtitle_source_lineage.json').read_text()) != lineage:
            raise ValueError('The original locked translation or audio changed before resume')
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name in REQUIRED + OPTIONAL:
            if (source / name).is_file():
                target = output / 'output' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, target)
        for name in ('audio/segs', 'audio/refers', 'audio/emotion_refers'):
            if (source / name).is_dir():
                shutil.copytree(source / name, output / 'output' / name)
        for item in preflight['audio']:
            raw = source.parent / item['raw_path']
            for original in (raw, raw.with_suffix('.conditioning.json')):
                if original.is_file():
                    relative = original.relative_to(source)
                    target = output / 'output' / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, target)
        shutil.copy2(source.parent / 'config.yaml', output / 'config.yaml')
        (output / 'subtitle_source_lineage.json').write_text(json.dumps(lineage, ensure_ascii=False, indent=2))
    os.chdir(output)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get('PYTHONPATH', '')]))
    from core import config_utils
    config_utils.CONFIG_PATH = str(output / 'config.yaml')
    from core.utterance_plan import load_plan
    from core.atomic_dubbing import verify_preflight_identity, unit_clock_signature
    import pandas as pd
    plan = load_plan()
    clock = unit_clock_signature(plan)
    frame = pd.DataFrame([{'LineID': u['unit_id'], 'Source': u['source'], 'Translation': u['target']}
                          for u in plan['units']])
    verify_preflight_identity(frame)
    run = PipelineRun(run_id=output.name, profile='current_configuration', steps=[s.key for s in steps], input=str(source / 'source.mp4'),
                      source='zh', target='en', llm='omlx', tts='mlx_indextts2')
    result = {**lineage, 'status': 'fail'}
    started = time.monotonic()
    try:
        run_pipeline(run, steps, resume=True)
        current = load_plan()
        if unit_clock_signature(current) != clock:
            raise ValueError('Subtitle rebuilding changed the source speech clocks')
        verify_preflight_identity(frame)
        from core.bilingual_alignment import verify_bilingual_display
        verify_bilingual_display(current, dubbed=True)
        from core.dubbing_quality import write_dubbing_eval
        summary = write_dubbing_eval()
        if not summary['quality_gate']['passed']:
            raise ValueError('Retained dubbing quality failed after subtitle rebuilding')
        for item in preflight['audio']:
            if hash_path(item['path']) != item['sha256'] or hash_path(item['raw_path']) != item['raw_sha256']:
                raise ValueError('Subtitle rebuilding changed accepted audio')
        alignment = json.loads(Path('output/log/bilingual_alignment.json').read_text())
        layout = json.loads(Path('output/log/dub_subtitle_layout.json').read_text())
        result.update(status='rendered', video=str(output / 'output/AI配音.mp4'),
                      semantic_alignment=alignment['status'], units=len(alignment['units']),
                      display_pairs=sum(len(u['pairs']) for u in alignment['units']),
                      layout=layout, dubbing_summary=summary, source_clocks_unchanged=True,
                      accepted_audio_unchanged=True)
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result.update(run=run.to_dict(), elapsed_seconds=round(time.monotonic() - started, 3))
    Path('bilingual_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: result.get(k) for k in ('status', 'error', 'elapsed_seconds')}, ensure_ascii=False), flush=True)
    return int(result['status'] != 'rendered')


if __name__ == '__main__':
    raise SystemExit(main())
