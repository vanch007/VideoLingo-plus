"""Fresh translation-to-video verification on a retained, reviewed source checkpoint.

Source ASR/diarization and terminology are reused with their original evidence.
No prior translated text, synthetic speech, or acceptance state is imported.
The ordinary pipeline runner performs every stage from translation onwards.
"""
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


SOURCE_ASSETS = (
    "source.mp4", "audio/vocal.wav", "audio/background.wav", "audio/raw_master.wav",
    "audio/demucs_receipt.json", "log/cleaned_chunks.xlsx", "log/sentence_splitbynlp.txt",
    "log/sentence_splitbymeaning.txt", "log/utterance_plan.json", "log/terminology.json",
    "log/source_pause_budget.json", "log/source_verification.json", "log/source_reference_assessment.json",
    "log/asr_final_results.json", "log/asr_raw_results.json", "log/asr_fingerprint.json",
    "log/speaker_diarization.json", "log/source_asr_coverage.json", "log/source_segmentation.json",
    "log/speaker_sentence_boundaries.json", "log/lexical_boundary_review.json",
    "log/asr_timing_review.json", "log/source_anomalies.json", "log/dialogue_role_review.json",
    "log/dialogue_units.json", "log/content_context.json", "log/speaker_profiles.json",
)
SOURCE_DIRECTORIES = ("log/local_source_asr", "log/source_witness", "log/nemotron_diarization")
REQUIRED = {"source.mp4", "audio/vocal.wav", "audio/background.wav", "log/cleaned_chunks.xlsx",
            "log/sentence_splitbymeaning.txt", "log/utterance_plan.json", "log/terminology.json",
            "log/source_reference_assessment.json"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue this verification workspace after a failure")
    args = parser.parse_args()
    source = args.source_workspace.resolve() / "output"
    output = args.output.resolve()
    if output.exists() != args.resume:
        raise ValueError("Use a new directory, or explicitly --resume an existing verification workspace")
    missing = [name for name in REQUIRED if not (source / name).is_file()]
    if missing:
        raise ValueError(f"Required source evidence is missing: {sorted(missing)}")
    assessment = json.loads((source / "log/source_reference_assessment.json").read_text())
    if assessment.get("status") != "pass":
        raise ValueError("The retained source reference assessment must pass")
    assets = [name for name in SOURCE_ASSETS if (source / name).is_file()]
    hashes = {name: hash_path(source / name) for name in assets}
    from core.pipeline.runner import PipelineRun, build_steps_for_input, run_pipeline
    steps = build_steps_for_input(str(source / "source.mp4"))
    steps = steps[next(i for i, step in enumerate(steps) if step.key == "translate"):]
    if args.dry_run:
        print(json.dumps({"scope": __doc__, "source_workspace": str(source.parent),
                          "output": str(output), "steps": [step.key for step in steps],
                          "source_assets": hashes}, ensure_ascii=False, indent=2))
        return 0
    if args.resume:
        retained = json.loads((output / "source_lineage.json").read_text())
        if retained["source_workspace"] != str(source.parent) or retained["source_assets"] != hashes:
            raise ValueError("Resume source evidence does not match the retained verification")
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name in assets:
            target = output / "output" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
        for name in SOURCE_DIRECTORIES:
            if (source / name).is_dir():
                shutil.copytree(source / name, output / "output" / name)
        shutil.copy2(ROOT / "config.yaml", output / "config.yaml")
    os.chdir(output)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH", "")]))
    from core import config_utils
    config_utils.CONFIG_PATH = str(output / "config.yaml")
    from core.utterance_plan import load_plan, save_plan
    from core.atomic_dubbing import unit_clock_signature
    plan = load_plan()
    clocks = unit_clock_signature(plan)
    if not args.resume:
        for unit in plan["units"]:
            unit.pop("target", None)
            unit["status"] = "planned"
        plan["status"] = "planned"
        save_plan(plan)
    lineage = {"scope": "fresh_translation_to_video; retained_source_checkpoint",
               "source_workspace": str(source.parent), "source_assets": hashes,
               "source_unit_count": len(plan["units"]), "unit_clock_signature": clocks,
               "primary_drafts_reused": False, "human_perceptual_acceptance": "missing evidence"}
    if not args.resume:
        Path("source_lineage.json").write_text(json.dumps(lineage, ensure_ascii=False, indent=2))
    elif retained["unit_clock_signature"] != clocks:
        raise ValueError("The verification source clocks changed before resume")
    run = PipelineRun(run_id=output.name, profile="current_configuration", source="zh", target="en",
                      input=str(source / "source.mp4"), steps=[step.key for step in steps],
                      llm="omlx", tts="mlx_indextts2")
    result = {**lineage, "status": "fail"}
    started = time.monotonic()
    try:
        from core.llm_provider import get_llm_provider_config
        from core.homura_translation import is_homura
        if (config_utils.load_key("llm.provider") != "omlx"
                or not is_homura(get_llm_provider_config("omlx_homura"))
                or config_utils.load_key("translation.speech_budget.candidate_provider") != "omlx_homura"):
            raise ValueError("This verification requires the local Index/Homura configuration")
        run_pipeline(run, steps, resume=args.resume)
        if unit_clock_signature(load_plan()) != clocks:
            raise ValueError("Translation or rendering changed the retained source clocks")
        from core.providers.asr_readback import verify_tasks_file
        from core.dubbing_quality import write_dubbing_eval
        readback = verify_tasks_file()
        summary = write_dubbing_eval()
        result.update(status="rendered", run=run.to_dict(), readback=readback.__dict__,
                      dubbing_summary=summary, video=str(output / "output/AI配音.mp4"))
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", run=run.to_dict())
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    Path("integration_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: result[k] for k in ("status", "elapsed_seconds")}, ensure_ascii=False), flush=True)
    if result.get("error"):
        print(result["error"], flush=True)
    return int(result["status"] != "rendered")


if __name__ == "__main__":
    raise SystemExit(main())
