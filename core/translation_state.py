from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.llm_provider import get_llm_provider_config


MANIFEST_PATH = Path("output/log/llm_artifacts_manifest.json")
GPT_LOG_DIR = Path("output/gpt_log")

TRANSLATION_STAGES: dict[str, dict[str, list[str]]] = {
    "split_meaning": {
        "artifacts": ["output/log/sentence_splitbymeaning.txt"],
        "logs": ["sentence_splitbymeaning"],
    },
    "summarize": {
        "artifacts": ["output/log/terminology.json", "output/log/sentence_splitbymeaning.txt"],
        "logs": ["summary", "stt_correction_batch_*"],
    },
    "translate": {
        "artifacts": ["output/log/translation_results.xlsx"],
        "logs": ["translate_faithfulness", "translate_expressiveness"],
    },
    "split_subtitle": {
        "artifacts": [
            "output/log/translation_results_for_subtitles.xlsx",
            "output/log/translation_results_remerged.xlsx",
        ],
        "logs": ["align_subs", "sentence_splitbymeaning"],
    },
}

TRANSLATION_DEPENDENT_STEPS = {
    "split_meaning",
    "summarize",
    "translate",
    "split_subtitle",
    "timeline",
    "merge_subtitle",
    "gen_audio_task",
    "gen_dub_chunks",
    "extract_refer",
    "gen_audio",
    "merge_audio",
    "merge_video",
}

TRANSLATION_DERIVED_PATHS = [
    "output/log/translation_results.xlsx",
    "output/log/translation_results_for_subtitles.xlsx",
    "output/log/translation_results_remerged.xlsx",
    "output/log/terminology.json",
    "output/log/translation_speaker_context.json",
    "output/log/translation_workflow_metrics.json",
    "output/log/sentence_splitbymeaning.txt",
    "output/final_timeline.xlsx",
    "output/src_trans.srt",
    "output/trans.srt",
    "output/dub_orig.srt",
    "output/dub.srt",
    "output/dub.mp3",
    "output/AI字幕.mp4",
    "output/AI配音.mp4",
    "output/audio/final_timeline.xlsx",
    "output/audio/src_subs_for_audio.srt",
    "output/audio/trans_subs_for_audio.srt",
    "output/audio/tts_tasks.xlsx",
    "output/audio/dubbing_eval.json",
    "output/audio/dubbing_eval.xlsx",
    "output/audio/dubbing_repair_plan.json",
    "output/audio/dubbing_over_duration_report.json",
    "output/audio/timeline_rescue_report.json",
    "output/audio/timeline_rescue_report.xlsx",
    "output/audio/tts_tasks_timeline_rescue.xlsx",
    "output/audio/refers",
    "output/audio/segs",
]

TRANSLATION_WORKFLOW_VERSION = "speaker-aware-dialogue-v2"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _provider_dict() -> dict[str, Any]:
    provider = get_llm_provider_config()
    data = asdict(provider)
    data.pop("api_key", None)
    data["fingerprint"] = _provider_fingerprint(data)
    return data


def _provider_fingerprint(provider: dict[str, Any]) -> str:
    return "|".join(
        str(provider.get(key, ""))
        for key in ("name", "provider_type", "base_url", "model", "supports_json_object")
    )


def _read_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"version": 1, "stages": {}}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "stages": {}, "manifest_error": "invalid_json"}


def _write_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _artifact_info(path: str) -> dict[str, Any]:
    p = Path(path)
    exists = p.exists()
    info: dict[str, Any] = {"path": path, "exists": exists}
    if exists:
        stat = p.stat()
        info.update(
            {
                "type": "dir" if p.is_dir() else "file",
                "size": stat.st_size if p.is_file() else None,
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)),
            }
        )
    return info


def _matching_log_paths(pattern: str) -> list[Path]:
    if "*" in pattern:
        return sorted(GPT_LOG_DIR.glob(f"{pattern}.json"))
    path = GPT_LOG_DIR / f"{pattern}.json"
    return [path] if path.exists() else []


def _log_models(patterns: list[str]) -> dict[str, Any]:
    models: Counter[str] = Counter()
    files: list[str] = []
    bad_files: list[str] = []
    total_entries = 0
    for pattern in patterns:
        for path in _matching_log_paths(pattern):
            files.append(str(path))
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                bad_files.append(str(path))
                continue
            if not isinstance(data, list):
                bad_files.append(str(path))
                continue
            for item in data:
                if isinstance(item, dict):
                    models[str(item.get("model", ""))] += 1
                    total_entries += 1
    return {
        "files": files,
        "models": dict(models),
        "bad_files": bad_files,
        "entry_count": total_entries,
    }


def record_llm_stage(stage: str, artifacts: list[str] | None = None, logs: list[str] | None = None) -> dict[str, Any]:
    if stage not in TRANSLATION_STAGES:
        raise ValueError(f"Unknown translation stage: {stage}")

    stage_config = TRANSLATION_STAGES[stage]
    artifacts = artifacts or stage_config["artifacts"]
    logs = logs or stage_config["logs"]
    provider = _provider_dict()
    manifest = _read_manifest()
    manifest["version"] = 1
    manifest["updated_at"] = _now_iso()
    manifest["current_provider"] = provider
    manifest.setdefault("stages", {})[stage] = {
        "stage": stage,
        "updated_at": _now_iso(),
        "provider": provider,
        "workflow_version": TRANSLATION_WORKFLOW_VERSION,
        "artifacts": [_artifact_info(path) for path in artifacts],
        "logs": _log_models(logs),
    }
    _write_manifest(manifest)
    return manifest["stages"][stage]


def build_translation_status() -> dict[str, Any]:
    current_provider = _provider_dict()
    manifest = _read_manifest()
    stages: dict[str, Any] = {}
    stale_stage_count = 0
    warning_stage_count = 0

    for stage, stage_config in TRANSLATION_STAGES.items():
        stage_manifest = manifest.get("stages", {}).get(stage, {})
        manifest_provider = stage_manifest.get("provider")
        log_state = _log_models(stage_config["logs"])
        artifact_state = [_artifact_info(path) for path in stage_config["artifacts"]]
        artifact_exists = any(item["exists"] for item in artifact_state)
        reasons: list[str] = []
        warnings: list[str] = []

        if artifact_exists and not stage_manifest:
            warnings.append("artifact_exists_without_manifest")
        if artifact_exists and stage in {"summarize", "translate"}:
            if stage_manifest and stage_manifest.get("workflow_version") != TRANSLATION_WORKFLOW_VERSION:
                reasons.append("workflow_version_mismatch")
            reasons.extend(_speaker_artifact_reasons(stage))
        if manifest_provider and manifest_provider.get("fingerprint") != current_provider["fingerprint"]:
            reasons.append("manifest_provider_mismatch")

        log_models = {model for model, count in log_state["models"].items() if count and model}
        current_model = current_provider["model"]
        if log_models and current_model not in log_models:
            reasons.append("log_model_mismatch")
        elif log_models - {current_model}:
            warnings.append("log_contains_other_models")

        if log_state["bad_files"]:
            reasons.append("bad_log_files")

        stale = bool(reasons)
        warning = bool(warnings)
        if stale:
            stale_stage_count += 1
        if warning:
            warning_stage_count += 1

        stages[stage] = {
            "stage": stage,
            "stale": stale,
            "warning": warning,
            "reasons": reasons,
            "warnings": warnings,
            "artifacts": artifact_state,
            "log_state": log_state,
            "manifest_provider": manifest_provider,
        }

    return {
        "current_provider": current_provider,
        "manifest_path": str(MANIFEST_PATH),
        "manifest_exists": MANIFEST_PATH.exists(),
        "stale_stage_count": stale_stage_count,
        "warning_stage_count": warning_stage_count,
        "stages": stages,
        "next_action": _next_action(stale_stage_count, warning_stage_count),
    }


def _speaker_artifact_reasons(stage: str) -> list[str]:
    reasons: list[str] = []
    if stage == "summarize":
        path = Path("output/log/terminology.json")
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload.get("speaker_profiles"), list) or not payload.get("speaker_profiles"):
                    reasons.append("speaker_profiles_missing")
            except (json.JSONDecodeError, OSError):
                reasons.append("speaker_profiles_unreadable")
    elif stage == "translate":
        path = Path("output/log/translation_results.xlsx")
        if path.exists():
            try:
                import pandas as pd

                columns = set(pd.read_excel(path, nrows=1).columns)
                if not {"LineID", "Speaker"}.issubset(columns):
                    reasons.append("speaker_translation_columns_missing")
            except Exception:
                reasons.append("speaker_translation_artifact_unreadable")
    return reasons


def _stage_has_evidence(stage_state: dict[str, Any]) -> bool:
    return any(item["exists"] for item in stage_state["artifacts"]) or bool(stage_state["log_state"]["files"])


def adopt_current_translation_artifacts(*, dry_run: bool = True) -> dict[str, Any]:
    status = build_translation_status()
    if status["stale_stage_count"]:
        return {
            "ok": False,
            "dry_run": dry_run,
            "reason": "stale_translation_artifacts",
            "stale_stages": [
                stage for stage, state in status["stages"].items() if state["stale"]
            ],
            "hint": "archive stale artifacts or rerun translation before adopting current evidence",
        }

    stages = [stage for stage, state in status["stages"].items() if _stage_has_evidence(state)]
    if not dry_run:
        for stage in stages:
            record_llm_stage(stage)
    return {
        "ok": True,
        "dry_run": dry_run,
        "adopted_stages": [] if dry_run else stages,
        "planned_stages": stages,
        "manifest_path": str(MANIFEST_PATH),
    }


def guard_translation_artifacts_for_steps(step_keys: list[str], *, auto_archive: bool = False) -> dict[str, Any]:
    impacted = [step for step in step_keys if step in TRANSLATION_DEPENDENT_STEPS]
    if not impacted:
        return {"ok": True, "impacted_steps": [], "action": "not_applicable"}

    status = build_translation_status()
    stale_stages = [stage for stage, state in status["stages"].items() if state["stale"]]
    if not stale_stages:
        return {
            "ok": True,
            "impacted_steps": impacted,
            "action": "none",
            "warning_stage_count": status["warning_stage_count"],
            "manifest_path": status["manifest_path"],
        }

    plan = build_translation_archive_plan(reason="stale-translation-provider")
    if auto_archive:
        applied = apply_translation_archive_plan(plan)
        return {
            "ok": True,
            "impacted_steps": impacted,
            "action": "auto_archived",
            "stale_stages": stale_stages,
            "archive": applied,
        }

    return {
        "ok": False,
        "impacted_steps": impacted,
        "action": "blocked",
        "stale_stages": stale_stages,
        "reason": "translation artifacts do not match the current configured LLM provider/model",
        "archive_plan": {
            "destination": plan["destination"],
            "item_count": plan["item_count"],
        },
        "hint": "run `python -m core.cli translation archive --apply` or rerun with `--auto-archive-stale-translation`",
    }


def _next_action(stale_stage_count: int, warning_stage_count: int) -> str:
    if stale_stage_count == 0:
        if warning_stage_count:
            return "translation logs match the current model, but rerun LLM steps once to write a manifest before long production jobs"
        return "translation artifacts match the current model evidence"
    return "archive stale translation artifacts, then rerun LLM steps from step3_2 or step4_1 depending on the source state"


def _archive_candidates() -> list[Path]:
    candidates = {Path(path) for path in TRANSLATION_DERIVED_PATHS}
    candidates.add(MANIFEST_PATH)
    if GPT_LOG_DIR.exists():
        candidates.update(GPT_LOG_DIR.glob("*.json"))
    return sorted(path for path in candidates if path.exists())


def build_translation_archive_plan(reason: str = "translation-reset") -> dict[str, Any]:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = Path("output/history") / f"translation_{stamp}"
    items = []
    for path in _archive_candidates():
        target = destination / path
        items.append({"source": str(path), "target": str(target), "type": "dir" if path.is_dir() else "file"})
    return {
        "reason": reason,
        "destination": str(destination),
        "item_count": len(items),
        "items": items,
    }


def apply_translation_archive_plan(plan: dict[str, Any]) -> dict[str, Any]:
    moved = []
    destination = Path(plan["destination"])
    destination.mkdir(parents=True, exist_ok=True)

    for item in plan.get("items", []):
        source = Path(item["source"])
        target = Path(item["target"])
        if not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved.append({"source": str(source), "target": str(target)})

    marker = destination / "archive_manifest.json"
    marker.write_text(json.dumps({**plan, "moved": moved, "applied_at": _now_iso()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"destination": str(destination), "moved_count": len(moved), "moved": moved}
