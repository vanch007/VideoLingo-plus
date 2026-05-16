from __future__ import annotations

import argparse
import json
import time

from core.config_utils import update_key
from core.dubbing_quality import write_dubbing_eval
from core.pipeline.profiles import apply_profile
from core.pipeline.runner import PipelineRun, build_steps_for_input, load_state, run_pipeline
from core.providers.mlx_tts import list_backend_status
from core.providers.omlx import list_omlx_models, smoke_chat


def _cmd_doctor(args: argparse.Namespace) -> int:
    from core.doctor import print_report

    return print_report()


def _cmd_models(args: argparse.Namespace) -> int:
    payload = {"omlx": [], "mlx_tts": list_backend_status()}
    try:
        payload["omlx"] = [model.__dict__ for model in list_omlx_models()]
    except Exception as exc:
        payload["omlx_error"] = f"{type(exc).__name__}: {exc}"
    if args.smoke:
        try:
            payload["omlx_smoke"] = smoke_chat()
        except Exception as exc:
            payload["omlx_smoke_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    if args.target != "dubbing":
        raise SystemExit(f"Unknown eval target: {args.target}")
    if args.readback:
        from core.providers.asr_readback import verify_tasks_file

        summary = verify_tasks_file(limit=args.limit, force=args.force)
        print(json.dumps({"readback": summary.__dict__}, ensure_ascii=False, indent=2))
    print(json.dumps(write_dubbing_eval(), ensure_ascii=False, indent=2))
    return 0


def _cmd_repair(args: argparse.Namespace) -> int:
    if args.target != "dubbing":
        raise SystemExit(f"Unknown repair target: {args.target}")
    if args.readback:
        from core.providers.asr_readback import verify_tasks_file

        verify_tasks_file(limit=args.limit, force=args.force_readback)

    from core.providers.dubbing_repair import (
        apply_repair_plan,
        build_repair_plan,
        run_repair_batches,
        write_over_duration_report,
        write_repair_plan,
    )

    if args.apply and args.batches > 1:
        summary = run_repair_batches(
            batch_limit=args.limit,
            max_batches=args.batches,
            backend_fallback=args.backend_fallback,
            reasons=args.reasons,
            actions=args.actions,
            full_remap=args.full_remap,
        )
        payload = summary.__dict__
        payload["over_duration_report_path"] = write_over_duration_report()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    plan = build_repair_plan(
        limit=args.limit,
        backend_fallback=args.backend_fallback,
        reasons=args.reasons,
        actions=args.actions,
    )
    plan_path = write_repair_plan(plan)
    payload = {
        "plan_path": plan_path,
        "planned": plan["planned"],
        "reason_counts": plan["reason_counts"],
        "action_counts": plan.get("action_counts", {}),
        "reason_filter": plan.get("reason_filter", []),
        "action_filter": plan.get("action_filter", []),
    }
    if args.apply:
        summary = apply_repair_plan(plan, dry_run=False, full_remap=args.full_remap)
        payload["apply"] = summary.__dict__
    else:
        payload["apply"] = {"dry_run": True, "hint": "pass --apply to regenerate selected rows"}
    payload["over_duration_report_path"] = write_over_duration_report()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_rescue(args: argparse.Namespace) -> int:
    if args.target != "timeline":
        raise SystemExit(f"Unknown rescue target: {args.target}")
    from core.providers.timeline_rescue import (
        build_timeline_rescue_report,
        write_timeline_rescue_candidate_tasks,
        write_timeline_rescue_report,
    )

    report = build_timeline_rescue_report(max_gap=args.max_gap, limit_clusters=args.limit_clusters)
    paths = write_timeline_rescue_report(report)
    payload = {
        "paths": paths,
        "summary": report["summary"],
        "top_clusters": report["clusters"][: min(5, len(report["clusters"]))],
    }
    if args.write_candidate_tasks:
        payload["candidate_tasks"] = write_timeline_rescue_candidate_tasks(pre_merge=not args.no_pre_merge)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_translation(args: argparse.Namespace) -> int:
    from core.translation_state import (
        apply_translation_archive_plan,
        build_translation_archive_plan,
        build_translation_status,
    )

    if args.translation_command == "status":
        status = build_translation_status()
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 2 if args.fail_on_stale and status["stale_stage_count"] else 0
    if args.translation_command == "archive":
        plan = build_translation_archive_plan(reason=args.reason)
        payload = {"plan": plan, "apply": {"dry_run": not args.apply}}
        if args.apply:
            payload["apply"] = apply_translation_archive_plan(plan)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.translation_command == "adopt-current":
        from core.translation_state import adopt_current_translation_artifacts

        payload = adopt_current_translation_artifacts(dry_run=not args.apply)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["ok"] else 2
    raise SystemExit(f"Unknown translation command: {args.translation_command}")


def _apply_run_config(args: argparse.Namespace) -> None:
    if args.source:
        update_key("source_language", args.source)
        update_key("whisper.language", args.source)
    if args.target:
        update_key("target_language", args.target)
    if args.llm:
        update_key("llm.provider", "openai_compatible" if args.llm == "config" else args.llm)
    if args.tts:
        update_key("tts_method", "mlx_router" if args.tts == "auto" else args.tts)
    apply_profile(args.profile)


def _cmd_run(args: argparse.Namespace) -> int:
    _apply_run_config(args)
    steps = build_steps_for_input(args.input, dubbing=not args.subtitle_only, subtitles=not args.no_subtitles)
    from core.translation_state import guard_translation_artifacts_for_steps

    translation_guard = guard_translation_artifacts_for_steps(
        [step.key for step in steps],
        auto_archive=args.auto_archive_stale_translation and not args.dry_run,
    )
    if not translation_guard["ok"]:
        print(json.dumps({"translation_guard": translation_guard}, ensure_ascii=False, indent=2))
        return 2
    run = PipelineRun(
        run_id=args.run_id or time.strftime("%Y%m%d-%H%M%S"),
        profile=args.profile or "default",
        source=args.source or "auto",
        target=args.target or "auto",
        steps=[],
    )
    result = run_pipeline(run, steps, resume=not args.no_resume, dry_run=args.dry_run)
    print(json.dumps({"run": result.to_dict(), "translation_guard": translation_guard}, ensure_ascii=False, indent=2))
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    state = load_state()
    if args.run_id and state.get("run_id") != args.run_id:
        raise SystemExit(f"Last run id is {state.get('run_id')}, not {args.run_id}")
    if not args.input:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    return _cmd_run(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m core.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Run environment and service checks")
    doctor.set_defaults(func=_cmd_doctor)

    models = sub.add_parser("models", help="Inspect local model providers")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_list = models_sub.add_parser("list", help="List oMLX and MLX TTS backends")
    models_list.add_argument("--smoke", action="store_true", help="Run a short oMLX chat smoke")
    models_list.set_defaults(func=_cmd_models)

    run = sub.add_parser("run", help="Run the shared VideoLingo pipeline")
    run.add_argument("--input", required=True, help="Video URL, local video path, or SRT path")
    run.add_argument("--source", default=None)
    run.add_argument("--target", default=None)
    run.add_argument("--profile", default="cinematic")
    run.add_argument("--llm", default=None, help="LLM provider. Use 'config' for config.yaml api.base_url/api.model.")
    run.add_argument("--tts", default="auto")
    run.add_argument("--run-id", default=None)
    run.add_argument("--subtitle-only", action="store_true")
    run.add_argument("--no-subtitles", action="store_true")
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--auto-archive-stale-translation", action="store_true", help="Move stale translation-derived artifacts to history before running")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=_cmd_run)

    resume = sub.add_parser("resume", help="Inspect or resume the last run")
    resume.add_argument("--run-id", default=None)
    resume.add_argument("--input", default=None, help="Required only when actually resuming execution")
    resume.add_argument("--source", default=None)
    resume.add_argument("--target", default=None)
    resume.add_argument("--profile", default="cinematic")
    resume.add_argument("--llm", default=None, help="LLM provider. Use 'config' for config.yaml api.base_url/api.model.")
    resume.add_argument("--tts", default="auto")
    resume.add_argument("--subtitle-only", action="store_true")
    resume.add_argument("--no-subtitles", action="store_true")
    resume.add_argument("--no-resume", action="store_true")
    resume.add_argument("--auto-archive-stale-translation", action="store_true", help="Move stale translation-derived artifacts to history before running")
    resume.add_argument("--dry-run", action="store_true")
    resume.set_defaults(func=_cmd_resume)

    eval_parser = sub.add_parser("eval", help="Run quality evaluation")
    eval_parser.add_argument("target", choices=["dubbing"])
    eval_parser.add_argument("--readback", action="store_true", help="Run configured ASR readback before scoring")
    eval_parser.add_argument("--limit", type=int, default=None, help="Limit ASR readback segment count")
    eval_parser.add_argument("--force", action="store_true", help="Re-run ASR readback even when scores exist")
    eval_parser.set_defaults(func=_cmd_eval)

    repair = sub.add_parser("repair", help="Plan or apply dubbing repairs")
    repair.add_argument("target", choices=["dubbing"])
    repair.add_argument("--limit", type=int, default=None, help="Limit repaired segment count")
    repair.add_argument(
        "--backend-fallback",
        choices=["auto", "indextts2", "omnivoice", "qwen3_tts", "voxcpm2"],
        default="auto",
        help="Force a backend for regenerated rows or let the repair planner choose",
    )
    repair.add_argument("--readback", action="store_true", help="Run configured ASR readback before planning")
    repair.add_argument("--force-readback", action="store_true", help="Re-run ASR readback even when scores exist")
    repair.add_argument("--reasons", default=None, help="Comma-separated reason filter, e.g. missing_audio,over_duration")
    repair.add_argument("--actions", default=None, help="Comma-separated action filter, e.g. slow_fit_existing_audio")
    repair.add_argument("--apply", action="store_true", help="Regenerate selected rows instead of writing plan only")
    repair.add_argument("--batches", type=int, default=1, help="Repeat plan/apply for N batches when --apply is set")
    repair.add_argument("--full-remap", action="store_true", help="Recompute all chunk timings after repair")
    repair.set_defaults(func=_cmd_repair)

    rescue = sub.add_parser("rescue", help="Plan non-destructive rescue actions for broken artifacts")
    rescue.add_argument("target", choices=["timeline"])
    rescue.add_argument("--max-gap", type=float, default=0.05, help="Seconds between rows treated as the same timing cluster")
    rescue.add_argument("--limit-clusters", type=int, default=None, help="Limit clusters written to the report")
    rescue.add_argument("--write-candidate-tasks", action="store_true", help="Write non-destructive tts_tasks_timeline_rescue.xlsx from the current fixed timeline")
    rescue.add_argument("--no-pre-merge", action="store_true", help="Disable short-chunk pre-merge for candidate task export")
    rescue.set_defaults(func=_cmd_rescue)

    translation = sub.add_parser("translation", help="Inspect or archive LLM translation artifacts")
    translation_sub = translation.add_subparsers(dest="translation_command", required=True)
    translation_status = translation_sub.add_parser("status", help="Show whether translation artifacts match the current LLM config")
    translation_status.add_argument("--fail-on-stale", action="store_true", help="Exit 2 when stale translation artifacts are detected")
    translation_status.set_defaults(func=_cmd_translation)
    translation_archive = translation_sub.add_parser("archive", help="Plan or apply archiving of translation-derived artifacts")
    translation_archive.add_argument("--reason", default="translation-reset")
    translation_archive.add_argument("--apply", action="store_true", help="Move files into output/history/translation_*")
    translation_archive.set_defaults(func=_cmd_translation)
    translation_adopt = translation_sub.add_parser("adopt-current", help="Write a manifest for current artifacts when logs match the current model")
    translation_adopt.add_argument("--apply", action="store_true", help="Write output/log/llm_artifacts_manifest.json")
    translation_adopt.set_defaults(func=_cmd_translation)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
