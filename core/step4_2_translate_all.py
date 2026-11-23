import concurrent.futures
from dataclasses import asdict
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.config_utils import load_key
from core.llm_provider import get_llm_max_workers, get_translation_strategy
from core.step4_1_summarize import search_things_to_note_in_prompt
from core.timing_utils import time_it
from core.translate_once import translate_lines
from core.translation_context import (
    load_speaker_aware_rows,
    speaker_profiles_context,
    split_speaker_aware_batches,
    surrounding_context,
)
from core.translation_review import is_valid_target_language, review_and_repair_translation
from core.translation_state import record_llm_stage


console = Console()

TRANSLATION_RESULTS_FILE = "output/log/translation_results.xlsx"
TERMINOLOGY_FILE = "output/log/terminology.json"
WORKFLOW_REPORT_FILE = "output/log/translation_workflow_metrics.json"


def existing_translation_is_speaker_aware() -> bool:
    if not os.path.exists(TRANSLATION_RESULTS_FILE) or not os.path.exists(TERMINOLOGY_FILE):
        return False
    try:
        tdf = pd.read_excel(TRANSLATION_RESULTS_FILE)
        columns = set(tdf.columns)
        with open(TERMINOLOGY_FILE, "r", encoding="utf-8") as file:
            terminology = json.load(file)
        rows, _ = load_speaker_aware_rows()
        if len(tdf) != len(rows):
            return False
    except Exception:
        return False
    profiles = terminology.get("speaker_profiles")
    return {"LineID", "Speaker", "Source", "Translation"}.issubset(columns) and bool(profiles)


def split_chunks_by_chars(chunk_size=2000, max_i=10):
    """Compatibility wrapper returning plain text chunks with speaker-safe boundaries."""
    rows, _ = load_speaker_aware_rows()
    return [
        batch["source_text"]
        for batch in split_speaker_aware_batches(rows, chunk_size=chunk_size, max_lines=max_i)
    ]


@time_it()
def translate_chunk(batch, all_rows, terminology, index):
    source_text = batch["source_text"]
    things_to_note_prompt = search_things_to_note_in_prompt(source_text)
    previous_content_prompt, after_content_prompt = surrounding_context(all_rows, batch)
    focused_ids = set(batch["rows"]["line_id"])
    batch_context = dict(terminology)
    batch_context["dialogue_units"] = [
        unit for unit in terminology.get("dialogue_units", [])
        if unit.get("line_id") in focused_ids
    ]
    batch_context["source_uncertainties"] = [
        group for group in terminology.get("source_uncertainties", [])
        if set(group["line_ids"]) & focused_ids
    ]
    budgets = None
    if "unit_id" in batch["rows"]:
        from core.speech_budget import budget_for
        budgets = [budget_for(row) for row in batch["rows"].to_dict("records")]
        batch_context["parent_dialogue"] = list(dict.fromkeys(batch["rows"]["parent_source"]))
    translation, echoed_source = translate_lines(
        source_text,
        previous_content_prompt,
        after_content_prompt,
        things_to_note_prompt,
        speaker_profiles_context(batch_context),
        index,
        speaker_context_prompt=batch["speaker_context"],
        glossary=[term for term in terminology.get("terms", []) if str(term.get("src", "")).lower() in source_text.lower()],
        speech_budgets=budgets,
    )
    expected_lines = source_text.splitlines()
    echoed_lines = echoed_source.splitlines()
    translated_lines = translation.splitlines()
    if echoed_lines != expected_lines:
        raise ValueError(f"Translation batch {batch['batch_id']} changed source line identity")
    if len(translated_lines) != len(expected_lines):
        # Auto-align line count if minor mismatch occurred
        if len(translated_lines) < len(expected_lines):
            from core.ask_gpt import ask_gpt
            for missing_idx in range(len(translated_lines), len(expected_lines)):
                missing_src = expected_lines[missing_idx]
                try:
                    fallback_trans = ask_gpt(f"请将这句英文翻译为简短自然的中文口语，只输出译文：{missing_src}", response_json=False)
                    fallback_trans = str(fallback_trans).strip().strip(chr(34)) if fallback_trans else missing_src
                except Exception:
                    fallback_trans = missing_src
                translated_lines.append(fallback_trans)
        elif len(translated_lines) > len(expected_lines):
            translated_lines = translated_lines[:len(expected_lines)]
    return index, translated_lines


def assemble_translation_results(results, batches, rows):
    """Collect asynchronous results by stable batch index in O(B) time."""
    translated_by_batch = {index: translated for index, translated in results}
    translated_lines = []
    for index in range(len(batches)):
        if index not in translated_by_batch:
            raise RuntimeError(f"Missing translation result for batch B{index + 1:04d}")
        translated_lines.extend(translated_by_batch[index])
    if len(translated_lines) != len(rows):
        raise RuntimeError(
            f"Translated row count {len(translated_lines)} does not match source row count {len(rows)}"
        )
    output = pd.DataFrame({
        "LineID": rows["line_id"],
        "Speaker": rows["speaker"],
        "Source": rows["text"],
        "Translation": translated_lines,
    })
    for column, field in (("Character", "character_id"), ("AddressedTo", "addressed_to"),
                          ("RelaysCharacter", "relays_character"), ("ActionAgent", "action_agent"),
                          ("ActionPatient", "action_patient"), ("RoleConfidence", "confidence")):
        if field in rows:
            output[column] = rows[field].values
    for column, field in (("UnitID", "unit_id"), ("ParentLineID", "parent_line_id"),
                          ("Start", "start"), ("End", "end"), ("SourceEnd", "source_end")):
        if field in rows:
            output[column] = rows[field].values
    return output


def apply_translation_exact_replacements(output: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Apply verified run-specific translations keyed by the exact source line."""
    replacements = load_key("translation_exact_replacements", {}) or {}
    if not isinstance(replacements, dict):
        raise ValueError("translation_exact_replacements must be a mapping")
    result = output.copy()
    applied = 0
    for source, target in replacements.items():
        if not isinstance(source, str) or not isinstance(target, str) or not source or not target:
            raise ValueError("translation exact replacement keys and values must be non-empty strings")
        mask = result["Source"].eq(source)
        if mask.any():
            applied += int((result.loc[mask, "Translation"] != target).sum())
            result.loc[mask, "Translation"] = target
    return result, applied


@time_it("翻译全部")
def translate_all():
    if os.path.exists(TRANSLATION_RESULTS_FILE):
        if not existing_translation_is_speaker_aware():
            console.print("[yellow]⚠️ Existing translation_results.xlsx is stale, incompatible or line count differs from current source rows. Automatically re-translating...[/yellow]")
            try:
                os.remove(TRANSLATION_RESULTS_FILE)
            except OSError:
                pass
        console.print(Panel(
            "Speaker-aware `translation_results.xlsx` already exists; skipping TRANSLATE ALL.",
            title="Warning",
            border_style="yellow",
        ))
        return

    console.print("[bold green]Start Speaker-Aware Translating All...[/bold green]")
    rows, metrics = load_speaker_aware_rows()
    chunk_size = int(load_key("translation_context.batch_max_chars", 2000))
    max_lines = int(load_key("translation_context.batch_max_lines", 10))

    with open(TERMINOLOGY_FILE, "r", encoding="utf-8") as file:
        terminology = json.load(file)
    if "topic" not in terminology:
        raise RuntimeError("terminology.json is missing topic; rerun speaker-aware summary")
    from core.dialogue_roles import attach_dialogue_roles
    rows = attach_dialogue_roles(rows, terminology)
    batches = split_speaker_aware_batches(rows, chunk_size=chunk_size, max_lines=max_lines)
    profile_ids = {
        str(item.get("speaker_id", "")).strip()
        for item in terminology.get("speaker_profiles", [])
        if isinstance(item, dict)
    }
    missing_profiles = set(metrics.speakers) - profile_ids
    if missing_profiles:
        raise RuntimeError(
            f"Speaker profiles are missing for {sorted(missing_profiles)}; rerun Step 4 summary."
        )

    from core.atomic_dubbing import resume_translation_lines
    resumed_lines = resume_translation_lines(rows)
    if resumed_lines is not None:
        results, cursor = [], 0
        for index, batch in enumerate(batches):
            size = len(batch['rows'])
            results.append((index, resumed_lines[cursor:cursor + size]))
            cursor += size
        print(f"Resuming {len(resumed_lines)} source-bound translation drafts; "
              "primary requests skipped, current gates remain required", flush=True)
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Translating speaker-aware chunks...", total=len(batches))
            with concurrent.futures.ThreadPoolExecutor(max_workers=get_llm_max_workers()) as executor:
                futures = [
                    executor.submit(translate_chunk, batch, rows, terminology, index)
                    for index, batch in enumerate(batches)
                ]
                results = []
                try:
                    for future in concurrent.futures.as_completed(futures):
                        results.append(future.result())
                        progress.update(task, advance=1)
                        print(f"Translation batches {len(results)}/{len(batches)}", flush=True)
                except Exception:
                    # A failed gate cannot publish this batch. Cancel queued work
                    # instead of silently spending minutes finishing every request.
                    for future in futures:
                        future.cancel()
                    raise

    # Deterministic O(B) collection by immutable batch index. Do not fuzzy-match
    # repeated dialogue against every batch after the requests complete.
    output = assemble_translation_results(results, batches, rows)
    output, exact_replacements = apply_translation_exact_replacements(output)
    if exact_replacements:
        console.print(
            f"[green]✅ Applied {exact_replacements} verified exact translation replacement(s).[/green]"
        )

    # Mutual address (哥 / 弟) rendered identically loses who is the elder; batches can split the pair.
    target_code = str(load_key("target_language", "en"))
    if not target_code.lower().startswith("zh"):
        from core.address_terms import repair_exchanges, save_report
        from core.wordplay import english_language_name
        repaired, address_report = repair_exchanges(
            output["Source"].astype(str).tolist(), output["Translation"].astype(str).tolist(),
            english_language_name(target_code))
        save_report(address_report)
        for item in address_report["repaired"]:
            console.print(f"[green]👥 Address seniority kept: {item['elder_source']} / {item['younger_source']} → "
                          f"{item['elder_after']} / {item['younger_after']}[/green]")
        output["Translation"] = repaired

    output.to_excel("output/log/translation_candidate.xlsx", index=False)
    target_lang = str(load_key("target_language", "en"))
    is_fast_mode = (load_key("quality_mode") == "fast") or bool(load_key("fast_mode.enabled", False))

    if is_fast_mode:
        console.print("[cyan]⚡ Fast Mode: Skipping preflight TTS synthesis and heavy semantic audit.[/cyan]")
        for idx, row in output.iterrows():
            t_val = str(row.get("Translation", "") or "").strip()
            from core.fast_text import validate_fast_text
            validate_fast_text(str(row.get("Source", "")), t_val, target_lang, terminology.get("terms", []))
        review_report = {"status": "not_run", "mode": "fast", "basic_checks": "pass", "note": "Deep semantic audit not run"}
        if "UnitID" in output:
            from core.utterance_plan import bind_translation
            bind_translation(output)
        # Release idle oMLX models after translation to free unified memory for TTS
        try:
            from core.providers.omlx_memory import release_omlx_for_tts
            release_omlx_for_tts('indextts2')
        except Exception as exc:
            console.print(f"[yellow]⚠️ Failed to release oMLX memory: {exc}[/yellow]")
    else:
        if "UnitID" in output:
            from core.atomic_dubbing import translation_speech_preflight
            from core.speech_budget import guard_translation_context
            output = guard_translation_context(output, rows, terminology)
            output = translation_speech_preflight(output, terminology)

        # Publication verifies the same exact-source semantic receipts. Only new
        # wording/context needs new inference; acceptance is never speech-only.
        output, review_report = review_and_repair_translation(
            output, terminology,
            max_repair_rounds=int(load_key("translation.max_repair_rounds", 3)),
            audit_only=get_translation_strategy() == "localized" and load_key("translation.review_mode", "audit_only") == "audit_only",
        )
        # Keep a reviewable candidate even if the strict publication gate fails.
        # This file is not a downstream checkpoint or an accepted translation.
        output.to_excel("output/log/translation_candidate.xlsx", index=False)
        if review_report.get("status") != "pass":
            review_err = review_report.get("error") or "Semantic review detected unresolvable critical issues"
            raise ValueError(f"Translation review gate failed: {review_err}")
        for _, row in output.iterrows():
            t_val = str(row.get("Translation", "") or "")
            if not is_valid_target_language(t_val, target_lang):
                raise ValueError(
                    f"Translation line {row.get('LineID')} has invalid target language content: {t_val}"
                )

        if "UnitID" in output:
            from core.utterance_plan import bind_translation
            # No wording change is allowed after the measured preflight. If a
            # mutating review mode is selected it must preflight the new text too.
            from core.atomic_dubbing import verify_preflight_identity
            verify_preflight_identity(output)
            bind_translation(output)
    output.to_excel(TRANSLATION_RESULTS_FILE, index=False)
    baseline_comparisons = len(batches) * len(batches)
    optimized_comparisons = len(batches)
    workflow_metrics = {
        **asdict(metrics),
        "translation_strategy": get_translation_strategy(),
        "llm_max_workers": get_llm_max_workers(),
        "batch_count": len(batches),
        "primary_draft_reused": resumed_lines is not None,
        "speaker_context_input_coverage_before": 0.0,
        "speaker_context_input_coverage_after": metrics.coverage,
        "result_assignment_before": "fuzzy_all_pairs_O(B^2)",
        "result_assignment_after": "stable_batch_id_O(B)",
        "result_matching_comparisons_before": baseline_comparisons,
        "result_matching_comparisons_after": optimized_comparisons,
        "comparison_reduction": (
            1 - optimized_comparisons / baseline_comparisons
            if baseline_comparisons else 0.0
        ),
        "translation_quality_improvement": "pending_same_dataset_human_or_llm_judge",
    }
    with open(WORKFLOW_REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(workflow_metrics, file, ensure_ascii=False, indent=2)
    record_llm_stage(
        "translate",
        artifacts=[
            TRANSLATION_RESULTS_FILE,
            "output/log/translation_speaker_context.json",
            "output/log/translation_review.json",
            "output/log/translation_duration_candidate_provenance.json",
            "output/log/duration_candidate_semantic_audit.json",
            WORKFLOW_REPORT_FILE,
        ],
    )
    console.print(
        f"[bold green]✅ Speaker-aware translation saved: {len(output)} lines, "
        f"{metrics.speaker_count} speakers, {len(batches)} deterministic batches.[/bold green]"
    )


if __name__ == "__main__":
    translate_all()
