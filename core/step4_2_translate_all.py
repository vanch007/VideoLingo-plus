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
from core.step4_1_summarize import search_things_to_note_in_prompt
from core.timing_utils import time_it
from core.translate_once import translate_lines
from core.translation_context import (
    load_speaker_aware_rows,
    speaker_profiles_context,
    split_speaker_aware_batches,
    surrounding_context,
)
from core.translation_state import record_llm_stage


console = Console()

TRANSLATION_RESULTS_FILE = "output/log/translation_results.xlsx"
TERMINOLOGY_FILE = "output/log/terminology.json"
WORKFLOW_REPORT_FILE = "output/log/translation_workflow_metrics.json"


def existing_translation_is_speaker_aware() -> bool:
    if not os.path.exists(TRANSLATION_RESULTS_FILE) or not os.path.exists(TERMINOLOGY_FILE):
        return False
    try:
        columns = set(pd.read_excel(TRANSLATION_RESULTS_FILE, nrows=1).columns)
        with open(TERMINOLOGY_FILE, "r", encoding="utf-8") as file:
            terminology = json.load(file)
    except (OSError, ValueError, json.JSONDecodeError):
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
    translation, echoed_source = translate_lines(
        source_text,
        previous_content_prompt,
        after_content_prompt,
        things_to_note_prompt,
        speaker_profiles_context(terminology),
        index,
        speaker_context_prompt=batch["speaker_context"],
    )
    expected_lines = source_text.splitlines()
    echoed_lines = echoed_source.splitlines()
    translated_lines = translation.splitlines()
    if echoed_lines != expected_lines:
        raise ValueError(f"Translation batch {batch['batch_id']} changed source line identity")
    if len(translated_lines) != len(expected_lines):
        raise ValueError(
            f"Translation batch {batch['batch_id']} line mismatch: "
            f"expected {len(expected_lines)}, got {len(translated_lines)}"
        )
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
    return pd.DataFrame({
        "LineID": rows["line_id"],
        "Speaker": rows["speaker"],
        "Source": rows["text"],
        "Translation": translated_lines,
    })


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
            raise RuntimeError(
                "Existing translation_results.xlsx predates the speaker-aware workflow. "
                "Archive stale translation artifacts, then rerun Step 4 summary and translation."
            )
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
    batches = split_speaker_aware_batches(rows, chunk_size=chunk_size, max_lines=max_lines)

    with open(TERMINOLOGY_FILE, "r", encoding="utf-8") as file:
        terminology = json.load(file)
    if "topic" not in terminology:
        raise RuntimeError("terminology.json is missing topic; rerun speaker-aware summary")
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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Translating speaker-aware chunks...", total=len(batches))
        with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
            futures = [
                executor.submit(translate_chunk, batch, rows, terminology, index)
                for index, batch in enumerate(batches)
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            progress.update(task, advance=len(batches))

    # Deterministic O(B) collection by immutable batch index. Do not fuzzy-match
    # repeated dialogue against every batch after the requests complete.
    output = assemble_translation_results(results, batches, rows)
    output, exact_replacements = apply_translation_exact_replacements(output)
    if exact_replacements:
        console.print(
            f"[green]✅ Applied {exact_replacements} verified exact translation replacement(s).[/green]"
        )
    output.to_excel(TRANSLATION_RESULTS_FILE, index=False)
    baseline_comparisons = len(batches) * len(batches)
    optimized_comparisons = len(batches)
    workflow_metrics = {
        **asdict(metrics),
        "batch_count": len(batches),
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
            WORKFLOW_REPORT_FILE,
        ],
    )
    console.print(
        f"[bold green]✅ Speaker-aware translation saved: {len(output)} lines, "
        f"{metrics.speaker_count} speakers, {len(batches)} deterministic batches.[/bold green]"
    )


if __name__ == "__main__":
    translate_all()
