"""Measure and validate speech before locking translation; reuse it downstream.

Step 8–12 cannot rewrite or reshape an accepted utterance. Candidate takes and
their evidence remain inspectable when a translation cannot meet both gates.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import statistics

import pandas as pd
from pydub import AudioSegment
from rich import print as rprint

from core.config_utils import load_key
from core.pipeline.artifact_manifest import hash_path
from core.utterance_plan import load_plan, save_plan, signature, tasks_from_plan, validate_tasks

PREFLIGHT_TASKS = Path("output/audio/translation_speech_tasks.xlsx")
PREFLIGHT_REPORT = Path("output/log/translation_speech_preflight.json")
CANDIDATE_STATE = Path("output/log/translation_speech_candidate_state.json")


def native_tts_runtime_signature() -> str | None:
    if load_key("tts_method", "mlx_indextts2") != "mlx_indextts2":
        return None
    root = Path(load_key("mlx_tts.backends.indextts2.root", "/Users/vanch/mlx-indextts2"))
    return signature({name: hash_path(root / "mlx_indextts" / name) for name in (
        "cli.py", "runtime.py", "model_version.py", "normalize.py", "tokenizer.py", "generate_v2.py",
        "normalizer_v25.py", "text_frontend_v25.py", "tokenizer_v25.py", "generate_v25.py")})


def synthesis_contract_signature() -> str:
    """Duration acceptance is specific to the selected voice model and policy."""
    from core.speech_budget import candidate_model_contract
    from core.tts_pronunciation import pronunciation_policy_signature
    return signature({"tts_method": load_key("tts_method", "mlx_indextts2"),
                      "native_tts_runtime": native_tts_runtime_signature(),
                      "pronunciation_glossary": hash_path("output/log/terminology.json"),
                      "pronunciation_policy": pronunciation_policy_signature(),
                      "mlx_tts": load_key("mlx_tts", {}),
                      "speech_budget": load_key("translation.speech_budget", {}),
                      "candidate_provider": load_key("translation.speech_budget.candidate_provider", ""),
                      "candidate_model_contract": candidate_model_contract(),
                      "llm_provider_settings": load_key("llm.providers", {}),
                      "readback_policy": load_key("dubbing_quality", {}),
                      "workers": {name: hash_path(Path(__file__).resolve().parent / name) for name in (
                          "atomic_dubbing.py", "audio_speed.py", "tts_reference_plan.py",
                          "tts_pronunciation.py",
                          "providers/mlx_tts.py", "providers/asr_readback.py", "ask_gpt.py",
                          "dubbing_content_gate.py", "providers/quality.py",
                          "local_translation.py", "speech_budget.py", "llm_provider.py", "translation_invariants.py",
                          "spoken_length.py", "duration_candidate_audit.py", "homura_translation.py", "model_artifacts.py", "local_mlx_chat.py",
                          "../scripts/local_mlx_chat_worker.py")}})


def candidate_policy_signature() -> str:
    from core.speech_budget import candidate_model_contract
    return signature({"speech_budget":load_key("translation.speech_budget",{}),
                      "candidate_model_contract":candidate_model_contract(),
                      "llm":load_key("llm",{}), "workers":{name:hash_path(Path(__file__).resolve().parent/name)
                      for name in ("speech_budget.py","spoken_length.py","local_translation.py","translation_invariants.py",
                                   "duration_candidate_audit.py","homura_translation.py","model_artifacts.py",
                                   "local_mlx_chat.py","../scripts/local_mlx_chat_worker.py")}})


def unit_clock_signature(plan: dict) -> str:
    return signature([{**{k: u[k] for k in ("unit_id", "source", "speaker", "start", "source_end", "end")},
                       "source_start": u.get("source_start", u["start"])}
                      for u in plan["units"]])


def materialize_references(tasks: pd.DataFrame) -> None:
    """Use exact source windows; voice and emotion conditioning share this clock."""
    root = Path("output/audio")
    vocal = root / "vocal.wav"
    if not vocal.is_file():
        from core.all_whisper_methods.demucs_vl import demucs_main
        demucs_main()
    if not vocal.is_file():
        vocal = root / "vocal.mp3"
    audio = AudioSegment.from_file(vocal)
    audio = audio.set_channels(1)
    (root / "refers").mkdir(parents=True, exist_ok=True)
    for row in tasks.to_dict("records"):
        start, end = row["source_start"], row["source_end"]
        out = root / "refers" / f"{int(row['number'])}.wav"
        # Padding is silence, never a neighbouring character's speech.
        clip = audio[round(start * 1000):round(end * 1000)]
        if len(clip) < 250:
            clip += AudioSegment.silent(duration=250 - len(clip), frame_rate=clip.frame_rate)
        clip.export(out, format="wav")


def _raw_path(uid: str) -> Path:
    return Path("output/audio/preflight") / f"{uid}.wav"


def _preferred_natural_takes(tasks: pd.DataFrame, attempts: list[dict]) -> dict:
    """Choose retained speech for revalidation, not the last random seed.

    A faster complete take of the same script must not be forgotten when a
    later variant is slower. Exact audio identity is checked here; generation
    and reference receipts are still checked by _resume_natural_take, then
    current fitting and content gates revalidate the selected voice normally.
    """
    wording = {row.unit_id: row.text for row in tasks.itertuples()}
    grouped = {}
    for take in attempts:
        uid, path = take.get('unit_id'), take.get('natural_audio_path')
        if (uid not in wording or take.get('text') != wording[uid] or not path
                or hash_path(path) != take.get('candidate_sha256')):
            continue
        grouped.setdefault(uid, []).append(take)
    preferred = {}
    for uid, takes in grouped.items():
        complete = [take for take in takes if take.get('content_complete') is True
                    and isinstance(take.get('natural_seconds'), (int, float))
                    and math.isfinite(take['natural_seconds']) and take['natural_seconds'] > 0]
        preferred[uid] = min(complete, key=lambda take: take['natural_seconds']) if complete else takes[-1]
    return preferred


def _duration_calibration_take(row, current: dict, attempts: list[dict]) -> dict:
    """Use the fastest complete same-script witness under current conditioning."""
    eligible = [take for take in attempts if take.get('unit_id') == row.unit_id
                and take.get('text') == row.text and take.get('content_complete') is True
                and take.get('conditioning_signature') == current.get('conditioning_signature')
                and take.get('readback_signature') and take.get('natural_audio_path')
                and isinstance(take.get('natural_seconds'), (int, float))
                and math.isfinite(take['natural_seconds']) and take['natural_seconds'] > 0
                and hash_path(take['natural_audio_path']) == take.get('candidate_sha256')]
    return min(eligible, key=lambda take: take['natural_seconds']) if eligible else current


def _resume_natural_take(request, previous: dict | None):
    """Reuse a bound natural take as input to fresh measurement, not acceptance."""
    from core.tts_reference_plan import conditioning_receipt_matches
    if not previous or previous.get('text') != request.text:
        return None
    saved = previous.get('generation_settings')
    wanted = request.metadata.get('generation_settings')
    if not isinstance(saved, dict) or not isinstance(wanted, dict):
        return None
    if ({k: v for k, v in saved.items() if k != 'seed'} !=
            {k: v for k, v in wanted.items() if k != 'seed'}):
        return None
    path = previous.get('natural_audio_path')
    saved_hash = str(previous.get('candidate_sha256') or '').removeprefix('sha256:')
    if not path or str(hash_path(path)).removeprefix('sha256:') != saved_hash:
        return None
    candidate = replace(request, output_path=str(Path(path).resolve()),
                        metadata={**request.metadata, 'generation_settings': saved})
    return candidate if conditioning_receipt_matches(candidate) else None


def _candidate_receipts(uid, output_path):
    """Managed peer workspaces may contain short-verified, immutable takes.

    Discovery is not approval. Every caller checks current conditioning,
    runtime, wording, audio hash and then the ordinary speech gates.
    """
    roots = [Path(output_path).parent / 'candidates' / uid]
    managed = Path(__file__).resolve().parents[1] / 'run_workspaces'
    roots.extend(managed.glob(f'*/output/audio/preflight/candidates/{uid}'))
    seen = set()
    for root in roots:
        for path in root.glob('*.conditioning.json'):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def _cached_natural_take(request, excluded_sha256=()):
    """Discover an existing seed variant as input, never as acceptance.

    Short verification and the normal run share content-addressed candidates.
    Their local attempt journals may differ. Search the exact current wording,
    runtime, pronunciation and reference receipts before synthesizing again;
    fitting and independent content readback still run on the selected bytes.
    """
    from core.audio_speed import trim_edge_silence
    uid = request.task_row.get('unit_id') or Path(request.output_path).stem
    candidates = []
    excluded = {str(value).removeprefix('sha256:') for value in excluded_sha256}
    for receipt_path in _candidate_receipts(uid, request.output_path):
        try:
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            path = receipt_path.with_name(receipt_path.name[:-len('.conditioning.json')] + '.wav')
            if str(receipt.get('audio', {}).get('sha256', '')).removeprefix('sha256:') in excluded:
                continue
            previous = {'text': receipt.get('text'), 'generation_settings': receipt.get('generation_settings'),
                        'natural_audio_path': str(path), 'candidate_sha256': receipt.get('audio', {}).get('sha256')}
            retained = _resume_natural_take(request, previous)
            if retained is None:
                continue
            audio = AudioSegment.from_file(path)
            if audio.rms == 0:
                continue
            natural = trim_edge_silence(audio, keep_silence=10, min_silence_len=80, silence_thresh=-35)
            if len(natural) > 0:
                candidates.append((len(natural), retained))
        except (ValueError, OSError, KeyError, TypeError):
            continue
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _cached_duration_draft(row, tasks, unit, context, attempts):
    """Offer a source/conditioning-bound short take before another LLM edit.

    Acoustic fit here is only a search filter. The entire actual target must
    pass current semantic/name/language/display review and fresh speech
    fitting/readback; an old short-pass label is never consumed as acceptance.
    """
    if load_key('tts_method', '') != 'mlx_indextts2':
        return None
    from core.providers.mlx_tts import resolve_indextts_config
    from core.runtime_context import effective_target_language
    from core.audio_speed import trim_edge_silence
    from core.tts_pronunciation import (spoken_plan, pronunciation_terms, synthesis_text,
                                       annotate_standard_mandarin_spans, original_name_conflicts)
    from core.translation_review import is_valid_target_language
    from core.translation_invariants import source_fact_conflicts
    from core.speech_budget import audit_duration_candidate
    from core.bilingual_alignment import prepare_candidate_alignment
    from types import SimpleNamespace
    language = effective_target_language('en')
    cfg = resolve_indextts_config(dict(load_key('mlx_tts.backends.indextts2', {})), language)
    cfg['fit_duration'] = False
    generation = signature({key:value for key,value in cfg.items() if key != 'seed'})
    runtime = native_tts_runtime_signature()
    candidates = {}
    rejected_audio = {str(take.get('candidate_sha256', '')).removeprefix('sha256:')
                      for take in attempts if take.get('unit_id') == row.unit_id
                      and (take.get('fits') is False or take.get('content_complete') is False)}
    for path in _candidate_receipts(row.unit_id, _raw_path(row.unit_id)):
        try:
            receipt = json.loads(path.read_text())
            text, settings = receipt['text'], receipt['generation_settings']
            audio = path.with_name(path.name[:-len('.conditioning.json')] + '.wav')
            if (text == row.text or not text or receipt.get('backend') != 'indextts2'
                    or receipt.get('language') != language or receipt.get('target_duration') is not None
                    or receipt['reference_plan']['fingerprint'] != row.reference_plan_fingerprint
                    or settings.get('runtime_signature') != runtime
                    or settings.get('backend_config_signature') != generation
                    or str(hash_path(audio)).removeprefix('sha256:') !=
                       str(receipt['audio']['sha256']).removeprefix('sha256:')
                    or str(receipt['audio']['sha256']).removeprefix('sha256:') in rejected_audio):
                continue
            plan = spoken_plan(text, language, pronunciation_terms())
            forms = [(plan['spoken_text'], [], False)]
            if plan['spans']:
                annotated, guidance = annotate_standard_mandarin_spans(plan)
                forms.append((annotated, guidance, True))
            if not any(receipt.get('pronunciation_guidance', []) == guidance
                       and settings.get('pronunciation_signature') == signature(guidance)
                       and receipt.get('synthesis_text',text) == synthesis_text(SimpleNamespace(
                           text=text,metadata={'backend':'indextts2','synthesis_text':spoken,
                               'spoken_plan':plan,'spoken_transliterations':plan['matched_transliterations'],
                               'standard_mandarin_guidance':standard,'spoken_pinyin':guidance}))
                       for spoken,guidance,standard in forms):
                continue
            wave = trim_edge_silence(AudioSegment.from_file(audio), keep_silence=10,
                                     min_silence_len=80, silence_thresh=-35)
            if wave.rms == 0 or len(wave)/1000 > float(row.duration)*float(load_key('translation.speech_budget.max_speed_factor',1.08)):
                continue
            if (not is_valid_target_language(text,language) or source_fact_conflicts(unit['source'],text,language)
                    or original_name_conflicts(unit['source'],text,language,pronunciation_terms())):
                continue
            candidates[text] = min(candidates.get(text,float('inf')),len(wave))
        except (OSError,ValueError,KeyError,TypeError):
            continue
    for text in sorted(candidates,key=candidates.get):
        verdict = audit_duration_candidate(unit['source'], row.text, text, context)
        if verdict.get('equivalent') is not True:
            continue
        try:
            prepare_candidate_alignment({**unit,'target':text}, context)
        except ValueError:
            continue
        return text
    return None


def _generate(tasks: pd.DataFrame, indices: list[int], *, seed_offset=0,
              pronunciation_retries=None, resume_takes=None, excluded_takes=None) -> None:
    from core.all_tts_functions.mlx_router import build_mlx_tts_request
    from core.providers.mlx_tts import IndexTTS2Backend
    from core.all_tts_functions.tts_registry import MLX_ROUTER_BACKENDS
    from core.tts_reference_plan import conditioning_receipt_matches, refresh_conditioning_receipt_audio
    from core.tts_pronunciation import synthesis_text
    method = str(load_key("tts_method", "mlx_indextts2"))
    cfg = dict(load_key("mlx_tts.backends.indextts2", {}))
    from core.providers.mlx_tts import resolve_indextts_config
    from core.runtime_context import effective_target_language
    cfg = resolve_indextts_config(cfg, effective_target_language('auto'))
    cfg["seed"] = int(cfg.get("seed", 42)) + seed_offset
    cfg["fit_duration"] = False
    requests, selected = [], []
    for idx in indices:
        row = tasks.loc[idx].to_dict()
        row["disable_native_fit"] = True
        if method in MLX_ROUTER_BACKENDS:
            row["tts_backend"] = MLX_ROUTER_BACKENDS[method]
        output = _raw_path(row["unit_id"])
        output.parent.mkdir(parents=True, exist_ok=True)
        request = build_mlx_tts_request(row["text"], str(output.resolve()), int(row["number"]), tasks,
                                        task_row=row, target_duration=None)
        from core.tts_pronunciation import spoken_plan
        pronunciation = spoken_plan(request.text, request.language,
            tasks.attrs.get('source_terminology', {}).get('terms', []))
        if pronunciation['spans']:
            request.metadata.update(synthesis_text=pronunciation['spoken_text'],
                spoken_transliterations=pronunciation['matched_transliterations'], spoken_plan=pronunciation)
        tasks.at[idx, 'spoken_text'] = pronunciation['spoken_text']
        tasks.at[idx, 'spoken_transliterations'] = json.dumps(pronunciation['matched_transliterations'], ensure_ascii=False)
        if method == "mlx_indextts2":
            guidance = []
            root = Path(cfg.get("root", "/Users/vanch/mlx-indextts2"))
            v25 = (cfg.get("profile") == "v25" or (cfg.get("profile", "auto") == "auto"
                   and (root / "models/mlx-IndexTTS-2.5-8bit").is_dir()))
            if v25 and cfg.get("source_term_pronunciation", True):
                from core.tts_pronunciation import (source_pronunciations, source_compound_pronunciations,
                                                    short_english_pronunciations, annotate_pronunciations)
                guidance = source_pronunciations(row["origin"], tasks.attrs.get("source_terminology", {}).get("terms", []))
                if request.language.startswith("en"):
                    guidance += source_compound_pronunciations(row["origin"], request.text,
                        tasks.attrs.get("source_terminology", {}).get("terms", []))
                    guidance += short_english_pronunciations(request.text, guidance)
                elif request.language.startswith("zh"):
                    from core.tts_pronunciation import (mandarin_transliterations, apply_mandarin_transliterations, mixed_english_pronunciations)
                    terms = tasks.attrs.get("source_terminology", {}).get("terms", [])
                    trans_map = mandarin_transliterations(terms)
                    spoken_text, matched_trans = apply_mandarin_transliterations(request.text, trans_map)
                    if matched_trans:
                        request.metadata["synthesis_text"] = spoken_text
                        request.metadata["spoken_transliterations"] = matched_trans
                        tasks.at[idx, "spoken_transliterations"] = json.dumps(matched_trans, ensure_ascii=False)
                        guidance.append({"type": "mandarin_transliterations", "mappings": matched_trans})
                    elif (idx in (pronunciation_retries or set()) or
                            ((resume_takes or {}).get(row['unit_id'], {}).get('generation_settings') or {}).get(
                                'pronunciation_signature', signature([])) != signature([])):
                        guidance += mixed_english_pronunciations(row["origin"], request.text)
                    if matched_trans:
                        from core.tts_pronunciation import annotate_mandarin_spans
                        annotated, pinyin_receipts = annotate_mandarin_spans(pronunciation)
                        request.metadata['synthesis_text'] = annotated
                        request.metadata['spoken_pinyin'] = pinyin_receipts
                        guidance.extend(pinyin_receipts)
                    else:
                        request.metadata['synthesis_text'] = annotate_pronunciations(request.text, guidance)
                    request.metadata["pronunciation_guidance"] = guidance
            elif (not v25 and pronunciation['spans'] and
                  (idx in (pronunciation_retries or set()) or
                   ((resume_takes or {}).get(row['unit_id'], {}).get('generation_settings') or {}).get(
                       'pronunciation_signature', signature([])) != signature([]))):
                from core.tts_pronunciation import annotate_standard_mandarin_spans
                annotated, guidance = annotate_standard_mandarin_spans(pronunciation)
                request.metadata.update(synthesis_text=annotated, spoken_pinyin=guidance,
                    standard_mandarin_guidance=True, pronunciation_guidance=guidance)
            request.metadata["generation_settings"] = {"seed": cfg["seed"], "fit_duration": False,
                                                        "mode": "translation_natural_take",
                                                        "backend_config_signature": signature({
                                                            key: value for key, value in cfg.items() if key != "seed"}),
                                                        "runtime_signature": native_tts_runtime_signature(),
                                                        "pronunciation_signature": signature(guidance)}
        retained = _resume_natural_take(request, (resume_takes or {}).get(row['unit_id']))
        if retained is None and (seed_offset == 0 or excluded_takes is not None):
            retained = _cached_natural_take(request, (excluded_takes or {}).get(row['unit_id'], ()))
        if retained is not None:
            selected.append((Path(retained.output_path), output))
            continue
        variant_key = signature({"unit_id": row["unit_id"], "text": request.text,
                                 "synthesis_text": synthesis_text(request),
                                 "reference": request.metadata["reference_plan"]["fingerprint"],
                                 "generation": request.metadata.get("generation_settings"),
                                 "variant_index": seed_offset})
        variant = output.parent / "candidates" / row["unit_id"] / f"{variant_key[:24]}.wav"
        variant.parent.mkdir(parents=True, exist_ok=True)
        request = replace(request, output_path=str(variant.resolve()))
        selected.append((variant, output))
        if not conditioning_receipt_matches(request):
            requests.append(request)
    if requests:
        if method == "mlx_indextts2":
            IndexTTS2Backend(cfg).synthesize_batch(requests)
        else:
            from core.providers.mlx_tts import synthesize_with_mlx_router
            for request in requests:
                synthesize_with_mlx_router(request)
    for variant, output in selected:
        shutil.copy2(variant, output)
        receipt_path = variant.with_suffix(".conditioning.json")
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["selected_candidate"] = str(variant)
            output.with_suffix(".conditioning.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
            refresh_conditioning_receipt_audio(output, "natural_candidate_selected")


def fit_take(raw_path: Path, output: Path, seconds: float, *, max_speed=1.08, min_speed=.90) -> dict:
    from core.audio_speed import trim_edge_silence, adjust_audio_speed
    if not 0 < min_speed <= 1 <= max_speed or seconds <= 0:
        raise ValueError("Invalid natural speech fitting limits")
    raw = AudioSegment.from_file(raw_path).set_channels(1).set_frame_rate(24000)
    if not math.isfinite(raw.dBFS) or raw.rms == 0:
        raise ValueError("Generated speech is silent")
    # Match final acoustic onset detection; readback below must prove that no
    # weak opening word was lost. Sub-threshold pre-roll cannot lock a take.
    trimmed = trim_edge_silence(raw, keep_silence=10, min_silence_len=80, silence_thresh=-35)
    natural = len(trimmed) / 1000
    required = natural / seconds
    window_ms = round(seconds * 1000)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Overlong candidate takes remain audible for readback, but cannot be merged.
    if required > max_speed + 1e-6:
        trimmed.export(output, format="wav")
        return {"natural_seconds": natural, "required_speed": required,
                "speed_factor": 1., "fits": False, "dbfs": trimmed.dBFS}
    # A short, complete delivery keeps its natural pace; the remaining source
    # window becomes silence rather than artificial slow speech or new words.
    speed = max(1., required)
    # A nearly-unity ratio can still overrun a short window. Only skip fitting
    # when the complete take really fits the sample's millisecond clock.
    if len(trimmed) <= window_ms:
        fitted = trimmed
        speed = 1.
    else:
        trim_path = output.with_suffix(".trim.wav")
        trim_path.parent.mkdir(parents=True, exist_ok=True)
        trimmed.export(trim_path, format="wav")
        adjust_audio_speed(str(trim_path), str(output), speed)
        fitted = AudioSegment.from_file(output)
        # atempo has a finite analysis window. Retry the complete take with a
        # measured rounding margin; no audible tail may be cropped.
        if len(fitted) > round(seconds * 1000):
            # The rounding allowance is advisory, not a stricter speed gate.
            # Try the allowed cap before rejecting an otherwise complete take.
            corrected = min(max_speed, speed * len(fitted) / max(seconds * 1000 - 15, 1))
            adjust_audio_speed(str(trim_path), str(output), corrected)
            fitted = AudioSegment.from_file(output)
            speed = corrected
    if len(fitted) > window_ms and (len(fitted) - window_ms) <= 15 and fitted[window_ms:].rms == 0:
        fitted = fitted[:window_ms]
    if len(fitted) > window_ms:
        # A timing rejection must still have a complete readback witness.
        # Never leave a missing/stale file that looks like missing speech and
        # triggers synthesis of an otherwise reusable natural take.
        fitted.export(output, format="wav")
        processed = len(fitted) / 1000
        overhead = max(0., processed - natural / speed)
        return {"natural_seconds": natural, "required_speed": required,
                "speed_factor": speed, "fits": False, "dbfs": fitted.dBFS,
                "processed_seconds": processed, "processing_overhead_seconds": overhead,
                "calibrated_max_natural_seconds": max(0., (seconds - overhead) * max_speed),
                "timing_issue": "processed_duration_over_budget"}
    speech_seconds = len(fitted) / 1000
    fitted += AudioSegment.silent(duration=window_ms - len(fitted), frame_rate=24000)
    fitted.export(output, format="wav")
    from core.dubbing_quality import measure_acoustic_activity
    activity = measure_acoustic_activity(str(output))
    onset = activity.get('onset')
    if onset is None or onset > float(load_key('dubbing_quality.max_start_drift', .12)):
        return {'natural_seconds': natural, 'required_speed': required, 'speed_factor': speed,
                'fits': False, 'dbfs': fitted.dBFS, 'acoustic_onset_seconds': onset,
                'timing_issue': 'acoustic_onset_over_limit', 'retry_unchanged_script': True}
    conditioning = raw_path.with_suffix(".conditioning.json")
    if conditioning.is_file():
        shutil.copy2(conditioning, output.with_suffix(".conditioning.json"))
        from core.tts_reference_plan import refresh_conditioning_receipt_audio
        refresh_conditioning_receipt_audio(output, "translation_speech_preflight_fit")
    return {"natural_seconds": natural, "required_speed": required, "speed_factor": speed,
            "fits": True, "dbfs": trimmed.dBFS, "speech_seconds": speech_seconds,
            "acoustic_onset_seconds": onset,
            "trailing_silence_seconds": (window_ms / 1000) - speech_seconds}


def _content_issues(tasks: pd.DataFrame) -> dict[int, list[str]]:
    import re
    from core.dubbing_content_gate import probable_tail_truncation, probable_prefix_truncation, unexpected_boundary_speech
    from core.translation_invariants import number_annotations
    from core.tts_pronunciation import mandarin_transliterations, apply_mandarin_transliterations
    threshold = float(load_key("dubbing_quality.content_score_min", .88))
    from core.tts_pronunciation import pronunciation_terms
    terms = tasks.attrs.get("source_terminology", {}).get("terms", pronunciation_terms())
    from core.runtime_context import effective_target_language
    trans_map = mandarin_transliterations(terms) if effective_target_language('en').startswith('zh') else {}
    issues = {}
    for idx, row in tasks.iterrows():
        transcript = row.get("asr_transcript")
        score = row.get("asr_content_score")
        status = row.get("asr_status")
        reasons = []
        if (not isinstance(transcript, str) or not transcript.strip()
                or not isinstance(status, str) or status != "ok"):
            reasons.append("empty_or_unavailable_readback")
        elif score is None or pd.isna(score) or float(score) < threshold:
            reasons.append("readback_below_threshold")
        leakage = row.get("asr_leakage_score")
        if pd.notna(leakage) and float(leakage) > float(load_key("dubbing_quality.leak_score_max", .12)):
            reasons.append("reference_leak")
        # An absent readback is a failed content witness, not text. Do not send
        # None/NaN/pd.NA (or stale text from a failed recognizer) to lexical
        # boundary checks; the failure remains eligible for bounded recovery.
        if "empty_or_unavailable_readback" in reasons:
            issues[idx] = reasons
            continue
        expected_text = str(row.get("text", "")).strip()
        expected_spoken = apply_mandarin_transliterations(expected_text, trans_map)[0] if trans_map else expected_text
        if isinstance(transcript, str):
            from core.tts_pronunciation import spoken_name_issues, spoken_name_witnesses
            # Recompute witnesses from current, audio-bound raw readbacks.
            # Never trust a saved "passed" flag or splice ASR transcripts.
            line_evidence = []
            try:
                line_evidence = json.loads(row.get('asr_line_results',''))
            except (ValueError,TypeError):
                pass
            from core.dubbing_quality import normalize_lines
            from core.providers.asr_readback import _audio_file_for
            expected_lines = normalize_lines(row.get('lines',expected_text))
            bound = (isinstance(line_evidence,list) and len(line_evidence)==len(expected_lines)
                     and bool(line_evidence) and all(isinstance(item,dict) and item.get('status')=='ok'
                         and item.get('expected')==line and item.get('audio_path')==_audio_file_for(int(row['number']),i)
                         and item.get('audio_sha256') and item['audio_sha256']==hash_path(item['audio_path'])
                         for i,(item,line) in enumerate(zip(line_evidence,expected_lines))))
            if bound:
                for item in line_evidence:
                    witnesses = spoken_name_witnesses(item['expected'],[
                        {'backend':'primary','status':'ok','transcript':item.get('primary_transcript')},
                        {'backend':'secondary',**(item.get('secondary_readback') or {'status':'unavailable'})}],
                        effective_target_language('en'),terms)
                    reasons += ['spoken_name_mismatch:'+w['spoken'] for w in witnesses if not w['passed']]
            else:
                reasons += spoken_name_issues(expected_text, transcript, effective_target_language('en'), terms)
        tail_loss = probable_tail_truncation(expected_text, transcript) and probable_tail_truncation(expected_spoken, transcript)
        prefix_loss = probable_prefix_truncation(expected_text, transcript) and probable_prefix_truncation(expected_spoken, transcript)
        if isinstance(transcript, str) and (tail_loss or prefix_loss):
            reasons.append("incomplete_speech")
        boundary_extra = unexpected_boundary_speech(expected_text, transcript) and unexpected_boundary_speech(expected_spoken, transcript)
        if isinstance(transcript, str) and boundary_extra:
            reasons.append("unexpected_boundary_speech")
        if isinstance(transcript, str):
            from core.providers.quality import spoken_numeric_values
            expected_values = spoken_numeric_values(row['text'])
            if expected_values and expected_values != spoken_numeric_values(transcript):
                reasons.append("spoken_numeric_value_mismatch")
            if isinstance(row.get("origin"),str) and row["origin"].strip():
                from core.translation_invariants import source_fact_conflicts
                from core.runtime_context import effective_target_language
                if source_fact_conflicts(row["origin"],transcript,effective_target_language("en")):
                    reasons.append("spoken_source_fact_mismatch")
        if reasons:
            issues[idx] = reasons
    return issues


def _saved_input_rows(saved: dict, current_rows: list[dict]) -> list[dict] | None:
    """Recover draft lineage; reuse never certifies wording or accepted audio."""
    rows = saved.get("input_rows")
    if rows is None:
        # Older checkpoints retained their first measured scripts. Reconstruct
        # a baseline only when its full hash proves it was the original input.
        first = {}
        for take in saved.get("attempts", []):
            if isinstance(take.get("text"), str):
                first.setdefault(take["unit_id"], take["text"])
        if any(row["LineID"] not in first for row in current_rows):
            return None
        rows = [{**row, "Translation": first[row["LineID"]]} for row in current_rows]
    if not isinstance(rows, list) or signature(rows) != saved.get("input_signature"):
        return None
    if (len(rows) != len(current_rows) or
            any((old.get("LineID"), old.get("Source")) != (new["LineID"], new["Source"])
                for old, new in zip(rows, current_rows))):
        return None
    return rows


def resume_translation_lines(rows: pd.DataFrame) -> list[str] | None:
    """Restore source-bound input drafts, never an acceptance or audio receipt.

    Repeating primary translation on a failed preflight can change otherwise
    verified scripts and unnecessarily regenerate their speech. Retained
    drafts still go through every current semantic/display/speech gate.
    A changed source or clock requires a separate run rather than stale reuse.
    """
    if 'unit_id' not in rows or not CANDIDATE_STATE.is_file():
        return None
    saved = json.loads(CANDIDATE_STATE.read_text(encoding='utf-8'))
    current = [{'LineID': row.line_id, 'Source': row.text, 'Translation': ''}
               for row in rows.itertuples()]
    plan = load_plan()
    if (saved.get('unit_clock_signature') != unit_clock_signature(plan)
            or [(row['LineID'], row['Source']) for row in current]
            != [(unit['unit_id'], unit['source']) for unit in plan['units']]):
        raise ValueError('Cannot resume translation drafts: source identities or measured clock changed')
    baseline = _saved_input_rows(saved, current)
    candidates = saved.get('candidates')
    if (baseline is None or not isinstance(candidates, list)
            or len(candidates) != len(current)
            or any(not isinstance(candidate, dict)
                   or candidate.get('unit_id') != row['LineID']
                   or not isinstance(candidate.get('target'), str)
                   or not candidate['target'].strip()
                   for candidate, row in zip(candidates, current))
            or any(not isinstance(row.get('Translation'), str)
                   or not row['Translation'].strip() for row in baseline)):
        raise ValueError('Cannot resume translation drafts: incomplete or unbound candidate lineage')
    return [row['Translation'] for row in baseline]


def _audible_resume_drafts(saved: dict) -> dict[str, str]:
    """Prefer the last audio-bound, complete fitted script over an unvoiced edit.

    These are starting drafts, NOT semantic acceptances. All current source
    gates run again and only a confirmed wording defect can replace the voice.
    Keeping a later unsynthesized audit edit would silently discard a correct
    audible draft after a false objection or an interrupted review.
    """
    drafts = {}
    for take in reversed(saved.get('attempts', [])):
        uid, text = take.get('unit_id'), take.get('text')
        if (isinstance(uid, str) and uid not in drafts and isinstance(text, str) and text.strip()
                and take.get('fits') is True and take.get('content_complete') is True
                and take.get('content_issues') == [] and take.get('readback_signature')
                and take.get('natural_audio_path') and take.get('candidate_sha256')
                and hash_path(take['natural_audio_path']) == take['candidate_sha256']):
            drafts[uid] = text
    return drafts


def _repeated_delivery_failure(recent: list[dict], reasons: list[str]) -> bool:
    """Two successful recognitions can justify another source proposal.

    A repeated registered-name defect may differ in ASR spelling elsewhere.
    Missing readback, leakage and boundary loss never become duration evidence
    or a reason to reword an otherwise unverified voice.
    """
    from core.source_verification import normalize_transcript
    if (len(recent) != 2 or not reasons or any(
            reason != 'readback_below_threshold' and not reason.startswith('spoken_name_mismatch:')
            for reason in reasons)):
        return False
    if any(not take.get('readback_signature') or not isinstance(take.get('readback'), str)
           or not take['readback'].strip() or take.get('content_issues') != reasons for take in recent):
        return False
    if any(reason.startswith('spoken_name_mismatch:') for reason in reasons):
        return True
    return normalize_transcript(recent[0]['readback']) == normalize_transcript(recent[1]['readback'])


def translation_speech_preflight(frame: pd.DataFrame, terminology: dict) -> pd.DataFrame:
    """Converge candidate translations using real natural takes, before review."""
    from core.tts_reference_plan import prepare_reference_plan
    from core.providers.asr_readback import verify_tasks_df
    from core.speech_budget import (budget_for, spoken_word_count, retranslate_unit, duration_context,
                                   audit_duration_candidate, with_source_category_terms)
    from core.translation_invariants import source_fact_conflicts
    from core.runtime_context import effective_target_language
    from core.homura_translation import record_homura_speech_result
    PREFLIGHT_REPORT.parent.mkdir(parents=True,exist_ok=True)
    plan = copy.deepcopy(load_plan())
    if list(frame.LineID) != [u["unit_id"] for u in plan["units"]]:
        raise ValueError("Translation preflight source mapping changed")
    if list(frame.Source) != [u["source"] for u in plan["units"]]:
        raise ValueError("Translation preflight source text changed")
    from core.tts_pronunciation import canonical_name_spellings
    name_changes = []
    frame = frame.copy()
    for idx, row in frame.iterrows():
        canonical, proof = canonical_name_spellings(row.Source, row.Translation,
            effective_target_language('en'), terminology.get('terms', []))
        if proof:
            name_changes.append({'unit_id':row.LineID, 'source':row.Source,
                'before':row.Translation, 'after':canonical, 'reason':'registered_display_name_before_audit', 'evidence':proof})
            frame.at[idx,'Translation'] = canonical
    input_rows = frame[["LineID", "Source", "Translation"]].to_dict("records")
    input_signature = signature(input_rows)
    contract, clock = synthesis_contract_signature(), unit_clock_signature(plan)
    previous = {}
    semantic_changes = name_changes
    if CANDIDATE_STATE.is_file():
        saved = json.loads(CANDIDATE_STATE.read_text(encoding="utf-8"))
        baseline = _saved_input_rows(saved, input_rows)
        if (saved.get("unit_clock_signature") == clock and
                (saved.get("input_signature") == input_signature or baseline is not None)):
            # Candidate wording is not accepted audio. Preserve it across a
            # fitting-policy change, then measure every take under this contract.
            previous = saved
            if [u["unit_id"] for u in saved["candidates"]] != list(frame.LineID):
                raise ValueError("Speech candidate checkpoint changed source identities")
            frame = frame.copy()
            audible_drafts = _audible_resume_drafts(saved)
            for i, idx in enumerate(frame.index):
                # A new reviewed input invalidates only its own candidate. All
                # retained drafts are measured and reviewed again below.
                if baseline is None or baseline[i] == input_rows[i]:
                    frame.at[idx, "Translation"] = audible_drafts.get(
                        saved['candidates'][i]['unit_id'], saved["candidates"][i]["target"])
    for unit, row in zip(plan["units"], frame.to_dict("records")):
        unit.update(target=row["Translation"], status="translated")
    # Primary drafts and resumed alternatives share the same source gate.
    # No duration search or voice synthesis can excuse missing source meaning.
    if load_key("translation.speech_budget.semantic_candidate_gate", True):
        from core.homura_translation import is_homura
        from core.llm_provider import get_llm_provider_config
        from core.duration_candidate_audit import source_category_terms
        from core.spoken_length import source_form_conflicts
        candidate_name=load_key("translation.speech_budget.candidate_provider", "")
        category_planning=(load_key("translation.speech_budget.source_category_guidance", False)
                           and effective_target_language("en").startswith("en") and candidate_name
                           and is_homura(get_llm_provider_config(str(candidate_name))))
        semantic_issues = []
        semantic_retries = max(0, int(load_key("translation.speech_budget.semantic_edit_retries", 2)))
        for idx, (unit, anchor) in enumerate(zip(plan["units"], input_rows)):
            print(f"Translation semantic/display preflight {idx + 1}/{len(plan['units'])}: {unit['unit_id']}", flush=True)
            conflicts = source_fact_conflicts(unit["source"], unit["target"], effective_target_language("en"))
            context=duration_context(unit, terminology)
            if category_planning and not conflicts:
                unit["source_category_terms"]=source_category_terms(unit["source"],context)
                context=with_source_category_terms(context,unit["source_category_terms"])
            for repair in range(semantic_retries + 1):
                conflicts = source_fact_conflicts(unit["source"], unit["target"], effective_target_language("en"))
                from core.tts_pronunciation import original_name_conflicts
                conflicts += original_name_conflicts(unit['source'], unit['target'], effective_target_language('en'), terminology.get('terms', []))
                from core.translation_review import is_valid_target_language
                if not is_valid_target_language(unit['target'], effective_target_language('en')):
                    conflicts.append('Return the complete source meaning in the requested target language; '
                                     'shared Han characters do not make Japanese or Korean text Chinese.')
                if effective_target_language("en").startswith("en"):
                    conflicts += source_form_conflicts(unit["source"], unit["target"], unit["duration"])
                try:
                    verdict = ({"equivalent":False,"issues":conflicts} if conflicts else
                               audit_duration_candidate(unit["source"], anchor["Translation"], unit["target"], context))
                except Exception as exc:
                    verdict = {"equivalent": False, "issues": [f"Source audit unavailable: {exc}"],
                               "failure_kind": "audit_unavailable"}
                    # No semantic judgment was returned. Keep the draft and
                    # report the operational failure instead of asking the
                    # translator to fix a fact that nobody found to be wrong.
                    break
                if verdict['equivalent']:
                    from core.bilingual_alignment import prepare_candidate_alignment
                    try:
                        prepare_candidate_alignment(unit, context)
                    except ValueError as exc:
                        verdict = {'equivalent': False, 'issues': [
                            'Bilingual display cannot partition this wording in source order. Preserve all meaning and '
                            'natural target speech, but keep clause/action order compatible with short corresponding '
                            f'subtitles. Rewrite the complete draft before synthesis. Alignment evidence: {exc}']}
                if verdict["equivalent"] or repair == semantic_retries:
                    break
                budget = budget_for({**unit, "line_id": unit["unit_id"]})
                budget["meaning_anchor"] = anchor["Translation"]
                budget["source_semantic_feedback"] = verdict
                try:
                    revised = retranslate_unit(unit["source"], unit["target"], budget, context)
                except Exception as exc:
                    verdict["repair_error"] = str(exc)
                    break
                # Ordered recovery can repair the display correspondence while
                # returning the identical wording. Reaudit it on the next
                # bounded iteration instead of retaining the obsolete failure.
                if revised != unit["target"]:
                    semantic_changes.append({"unit_id": unit["unit_id"], "source": unit["source"],
                        "before": unit["target"], "after": revised, "reason": "source_semantics_before_synthesis",
                        "source_verdict": verdict})
                    unit["target"] = revised
                    frame.at[frame.index[idx], "Translation"] = revised
            if not verdict["equivalent"]:
                semantic_issues.append({"unit_id":unit["unit_id"],"source":unit["source"],
                                        "target":unit["target"],"verdict":verdict})
            # Persist candidate wording before expensive synthesis. This is
            # only a draft checkpoint: resumed drafts still pass every gate.
            draft_state = {"input_signature": input_signature, "input_rows": input_rows,
                "synthesis_contract_signature": contract, "unit_clock_signature": clock,
                "candidate_policy_signature": candidate_policy_signature(),
                "candidates": [{"unit_id": u["unit_id"], "target": u["target"]} for u in plan["units"]],
                "changes": previous.get("changes", []) + semantic_changes,
                "attempts": previous.get("attempts", []), "status": "unaccepted_drafts"}
            draft_tmp = CANDIDATE_STATE.with_suffix(".tmp")
            draft_tmp.write_text(json.dumps(draft_state, ensure_ascii=False, indent=2), encoding="utf-8")
            draft_tmp.replace(CANDIDATE_STATE)
            report_tmp = PREFLIGHT_REPORT.with_suffix('.tmp')
            report_tmp.write_text(json.dumps({'version': 2, 'status': 'running',
                'stage': 'source_semantics_before_synthesis', 'processed_units': idx + 1,
                'total_units': len(plan['units']), 'issues': semantic_issues},
                ensure_ascii=False, indent=2), encoding='utf-8')
            report_tmp.replace(PREFLIGHT_REPORT)
            print(f"Translation semantic/display result {idx + 1}/{len(plan['units'])}: "
                  f"{unit['unit_id']} {'pass' if verdict['equivalent'] else 'needs_review'}", flush=True)
        if semantic_issues:
            PREFLIGHT_REPORT.write_text(json.dumps({"version":2,"status":"needs_review",
                "stage":"source_semantics_before_synthesis","issues":semantic_issues},
                ensure_ascii=False,indent=2),encoding="utf-8")
            frame.to_excel("output/log/translation_candidate.xlsx", index=False)
            raise RuntimeError(f"Translation source semantic gate needs review: {len(semantic_issues)} unit(s)")
    tasks = tasks_from_plan(plan)
    materialize_references(tasks)
    tasks = prepare_reference_plan(tasks)
    if previous.get("synthesis_contract_signature") == contract and PREFLIGHT_TASKS.is_file():
        cached_tasks = pd.read_excel(PREFLIGHT_TASKS)
        if list(cached_tasks.text) == list(tasks.text):
            validate_tasks(cached_tasks, plan)
            if list(cached_tasks.reference_plan_fingerprint) == list(tasks.reference_plan_fingerprint):
                tasks = cached_tasks
    max_speed = float(load_key("translation.speech_budget.max_speed_factor", 1.08))
    min_speed = float(load_key("translation.speech_budget.min_speed_factor", .90))
    rounds = int(load_key("translation.speech_budget.max_retries", 2))
    reviewed_translations = load_key("translation_exact_replacements", {}) or {}
    pending = list(tasks.index)
    # Each wording gets its own complete-speech retry allowance. Generating a
    # duration candidate must not consume its retakes before it is ever heard.
    from core.spoken_length import english_syllables
    short_tries = max(rounds + 1, min(8, int(load_key("translation.speech_budget.short_natural_takes", 6))))
    natural_tries = {idx: short_tries if (effective_target_language("en").startswith("en")
                    and float(row.duration) <= 1.2 and english_syllables(row.text) <= 3)
                    else rounds + 1 for idx, row in tasks.iterrows()}
    script_takes = {idx: 0 for idx in tasks.index}
    duration_edits = {idx: 0 for idx in tasks.index}
    exhausted_scripts = set()
    total_rounds = (rounds + 1) * short_tries
    changes, attempts = previous.get("changes", []) + semantic_changes, previous.get("attempts", [])
    prior_attempt_count = len(attempts)
    resume_takes = _preferred_natural_takes(tasks, attempts)
    base_seed = int(load_key('mlx_tts.backends.indextts2.seed', 42))
    prior_seed_offset = max([0] + [int(take['generation_settings']['seed']) - base_seed
        for take in attempts if isinstance(take.get('generation_settings'), dict)
        and isinstance(take['generation_settings'].get('seed'), int)])
    fit_metrics, failures = {}, {}
    checked_indices = set()
    def checkpoint(stage, round_number):
        CANDIDATE_STATE.write_text(json.dumps({"input_signature": input_signature,
            "input_rows": input_rows,
            "synthesis_contract_signature": contract, "unit_clock_signature": clock,
            "candidate_policy_signature": candidate_policy_signature(),
            "candidates": [{"unit_id": row.unit_id, "target": row.text} for row in tasks.itertuples()],
            "changes": changes, "attempts": attempts}, ensure_ascii=False, indent=2), encoding="utf-8")
        PREFLIGHT_REPORT.write_text(json.dumps({'version': 2, 'status': 'running',
            'stage': stage, 'round': round_number, 'batch_units': len(pending),
            'units': len(tasks), 'checked_units': len(checked_indices),
            'passed_units': sum(fit_metrics[i]['fits'] and i not in failures for i in checked_indices)},
            ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(total_rounds):
        rprint(f"[cyan]Translation speech preflight {attempt + 1}/{total_rounds}: {len(pending)} candidate take(s)[/cyan]")
        # Pandas copies attrs while constructing rows/frames. Only the compact
        # glossary is needed by TTS; copying the whole dialogue for every row
        # made reference planning and readback quadratic in document size.
        tasks.attrs["source_terminology"] = {"terms": terminology.get("terms", [])}
        checked_indices.difference_update(pending)
        checkpoint('speech_generation', attempt + 1)
        _generate(tasks, pending, seed_offset=attempt + (prior_seed_offset if attempt else 0),
                  pronunciation_retries={idx for idx in failures if script_takes[idx] > 0},
                  resume_takes=resume_takes if attempt == 0 else None,
                  excluded_takes={uid: [take['candidate_sha256'] for take in attempts[prior_attempt_count:]
                      if take.get('unit_id') == uid and take.get('candidate_sha256')
                      and (take.get('fits') is False or take.get('content_complete') is False)]
                      for uid in tasks.unit_id})
        batch_attempts = {}
        for idx in pending:
            script_takes[idx] += 1
            row = tasks.loc[idx]
            output = Path("output/audio/segs") / f"{int(row['number'])}_0.wav"
            fit = fit_take(_raw_path(row.unit_id), output, float(row.duration),
                           max_speed=max_speed, min_speed=min_speed)
            fit_metrics[idx] = fit
            tasks.at[idx, "real_dur"] = fit["natural_seconds"]
            tasks.at[idx, "speed_factor"] = fit["speed_factor"]
            tasks.at[idx, "speech_seconds"] = fit.get("speech_seconds", fit["natural_seconds"])
            tasks.at[idx, "planned_trailing_silence"] = fit.get("trailing_silence_seconds", 0.)
            tasks.at[idx, "translation_max_speed_factor"] = max_speed
            receipt_path = _raw_path(row.unit_id).with_suffix(".conditioning.json")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
            attempts.append({"unit_id": row.unit_id, "text": row.text, "round": attempt,
                             "candidate_audio": receipt.get("selected_candidate"),
                             "natural_audio_path": receipt.get("selected_candidate") or str(_raw_path(row.unit_id)),
                             "candidate_sha256": hash_path(_raw_path(row.unit_id)),
                             "conditioning_receipt_sha256": hash_path(receipt_path),
                             "conditioning_signature": signature({"synthesis_contract":contract,
                                 "reference":row.get("reference_plan_fingerprint")}),
                             "content_complete": None,
                             "generation_settings": receipt.get("generation_settings"), **fit})
            batch_attempts[idx] = attempts[-1]
        # Preserve expensive natural takes and measured progress even if the
        # independent readback runtime is unavailable on this machine.
        tasks.to_excel(PREFLIGHT_TASKS, index=False)
        checkpoint('content_readback', attempt + 1)
        tasks, readback = verify_tasks_df(tasks)
        # Keep the independent witness even if a later acceptance check fails.
        tasks.to_excel(PREFLIGHT_TASKS, index=False)
        failures = _content_issues(tasks)
        for idx, take in batch_attempts.items():
            row = tasks.loc[idx]
            transcript = row.get("asr_transcript")
            transcript = transcript if isinstance(transcript, str) else None
            take.update(content_complete=idx not in failures,
                        content_issues=failures.get(idx, []), readback=transcript,
                        readback_signature=signature({"expected_text":row.text,
                            "fitted_audio_sha256":hash_path(f"output/audio/segs/{int(row.number)}_0.wav"),
                            "transcript":transcript,"status":str(row.get("asr_status","unknown"))}))
            record_homura_speech_result(row.unit_id, row.text, take)
        checked_indices.update(fit_metrics)
        checkpoint('duration_candidates', attempt + 1)
        overlong = {idx for idx, fit in fit_metrics.items() if not fit["fits"]}
        PREFLIGHT_TASKS.parent.mkdir(parents=True, exist_ok=True)
        tasks.to_excel(PREFLIGHT_TASKS, index=False)
        if not failures and not overlong:
            break
        if attempt + 1 >= total_rounds:
            break
        pending = []
        # Missing/misread speech regenerates the identical script. Its duration
        # cannot calibrate translation, even if that partial take also overran.
        for idx in sorted(set(failures) | overlong):
            row = tasks.loc[idx]
            unit = plan['units'][idx]
            cached_draft = _cached_duration_draft(row, tasks, unit, duration_context(unit,terminology),
                                                attempts[prior_attempt_count:])
            if cached_draft:
                changes.append({'unit_id':unit['unit_id'],'source':unit['source'],
                    'before':row.text,'after':cached_draft,'reason':'current_source_reviewed_cached_duration_candidate'})
                tasks.at[idx,'text'] = cached_draft
                tasks.at[idx,'lines'] = [cached_draft]
                unit['target'] = cached_draft
                frame.at[idx,'Translation'] = cached_draft
                script_takes[idx] = 0
                pending.append(idx)
                continue
            if idx in failures:
                if script_takes[idx] < natural_tries[idx]:
                    pending.append(idx)
                elif (duration_edits[idx] < rounds
                      and plan['units'][idx]['source'] not in reviewed_translations):
                    # Consistent lexical misdelivery is not repaired by more
                    # seeds. Use bounded source retranslation without treating
                    # a failed content witness as duration calibration.
                    recent = [a for a in attempts[prior_attempt_count:] if a.get('unit_id') == row.unit_id
                              and a.get('text') == row.text and a.get('content_complete') is False][-2:]
                    if not _repeated_delivery_failure(recent, failures[idx]):
                        continue
                    unit = plan['units'][idx]
                    budget = budget_for({**unit, 'line_id':unit['unit_id']})
                    budget['speech_delivery_feedback'] = {'expected':row.text,
                        'repeated_readbacks':[a['readback'] for a in recent], 'issues':failures[idx]}
                    try:
                        revised = retranslate_unit(unit['source'], row.text, budget,
                                                   duration_context(unit, terminology))
                    except Exception as exc:
                        attempts.append({'unit_id':unit['unit_id'], 'status':'needs_review', 'error':str(exc)})
                        continue
                    if revised == row.text:
                        continue
                    changes.append({'unit_id':unit['unit_id'], 'source':unit['source'],
                        'before':row.text, 'after':revised, 'reason':'consistent_lexical_readback_failure',
                        'readback_evidence':[a['readback'] for a in recent]})
                    tasks.at[idx,'text'] = revised
                    tasks.at[idx,'lines'] = [revised]
                    unit['target'] = revised
                    frame.at[idx,'Translation'] = revised
                    duration_edits[idx] += 1
                    script_takes[idx] = 0
                    pending.append(idx)
                continue
            if fit_metrics[idx].get('retry_unchanged_script'):
                if script_takes[idx] < natural_tries[idx]:
                    pending.append(idx)
                continue
            # A reviewed run-specific script is not permission to override its
            # wording with another model draft. It still needs a complete take
            # that passes the same duration and readback gates.
            if (idx in overlong and plan["units"][idx]["source"] not in reviewed_translations
                    and duration_edits[idx] < rounds and (idx, row.text) not in exhausted_scripts):
                marginal = float(load_key("translation.speech_budget.natural_variant_margin",1.08))
                if (script_takes[idx] < min(3, natural_tries[idx])
                        and fit_metrics[idx]["required_speed"] <= max_speed * marginal):
                    pending.append(idx)
                    continue
                unit = plan["units"][idx]
                budget = budget_for({"line_id": unit["unit_id"], "duration": unit["duration"]})
                take = _duration_calibration_take(row, batch_attempts[idx], attempts)
                # atempo's analysis window can exceed the slot even when the
                # natural ratio is just below the cap. Its measured overhead
                # guides a smaller candidate; it never licenses tail cropping.
                fit_cap = take.get("calibrated_max_natural_seconds")
                if isinstance(fit_cap, (int, float)) and 0 < fit_cap < budget["max_natural_speech_seconds"]:
                    budget["max_natural_speech_seconds"] = fit_cap
                current_count = spoken_word_count(row.text)
                measured = take["natural_seconds"]
                budget["target_max_words"] = max(1, min(current_count - 1,
                    int(math.ceil(current_count * unit["duration"] * max_speed / measured))))
                budget["meaning_anchor"] = input_rows[idx]["Translation"]
                budget["source_category_terms"]=unit.get("source_category_terms")
                budget["duration_witness"] = {"text":row.text,"content_complete":True,
                    "natural_seconds":measured,"natural_audio_sha256":take["candidate_sha256"],
                    "conditioning_signature":take["conditioning_signature"],
                    "readback_signature":take["readback_signature"]}
                context = duration_context(unit,terminology)
                try:
                    measured_candidates = [a for a in attempts if a.get("unit_id") == unit["unit_id"]
                                           and a.get("fits") is False and a.get("content_complete") is True
                                           and a.get("conditioning_signature") == take["conditioning_signature"]
                                           and a.get("natural_audio_path")
                                           and hash_path(a["natural_audio_path"]) == a.get("candidate_sha256")
                                           and isinstance(a.get("text"), str)]
                    revised = retranslate_unit(unit["source"], row.text, budget, context,
                                               measured_seconds=measured, measured_candidates=measured_candidates)
                except Exception as exc:
                    attempts.append({"unit_id": unit["unit_id"], "status": "needs_review", "error": str(exc)})
                    exhausted_scripts.add((idx, row.text))
                    # A safe shorter paraphrase may not exist. Try another
                    # complete natural take of the unchanged wording instead.
                    if script_takes[idx] < natural_tries[idx]:
                        pending.append(idx)
                    continue
                if revised == row.text:
                    if script_takes[idx] < natural_tries[idx]:
                        pending.append(idx)
                    continue
                changes.append({"unit_id": unit["unit_id"], "source": unit["source"],
                                "before": row.text, "after": revised, "reason": "measured_translation_duration"})
                tasks.at[idx, "text"] = revised
                tasks.at[idx, "lines"] = [revised]
                plan["units"][idx]["target"] = revised
                frame.at[idx, "Translation"] = revised
                duration_edits[idx] += 1
                script_takes[idx] = 0
                if (effective_target_language("en").startswith("en") and float(row.duration) <= 1.2
                        and english_syllables(revised) <= 3):
                    natural_tries[idx] = short_tries
            if script_takes[idx] < natural_tries[idx]:
                pending.append(idx)
        if not pending:
            break
    issues = [{"unit_id": tasks.loc[idx, "unit_id"], "reasons": failures.get(idx, []) +
               ([fit_metrics[idx].get("timing_issue", "natural_duration_over_budget")]
                if not fit_metrics[idx]["fits"] else []),
               "source": tasks.loc[idx, "origin"], "target": tasks.loc[idx, "text"],
               "next_action": "requires_speech_review" if idx in failures else "requires_plan_review",
               "transcript": tasks.loc[idx].get("asr_transcript") if isinstance(tasks.loc[idx].get("asr_transcript"), str) else None,
               **fit_metrics[idx]}
              for idx in sorted(set(failures) | {i for i, fit in fit_metrics.items() if not fit["fits"]})]
    rates = [spoken_word_count(tasks.loc[i, "text"]) / fit["natural_seconds"]
             for i, fit in fit_metrics.items() if fit["natural_seconds"] > 1 and i not in failures]
    if synthesis_contract_signature() != contract:
        issues.append({"unit_id": None, "reasons": ["synthesis_contract_changed_during_run"],
                       "next_action": "restart_translation_preflight"})
    report = {"version": 2, "status": "needs_review" if issues else "pass", "stage": "translation_before_acceptance",
              "units": len(tasks), "max_speed_factor": max_speed, "changes": changes,
              "synthesis_contract_signature": contract,
              "unit_clock_signature": unit_clock_signature(plan),
              "source_category_terms":{u["unit_id"]:u.get("source_category_terms",[]) for u in plan["units"]},
              "attempts": attempts, "issues": issues, "readback": readback.__dict__,
              "measured_median_words_per_second": statistics.median(rates) if rates else None,
              "translation_signature": signature(frame[["LineID", "Source", "Translation"]].to_dict("records")),
              "audio": [{"unit_id": row.unit_id, "path": f"output/audio/segs/{int(row.number)}_0.wav",
                         "sha256": hash_path(f"output/audio/segs/{int(row.number)}_0.wav"),
                         "raw_path": str(_raw_path(row.unit_id)), "raw_sha256": hash_path(_raw_path(row.unit_id)),
                         **fit_metrics[idx]} for idx, row in tasks.iterrows()]}
    PREFLIGHT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tasks["new_sub_times"] = [[[u["start"], u["end"]]] for u in plan["units"]]
    tasks.to_excel(PREFLIGHT_TASKS, index=False)
    frame.to_excel("output/log/translation_candidate.xlsx", index=False)
    if issues:
        raise RuntimeError(f"Translation speech preflight needs review: {len(issues)} unit(s); see {PREFLIGHT_REPORT}")
    return frame


def verify_preflight_identity(frame: pd.DataFrame) -> None:
    report = json.loads(PREFLIGHT_REPORT.read_text(encoding="utf-8"))
    if report["status"] != "pass" or report.get("issues") or report["translation_signature"] != signature(frame[["LineID", "Source", "Translation"]].to_dict("records")):
        raise ValueError("Translation changed after measured speech validation; repeat preflight before acceptance")
    if report.get("version") != 2 or report.get("synthesis_contract_signature") != synthesis_contract_signature():
        raise ValueError("Speech synthesis configuration changed; repeat translation speech preflight")
    if report.get("unit_clock_signature") != unit_clock_signature(load_plan()):
        raise ValueError("Utterance clock changed after measured speech validation")
    for take in report["audio"]:
        if not take.get("sha256") or hash_path(take.get("path", "")) != take["sha256"]:
            raise ValueError(f"Validated speech changed at {take['unit_id']}")
        if not take.get("raw_sha256") or hash_path(take.get("raw_path", "")) != take["raw_sha256"]:
            raise ValueError(f"Validated natural speech changed at {take['unit_id']}")


def preflight_validated_units(tasks: pd.DataFrame) -> set[str]:
    """Read-only proof used by quality evaluation; no inference from filenames."""
    if "unit_id" not in tasks or not PREFLIGHT_REPORT.is_file():
        return set()
    frame = pd.DataFrame({"LineID": tasks.unit_id, "Source": tasks.origin, "Translation": tasks.text})
    try:
        verify_preflight_identity(frame)
        validate_tasks(tasks, load_plan(), generated=True)
    except (OSError, ValueError, KeyError):
        return set()
    return set(tasks.unit_id)


def create_tasks() -> None:
    plan = load_plan()
    tasks = pd.read_excel(PREFLIGHT_TASKS)
    validate_tasks(tasks, plan, generated=True)
    path = Path("output/audio/tts_tasks.xlsx")
    if path.is_file():
        validate_tasks(pd.read_excel(path), plan)
        return
    tasks.to_excel(path, index=False)


def reuse_locked_speech() -> None:
    """No rewrite, remerge or second split after translation acceptance."""
    from core.providers.asr_readback import verify_tasks_df
    from core.dubbing_quality import write_dubbing_eval
    plan = load_plan()
    frame = pd.DataFrame([{"LineID": u["unit_id"], "Source": u["source"], "Translation": u["target"]} for u in plan["units"]])
    verify_preflight_identity(frame)
    tasks = pd.read_excel("output/audio/tts_tasks.xlsx")
    validate_tasks(tasks, plan, generated=True)
    tasks, summary = verify_tasks_df(tasks)
    failures = _content_issues(tasks)
    if failures:
        raise RuntimeError(f"Locked speech readback failed for {len(failures)} unit(s)")
    tasks.to_excel("output/audio/tts_tasks.xlsx", index=False)
    for unit, row in zip(plan["units"], tasks.to_dict("records")):
        unit.update(status="synthesized", audio_path=f"output/audio/segs/{int(row['number'])}_0.wav",
                    audio_sha256=hash_path(f"output/audio/segs/{int(row['number'])}_0.wav"),
                    speed_factor=float(row["speed_factor"]), readback=row["asr_transcript"])
    plan["status"] = "synthesized"
    save_plan(plan)
    result = write_dubbing_eval(tasks)
    if not result.get('quality_gate', {}).get('passed'):
        raise RuntimeError(f"Locked speech quality gate failed: {result.get('quality_gate')}")
    rprint(f"[green]Reused {len(tasks)} locked translation takes; readback: {summary.status}; quality: {result}[/green]")
