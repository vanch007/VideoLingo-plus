"""Display subtitles from locked utterances and measured target word timing."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess

import pandas as pd

from core.config_utils import load_key
from core.pipeline.artifact_manifest import hash_path
from core.source_verification import normalize_transcript
from core.subtitle_layout import split_display_text
from core.utterance_plan import load_plan, save_plan, signature


def attach_source_words(plan):
    words = pd.read_excel('output/log/cleaned_chunks.xlsx')
    for unit in plan['units']:
        selected = words.iloc[unit['word_start']:unit['word_end'] + 1]
        unit['source_words'] = [{'word': str(row.text).strip().strip('"'),
                                 'start': float(row.start), 'end': float(row.end)}
                                for row in selected.itertuples()]


def source_boundaries(unit):
    offset, bounds = 0, {0: float(unit.get('source_start', unit['start']))}
    for word in unit.get('source_words', []):
        if offset:
            bounds[offset] = float(word['start'])
        offset += len(normalize_transcript(word['word']))
    if offset != len(normalize_transcript(unit['source'])):
        raise ValueError(f"Source word evidence does not cover locked text at {unit['unit_id']}")
    bounds[offset] = float(unit.get('source_end', unit['end']))
    return bounds


def display_clock_signature(plan, *, dubbed):
    if dubbed:
        from core.bilingual_alignment import target_clock_signature
        return target_clock_signature(plan)
    return signature([{k: u[k] for k in ('unit_id', 'source_start', 'source_end', 'source_words')}
                      for u in plan['units']])


def align_target_words(plan: dict) -> dict:
    from core.runtime_context import effective_target_language
    language = effective_target_language("en")
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/align_dubbing_units.py"
    model = str(load_key("source_asr.aligner", "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"))
    from core.tts_pronunciation import spoken_plan, pronunciation_terms, pronunciation_policy_signature, display_aligned_words
    terms = pronunciation_terms()
    pronunciation = {u['unit_id']: spoken_plan(u['target'], language, terms) for u in plan['units']}
    manifest = [{"unit_id": u["unit_id"], "text": pronunciation[u['unit_id']]['spoken_text'],
                 "audio": str(Path(f"output/audio/segs/{i + 1}_0.wav").resolve()),
                 "audio_sha256": hash_path(f"output/audio/segs/{i + 1}_0.wav")}
                for i, u in enumerate(plan["units"])]
    fingerprint = signature({"manifest": manifest, "model": model,
                             "display_texts": [u['target'] for u in plan['units']],
                             "pronunciation_policy": pronunciation_policy_signature(terms),
                             "language": language, "worker": hash_path(script)})
    path = Path("output/log/target_word_alignment.json")
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved.get("fingerprint") == fingerprint:
            return saved
    inp = Path("output/log/target_alignment_input.json")
    inp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    names = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "fr": "French",
             "de": "German", "it": "Italian", "pt": "Portuguese", "ru": "Russian", "es": "Spanish"}
    if language not in names:
        raise ValueError(f"Target word alignment has no verified language route for {language}")
    completed = subprocess.run([str(load_key("source_asr.python")), str(script), "--manifest", str(inp.resolve()),
                                "--model", model, "--language", names[language]],
                               capture_output=True, text=True, timeout=600,
                               env={**os.environ, "HF_HUB_OFFLINE": "1"})
    if completed.returncode:
        raise RuntimeError(f"Target speech alignment failed: {completed.stderr[-1500:]}")
    aligned = json.loads(completed.stdout)
    if [item["unit_id"] for item in aligned] != [u["unit_id"] for u in plan["units"]]:
        raise ValueError("Target word alignment changed utterance identities")
    for item, unit in zip(aligned, plan["units"]):
        spoken = pronunciation[unit['unit_id']]
        if normalize_transcript("".join(w["word"] for w in item["words"])) != normalize_transcript(spoken['spoken_text']):
            raise ValueError(f"Target aligner changed locked text at {unit['unit_id']}")
        if not item["words"] or any(not math.isfinite(float(w[k])) for w in item["words"] for k in ("start", "end")):
            raise ValueError(f"Target word timing missing at {unit['unit_id']}")
        unit['spoken_words'] = item['words']
        unit['spoken_text'] = spoken['spoken_text']
        unit['pronunciation_plan'] = spoken
        item['words'] = display_aligned_words(spoken, item['words'])
        unit["target_words"] = item["words"]
    saved = {"status": "pass", "fingerprint": fingerprint, "model": model,
             "timing_basis": "target_audio_forced_alignment_80ms", "units": aligned}
    path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    save_plan(plan)
    return saved


def layout_plan(plan: dict, *, chars=38, source_chars=22, bilingual=True,
                bilingual_alignment=None, dubbed=True) -> tuple[pd.DataFrame, dict]:
    """Display meaning pairs on the clock of the speech present in the video."""
    from core.bilingual_alignment import align_unit, validate_display_pairs, target_boundaries, display_line_limit
    by_id = {u['unit_id']: u for u in (bilingual_alignment or {}).get('units', [])}
    records, issues = [], []
    for unit in plan["units"]:
        if bilingual:
            aligned = by_id.get(unit['unit_id'])
            if aligned is None:
                if len(unit['target']) > chars or len(unit['source']) > source_chars:
                    raise ValueError(f"Bilingual semantic alignment is required at {unit['unit_id']}")
                pairs = align_unit(unit, chars=chars, source_chars=source_chars)
            else:
                if (aligned.get('source') != unit['source'] or aligned.get('target') != unit['target']
                        or aligned.get('status') != 'pass'):
                    raise ValueError('Bilingual alignment does not match the locked utterance')
                pairs = validate_display_pairs(unit, aligned, chars=chars, source_chars=source_chars, require_review=True)
            boundaries = target_boundaries(unit) if dubbed else source_boundaries(unit)
            for n, pair in enumerate(pairs, 1):
                field = 'target' if dubbed else 'source'
                left, right = pair[field + '_char_span']
                at, stop = (len(normalize_transcript(unit[field][:index])) for index in (left, right))
                def _snap_b(pos, min_val=0):
                    if pos in boundaries and pos >= min_val:
                        return pos
                    candidates = [k for k in boundaries if k >= min_val]
                    closest = min(candidates, key=lambda k: abs(k - pos)) if candidates else pos
                    return closest if abs(closest - pos) <= 8 else pos
                at = _snap_b(at, 0)
                stop = _snap_b(stop, at + 1)
                if at not in boundaries or stop not in boundaries:
                    raise ValueError(f"Subtitle boundary lacks measured {field} word timing at {unit['unit_id']}")
                start, end = boundaries[at], boundaries[stop]
                if not math.isfinite(start) or not math.isfinite(end) or end - start < .08 - 1e-6:
                    raise ValueError(f"Collapsed {field} subtitle boundary at {unit['unit_id']}")
                if len(pair['target']) / (end - start) > float(load_key('subtitle.max_cps', 20)):
                    issues.append({'unit_id': unit['unit_id'], 'part': n, 'reason': 'reading_speed'})
                records.append({'UnitID': unit['unit_id'], 'LineID': unit['unit_id'],
                                'parent_line_id': unit['parent_line_id'], 'display_part': n,
                                'Source': pair['source'], 'Translation': pair['target'],
                                'timestamp': (start, end), 'duration': end - start,
                                'speaker': unit.get('speaker'),
                                'source_char_span': pair['source_char_span'], 'target_char_span': pair['target_char_span'],
                                'display_alignment': 'verified_bilingual_semantic_pair+measured_' + field + '_words'})
            continue
        target = split_display_text(unit["target"], chars)
        if any(len(p) > chars for p in target):
            raise ValueError(f"An unbreakable subtitle token exceeds its line limit at {unit['unit_id']}")
        target = ["\n".join(target[i:i + 2]) for i in range(0, len(target), 2)]
        count = len(target)
        bounds = [float(unit["start"])]
        words = unit.get("target_words", [])
        cursor = 0
        for part in target[:-1]:
            cursor += len(normalize_transcript(part))
            position = 0
            at = None
            for word in words:
                if position >= cursor:
                    at = float(unit["start"]) + float(word["start"])
                    break
                position += len(normalize_transcript(word["word"]))
            if at is None:
                raise ValueError(f"Target caption boundary lacks word timing at {unit['unit_id']}")
            # Keep display partitions positive on the aligner's coarse grid.
            if not bounds[-1] + .08 <= at < unit["end"]:
                raise ValueError(f"Collapsed target subtitle boundary at {unit['unit_id']}")
            bounds.append(at)
        bounds.append(float(unit["end"]))
        for n in range(count):
            src = unit["source"]
            trans = target[n]
            start, end = bounds[n], bounds[n + 1]
            if len(trans.replace("\n", "")) / max(end - start, .001) > float(load_key("subtitle.max_cps", 20)):
                issues.append({"unit_id": unit["unit_id"], "part": n + 1, "reason": "reading_speed"})
            records.append({"UnitID": unit["unit_id"], "LineID": unit["unit_id"],
                            "parent_line_id": unit["parent_line_id"], "display_part": n + 1,
                            "Source": src, "Translation": trans, "timestamp": (start, end),
                            "duration": end - start, "speaker": unit.get("speaker"),
                            "display_alignment": "measured_target_words; monolingual"})
    report = {"status": "needs_review" if issues else "pass", "semantic_rows": len(plan["units"]),
              "display_cues": len(records), "max_chars_per_line": chars,
              "max_source_chars_per_line": source_chars, "max_lines": display_line_limit(), "issues": issues,
              "bilingual_semantic_alignment": 'pass' if bilingual else 'not_applicable',
              "bilingual_alignment_fingerprint": (bilingual_alignment or {}).get('fingerprint'),
              "display_clock": 'target_speech' if dubbed else 'source_speech',
              "alignment_basis": "locked_semantic_pairs_and_" + ('target' if dubbed else 'source') + "_audio_word_timing"
                                 if bilingual else "target_audio_words"}
    return pd.DataFrame(records), report


def _format_srt_text(val: str, *, limit=None, max_lines=2) -> str:
    """Wrap only. Never discard words to make a subtitle fit."""
    val = str(val).strip()
    if not val:
        return ""
    if limit is None:
        limit = 22 if re.search(r"[\u3400-\u9fff\u3040-\u30ff]", val) else 42
    parts = split_display_text(val, limit)
    if (max_lines is not None and len(parts) > max_lines) or any(len(p) > limit for p in parts):
        raise ValueError("Subtitle needs timed segmentation before export; text cannot be truncated")
    if re.sub(r"\s+", "", "".join(parts)) != re.sub(r"\s+", "", val):
        raise ValueError("Subtitle wrapping changed locked text")
    return "\n".join(parts)


def _write_srt(frame, path: Path, fields, *, display=True) -> None:
    from core.step6_generate_final_timeline import convert_to_srt_format
    from core.bilingual_alignment import display_line_limit
    blocks = []
    for i, row in frame.iterrows():
        cue_lines = []
        for f in fields:
            raw = str(row[f]).strip()
            if not raw:
                continue
            cue_lines.append(_format_srt_text(raw, max_lines=display_line_limit()) if display else raw)
        text = chr(10).join(cue_lines)
        if not text:
            raise ValueError("Empty visual cue")
        time_str = convert_to_srt_format(*row.timestamp)
        blocks.append(str(i + 1) + chr(10) + time_str + chr(10) + text + chr(10))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(blocks), encoding="utf-8")


def render_plan_subtitles(*, dubbed=False) -> None:
    plan = load_plan()
    aligned = align_target_words(plan)
    by_id = {u["unit_id"]: u["words"] for u in aligned["units"]}
    for unit in plan["units"]:
        unit["target_words"] = by_id[unit["unit_id"]]
    from core.bilingual_alignment import ensure_bilingual_alignment, ALIGNMENT_PATH
    from core.bilingual_alignment import display_limits
    chars, source_chars = display_limits()
    # Original-audio captions require the source clock. Dubbing uses the target
    # clock and may be rendered from a self-contained accepted plan.
    if not dubbed or all("word_start" in u and "word_end" in u for u in plan["units"]):
        attach_source_words(plan)
    paired = ensure_bilingual_alignment(plan, chars=chars, source_chars=source_chars)
    frame, report = layout_plan(plan, chars=chars, source_chars=source_chars,
                                bilingual_alignment=paired, dubbed=dubbed)
    output = Path("output")
    names = [("src.srt", ["Source"]), ("trans.srt", ["Translation"]),
             ("src_trans.srt", ["Source", "Translation"]), ("trans_src.srt", ["Translation", "Source"])]
    if dubbed:
        names = [("dub_orig.srt", ["Source"]), ("dub.srt", ["Translation"]),
                 ("dub_bilingual.srt", ["Source", "Translation"])]
    for name, fields in names:
        _write_srt(frame, output / name, fields)
    report['display_clock_signature'] = display_clock_signature(plan, dubbed=dubbed)
    report['bilingual_alignment_sha256'] = hash_path(ALIGNMENT_PATH)
    report['display_artifacts'] = {str(output / name): hash_path(output / name) for name, _ in names}
    timeline_path = output / ('audio/dub_display_timeline.xlsx' if dubbed else 'final_timeline.xlsx')
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(timeline_path, index=False)
    (output / "log" / ("dub_subtitle_layout.json" if dubbed else "subtitle_layout.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    atomic = pd.DataFrame([{"LineID": u["unit_id"], "Source": u["source"], "Translation": u["target"],
                           "duration": u["duration"], "timestamp": (u["start"], u["end"])} for u in plan["units"]])
    if not dubbed:
        for name, fields in (("src_subs_for_audio.srt", ["Source"]), ("trans_subs_for_audio.srt", ["Translation"])):
            _write_srt(atomic, output / "audio" / name, fields, display=False)
        atomic.to_excel(output / "audio/final_timeline.xlsx", index=False)
