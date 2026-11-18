"""Preserve local ASR text, optionally review it, then measure word alignment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from core.asr_schema import normalize_asr_result, validate_word_timestamps
from core.all_whisper_methods.qwen3_aligned import merge_zero_duration_words
from core.config_utils import load_key
from core.pipeline.artifact_manifest import hash_path
from core.source_verification import normalize_transcript


def apply_source_review(segments, review, *, audio_sha256, start, end):
    """Apply explicit timed source corrections with matched audio and receipts.

    This is a review import, not model recognition. Every edit must match the
    actual ASR text and identify its evidence; it never modifies the raw receipt.
    """
    if review.get("audio_sha256") != audio_sha256:
        raise ValueError("Source review belongs to different original audio")
    edits, updated = [], [dict(s) for s in segments]
    for edit in review.get("edits", []):
        left, right = float(edit["start"]), float(edit["end"])
        if right <= start or left >= end:
            continue
        if not edit.get("evidence") or not str(edit.get("after", "")).strip():
            raise ValueError("Source edit requires nonempty reviewed text and evidence")
        matches = [i for i, s in enumerate(updated)
                   if left <= (float(s["start"]) + float(s["end"])) / 2 + start <= right]
        before = "".join(updated[i]["text"] for i in matches)
        if not matches or normalize_transcript(before) != normalize_transcript(edit["before"]):
            raise ValueError(f"Source edit no longer matches raw ASR: {edit.get('id', left)}")
        first, last = matches[0], matches[-1]
        replacement = {**updated[first], "text": edit["after"], "end": updated[last]["end"],
                       "review_id": edit.get("id"), "source_evidence": edit["evidence"]}
        updated[first:last + 1] = [replacement]
        edits.append({**edit, "observed_before": before})
    return updated, edits


def bind_recognition_words(raw_words, segments, start, language):
    # Tag characters before coalescing zero-width alignment records. Otherwise
    # a collapsed last character can migrate into the next recognized phrase.
    raw_words = [dict(w) for w in raw_words]
    from bisect import bisect_right
    offsets = [0]
    for segment in segments:
        offsets.append(offsets[-1] + len(normalize_transcript(segment["text"])))
    position = 0
    for word in raw_words:
        if segments:
            index = min(len(segments) - 1, bisect_right(offsets, position) - 1)
            segment = segments[index]
            word.update(recognition_segment_id=f"{start:.3f}-{index}", recognition_text=segment["text"],
                        recognition_start=float(segment["start"]) + start,
                        recognition_end=float(segment["end"]) + start)
        position += len(normalize_transcript(word["word"]))
    words = merge_zero_duration_words(raw_words, language=language)
    if not words:
        raise ValueError("Forced alignment returned no spoken words")
    if segments and normalize_transcript("".join(w["word"] for w in words)) != normalize_transcript("".join(s["text"] for s in segments)):
        raise ValueError("Forced alignment changed recognized source text; rejected")
    # Alignment owns timing, never recognition text. In particular, joining
    # stripped English aligner tokens erases spaces and punctuation upstream of
    # translation. Verify each phrase without replacing its canonical wording.
    for segment_id in {w.get("recognition_segment_id") for w in words}:
        phrase = [w for w in words if w.get("recognition_segment_id") == segment_id]
        if phrase and segment_id and normalize_transcript("".join(w["word"] for w in phrase)) != normalize_transcript(phrase[0]["recognition_text"]):
            raise ValueError("Forced alignment crossed a recognized phrase; realignment required")
    validate_word_timestamps({'segments': [{'text': ' '.join(w['word'] for w in words), 'words': words}]},
                             backend='source-alignment')
    return words


def _weights_identity(model) -> str | None:
    """Size+mtime of every weight file of a local model dir, so replacing weights in place changes
    the cache key. Hub ids (not a directory) are identified by their name."""
    if not model:
        return None
    path = Path(str(model)).expanduser()
    if not path.is_dir():
        return str(model)
    return "|".join(f"{f.name}:{f.stat().st_size}:{int(f.stat().st_mtime)}"
                    for f in sorted(path.glob("*.safetensors")))


def _global_key(fingerprint, start, end, cfg):
    """Cross-task cache key for one chunk, or None when human review shaped the result."""
    if cfg.get("review_file"):
        return None
    from core import asset_cache
    return asset_cache.make_key(kind="asr_chunk", fingerprint=fingerprint, start=round(start, 3), end=round(end, 3),
                                models=[_weights_identity(cfg.get("model")), _weights_identity(cfg.get("aligner"))])


def _global_get(key):
    if not key:
        return None
    from core import asset_cache
    hit = asset_cache.lookup("asr_chunk", key, ["receipt.json"])
    if not hit:
        return None
    try:
        return json.loads(hit["files"]["receipt.json"].read_text())
    except (OSError, ValueError):
        return None


def _global_put(key, receipt_path):
    if key and Path(receipt_path).is_file():
        from core import asset_cache
        asset_cache.store("asr_chunk", key, {"receipt.json": str(receipt_path)})


def transcribe_audio(audio_file: str, start: float, end: float) -> dict:
    from core.runtime_context import effective_source_language
    language = effective_source_language("zh")
    names = {"zh": "Chinese", "en": "English", "yue": "Cantonese", "fr": "French",
             "de": "German", "it": "Italian", "ja": "Japanese", "ko": "Korean",
             "pt": "Portuguese", "ru": "Russian", "es": "Spanish"}
    if language not in names:
        raise ValueError(f"Local forced alignment does not support {language}; choose a native Whisper runtime")
    if end - start > 300:
        raise ValueError("Local alignment requires source chunks of at most 300 seconds")
    cfg = dict(load_key("source_asr", {}) or {})
    project = Path(__file__).resolve().parents[2]
    output = Path("output/log/local_source_asr")
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / f"{start:.3f}-{end:.3f}"
    clip, transcript = Path(str(prefix) + ".wav"), Path(str(prefix) + ".txt")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", audio_file,
                    "-t", str(end - start), "-ar", "16000", "-ac", "1", str(clip)], check=True)
    source_hash = hash_path("output/audio/raw_master.wav")
    review_path = Path(str(cfg.get("review_file") or ""))
    fingerprint = {"audio_sha256": hash_path(clip), "source_sha256": source_hash,
                   "language": language, "config": cfg,
                   "review_sha256": hash_path(review_path) if cfg.get("review_file") else None,
                   "worker_sha256": hash_path(project / "scripts/recognize_source_audio.py"),
                   "align_worker_sha256": hash_path(project / "scripts/align_source_audio.py"),
                   "adapter_sha256": hash_path(Path(__file__)),
                   "collapsed_words_sha256": hash_path(project / "core/all_whisper_methods/qwen3_aligned.py")}
    receipt = Path(str(prefix) + ".json")
    if receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved.get("fingerprint") == fingerprint:
            return saved["result"]
    global_key = _global_key(fingerprint, start, end, cfg)
    cached = _global_get(global_key)
    if cached and cached.get("fingerprint") == fingerprint:
        receipt.write_text(json.dumps(cached, ensure_ascii=False, indent=2))
        print(f"♻️ ASR chunk {start:.1f}-{end:.1f}s restored from the cross-task cache")
        return cached["result"]
    hotwords = Path(str(prefix) + ".hotwords.json")
    hotwords.write_text(json.dumps(cfg.get("hotwords", []), ensure_ascii=False))
    cmd = [str(cfg["python"]), str(project / "scripts/recognize_source_audio.py"),
           "--audio", str(clip.resolve()), "--backend", cfg.get("recognizer", "moss"),
           "--model", str(cfg["model"]), "--language", language,
           "--max-tokens", str(cfg.get("max_tokens", 8192)), "--hotwords-file", str(hotwords.resolve()),
           "--moss-root", str(load_key("moss_asr.root"))]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    recognized = subprocess.run(cmd, capture_output=True, text=True, timeout=int(cfg.get("timeout_seconds", 1800)), env=env)
    if recognized.returncode:
        raise RuntimeError(f"Local source recognition failed: {recognized.stderr[-1800:]}")
    raw = json.loads(recognized.stdout)
    Path(str(prefix) + ".raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2))
    segments, edits = raw["segments"], []
    if cfg.get("review_file"):
        if not segments:
            raise ValueError("Timed source review currently requires MOSS segment receipts")
        segments, edits = apply_source_review(segments, json.loads(review_path.read_text()),
                            audio_sha256=source_hash, start=start, end=end)
    joiner = "" if language in {"zh", "ja", "ko", "yue"} else " "
    text = joiner.join(s["text"] for s in segments) if segments else raw["text"]
    transcript.write_text(text, encoding="utf-8")
    if not text.strip():
        result = {"language": language, "segments": [], "text": "",
                  "timing_source": "qwen3_forced_alignment_80ms",
                  "word_timestamp_contract": "acoustic-forced-alignment-80ms",
                  "word_timestamp_count": 0}
        receipt.write_text(json.dumps({"result": result, "fingerprint": fingerprint}, ensure_ascii=False, indent=2))
        _global_put(global_key, receipt)
        return result
    align_command = [str(cfg["python"]), str(project / "scripts/align_source_audio.py"),
                "--audio", str(clip.resolve()), "--text-file", str(transcript.resolve()),
                "--model", str(cfg["aligner"]), "--language", names[language]]
    if segments:
        segment_file = Path(str(prefix) + ".segments.json")
        segment_file.write_text(json.dumps(segments, ensure_ascii=False, indent=2))
        align_command.extend(["--segments-file", str(segment_file.resolve())])
    aligned = subprocess.run(align_command, capture_output=True, text=True,
                            timeout=int(cfg.get("timeout_seconds", 1800)), env=env)
    if aligned.returncode:
        raise RuntimeError(f"Local source alignment failed: {aligned.stderr[-1800:]}")
    alignment_receipt = json.loads(aligned.stdout)
    try:
        words = bind_recognition_words(alignment_receipt["words"], segments, start, language)
    except ValueError:
        if not segments:
            raise
        # A measured per-phrase retry cannot borrow timing or text from a
        # neighboring speaker. Preserve both attempts for diagnosis.
        Path(str(prefix) + ".alignment-first.json").write_text(json.dumps(alignment_receipt, ensure_ascii=False))
        retried = subprocess.run(align_command + ["--individual-phrases"], capture_output=True, text=True,
                                timeout=int(cfg.get("timeout_seconds", 1800)), env=env)
        if retried.returncode:
            raise RuntimeError(f"Phrase-local alignment retry failed: {retried.stderr[-1800:]}")
        alignment_receipt = json.loads(retried.stdout)
        Path(str(prefix) + ".alignment-phrases.json").write_text(json.dumps(alignment_receipt, ensure_ascii=False))
        words = bind_recognition_words(alignment_receipt["words"], segments, start, language)
    if normalize_transcript("".join(w["word"] for w in words)) != normalize_transcript(text):
        raise ValueError("Forced alignment changed recognized source text; rejected")
    # Retain recognizer phrase boundaries separately from acoustic speaker
    # labels. No word timing is copied from the recognizer's segment estimates.
    for word in words:
        word["start"] += start
        word["end"] += start
    data = normalize_asr_result({"language": language, "segments": [{"text": text,
                  "start": words[0]["start"], "end": words[-1]["end"], "words": words}],
                  "recognition_segments": segments,
                  "timing_source": "qwen3_forced_alignment_80ms; not native ASR timing",
                  "source_review": {"status": "applied" if edits else "not_applied", "edits": edits,
                                    "basis": "external_source_review" if edits else "acoustic_recognition"}}, "local-aligned")
    validate_word_timestamps(data, backend="local-aligned")
    receipt.write_text(json.dumps({"fingerprint": fingerprint, "recognition": raw,
                                  "alignment": alignment_receipt, "result": data}, ensure_ascii=False, indent=2))
    _global_put(global_key, receipt)
    return data
