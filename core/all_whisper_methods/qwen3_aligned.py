"""Local Qwen ASR + acoustic forced alignment, with explicit timing provenance."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from core.asr_schema import normalize_asr_result, validate_word_timestamps, _join_collapsed_tokens
from core.config_utils import load_key
from core.pipeline.artifact_manifest import hash_path
from core.source_verification import transcribe_witness


def merge_zero_duration_words(words: list[dict], *, language="auto") -> list[dict]:
    """Keep text when the aligner's 80 ms grid collapses a short token."""
    output, pending = [], []
    def same_phrase(left, right):
        return left.get("recognition_segment_id") == right.get("recognition_segment_id")
    for index, item in enumerate(words):
        word = dict(item)
        if word["end"] <= word["start"]:
            if pending:
                if not same_phrase(pending[-1], word):
                    raise ValueError("Collapsed alignment crosses a recognized phrase; realignment required")
                pending.append(word)
                continue
            following = next((w for w in words[index + 1:]
                              if w["end"] > w["start"] and same_phrase(w, word)), None)
            preceding = output[-1] if output and same_phrase(output[-1], word) else None
            # Choose the nearest measured neighbour in the same ASR phrase.
            # A collapsed final character must not become the next turn's prefix.
            if preceding and (not following or
                    abs(word["start"] - preceding["end"]) <= abs(following["start"] - word["end"])):
                preceding["word"] = _join_collapsed_tokens([preceding["word"], word["word"]], language)
                preceding["end"] = max(preceding["end"], word["end"])
            else:
                pending.append(word)
            continue
        if pending and any(not same_phrase(w, word) for w in pending):
            raise ValueError("Collapsed alignment crosses a recognized phrase; realignment required")
        word["word"] = _join_collapsed_tokens([*[w["word"] for w in pending], word["word"]], language) if pending else word["word"]
        if pending:
            word["start"] = min(word["start"], pending[0]["start"])
        pending = []
        output.append(word)
    if pending:
        if not output or any(not same_phrase(w, output[-1]) for w in pending):
            raise ValueError("Aligner could not place any spoken text")
        output[-1]["word"] = _join_collapsed_tokens([output[-1]["word"], *[w["word"] for w in pending]], language)
    return output


def transcribe_audio(audio_file: str, start: float, end: float) -> dict:
    from core.runtime_context import effective_source_language
    language = effective_source_language("zh")
    model = str(load_key("qwen3_source.model", "mlx-community--Qwen3-ASR-0.6B-8bit"))
    aligner = str(load_key("qwen3_source.aligner", "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"))
    python = str(load_key("qwen3_source.python", load_key("nemotron_diarization.python")))
    output = Path("output/log/qwen3_source")
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / f"{start:.3f}-{end:.3f}"
    # Prefix contains decimal points, so append extensions rather than replace them.
    audio = Path(str(prefix) + ".wav")
    text_file = Path(str(prefix) + ".txt")
    receipt = Path(str(prefix) + ".json")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", audio_file,
                    "-t", str(end - start), "-ar", "16000", "-ac", "1", str(audio)], check=True)
    fingerprint = {"audio_sha256": hash_path(audio), "model": model, "aligner": aligner, "language": language}
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved.get("fingerprint") == fingerprint:
            return saved["result"]
    text = transcribe_witness(audio, language, model)
    text_file.write_text(text, encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "scripts" / "align_source_audio.py"
    result = subprocess.run([python, str(script), "--audio", str(audio.resolve()),
                             "--text-file", str(text_file.resolve()), "--model", aligner,
                             "--language", {"zh": "Chinese", "en": "English", "ja": "Japanese"}.get(language, language)],
                            capture_output=True, text=True, timeout=180,
                            env={**os.environ, "HF_HUB_OFFLINE": "1"})
    if result.returncode:
        raise RuntimeError(f"Source forced alignment failed: {result.stderr[-1800:]}")
    aligned = json.loads(result.stdout)["words"]
    words = merge_zero_duration_words(aligned, language=language)
    for word in words:
        word["start"] = float(word["start"]) + start
        word["end"] = float(word["end"]) + start
    data = normalize_asr_result({
        "language": language, "segments": [
            {"text": text, "start": words[0]["start"], "end": words[-1]["end"], "words": words}
        ] if words else [],
        "timing_source": "qwen3_forced_alignment_80ms; not ASR native timing",
    }, "qwen3-aligned")
    validate_word_timestamps(data, backend="qwen3-aligned", allow_empty=True)
    receipt.write_text(json.dumps({"fingerprint": fingerprint, "raw_words": aligned, "result": data}, ensure_ascii=False, indent=2))
    return data
