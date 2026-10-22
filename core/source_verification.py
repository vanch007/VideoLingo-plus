"""Independent acoustic witness for translation source text; never invent repairs."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

import requests

from core.config_utils import load_key
from core.pipeline.artifact_manifest import hash_path
from core.providers.omlx import get_omlx_base_url

REPORT_PATH = Path("output/log/source_verification.json")


def normalize_transcript(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold(), flags=re.UNICODE)


def compare_transcripts(source: str, witness: str, *, fail_below: float = 0.65) -> dict:
    left, right = normalize_transcript(source), normalize_transcript(witness)
    score = SequenceMatcher(None, left, right, autojunk=False).ratio()
    status = "pass" if left and left == right else "needs_review"
    # Edit distance ranks review urgency; it cannot establish which ASR is
    # correct. A disagreement must never be relabeled as verified bad source.
    if not left or not right:
        status = "failed"
    return {
        "status": status,
        "similarity": round(score, 4),
        "disagreement_level": "high" if score < fail_below else ("none" if left == right else "moderate"),
        "differences": [
            {"source": left[a:b], "witness": right[c:d], "operation": op}
            for op, a, b, c, d in SequenceMatcher(None, left, right, autojunk=False).get_opcodes()
            if op != "equal"
        ],
    }


def transcribe_witness(audio: Path, language: str, model: str) -> str:
    key = load_key("llm.providers.omlx.api_key", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    with audio.open("rb") as handle:
        response = requests.post(
            f"{get_omlx_base_url()}/audio/transcriptions",
            headers=headers,
            files={"file": (audio.name, handle, "audio/wav")},
            data={"model": model, "language": language, "max_tokens": 2048},
            timeout=float(load_key("source_quality.timeout_seconds", 180)),
        )
    response.raise_for_status()
    text = response.json().get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Independent ASR returned no transcript")
    return text.strip()


def verify_source_rows(rows, *, report_path: Path = REPORT_PATH) -> dict:
    """Compare intact speaker turns against independent ASR on original audio.

    Agreements are model agreement, not human acceptance. Disagreements are
    recorded without replacing text or native word timestamps.
    """
    from core.runtime_context import effective_source_language

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if load_key("quality_mode", "high_sync") == "fast":
        report = {"status": "not_run", "reason": "fast_profile", "groups": []}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    if not load_key("source_quality.enabled", False):
        report = {"status": "missing_evidence", "reason": "independent_source_verification_disabled", "groups": []}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    if Path("output/log/srt_chunks.xlsx").exists():
        report = {"status": "not_applicable", "reason": "external_subtitle_input_without_audio_witness", "groups": []}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    audio = Path("output/audio/raw_master.wav")
    if not audio.is_file():
        raise RuntimeError("Original lossless audio is missing; cannot verify source text")
    language = effective_source_language("auto")
    model = str(load_key("source_quality.model", "mlx-community--Qwen3-ASR-0.6B-8bit"))
    if load_key("whisper.runtime", "stable-ts") == "qwen3-aligned" and model == load_key("qwen3_source.model", "mlx-community--Qwen3-ASR-0.6B-8bit"):
        raise RuntimeError("Source witness must use a different ASR model from primary recognition")
    audio_hash = hash_path(audio)
    joiner = "" if language.split("-")[0] in {"zh", "ja", "ko"} else " "
    groups = []
    for _, row in rows.iterrows():
        speaker = str(row.get("speaker") or "unknown")
        if (groups and float(row["start"]) - groups[-1]["end"] < 1.2
                and float(row["end"]) - groups[-1]["start"] <= 18):
            group = groups[-1]
            if group["speaker"] != speaker:
                group["speaker"] = "mixed"
            group["source"] += joiner + str(row["text"])
            group["end"] = float(row["end"])
            group["line_ids"].append(str(row["line_id"]))
        else:
            groups.append({"speaker": speaker, "source": str(row["text"]),
                           "start": float(row["start"]), "end": float(row["end"]),
                           "line_ids": [str(row["line_id"])]})
    evidence = report_path.parent / "source_witness"
    evidence.mkdir(exist_ok=True)
    for index, group in enumerate(groups, 1):
        start, end = max(0, group['start'] - .25), group['end'] + .25
        if index > 1:
            start = max(start, (groups[index - 2]['end'] + group['start']) / 2)
        if index < len(groups):
            end = min(end, (group['end'] + groups[index]['start']) / 2)
        if end <= start:
            raise ValueError('Source witness has an invalid or overlapping speech window')
        fingerprint = hashlib.sha256(json.dumps(
            {"audio": audio_hash, "model": model, "language": language, "group": group,
             "clip_start": start, "clip_end": end},
            sort_keys=True, ensure_ascii=False,
        ).encode()).hexdigest()
        receipt = evidence / f"{fingerprint}.json"
        clip = evidence / f"{fingerprint}.wav"
        try:
            if receipt.exists():
                data = json.loads(receipt.read_text())
                if data.get("audio_sha256") != hash_path(clip):
                    raise ValueError("Source witness clip no longer matches its receipt")
                witness = data["text"]
            else:
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start),
                                "-i", str(audio), "-t", str(end - start),
                                "-ar", "16000", "-ac", "1", str(clip)], check=True)
                witness = transcribe_witness(clip, language, model)
                receipt.write_text(json.dumps({"text": witness, "audio_sha256": hash_path(clip),
                                              "model": model}, ensure_ascii=False, indent=2))
            group.update(compare_transcripts(group["source"], witness,
                         fail_below=float(load_key("source_quality.fail_below", 0.65))))
            group.update(witness=witness, audio=str(clip), fingerprint=fingerprint)
        except Exception as exc:
            group.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(f"Source witness {index}/{len(groups)}: {group['status']}", flush=True)
    status = "failed" if any(g["status"] == "failed" for g in groups) else (
        "needs_review" if any(g["status"] != "pass" for g in groups) else "pass")
    report = {"status": status, "basis": "independent_asr_agreement_not_human_verification",
              "model": model, "audio_sha256": audio_hash, "groups": groups}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if status == "failed" and load_key("source_quality.strict_gate", True):
        raise RuntimeError(f"Source verification failed; inspect audio and disagreements in {report_path}")
    return report
