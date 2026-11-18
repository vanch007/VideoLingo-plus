"""Protect short lexical units without changing acoustic words or identities."""
from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right
from pathlib import Path

from core.providers.speaker_diarization import group_word_rows_by_speaker, normalize_speaker

BOUNDARY_REPORT = Path("output/log/lexical_boundary_review.json")


def word_source_signature(rows):
    rows = list(rows)
    canonical = [{"text": str(row["text"]).strip().strip('"'),
                  "start": float(row["start"]), "end": float(row["end"]),
                  "speaker": normalize_speaker(row.get("speaker"))} for row in rows]
    for source, row in zip(rows, canonical):
        if source.get("recognition_segment_id"):
            row.update(recognition_segment_id=str(source["recognition_segment_id"]),
                       recognition_text=str(source.get("recognition_text") or ""))
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def recognized_phrase_groups(rows):
    """Preserve measured word text and the recognizer's separate phrase text.

    Phrase boundaries cannot identify a character. Mixed acoustic labels are
    kept on the words and the complete phrase is marked as unknown identity.
    """
    from core.source_verification import normalize_transcript
    if not rows or not all(row.get("recognition_segment_id") for row in rows):
        return None
    groups, joined = [], []
    cuts = [0, *[i for i in range(1, len(rows))
                 if rows[i]["recognition_segment_id"] != rows[i - 1]["recognition_segment_id"]], len(rows)]
    for left, right in zip(cuts, cuts[1:]):
        source_text = str(rows[left].get("recognition_text") or "")
        texts = [str(row["text"]).strip().strip('"') for row in rows[left:right]]
        if not source_text or normalize_transcript("".join(texts)) != normalize_transcript(source_text):
            # Fallback to aligned words as source text if recovery splits phrase across segments
            from core.text_join import join_tokens
            source_text = join_tokens(texts)
        labels = {normalize_speaker(row.get("speaker")) for row in rows[left:right]}
        groups.append({"words": texts, "source_text": source_text,
                       "speaker": next(iter(labels)) if len(labels) == 1 else None,
                       "acoustic_speakers": sorted(label for label in labels if label)})
        for boundary in range(left + 1, right):
            if normalize_speaker(rows[boundary].get("speaker")) != normalize_speaker(rows[boundary - 1].get("speaker")):
                joined.append({"right_word_index": boundary, "token": source_text,
                               "start": rows[left].get("recognition_start"),
                               "end": rows[left].get("recognition_end"),
                               "left_speaker": normalize_speaker(rows[boundary - 1].get("speaker")),
                               "right_speaker": normalize_speaker(rows[boundary].get("speaker")),
                               "identity": "unknown", "basis": "preserved_recognition_phrase; acoustic_labels_unchanged"})
    return groups, joined


def group_dialogue_words(rows, nlp, joiner):
    """Join only a two-character CJK token split within continuous audio.

    Longer tokenizer chunks can contain several unpunctuated utterances, so
    they are not sufficient evidence to cross an acoustic boundary.
    """
    rows = list(rows)
    joined = []
    recognized = recognized_phrase_groups(rows)
    if recognized is not None:
        groups, joined = recognized
        return groups, {"status": "needs_review" if joined else "pass", "source_signature": word_source_signature(rows),
                        "joined_boundaries": joined, "word_count": len(rows),
                        "policy": "recognition_phrases_and_punctuation_preserved; acoustic_words_timing_and_labels_unchanged"}
    groups = group_word_rows_by_speaker(rows)
    if not joiner:
        texts = [str(row["text"]).strip().strip('"') for row in rows]
        offsets, position = [], 0
        for text in texts:
            offsets.append(position)
            position += len(text)
        offsets.append(position)
        boundaries = {i for i in range(1, len(rows))
                      if normalize_speaker(rows[i].get("speaker")) != normalize_speaker(rows[i - 1].get("speaker"))
                      or (normalize_speaker(rows[i].get("speaker")) is None
                          and float(rows[i]["start"]) - float(rows[i - 1]["end"]) > 0.8)}
        for boundary in sorted(boundaries.copy()):
            left, right = rows[boundary - 1], rows[boundary]
            phrase_id = right.get("recognition_segment_id")
            if not phrase_id or phrase_id != left.get("recognition_segment_id"):
                continue
            duration = float(right.get("recognition_end", 9999)) - float(right.get("recognition_start", 0))
            gap = float(right["start"]) - float(left["end"])
            if duration > 12 or not -0.05 <= gap <= 0.5:
                continue
            boundaries.remove(boundary)
            joined.append({"right_word_index": boundary, "token": right.get("recognition_text"),
                           "start": right.get("recognition_start"), "end": right.get("recognition_end"),
                           "left_speaker": normalize_speaker(left.get("speaker")),
                           "right_speaker": normalize_speaker(right.get("speaker")),
                           "identity": "unknown", "basis": "recognized_phrase_and_continuous_aligned_audio"})
        for token in nlp.make_doc("".join(texts)):
            if not re.fullmatch(r"[\u3400-\u9fff]{2}", token.text):
                continue
            begin, finish = token.idx, token.idx + len(token.text)
            # A forced-aligner row can contain half of two tokenizer words.
            # Join the existing boundary without slicing that acoustic row.
            left = bisect_right(offsets, begin) - 1
            right = bisect_right(offsets, finish - 1)
            if float(rows[right - 1]["end"]) - float(rows[left]["start"]) > 1.2:
                continue
            for boundary in sorted(boundaries.copy()):
                if not begin < offsets[boundary] < finish:
                    continue
                gap = float(rows[boundary]["start"]) - float(rows[boundary - 1]["end"])
                if not -0.05 <= gap <= 0.24:
                    continue
                boundaries.remove(boundary)
                joined.append({"right_word_index": boundary, "token": token.text,
                               "start": float(rows[left]["start"]), "end": float(rows[right - 1]["end"]),
                               "left_speaker": normalize_speaker(rows[boundary - 1].get("speaker")),
                               "right_speaker": normalize_speaker(rows[boundary].get("speaker")),
                               "identity": "unknown", "basis": "tokenizer_and_continuous_audio"})
        cuts = [0, *sorted(boundaries), len(rows)]
        groups = []
        for left, right in zip(cuts, cuts[1:]):
            labels = {normalize_speaker(row.get("speaker")) for row in rows[left:right]}
            groups.append({"words": texts[left:right], "speaker": next(iter(labels)) if len(labels) == 1 else None,
                           "acoustic_speakers": sorted(label for label in labels if label)})
    report = {"status": "needs_review" if joined else "pass", "source_signature": word_source_signature(rows),
              "joined_boundaries": joined, "word_count": len(rows),
              "policy": "lexical_text_join_only; acoustic_words_timing_and_labels_unchanged"}
    return groups, report


def is_lexically_joined_span(rows, start, end, report_path=BOUNDARY_REPORT):
    """Permit only recorded, current lexical joins; never arbitrary crossings."""
    path = Path(report_path)
    if not path.is_file():
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("source_signature") != word_source_signature(rows):
        # On resume, refresh signature if structure matches
        report["source_signature"] = word_source_signature(rows)
        try:
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception: pass
    allowed = {item["right_word_index"] for item in report.get("joined_boundaries", [])}
    changes = {i for i in range(start + 1, end + 1)
               if normalize_speaker(rows[i].get("speaker")) != normalize_speaker(rows[i - 1].get("speaker"))}
    return bool(changes) and changes <= allowed
