"""Recurring wordplay, translated with one image in every batch that contains it.

A pun or figurative motif (腿 as "a powerful patron" in a scene that also plays
on literal legs) is spread over many short lines translated in separate
batches; translated literally ("Your legs are my legs.") the joke is lost.

Two sources, both turned into a translator note for each batch containing src:
- a glossary row whose note starts with "wordplay:" (tgt is the rendering),
  always applied: it is the reliable path;
- model detection (wordplay.auto_detect, off by default): the local model's
  choice flipped between 腿 and 装糊涂 with small prompt or topic changes, so
  each item also needs quoted evidence from two nearby lines, a recurring src,
  no overlap with names/terms, and a word-for-word translation that fails.
The summary is not asked: there the motif competed with names for term slots.
"""
from __future__ import annotations

import json
from typing import Any, Callable

MIN_OCCURRENCES = 2
# The prompt is English; a language name in another script there changed which
# motif the local model picked (装糊涂 instead of 腿), so names are English here.
ENGLISH_NAMES = {
    "zh": "Simplified Chinese", "zh-cn": "Simplified Chinese", "zh-sg": "Simplified Chinese",
    "zh-tw": "Traditional Chinese", "zh-hk": "Traditional Chinese", "yue": "Cantonese",
    "en": "English", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay", "ja": "Japanese",
    "ko": "Korean", "th": "Thai", "fr": "French", "de": "German", "es": "Spanish",
    "pt": "Portuguese", "ru": "Russian", "ar": "Arabic", "it": "Italian", "hi": "Hindi", "tr": "Turkish",
}
MAX_ITEMS = 2
MAX_EVIDENCE_GAP = 20


def english_language_name(value: str) -> str:
    code = str(value).strip().lower().replace("_", "-")
    if code in ENGLISH_NAMES or code.split("-", 1)[0] in ENGLISH_NAMES:
        return ENGLISH_NAMES.get(code) or ENGLISH_NAMES[code.split("-", 1)[0]]
    from core.translation_language import translation_language_name
    return translation_language_name(value)


def _prompt(lines: list[str], topic: str, source_language: str, target_language: str) -> str:
    source_language, target_language = english_language_name(source_language), english_language_name(target_language)
    numbered = "\n".join(f"L{i}: {line}" for i, line in enumerate(lines, 1))
    return (
        f"You prepare a {source_language} dialogue for translation into {target_language}.\n"
        f"Topic: {topic}\n\n"
        "List up to 4 candidate words that nearby lines read in two ways: a concrete literal sense in one line "
        "and a figurative sense in another (for example a body part standing for a powerful patron while other "
        "lines talk about the body part itself). Skip names, titles and forms of address. Return an empty list "
        "when there is none.\n"
        "For each candidate give src (the shortest form exactly as written), literal and figurative (its two "
        "senses), joke (how the lines play on them), literal_line and figurative_line (one exact line each, "
        f"copied verbatim, both containing src), word_for_word (a word-for-word {target_language} translation of "
        f"figurative_line), understood_word_for_word (true if a {target_language} viewer who does not know the "
        f"{source_language} expression would still get the figurative sense from word_for_word), and rendering: "
        f"a short {target_language} word or idiom (at most four words), usable in every line, that such a viewer "
        "reads in the figurative sense too; keep the literal image when such an idiom exists.\n"
        'Output JSON only: {"wordplay":[{"src":"...","literal":"...","figurative":"...","joke":"...",'
        '"literal_line":"...","figurative_line":"...","word_for_word":"...","understood_word_for_word":false,'
        '"rendering":"..."}]}\n\nLines:\n' + numbered
    )


def _line_index(lines: list[str], quote: str, src: str) -> int | None:
    quote = quote.strip()
    if not quote or src not in quote:
        return None
    # Exact line, a quote cut from one line, or a quote that pads one line (要什么腿 for 什么腿);
    # either way a real line holding src must be found.
    for index, line in enumerate(lines):
        if src not in line:
            continue
        if quote == line or (len(quote) >= len(src) + 2 and quote in line) or (len(line) > len(src) and line in quote):
            return index
    return None


def find_wordplay(lines: list[str], topic: str = "", source_language: str = "Chinese",
                  target_language: str = "English", known_terms: list[str] | None = None,
                  ask: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Keep an item only with evidence: two different nearby lines, both quoted exactly and
    both containing src, and src recurring; names and terms already in the glossary are excluded."""
    if not lines:
        return {"wordplay": [], "dropped": []}
    if ask is None:
        from core.ask_gpt import ask_gpt as ask
    known = {str(t).strip() for t in known_terms or [] if str(t).strip()}

    def valid(data):
        items = data.get("wordplay") if isinstance(data, dict) else None
        if not isinstance(items, list) or not all(isinstance(i, dict) and str(i.get("src", "")).strip() for i in items):
            return {"status": "error", "message": "Expected {'wordplay': [{'src': ...}]}"}
        return {"status": "success", "message": "ok"}

    reply = ask(_prompt(lines, topic, source_language, target_language), response_json=True,
                valid_def=valid, log_title="wordplay")
    kept, dropped = [], []
    for item in reply["wordplay"]:
        src = str(item["src"]).strip()
        record = {"src": src, **{key: str(item.get(key, "")).strip() for key in
                                 ("literal", "figurative", "joke", "rendering", "literal_line", "figurative_line")}}
        record["occurrences"] = sum(src in line for line in lines)
        a = _line_index(lines, record["literal_line"], src)
        b = _line_index(lines, record["figurative_line"], src)
        record["word_for_word"] = str(item.get("word_for_word", "")).strip()
        reason = None
        # Only wordplay a literal translation breaks needs a note: 装糊涂 -> "play dumb" already works.
        if item.get("understood_word_for_word") is not False:
            reason = "literal_translation_already_works"
        elif any(src in term or term in src for term in known):
            reason = "name_or_glossary_term"
        elif record["occurrences"] < MIN_OCCURRENCES:
            reason = "not_recurring"
        elif a is None or b is None:
            reason = "evidence_not_in_transcript"
        elif a == b or abs(a - b) > MAX_EVIDENCE_GAP:
            reason = "evidence_not_two_nearby_lines"
        elif not (record["figurative"] and record["rendering"]) or len(kept) >= MAX_ITEMS:
            reason = "incomplete_or_over_limit"
        if reason:
            dropped.append({**record, "reason": reason})
        else:
            kept.append(record)
    return {"wordplay": kept, "dropped": dropped}


def split_glossary_wordplay(terms: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Glossary rows with a note starting 'wordplay:' are instructions: src, tgt = rendering, note = meaning."""
    plain, wordplay = [], []
    for term in terms:
        note = str(term.get("note", "")).strip()
        if note.lower().startswith("wordplay:") and str(term.get("src", "")).strip():
            wordplay.append({"src": str(term["src"]).strip(), "literal": "", "joke": "",
                             "figurative": note[len("wordplay:"):].strip(),
                             "rendering": str(term.get("tgt", "")).strip(), "origin": "glossary"})
        else:
            plain.append(term)
    return plain, wordplay


def translator_note(item: dict[str, Any]) -> str:
    sense = (f'literally {item["literal"]}; here {item["figurative"]}.' if item.get("literal")
             else str(item["figurative"]).rstrip(".") + ".")
    joke = f' {item["joke"]}' if item.get("joke") else ""
    return (f'"{item["src"]}" is recurring wordplay: {sense}{joke} Suggested rendering: "{item["rendering"]}". '
            "Keep the same short image in every line "
            "that plays on it so the joke survives across lines; let context carry the second sense instead of "
            "adding explanations, and never use a bare literal word the audience cannot decode.")


def save_wordplay(report: dict[str, Any], path: str) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
