"""Seniority in mutual address (哥 / 弟) must survive translation.

Two people calling each other 哥 and 弟 state who is the elder. Each line
alone translates fine as "Brother.", so per-line checks pass while the pair
loses the distinction. Exchanges are found deterministically in the source;
a batch that contains one gets a translator note, and after all batches are
assembled a pair rendered identically is repaired. A line that is only the
address has one correct rendering, so English uses a fixed table (keeping the
draft's punctuation); other targets ask the LLM for the two words alone, with
no dialogue context: with context it wrote "Brother, your backer..." for 哥.
A pair inside longer sentences is reported, never rewritten.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

REPORT_PATH = "output/log/address_terms.json"
MAX_GAP = 2
# form -> (seniority, sense)
FORMS = {
    "哥": ("elder", "older brother"), "哥哥": ("elder", "older brother"), "大哥": ("elder", "older brother"),
    "兄": ("elder", "older brother"), "兄长": ("elder", "older brother"), "师兄": ("elder", "senior fellow disciple"),
    "姐": ("elder", "older sister"), "姐姐": ("elder", "older sister"), "大姐": ("elder", "older sister"),
    "师姐": ("elder", "senior fellow disciple (female)"),
    "弟": ("younger", "younger brother"), "弟弟": ("younger", "younger brother"), "老弟": ("younger", "younger brother"),
    "小弟": ("younger", "younger brother"), "贤弟": ("younger", "younger brother"),
    "师弟": ("younger", "junior fellow disciple"),
    "妹": ("younger", "younger sister"), "妹妹": ("younger", "younger sister"), "小妹": ("younger", "younger sister"),
    "师妹": ("younger", "junior fellow disciple (female)"),
}
ENGLISH = {
    "older brother": "Big brother", "younger brother": "Little brother",
    "older sister": "Big sister", "younger sister": "Little sister",
    "senior fellow disciple": "Senior brother", "junior fellow disciple": "Junior brother",
    "senior fellow disciple (female)": "Senior sister", "junior fellow disciple (female)": "Junior sister",
}
PARTICLES = "啊呀哦吧嘛呢哎诶"
PUNCT = "，。！？、；：,.!?;:…—~～ \"“”'‘’"


def address_form(line: str) -> str | None:
    """The form when the line is an address: the whole line, or a vocative before a comma."""
    text = str(line).strip()
    core = text.strip(PUNCT).rstrip(PARTICLES).strip(PUNCT)
    if core in FORMS:
        return core
    match = re.match(r"^([一-鿿]{1,2})[，,！!]", text)
    if match and match.group(1) in FORMS:
        return match.group(1)
    return None


def find_exchanges(lines: list[str]) -> list[dict[str, Any]]:
    forms = [address_form(line) for line in lines]
    exchanges, used = [], set()
    for i, form in enumerate(forms):
        if not form or i in used:
            continue
        for j in range(i + 1, min(len(lines), i + MAX_GAP + 1)):
            other = forms[j]
            if other and j not in used and FORMS[form][0] != FORMS[other][0]:
                elder, younger = (i, j) if FORMS[form][0] == "elder" else (j, i)
                exchanges.append({"elder": elder, "younger": younger,
                                  "elder_form": forms[elder], "younger_form": forms[younger]})
                used.update((i, j))
                break
    return exchanges


def batch_note(lines: list[str]) -> str | None:
    notes = []
    for ex in find_exchanges(lines):
        notes.append(
            f'"{lines[ex["elder"]]}" and "{lines[ex["younger"]]}" are two people addressing each other: '
            f'{ex["elder_form"]} = {FORMS[ex["elder_form"]][1]} (said to the elder), '
            f'{ex["younger_form"]} = {FORMS[ex["younger_form"]][1]} (said to the younger). Keep who is the elder '
            "visible, e.g. Big brother. / Little brother.; never render both the same.")
    return "\n".join(notes) or None


def _same(a: str, b: str) -> bool:
    norm = lambda s: re.sub(r"[\W_]+", " ", str(s).lower()).strip()
    return norm(a) == norm(b)


def _whole_line(line: str, form: str) -> bool:
    return str(line).strip().strip(PUNCT).rstrip(PARTICLES).strip(PUNCT) == form


def _with_punctuation(word: str, draft: str) -> str:
    end = re.search(r"[.!?…]+$", str(draft).strip())
    return word + (end.group(0) if end else ".")


def _ask_pair(ex: dict[str, Any], target_language: str, ask: Callable[..., Any] | None) -> tuple[str, str]:
    if ask is None:
        from core.ask_gpt import ask_gpt as ask
    elder, younger = FORMS[ex["elder_form"]][1], FORMS[ex["younger_form"]][1]
    prompt = (f"Translate two forms of address into {target_language} so the seniority stays visible: "
              f'"{ex["elder_form"]}" ({elder}, said to the elder) and "{ex["younger_form"]}" ({younger}, said to '
              "the younger). Give only the address itself, at most 3 words each, nothing else. "
              'Output JSON only: {"elder":"...","younger":"..."}')

    def valid(data):
        a = str(data.get("elder", "")).strip() if isinstance(data, dict) else ""
        b = str(data.get("younger", "")).strip() if isinstance(data, dict) else ""
        if not a or not b or _same(a, b) or len(a.split()) > 3 or len(b.split()) > 3:
            return {"status": "error", "message": "Two different addresses of at most 3 words are required"}
        return {"status": "success", "message": "ok"}

    reply = ask(prompt, response_json=True, valid_def=valid, log_title="address_terms")
    if valid(reply)["status"] != "success":
        raise ValueError(valid(reply)["message"])
    return str(reply["elder"]).strip().rstrip(".!?"), str(reply["younger"]).strip().rstrip(".!?")


def repair_exchanges(sources: list[str], translations: list[str], target_language: str,
                     ask: Callable[..., Any] | None = None) -> tuple[list[str], dict[str, Any]]:
    """Repair whole-line address pairs rendered identically; returns new translations and a report."""
    fixed = list(translations)
    report: dict[str, Any] = {"exchanges": [], "repaired": [], "kept": []}
    for ex in find_exchanges(sources):
        e, y = ex["elder"], ex["younger"]
        record = {**ex, "elder_source": sources[e], "younger_source": sources[y],
                  "elder_before": fixed[e], "younger_before": fixed[y]}
        report["exchanges"].append(record)
        if not _same(fixed[e], fixed[y]):
            continue
        if not (_whole_line(sources[e], ex["elder_form"]) and _whole_line(sources[y], ex["younger_form"])):
            report["kept"].append({**record, "reason": "address inside a longer line; not rewritten"})
            continue
        if english_target(target_language):
            words = ENGLISH[FORMS[ex["elder_form"]][1]], ENGLISH[FORMS[ex["younger_form"]][1]]
        else:
            try:
                words = _ask_pair(ex, target_language, ask)
            except Exception as error:  # A failed repair keeps the draft; it never fails translation.
                report["kept"].append({**record, "reason": str(error)[:200]})
                continue
        fixed[e], fixed[y] = _with_punctuation(words[0], fixed[e]), _with_punctuation(words[1], fixed[y])
        report["repaired"].append({**record, "elder_after": fixed[e], "younger_after": fixed[y]})
    return fixed, report


def english_target(target_language: str) -> bool:
    return str(target_language).strip().lower() in {"en", "english"} or str(target_language).lower().startswith("en-")


def save_report(report: dict[str, Any], path: str = REPORT_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
