"""Seniority in mutual address (哥 / 弟) must survive translation.

Two people calling each other 哥 and 弟 state who is the elder. Each line
alone translates fine as "Brother.", so per-line checks pass while the pair
loses the distinction. Exchanges are found deterministically in the source;
a batch that contains one gets a translator note, and the assembled script is
checked against them in core.translation_constraints.
"""
from __future__ import annotations

import re
from typing import Any

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
            "visible and the sex right, e.g. Big brother. / Little brother.; never render both the same.")
    return "\n".join(notes) or None
