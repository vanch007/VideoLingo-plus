"""Cross-line name consistency for Chinese transcripts, before translation.

ASR spells a name right where the context is rich and wrong where it is
spoken alone (黛玉 -> 大玉, 晴雯 -> 添温). The summary already knows the names;
this module finds spans that sound like one of them, asks the LLM to confirm
each one against the surrounding lines, and replaces only confirmed spans.
A replacement always has the name's own length, so character offsets and the
Step 6 fuzzy timestamp match are unaffected. Every decision is logged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

REPORT_PATH = "output/log/name_consistency.json"
CJK = re.compile(r"^[一-鿿]+$")
PUNCT = re.compile(r"[\s，。！？、；：,.!?;:'\"“”‘’…—\-（）()《》]+")
MAX_NAME_CHARS = 6
CONTEXT_LINES = 3


def _syllables(text: str) -> list[str]:
    from pypinyin import lazy_pinyin
    return lazy_pinyin(text)


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def name_pool(terms: list[dict[str, Any]], lines: list[str]) -> list[str]:
    """CJK term sources plus the parts of a compound name that are spoken on their own.

    '黛玉晴雯子' is one term, but '不是，黛玉和晴雯吗' uses 黛玉 and 晴雯 separately.
    A part counts only where its neighbours in the line differ from its
    neighbours in the term, so '玉晴' inside '黛玉晴雯' is never a name.
    """
    names: set[str] = set()
    for term in terms:
        src = str(term.get("src", "")).strip()
        if not (2 <= len(src) <= MAX_NAME_CHARS and CJK.match(src)):
            continue
        names.add(src)
        for i in range(len(src)):
            for j in range(i + 2, len(src) + 1):
                part = src[i:j]
                if part == src:
                    continue
                for line in lines:
                    for match in re.finditer(re.escape(part), line):
                        p, q = match.start(), match.end()
                        before_free = i == 0 or p == 0 or line[p - 1] != src[i - 1]
                        after_free = j == len(src) or q == len(line) or line[q] != src[j]
                        if before_free and after_free:
                            names.add(part)
    return sorted(names, key=lambda n: (-len(n), n))


def _sounds_like(window: str, name: str, standalone: bool) -> bool:
    """A misheard name: same length, part of it kept, the rest close in sound.

    Inside a longer line one character must be identical and every other
    syllable one edit away in toneless pinyin (大玉 for 黛玉); a shared
    syllable alone matches too many ordinary words there. A line that is only
    the span is a stronger signal of a name being called out, so sharing half
    the syllables suffices (添温 for 晴雯) and the LLM decides.
    """
    if window == name or len(window) != len(name):
        return False
    heard, expected = _syllables(window), _syllables(name)
    if len(heard) != len(expected):
        return False
    same = [w == n or a == b for w, n, a, b in zip(window, name, heard, expected)]
    if not any(same):
        return False
    if standalone:
        return 2 * sum(same) >= len(same)
    if not any(w == n for w, n in zip(window, name)):
        return False
    return all(s or _edit_distance(a, b) <= 1 for s, a, b in zip(same, heard, expected))


def find_candidates(lines: list[str], names: list[str]) -> list[dict[str, Any]]:
    candidates = []
    for index, line in enumerate(lines):
        bare = PUNCT.sub("", line)
        covered = [False] * len(line)
        for name in names:
            for match in re.finditer(re.escape(name), line):
                for k in range(match.start(), match.end()):
                    covered[k] = True
        for name in names:
            n = len(name)
            for start in range(len(line) - n + 1):
                window = line[start:start + n]
                if not CJK.match(window) or any(covered[start:start + n]):
                    continue
                if _sounds_like(window, name, standalone=bare == window):
                    candidates.append({"line_index": index, "start": start, "heard": window, "name": name})
    # One proposal per span: the longest name, then the first in pool order.
    seen, unique = set(), []
    for item in candidates:
        key = (item["line_index"], item["start"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _prompt(lines: list[str], names: list[str], candidates: list[dict[str, Any]], topic: str) -> str:
    cases = []
    for number, item in enumerate(candidates, 1):
        i = item["line_index"]
        context = [f"{'>>' if k == i else '  '} L{k + 1}: {lines[k]}"
                   for k in range(max(0, i - CONTEXT_LINES), min(len(lines), i + CONTEXT_LINES + 1))]
        name_lines = [f"   L{k + 1}: {line}" for k, line in enumerate(lines) if item["name"] in line][:3]
        pairs = [f"   '{other['heard']}' in L{other['line_index'] + 1} sounds like '{other['name']}', and "
                 f"'{item['name']}' and '{other['name']}' are said together in L{k + 1}: {lines[k]}"
                 for other in candidates if other is not item and abs(other["line_index"] - i) <= 2
                 for k in [next((k for k, line in enumerate(lines)
                                 if item["name"] in line and other["name"] in line), None)] if k is not None]
        cases.append(f"Case {number}: is '{item['heard']}' in line L{i + 1} a mis-transcription of the name "
                     f"'{item['name']}'?\nContext:\n" + "\n".join(context)
                     + ("\nWhere the name is transcribed correctly:\n" + "\n".join(name_lines) if name_lines else "")
                     + ("\nNames heard together nearby:\n" + "\n".join(pairs[:2]) if pairs else ""))
    return (
        "You check a Chinese speech transcript for names that the speech recognizer misheard.\n"
        f"Topic: {topic}\nNames used in this video: {', '.join(names)}\n\n"
        "For each case decide whether the speaker actually said the name. Answer true only when the context "
        "shows the same person or thing is meant (for example the name is introduced or repeated nearby). "
        "Answer false when the heard words are ordinary words that make sense as they are. Names that are "
        "spoken alone are often misheard in several syllables, so judge by context, not by sound alone.\n\n"
        + "\n\n".join(cases)
        + '\n\nOutput JSON only: {"decisions":[{"case":1,"is_name":true,"reason":"..."}]}'
    )


def check_names(lines: list[str], terms: list[dict[str, Any]], topic: str = "",
                ask: Callable[..., Any] | None = None) -> tuple[list[str], dict[str, Any]]:
    """Return corrected lines and a report; lines change only where the LLM confirms."""
    names = name_pool(terms, lines)
    candidates = find_candidates(lines, names)
    report: dict[str, Any] = {"names": names, "candidates": candidates, "applied": [], "rejected": []}
    if not candidates:
        return list(lines), report
    if ask is None:
        from core.ask_gpt import ask_gpt as ask

    def valid(data):
        decisions = data.get("decisions") if isinstance(data, dict) else None
        if not isinstance(decisions, list):
            return {"status": "error", "message": "Missing 'decisions' list"}
        cases = {d.get("case") for d in decisions if isinstance(d, dict)}
        if cases != set(range(1, len(candidates) + 1)):
            return {"status": "error", "message": f"Expected decisions for cases 1..{len(candidates)}"}
        return {"status": "success", "message": "ok"}

    reply = ask(_prompt(lines, names, candidates, topic), response_json=True, valid_def=valid,
                log_title="name_consistency")
    decided = {d["case"]: d for d in reply["decisions"]}
    corrected = list(lines)
    for number, item in enumerate(candidates, 1):
        decision = decided[number]
        record = {**item, "reason": str(decision.get("reason", ""))}
        if decision.get("is_name") is True:
            i, s, n = item["line_index"], item["start"], len(item["name"])
            line = corrected[i]
            if line[s:s + n] == item["heard"]:
                corrected[i] = line[:s] + item["name"] + line[s + n:]
                report["applied"].append({**record, "before": line, "after": corrected[i]})
                continue
        report["rejected"].append(record)
    return corrected, report


def run_name_consistency(sentences_path: str, terms: list[dict[str, Any]], topic: str = "",
                         report_path: str = REPORT_PATH, ask: Callable[..., Any] | None = None) -> dict[str, Any]:
    path = Path(sentences_path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    corrected, report = check_names(lines, terms, topic, ask)
    if corrected != lines:
        path.write_text("\n".join(corrected), encoding="utf-8")
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
