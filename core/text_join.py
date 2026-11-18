"""Joining word tokens of languages written without spaces (zh/ja/ko/yue).

A plain ``"".join`` turns the recognized "US dollar" into "USdollar" and "easy easy" into
"easyeasy". Tokens are joined without a separator except between two Latin/digit tokens, where
a single space is kept.
"""
from __future__ import annotations

import re
from typing import Iterable

_EDGE_LATIN = re.compile(r"[A-Za-z0-9]")


def join_tokens(tokens: Iterable[str]) -> str:
    out = ""
    for token in tokens:
        token = str(token)
        if out and token and _EDGE_LATIN.fullmatch(out[-1]) and _EDGE_LATIN.fullmatch(token[0]):
            out += " "
        out += token
    return out
