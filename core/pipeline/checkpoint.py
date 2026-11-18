"""Human checkpoint after the translation (``pause_after_translate``).

The pipeline stops right after the translate step, the user edits the ``Translation`` column of
``output/log/translation_results.xlsx`` (the ``Source`` column and the rows must stay as they
are), and ``resume`` continues. Edited rows are identified by hash, written to
``output/log/translation_manual_edits.json`` and noted in the run events, so later stages and
audits can tell reviewed wording from model output.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from core.config_utils import load_key

TRANSLATION_FILE = "output/log/translation_results.xlsx"
PAUSE_FILE = Path("output/log/pause_after_translate.json")
EDITS_FILE = Path("output/log/translation_manual_edits.json")


def _digest(text) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def _rows(path: str = TRANSLATION_FILE) -> list[dict]:
    import pandas as pd
    frame = pd.read_excel(path)
    if not {"Source", "Translation"} <= set(frame.columns):
        raise ValueError(f"{path} needs the columns Source and Translation")
    return [{"source": _digest(s), "translation": _digest(t), "text": "" if t != t else str(t)}
            for s, t in zip(frame["Source"], frame["Translation"])]


def pause_requested(step_key: str, has_remaining_steps: bool) -> bool:
    return step_key == "translate" and has_remaining_steps and bool(load_key("pause_after_translate", False))


def write_pause() -> str:
    rows = _rows()
    PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAUSE_FILE.write_text(json.dumps({"created": time.time(), "rows": [
        {"source": r["source"], "translation": r["translation"]} for r in rows]}, indent=2), encoding="utf-8")
    return (f"Translation finished. Edit the Translation column of {TRANSLATION_FILE} "
            "(keep the Source column and the rows), then resume.")


def consume_pause() -> dict | None:
    """Validate and record manual edits when resuming after a pause. None when not paused."""
    if not PAUSE_FILE.is_file():
        return None
    saved = json.loads(PAUSE_FILE.read_text(encoding="utf-8"))["rows"]
    current = _rows()
    if len(current) != len(saved):
        raise ValueError(f"Translation checkpoint: the number of rows changed ({len(saved)} -> {len(current)}); "
                         "restore the rows and edit only the Translation column")
    moved = [i + 1 for i, (a, b) in enumerate(zip(saved, current)) if a["source"] != b["source"]]
    if moved:
        raise ValueError(f"Translation checkpoint: the Source column changed in rows {moved[:10]}; edit only Translation")
    empty = [i + 1 for i, row in enumerate(current) if not row["text"].strip()]
    if empty:
        raise ValueError(f"Translation checkpoint: empty translation in rows {empty[:10]}")
    edited = [i + 1 for i, (a, b) in enumerate(zip(saved, current)) if a["translation"] != b["translation"]]
    summary = {"rows": len(current), "edited_rows": edited, "resumed": time.time()}
    EDITS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    PAUSE_FILE.rename(PAUSE_FILE.with_suffix(".done"))
    return summary
