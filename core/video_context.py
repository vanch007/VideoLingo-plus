"""Optional visual context from WeMM-Video, gathered before transcription.

What it is used for (and only this):
  * screen terms  -> extra ASR hotwords for this run (MOSS uses them only when actually spoken)
  * hard subtitles -> skip burning the source-language row when the video already shows it
  * characters     -> optional hints for the summary/terminology prompt (off by default, see DEFAULTS)
Dialogue text, speaker attribution and timing from the vision model are never used: on the test clips
its transcript was less accurate than MOSS and its per-line speakers were wrong.

The step never fails the pipeline: a missing model or runtime, a timeout or an unreadable reply is
recorded as status "skipped"/"failed" and every consumer falls back to the configured behaviour.

Config (all optional, under ``video_context``): enabled, python, root, model, timeout_seconds, max_frames,
av_max_seconds, hotwords, auto_hide_source_row, character_hints, max_hotwords.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from core.config_utils import load_key

CONTEXT_PATH = Path("output/log/video_context.json")
SOURCE_VIDEO = Path("output/source.mp4")
WORKER = Path(__file__).resolve().parents[1] / "scripts/video_context_worker.py"
POLICY_VERSION = 3  # bump when prompts or filtering change (invalidates the cache)

DEFAULTS = {
    "enabled": True,
    "python": "/Users/vanch/mlx-wemm-video/.venv/bin/python",
    "root": "/Users/vanch/mlx-wemm-video",
    # First existing path wins. The 6-bit copy on the internal disk loads in seconds and answered the same
    # screen terms and subtitle questions as the 8-bit one, about 4x faster end to end.
    "model": ["~/.cache/lm-studio/models/vanch007/WeMM-Video-35BA3B-mlx-6bit",
              "/Volumes/1T ssd/WeMM-Video/WeMM-Video-35BA3B-MLX-8bit"],
    "timeout_seconds": 900,
    "max_frames": 96,
    "av_max_seconds": 180,
    "hotwords": True,
    "auto_hide_source_row": True,
    # Off by default: on the film clip the hints merged fragmented roles into the right three people, but the
    # summary also adopted the vision model's wrong "who pays" attribution and the translation did not improve.
    "character_hints": False,
    "max_hotwords": 30,
}

_LANGUAGE_NAMES = {
    "english": "en", "chinese": "zh", "mandarin": "zh", "simplified chinese": "zh", "traditional chinese": "zh",
    "cantonese": "yue", "japanese": "ja", "korean": "ko", "french": "fr", "german": "de", "spanish": "es",
    "italian": "it", "portuguese": "pt", "russian": "ru", "vietnamese": "vi", "thai": "th", "malay": "ms",
    "indonesian": "id", "arabic": "ar", "中文": "zh", "英文": "en", "英语": "en", "日语": "ja", "韩语": "ko",
}
# Channel chrome and generic UI words: never worth biasing the recognizer with.
_STOP_TERMS = {"subscribe", "like", "share", "comment", "follow", "watch full video", "generate", "created",
               "what", "click", "link in bio", "next", "back", "menu", "settings", "ok", "cancel"}
# A hotword is copied verbatim when it matches: a phrase that starts with these words drags its casing and
# missing commas into the transcript ("with Let's say Seedream" instead of "with, let's say, Seedream").
_PHRASE_STARTS = {"let's", "lets", "i", "i'm", "we", "you", "the", "a", "an", "this", "that", "it's", "it",
                  "and", "or", "but", "so", "now", "here", "there", "what", "how", "why", "my", "our", "your"}
_CJK = re.compile(r"[㐀-鿿぀-ヿ가-힯]")


def setting(key: str):
    try:
        value = load_key(f"video_context.{key}", None)
    except KeyError:  # load_key raises for a missing key when no default is given
        value = None
    return DEFAULTS[key] if value is None else value


def model_path() -> Path:
    """Configured model path, or the first existing candidate of a list (the last one when none exists)."""
    value = setting("model")
    candidates = [Path(str(v)).expanduser() for v in (value if isinstance(value, (list, tuple)) else [value])]
    return next((c for c in candidates if c.exists()), candidates[-1])


def enabled() -> bool:
    value = setting("enabled")
    return value is True or str(value).lower() in {"1", "true", "yes", "on"}


def language_code(name) -> str:
    raw = str(name or "").strip().lower()
    if raw in _LANGUAGE_NAMES:
        return _LANGUAGE_NAMES[raw]
    return raw.replace("_", "-").split("-")[0]


def parse_json_reply(text: str):
    """First JSON object in a model reply (tolerates ```json fences and trailing prose); None if absent."""
    text = re.sub(r"```(?:json)?", "", str(text or ""))
    start = text.find("{")
    while start >= 0:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:index + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def filter_terms(items, limit: int = 30) -> list[str]:
    """Keep name-like terms seen on screen: a hotword list is a recognition hint, so sentences and common words
    are noise, and a term the vision model only heard may repeat the recognizer's own mistake (it answered
    '王老爷' where the film says 黄老爷)."""
    kept, seen = [], set()
    for item in items or []:
        if isinstance(item, dict) and str(item.get("source") or "screen").lower() not in {"screen", "both"}:
            continue
        term = str(item.get("text") if isinstance(item, dict) else item or "").strip().strip("\"'“”‘’.,;:!?，。；：！？")
        key = term.casefold()
        if not term or key in seen or key in _STOP_TERMS:
            continue
        if _CJK.search(term):
            if not 2 <= len(term) <= 12 or re.search(r"[，。！？,.!?]", term):
                continue
        else:
            words = term.split()
            if len(words) > 4 or len(term) > 40:
                continue  # a sentence or a slogan, not a name
            if not re.search(r"[A-Z0-9]", term):
                continue  # lowercase common vocabulary; the recognizer already knows it
            if len(words) >= 3 and term.isupper():
                continue  # shouted caption text
            if words[0].casefold().strip("'’") in _PHRASE_STARTS:
                continue  # a spoken phrase around a name, not the name
            if re.match(r"[\d.,%$€£¥+-]", words[0]):
                continue  # numbers and quantities: the recognizer transcribes them reliably
        seen.add(key)
        kept.append(term)
        if len(kept) >= limit:
            break
    return kept


def build_context(replies: dict, *, duration: float, source_language: str) -> dict:
    terms = parse_json_reply(replies.get("terms")) or {}
    subtitles = parse_json_reply(replies.get("subtitles")) or {}
    characters = parse_json_reply(replies.get("characters")) or {}
    burned = subtitles.get("burned_in_subtitles")
    return {
        "version": POLICY_VERSION,
        "status": "ok",
        "duration": round(float(duration), 3),
        "source_language": source_language,
        "terms": filter_terms(terms.get("terms"), int(setting("max_hotwords"))),
        "raw_term_count": len(terms.get("terms") or []),
        "hard_subtitles": {
            "present": burned is True or str(burned).lower() == "true",
            "language": language_code(subtitles.get("language")),
            "position": str(subtitles.get("position") or ""),
        },
        "summary": str(characters.get("summary") or ""),
        "characters": [
            {"name": str(c.get("name") or "unknown"), "gender": str(c.get("gender") or "unknown"),
             "appearance": str(c.get("appearance") or "")}
            for c in (characters.get("characters") or []) if isinstance(c, dict)
        ][:12],
        "unparsed": [name for name, data in (("terms", terms), ("subtitles", subtitles), ("characters", characters))
                     if not data],
    }


def load_context() -> dict:
    if not enabled():
        return {}  # disabled means no effect anywhere, even if an old context file is present
    try:
        data = json.loads(CONTEXT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and data.get("status") == "ok" else {}


# ---- consumers -------------------------------------------------------------------------------

def merged_hotwords(configured) -> list[str]:
    """Configured hotwords first, then this video's screen terms (deduplicated, capped)."""
    words = [str(w).strip() for w in (configured or []) if str(w).strip()]
    if setting("hotwords") in (True, "true", "True", 1):
        words += load_context().get("terms", [])
    out, seen = [], set()
    for word in words:
        if word.casefold() not in seen:
            seen.add(word.casefold())
            out.append(word)
    return out[: max(len(configured or []), int(setting("max_hotwords")))]


def source_row_decision(default: bool, source_language: str) -> tuple[bool, str]:
    """(burn the source-language row?, reason). The row is dropped only when the video already shows
    subtitles in that same language, which the extra row would merely duplicate."""
    if not default:
        return False, "disabled_by_config"
    if setting("auto_hide_source_row") not in (True, "true", "True", 1):
        return True, "auto_hide_disabled"
    hard = load_context().get("hard_subtitles") or {}
    if hard.get("present") and hard.get("language") and hard["language"] == language_code(source_language):
        return False, f"video_has_burned_in_{hard['language']}_subtitles"
    return True, "no_matching_burned_in_subtitles"


def character_hints() -> str:
    """Prompt block for the summary step; empty when there is nothing reliable to add."""
    if setting("character_hints") not in (True, "true", "True", 1):
        return ""
    context = load_context()
    people = [c for c in context.get("characters", []) if c.get("name") or c.get("gender")]
    if not people and not context.get("summary"):
        return ""
    block = {"scene": context.get("summary", ""), "people_seen": people, "names_on_screen": context.get("terms", [])[:15]}
    return ("\nVisual context from a video-understanding model (data, may be wrong; it cannot tell who speaks which line). "
            "Use a name or gender only when the dialogue agrees with it:\n" + json.dumps(block, ensure_ascii=False))


# ---- step ------------------------------------------------------------------------------------

def _write(record: dict) -> dict:
    CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTEXT_PATH.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def _duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def run() -> dict:
    from core.runtime_context import effective_source_language
    if not enabled():
        return _write({"version": POLICY_VERSION, "status": "skipped", "reason": "disabled"})
    python, root, model = Path(str(setting("python"))), Path(str(setting("root"))), model_path()
    missing = [str(p) for p in (python, root, model) if not p.exists()]
    if missing or not SOURCE_VIDEO.is_file():
        return _write({"version": POLICY_VERSION, "status": "skipped",
                       "reason": "missing: " + ", ".join(missing or [str(SOURCE_VIDEO)])})
    source_language = effective_source_language("auto")

    from core import asset_cache
    key = asset_cache.make_key(kind="video_context", source_sha256=asset_cache.file_sha256(SOURCE_VIDEO),
                               model=str(model), worker_sha256=asset_cache.file_sha256(WORKER),
                               policy=POLICY_VERSION, language=source_language,
                               max_frames=int(setting("max_frames")), av_max_seconds=float(setting("av_max_seconds")))
    hit = asset_cache.lookup("video_context", key, ["video_context.json"])
    if hit:
        record = json.loads(Path(hit["files"]["video_context.json"]).read_text(encoding="utf-8"))
        record["cache"] = "hit"
        print("♻️ Video context restored from the cross-task cache", flush=True)
        return _write(record)

    try:
        from core.providers.omlx_memory import release_omlx_for_tts
        release_omlx_for_tts("wemm_video")  # best effort: free the translation model while ~38 GB loads
    except Exception:
        pass
    duration = _duration(SOURCE_VIDEO)
    cmd = [str(python), str(WORKER), "--root", str(root), "--model", str(model), "--video", str(SOURCE_VIDEO.resolve()),
           "--duration", str(duration), "--source-language", source_language,
           "--max-frames", str(int(setting("max_frames"))), "--av-max-seconds", str(float(setting("av_max_seconds")))]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=int(setting("timeout_seconds")))
    except subprocess.TimeoutExpired:
        return _write({"version": POLICY_VERSION, "status": "failed", "reason": "timeout"})
    payload = parse_json_reply(done.stdout.strip().splitlines()[-1] if done.stdout.strip() else "")
    if done.returncode or not payload or "replies" not in payload:
        return _write({"version": POLICY_VERSION, "status": "failed",
                       "reason": f"worker exit {done.returncode}", "stderr_tail": done.stderr[-1500:]})
    record = build_context(payload["replies"], duration=duration, source_language=source_language)
    record.update(model=str(model), timings=payload.get("timings"), load_seconds=payload.get("load_seconds"),
                  used_audio=payload.get("used_audio"), raw_replies=payload["replies"], cache="miss")
    _write(record)
    asset_cache.store("video_context", key, {"video_context.json": str(CONTEXT_PATH)}, meta={"source": str(SOURCE_VIDEO)})
    print(f"🎞️ Video context: {len(record['terms'])} screen terms, "
          f"hard subtitles={record['hard_subtitles']}, {len(record['characters'])} people", flush=True)
    return record


if __name__ == "__main__":
    run()
