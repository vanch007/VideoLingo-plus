"""Cross-task cache for expensive model outputs (separated stems, recognized chunks).

Lives outside every workspace (``~/.cache/videolingo-plus`` or ``$VIDEOLINGO_CACHE_DIR``), so a
new task on the same media does not repeat Demucs / MOSS / alignment. Keys are content hashes:
source bytes, model identity, settings and a schema number; file names never matter.

Only raw model output is cached. Human review files, translations and anything derived from them
stay in the workspace. A file is trusted only when its recorded SHA-256 still matches; entries are
written to a temporary directory and renamed, so a cancelled run never leaves a partial hit.
There is no automatic eviction: use ``stats()`` / ``purge()``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

SCHEMA = 1
MANIFEST = "cache_manifest.json"


def cache_root() -> Path:
    return Path(os.environ.get("VIDEOLINGO_CACHE_DIR") or Path.home() / ".cache" / "videolingo-plus").expanduser()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_key(**identity) -> str:
    payload = json.dumps({"schema": SCHEMA, **identity}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]


def _entry_dir(kind: str, key: str) -> Path:
    return cache_root() / kind / key


def _clone(src: Path, dst: Path) -> None:
    """Copy-on-write clone on APFS (no extra disk for large WAVs); plain copy elsewhere."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if os.uname().sysname == "Darwin":
        if subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True).returncode == 0:
            return
    shutil.copy2(src, dst)


def store(kind: str, key: str, files: dict[str, str], meta: dict | None = None) -> Path | None:
    """Store {name: path} atomically. Returns the entry directory, or None if a file is missing."""
    sources = {name: Path(p) for name, p in files.items()}
    if not all(p.is_file() for p in sources.values()):
        return None
    entry = _entry_dir(kind, key)
    entry.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=entry.parent))
    try:
        hashes = {}
        for name, src in sources.items():
            _clone(src, tmp / name)
            hashes[name] = file_sha256(tmp / name)
        (tmp / MANIFEST).write_text(json.dumps({"schema": SCHEMA, "files": hashes, "meta": meta or {}},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
        if entry.exists():
            shutil.rmtree(entry)
        tmp.rename(entry)
        prune_from_config(keep=entry)
        return entry
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        return None


def _touch(entry: Path) -> None:
    try:
        os.utime(entry / MANIFEST)  # recency for LRU pruning
    except OSError:
        pass


def lookup(kind: str, key: str, names: list[str] | None = None) -> dict | None:
    """Verified entry as {'dir': Path, 'meta': dict, 'files': {name: Path}}, else None."""
    entry = _entry_dir(kind, key)
    manifest = entry / MANIFEST
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            return None
        files = data["files"]
        if names and not set(names) <= set(files):
            return None
        for name, digest in files.items():
            if not (entry / name).is_file() or file_sha256(entry / name) != digest:
                return None
        _touch(entry)
        return {"dir": entry, "meta": data.get("meta", {}), "files": {n: entry / n for n in files}}
    except (OSError, ValueError, KeyError):
        return None


def restore(hit: dict, destinations: dict[str, str]) -> None:
    for name, dst in destinations.items():
        _clone(hit["files"][name], Path(dst))


def stats() -> dict:
    root = cache_root()
    out = {"root": str(root), "kinds": {}}
    if root.is_dir():
        for kind in sorted(p for p in root.iterdir() if p.is_dir()):
            entries = [p for p in kind.iterdir() if p.is_dir() and not p.name.startswith(".")]
            size = sum(f.stat().st_size for e in entries for f in e.rglob("*") if f.is_file())
            out["kinds"][kind.name] = {"entries": len(entries), "bytes": size}
    return out


def _entries() -> list[tuple[float, int, Path]]:
    """(last_used, bytes, dir) of every finished entry; unfinished temporaries are included as stale."""
    root, out = cache_root(), []
    if not root.is_dir():
        return out
    for kind in (p for p in root.iterdir() if p.is_dir()):
        for e in (p for p in kind.iterdir() if p.is_dir()):
            files = [f for f in e.rglob("*") if f.is_file()]
            size = sum(f.stat().st_size for f in files)
            manifest = e / MANIFEST
            if e.name.startswith("."):  # interrupted store
                used = 0.0
            else:
                used = manifest.stat().st_mtime if manifest.is_file() else 0.0
            out.append((used, size, e))
    return out


def prune(max_bytes: int | None = None, max_age_days: float | None = None, keep: Path | None = None) -> dict:
    """Delete least-recently-used entries until the cache fits ``max_bytes`` and nothing is older
    than ``max_age_days``. ``keep`` (the entry just stored) is never removed. Returns what was freed."""
    import time
    entries = sorted(_entries(), key=lambda t: t[0])
    total = sum(e[1] for e in entries)
    freed = removed = 0
    now = time.time()
    for used, size, path in entries:
        stale = max_age_days is not None and now - used > max_age_days * 86400
        over = max_bytes is not None and total > max_bytes
        if keep is not None and path == keep:
            continue
        if stale or over or path.name.startswith("."):
            shutil.rmtree(path, ignore_errors=True)
            total -= size
            freed += size
            removed += 1
    return {"removed": removed, "freed_bytes": freed, "remaining_bytes": total}


def prune_from_config(keep: Path | None = None) -> None:
    """Apply asset_cache_max_gb / asset_cache_max_age_days from config.yaml (0 or missing = unlimited)."""
    try:
        from core.config_utils import load_key
        gb = float(load_key("asset_cache_max_gb", 20) or 0)
        days = float(load_key("asset_cache_max_age_days", 90) or 0)
    except Exception:
        return
    if gb > 0 or days > 0:
        prune(int(gb * 1024 ** 3) if gb > 0 else None, days or None, keep)


def purge(kind: str | None = None) -> None:
    root = cache_root()
    target = root / kind if kind else root
    if target.is_dir():
        shutil.rmtree(target)
