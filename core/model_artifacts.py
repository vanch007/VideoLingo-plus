"""Content fingerprints for a pinned local model served through an API."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path


@lru_cache(maxsize=8)
def _fingerprint_files(stamps: tuple) -> str:
    files = []
    for filename, _size, _mtime, _ctime, _inode in stamps:
        digest = hashlib.sha256()
        with Path(filename).open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 << 20), b""):
                digest.update(chunk)
        files.append((Path(filename).name, digest.hexdigest()))
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def local_model_fingerprint(model_path: str) -> str:
    """Hash once per file revision; same-name weight changes invalidate caches."""
    root = Path(model_path).expanduser().resolve()
    weights = sorted(root.glob("model*.safetensors"))
    if not weights or not (root / "config.json").is_file() or not (root / "tokenizer.json").is_file():
        raise ValueError(f"Pinned local model is incomplete: {root}")
    paths = sorted(set(weights + [p for pattern in (
        "config.json", "generation_config.json", "model*.index.json", "tokenizer*.json",
        "chat_template*", "vocab.json", "merges.txt") for p in root.glob(pattern) if p.is_file()]))
    stamps = tuple((str(p), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
                   for p in paths for stat in (p.stat(),))
    return _fingerprint_files(stamps)
