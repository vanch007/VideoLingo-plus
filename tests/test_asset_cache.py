import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from core import asset_cache


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOLINGO_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


def test_store_lookup_restore_roundtrip(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"alpha"); b.write_bytes(b"beta")
    key = asset_cache.make_key(kind="t", sha="x")
    assert asset_cache.lookup("t", key) is None
    assert asset_cache.store("t", key, {"a": str(a), "b": str(b)}, meta={"n": 1})
    hit = asset_cache.lookup("t", key, ["a", "b"])
    assert hit["meta"] == {"n": 1}
    out = tmp_path / "out"
    asset_cache.restore(hit, {"a": str(out / "a2"), "b": str(out / "b2")})
    assert (out / "a2").read_bytes() == b"alpha"
    assert asset_cache.stats()["kinds"]["t"]["entries"] == 1


def test_key_depends_on_content_identity_not_names():
    assert asset_cache.make_key(sha="1", model="m") == asset_cache.make_key(model="m", sha="1")
    assert asset_cache.make_key(sha="1", model="m") != asset_cache.make_key(sha="2", model="m")
    assert asset_cache.make_key(sha="1", chunk=0) != asset_cache.make_key(sha="1", chunk=300)


def test_corrupted_or_missing_entry_is_a_miss(tmp_path, cache_dir):
    a = tmp_path / "a.bin"; a.write_bytes(b"data")
    key = asset_cache.make_key(sha="x")
    asset_cache.store("t", key, {"a": str(a)})
    (cache_dir / "t" / key / "a").write_bytes(b"tampered")
    assert asset_cache.lookup("t", key) is None
    assert asset_cache.store("t", key, {"a": str(tmp_path / "nope")}) is None  # missing source


def test_purge(tmp_path):
    a = tmp_path / "a.bin"; a.write_bytes(b"x")
    asset_cache.store("t", "k", {"a": str(a)})
    asset_cache.purge("t")
    assert asset_cache.stats()["kinds"] == {}


def test_asr_chunk_cache_shared_between_workspaces(tmp_path, monkeypatch):
    from core.all_whisper_methods import local_aligned as la
    key = la._global_key({"fp": 1}, 0.0, 180.0, {})
    assert key and la._global_key({"fp": 1}, 0.0, 180.0, {"review_file": "r.json"}) is None  # human review: never shared
    receipt = tmp_path / "r.json"
    receipt.write_text(json.dumps({"fingerprint": {"fp": 1}, "result": {"segments": []}}))
    la._global_put(key, receipt)
    assert la._global_get(key)["result"] == {"segments": []}
    assert la._global_get(la._global_key({"fp": 1}, 0.0, 181.0, {})) is None  # other window


class FakeSeparator:
    """vocals = 0.5 * input, background = 0.25 * input (context-free, so chunking must match)."""
    def separate_tensor(self, wav, sr):
        return wav, {"vocals": wav * 0.5, "drums": wav * 0.15, "bass": wav * 0.1}


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_chunked_separation_matches_whole_file_and_length(tmp_path):
    from core.all_whisper_methods.demucs_vl import separate_in_chunks
    sr = 44100
    t = np.arange(int(23.37 * sr)) / sr
    audio = np.stack([0.4 * np.sin(2 * np.pi * 220 * t), 0.3 * np.sin(2 * np.pi * 330 * t)], axis=1)
    src = tmp_path / "in.wav"
    sf.write(str(src), audio, sr, subtype="PCM_16")
    vp, bp = separate_in_chunks(FakeSeparator(), str(src), sr, str(tmp_path / "v.wav"), str(tmp_path / "b.wav"),
                                chunk_seconds=7, context_seconds=1.0)
    v, _ = sf.read(str(tmp_path / "v.wav")); b, _ = sf.read(str(tmp_path / "b.wav"))
    ref, _ = sf.read(str(src))
    assert len(v) == len(b) == len(ref)                       # sample-exact length
    assert np.abs(v - 0.5 * ref).max() < 3e-4                  # seams invisible (16-bit rounding only)
    assert np.abs(b - 0.25 * ref).max() < 3e-4
    assert vp == pytest.approx(0.5 * 0.4, abs=0.01)
