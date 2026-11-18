import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest
from contextlib import nullcontext


@pytest.fixture(autouse=True)
def no_live_omlx_memory_mutations(monkeypatch):
    """Unit tests must never unload models from the user's running server."""
    monkeypatch.setattr("core.providers.mlx_tts.local_tts_memory_phase",
                        lambda backend: nullcontext())


@pytest.fixture(autouse=True)
def isolated_asset_cache(monkeypatch, tmp_path_factory):
    """Tests must neither read nor pollute the user's cross-task cache."""
    monkeypatch.setenv("VIDEOLINGO_CACHE_DIR", str(tmp_path_factory.mktemp("asset_cache")))


# Mode keys the contracts under test assume. The user's live config.yaml is in fast mode, which skips
# several review/repair stages by design; tests of those stages must not depend on it.
TEST_CONFIG_OVERRIDES = {
    "quality_mode": "high_sync",
    "fast_mode.enabled": False,
    # Thresholds of the repair/rescue classification tests (the live config tunes the speed cap higher).
    "dubbing_quality.max_speed_factor": 1.08,
}


@pytest.fixture(autouse=True)
def hermetic_config(monkeypatch, tmp_path_factory):
    """Each test reads (and may update_key) a private copy of config.yaml in a neutral mode, never the
    user's real file."""
    from core import config_utils
    source = config_utils.CONFIG_PATH if Path(config_utils.CONFIG_PATH).is_file() else str(ROOT / "config.yaml")
    with open(source, "r", encoding="utf-8") as handle:
        data = config_utils.yaml.load(handle)
    for dotted, value in TEST_CONFIG_OVERRIDES.items():
        node = data
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node[part]
        node[leaf] = value
    path = tmp_path_factory.mktemp("config") / "config.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        config_utils.yaml.dump(data, handle)
    monkeypatch.setattr(config_utils, "CONFIG_PATH", str(path))
