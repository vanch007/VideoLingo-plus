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
