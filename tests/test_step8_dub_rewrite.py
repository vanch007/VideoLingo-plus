from types import SimpleNamespace

import pandas as pd

from core import step8_2_gen_dub_chunks


def test_estimated_overlong_rewrite_uses_all_allowed_rounds(monkeypatch):
    rewrites = iter([["Still too long"], ["Fits now"]])
    monkeypatch.setattr(
        step8_2_gen_dub_chunks,
        "get_quality_config",
        lambda: SimpleNamespace(max_rewrite_rounds=2),
    )
    monkeypatch.setattr(
        step8_2_gen_dub_chunks,
        "rewrite_task_lines",
        lambda *_args, **_kwargs: next(rewrites),
    )
    monkeypatch.setattr(
        step8_2_gen_dub_chunks,
        "estimate_duration",
        lambda text, _estimator: 1.4 if text == "Still too long" else 0.9,
    )
    monkeypatch.setattr(
        step8_2_gen_dub_chunks,
        "estimated_rewrite_needed",
        lambda row: float(row.get("est_dur", 0)) > 1.08,
    )
    tasks = pd.DataFrame([{
        "number": 1,
        "text": "A sentence far too long",
        "lines": ["A sentence far too long"],
        "est_dur": 2.0,
        "duration": 1.0,
        "tol_dur": 1.0,
        "tolerance": 0.0,
        "available_duration": 1.0,
        "if_too_fast": 1,
    }])

    result = step8_2_gen_dub_chunks.rewrite_estimated_overlong_rows(tasks)

    assert result.at[0, "text"] == "Fits now"
    assert result.at[0, "dubbing_rewrite_rounds"] == 2
    assert bool(result.at[0, "rewritten_for_dubbing"]) is True
