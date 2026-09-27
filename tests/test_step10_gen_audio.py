import pandas as pd
import pytest
from types import SimpleNamespace

from core import step10_gen_audio


def test_identical_pcm_export_preserves_readback_cache_mtime(tmp_path):
    import os
    from pydub.generators import Sine
    from pydub import AudioSegment

    output = tmp_path / "same.wav"
    Sine(440).to_audio_segment(duration=250).export(output, format="wav")
    audio = AudioSegment.from_wav(output)
    fixed_ns = 1_700_000_000_000_000_000
    os.utime(output, ns=(fixed_ns, fixed_ns))

    step10_gen_audio._export_wav_if_audio_changed(audio, str(output))

    assert output.stat().st_mtime_ns == fixed_ns


def test_merge_chunks_refuses_to_truncate_last_spoken_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(
        step10_gen_audio,
        "load_key",
        lambda key, default=None: {
            "speed_factor.accept": 1.0,
            "speed_factor.min": 1.0,
        }.get(key, default),
    )
    monkeypatch.setattr(step10_gen_audio, "adjust_audio_speed", lambda *_args: None)
    monkeypatch.setattr(step10_gen_audio, "get_audio_duration", lambda _path: 1.2)
    monkeypatch.setattr(
        step10_gen_audio,
        "TEMP_FILE_TEMPLATE",
        str(tmp_path / "{}_temp.wav"),
    )
    monkeypatch.setattr(
        step10_gen_audio,
        "OUTPUT_FILE_TEMPLATE",
        str(tmp_path / "{}.wav"),
    )
    tasks = pd.DataFrame([{
        "number": 1,
        "lines": ["Complete this sentence"],
        "real_dur": 1.2,
        "tol_dur": 1.0,
        "tolerance": 0.0,
        "gap": 0.0,
        "cut_off": 1,
        "start_time": "00:00:00,000",
        "end_time": "00:00:01,000",
    }])

    with pytest.raises(RuntimeError, match="Refusing to truncate spoken content"):
        step10_gen_audio.merge_chunks(tasks)


def test_indextts2_measured_overlong_retries_are_batched(monkeypatch):
    calls = []
    monkeypatch.setattr(
        step10_gen_audio,
        "get_quality_config",
        lambda: SimpleNamespace(enabled=True, max_rewrite_rounds=2),
    )
    monkeypatch.setattr(
        step10_gen_audio,
        "load_key",
        lambda key, default=None: "mlx_indextts2" if key == "tts_method" else default,
    )
    monkeypatch.setattr(
        step10_gen_audio,
        "actual_rewrite_needed",
        lambda row: float(row.get("real_dur", 0)) > float(row.get("available_duration", 1)),
    )
    monkeypatch.setattr(
        step10_gen_audio,
        "rewrite_task_lines",
        lambda row, **_kwargs: [f"short {int(row['number'])}"],
    )
    monkeypatch.setattr(step10_gen_audio, "_delete_temp_files_for_row", lambda _row: None)

    def fake_batch(_tasks_df, rows_df=None):
        calls.append(list(rows_df["number"]))
        return [(int(number), 0.8) for number in rows_df["number"]]

    monkeypatch.setattr(step10_gen_audio, "process_indextts2_batch", fake_batch)
    tasks = pd.DataFrame([
        {"number": 1, "text": "long one", "lines": ["long one"], "real_dur": 2.0, "available_duration": 1.0},
        {"number": 2, "text": "long two", "lines": ["long two"], "real_dur": 2.0, "available_duration": 1.0},
    ])

    result = step10_gen_audio.retry_overlong_rows(tasks)

    assert calls == [[1, 2]]
    assert result["real_dur"].tolist() == [0.8, 0.8]
    assert result["dubbing_rewrite_rounds"].tolist() == [1, 1]


def test_indextts2_fast_speech_retries_are_batched(monkeypatch, tmp_path):
    monkeypatch.setattr(step10_gen_audio, "TTS_TASKS_FILE", str(tmp_path / "tts_tasks.xlsx"))
    calls = []
    monkeypatch.setattr(
        step10_gen_audio,
        "get_quality_config",
        lambda: SimpleNamespace(enabled=True, max_rewrite_rounds=2, max_natural_speed_factor=1.12),
    )
    monkeypatch.setattr(
        step10_gen_audio,
        "load_key",
        lambda key, default=None: "mlx_indextts2" if key == "tts_method" else default,
    )
    monkeypatch.setattr(
        step10_gen_audio,
        "rewrite_task_lines",
        lambda row, **_kwargs: [f"short {int(row['number'])}"],
    )
    monkeypatch.setattr(step10_gen_audio, "_delete_temp_files_for_row", lambda _row: None)

    def fake_batch(_tasks_df, rows_df=None):
        calls.append(list(rows_df["number"]))
        return [(int(number), 0.8) for number in rows_df["number"]]

    monkeypatch.setattr(step10_gen_audio, "process_indextts2_batch", fake_batch)
    tasks = pd.DataFrame([
        {"number": 1, "text": "fast one", "lines": ["fast one"], "real_dur": 1.0, "speed_factor": 1.3},
        {"number": 2, "text": "fast two", "lines": ["fast two"], "real_dur": 1.0, "speed_factor": 1.2},
    ])

    result, changed = step10_gen_audio.retry_fast_speech_rows(tasks)

    assert changed is True
    assert calls == [[1, 2]]
    assert result["dubbing_rewrite_rounds"].tolist() == [1, 1]


def test_indextts2_native_fit_merge_pads_short_audio_without_speeding(monkeypatch, tmp_path):
    from pydub import AudioSegment

    monkeypatch.setattr(
        step10_gen_audio,
        "TEMP_FILE_TEMPLATE",
        str(tmp_path / "{}_temp.wav"),
    )
    monkeypatch.setattr(
        step10_gen_audio,
        "OUTPUT_FILE_TEMPLATE",
        str(tmp_path / "{}_final.wav"),
    )
    AudioSegment.silent(duration=400, frame_rate=16000).export(
        tmp_path / "1_0_temp.wav", format="wav"
    )
    tasks = pd.DataFrame([{
        "number": 1,
        "text": "Yes.",
        "lines": ["Yes."],
        "start_time": "00:00:01,000",
        "end_time": "00:00:02,000",
        "duration": 1.0,
        "tolerance": 0.0,
        "available_duration": 1.0,
    }])

    result = step10_gen_audio.merge_indextts2_native_fit(tasks)

    final = AudioSegment.from_wav(tmp_path / "1_0_final.wav")
    assert len(final) == 1000
    assert result.at[0, "speed_factor"] == 1.0
    assert result.at[0, "new_sub_times"] == [[1.0, 2.0]]


def test_indextts2_native_fit_merge_refuses_real_overflow(monkeypatch, tmp_path):
    from pydub import AudioSegment

    monkeypatch.setattr(step10_gen_audio, "TEMP_FILE_TEMPLATE", str(tmp_path / "{}_temp.wav"))
    monkeypatch.setattr(step10_gen_audio, "OUTPUT_FILE_TEMPLATE", str(tmp_path / "{}_final.wav"))
    AudioSegment.silent(duration=1200, frame_rate=16000).export(
        tmp_path / "1_0_temp.wav", format="wav"
    )
    tasks = pd.DataFrame([{
        "number": 1,
        "text": "Complete line.",
        "lines": ["Complete line."],
        "start_time": "00:00:00,000",
        "end_time": "00:00:01,000",
        "duration": 1.0,
        "tolerance": 0.0,
        "available_duration": 1.0,
    }])

    with pytest.raises(RuntimeError, match="Refusing to truncate spoken content"):
        step10_gen_audio.merge_indextts2_native_fit(tasks)


def test_indextts2_native_fit_removes_only_silent_rounding_tail(monkeypatch, tmp_path):
    from pydub import AudioSegment
    from pydub.generators import Sine

    monkeypatch.setattr(step10_gen_audio, "TEMP_FILE_TEMPLATE", str(tmp_path / "{}_temp.wav"))
    monkeypatch.setattr(step10_gen_audio, "OUTPUT_FILE_TEMPLATE", str(tmp_path / "{}_final.wav"))
    monkeypatch.setattr(
        step10_gen_audio,
        "adjust_audio_speed",
        lambda _src, dst, _speed: (
            Sine(440).to_audio_segment(duration=992).apply_gain(-6)
            + AudioSegment.silent(duration=16, frame_rate=44100)
        ).export(dst, format="wav"),
    )
    AudioSegment.silent(duration=1010, frame_rate=44100).export(
        tmp_path / "1_0_temp.wav", format="wav"
    )
    tasks = pd.DataFrame([{
        "number": 1,
        "text": "Complete line.",
        "lines": ["Complete line."],
        "start_time": "00:00:00,000",
        "end_time": "00:00:01,000",
        "duration": 1.0,
        "available_duration": 1.0,
    }])

    result = step10_gen_audio.merge_indextts2_native_fit(tasks)

    assert len(AudioSegment.from_wav(tmp_path / "1_0_final.wav")) == 1000
    assert result.at[0, "new_sub_times"] == [[0.0, 1.0]]


def test_indextts2_native_fit_allocates_multiline_window_by_measured_duration(monkeypatch, tmp_path):
    from pydub import AudioSegment
    from pydub.generators import Sine

    monkeypatch.setattr(step10_gen_audio, "TEMP_FILE_TEMPLATE", str(tmp_path / "{}_temp.wav"))
    monkeypatch.setattr(step10_gen_audio, "OUTPUT_FILE_TEMPLATE", str(tmp_path / "{}_final.wav"))
    Sine(440).to_audio_segment(duration=400).export(tmp_path / "1_0_temp.wav", format="wav")
    Sine(440).to_audio_segment(duration=800).export(tmp_path / "1_1_temp.wav", format="wav")
    tasks = pd.DataFrame([{
        "number": 1,
        "text": "Short. A much longer sentence.",
        "lines": ["Short.", "A much longer sentence."],
        "start_time": "00:00:00,000",
        "end_time": "00:00:01,500",
        "duration": 1.5,
        "available_duration": 1.5,
    }])

    result = step10_gen_audio.merge_indextts2_native_fit(tasks)

    windows = result.at[0, "new_sub_times"]
    assert result.at[0, "speed_factor"] == 1.0
    assert windows[0][1] - windows[0][0] == pytest.approx(0.5)
    assert windows[1][1] - windows[1][0] == pytest.approx(1.0)
    assert len(AudioSegment.from_wav(tmp_path / "1_0_final.wav")) == 500
    assert len(AudioSegment.from_wav(tmp_path / "1_1_final.wav")) == 1000


def test_tts_generation_checkpoint_invalidates_when_text_changes(monkeypatch, tmp_path):
    from pydub import AudioSegment

    monkeypatch.setattr(step10_gen_audio, "TEMP_FILE_TEMPLATE", str(tmp_path / "{}_temp.wav"))
    AudioSegment.silent(duration=500).export(tmp_path / "1_0_temp.wav", format="wav")
    tasks = pd.DataFrame([{
        "number": 1,
        "text": "Original line.",
        "lines": ["Original line."],
        "real_dur": 0.5,
    }])

    checkpoint = step10_gen_audio.mark_tts_generation_checkpoint(tasks.copy())
    assert step10_gen_audio.tts_generation_checkpoint_valid(checkpoint)

    checkpoint.at[0, "lines"] = ["Edited line."]
    checkpoint.at[0, "text"] = "Edited line."
    assert not step10_gen_audio.tts_generation_checkpoint_valid(checkpoint)


def test_indextts2_generation_reuses_valid_rows(monkeypatch):
    selected_numbers = []
    monkeypatch.setattr(step10_gen_audio, "apply_dubbing_budget_columns", lambda df: df)
    monkeypatch.setattr(
        step10_gen_audio,
        "load_key",
        lambda key, default=None: "mlx_indextts2" if key == "tts_method" else default,
    )
    monkeypatch.setattr(
        step10_gen_audio,
        "_tts_generation_row_valid",
        lambda row: int(row["number"]) == 1,
    )

    def fake_batch(_all_rows, selected):
        selected_numbers.extend(selected["number"].astype(int).tolist())
        return [(2, 1.25)]

    monkeypatch.setattr(step10_gen_audio, "process_indextts2_batch", fake_batch)
    monkeypatch.setattr(step10_gen_audio, "retry_overlong_rows", lambda df: df)
    tasks = pd.DataFrame([
        {"number": 1, "text": "Cached.", "real_dur": 0.9},
        {"number": 2, "text": "Changed.", "real_dur": None},
    ])

    result = step10_gen_audio.generate_tts_audio(tasks)

    assert selected_numbers == [2]
    assert result.loc[result.number.eq(1), "real_dur"].iat[0] == 0.9
    assert result.loc[result.number.eq(2), "real_dur"].iat[0] == 1.25


def test_trim_edge_silence_preserves_internal_pause():
    from pydub import AudioSegment
    from pydub.generators import Sine
    from core.audio_speed import trim_edge_silence

    tone = Sine(440).to_audio_segment(duration=200).apply_gain(-6)
    audio = AudioSegment.silent(duration=400) + tone + AudioSegment.silent(duration=500) + tone + AudioSegment.silent(duration=400)

    trimmed = trim_edge_silence(audio, keep_silence=50)

    assert 950 <= len(trimmed) <= 1050


def test_trim_edge_silence_preserves_quiet_cloned_speech():
    from pydub import AudioSegment
    from pydub.generators import Sine
    from core.audio_speed import trim_edge_silence

    quiet_word = Sine(330).to_audio_segment(duration=220).apply_gain(-52)
    audio = (
        AudioSegment.silent(duration=180)
        + quiet_word
        + AudioSegment.silent(duration=380)
        + quiet_word
        + AudioSegment.silent(duration=180)
    )

    trimmed = trim_edge_silence(audio, keep_silence=50)

    assert len(trimmed) >= 850


def test_speed_validation_never_clips_short_audio_tail(monkeypatch, tmp_path):
    from pydub import AudioSegment
    from pydub.generators import Sine
    from core import audio_speed

    output = tmp_path / "short.wav"
    Sine(440).to_audio_segment(duration=1090).export(output, format="wav")
    monkeypatch.setattr(audio_speed, "get_audio_duration", lambda _path: 1.09)

    audio_speed._validate_adjusted_duration(str(output), 1.1, 1.0)

    assert len(AudioSegment.from_wav(output)) == 1090


def test_indextts2_native_fit_uses_natural_speed_instead_of_clipping(monkeypatch, tmp_path):
    from pydub import AudioSegment
    from pydub.generators import Sine

    monkeypatch.setattr(step10_gen_audio, "TEMP_FILE_TEMPLATE", str(tmp_path / "{}_temp.wav"))
    monkeypatch.setattr(step10_gen_audio, "OUTPUT_FILE_TEMPLATE", str(tmp_path / "{}_final.wav"))
    monkeypatch.setattr(
        step10_gen_audio,
        "get_quality_config",
        lambda: SimpleNamespace(max_natural_speed_factor=1.12),
    )
    tone = Sine(440).to_audio_segment(duration=980).apply_gain(-6)
    (AudioSegment.silent(duration=300) + tone + AudioSegment.silent(duration=300)).export(
        tmp_path / "1_0_temp.wav", format="wav"
    )
    tasks = pd.DataFrame([{
        "number": 1,
        "text": "Complete line.",
        "lines": ["Complete line."],
        "start_time": "00:00:00,000",
        "end_time": "00:00:01,000",
        "duration": 1.0,
        "tolerance": 0.0,
        "available_duration": 1.0,
    }])

    result = step10_gen_audio.merge_indextts2_native_fit(tasks)

    final = AudioSegment.from_wav(tmp_path / "1_0_final.wav")
    assert len(final) == 1000
    assert 1.04 <= result.at[0, "speed_factor"] <= 1.12


def test_indextts2_native_fit_allows_short_reply_speed_limit(monkeypatch, tmp_path):
    from pydub.generators import Sine

    monkeypatch.setattr(step10_gen_audio, "TEMP_FILE_TEMPLATE", str(tmp_path / "{}_temp.wav"))
    monkeypatch.setattr(step10_gen_audio, "OUTPUT_FILE_TEMPLATE", str(tmp_path / "{}_final.wav"))
    monkeypatch.setattr(
        step10_gen_audio,
        "get_quality_config",
        lambda: SimpleNamespace(max_natural_speed_factor=1.12),
    )
    monkeypatch.setattr(
        step10_gen_audio,
        "load_key",
        lambda key, default=None: 1.35
        if key == "dubbing_quality.short_utterance_max_speed_factor"
        else default,
    )
    Sine(440).to_audio_segment(duration=450).export(tmp_path / "1_0_temp.wav", format="wav")
    tasks = pd.DataFrame([{
        "number": 1,
        "text": "Yes",
        "lines": ["Yes"],
        "start_time": "00:00:00,000",
        "end_time": "00:00:00,350",
        "duration": 0.35,
        "tolerance": 0.0,
        "available_duration": 0.35,
    }])

    result = step10_gen_audio.merge_indextts2_native_fit(tasks)

    assert (tmp_path / "1_0_final.wav").exists()
    # The actual atempo factor includes the configured timeline rounding margin.
    # Speed factor must strictly respect the configured limit (1.35) without margin bypass
    assert 1.25 <= result.at[0, "speed_factor"] <= 1.35


def test_content_repair_regenerates_selected_indextts2_rows_without_native_fit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        step10_gen_audio,
        "process_indextts2_batch",
        lambda _all, selected: calls.append(selected.copy()) or [],
    )
    monkeypatch.setattr(step10_gen_audio, "_trim_generated_edges", lambda *_args: None)
    monkeypatch.setattr(step10_gen_audio, "merge_indextts2_native_fit", lambda df: df)
    monkeypatch.setattr(
        step10_gen_audio,
        "get_quality_config",
        lambda: SimpleNamespace(max_natural_speed_factor=1.12),
    )
    tasks = pd.DataFrame([{
        "number": 1,
        "text": "Complete sentence.",
        "lines": ["Complete sentence."],
        "real_dur": 0.8,
        "available_duration": 1.0,
    }])

    result, rewritten = step10_gen_audio.repair_indextts2_content_rows(tasks, [0], False)

    assert rewritten == []
    assert result.at[0, "disable_native_fit"]
    assert len(calls) == 1
    assert calls[0].at[0, "disable_native_fit"]


def test_content_repair_reenables_duration_control_after_compact_rewrite(monkeypatch):
    calls = []
    monkeypatch.setattr(
        step10_gen_audio,
        "process_indextts2_batch",
        lambda _all, selected: calls.append(selected.copy()) or [],
    )
    durations = iter([1.8, 0.9])
    monkeypatch.setattr(
        step10_gen_audio,
        "_trim_generated_edges",
        lambda df, indices: [df.__setitem__("real_dur", [next(durations)]) for _ in [indices]],
    )
    monkeypatch.setattr(step10_gen_audio, "merge_indextts2_native_fit", lambda df: df)
    monkeypatch.setattr(step10_gen_audio, "_rewrite_content_rows", lambda _df, _idx: [1])
    monkeypatch.setattr(
        step10_gen_audio,
        "get_quality_config",
        lambda: SimpleNamespace(max_natural_speed_factor=1.12),
    )
    tasks = pd.DataFrame([{
        "number": 1,
        "text": "Compact sentence.",
        "lines": ["Compact sentence."],
        "real_dur": 0.8,
        "available_duration": 1.0,
    }])

    result, rewritten = step10_gen_audio.repair_indextts2_content_rows(tasks, [0], False)

    assert rewritten == [1]
    assert len(calls) == 2
    assert calls[0].at[0, "disable_native_fit"]
    assert not calls[1].at[0, "disable_native_fit"]
    assert not result.at[0, "disable_native_fit"]


def test_persistent_content_repair_uses_configured_voice_clone_fallback(monkeypatch):
    index_calls = []
    row_calls = []
    monkeypatch.setattr(
        step10_gen_audio,
        "load_key",
        lambda key, default=None: (
            "dots" if key == "dubbing_repair.low_content_fallback_backend" else default
        ),
    )
    monkeypatch.setattr(
        step10_gen_audio,
        "process_indextts2_batch",
        lambda _all, selected: index_calls.append(selected.copy()) or [],
    )
    monkeypatch.setattr(
        step10_gen_audio,
        "process_row",
        lambda row, _all: row_calls.append(dict(row)) or (int(row["number"]), 0.9),
    )
    monkeypatch.setattr(step10_gen_audio, "_delete_temp_files_for_row", lambda _row: None)
    monkeypatch.setattr(step10_gen_audio, "_trim_generated_edges", lambda *_args: None)
    monkeypatch.setattr(step10_gen_audio, "merge_indextts2_native_fit", lambda df: df)
    monkeypatch.setattr(step10_gen_audio, "_rewrite_content_rows", lambda _df, _idx: [])
    monkeypatch.setattr(
        step10_gen_audio,
        "get_quality_config",
        lambda: SimpleNamespace(max_natural_speed_factor=1.12),
    )
    tasks = pd.DataFrame([{
        "number": 43,
        "text": "By kneeling, this time.",
        "lines": ["By kneeling, this time."],
        "real_dur": 2.0,
        "available_duration": 2.048,
    }])

    result, rewritten = step10_gen_audio.repair_indextts2_content_rows(tasks, [0], True)

    assert rewritten == []
    assert index_calls == []
    assert len(row_calls) == 1
    assert row_calls[0]["tts_method"] == "mlx_dots_tts"
    assert row_calls[0]["tts_backend"] == "dots"
    assert result.at[0, "tts_method"] == "mlx_dots_tts"
    assert result.at[0, "tts_backend"] == "dots"
    assert result.at[0, "real_dur"] == 0.9


def test_content_repair_isolates_indextts2_rows_before_fallback(monkeypatch):
    selected_batches = []
    monkeypatch.setattr(
        step10_gen_audio,
        "process_indextts2_batch",
        lambda _all, selected: selected_batches.append(selected["number"].tolist()) or [],
    )
    tasks = pd.DataFrame([
        {"number": 1, "text": "First.", "lines": ["First."], "tts_method": "mlx_indextts2"},
        {"number": 2, "text": "Second.", "lines": ["Second."], "tts_method": "mlx_indextts2"},
    ])

    step10_gen_audio._regenerate_content_rows(tasks, [0, 1])

    assert selected_batches == [[1], [2]]


def test_persistent_unreadable_repair_does_not_shorten_complete_translation(monkeypatch):
    rewritten_indices = []
    monkeypatch.setattr(
        step10_gen_audio,
        "load_key",
        lambda key, default=None: (
            "dots" if key == "dubbing_repair.low_content_fallback_backend" else default
        ),
    )
    monkeypatch.setattr(
        step10_gen_audio,
        "_rewrite_content_rows",
        lambda _df, indices: rewritten_indices.extend(indices) or [],
    )
    monkeypatch.setattr(step10_gen_audio, "process_row", lambda row, _df: (row["number"], 1.4))
    monkeypatch.setattr(step10_gen_audio, "_delete_temp_files_for_row", lambda _row: None)
    monkeypatch.setattr(step10_gen_audio, "_trim_generated_edges", lambda *_args: None)
    monkeypatch.setattr(step10_gen_audio, "merge_indextts2_native_fit", lambda df: df)
    monkeypatch.setattr(
        step10_gen_audio,
        "get_quality_config",
        lambda: SimpleNamespace(max_natural_speed_factor=1.12),
    )
    tasks = pd.DataFrame([{
        "number": 43,
        "text": "By kneeling, this time.",
        "lines": ["By kneeling, this time."],
        "asr_transcript": "Nelly.",
        "real_dur": 1.4,
        "available_duration": 2.048,
    }])

    result, _ = step10_gen_audio.repair_indextts2_content_rows(tasks, [0], True)

    assert rewritten_indices == []
    assert result.at[0, "text"] == "By kneeling, this time."
    assert result.at[0, "lines"] == ["By kneeling, this time."]


def test_incomplete_content_retry_splits_clauses_without_dropping_words():
    tasks = pd.DataFrame([{
        "number": 42,
        "text": "You can. On your knees.",
        "lines": ["You can. On your knees."],
        "dubbing_rewrite_rounds": 0,
        "rewritten_for_dubbing": False,
        "rewrite_reason": "",
    }])

    changed = step10_gen_audio._split_content_rows_for_retry(tasks, [0])

    assert changed == [42]
    assert tasks.at[0, "lines"] == ["You can.", "On your knees."]
    assert tasks.at[0, "text"] == "You can. On your knees."
    assert tasks.at[0, "rewrite_reason"] == "asr_incomplete_split_retry"


def test_severely_unreadable_dots_retry_splits_even_without_tail_match(monkeypatch):
    generated = []
    monkeypatch.setattr(
        step10_gen_audio,
        "load_key",
        lambda key, default=None: "dots"
        if key == "dubbing_repair.low_content_fallback_backend"
        else default,
    )
    monkeypatch.setattr(
        step10_gen_audio,
        "process_row",
        lambda row, _df: generated.append(dict(row)) or (int(row["number"]), 0.8),
    )
    monkeypatch.setattr(step10_gen_audio, "_delete_temp_files_for_row", lambda _row: None)
    monkeypatch.setattr(step10_gen_audio, "_trim_generated_edges", lambda *_args: None)
    monkeypatch.setattr(step10_gen_audio, "merge_indextts2_native_fit", lambda df: df)
    monkeypatch.setattr(
        step10_gen_audio,
        "get_quality_config",
        lambda: SimpleNamespace(max_natural_speed_factor=1.12),
    )
    tasks = pd.DataFrame([{
        "number": 42,
        "text": "You can. On your knees.",
        "lines": ["You can. On your knees."],
        "asr_transcript": "can",
        "real_dur": 0.5,
        "available_duration": 2.048,
        "dubbing_rewrite_rounds": 0,
        "rewritten_for_dubbing": False,
        "rewrite_reason": "",
    }])

    result, rewritten = step10_gen_audio.repair_indextts2_content_rows(tasks, [0], True)

    assert rewritten == [42]
    assert generated[0]["lines"] == ["You can.", "On your knees."]
    assert result.at[0, "tts_method"] == "mlx_dots_tts"


def test_content_rewrite_provider_failure_is_nonfatal_and_returns_no_change(monkeypatch):
    monkeypatch.setattr(
        step10_gen_audio,
        "rewrite_task_lines",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
    )
    tasks = pd.DataFrame([{
        "number": 42,
        "text": "You can. On your knees.",
        "lines": ["You can. On your knees."],
        "asr_transcript": "On your knees.",
    }])

    assert step10_gen_audio._rewrite_content_rows(tasks, [0]) == []


def test_gen_audio_runs_content_gate_before_writing_tasks(monkeypatch):
    calls = []
    tasks = pd.DataFrame([{"number": 1, "text": "Complete."}])
    monkeypatch.setattr(step10_gen_audio, "prepare_reference_plan", lambda df: df)
    monkeypatch.setattr(
        step10_gen_audio,
        "load_key",
        lambda key, default=None: {
            "tts_method": "mlx_indextts2",
            "mlx_tts.backends.indextts2.fit_duration": True,
            "dubbing_quality.content_completion_gate": True,
            "dubbing_quality.asr_readback": True,
        }.get(key, default),
    )
    monkeypatch.setattr(step10_gen_audio.pd, "read_excel", lambda _path: tasks.copy())
    monkeypatch.setattr(step10_gen_audio, "generate_tts_audio", lambda df: df)
    monkeypatch.setattr(step10_gen_audio, "merge_indextts2_native_fit", lambda df: df)
    monkeypatch.setattr(
        step10_gen_audio,
        "run_indextts2_content_gate",
        lambda df: calls.append("content_gate") or df,
    )
    monkeypatch.setattr(step10_gen_audio, "write_dubbing_eval", lambda _df: {})
    monkeypatch.setattr(pd.DataFrame, "to_excel", lambda *_args, **_kwargs: calls.append("write"))

    step10_gen_audio.gen_audio()

    assert calls == ["write", "content_gate", "write"]
