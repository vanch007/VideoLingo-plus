# Benchmark Plan

- Target: `tts`
- Dataset: `tests/fixtures/tts_lines.yaml`
- Status: `ready`
- Dataset status: `valid`
- Planned only: `True`
- Providers: mlx_indextts2, edge_tts
- Metrics: duration_ratio, asr_content_score, reference_leak_score, natural_speed_factor, rtf, missing_or_tiny_audio_rate

This is a non-mutating benchmark plan. It does not call external model providers.
