# Benchmark Plan

- Target: `asr`
- Dataset: `tests/fixtures/asr_samples.yaml`
- Status: `seed_schema_only`
- Dataset status: `seed_schema_only`
- Planned only: `True`
- Providers: stable_ts_mlx, whisperx, funasr
- Metrics: cer, wer, timestamp_drift_seconds, word_timestamp_coverage, rtf

## Notes
- Dataset is a schema seed with placeholder media paths; replace placeholders before running a real benchmark.

## Validation Warnings
- item 1: audio is empty; ASR benchmark cannot run yet.

This is a non-mutating benchmark plan. It does not call external model providers.
