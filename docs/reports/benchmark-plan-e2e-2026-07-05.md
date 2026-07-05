# Benchmark Plan

- Target: `e2e`
- Dataset: `tests/fixtures/e2e_smoke.yaml`
- Status: `seed_schema_only`
- Dataset status: `seed_schema_only`
- Planned only: `True`
- Providers: baseline_current
- Metrics: total_runtime_seconds, repair_count, quality_pass_rate, manual_review_count, cost_per_finished_minute

## Notes
- Dataset is a schema seed with placeholder media paths; replace placeholders before running a real benchmark.

## Validation Warnings
- item 1: input is empty; E2E smoke benchmark cannot run yet.

This is a non-mutating benchmark plan. It does not call external model providers.
