# Benchmark Plan

- Target: `llm`
- Dataset: `tests/fixtures/translation_zh_vi.jsonl`
- Status: `ready`
- Dataset status: `valid`
- Planned only: `True`
- Providers: deepseek_v4_pro, gpt_5_5
- Metrics: json_validity, line_count_preservation, terminology_accuracy, retry_count, latency_seconds, estimated_cost

This is a non-mutating benchmark plan. It does not call external model providers.
