# SiliconFlow 非 `Pro/` 聊天模型速度测试

- 生成时间（UTC）：2026-07-31T04:02:51.220192+00:00
- 平台：SiliconFlow OpenAI-compatible API（`https://api.siliconflow.cn/v1`）
- 样本：项目实际的中文→越南语三行访谈字幕批次，使用 `core.translate_once.translate_lines` 的原始忠实翻译与表达优化提示词；每阶段流式请求统一 `temperature=0`、`max_tokens=1024`，并校验 3 行 JSON 结构。
- 筛选：API 当前可访问、模型 ID 不以 `Pro/` 开头、且官方目录标为 `text/chat`。 其他类型保留在 JSON 中并标注为不适用。
- L0：官方目录的当前 L0 限速。`—` 表示该维度未设置；速率策略可调整，应以模型中心实时页面为准。

| 模型 | 平台 | 规格 | 上下文 | L0 RPM | L0 TPM | 忠实阶段首 Token (s) | 表达阶段首 Token (s) | 两阶段总耗时 (s) | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen/Qwen3.5-122B-A10B | SiliconFlow | 122B | 256K | 1000 | 20000 | — | — | — | fail: faithfulness |
| zai-org/GLM-4.5-Air | SiliconFlow | 106B | 128K | 1000 | 20000 | — | — | — | fail: faithfulness |
| zai-org/GLM-4.5V | SiliconFlow | 106B | 64K | 1000 | 20000 | — | — | — | fail: faithfulness |

## Chat 模型失败摘要

- `faithfulness`：3 个（忠实翻译 API 拒绝或请求失败）。

## 非聊天模型（未进行 Chat 速度测试）

这些模型没有作为 VideoLingo 的 LLM 翻译候选进行请求：

