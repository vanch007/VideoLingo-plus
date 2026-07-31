# SiliconFlow 非 `Pro/` 聊天模型速度测试

- 生成时间（UTC）：2026-07-30T16:28:04.766387+00:00
- 平台：SiliconFlow OpenAI-compatible API（`https://api.siliconflow.cn/v1`）
- 样本：项目实际的中文→越南语三行访谈字幕批次，使用 `core.translate_once.translate_lines` 的原始忠实翻译与表达优化提示词；每阶段流式请求统一 `temperature=0`、`max_tokens=1024`，并校验 3 行 JSON 结构。
- 筛选：API 当前可访问、模型 ID 不以 `Pro/` 开头、且官方目录标为 `text/chat`。其他类型保留在 JSON 中并标注为不适用。
- L0：官方目录的当前 L0 限速。`—` 表示该维度未设置；速率策略可调整，应以模型中心实时页面为准。

| 模型 | 平台 | 规格 | 上下文 | L0 RPM | L0 TPM | 忠实阶段首 Token (s) | 表达阶段首 Token (s) | 两阶段总耗时 (s) | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ByteDance-Seed/Seed-OSS-36B-Instruct | SiliconFlow | 36B | 256K | 1000 | 20000 | 1.042 | 1.328 | 141.429 | pass |
| MiniMaxAI/MiniMax-M2.5 | SiliconFlow | 229B | 192K | 1000 | 100000 | 1.137 | — | — | fail |
| PaddlePaddle/PaddleOCR-VL-1.5 | SiliconFlow | 1B | 未公布 | 1000 | 80000 | 14.165 | — | — | fail |
| Qwen/Qwen2.5-14B-Instruct | SiliconFlow | 14B | 32K | 1000 | 40000 | 0.855 | 0.917 | 28.15 | pass |
| Qwen/Qwen2.5-32B-Instruct | SiliconFlow | 32B | 32K | 1000 | 40000 | 1.138 | 1.433 | 39.559 | pass |
| Qwen/Qwen2.5-72B-Instruct | SiliconFlow | 72B | 32K | 1000 | 20000 | 0.981 | 1.451 | 40.63 | pass |
| Qwen/Qwen2.5-72B-Instruct-128K | SiliconFlow | 72B | 128K | 1000 | 20000 | 1.045 | 1.163 | 31.575 | pass |
| Qwen/Qwen2.5-7B-Instruct | SiliconFlow | 7B | 32K | 1000 | 50000 | 1.049 | — | — | fail |
| Qwen/Qwen3-14B | SiliconFlow | 14B | 128K | 1000 | 40000 | 1.871 | 0.859 | 18.309 | pass |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SiliconFlow | 30B | 256K | 1000 | 40000 | 1.009 | 1.267 | 20.444 | pass |
| Qwen/Qwen3-32B | SiliconFlow | 32B | 128K | 1000 | 40000 | 1.019 | 1.012 | 59.42 | pass |
| Qwen/Qwen3-8B | SiliconFlow | 8B | 128K | 1000 | 50000 | 1.251 | 2.058 | 390.257 | pass |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | SiliconFlow | 30B | 256K | 1000 | 40000 | 0.918 | 1.29 | 14.751 | pass |
| Qwen/Qwen3-Omni-30B-A3B-Captioner | SiliconFlow | 30B | 64K | 1000 | 40000 | 0.918 | 0.887 | — | fail |
| Qwen/Qwen3-Omni-30B-A3B-Instruct | SiliconFlow | 30B | 64K | 1000 | 40000 | 0.972 | 0.947 | 16.024 | pass |
| Qwen/Qwen3-Omni-30B-A3B-Thinking | SiliconFlow | 30B | 64K | 1000 | 40000 | 0.747 | 0.766 | 78.019 | pass |
| Qwen/Qwen3-VL-30B-A3B-Instruct | SiliconFlow | 30B | 256K | 1000 | 40000 | 1.312 | 1.085 | 26.597 | pass |
| Qwen/Qwen3-VL-30B-A3B-Thinking | SiliconFlow | 30B | 256K | 1000 | 40000 | 0.909 | 0.864 | 124.173 | pass |
| Qwen/Qwen3-VL-32B-Instruct | SiliconFlow | 32B | 256K | 1000 | 80000 | 0.892 | 0.966 | — | fail |
| Qwen/Qwen3-VL-32B-Thinking | SiliconFlow | 32B | 256K | 1000 | 80000 | 0.891 | — | — | fail |
| Qwen/Qwen3-VL-8B-Instruct | SiliconFlow | 8B | 256K | 1000 | 80000 | 1.107 | 1.116 | 21.936 | pass |
| Qwen/Qwen3-VL-8B-Thinking | SiliconFlow | 8B | 256K | 1000 | 80000 | 1.199 | 1.13 | 223.596 | pass |
| Qwen/Qwen3.5-122B-A10B | SiliconFlow | 122B | 256K | 1000 | 20000 | 1.094 | — | — | fail |
| Qwen/Qwen3.5-27B | SiliconFlow | 27B | 256K | 1000 | 40000 | 0.884 | — | — | fail |
| Qwen/Qwen3.5-35B-A3B | SiliconFlow | 35B | 256K | 1000 | 40000 | 0.854 | — | — | fail |
| Qwen/Qwen3.5-397B-A17B | SiliconFlow | 397B | 256K | 500 | 2000000 | 1.505 | 2.036 | 122.125 | pass |
| Qwen/Qwen3.5-4B | SiliconFlow | 4B | 256K | 1000 | 80000 | 0.753 | — | — | fail |
| Qwen/Qwen3.5-9B | SiliconFlow | 9B | 256K | 1000 | 80000 | 1.033 | — | — | fail |
| Qwen/Qwen3.6-27B | SiliconFlow | 27B | 256K | 1000 | 40000 | 0.863 | — | — | fail |
| Qwen/Qwen3.6-35B-A3B | SiliconFlow | 35B | 256K | 1000 | 40000 | 1.052 | — | — | fail |
| THUDM/GLM-4-32B-0414 | SiliconFlow | 32B | 32K | 1000 | 40000 | 0.957 | — | — | fail |
| THUDM/GLM-4-9B-0414 | SiliconFlow | 9B | 32K | 1000 | 50000 | 1.967 | — | — | fail |
| THUDM/GLM-Z1-9B-0414 | SiliconFlow | 9B | 128K | 1000 | 50000 | 0.793 | 2.194 | 90.336 | pass |
| deepseek-ai/DeepSeek-OCR | SiliconFlow | 3B | 8K | 1000 | 80000 | 0.662 | — | — | fail |
| deepseek-ai/DeepSeek-R1 | SiliconFlow | 671B | 160K | 1000 | 100000 | 1.387 | 1.736 | — | fail |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | SiliconFlow | 8B | 128K | 1000 | 50000 | 0.92 | — | — | fail |
| deepseek-ai/DeepSeek-V3 | SiliconFlow | 671B | 160K | 1000 | 100000 | — | — | — | fail |
| deepseek-ai/DeepSeek-V3.1-Terminus | SiliconFlow | 671B | 160K | 1000 | 100000 | 1.467 | 1.143 | — | fail |
| deepseek-ai/DeepSeek-V3.2 | SiliconFlow | 671B | 160K | 1000 | 100000 | 4.842 | — | — | fail |
| deepseek-ai/DeepSeek-V4-Flash | SiliconFlow | 284B | 1024K | 500 | 2000000 | 1.245 | — | — | fail |
| deepseek-ai/DeepSeek-V4-Pro | SiliconFlow | 1600B | 1024K | 500 | 2000000 | — | — | — | fail |
| inclusionAI/Ling-flash-2.0 | SiliconFlow | 100B | 128K | 1000 | 40000 | — | — | — | fail |
| inclusionAI/Ling-mini-2.0 | SiliconFlow | 16B | 128K | 1000 | 40000 | — | — | — | fail |
| meituan-longcat/LongCat-2.0 | SiliconFlow | 1600B | 1024K | 500 | 2000000 | — | — | — | fail |
| moonshotai/Kimi-K2.7-Code | SiliconFlow | 1000B | 256K | 500 | 2000000 | — | — | — | fail |
| nex-agi/Nex-N2-Pro | SiliconFlow | 397B | 256K | 1000 | 80000 | — | — | — | fail |
| stepfun-ai/Step-3.5-Flash | SiliconFlow | 196B | 256K | 1000 | 10000 | — | — | — | fail |
| tencent/Hunyuan-A13B-Instruct | SiliconFlow | 80B | 128K | 1000 | 20000 | — | — | — | fail |
| tencent/Hunyuan-MT-7B | SiliconFlow | 7B | 32K | 1000 | 80000 | — | — | — | fail |
| zai-org/GLM-4.5-Air | SiliconFlow | 106B | 128K | 1000 | 20000 | — | — | — | fail |
| zai-org/GLM-4.5V | SiliconFlow | 106B | 64K | 1000 | 20000 | — | — | — | fail |
| zai-org/GLM-5.2 | SiliconFlow | 753B | 1024K | 500 | 2000000 | — | — | — | fail |

## Chat 模型失败摘要

- `faithfulness_json`：17 个模型在忠实翻译阶段未返回可解析 JSON。
- `expressiveness_json`：4 个模型在表达优化阶段未返回可解析 JSON。
- `faithfulness`：13 个模型在忠实翻译阶段被 API 拒绝，其中 12 个为账户余额不足，`deepseek-ai/DeepSeek-V3` 为服务繁忙限流。

这些项目是本次账户和请求下的不可用结果，不代表模型本身的永久质量结论。

## ≥100B 可用候选的输出吞吐

| 模型 | 忠实翻译 | 表达优化 | 说明 |
|---|---:|---:|---|
| `Qwen/Qwen3.5-397B-A17B` | 5,749 tokens / 64.725s = **88.82 tok/s** | 4,834 tokens / 57.400s = **84.22 tok/s** | SiliconFlow usage 返回的 completion tokens；思考模型的该数字可能包含 reasoning tokens。 |

其余 ≥100B 模型本次未完成两阶段输出，因而没有可比较的输出吞吐值。

## 非聊天模型（未进行 Chat 速度测试）

这些模型没有作为 VideoLingo 的 LLM 翻译候选进行请求：

- `BAAI/bge-large-en-v1.5`：`text/embedding`；规格 未公布；L0 RPM/TPM 2000/500000。
- `BAAI/bge-large-zh-v1.5`：`text/embedding`；规格 未公布；L0 RPM/TPM 2000/500000。
- `BAAI/bge-m3`：`text/embedding`；规格 未公布；L0 RPM/TPM 2000/500000。
- `BAAI/bge-reranker-v2-m3`：`text/reranker`；规格 未公布；L0 RPM/TPM 2000/500000。
- `FunAudioLLM/CosyVoice2-0.5B`：`audio/text-to-speech`；规格 1B；L0 RPM/TPM 1000/40000。
- `FunAudioLLM/SenseVoiceSmall`：`audio/speech-to-text`；规格 未公布；L0 RPM/TPM 1000/50000。
- `Kwai-Kolors/Kolors`：`image/text-to-image`；规格 未公布；L0 RPM/TPM —/—。
- `Qwen/Qwen-Image`：`image/text-to-image`；规格 20B；L0 RPM/TPM —/—。
- `Qwen/Qwen-Image-Edit`：`image/text-to-image`；规格 20B；L0 RPM/TPM —/—。
- `Qwen/Qwen-Image-Edit-2509`：`image/text-to-image`；规格 20B；L0 RPM/TPM —/—。
- `Qwen/Qwen3-Embedding-0.6B`：`text/embedding`；规格 1B；L0 RPM/TPM 2000/1000000。
- `Qwen/Qwen3-Embedding-4B`：`text/embedding`；规格 4B；L0 RPM/TPM 2000/1000000。
- `Qwen/Qwen3-Embedding-8B`：`text/embedding`；规格 8B；L0 RPM/TPM 2000/1000000。
- `Qwen/Qwen3-Reranker-0.6B`：`text/reranker`；规格 1B；L0 RPM/TPM 2000/1000000。
- `Qwen/Qwen3-Reranker-4B`：`text/reranker`；规格 4B；L0 RPM/TPM 2000/1000000。
- `Qwen/Qwen3-Reranker-8B`：`text/reranker`；规格 8B；L0 RPM/TPM 2000/1000000。
- `Qwen/Qwen3-VL-Embedding-8B`：`text/embedding`；规格 8B；L0 RPM/TPM 2000/1000000。
- `Qwen/Qwen3-VL-Reranker-8B`：`text/reranker`；规格 8B；L0 RPM/TPM 2000/1000000。
- `TeleAI/TeleSpeechASR`：`audio/speech-to-text`；规格 未公布；L0 RPM/TPM 1000/50000。
- `Tongyi-MAI/Z-Image`：`image/text-to-image`；规格 6B；L0 RPM/TPM —/—。
- `Tongyi-MAI/Z-Image-Turbo`：`image/text-to-image`；规格 6B；L0 RPM/TPM —/—。
- `Wan-AI/Wan2.2-I2V-A14B`：`video/image-to-video`；规格 27B；L0 RPM/TPM —/—。
- `Wan-AI/Wan2.2-T2V-A14B`：`video/text-to-video`；规格 27B；L0 RPM/TPM —/—。
- `baidu/ERNIE-Image-Turbo`：`image/text-to-image`；规格 8B；L0 RPM/TPM —/—。
- `fnlp/MOSS-TTSD-v0.5`：`audio/text-to-speech`；规格 2B；L0 RPM/TPM 1000/40000。
