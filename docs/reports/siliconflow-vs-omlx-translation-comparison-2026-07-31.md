# SiliconFlow >100B 与本地 oMLX 翻译模型全面对比

- 生成时间（UTC）：2026-07-31T09:53:49.863528+00:00
- 工作负载：相同三行中文→越南语真实字幕，忠实翻译 + 表达优化两阶段。
- 质量口径：Qwen3.5-35B-A3B 与 DeepSeek-V3.1-Terminus 独立盲评；候选模型名称未写入评分提示。
- 限制：单一短样本，仅用于初步选型；不构成人工译审或整片质量结论。

## 速度、兼容性与质量

| 模型 | 平台 | 参数 | 两阶段总耗时 | 输出 tok/s（两阶段） | 质量 /100 | 结构状态 |
|---|---|---:|---:|---:|---:|---|
| oMLX Qwen3.6-35B-A3B | 本地 oMLX | 35B（A3B 激活，oQ6） | 24.612s | 49.21 / 52.85 | 81 | 通过 |
| GLM-4.5-Air | SiliconFlow | 106B | 26.267s | 53.66 / 42.27 | 83 | 通过 |
| GLM-4.5V | SiliconFlow | 106B | 42.715s | 51.74 / 77.76 | 78.5 | 通过 |
| DeepSeek-V3 | SiliconFlow | 671B | 84.871s | 24.58 / 14.81 | 83.5 | 通过 |
| DeepSeek-R1 | SiliconFlow | 671B | 98.345s | 21.93 / 23.96 | 83.5 | 通过 |
| DeepSeek-V3.1-Terminus | SiliconFlow | 671B | — | 24.69 / 24.49 | — | 失败：expressiveness_schema |
| DeepSeek-V3.2 | SiliconFlow | 671B | — | 23.09 / 22.41 | — | 失败：expressiveness_schema |

## 质量分项

| 模型 | 忠实度 /40 | 完整性 /20 | 越南语自然度 /20 | 术语 /10 | 配音简洁度 /10 | 有效评审 |
|---|---:|---:|---:|---:|---:|---:|
| oMLX Qwen3.6-35B-A3B | 33.5 | 17 | 15.5 | 8 | 7 | 2/2 |
| GLM-4.5-Air | 33.5 | 18 | 15.5 | 8.5 | 7.5 | 2/2 |
| GLM-4.5V | 33.5 | 18 | 14 | 7 | 6 | 2/2 |
| DeepSeek-V3 | 33.5 | 18 | 16 | 8.5 | 7.5 | 2/2 |
| DeepSeek-R1 | 33.5 | 18 | 16 | 8.5 | 7.5 | 2/2 |
| DeepSeek-V3.1-Terminus | — | — | — | — | — | 0/2 |
| DeepSeek-V3.2 | — | — | — | — | — | 0/2 |

## 规格、计费与限速

| 模型 | 上下文 | 输入价 | 输出价 | 抵用券 | L0 RPM | L0 TPM |
|---|---:|---:|---:|---|---:|---:|
| oMLX Qwen3.6-35B-A3B | 未在服务元数据公布 | 本地 ¥0 | 本地 ¥0 | 不需要 | — | — |
| GLM-4.5-Air | 128K | ¥1 / M Tokens | ¥6 / M Tokens | 适用 | 1000 | 20000 |
| GLM-4.5V | 64K | ¥1 / M Tokens | ¥6 / M Tokens | 适用 | 1000 | 20000 |
| DeepSeek-V3 | 160K | ¥2 / M Tokens | ¥8 / M Tokens | 适用 | 1000 | 100000 |
| DeepSeek-R1 | 160K | ¥4 / M Tokens | ¥16 / M Tokens | 适用 | 1000 | 100000 |
| DeepSeek-V3.1-Terminus | 160K | ¥4 / M Tokens | ¥12 / M Tokens | 适用 | 1000 | 100000 |
| DeepSeek-V3.2 | 160K | ¥2 / M Tokens | ¥3 / M Tokens | 适用 | 1000 | 100000 |

## 评审摘要

- **oMLX Qwen3.6-35B-A3B**（81）：The translation captures the general meaning but fails on specific entity names in line 2 (merging 'Mei Wan' and 'Yao Wan' into 'Meiwanyao'), which is a significant faithfulness error. The Vietnamese phrasing is generally understandable but occasionally slightly unnatural or verbose for subtitles (e.g., line 3). Terminology is mostly consistent, though 'đấu giá' (bidding) is a slight over-translation of '卷价格' (price war). / Bản dịch truyền tải tốt ý chính nhưng thiếu một số chi tiết quan trọng. Ngôn ngữ tự nhiên, thuật ngữ nhất quán, nhưng có thể ngắn gọn hơn cho phụ đề.
- **GLM-4.5-Air**（83）：The translation fails significantly on faithfulness due to mistranslating specific brand names (Jiao Ge Peng You -> Giao Bạn Thật) and confusing a person (Li Jiaqi) with a company entity. It also adds a question 'Why choose' in line 3 which is not in the source. While the general structure is preserved, the specific entity errors and added content reduce the score substantially. / High-quality translation with minor semantic losses. Faithful to core meaning but some nuanced expressions simplified. Vietnamese flows naturally with consistent terminology. Slightly verbose in places but overall suitable for subtitles.
- **GLM-4.5V**（78.5）：The translation fails significantly on brand names in Line 2, translating proper nouns (Li Jiaqi, Meiwan, Yao Wan, Make Friends) literally or incorrectly, which destroys the meaning. Line 1 is accurate but verbose. Line 3 suffers from unnatural phrasing and awkward terminology. The overall quality is low due to the critical brand name errors and lack of subtitle conciseness. / Bản dịch truyền tải đủ nội dung cơ bản nhưng có điểm yếu về tính tự nhiên và súc tích. Một số thuật ngữ và cụm từ dịch chưa tối ưu cho phụ đề, cần cải thiện độ mượt và độ dài câu.
- **DeepSeek-V3**（83.5）：The translation contains a major factual error in Line 2 by mistranslating the famous influencer Li Jiaqi's name and failing to recognize 'Jiao Ge Peng You' as a proper brand name, significantly harming faithfulness. While the structure is generally preserved, the loss of specific nuance regarding 'price wars' and the addition of 'supply chain operation' reduce accuracy. The Vietnamese phrasing is understandable but occasionally stiff. / Bản dịch trung thành về ý chính, ngôn ngữ tự nhiên, thuật ngữ nhất quán. Một số điểm mất sắc thái diễn đạt và thiếu chi tiết nhỏ, nhưng đạt chất tốt cho phụ đề.
- **DeepSeek-R1**（83.5）：The translation fails significantly on line 2 due to incorrect transliteration of famous Chinese names (Li Jiaqi, etc.) and mistranslation of the brand name 'Jiao Ge Peng You'. Line 1 adds an unnecessary verb ('chia sẻ'). While the general meaning is understandable, the factual errors regarding names and the unnatural phrasing of brand names severely impact faithfulness and naturalness. / Bản dịch tốt, tự nhiên, thuật ngữ nhất quán. Điểm trừ chính ở Line 2: hiểu sai đại từ '他们' (họ) dẫn đến sai lệch ý nghĩa câu hỏi then chốt. Một số chỗ có thể ngắn gọn hơn.
- **DeepSeek-V3.1-Terminus**：结构门禁未通过，未进入质量评分。
- **DeepSeek-V3.2**：结构门禁未通过，未进入质量评分。
