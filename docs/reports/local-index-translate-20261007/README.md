# 本地 Index-Translate 接入与验证

最新原片重跑与后续优化：[optimization-final-20261007.md](optimization-final-20261007.md)。372 项回归通过，字幕候选已生成；整体质量 pending，配音未重新生成。下面的接入期结果与数量为历史快照。

日期：2026-10-07。VideoLingo 默认翻译已切换到本地 oMLX 的 `Index-Translate-35B-A3B-preview-oQ6e-fp16-mtp`。本次验证范围是文本处理、翻译、校审与字幕生成；不包含重新转录整部视频或生成配音。

## 最终流程

1. 保留源字幕分句、人物分析、前后文和术语提取。
2. Index 使用接近官方训练方式的中文指令和完整语言名称，批量翻译编号 JSON 的值。程序按编号恢复顺序，原文、人物和时间由输入持有，不让模型重新生成。
3. CJK 译文若残留可疑的普通英文词，仅重译该句。专名、缩写、代码、URL、占位符及明确认可的术语拼写受保护；仍无法处理时停止，不静默放行。
4. Index 默认分批进行只读事实自审（`audit_only`），保留邻接对白和术语；人称事实采用窄约束和修正记录。其他修复模式的末轮改写仍须复核。报告明确标注 `same_model_self_review`，不得代替来源、角色或独立质量验收。
5. 保留完整对白与配音用原始时间范围。显示字幕单独换行/分页，并可利用已观察的语音间隙短暂延长显示；不为字数反复改写译文。

全链路使用同一本地 Index 模型，没有云端失败回退。其他模型仍可使用原有直译/润色路径。

## 配置和使用

- `llm.provider: omlx`，地址 `http://127.0.0.1:8000/v1`，模型 ID 固定为上述 Index 模型。
- 请求显式使用 `temperature: 0`、关闭思考输出、`max_tokens: 8192`、300 秒超时、单并发。现有服务的纯文本引擎和 MTP depth 1 已启用，本次未修改服务持久设置。
- `translation.strategy: auto` 对 Index 选择本地直接翻译；设为 `legacy` 可恢复旧路径作对照。
- `translation.preserve_cues: true` 对本地路径保留完整字幕段。`subtitle.max_length` 不再触发该路径中的 LLM 重分句；如必须严格拆成短字幕，可设为 `false`，但需关注下述已知问题。
- 在界面选择 `omlx`，或在既有 CLI 命令中使用 `--llm omlx`。新任务直接使用本地配置。
- 旧云端产物保留在原处。CLI 与 Streamlit 都会阻止跨模型复用；现有根目录检测到四个旧阶段。先用 `python -m core.cli translation status` 查看，再按需使用新的独立工作目录，或明确授权后归档旧译文。不要直接混用旧配音任务。

缓存包含模型、服务地址、生成参数和请求格式。截断回复会在 JSON 修复之前被拒绝；不会把半截回复补成成功结果。来源清单记录实际模型、翻译策略及字幕处理方式。

## 验证结果

| 检查 | 结果 | 范围 |
|---|---|---|
| 修改前回归 | pass：300 项，45.07 秒 | 当前工作区原始基线 |
| 修改后回归 | pass：324 项，48.88 秒 | 全部 `tests/`，含新增缺行、错序、截断、缓存、术语保护及字幕保真检查 |
| 本地多语言样例 | pass：5 组、17 句、35 项断言 | 中英、中越、英中；人物称谓、金额/否定、技术术语、习语、电商文案 |
| 故意注入错误的反例 | pass：两句修复后通过复核 | `audition` 残留；否定丢失；两天→一天；五千→两千 |
| 现有台词片段 | pass：7 行、3 人物→7 条字幕，34.78 秒 | 人物/术语分析→翻译→校审→字幕时间轴；来源检查 0 stale，0 跨人物字幕行 |
| 独立母语盲审 | pending | 自动断言及同模型校审不能代替母语审校 |
| 完整配音/视频重新生成 | 未执行 | 原环境存在配音运行路径缺失及 OpenCV/NumPy 版本冲突 |

真实片段的摘要实际输出 **1,718 tokens**，超过服务原先的 1,024 默认额度，验证了显式输出额度的必要性。该片段翻译输出 139 tokens、校审输出 50 tokens，所有回复均正常结束。

修复前：`明天 audition 加油啊！`；修复后：`祝你明天试演成功！`。

数字/否定反例修复前：`今天发货，等一天收两千美元。`；修复后：`今天不要发货。等两天收到五千美元后再发货。`。

初始同模型新旧流程对照保留在原始运行目录。两条流程都曾漏译 `audition`，因此没有把模型自评通过当作充分证据。断言还修正了对合法 `percentage-point` 连字符及越南语货币简称 `tệ` 的误报，旧结果未覆盖。本次没有据此宣称普遍质量提升或稳定吞吐倍增。

## 证据与复现

- 汇总数据：[validation.json](validation.json)。
- 最终五组样例及原始请求：`run_workspaces/index_translate_quality_verified_20261007/`。
- 错译反例与复核：上述目录的 `bad_draft_check_v2/verification.json`。
- 真实片段及双语 SRT：`run_workspaces/index_translate_pipeline_verified_20261007/`，摘要到字幕的记录为 `verification.json`，原始片段来源为 `source_excerpt.json`。
- 初始流程对照：`run_workspaces/index_translate_integration_20261007/results.json`；断言复核为 `lexical_recheck.json`。
- 失败试验保留在 `index_translate_verified_20261007`、`index_translate_final_20261007`、`index_translate_native_prompt_20261007`、`index_translate_pipeline_excerpt_20261007`，均位于 `run_workspaces/` 下。

```bash
conda run --no-capture-output -n videolingo python -m pytest -q
conda run --no-capture-output -n videolingo python scripts/verify_local_translation.py --output /absolute/path/to/new-directory
```

第二条命令只允许本地 oMLX；输出目录必须尚不存在。可加 `--compare-legacy` 比较同一模型两种流程。词汇断言是针对固定样例的检查，不是通用语义评分。

## 已知边界

- Index 仍会漏译、误审或提出不必要的改写；有问题且无法收敛时应停止并保留证据。同模型校审不是独立质量证明。
- 疑似漏译检查只覆盖 CJK 输出中的小写拉丁词。合法的小写品牌或技术名应放入术语表，避免误报；它不是完整语言识别器。
- 旧分句器会拒绝把语义完整的短句再拆开，原有多重重试会重复失败。本地默认保留字幕段避开该问题，而不是删减译文满足长度。
- 真实片段验证沿用已有 ASR 时间戳，只覆盖连续七行；旧完整视频中的识别错误、独立人物/音色情绪评价和全片观感不在本次验收范围内。
- 服务健康信息中的旧 Qwen 默认模型 ID 实际不可调用（404）；最终配置显式选择已验证可用的 Index，不依赖该过期记录。

本次保留既有输出、冻结交付及原有未提交的音频改动；未提交 Git、发布或上传任何内容。
