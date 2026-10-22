# 本地翻译优化：原片验证与剩余问题

本轮代码修复、回归和字幕生成为 **pass**；外部对白核验后的整体翻译质量为 **fail**，`accepted: false`。当前候选有明确剧情和动作错译，不适合正式交付。完整评估见 [29 个语义单元对照报告](/Users/vanch/VideoLingo-plus/docs/reports/local-index-translate-20261007/reference-assessment-20261007/reference-quality-assessment.md)。本轮尚未生成新的配音。

## 当前产物

- [英文字幕预览视频](/Users/vanch/VideoLingo-plus/run_workspaces/index_quality_v12_readability_20261007/review_preview.mp4)
- [双语字幕视频](/Users/vanch/VideoLingo-plus/run_workspaces/index_quality_v12_readability_20261007/output/AI字幕.mp4)
- [逐句原声、角色与识别分歧复核页](/Users/vanch/VideoLingo-plus/run_workspaces/index_quality_v12_readability_20261007/review.html)
- [英文 SRT](/Users/vanch/VideoLingo-plus/run_workspaces/index_quality_v12_readability_20261007/output/trans.srt)
- [机器可读验证记录](/Users/vanch/VideoLingo-plus/docs/reports/local-index-translate-20261007/optimization-final-20261007.json)

输入为冻结 V9 目录里的原始 `source.mp4`，约 170.85 秒，中文译英文。本轮 fresh ASR 在 `index_quality_v9_original_audio_20261007` 使用原始无损音频和本地 MLX Whisper large-v3；`index_quality_v11_validated_20261007` 重新运行摘要、角色分析、Index 翻译和审校；最终 `index_quality_v12_readability_20261007` 校验并复用这些上游结果，只重建字幕显示和视频。复用范围记录在最终工作区的 `upstream_reuse.json`，没有把全部阶段描述为最后一次运行时重新执行。

## 修复及实测

| 项目 | 结果 | 证据与限制 |
|---|---|---|
| 原始音频转录覆盖 | pass | 60/60 个合格声学片段有文本覆盖，缺失 0；覆盖不证明识别文字正确 |
| 独立 ASR 核对 | needs_review | 13 组识别分歧，其中 3 组为高分歧；保留音频、两套文字和哈希 |
| 说话角色分析 | needs_review | 隐藏声纹标签后逐行重新推断；45 行中 21 行有带引用的角色假设，28 行动作/指代关系待复核；这些数字不等于正确率 |
| 人称事实约束 | pass（有限范围） | 两处真实错误得到窄规则修正，最终 45 行没有触发已有事实约束；不等于完整语义通过 |
| 翻译自审 | pass（同模型自审） | 45 行，0 个模型报告的问题，0 次审校改写；来源为 `same_model_self_review`，不是独立验收 |
| 字幕显示 | needs_review | 45 个完整语义单元排成 57 条显示字幕；9 句利用语音间隙延长显示，警告从 16 处降至 12 处 |
| 原文/译文及配音时间范围 | pass | R11/R12 译文表和配音用 SRT 字节相同；显示延时只使用已观察的语音间隙，留存调整前后时间 |
| 阶段来源清单 | pass | 8/8 个字幕链阶段清单通过；模型来源检查 0 stale / 0 warning |
| 自动回归 | pass | 372 项，53.74 秒，见 `optimization-tests-v12.txt`；`git diff --check` 通过 |
| 视频长度 | pass（长度检查） | 候选 170.851995 秒，与原片差 0.431 毫秒；不能代替字幕逐句同步或配音评价 |
| 整体翻译质量 | fail | 外部对白核验发现 1 个核心剧情错误、10 个重大错误；自动报告的 pending 为核验前状态，不能继续作为本例结论 |

具体修复包括：空白音频探测不再触发 stable-ts 的越界崩溃；转录和恢复优先使用原始无损音频；短词被声纹边界切开时只调整文本分组，保留原声纹和词时间；角色身份与动作关系分别验证；审校引用异常不再阻断可查看的候选，但仍标记待复核；字幕显示与完整译文、配音单元分离。

字幕显示最多提前 0.15 秒、延后 0.65 秒，只能进入原生词时间所支持的空隙，相邻字幕共享间隙边界。回归包含其他语音挡住空隙、重叠语音、相邻字幕争用空隙，以及完整译文/时间范围保持不变的反例。

## 这个案例中的人称与转述

原识别行 L00035 含“我出一百八十万……接上我的腿”。Index 草稿把所有者译为 `your leg`；最终窄规则改为 `my leg`。规则只在原文所有者明确、译文也只有一个所有者时应用，修改前后记录在 `translation_fact_normalizations.json`。

L00040 含“给你一百八十万两银子……接上他的腿”，没有说出付款主语“我”。Index 草稿插入了 `I'll give you`；最终改为 `you'll receive`，保持付款人的不确定性，金额、银两和后续动作不变。

这两行重复金额，并从“我的”变为“他的”。角色模型仍把它们都分给同一人物，程序因此记录 `possible_relay_identity_collision`，将后一行角色和动作关系降为 unknown。程序拦截了这种混同，**没有据此证明前一句角色命名正确，也没有自动确定真正的转述者**。

以上“修正”是相对于当前识别文本的事实一致性修正。原音频仍存在分歧，不能把它写成对电影原台词已完成独立核验。

## 与历史 API V9/V13 的关系

历史交付保留。外部对白及前一场景确认，“腿”是黄四郎经营生意的比喻。当前候选使用 `have my leg reattached` / `get his leg reattached` 将它医疗化；此前认为这是改进的判断撤回。V9 的 `pull our legs` / `connect with him` 和 V13 的遗漏也有错误。窄人称规则只能保留识别文本的表面语法，不能证明完整剧情含义正确。

对照材料为 [V9 英文字幕](/Users/vanch/VideoLingo-plus/deliverables/e0c803_selected_v9_20260927/English_Subtitles.srt) 和 [V13 英文字幕](/Users/vanch/VideoLingo-plus/deliverables/e0c803_final_optimal_version_20260927/English_Subtitles_V13.srt)。

三个版本已按相同的 29 个语义单元评估，均为 **fail**。Index：8 正确、10 轻微、10 重大、1 核心剧情错误；V9：11/6/11/1；V13：6/10/12/1。Index 与 V9 互有得失，V13 的重大错误最多。成品质量已有明确结论；不同 ASR 与提示词条件不能用于隔离模型本身的优劣。

## 仍未解决的质量问题

1. 原片没有内嵌字幕。3–32 秒、117–150 秒、152–170.85 秒的扩大上下文复识别，仍与主转录和 Qwen 见证结果存在分歧。记录位于 `source_context_probes/`；它们是声学见证，不能充当正确答案。
2. 片尾“要人要人”与“了然了然”仍未可靠裁定；其他同音字、姓名和断句错误也可能影响译文。没有根据电影记忆替换原文。
3. 当前 Index 角色分析有不可靠命名和角色混同；unknown 与冲突保留，后续翻译不得把这些假设当作事实。独立角色正确率为 **unknown**。
4. 12 条显示字幕仍过短或过快，多发生在被切碎的对白。没有为了消除警告而删除对白、提高阅读速度阈值，或改动原声。
5. 同模型自审漏掉已观察到的人称问题，因此整体质量判断由来源、角色和显示报告共同决定。新配音、人物音色/情绪、完整同步及母语观感评价仍为 **pending**。

旧阶段报告和测试数量是当时的快照。本报告及相邻 JSON 为这轮原片字幕候选的当前证据；它们不是正式译配成品的验收证书。
