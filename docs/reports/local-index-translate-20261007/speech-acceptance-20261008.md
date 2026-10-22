# 1.08 倍边界与翻译阶段配音验收 — 2026-10-08

结论：1.08 倍属于接受范围，包含上限。边界修复与代码回归 pass；整段配音验收 fail，52 个单元中 44 个通过时长和回读检查，8 个仍超时。没有生成本轮最终视频。

## 修复与实际边界验证

- 自然配音时长 / 源时间窗 ≤ 1.08 时允许轻微加速。完整短句保留自然语速并补静音；不能裁切实质尾音。
- 旧拟合程序加入 15ms 余量后，若计算倍率超过 1.08 就直接拒绝。现在先在合法上限重新拟合完整音频，仍放不下才拒绝。
- 真实诊断 L00022.01：自然音频 3.092s，时间窗 2.880s，理论倍率 1.073611；旧 atempo 输出 2.890s 导致余量推到约 1.083 倍并误拒绝。完整音频按 1.08 倍拟合后为 2.866s，再补 14ms 静音，时长 pass。该诊断只证明时长，当时该条朗读内容仍有问题。
- 单词数改为参考预算，提示词给出包含 1.08 倍余量的自然配音时长。实测超时却原样返回的候选须尝试更紧凑的完整表达；实测正好 1.08 倍时不要求改写。源文事实、语气和重复不得为凑时长而删除。
- 输入和源时间轴相同的候选文本可在拟合政策变化后继续重测；已接受的音频仍绑定具体合成/拟合配置，变更后必须重新验收。

## 验证

- 完整 pytest：438 passed in 61.02s；相关翻译、回读、检查点和边界测试：109 passed in 9.44s。
- 本轮 52 个音频哈希均匹配；44 个拟合音频最大速度 1.08 倍。52 个单元均完成回读，最新失败原因全部是实际时长超限。语义审查和最终视频阶段未执行，不能声称整段译文语义或最终观感通过。
- 已保存的外部电影台词 12 个检查点：当前源文 160 个参照字符 0 错误，关键事实 0 失败。仅评价选定文本检查点，不能解释为原始 ASR 全片错误率或时间戳精度。
- doctor_current.log 综合检查 fail，包含已有 NumPy/OpenCV 依赖不匹配及未使用的 TTS 后端不可用。本次实际执行了指定 IndexTTS2 路线，未扩大到整个环境的修复。

## 未通过的单元

| 单元 | 源文 | 当前候选 | 实测所需倍率 |
| --- | --- | --- | --- |
| L00004.02 | 四筒啊！ | Four-dot mask! | 1.397 |
| L00006.01 | 马匪呢？ | Where are the bandits? | 1.798 |
| L00012.01 | 早晚！ | Eventually! | 1.149 |
| L00020.01 | 不用翻译，就是惊喜。 | No need to explain. It's a surprise. | 1.711 |
| L00023.01 | 惊喜嘛。 | A surprise. | 1.530 |
| L00028.02 | 给你们出城剿匪，接上我的腿，明白了吗？ | I'll send you out to wipe out the bandits. That'll get my business back on its feet. Understand? | 1.141 |
| L00031.01 | 翻译翻译。 | Spell it out. | 1.386 |
| L00038.01 | 这个，你这个。 | This... you, here. | 1.275 |

这些是真实的 >1.08 时长冲突。流水线在翻译验收阶段停止，保留候选及音频；未绕过门禁进入后期强制变速或再次翻译。

## 证据

`run_workspaces/index_quality_v26_translation_speech_acceptance_20261008/`：

- `speed_boundary_diagnostic/result.json`：真实音频边界复核。
- `output/log/translation_speech_preflight.json`：最新 44 / 8 结果、回读及音频哈希。
- `output/log/translation_candidate.xlsx`、`output/audio/translation_speech_tasks.xlsx`：未接受的整段候选。
- `before_boundary_fix/`、`before_progress_guard/`：先前失败报告保留。词数提示单独软化曾使压缩不收敛；加入实测无进展检查后失败数 18 → 8。
- `code_snapshot_progress_guard.json`、`code_regression_progress_guard.log`：代码版本及回归。
- `source_assessment.json`：12 个源文检查点。
- `resume_progress_run.log`、`output/pipeline_state.json`：实际 fail，失败阶段 translate。
