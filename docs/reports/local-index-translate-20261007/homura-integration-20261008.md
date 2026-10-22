# Homura 接入与验收记录（2026-10-08）

## 结果

**pass：接入和整片自动内容、时长、音频完整性验收。** 本地 `Index-Homura-9B-8bit` 已配置为英语时长候选，主翻译与源文语义审核继续使用 `Index-Translate-35B-A3B-preview-oQ6e-fp16-mtp`。本次已生成新的 170.852 秒完整配音视频。

**pass：当前双语配对与同步复跑。** V4 的独立显示切分绕过了双语对应步骤，其空间检查不代表语义同步。最新字幕流程结果为 `run_workspaces/homura_bilingual_aligned_v6_20261008/output/AI配音.mp4`：41 个单元、82 对；最终解码音轨与 V4 完全相同。详见 [双语字幕对齐记录](bilingual-alignment-20261008.md)。

**needs_review：字幕阅读速度。** 最新 V6 有 10 条 CPS 提示；V4 历史记录为 74 条显示字幕、6 处提示。未通过删除意思、改写已锁定配音或移动对白消除这些提示。

**missing evidence：独立人工听感。** 当前声纹和情绪评分仅是声学代理指标；不把同模型语义审核和 ASR 回读称作独立人工质量认证。

## Homura 翻译与配音整片验收（V4，字幕同步修正前）

历史工作区：`run_workspaces/homura_full_workflow_v4_20261008`，已于 2026-10-09 按用户要求清理。汇总与运行记录保留在当前 V6 的 `audit/source_workflow/delivery_validation.json`、`integration_result.json`；源 ASR、候选、审核及锁定语音已迁入 V6。当前视频使用上文 V6 结果。本报告其他旧工作区路径仅记录历史实验位置。

验证从已审核的原始视频和源文 checkpoint 开始，未导入旧译文、旧合成语音或旧接受状态；后续恢复仅继续本验证工作区的候选。ASR 和说话人证据本次复用，未重新识别。来源、资产哈希与不可变时钟保存在 `source_lineage.json`。

| 检查 | 结果与证据 |
|---|---|
| 源文逐句语义审核 | **pass**：41/41，`output/log/translation_review.json`，按当前原句独立比较，零问题 |
| 翻译阶段完整自然语音预验收 | **pass**：41/41，`output/log/translation_speech_preflight.json`，最高 1.08×，零问题 |
| 后续配音复用 | **pass**：41 段锁定语音直接复用，无后置重译、缺失音频或静音替代 |
| 配音内容、时间和完整性 | **pass**：41 ok / 0 warn / 0 fail，回读平均 0.99335，源语言泄漏 0 |
| 一分钟之后 | **pass**：28 段配音均存在且回读合格；最终 AAC 音轨与混音母带逐窗口相关系数最低 0.9996957 |
| 原始时钟 | **pass**：源文/词时钟签名未改变；最终视频流时长差 0，音频流时长差约 0.000426 秒 |
| 字幕空间和重叠 | **pass**：74 条显示字幕均在行长限制内，无时间窗口重叠 |
| 双语片段对应 | **fail**：V4 独立切分导致内容串位；由上述 V6 流程修正替代 |
| 字幕阅读速度 | **needs_review**：6 条警告，见 `output/log/dub_subtitle_layout.json` |

本次留下 60 条 Homura 候选历史，其中 8 个最终单元的锁定文案与通过实测的 Homura 文案完全一致；其余由主译稿或受源文约束的 Index 重译提供。拒绝的候选保留原始原因，不把模型建议直接视作已接受配音。

V4 阶段回归：**pass**，全项目 575 项，145.18 秒；整片合成入口相关 32 项通过；`compileall` 和 `git diff --check` 通过。日志保留于当前 V6 的 `audit/source_workflow/pytest_release_rendered.txt`、`pytest_render_entry.txt`、`full_workflow_v4_render_entry.log`。后续双语修正完整回归为 598 项，见双语对齐记录。

## 本次流程修正

- 源文事实重译保留对应语音预算；地点/方向、否定、金额与所有者冲突获得明确的源文反馈。
- 最终审核逐句绑定原文、当前文案和上下文，避免批量自审把陈旧 ASR 备选或角色假设当作当前事实。
- 对逗号后嵌入的重复分句计数；紧迫的名词式反问保持紧凑形式，并逐次保留重复。当前案例的三次反问恢复后，自然语音无需加速即可放入原窗口。
- 对当前源文术语中的短复合词提供字典发音标注，完整保留字幕/配音的规范文案；数字回读冲突可触发独立的第二识别器。
- 发现并修复整片合成入口缺少 `pandas` 导入的问题；新增锁定任务和预验收必须先于渲染执行的回归用例。

## 接入方式

1. Index 生成完整主译稿；源文事实审核和角色、术语背景在翻译阶段处理。
2. 当前源句中明确出现的术语按原句检查，避免时长压缩把具体类别泛化。可选的模型类别提取默认关闭，不能替代源文语义证据。
3. 翻译阶段预合成自然语音并回读。只有文本、自然音频、声音条件和完整回读绑定一致的超时样本，才用于校准 Homura 的正整数音节目标。
4. Homura 直接翻译原句，结合相关术语与上下文。角色推断仅辅助理解；邻句和未核实角色不能增加当前句的事实。
5. 候选先通过源文语义审核，再合成、回读和测量。语义失败可提供源文短语映射，在同一音节目标下有限重试。
6. 仅完整内容且实际时长符合最多 1.08× 语速的语音可锁定。较短的完整语音保留自然语速，以静音填充窗口。
7. 漏读、错读和回读不可用时，保留原译文重生成；这种语音的时长不能触发压缩翻译。
8. 最终语义审核必须为 `pass` 才能导出。后续配音复用锁定文案和音频，字幕按目标语音词对齐进行显示分段。

通用模型设置中不提供 Homura 作为默认通用模型。它的原生接口是源文与整数音节约束，不用于 JSON 场景分析或通用审核。非英语目标和没有安全候选的情况继续走主翻译器。

## 本地配置

- 路由：`translation.speech_budget.candidate_provider = omlx_homura`。
- 服务：`http://127.0.0.1:8000/v1`。
- 模型：`/Users/vanch/.cache/lm-studio/models/IndexTeam/Index-Homura-9B-8bit`。
- 源版本：`3f269ba26d87015c1a537ad42df6d586d3ab03dc`；affine 8bit、group size 64。
- 权重、配置和 tokenizer 的本地内容哈希进入模型契约；相同模型名称下更换文件会使请求缓存、翻译清单与语音接受状态失效。
- 候选与后续语音验收记录：`output/log/translation_duration_candidate_provenance.json`。

模型转换与下载证据另见：`/Users/vanch/Documents/Index-Homura_Research_20261008/README.md`。

## 早期电影台词实验（历史记录）

验证使用原工作区 `run_workspaces/index_quality_v30_turn_budget_workflow_20261008` 的源句、原时间窗口、原 Index 译稿与本地声音配置，没有整体替换历史交付物。

源句 `L00008.01`：

> 杀县长夫人绑架豪绅，祸害鹅城百姓，就是你黄老爷家的胡万。

| 项目 | Index 原稿 | Homura 候选 |
|---|---|---|
| 自然语音时长 | 10.82 秒 | 首次 8.58 秒；同文重试 8.858 秒 |
| 原时间窗口 | 9.15 秒 | 9.15 秒 |
| 放入窗口所需倍率 | 1.1825×，超过上限 | 两次均无需加速 |
| 源文语义审核 | 通过 | 通过，保留 `local gentry` 等类别 |
| 内容回读 | 用于完整超时校准 | 未通过，末尾人名未可靠识别 |
| 最终接受 | 不接受超时语音 | 不接受回读存疑语音 |

Homura 候选：

> The one who killed the County Magistrate's wife, kidnapped the local gentry, and harmed the people of Goose Town is your man, Hu Wan.

此结果证明模型可改善实际时长；不能由此证明这句配音已合格或整片效果更好。高文本相似度不再跳过句首、句尾完整性检查；边界异常可触发第二个识别模型。对现有音频的交叉回读仍未给出可靠的末尾人名证据，继续拒绝接受。

可查证据：

- `run_workspaces/homura_integrated_source_terms_20261008/integration_result.json`
- `run_workspaces/homura_integrated_source_terms_20261008/output/log/translation_speech_preflight.json`
- `run_workspaces/homura_integrated_source_terms_20261008/output/log/translation_duration_candidate_provenance.json`
- `run_workspaces/homura_readback_policy_verified_20261008/result.json`

新增验证选取 `L00017.01`（短句）、`L00032.01`（金额、单位及隐喻）、`L00033.01`（情绪和承诺）。结果记录在 `run_workspaces/homura_acceptance_suite_20261008/integration_result.json`。本地服务返回 507：Index-Translate 加载需求 28.74 GiB，运行时动态门限约 26.8–26.9 GiB。诊断时 oMLX 无已加载模型、无活动推理请求；未修改服务内存门限或关闭其他任务。

## 早期回归与局限（历史记录）

- **pass**：相关路由、回读、原子配音测试 104 项通过。
- **pass**：全项目 `pytest -q --tb=short`，511 项通过，129.50 秒。
- **pass**：`compileall` 检查 `core` 与验证脚本；`git diff --check`。
- **pass**：现有真实样本中，Homura 调用、语义审核、实际合成与时长测量均有证据。
- **fail**：上述人名句的最终内容回读；不能用时长合格豁免。
- **fail**：新增三组真实案例因内存门限未运行到底。
- **missing evidence**：Homura 参与后整片视频的完整验收及独立人工听感评估。

模型自审和语音识别均可能误判。音节数量用于有限候选搜索，实际完整语音才决定时长是否合格；这份记录不把模型自审等同于人工质量认证。
