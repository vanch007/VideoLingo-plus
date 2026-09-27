# Nemotron Diarization Implementation Plan

> **Execution:** Follow the plan in the current authorized task, using available tools and focused skills as needed. Preserve separate approval gates for consequential external or destructive actions.

**Goal:** 将 Nemotron-3-Diarization-8bit 接入 VideoLingo-plus，提升说话人归属及配音音色一致性，并保留原生词级时间戳、MOSS 内容检查和可审计的恢复执行能力。

**Architecture:** 独立 MLX 运行环境和子进程提供说话人活动概率，主项目负责归属、证据、质量门槛及下游传递。stable-ts/WhisperX 继续提供文字和原生词时间戳；MOSS 保留独立文本覆盖检查与 TTS 回读。先影子评测，再在质量证据充分时切换默认。

**Tech Stack:** Python 3.10 主项目；专用 Python 3.12 MLX 环境；mlx-audio Nemotron；FFmpeg；现有 pytest、provider、pipeline、manifest 体系。

---

## 授权、执行模型与状态

- 用户在本任务最新消息明确要求：设置目标、形成详细执行方案、切换 Gemini 3.8 Flash 执行。此要求将前一轮“只分析、不编写代码”推进为实施授权。
- 指定执行模型：`google-antigravity/gemini-3.8-flash`。模型实际切换结果以应用工具返回为准，本文不是切换成功证据。
- 本任务已有 active goal；接续时读取已有目标，不能重复创建。
- 当前计划状态：ready for execution；代码实施、真实推理、评测和默认切换均为 pending。
- 不创建新用户任务、不派生子代理；在当前任务接续，除非用户另行要求。
- 不删除/覆盖无关用户产物，不清空 output，不撤销已有改动，不升级共享语音环境，不发布/推送，不发送外部消息。
- 模型下载、专用本地环境、必要代码编辑及隔离验证属于此次实施范围。计费外部调用、上传私有音视频、生产部署等不在本方案内。

## 已读取证据与实施前复核

1. `.ai_project.md`、`.ai_memory.md`、README、requirements.txt、config.yaml。
2. `core/providers/speaker_diarization.py`：后端硬编码 MOSS；按时间重叠归属；未知分组可变为 SPEAKER_00；缓存只检查标签覆盖率。
3. `core/providers/source_coverage.py`：没有参考文本的片段被跳过，零个有效片段可能报告 100% 覆盖；不得直接用无文本 Nemotron turns 替代。
4. `core/step2_whisperX.py`：ASR 合并后、表格导出前已有说话人接入点。
5. `core/step6_generate_final_timeline.py`：多数票、相邻人物及 S01 回退；cross_speaker_rows 字段写死为零。
6. `core/step9_extract_refer_audio.py`：参考时间窗前后扩展，未依据声学人物边界。
7. `core/all_tts_functions/tts_utils.py`：可回退任意人物参考；speaker_anchor 基于时长筛选。
8. `core/step10_gen_audio.py`：生成指纹未包含说话人和参考身份。
9. 实际配置与文档有偏差：当前说话人 enabled=false、required=false、覆盖门槛 0.80，translation_context.require_speakers=false；不可当作当前严格模式已生效。
10. 已检查硬件 M3 Max/128 GiB。现有 bluestone-mlx-audio 环境 mlx-audio=0.4.2，无 Nemotron 实现；本地 /Users/vanch/mlx-audio 也缺少该模块。
11. 查询时上游 mlx-audio=0.5.6，commit `4ab7e6f7dedd69a136cfaa318c5dc8aed5119446`。其 transformers>=5.14.0、huggingface_hub>=1.0 与主项目上界冲突。
12. 8bit 模型 revision `dd8b8ce3d69a80540a7e50150e2beed1c34fee37`；配置记录的 NVIDIA source revision `723e19c601d99b7e58fba6a14e32153e0afe48d9`。实施时复核可获取性并锁定实际版本。

主资料：
- https://huggingface.co/mlx-community/Nemotron-3-Diarization-8bit
- https://huggingface.co/nvidia/Nemotron-3-Diarization
- https://github.com/Blaizzy/mlx-audio/blob/4ab7e6f7dedd69a136cfaa318c5dc8aed5119446/mlx_audio/vad/models/nemotron_diarization/README.md
- https://github.com/Blaizzy/mlx-audio/blob/4ab7e6f7dedd69a136cfaa318c5dc8aed5119446/pyproject.toml

## 固定设计边界

- ASR 文字/词时间戳与 speaker overlay 分别缓存；仅补识别流程可新增或修复经过验证的词。
- 8bit 是首选实施候选，不预先宣称比 MOSS/BF16 更准确。
- 16 kHz 单声道 PCM，共用源视频零点。记录转码、裁切、重采样与 offset；不删除静音后拼接时轴。
- 默认评测 preset=offline：chunk_len=340、right_context=40、fifo_len=40、spkcache_len=264、update_period=300，单位 80 ms；输出 10 ms。
- threshold=0.5、min_duration=0、merge_gap=0 为上游基线，不代表项目最优值；在独立调参集调优。
- 同一视频保持一个连续说话人状态，不能按 ASR 分块重置编号。每个新视频必须重置状态。
- 片内 speaker index 映射为 S01 等，并保留原始 index 和映射；未知不能变成一个假人物。
- 多通道活动概率不互斥；不能仅 argmax 抹除重叠，不能当作已校准身份准确率。
- Nemotron 不转录文字、不分离重叠音源、不识别人名、不提供跨视频人物身份。
- 第一阶段保留 MOSS 全量文本覆盖见证；减少 MOSS 调用必须有独立非退化证据。
- 当前句纯净参考优先，同人物纯净候选兜底；不默认全量固定 anchor，不允许严格模式借用其他人声音。
- 无音频的纯 SRT 不能声学分人；有音频分支需真实词级对齐，不能均分字幕伪造词时间戳。
- 已知超过八人的素材不承诺完整区分；仅输出八通道无法可靠检测第九人。

## Task 0：保全工作区与建立基线

**Files:** 读取现有 AGENTS.md（若存在）、项目文档和 git diff；新建 `docs/reports/nemotron-diarization/` 的基线记录，重型数据置于独立本地 run 目录，不提交音视频/权重。

1. 读取当前项目规则和本计划，确认当前目标、模型及用户授权。
2. 检查 git status/diff；用户原有修改包含 config、speaker_diarization、stable_ts_local、profiles、mlx_tts、step4/11/12、memory、测试和未跟踪音视频，不能回滚或整体替换。
3. 创建本任务专用验证目录；需要隔离 checkout 时，确保带上任务所依赖的现行改动，不能只用 HEAD 而丢失当前实现。
4. 识别可用测试解释器，执行相关现有测试，记录命令、环境、结果与历史失败。建议入口：`conda run -n videolingo python -m pytest tests/test_speaker_diarization.py tests/test_source_coverage.py -q`，先确认 conda 环境实际存在。
5. 特别核对 unknown 分组实现与现有测试的矛盾；失败必须区分已有问题和新引入问题。

**验收:** 有基线及原有变更范围；正式 output 未被验证流程覆盖。

## Task 1：隔离运行时和模型能力验证

**Files:** 新建 `core/providers/nemotron_diarization_runner.py`、专用依赖锁定/安装说明；修改 `core/doctor.py` 的后端选择检查。文件名为计划新增项，落地前确认无同名实现。

1. 在专用本地 Python 3.12 环境安装固定 mlx-audio 版本/commit，不修改现有 bluestone 或主项目依赖。
2. 获取固定 8bit revision，记录模型校验信息、运行库/MLX版本和来源。
3. 用 strict=True 加载明确带 -8bit 的模型 ID。使用 mlx_audio.vad，不能照搬 NeMo 加载器；result.text 为 RTTM。
4. 在专用测试目录验证：短音频、静音、尾部不足一个窗口、长音频跨块、重复调用的新会话隔离。
5. 记录冷启动/热运行时间与峰值内存；长音频避免把所有 PCM、概率和中间结果无限常驻。
6. 实现健康检查及结构化失败，不把路径存在当作模型加载成功，不执行健康检查时自动改环境。

**验收:** 真实加载和推理有证据；绝对时间、有限概率、八通道、尾部 flush 正确；依赖不污染主环境。

## Task 2：统一后端契约与影子接入

**Files:** 修改 `core/providers/speaker_diarization.py`、`core/providers/contracts.py`（适用时）、`core/step2_whisperX.py`；新增 `core/providers/nemotron_diarization.py`；新增/扩展相关 tests。

1. 定义后端中立结果：schema_version、audio_fingerprint、sample_rate、duration、time_origin、model/revision/runtime、turns、frame_stride、概率数据路径、speaker_map、metrics、status/error。
2. turn 至少记录 start/end、原始 speaker index、映射 ID；词级附加 candidates、assignment_status、overlap、score、source，文本可选且不能伪造。
3. 保留 MOSS 适配器，新增显式 nemotron-mlx 路由；避免 UI 先硬编码分支。
4. 使用独立子进程交换路径/JSON和概率文件，校验返回值、超时、退出码及结果所属输入。
5. 增加 shadow 模式，仅写独立报告，不改正式 word speaker 或默认配置。
6. 测试空输出、损坏结果、错误 revision、进程失败和复用旧输入结果的拒绝行为。

**验收:** 两后端遵循契约；旧 MOSS 路线可用；shadow 不影响正式产物。

## Task 3：词级归属、未知与重叠

**Files:** `core/providers/speaker_diarization.py`、`core/asr_schema.py`、`core/all_whisper_methods/audio_preprocess.py`、`tests/test_speaker_diarization.py`；新增 sidecar schema/测试按需要复用现有框架。

1. 为词定义稳定 ID，用区间内概率均值、活跃覆盖和候选差距决定 assigned/unknown/ambiguous/overlap。
2. 将边界容差与活动阈值分开；避免 0.35 秒无条件借标签。零时长词明确处理，不除零。
3. 保存原始帧级证据与后处理结果，多人活跃时保留多候选；不要跨真实换人边界平滑。
4. 不因 unknown 自动丢弃“嗯/啊”等真实短回应；修正质量统计分母，显示未分配及丢弃情况。
5. 人工修正绑定音频指纹、词 ID、时间和原始证据，记录修正来源；不能误用于另一视频。
6. Excel 与内部 schema 不得吞掉新状态；以 sidecar + 稳定 ID 关联，保持旧消费者兼容。
7. 测试：短回应、相同时间多人、间隙词、短暂离场、换人边界、未知连续词及人工修正；断言未补识别时原词文字和时间完全不变。

**验收:** 无虚假人物、无强制重叠独占；已有词时间轴不变；覆盖率不冒充准确率。

## Task 4：分离声学覆盖和文本覆盖检查

**Files:** `core/providers/source_coverage.py`、`core/providers/speaker_diarization.py`、`tests/test_source_coverage.py`。

1. 声学检查独立使用 Nemotron speech activity 与原 ASR 词区间的时间并集，避免重叠重复计时。
2. MOSS 保留文本见证；无文本结果不能送入旧文本检查并得到零样本 pass。
3. 状态明确区分 pass/fail/pending/not_checked/not_applicable；静音与有语音但无有效证据分开。
4. 可疑范围仍调用原 ASR transcribe_range；记录 padding、绝对 offset、去重与补识别来源。
5. 防止重新赋值 alignment report 时覆盖之前的 source_coverage 报告。
6. 测试无文本 Nemotron、真实静音、漏词、异常长词、补识别合并和回调失败。

**验收:** 原内容完整性检查未退化，不再零有效样本虚假通过。

## Task 5：断句、翻译和时间轴贯通

**Files:** `core/spacy_utils/split_by_mark.py`、`core/translation_context.py`、`core/step4_1_summarize.py`、`core/step6_generate_final_timeline.py`、`core/step8_1_gen_audio_task.py`、`core/step8_2_gen_dub_chunks.py`；定位对应现有测试再扩展。

1. unknown/ambiguous/overlap 与确定换人形成明确边界，不合并为假 SPEAKER_00。
2. 保留 LineID/词 ID/人物 ID 的传递关系；翻译角色推断不能覆盖声学身份。
3. 移除严格模式下多数票掩盖混合人物及无证据 S01 回退；混合句拆分或标记复核。
4. cross_speaker_rows 从实际词归属重新计算；不能写死零。
5. 严格模式只在信息不足的依赖阶段阻断，不破坏允许无说话人的纯字幕模式。
6. SRT/内嵌字幕输入：有音频可进入真实声学及对齐流程；无音频明确不支持声学识别，不伪造结果。
7. 测试说话人从 Step2 经断句/翻译到 TTS task 一致，未知不被悄悄填补。

**验收:** 一条可执行配音任务只含一个确定人物；不确定区间有显式状态和处理入口。

## Task 6：参考音频纯净度与同人物约束

**Files:** `core/step9_extract_refer_audio.py`、`core/all_tts_functions/tts_utils.py`、对应 MLX TTS 调用适配器及测试。

1. 当前句参考优先；裁剪时间必须满足人物活动区间，并排除其他人物/重叠/无效音频。
2. 当前句不合格时，只从同人物候选选择，依据时长、纯净度和质量；不要默认全量 speaker_anchor。
3. 参考音频和参考文本配对更新，记录选取理由和来源。
4. 严格多人模式禁止任意人物 fallback；没有纯净参考则标记 manual_review。
5. 保持 TTS 后端的实际输入时长要求，不能用过短参考假通过。
6. 测试窗口扩展跨人、短句、所有候选混音、同人回退和参考文本关联。

**验收:** 跨人物参考回退为零，未满足纯净度的参考不进入自动克隆。

## Task 7：缓存与恢复执行溯源

**Files:** `core/pipeline/artifact_manifest.py`、`core/pipeline/runner.py`、`core/step_checker.py`、`core/providers/speaker_diarization.py`、`core/step10_gen_audio.py`；按现有翻译 provenance 接口扩展。

1. 分离 ASR、diarization、alignment 指纹，后端切换可复用可信原生 ASR。
2. 指纹包含音频/时轴、模型/revision/runtime、推理/后处理配置、身份映射、人工修正和 schema。
3. 变更归属时使断句、角色翻译、时间轴、参考及 TTS 依赖失效；不要只检查文件存在或 coverage。
4. TTS 指纹纳入人物、参考音频内容/版本、参考文本、实际后端模型。
5. 旧结果标记 stale 并给出恢复计划；不得擅自删除或移动已有正式产物。
6. 测试后端变更、阈值变更、参考变更、同名音频内容变更和人工修正后的选择性重算。

**验收:** 无旧人物缓存泄漏；相关步骤失效而无关 ASR 可复用。

## Task 8：真实评测、默认路由与文档

**Files:** `core/providers/benchmarking.py`（可复用时）、`tests/fixtures/` 索引、`docs/reports/nemotron-diarization/`、`config.yaml`、`core/pipeline/profiles.py`、`.ai_project.md`、`.ai_memory.md`、README及相关 CLI/UI 设置。

1. 盘点已有可用音视频与人工标签，禁止把旧模型输出当成 ground truth。缺标签明确 missing evidence，并完成独立可验证的推理/契约回归。
2. 调参集与验收集分开；同素材比较 MOSS、MLX 8bit 和 BF16，固定 ASR 输入及评分协议，DER 标签允许最优人物置换匹配。
3. 覆盖中文快速接话、相似音色、短回应、音乐、重叠、长视频人物重返、其他实际源语言及 >8 人边界。
4. 记录 DER 的 miss/FA/confusion、人物数误差、词归属准确率与 unknown/ambiguity、换人边界、参考纯净度、处理速度/内存及 TTS 内容回读。
5. 98% 仅为建议的 assigned coverage 门槛，不能通过强制赋值达成，也不是识别准确率；需以真实标注考察错分。
6. 默认切换条件：契约/回归全通过；人物混淆和词归属较 MOSS 不退化且有明确改善证据；重要样本无严重退化；不变时间戳、零跨人物参考、实际边界报告、缓存验证均通过；TTS 内容回读不退步。
7. 8bit 若显著劣于 BF16，记录结果并评估选择；不得为了满足预期宣称 8bit 最优。证据不足则保持 opt-in/shadow，不擅自修改默认。
8. 最小真实端到端在独立工作区执行，保留源视频/报告/新产物。尽量复用已有可信翻译避免新计费外部调用；若必须新增计费服务先展示具体需求并遵守授权边界。
9. 同步配置说明、CLI/UI、doctor、模型列表及文档；不把当前用户临时配置整体替换成文档默认。
10. `.ai_memory.md` 仅记录有复用价值的结果，单次不超过五行。最终交付修改清单、可运行入口、验证报告、真实限制和回退方式。

**验收:** 区分工程接入完成与质量晋级；默认切换若 pending，要列出具体缺失数据/结果，不能声称整个目标已完成。

## 执行纪律与完成定义

- 每个任务按“读现状→补必要行为测试→最小实现→相关测试→记录证据”推进；不生成镜像实现的无意义测试。
- 新失败继续修复；历史失败与基线对照。没有执行的检查一律 pending。
- 状态和评测记录写入本任务独立报告，保留命令、输入指纹、版本、结果及已知限制。
- 不批量 git add，不提交用户无关改动，不强推。不为已授权的常规实现反复询问。
- 模型下载或测试启动不等于完成；跟进进程到结果，保存可恢复位置。
- 活跃 goal 仅在所有必需交付和验证闭环时标记 complete；遇到缺标签等阻塞先继续其他独立工作，按当前 goal 工具规则处理阻塞。

## 进度表

| 阶段 | 状态 | 证据 |
|---|---|---|
| 目标与详细计划 | pass | 本文件；当前任务 active goal |
| Gemini 模型切换 | pass | 当前会话已切为 gemini-3.8-flash 继续执行 |
| Task 0 基线 | pass | 已修复现有 group_word_rows_by_speaker 未知合并缺陷，基线 12 项测试全通 |
| Task 1 独立环境/模型 | pass | 独立环境 /Users/vanch/.codex/envs/nemotron-diarization (Python 3.12, mlx-audio 0.5.6)；已验证 8bit 模型加载推理 (58x RTFx) |
| Task 2 后端/影子 | pass | core/providers/nemotron_diarization.py + runner；支持 nemotron-mlx 与 shadow_backend |
| Task 3 词归属 | pass | align_speakers_to_words 支持概率张量提取、置信度阈值、多候选与未知标记，已增单测 |
| Task 4 覆盖检查 | pass | source_coverage.py 分离声学与文本覆盖分析，消除无文本误判 100% pass 漏洞，已增单测 |
| Task 5 下游传播 | pass | step6 时间轴禁止跨角色合并与盲猜 S01，修复测试并如实计算 cross_speaker_rows |
| Task 6 参考音频 | pass | step9 限制跨角色扩展，tts_utils 增加严格模式禁止借用他人音色，已增单测 |
| Task 7 缓存 | pass | step10 指纹纳入 speaker 与 ref_audio，避免声学修正后复用错误语音缓存 |
| Task 8 评测与默认晋级 | ready for shadow eval | 后端架构完整就绪，默认仍为 moss-mlx，影子评估器可即时比对实际素材 |
