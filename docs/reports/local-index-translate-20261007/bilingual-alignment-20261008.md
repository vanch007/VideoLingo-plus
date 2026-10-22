# 双语字幕对齐流程修正（2026-10-08）

**pass：本案例的双语片段配对、共享显示时钟与导出完整性。** 新完整配音视频：`run_workspaces/homura_bilingual_aligned_v6_20261008/output/AI配音.mp4`，170.852 秒。

**needs_review：10 条字幕阅读速度提示。** 82 组字幕全部满足中文每行 22 字、英文每行 38 字、双语两行的限制，无时间窗口重叠。为保留已验收配音，不通过重译、删除内容或移动对白消除 CPS 提示。

## 根因

原项目 Step 5 有中英片段对应机制，但新的 `translation.utterance_plan` 分支跳过了它。`core/plan_subtitles.py` 独立切分两种语言，再按字数平衡片段数，或重复整个中文句子。目标语词时间只能使字幕落在声音窗口内，不能保证同屏两种语言表达同一片段。

V4 的“空间长度和无重叠 pass”不包含语义配对。V5 初次自动配对仍把“亲爹”与 `County Magistrate` 拆开，并出现动作跨片段；其自动审核 pass 不能作为这次最终交付证据。V6 才加入完整词语、数量和动作短语的程序约束，收紧逐对审核并完成复跑。

## 现行流程

1. 翻译阶段验收的原句、译句、语音和对白窗口继续锁定。
2. 对目标配音重新进行词对齐；对齐输入给破折号等连接标点补空白，防止 `people—it` 被当成 `peopleit`，不改变规范字幕文本或音频。
3. 显示阶段切分对应的原文/译文。中文使用已安装的 `zh_core_web_md` 词语切点；目标语保留完整单词、专名、数值与单位及离开地点短语。
4. 模型直接选择切点失败时，Index 对目标片段生成内部原语言参照，程序按词语对应选择连续原文切点。参照不进入最终字幕，不成为新配音文案；仅字串相似度不能证明语义正确。
5. 每一对重新核对局部意思，禁止用整句相符代替逐段相符。结构、覆盖或审核失败时停止，没有比例切中文或重复整句的保底分支。
6. 双语字幕共用一个显示时间窗口。配音视频使用实际目标语词时钟；保留原音的视频使用源语言词时钟。
7. Step 7/12 在导出前检查配对记录、当前文字、词时钟和 SRT 文件哈希。流水线清单保留相应依赖，字幕代码改变只影响字幕及合成阶段。

传统 `align_subs` 允许重写目标片段，不能直接用于已锁定语音。本次恢复的是对应功能，并保留不可修改配音的约束。

## 案例核对

| 原文片段 | 同屏译文片段 |
|---|---|
| 杀县长夫人 | Killing the county magistrate's wife, |
| 绑架豪绅， | kidnapping the gentry, |
| 就算是我亲爹 | Even if he were my own father, |
| 也得死， | he'd still have to die. |
| 给你们出城剿匪， | to leave town, wipe out the bandits, |
| 这他妈是惊喜啊， | this is a fucking surprise! |
| 小弟我愿意等你三天。 | I'll wait three days for you. |

实际烧录画面抽查：24.2、54.6、127.3、148.3 秒，记录在工作区 `audit/`。逐组核对所有 82 对的规范文字和 SRT 时间；抽查画面确认没有旧长行或两种语言串位。

## 验证证据

| 检查 | 结果 |
|---|---|
| 锁定台词和源文时钟 | **pass**：未改变 |
| 双语片段 | **pass**：41 个配音单元、82 对，精确顺序覆盖，无增删重复 |
| 原文/译文/双语 SRT | **pass**：逐条编号、起止时间一致，同屏内容匹配配对记录 |
| 显示空间及重叠 | **pass**：82/82 行长合规、0 重叠 |
| 两种视频的时钟选择 | **pass**：原音使用源词、配音使用目标词 |
| 下游清单 | **pass**：split_subtitle / timeline / merge_subtitle / merge_audio / merge_video 五份清单均为当前版本 |
| 配音和最终音轨 | **pass**：41 段接受音频未变；混音母带与 V4 哈希相同；最终 AAC 解码后逐样本相同 |
| 视频/音频流起点及时长 | **pass**：保持规范源视频各流，容器 170.852 秒 |
| 全项目回归 | **pass**：598 项，155.47 秒；`compileall`、`git diff --check` 通过 |
| 字幕阅读速度 | **needs_review**：10 条 CPS 提示，与同步验收分别记录 |
| 独立人工声纹/情绪及全片听感 | **missing evidence**：本次字幕修改没有重新认证这些项目 |

最终解码 PCM SHA256：`450dddcda556308efad25d4263f6d86930ed91391831ed97e781c9fb31049bf2`，与 V4 相同。

主要证据：`bilingual_delivery_validation.json`、`bilingual_result.json`、`output/log/bilingual_alignment.json`、`output/log/dub_subtitle_layout.json`、`output/audio/dub_display_timeline.xlsx`、`audit/pytest_release.txt`。

本次只重建字幕、混音与成片，复用了 V4 已验收的原片、源文、翻译、声音条件和自然语音；没有新 ASR、重译或重新生成配音。模型对局部语义的审核仍可能误判，不能由本案例推导所有语言或素材已获得独立质量认证。

复现入口：

```bash
python scripts/verify_bilingual_video.py \
  --source-workspace run_workspaces/homura_bilingual_aligned_v6_20261008 \
  --output <new-workspace> --dry-run
```

检查后移除 `--dry-run`，使用同一项目 `videolingo` 环境执行。源工作区不变；字幕策略更改后已生成的旧显示检查点需要新工作区，不能假装当前结果继续导出。

## 正式流程与工作区保留（2026-10-09）

上述约束已进入正常 Step 5/6/7/11/12 及共享流水线的依赖检查，默认配置与 V6 验收配置一致。新素材使用 [现行本地工作流](../../localization-workflow.md) 的 CLI/Streamlit 入口，不依赖本案例验证脚本。

用户授权清理 `run_workspaces`，仅保留 V6。V4 的 ASR、Index/Homura 候选与审核证据补入 V6；原源流程汇总保存在 `audit/source_workflow/`。运行清单与恢复状态的输入引用按相同原片哈希迁至 V6，原始清单另存 `audit/relocation/`，不改动视频、字幕或语音。原 V4 路径在历史验收记录中仍保留为来源说明，不作为运行依赖。

本次定向回归 **pass**：115 项，88.24 秒；清理及独立复查结果见 V6 的 `retention_manifest.json` 和 `audit/finalization_validation.json`。此前完整回归、阅读速度提示与人工观感证据边界保持上文记录。
