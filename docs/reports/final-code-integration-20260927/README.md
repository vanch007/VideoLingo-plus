# 最终代码落盘记录

日期：2026-09-27。范围：将现有 V13 实现收尾到项目主目录，保留已选成片，不开启新的效果优化或生成轮次。

## 代码位置与本次改动

V13 的 `run_pipeline.py` 将本项目根目录加入 `PYTHONPATH`，实际调用的是本项目的 `core/`。最终实现此前已直接落在主目录，无需用历史工作区副本覆盖代码。

本次完成：

- 将 `config.yaml` 的 `translation_context.require_speakers` 从 `false` 对齐为 `true`。解析后的整个配置与 V13 工作区配置一致。
- 将 V10、V11、V12 的 17 项契约回归纳入正式 `tests/`，分别为 `test_media_delivery_contract.py`、`test_perceptual_identity_contract.py`、`test_fidelity_evidence_contract.py`。原始审核文件保留。
- 添加 `pytest.ini`，默认仅收集 `tests/`，避免历史工作区测试被重复执行。
- 更新 README 与项目记录中的默认后端、混合参考策略、测试入口及验收边界。
- 在 `deliverables/code_final_v13_20260927/` 保存本次项目代码快照与逐文件 SHA256 清单。这是源码检查点，不包含外部模型环境，也不是一次 Git 提交。

## 已在项目中的核心链路

| 环节 | 项目入口与行为 |
| --- | --- |
| ASR / 角色 | `core/step2_whisperX.py`、`core/providers/nemotron_diarization*.py`、`speaker_diarization.py`；原生词时间戳与 Nemotron 角色概率结合，歧义身份不冒充已确认角色 |
| 翻译 | `translation_context.py`、`translation_review.py`、Step 4–8；上下文、角色资料、LineID 及最终 TTS 文本审计 |
| 音色 / 情绪 | `tts_reference_plan.py`、`providers/mlx_tts.py`；短句使用同说话人锚点与原句情绪，长句使用原句双参考，保留参考回执 |
| 时间轴 | `step10_gen_audio.py`、`step11_merge_full_audio.py`、`dubbing_quality.py`；生成时长适配、绝对时间轴、实际发声与排程分开评价 |
| 背景与封装 | `all_whisper_methods/demucs_vl.py`、`step12_merge_dub_to_vid.py`；连续背景、无损母带、PCM 尾部对齐、编码后峰值和流时长校验 |
| 字幕 | `step11_merge_full_audio.py` 与 Step 12；长英文句自然折行、合成时烧录字幕 |

## 本次验证

- **pass**：在 `/Users/vanch/pinokio/bin/miniconda/envs/videolingo/bin/python` 下执行 `python -m pytest -q`，`300 passed in 43.34s`。
- **pass**：最终根配置与 V13 配置解析值一致；显式 IndexTTS + cinematic 的 CLI dry-run 返回 0，计划包含全部 15 阶段，未运行生成。
- **pass**：doctor 中 Nemotron 与 IndexTTS 运行环境检查通过。仅说明该检查覆盖的路径/导入条件通过，不代表本轮重跑了推理。
- **fail**：doctor 总体返回 1。OpenCV 4.14.0.94 要求 NumPy >=2，当前为 1.26.4；备选 ZONOS2 服务未启动、MOSS-TTS 目录和 Ming 模型权重缺失。本次没有调整现有依赖或启动备选服务。
- **pending**：根目录 `output/` 是历史运行，当前清单检查显示 0/15 可复用检查点；其旧状态中的 `accepted` 不能作为当前代码验收依据。根目录旧输出保持原样，最终成片位于下述交付目录。

只读检查原始结果保存在同目录 `doctor.txt`、`status.txt`、`translation-status.txt`、`dry-run.txt`。

## 最终成片与证据边界

成片：`deliverables/e0c803_final_optimal_version_20260927/English_Dub_V13_Optimal_Subtitled.mp4`。

V13 原始 `output/audio/dubbing_eval.json`：`quality_gate.passed=false`，10 ok / 33 warn / 0 fail；实际发声结束偏差最大 3.237 秒，最大速度因子 1.2935。独立声纹与情绪分类均为 `missing evidence`，感知状态为 `perceptual_pending`。这些结果不能写成严格同步与感知质量全部通过。

历史 V13 “Astra PASS” 文件不作为本次独立验收证据；本次只验证代码集成与回归，不声称完成新的独立听审或 100% 复刻原片。用户已要求结束效果优化，因此保留最终媒体及历史审核证据。

V13 启动时的 `code_hashes_snapshot.json` 有 8 个文件与后来修复的根代码不同，不能用它覆盖当前代码；本次源码快照重新记录当前文件哈希。

## 后续使用

在项目环境中先预览：

```bash
python -m core.cli run --input /absolute/path/video.mp4 --source zh --target en --profile cinematic --tts mlx_indextts2 --dry-run
```

确认工作目录与旧产物复用范围后，移除 `--dry-run` 执行。UI 使用根配置；CLI 指定 `--tts mlx_indextts2` 可避免 `auto` 路由选择其他后端。模型运行时及模型权重继续使用 `config.yaml` 所指向的本机环境；复制源码包到其他机器不能替代环境安装。
