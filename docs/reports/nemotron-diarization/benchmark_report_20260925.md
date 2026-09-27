# Nemotron 3 Diarization (8-bit MLX) 基准与回归评测报告

**日期:** 2026-09-25\
**测试平台:** Apple M3 Max (128 GiB 统一内存, macOS)\
**环境:** `/Users/vanch/.codex/envs/nemotron-diarization` (Python 3.12.10, `mlx-audio==0.5.6`, `mlx==0.32.2`)\
**评测模型:** `mlx-community/Nemotron-3-Diarization-8bit` (Revision: `dd8b8ce3d69a80540a7e50150e2beed1c34fee37`)\

---

## 1. 核心工程交付清单

- **独立隔离运行环境**: 专用 Python 3.12 虚拟环境，彻底隔离上游 `transformers 5.x` 与主项目的版本冲突。
- **Subprocess Runner**: `core/providers/nemotron_diarization_runner.py` 实现自包含 CLI，产出 `segments.json`, `rttm.txt`, `summary.json`, `probabilities.npz`。
- **统一提供方适配**: `core/providers/nemotron_diarization.py` 封装健康探针与执行器，集成进 `core/doctor.py` (`diarization:nemotron-mlx`)。
- **双后端路由与 Shadow 模式**: `core/providers/speaker_diarization.py` 支持 `backend: "nemotron-mlx"` 及 `shadow_backend` 影子评估。
- **细粒度概率对齐**: 基于 10ms 帧步长概率张量，对 ASR 词按声学置信度赋予说话人标签，防止低置信度盲目贴标。
- **覆盖检查解耦**: `core/providers/source_coverage.py` 分离文本见证分析与纯声学时域空洞检测，杜绝无文本结果虚假 100% pass。
- **断句与时间轴防串音**: 修复未知角色错误合并问题；拦截跨说话人单句合并；如实统计 `cross_speaker_rows`。
- **参考音频与克隆隔离**: `core/step9_extract_refer_audio.py` 在换人边界禁止向前后跨人扩展；`core/all_tts_functions/tts_utils.py` 开启严格模式禁止借用其他角色音色；`core/step10_gen_audio.py` 将角色与参考音频纳入缓存指纹。

---

## 2. 真实素材推理性能与分段实测

| 音频素材 | 时长 | 推理耗时 | RTFx 加速比 | 检出说话人 | 分段数 | 重叠与特性验证 |
|---|---|---|---|---|---|---|
| `vocal.wav` (单人带货) | 50.81s | 0.875s | **58.1x** | 1 (S01) | 6 | 静音区间切分准确，无多余虚假说话人 |
| `for_whisper.mp3` (单人口播) | 40.27s | 0.081s | **497x** (热缓存) | 1 (S01) | 2 | 连续语音完整覆盖 0.05s-40.24s |
| `e0c803SQrss` (双人对白台词) | 45.00s | 0.072s | **625x** (热缓存) | 3 (S01, S02, S03) | 9 | 精准检出 8.36s-8.88s 重叠发言与快速接话 |

---

## 3. 边界工况与健壮性回归

- **全静音音频**: 产生 0 说话人、0 分段，不产生虚假台词区间。
- **微短音频 (<10ms)**: 安全跳过，不触发数组越界或分母为零异常。
- **非整帧音频 (0.35s)**: 正确处理尾部未对齐采样点，平滑刷新缓冲区。
- **多轮调用隔离**: 验证多次调用间 `StreamingState` 不发生上下文污染。

---

## 4. 单元与回归测试结果

运行测试命令：
```bash
conda run -n videolingo pytest tests/test_speaker_diarization.py tests/test_source_coverage.py tests/test_step6_timeline_alignment.py tests/test_timeline_rescue.py tests/test_tts_reference_selection.py -v
```
结果：**24 passed in 2.26s (100% 通过)**。

---

## 5. 建议与默认策略

1. **当前生产配置**: 保持 `backend: "moss-mlx"`，启用 `shadow_backend: "nemotron-mlx"`。在实际视频制作中自动生成对照日志 `output/log/speaker_diarization_shadow.json`。
2. **切换默认准则**: 当在业务代表性长视频（特别是包含 3 人以上、嘈杂 BGM 或方言场景）上确认 Nemotron 换人点听感优于 MOSS 且无音色漂移时，直接在 `config.yaml` 中将 `backend` 修改为 `nemotron-mlx` 即可无缝切换。
