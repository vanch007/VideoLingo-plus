# 环境依赖修复记录（2026-10-09）

当前 `videolingo` 环境的 10 条包依赖冲突已消除；默认本地路线的环境诊断、实际音频解码与完整代码回归通过。

## 根因与修复

原环境混用了 NumPy 1.26.4、SciPy 1.13.1、Torch/TorchAudio 2.3.1，以及需要更新版本的 OpenCV、WhisperX、Pyannote、TorchVision。旧安装程序还会先安装另一组旧 Torch，再安装项目依赖，且可能在安装失败后继续报告成功。

- `constraints.txt` 固定经过验证的科学计算与音频版本；`requirements.txt` 和安装程序共同使用它。
- 删除安装程序里单独安装旧 Torch 的路径；Stable-TS 使用固定发布版，Demucs 使用固定提交。
- 增加 `environment.yml`，声明 Python 3.10 和 FFmpeg 7 本地库。
- 当前 macOS 的 FFmpeg 命令版本为 9.0.2，TorchCodec 0.7 无法使用其本地库。安装 keg-only 的 Homebrew `ffmpeg@7`，只向当前 Python 环境暴露缺少的版本化本地库；现有环境库不会被覆盖。具体链接见 `native-libraries.json`。
- 安装程序在依赖检查或本地库导入失败时停止。诊断保留全部依赖冲突；已选配音后端必须通过，未选的实验后端缺失显示为 `WARN`。

## 当前验证版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.21，Apple Silicon |
| NumPy / SciPy | 2.2.6 / 1.15.3 |
| Torch / TorchAudio | 2.8.0 / 2.8.0 |
| TorchVision / TorchCodec | 0.23.0 / 0.7.0 |
| OpenCV | 4.14.0.94 |
| WhisperX / Stable-TS | 3.8.6 / 2.19.1 |
| Pyannote Audio | 4.0.7 |
| spaCy | 3.8.16 |
| TorchCodec 使用的 FFmpeg 库 | 7.1.5 |

## 验证证据

| 检查 | 结果 | 记录 |
| --- | --- | --- |
| 修复前依赖检查 | fail，10 条冲突 | `pip-check-before.txt` |
| 完整依赖清单安装 | pass | `requirements-install.txt` |
| 修复后依赖检查 | pass，No broken requirements found | `pip-check-after.txt` |
| 14 个关键模块导入 | pass | `imports-after.json` |
| 数值计算、重采样、图像运算、中文分词 | pass | `operations-after.json` |
| TorchCodec / Pyannote 实际 WAV 解码 | pass | `operations-after.json` |
| 默认路线环境诊断与 oMLX 请求 | pass，退出码 0 | `doctor.txt` |
| ASR/回读独立环境依赖 | pass | `asr-pip-check.txt` |
| IndexTTS2 独立环境依赖 | pass | `indextts2-pip-check.txt` |
| 新环境 Conda 依赖求解（dry-run） | pass，未执行安装 | `fresh-environment-plan.json` |
| 完整代码回归 | pass，609 项 | `pytest.txt` |
| 最后诊断改动相关回归 | pass，49 项 | `targeted-pytest.txt` |

Python 编译检查和 `git diff --check` 通过。修复针对现有环境，未另行安装一个完整的新环境，也未重新生成视频；既有媒体质量验收边界仍见 `docs/localization-workflow.md`。

## 安装与复核

新环境使用项目根的 `environment.yml`；现有环境可通过 `python -m pip install -r requirements.txt` 恢复版本约束。macOS 现有环境若缺少 FFmpeg 7 本地库，先安装 Homebrew `ffmpeg@7`，再运行安装程序。

```bash
conda env create -f environment.yml
conda activate videolingo
python install.py
python -m core.cli doctor
```

## 已知提示与证据边界

- Higgs、dots、ZONOS2、MOSS-TTS、Ming 尚未配置齐全或服务未启动；它们未被当前默认路线选用。切换使用时会成为必需检查，不能用 `WARN` 绕过。
- macOS 同一进程同时导入 OpenCV、PyAV 与 TorchCodec 时，仍可能出现 FFmpeg 的 AVFoundation 类重复加载提示，保留在 `runtime-probe.log` 和 `doctor.txt`。本次导入、实际解码及回归通过，但没有据此宣称该组合在所有视频输入下无崩溃风险。未改写第三方二进制或抑制这项提示。
- 环境诊断对本地 ASR/对齐检查入口及运行环境；没有执行新的模型全片识别。独立环境包兼容性已单独核对。
