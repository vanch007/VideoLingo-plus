# 🚀 开始使用

## 📋 API 配置指南
本项目翻译需要大模型；配音统一使用本地 MLX 音色克隆路由，无需 TTS API Key。

### 1. **大模型的 API_KEY**：

| 推荐模型 | 推荐提供商 | base_url | 价格 | 效果 |
|:-----|:---------|:---------|:-----|:---------|
| claude-3-5-sonnet-20240620 | [302AI](https://gpt302.saaslink.net/C2oHR9) | https://api.302.ai | $15 / 1M tokens | 🤩 |
| gemini-2.0-flash-exp | [302AI](https://gpt302.saaslink.net/C2oHR9) | https://api.302.ai | $0.3 / 1M tokens | 😃 |
| deepseek-chat(v3) | [302AI](https://gpt302.saaslink.net/C2oHR9) | https://api.302.ai | ¥4 / 1M tokens | 🥳 |
| qwen2.5-coder:32b | [Ollama](https://ollama.ai) | http://localhost:11434 | 本地 | 😃 |

注：支持 OpenAI 格式接口，可自行尝试不同模型。但处理过程涉及多步思维链和复杂的json格式，**不建议使用小于 30B 的模型**。

### 2. **本地 MLX TTS**

此分支仅提供 MLX 音色克隆路由：IndexTTS2、OmniVoice、Qwen3-TTS、VoxCPM2、Higgs Audio、dots.tts、ZONOS2 和 MOSS-TTS。配置入口为 `mlx_tts.backends`。

## 🛠️ 快速上手

VideoLingo 支持 Windows、macOS 和 Linux 系统，可使用 CPU 或 GPU 运行。

> **注意:** 在 Windows 上使用 NVIDIA GPU 加速，请先完成以下步骤:
> 1. 安装 [CUDA Toolkit 12.6](https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.76_windows.exe)
> 2. 安装 [CUDNN 9.3.0](https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn_9.3.0_windows.exe)
> 3. 将 `C:\Program Files\NVIDIA\CUDNN\v9.3\bin\12.6` 添加到系统 PATH
> 4. 重启电脑

> **注意:** FFmpeg 是必需的，请通过包管理器安装：
> - Windows：```choco install ffmpeg```（通过 [Chocolatey](https://chocolatey.org/)）
> - macOS：```brew install ffmpeg```（通过 [Homebrew](https://brew.sh/)）
> - Linux：```sudo apt install ffmpeg```（Debian/Ubuntu）

开始安装 VideoLingo 之前，请确保安装了 Git 和 Anaconda。

1. 克隆项目：
   ```bash
   git clone https://github.com/Huanshere/VideoLingo.git
   cd VideoLingo
   ```

2. 创建并活虚拟环境（**必须使用 3.10**）：
   ```bash
   conda create -n videolingo python=3.10.0 -y
   conda activate videolingo
   ```

3. 运行安装脚本：
   ```bash
   python install.py
   ```

4. 🎉 输入命令或点击 `一键启动.bat` 启动 Streamlit 应用：
   ```bash
   streamlit run st.py
   ```

5. 在弹出网页的侧边栏中设置key，开始使用~

   ![tutorial](./zh_page.png)

6. （可选）更多设置可以在 `config.yaml` 中手动修改，运行过程请注意命令行输出。如需使用自定义术语，请在处理前将术语添加到 `custom_terms.xlsx` 中，例如 `Biden | 登子 | 美国的瞌睡总统`。

> 需要帮助？我们的 [AI助手](https://share.fastgpt.in/chat/share?shareId=066w11n3r9aq6879r4z0v9rh) 随时解答问题！


## 🏭 批量模式（beta）

使用说明: [English](/batch/README.md) | [简体中文](/batch/README.zh.md)

这个模式仍处于早期开发阶段，可能有潜在的错误。

## 🚨 常见报错

1. **翻译过程的 'All array must be of the same length' 或 'Key Error'**: 
   - 原因1：弱模型遵循JSON格式能力较弱导致响应解析错误。
   - 原因2：对于敏感内容，LLM可能拒绝翻译。
   解决方案：检查 `output/gpt_log/error.json` 的 `response` 和 `msg` 字段，删掉 `output/gpt_log` 文件夹后重试。

2. **'Retry Failed', 'SSL', 'Connection', 'Timeout'**: 通常是网络问题。解决方案：中国大陆用户请切换网络节点重试。

3. **local_files_only=True**：网络问题引起的模型下载失败，需要确认网络能 ping 通 `huggingface.co`。
