#!/bin/bash
# VideoLingo-plus 一键启动脚本 (macOS)
# 自动激活 conda 环境并启动 Streamlit WebUI

set -e

CONDA_ENV="videolingo"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔══════════════════════════════════════╗"
echo "║   VideoLingo-plus WebUI Launcher     ║"
echo "╚══════════════════════════════════════╝"
echo ""

# 初始化 conda（兼容 bash/zsh）
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
else
    echo "❌ 未找到 conda 安装，请先安装 Miniconda 或 Anaconda"
    echo "   下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 检查 conda 环境是否存在
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "❌ conda 环境 '${CONDA_ENV}' 不存在"
    echo "   请先运行: python install.py"
    exit 1
fi

echo "✅ 激活 conda 环境: ${CONDA_ENV}"
conda activate "${CONDA_ENV}"

echo "✅ 切换工作目录: ${SCRIPT_DIR}"
cd "${SCRIPT_DIR}"

echo "🚀 启动 VideoLingo-plus WebUI..."
echo "   访问地址: http://localhost:8501"
echo ""

python -m streamlit run st.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --browser.gatherUsageStats false
