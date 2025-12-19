#!/bin/bash
# VideoLingo 批处理脚本启动器
# 自动使用正确的 conda 环境运行批处理任务

echo "🚀 Starting VideoLingo batch processor..."
echo "📦 Using conda environment: videolingo"
echo ""

conda run -n videolingo python batch/utils/batch_processor.py "$@"
