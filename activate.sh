#!/bin/bash
# 快速激活 FoloVLLM 虚拟环境

if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在"
    echo "请先运行: bash setup_env.sh"
    exit 1
fi

source venv/bin/activate
echo "✓ FoloVLLM 虚拟环境已激活"
echo "Python: $(python --version)"
echo "位置: $(which python)"

