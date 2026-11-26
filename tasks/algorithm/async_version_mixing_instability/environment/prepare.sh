#!/usr/bin/env bash
# AutoRLBench prepare.sh —— 唯一联网阶段。
# 素材阶段只需 numpy（模拟 run 用）。正式 task（真实小模型 async GRPO run）时替换为拉模型/数据的命令。
set -euo pipefail

if ! python3 -c "import numpy" 2>/dev/null; then
    pip install numpy
fi
echo "environment ready"
