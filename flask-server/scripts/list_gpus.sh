#!/bin/bash

# 列出系统中可用的GPU设备

echo "=========================================="
echo "GPU设备列表"
echo "=========================================="

# 检查nvidia-smi是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi 未找到，可能没有安装NVIDIA驱动或没有GPU"
    exit 1
fi

# 显示GPU信息
echo ""
echo "可用GPU设备:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s\n", $1, $2; printf "    显存: %s MB / %s MB (使用率: %.1f%%)\n", $4, $3, ($4/$3)*100; printf "    利用率: %s%%\n\n", $6}'

# 显示GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "总GPU数量: $GPU_COUNT"
echo ""

# 显示当前CUDA_VISIBLE_DEVICES设置
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "当前CUDA_VISIBLE_DEVICES环境变量: $CUDA_VISIBLE_DEVICES"
else
    echo "当前CUDA_VISIBLE_DEVICES环境变量: 未设置（将使用所有GPU）"
fi
echo ""

# 使用建议
echo "使用建议:"
echo "  1. 在gunicorn.conf.py中设置 gpu_device_id = <GPU编号>"
echo "  2. 或在启动时设置环境变量: export CUDA_VISIBLE_DEVICES=<GPU编号>"
echo "  3. 例如使用GPU 1: export CUDA_VISIBLE_DEVICES=1"
echo ""

