#!/bin/bash
# Docker 构建脚本

set -e

echo "=========================================="
echo "Grounded-SAM-2 Docker 构建脚本"
echo "=========================================="

# 检查是否在项目根目录
if [ ! -f "Dockerfile.server.new" ]; then
    echo "错误：请在项目根目录运行此脚本"
    exit 1
fi

# 构建代码容器
echo ""
echo "构建服务容器镜像..."
docker build -f Dockerfile.server -t grounded-sam2-server:v1.2 .

if [ $? -eq 0 ]; then
    echo "✅ 服务容器镜像构建成功"
else
    echo "❌ 服务容器镜像构建失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 镜像构建完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 确保模型文件已下载到以下位置："
echo "   - checkpoints/sam2.1_hiera_large.pt"
echo "   - gdino_checkpoints/groundingdino_swint_ogc.pth"
echo ""
echo "2. （可选）准备 Hugging Face 模型："
echo "   - 方式1: 使用本地缓存: $HOME/.cache/huggingface/hub"
echo "   - 方式2: 在项目目录创建 hf_models/ 目录并放置模型"
echo ""
echo "3. 运行 docker-run.sh 启动容器"
echo ""
