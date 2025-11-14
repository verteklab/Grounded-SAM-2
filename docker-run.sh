#!/bin/bash
# Docker 运行脚本

set -e

echo "=========================================="
echo "Grounded-SAM-2 Docker 运行脚本"
echo "=========================================="

# 检查镜像是否存在（支持 v1.0 和 latest 标签）
SERVER_IMAGE="grounded-sam2-server:v1.1"

if ! docker image inspect $SERVER_IMAGE > /dev/null 2>&1; then
    if docker image inspect grounded-sam2-server:latest > /dev/null 2>&1; then
        SERVER_IMAGE="grounded-sam2-server:latest"
    else
        echo "错误：代码容器镜像不存在，请先运行 docker-build.sh"
        exit 1
    fi
fi

echo "使用镜像: $SERVER_IMAGE"

# 端口配置（可通过环境变量 PORT 覆盖，默认 6155）
PORT=${PORT:-6156}
echo "服务端口: $PORT"

# 检查模型文件是否存在
if [ ! -f "checkpoints/sam2.1_hiera_large.pt" ]; then
    echo "警告：SAM2模型文件不存在: checkpoints/sam2.1_hiera_large.pt"
    echo "请确保模型文件已下载到正确位置"
fi

if [ ! -f "gdino_checkpoints/groundingdino_swint_ogc.pth" ]; then
    echo "警告：GroundingDINO模型文件不存在: gdino_checkpoints/groundingdino_swint_ogc.pth"
    echo "请确保模型文件已下载到正确位置"
fi

# 检查 bert-base-uncased 模型是否存在（可选）
# 优先级：1. flask-server/bert-base-uncased-local (项目本地模型)
#         2. $HOME/.cache/huggingface/hub/models--bert-base-uncased (标准 HF 缓存)
#         3. hf_models/hub/models--bert-base-uncased (项目目录下的 HF 格式模型)
HF_MODELS_DIR=""
BERT_LOCAL_MODEL=""
if [ -d "flask-server/bert-base-uncased-local" ]; then
    # 优先使用项目目录下的本地模型（推荐）
    BERT_LOCAL_MODEL="$(pwd)/flask-server/bert-base-uncased-local"
    echo "✅ 发现项目本地 bert-base-uncased 模型: $BERT_LOCAL_MODEL"
    echo "   将挂载到容器: /app/flask-server/bert-base-uncased-local"
    echo "   注意：此模型将直接通过路径使用，无需设置 TRANSFORMERS_CACHE"
elif [ -d "$HOME/.cache/huggingface/hub/models--bert-base-uncased" ]; then
    # 使用标准 Hugging Face 缓存格式
    HF_MODELS_DIR="$HOME/.cache/huggingface"
    echo "✅ 发现本地 Hugging Face 模型缓存: $HF_MODELS_DIR"
    echo "   模型路径: $HF_MODELS_DIR/hub/models--bert-base-uncased"
    echo "   将挂载到容器: /data/hf_models"
elif [ -d "hf_models/hub/models--bert-base-uncased" ]; then
    # 项目目录下的标准 HF 格式模型
    HF_MODELS_DIR="$(pwd)/hf_models"
    echo "✅ 发现项目目录下的 Hugging Face 模型: $HF_MODELS_DIR"
    echo "   模型路径: $HF_MODELS_DIR/hub/models--bert-base-uncased"
    echo "   将挂载到容器: /data/hf_models"
elif [ -d "hf_models" ]; then
    # 兼容旧格式（如果 hf_models 目录存在但结构不同）
    HF_MODELS_DIR="$(pwd)/hf_models"
    echo "⚠️  发现 hf_models 目录，但未找到 models--bert-base-uncased 子目录"
    echo "   将挂载到容器: /data/hf_models"
else
    echo "⚠️  警告：未找到 bert-base-uncased 模型目录"
    echo "   将尝试在线下载（如果网络可用）或使用容器内的默认路径"
    echo "   提示：可以运行 ./download_hf_models.sh 下载模型"
fi

# 停止并删除已存在的容器
echo ""
echo "清理已存在的容器..."
docker stop grounded-sam2-server 2>/dev/null || true
docker rm grounded-sam2-server 2>/dev/null || true

# 启动代码容器（直接挂载模型文件）
echo ""
echo "启动服务容器..."
docker run -d \
  --name grounded-sam2-server \
  --gpus all \
  -p ${PORT}:${PORT} \
  -v "$(pwd)/checkpoints:/data/checkpoints:ro" \
  -v "$(pwd)/gdino_checkpoints:/data/gdino_checkpoints:ro" \
  ${HF_MODELS_DIR:+-v "$HF_MODELS_DIR:/data/hf_models:ro"} \
  ${BERT_LOCAL_MODEL:+-v "$BERT_LOCAL_MODEL:/app/flask-server/bert-base-uncased-local:ro"} \
  -v "$(pwd)/flask-server/logs:/app/flask-server/logs" \
  -v "$(pwd)/flask-server/results:/app/flask-server/results" \
  -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} \
  -e GPU_DEVICE_ID=${GPU_DEVICE_ID:-2} \
  -e GUNICORN_BIND=0.0.0.0:${PORT} \
  -e GUNICORN_WORKERS=${GUNICORN_WORKERS:-5} \
  -e GUNICORN_THREADS=${GUNICORN_THREADS:-3} \
  ${BERT_LOCAL_MODEL:+-e BERT_MODEL_PATH=/app/flask-server/bert-base-uncased-local} \
  $SERVER_IMAGE

if [ $? -eq 0 ]; then
    echo "✅ 服务容器启动成功"
else
    echo "❌ 服务容器启动失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 容器启动完成！"
echo "=========================================="
echo ""
echo "服务信息："
echo "  - 服务地址: http://localhost:${PORT}"
echo "  - 健康检查: curl http://localhost:${PORT}/health"
echo ""
echo "提示：可以通过环境变量 PORT 指定端口，例如："
echo "  PORT=8080 ./docker-run.sh"
echo ""
echo "查看日志："
echo "  - 服务容器: docker logs -f grounded-sam2-server"
echo ""
echo "停止容器："
echo "  docker stop grounded-sam2-server"
echo ""
