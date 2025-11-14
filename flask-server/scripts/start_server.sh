#!/bin/bash

# Grounded-SAM2 Gunicorn启动脚本

# === 获取脚本所在目录和项目根目录 ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASK_SERVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$FLASK_SERVER_DIR/.." && pwd)"

# 切换到flask-server目录
cd "$FLASK_SERVER_DIR" || {
    echo "错误：无法切换到flask-server目录"
    exit 1
}

# === 环境设置 ===
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# === GPU配置 ===
# 单GPU模式：使用环境变量CUDA_VISIBLE_DEVICES指定GPU设备
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=1  # 默认使用GPU 1
    echo "ℹ️  使用GPU 1（可通过CUDA_VISIBLE_DEVICES环境变量修改）"
else
    echo "ℹ️  使用GPU $CUDA_VISIBLE_DEVICES"
fi

# === Hugging Face 离线模式配置 ===
# 使用本地缓存，避免网络连接问题
if [ -d "${HOME}/.cache/huggingface/hub" ]; then
    export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface"
    export HF_HOME="${HOME}/.cache/huggingface"
    echo "✅ 使用本地 Hugging Face 缓存: ${TRANSFORMERS_CACHE}"
else
    echo "⚠️  警告: 未找到 Hugging Face 缓存目录，将尝试在线下载"
fi

# 强制离线模式（如果网络不可达，会使用本地缓存）
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# === 参数处理 ===
CONFIG_MODE=${1:-"prod"}  # 默认生产环境，可传参dev

if [ "$CONFIG_MODE" = "dev" ]; then
    CONFIG_FILE="config/development.py"
    echo "启动开发环境..."
else
    CONFIG_FILE="config/gunicorn.conf.py"
    echo "启动生产环境..."
fi

# === 预检查 ===
echo "检查依赖..."
python -c "import torch, flask, gunicorn, psutil" || {
    echo "错误：缺少依赖包，请运行: pip install psutil"
    exit 1
}

echo "检查模型文件..."
# 检查SAM2模型文件
if [ ! -f "${PROJECT_ROOT}/checkpoints/sam2.1_hiera_large.pt" ]; then
    echo "错误：SAM2模型文件不存在: ${PROJECT_ROOT}/checkpoints/sam2.1_hiera_large.pt"
    exit 1
fi

# 检查GroundingDINO模型文件
if [ ! -f "${PROJECT_ROOT}/gdino_checkpoints/groundingdino_swint_ogc.pth" ]; then
    echo "错误：GroundingDINO模型文件不存在: ${PROJECT_ROOT}/gdino_checkpoints/groundingdino_swint_ogc.pth"
    exit 1
fi

echo "✅ 模型文件检查通过"

echo "检查端口是否占用..."
if lsof -i :6155 > /dev/null 2>&1; then
    echo "错误：端口6155已被占用"
    exit 1
fi

# === 创建日志目录 ===
mkdir -p logs

# === PID文件路径 ===
PID_FILE="${FLASK_SERVER_DIR}/logs/gunicorn.pid"

# === 检查是否已经在运行 ===
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  警告: 服务已在运行 (PID: $OLD_PID)"
        echo "   如需重启，请先停止服务: kill $OLD_PID 或删除 $PID_FILE"
        exit 1
    else
        echo "清理旧的PID文件..."
        rm -f "$PID_FILE"
    fi
fi

# === 启动服务（后台运行） ===
echo "正在启动Gunicorn服务（后台模式）..."
echo "工作目录: $(pwd)"
echo "项目根目录: ${PROJECT_ROOT}"
echo "使用配置: $CONFIG_FILE"
echo ""

# 使用 wsgi:app 或 app:app 都可以，但 wsgi.py 中已经导入了 app
# 当 preload_app=True 时，Gunicorn 会在主进程加载应用，模型也会在主进程加载
# 使用 nohup 和 & 让服务在后台运行，并将输出重定向到日志文件
nohup gunicorn -c "$CONFIG_FILE" wsgi:app > logs/gunicorn_startup.log 2>&1 &
GUNICORN_PID=$!

# 等待一下，检查进程是否成功启动
sleep 2

if ps -p "$GUNICORN_PID" > /dev/null 2>&1; then
    # 保存PID到文件
    echo "$GUNICORN_PID" > "$PID_FILE"
    
    echo "✅ 服务已在后台启动成功！"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  进程ID (PID): $GUNICORN_PID"
    echo "  访问地址: http://localhost:6155"
    echo "  PID文件: $PID_FILE"
    echo ""
    echo "  查看日志:"
    echo "    tail -f logs/app.log          # 应用日志"
    echo "    tail -f logs/access.log       # 访问日志"
    echo "    tail -f logs/error.log        # 错误日志"
    echo "    tail -f logs/gunicorn_startup.log  # 启动日志"
    echo ""
    echo "  停止服务:"
    echo "    kill $GUNICORN_PID"
    echo "    或"
    echo "    kill \$(cat $PID_FILE)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "❌ 服务启动失败！"
    echo "查看启动日志: cat logs/gunicorn_startup.log"
    exit 1
fi
