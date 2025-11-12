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
export CUDA_VISIBLE_DEVICES=1  # 指定使用的GPU

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

# === 启动服务 ===
echo "正在启动Gunicorn服务..."
echo "工作目录: $(pwd)"
echo "项目根目录: ${PROJECT_ROOT}"
echo "使用配置: $CONFIG_FILE"

# 使用 wsgi:app 或 app:app 都可以，但 wsgi.py 中已经导入了 app
# 当 preload_app=True 时，Gunicorn 会在主进程加载应用，模型也会在主进程加载
gunicorn -c "$CONFIG_FILE" wsgi:app

if [ $? -eq 0 ]; then
    echo "服务启动成功！"
    echo "访问地址: http://localhost:6155"
    echo "查看日志: tail -f logs/*.log"
else
    echo "服务启动失败！"
    exit 1
fi
