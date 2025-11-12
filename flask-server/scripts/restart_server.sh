#!/bin/bash

# Grounded-SAM2 服务重启脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASK_SERVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$FLASK_SERVER_DIR" || {
    echo "❌ 错误：无法切换到flask-server目录"
    exit 1
}

echo "=========================================="
echo "重启 Grounded-SAM2 服务"
echo "=========================================="
echo ""

# 第一步：停止服务
echo "步骤 1/2: 停止服务..."
if [ -f "$SCRIPT_DIR/stop_server.sh" ]; then
    bash "$SCRIPT_DIR/stop_server.sh"
    STOP_STATUS=$?
    if [ $STOP_STATUS -ne 0 ]; then
        echo "⚠️  停止服务时出现问题，但继续尝试启动..."
    fi
else
    echo "⚠️  未找到 stop_server.sh，尝试手动停止..."
    PIDS=$(pgrep -f "gunicorn.*grounded-sam2-server")
    if [ -n "$PIDS" ]; then
        kill -TERM $PIDS
        sleep 3
    fi
fi

# 等待端口释放
echo "等待端口释放..."
for i in {1..10}; do
    if ! lsof -i :6155 > /dev/null 2>&1; then
        echo "✓ 端口已释放"
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# 第二步：启动服务
echo "步骤 2/2: 启动服务..."
if [ -f "$SCRIPT_DIR/start_server.sh" ]; then
    bash "$SCRIPT_DIR/start_server.sh" "$@"
    START_STATUS=$?
    if [ $START_STATUS -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅ 服务重启成功！"
        echo "=========================================="
        echo "访问地址: http://localhost:6155"
        echo "查看日志: tail -f logs/*.log"
        echo "健康检查: curl http://localhost:6155/health"
    else
        echo ""
        echo "=========================================="
        echo "❌ 服务启动失败！"
        echo "=========================================="
        echo "请检查日志: tail -f logs/error.log"
        exit 1
    fi
else
    echo "❌ 错误：未找到 start_server.sh"
    exit 1
fi

