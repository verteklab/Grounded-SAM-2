#!/bin/bash

# 优雅停止Gunicorn服务

echo "正在查找Gunicorn进程..."

# 方法1: 通过端口6155查找主进程
MAIN_PID=$(lsof -ti :6155 2>/dev/null | head -1)

# 方法2: 如果方法1失败，通过进程命令查找
if [ -z "$MAIN_PID" ]; then
    # 查找包含 wsgi:app 的 gunicorn 主进程（PPID=1 或没有父gunicorn进程）
    MAIN_PID=$(ps aux | grep "gunicorn.*wsgi:app" | grep -v grep | awk '{print $2}' | while read pid; do
        ppid=$(ps -o ppid= -p $pid 2>/dev/null | tr -d ' ')
        # 主进程通常是PPID=1，或者是没有其他gunicorn作为父进程的
        if [ "$ppid" = "1" ] || ! ps -p $ppid -o comm= 2>/dev/null | grep -q gunicorn; then
            echo $pid
            break
        fi
    done | head -1)
fi

# 方法3: 如果还是找不到，尝试通过配置文件路径查找
if [ -z "$MAIN_PID" ]; then
    MAIN_PID=$(pgrep -f "gunicorn.*config/gunicorn.conf.py.*wsgi:app" | head -1)
fi

if [ -z "$MAIN_PID" ]; then
    echo "未找到运行中的Gunicorn服务（端口6155）"
    exit 0
fi

echo "找到主进程 PID: $MAIN_PID"
echo "发送优雅终止信号..."

# 发送TERM信号给主进程，Gunicorn会自动停止所有worker
kill -TERM $MAIN_PID 2>/dev/null

# 等待进程退出
TIMEOUT=60
COUNT=0
while [ $COUNT -lt $TIMEOUT ]; do
    # 检查端口是否还在使用
    if ! lsof -i :6155 > /dev/null 2>&1; then
        echo ""
        echo "✓ 服务已优雅停止（端口已释放）"
        exit 0
    fi
    # 检查主进程是否还在运行
    if ! ps -p $MAIN_PID > /dev/null 2>&1; then
        echo ""
        echo "✓ 服务已优雅停止（主进程已退出）"
        exit 0
    fi
    sleep 1
    COUNT=$((COUNT + 1))
    echo -n "."
done

# 如果优雅停止失败，强制终止所有相关进程
echo ""
echo "⚠️  优雅停止超时，强制终止所有相关进程..."

# 强制终止主进程
kill -KILL $MAIN_PID 2>/dev/null

# 强制终止所有通过端口6155的进程
lsof -ti :6155 2>/dev/null | xargs -r kill -KILL 2>/dev/null

# 强制终止所有相关的gunicorn进程
pkill -9 -f "gunicorn.*wsgi:app" 2>/dev/null

sleep 2
if ! lsof -i :6155 > /dev/null 2>&1; then
    echo "✓ 服务已强制停止"
else
    echo "⚠️  警告: 可能仍有进程占用端口6155"
    echo "请手动检查: lsof -i :6155"
    exit 1
fi
