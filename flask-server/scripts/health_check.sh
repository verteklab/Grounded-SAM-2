#!/bin/bash

# Grounded-SAM2健康检查脚本

# === 服务状态检查 ===
echo "=== 服务状态 ==="
if pgrep -f "gunicorn.*grounded-sam2-server" > /dev/null; then
    echo "✓ Gunicorn服务运行中"
    # 显示工作进程数
    PROC_COUNT=$(pgrep -f "gunicorn.*worker" | wc -l)
    echo "  工作进程数: $PROC_COUNT"
else
    echo "✗ Gunicorn服务未运行"
fi

# === 端口监听检查 ===
if lsof -i :6155 > /dev/null 2>&1; then
    echo "✓ 端口6155正在监听"
else
    echo "✗ 端口6155未监听"
fi

# === 系统资源检查 ===
echo -e "\n=== 系统资源 ==="
# CPU使用率
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
echo "CPU使用率: ${CPU_USAGE}%"

# 内存使用率
MEM_USAGE=$(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')
echo "内存使用率: $MEM_USAGE"

# GPU状态（如果安装了nvidia-smi）
if command -v nvidia-smi > /dev/null; then
    echo -e "\n=== GPU状态 ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx name used total util; do
        echo "GPU${idx}(${name}): 显存 ${used}MB/${total}MB, 利用率 ${util}%"
    done
fi

# === 接口测试 ===
echo -e "\n=== API测试 ==="
if curl -s http://localhost:6155/health > /dev/null 2>&1; then
    echo "✓ /health 接口正常"
    HEALTH_STATUS=$(curl -s http://localhost:6155/health | jq -r '.status' 2>/dev/null || echo "unknown")
    echo "  服务状态: $HEALTH_STATUS"
else
    echo "✗ /health 接口异常"
fi

# === 日志检查 ===
echo -e "\n=== 最近日志 ==="
echo "错误日志（最后5行）:"
tail -n 5 logs/error.log 2>/dev/null || echo "无错误日志"
echo -e "\n访问日志（最后5行）:"
tail -n 5 logs/access.log 2>/dev/null || echo "无访问日志"
