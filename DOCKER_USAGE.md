# Docker 构建和运行说明文档

## 📋 目录

- [概述](#概述)
- [前置要求](#前置要求)
- [快速开始](#快速开始)
- [详细步骤](#详细步骤)
- [配置说明](#配置说明)
- [模型文件准备](#模型文件准备)
- [环境变量配置](#环境变量配置)
- [常用操作](#常用操作)
- [故障排查](#故障排查)

## 概述

本项目提供了两个主要脚本用于 Docker 容器的构建和运行：

- **docker-build.sh**: 构建 Docker 镜像
- **docker-run.sh**: 运行 Docker 容器并启动服务

### 架构说明

- **单容器架构**: 所有代码和依赖都打包在一个镜像中
- **模型文件挂载**: 模型权重文件通过 volume 挂载，不包含在镜像中
- **支持 GPU**: 容器支持 NVIDIA GPU 加速

## 前置要求

### 1. 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **Docker**: 已安装 Docker Engine 20.10+
- **NVIDIA Docker**: 已安装 nvidia-docker2 或 nvidia-container-toolkit
- **GPU**: NVIDIA GPU，支持 CUDA 12.1+
- **磁盘空间**: 至少 20GB 可用空间（用于镜像构建）
- **内存**: 至少 16GB RAM

### 2. 验证环境

```bash
# 检查 Docker
docker --version

# 检查 GPU 支持
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 检查是否在项目根目录
ls -la Dockerfile.server docker-build.sh docker-run.sh
```

## 快速开始

### 一键构建和运行

```bash
# 1. 构建镜像（首次运行，需要 10-30 分钟）
./docker-build.sh

# 2. 运行容器
./docker-run.sh

# 3. 测试服务（等待 30-60 秒让模型加载）
curl http://localhost:6156/health
```

## 详细步骤

### 步骤 1: 构建 Docker 镜像

#### 使用脚本构建（推荐）

```bash
# 确保在项目根目录
cd /path/to/Grounded-SAM-2-1

# 给脚本添加执行权限（如果还没有）
chmod +x docker-build.sh

# 执行构建
./docker-build.sh
```

#### 手动构建

```bash
docker build -f Dockerfile.server -t grounded-sam2-server:v1.1 .
```

#### 构建说明

- **镜像名称**: `grounded-sam2-server:v1.1`
- **构建时间**: 约 10-30 分钟（取决于网络和系统性能）
- **包含内容**:
  - Flask 应用代码 (`flask-server/`)
  - SAM2 和 GroundingDINO 代码库
  - Python 依赖包
  - 系统依赖（ffmpeg 等）

**注意**: 模型权重文件（`.pt`, `.pth`）**不包含**在镜像中，需要通过 volume 挂载。

### 步骤 2: 准备模型文件

在运行容器之前，需要确保以下模型文件已准备好：

#### 必需模型文件

```bash
# SAM2 模型
checkpoints/sam2.1_hiera_large.pt

# GroundingDINO 模型
gdino_checkpoints/groundingdino_swint_ogc.pth
```

#### 下载模型文件

如果模型文件不存在，可以使用以下命令下载：

```bash
# 下载 SAM2 模型
cd checkpoints
bash download_ckpts.sh
cd ..

# 下载 GroundingDINO 模型
cd gdino_checkpoints
bash download_ckpts.sh
cd ..
```

#### bert-base-uncased 模型（可选）

脚本会自动检测以下位置的 bert-base-uncased 模型，按优先级排序：

1. **项目本地模型**（推荐）: `flask-server/bert-base-uncased-local/`
2. **标准 HF 缓存**: `$HOME/.cache/huggingface/hub/models--bert-base-uncased/`
3. **项目目录 HF 格式**: `hf_models/hub/models--bert-base-uncased/`

如果未找到，容器会尝试在线下载（需要网络连接）。

### 步骤 3: 运行容器

#### 使用脚本运行（推荐）

```bash
# 使用默认配置运行
./docker-run.sh

# 指定端口运行
PORT=8080 ./docker-run.sh

# 指定 GPU 设备运行
CUDA_VISIBLE_DEVICES=0 ./docker-run.sh
```

#### 手动运行

```bash
docker run -d \
  --name grounded-sam2-server \
  --gpus all \
  -p 6156:6156 \
  -v "$(pwd)/checkpoints:/data/checkpoints:ro" \
  -v "$(pwd)/gdino_checkpoints:/data/gdino_checkpoints:ro" \
  -v "$HOME/.cache/huggingface:/data/hf_models:ro" \
  -v "$(pwd)/flask-server/bert-base-uncased-local:/app/flask-server/bert-base-uncased-local:ro" \
  -v "$(pwd)/flask-server/logs:/app/flask-server/logs" \
  -v "$(pwd)/flask-server/results:/app/flask-server/results" \
  -e CUDA_VISIBLE_DEVICES=2 \
  -e GPU_DEVICE_ID=2 \
  -e GUNICORN_BIND=0.0.0.0:6156 \
  -e GUNICORN_WORKERS=5 \
  -e GUNICORN_THREADS=3 \
  grounded-sam2-server:v1.1
```

#### 运行说明

- **容器名称**: `grounded-sam2-server`
- **端口映射**: 默认 `6156:6156`（可通过 `PORT` 环境变量修改）
- **GPU 设备**: 默认使用 GPU 2（可通过 `CUDA_VISIBLE_DEVICES` 修改）
- **模型加载**: 首次启动需要 30-60 秒加载模型

### 步骤 4: 验证服务

```bash
# 等待模型加载完成（30-60 秒）
sleep 60

# 健康检查
curl http://localhost:6156/health

# 查看服务信息
curl http://localhost:6156/

# 查看日志
docker logs -f grounded-sam2-server
```

## 配置说明

### docker-build.sh 配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| 镜像名称 | Docker 镜像标签 | `grounded-sam2-server:v1.1` |
| Dockerfile | 构建文件 | `Dockerfile.server` |

### docker-run.sh 配置

脚本支持通过环境变量覆盖默认配置：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `PORT` | 服务端口 | `6156` |
| `CUDA_VISIBLE_DEVICES` | GPU 设备 ID | `2` |
| `GPU_DEVICE_ID` | GPU 设备 ID（兼容） | `2` |
| `GUNICORN_WORKERS` | Worker 进程数 | `5` |
| `GUNICORN_THREADS` | 每个 Worker 的线程数 | `3` |

### 使用示例

```bash
# 使用自定义端口和 GPU
PORT=8080 CUDA_VISIBLE_DEVICES=0 ./docker-run.sh

# 调整 Worker 数量
GUNICORN_WORKERS=3 GUNICORN_THREADS=2 ./docker-run.sh

# 组合配置
PORT=9000 CUDA_VISIBLE_DEVICES=1 GUNICORN_WORKERS=4 ./docker-run.sh
```

## 模型文件准备

### 模型文件结构

```
Grounded-SAM-2-1/
├── checkpoints/
│   └── sam2.1_hiera_large.pt          # SAM2 模型（必需）
├── gdino_checkpoints/
│   └── groundingdino_swint_ogc.pth    # GroundingDINO 模型（必需）
└── flask-server/
    └── bert-base-uncased-local/       # bert-base-uncased 模型（可选，推荐）
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── ...
```

### 模型文件检查

```bash
# 检查必需模型
ls -lh checkpoints/sam2.1_hiera_large.pt
ls -lh gdino_checkpoints/groundingdino_swint_ogc.pth

# 检查可选模型
ls -d flask-server/bert-base-uncased-local 2>/dev/null || \
ls -d $HOME/.cache/huggingface/hub/models--bert-base-uncased 2>/dev/null || \
echo "未找到 bert-base-uncased 模型"
```

## 环境变量配置

### 容器内环境变量

以下环境变量可以在 `docker run` 时通过 `-e` 参数设置：

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `CUDA_VISIBLE_DEVICES` | 可见的 GPU 设备 | `2` |
| `GPU_DEVICE_ID` | GPU 设备 ID | `2` |
| `GUNICORN_BIND` | Gunicorn 绑定地址 | `0.0.0.0:6156` |
| `GUNICORN_WORKERS` | Worker 进程数 | `5` |
| `GUNICORN_THREADS` | 每个 Worker 线程数 | `3` |
| `GUNICORN_TIMEOUT` | 请求超时时间（秒） | `300` |
| `BERT_MODEL_PATH` | bert-base-uncased 模型路径 | `/app/flask-server/bert-base-uncased-local` |

### 在 Dockerfile 中定义的环境变量

这些变量在 Dockerfile 中已定义，可以通过运行时环境变量覆盖：

- `TRANSFORMERS_CACHE`: `/data/hf_models`
- `HF_HOME`: `/data/hf_models`
- `TRANSFORMERS_OFFLINE`: `1`
- `HF_HUB_OFFLINE`: `1`

## 常用操作

### 查看容器状态

```bash
# 查看运行中的容器
docker ps | grep grounded-sam2-server

# 查看所有容器（包括已停止的）
docker ps -a | grep grounded-sam2-server
```

### 查看日志

```bash
# 实时查看日志
docker logs -f grounded-sam2-server

# 查看最后 100 行日志
docker logs --tail 100 grounded-sam2-server

# 查看最近 10 分钟的日志
docker logs --since 10m grounded-sam2-server
```

### 停止和启动容器

```bash
# 停止容器
docker stop grounded-sam2-server

# 启动已停止的容器
docker start grounded-sam2-server

# 重启容器
docker restart grounded-sam2-server
```

### 删除容器

```bash
# 停止并删除容器
docker stop grounded-sam2-server
docker rm grounded-sam2-server

# 或者强制删除（如果容器正在运行）
docker rm -f grounded-sam2-server
```

### 进入容器调试

```bash
# 进入运行中的容器
docker exec -it grounded-sam2-server /bin/bash

# 在容器内查看模型文件
docker exec grounded-sam2-server ls -lh /data/checkpoints/
docker exec grounded-sam2-server ls -lh /app/flask-server/bert-base-uncased-local/
```

### 查看资源使用

```bash
# 查看容器资源使用情况
docker stats grounded-sam2-server

# 查看容器详细信息
docker inspect grounded-sam2-server
```

## 故障排查

### 1. 构建失败

**问题**: `docker build` 失败

**可能原因**:
- 网络连接问题
- Dockerfile 语法错误
- 磁盘空间不足

**解决方法**:
```bash
# 检查网络连接
ping pypi.tuna.tsinghua.edu.cn

# 检查磁盘空间
df -h

# 清理 Docker 缓存
docker system prune -a
```

### 2. 运行失败：GPU 不可用

**问题**: `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`

**解决方法**:
```bash
# 安装 nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 验证 GPU 支持
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 3. 运行失败：端口被占用

**问题**: `Error: bind: address already in use`

**解决方法**:
```bash
# 查找占用端口的进程
sudo lsof -i :6156

# 停止占用端口的容器
docker stop $(docker ps -q --filter "publish=6156")

# 或使用其他端口
PORT=8080 ./docker-run.sh
```

### 4. 模型加载失败

**问题**: 容器启动后模型加载失败

**解决方法**:
```bash
# 检查模型文件是否存在
ls -lh checkpoints/sam2.1_hiera_large.pt
ls -lh gdino_checkpoints/groundingdino_swint_ogc.pth

# 检查模型文件权限
ls -l checkpoints/ gdino_checkpoints/

# 查看容器日志
docker logs grounded-sam2-server

# 检查挂载点
docker exec grounded-sam2-server ls -lh /data/checkpoints/
docker exec grounded-sam2-server ls -lh /data/gdino_checkpoints/
```

### 5. 显存不足

**问题**: `CUDA out of memory`

**解决方法**:
```bash
# 减少 Worker 数量
GUNICORN_WORKERS=2 ./docker-run.sh

# 或使用更小的模型
# 修改 Dockerfile 中的模型配置，使用 base_plus 而不是 large
```

### 6. 服务无响应

**问题**: 健康检查返回 503 或超时

**解决方法**:
```bash
# 等待更长时间（模型加载需要时间）
sleep 120
curl http://localhost:6156/health

# 查看日志确认模型是否加载完成
docker logs grounded-sam2-server | grep "模型加载"

# 检查容器是否正常运行
docker ps | grep grounded-sam2-server
```

### 7. bert-base-uncased 模型未找到

**问题**: 警告信息显示未找到 bert-base-uncased 模型

**解决方法**:
```bash
# 方式1: 使用项目本地模型（推荐）
# 确保 flask-server/bert-base-uncased-local/ 目录存在
ls -d flask-server/bert-base-uncased-local

# 方式2: 使用标准 HF 缓存
# 确保 $HOME/.cache/huggingface/hub/models--bert-base-uncased 存在
ls -d $HOME/.cache/huggingface/hub/models--bert-base-uncased

# 方式3: 下载模型
./download_hf_models.sh
```

## 性能调优

### GPU 显存配置

根据 GPU 显存调整 Worker 数量：

| GPU 显存 | 推荐 Workers | 说明 |
|----------|-------------|------|
| 24GB (RTX 3090) | 4-5 | 每个 worker 约 5-6GB |
| 16GB (RTX 4080) | 3-4 | 每个 worker 约 5-6GB |
| 12GB (RTX 3060) | 2 | 每个 worker 约 5-6GB |
| 8GB | 1 | 单个 worker |

### 修改配置

```bash
# 减少 Worker 数量以节省显存
GUNICORN_WORKERS=2 ./docker-run.sh

# 增加 Worker 数量以提高并发（需要足够显存）
GUNICORN_WORKERS=6 ./docker-run.sh
```

## API 使用

### 健康检查

```bash
curl http://localhost:6156/health
```

### 推理接口

```bash
curl -X POST http://localhost:6156/semantic-segmentation \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "base64_encoded_image_string",
    "text_prompt": "road surface.",
    "box_threshold": 0.1,
    "text_threshold": 0.25,
    "epsilon": 1.0
  }'
```

### 查看服务信息

```bash
curl http://localhost:6156/
curl http://localhost:6156/metrics
curl http://localhost:6156/stats
```

## 注意事项

1. **首次启动**: 模型加载需要 30-60 秒，请耐心等待
2. **GPU 显存**: 确保有足够的 GPU 显存（建议至少 12GB）
3. **端口冲突**: 如果端口被占用，使用 `PORT` 环境变量指定其他端口
4. **模型文件**: 确保模型文件路径正确且可读
5. **日志目录**: `flask-server/logs/` 目录会自动创建
6. **结果目录**: `flask-server/results/` 目录会自动创建

## 相关文件

- `Dockerfile.server`: Docker 镜像构建文件
- `docker-build.sh`: 构建脚本
- `docker-run.sh`: 运行脚本
- `docker-test.sh`: 测试脚本
- `docker-debug.sh`: 诊断脚本

## 获取帮助

如果遇到问题：

1. 查看日志: `docker logs -f grounded-sam2-server`
2. 运行诊断: `./docker-debug.sh`
3. 检查容器状态: `docker ps -a | grep grounded-sam2-server`
4. 查看资源使用: `docker stats grounded-sam2-server`

---

**最后更新**: 2024年11月
**版本**: v1.1

