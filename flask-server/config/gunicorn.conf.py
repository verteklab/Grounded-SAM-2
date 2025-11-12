# 生产环境Gunicorn配置
# 
# 系统资源分析：
# - CPU: 192核心（AMD EPYC 7K62），可用96核心
# - GPU: RTX 3090 (24GB显存)
# - 实际测量：每个worker占用约5-6GB显存（包括模型和推理临时显存）
# 
# 优化策略：
# 1. Workers数量：根据实际GPU显存占用计算
#    计算公式: workers = (GPU显存 - 系统预留) / 每个worker显存占用
#    RTX 3090 (24GB): (24 - 2) / 5.5 ≈ 4个workers（保守）
#    考虑到推理时的临时显存峰值和内存碎片，建议设置为3-4个workers
# 
# 2. Threads数量：由于模型锁的存在，每个worker内的推理是串行的
#    threads主要用于：接收请求、I/O处理、等待模型锁
#    推荐：2-4个threads/worker，既能处理并发请求，又不会浪费资源
# 
# 3. 总并发能力：workers × threads = 实际并发处理数
#    但受模型锁限制，真正的并行推理数 = workers数量
# 
# 4. 显存管理：已添加显存清理机制，防止内存泄漏
# 5. GPU设备选择：可以通过CUDA_VISIBLE_DEVICES或gpu_device_id指定

# GPU设备配置
# 方式1: 使用CUDA_VISIBLE_DEVICES环境变量（推荐）
# 方式2: 使用gpu_device_id配置（在post_fork中设置）
# 如果系统有多个GPU，可以指定使用哪个GPU
# 例如: gpu_device_id = 0 表示使用GPU 0, gpu_device_id = 1 表示使用GPU 1
gpu_device_id = 1  # 默认使用GPU 0，可以修改为0, 1, 2等

# 工作进程数：根据实际GPU显存占用调整（RTX 3090 24GB）
# 实际测量：每个worker占用约5-6GB显存
# 计算公式: (24GB - 2GB系统预留) / 5.5GB ≈ 4个workers
# 保守配置：3个workers（预留7GB显存缓冲，更安全）
# 平衡配置：4个workers（预留2GB，推荐）
workers = 6  # RTX 3090实际占用5-6GB/worker，推荐3-4个workers

# 每个进程线程数
# 由于模型锁的存在，每个worker内的推理是串行的
# threads主要用于接收新请求和处理I/O，推荐2-4个
threads = 3  # 每个worker 3个线程，用于处理请求队列和I/O

# 请求超时时间（模型推理可能需要较长时间）
timeout = 300

# CUDA与fork()不兼容，不能使用preload_app
# 每个worker进程需要独立加载模型
# 注意：这会导致每个进程都占用GPU内存，总内存 = 模型大小 × worker数量
preload_app = False

# 绑定地址和端口
bind = "0.0.0.0:6155"

# 守护进程模式（后台运行）
daemon = True

# 进程名称
proc_name = "grounded-sam2-server"

# 日志配置
loglevel = "info"
accesslog = "logs/access.log"
errorlog = "logs/error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 防止内存泄漏：每个进程处理一定数量请求后自动重启
# 对于深度学习模型，建议设置较低的值以防止内存泄漏
# 由于已添加显存清理机制，可以适当提高此值
max_requests = 200  # 降低到200，更频繁地重启worker以释放累积的内存碎片
max_requests_jitter = 20  # 随机抖动，避免所有worker同时重启

# 优雅超时设置
graceful_timeout = 60

# Worker启动后回调：在每个worker进程中加载模型
def on_starting(server):
    """主进程启动时调用"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("🚀 Gunicorn主进程启动")

def post_fork(server, worker):
    """每个worker进程fork后调用 - 在这里加载模型"""
    import logging
    import os
    import torch
    
    logger = logging.getLogger(__name__)
    logger.info(f"🔄 Worker进程 {os.getpid()} 启动，正在加载模型...")
    
    # 配置GPU设备
    # 如果环境变量CUDA_VISIBLE_DEVICES已设置，优先使用环境变量
    # 否则使用配置中的gpu_device_id
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        # 从配置文件中读取GPU设备ID
        try:
            # 读取配置文件中的gpu_device_id
            # 注意：这里需要访问当前模块的全局变量
            import importlib
            import sys
            # 获取当前配置模块
            current_module = sys.modules[__name__]
            if hasattr(current_module, 'gpu_device_id'):
                gpu_id = current_module.gpu_device_id
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                logger.info(f"📌 从配置文件读取GPU设备: {gpu_id} (设置CUDA_VISIBLE_DEVICES={gpu_id})")
            else:
                logger.warning("⚠️ 配置文件中未找到gpu_device_id，使用默认GPU 0")
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        except Exception as e:
            logger.warning(f"⚠️ 无法从配置文件读取GPU设备，使用默认GPU 0: {e}")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        logger.info(f"📌 使用环境变量指定的GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    # 配置PyTorch显存管理，减少内存碎片
    # expandable_segments: 允许PyTorch动态扩展内存段，减少碎片
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    
    # 在每个worker进程中加载模型
    from model_manager import model_manager
    try:
        model_manager.load_models()
        
        # 记录初始显存使用情况和GPU信息
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
            gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
            logger.info(f"✅ Worker进程 {os.getpid()} 模型加载完成")
            logger.info(f"   GPU设备: {gpu_id} ({gpu_name})")
            logger.info(f"   初始显存占用: {initial_memory:.2f} GB")
        else:
            logger.info(f"✅ Worker进程 {os.getpid()} 模型加载完成（使用CPU）")
    except Exception as e:
        logger.error(f"❌ Worker进程 {os.getpid()} 模型加载失败: {e}")
        raise
