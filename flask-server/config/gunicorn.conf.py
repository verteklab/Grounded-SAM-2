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

# GPU设备配置 - 单GPU模式
# 使用CUDA_VISIBLE_DEVICES环境变量指定GPU设备
import os

# 单GPU配置
gpu_device_id = int(os.getenv('GPU_DEVICE_ID', '0'))  # 支持环境变量覆盖

# 工作进程数：支持环境变量覆盖，默认3个workers
# 计算公式: workers = (GPU显存 - 系统预留) / 每个worker显存占用
# RTX 3090 (24GB): (24 - 2) / 5.5 ≈ 4个workers（保守）
# 考虑到推理时的临时显存峰值和内存碎片，建议设置为3-4个workers
workers = int(os.getenv('GUNICORN_WORKERS', '5'))  # 支持环境变量覆盖

# 每个进程线程数
# 由于模型锁的存在，每个worker内的推理是串行的
# threads主要用于接收新请求和处理I/O，推荐2-4个
# 
# 注意：Gunicorn会自动在workers之间轮询分配请求，实现worker级别的负载均衡
# - sync worker (threads=1): 每个worker一个线程，Gunicorn在workers之间轮询分配请求
# - gthread worker (threads>1): 每个worker多个线程，Gunicorn在workers之间分配请求，worker内部使用线程池
# 两种方式都会在workers之间自动负载均衡，无需额外配置
threads = int(os.getenv('GUNICORN_THREADS', '3'))  # 支持环境变量覆盖

# 请求超时时间（模型推理可能需要较长时间）
timeout = int(os.getenv('GUNICORN_TIMEOUT', '300'))  # 支持环境变量覆盖

# CUDA与fork()不兼容，不能使用preload_app
# 每个worker进程需要独立加载模型
# 注意：这会导致每个进程都占用GPU内存，总内存 = 模型大小 × worker数量
preload_app = False

# 绑定地址和端口
bind = os.getenv('GUNICORN_BIND', '0.0.0.0:6155')  # 支持环境变量覆盖

# 守护进程模式（后台运行）
# Docker中必须设为False，否则容器会立即退出
daemon = os.getenv('GUNICORN_DAEMON', 'False').lower() == 'true'  # 支持环境变量覆盖

# 进程名称
proc_name = "grounded-sam2-server"

# 日志配置
loglevel = os.getenv('GUNICORN_LOGLEVEL', 'info')  # 支持环境变量覆盖
accesslog = "logs/access.log"
errorlog = "logs/error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 防止内存泄漏：每个进程处理一定数量请求后自动重启
# 对于深度学习模型，建议设置较低的值以防止内存泄漏
# 由于已添加显存清理机制，可以适当提高此值
# 设置为0表示不限制（不自动重启），但需要确保没有内存泄漏
# 如果遇到内存泄漏问题，可以设置为1000-5000之间的值
max_requests = int(os.getenv('GUNICORN_MAX_REQUESTS', '0'))  # 0表示不限制，支持环境变量覆盖
max_requests_jitter = int(os.getenv('GUNICORN_MAX_REQUESTS_JITTER', '0'))  # 支持环境变量覆盖

# 优雅超时设置
graceful_timeout = int(os.getenv('GUNICORN_GRACEFUL_TIMEOUT', '60'))  # 支持环境变量覆盖

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
    worker_id = os.getpid()  # 使用进程ID作为worker标识
    logger.info(f"🔄 Worker进程 {worker_id} 启动，正在加载模型...")
    
    # 配置GPU设备 - 单GPU模式
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        # 从配置文件中读取GPU设备ID
        try:
            # 读取配置文件中的gpu_device_id
            import importlib
            import sys
            current_module = sys.modules[__name__]
            if hasattr(current_module, 'gpu_device_id'):
                gpu_id = current_module.gpu_device_id
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                logger.info(f"📌 从配置文件读取GPU设备: {gpu_id}")
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
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"✅✅✅ Worker进程 {os.getpid()} 模型加载成功！✅✅✅")
            logger.info(f"   GPU设备: {gpu_id} ({gpu_name})")
            logger.info(f"   初始显存占用: {initial_memory:.2f} GB")
            logger.info(f"   模型状态: SAM2 ✓ | GroundingDINO ✓")
            logger.info("=" * 60)
            logger.info("")
        else:
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"✅✅✅ Worker进程 {os.getpid()} 模型加载成功（使用CPU）✅✅✅")
            logger.info(f"   模型状态: SAM2 ✓ | GroundingDINO ✓")
            logger.info("=" * 60)
            logger.info("")
        
        # 启动线程池（如果启用）
        try:
            from pool_config import get_pool_config
            from thread_pool_manager import get_thread_pool_manager
            
            config = get_pool_config()
            if config.enable_thread_pool:
                thread_pool = get_thread_pool_manager()
                thread_pool.set_model_manager(model_manager)
                thread_pool.start()
                model_manager.enable_thread_pool(thread_pool)
                logger.info(f"✅ Worker进程 {os.getpid()} 线程池启动完成")
            else:
                logger.info(f"ℹ️  Worker进程 {os.getpid()} 线程池未启用（使用传统锁模式）")
        except Exception as e:
            logger.warning(f"⚠️  Worker进程 {os.getpid()} 线程池启动失败，回退到锁模式: {e}")
            # 不抛出异常，允许回退到锁模式
        
    except Exception as e:
        logger.error(f"❌ Worker进程 {os.getpid()} 模型加载失败: {e}")
        raise

def worker_exit(server, worker):
    """Worker进程退出时调用 - 用于清理资源和记录日志"""
    import logging
    import os
    import torch
    
    logger = logging.getLogger(__name__)
    pid = os.getpid()
    
    # 记录退出信息
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"🔄 Worker进程 {pid} 正在退出...")
    
    # 记录退出时的资源使用情况
    try:
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.info(f"   退出时显存占用: {final_memory:.2f} GB")
        
        # 记录进程统计信息
        try:
            import psutil
            process = psutil.Process(pid)
            mem_info = process.memory_info()
            logger.info(f"   退出时内存占用: {mem_info.rss / 1024**2:.1f} MB")
        except:
            pass
    except Exception as e:
        logger.warning(f"   无法获取退出时资源信息: {e}")
    
    # 清理线程池（如果启用）
    try:
        from thread_pool_manager import get_thread_pool_manager
        thread_pool = get_thread_pool_manager()
        if thread_pool and thread_pool._running:
            logger.info(f"   正在关闭线程池...")
            thread_pool.shutdown(timeout=10.0)
    except Exception as e:
        logger.warning(f"   线程池关闭失败: {e}")
    
    # 清理GPU显存
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass
    
    logger.info(f"✅ Worker进程 {pid} 退出完成")
    logger.info("=" * 60)
    logger.info("")

def worker_abort(worker):
    """Worker进程异常退出时调用"""
    import logging
    import os
    
    logger = logging.getLogger(__name__)
    pid = os.getpid()
    logger.error("")
    logger.error("=" * 60)
    logger.error(f"❌❌❌ Worker进程 {pid} 异常退出！❌❌❌")
    logger.error("   这可能是由于：")
    logger.error("   1. 内存不足（OOM）")
    logger.error("   2. 显存不足")
    logger.error("   3. 未捕获的异常")
    logger.error("   4. 系统信号（SIGKILL/SIGTERM）")
    logger.error("=" * 60)
    logger.error("")
