"""
工作线程池配置模块
提供线程池相关的配置参数和Feature Toggle
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThreadPoolConfig:
    """线程池配置类"""
    # Feature Toggle: 是否启用线程池模式
    enable_thread_pool: bool = True
    
    # 工作线程数量配置
    decode_threads: int = 2  # 解码线程数（I/O密集型）
    preprocess_threads: int = 1  # 预处理线程数（CPU密集型）
    postprocess_threads: int = 1  # 后处理线程数（轻量级）
    
    # 队列配置
    decode_queue_maxsize: int = 100  # 解码队列最大长度（0表示无界）
    inference_queue_maxsize: int = 50  # 推理队列最大长度（0表示无界）
    
    # 动态扩缩容配置
    enable_auto_scaling: bool = True
    scale_up_threshold: int = 10  # 队列长度超过此值触发扩容
    scale_down_threshold: int = 2  # 队列长度低于此值触发缩容
    scale_down_idle_seconds: int = 60  # 线程空闲超过此时间触发缩容
    max_total_threads: int = 8  # 最大总线程数（包括解码+预处理+后处理）
    
    # 超时配置
    task_timeout: float = 300.0  # 任务超时时间（秒）
    queue_wait_timeout: float = 5.0  # 队列等待超时（秒）
    
    # 监控配置
    metrics_collection_interval: float = 5.0  # 指标收集间隔（秒）
    
    @classmethod
    def from_env(cls) -> 'ThreadPoolConfig':
        """从环境变量加载配置"""
        return cls(
            enable_thread_pool=os.getenv('ENABLE_THREAD_POOL', 'true').lower() == 'true',
            decode_threads=int(os.getenv('DECODE_THREADS', '2')),
            preprocess_threads=int(os.getenv('PREPROCESS_THREADS', '1')),
            postprocess_threads=int(os.getenv('POSTPROCESS_THREADS', '1')),
            decode_queue_maxsize=int(os.getenv('DECODE_QUEUE_MAXSIZE', '100')),
            inference_queue_maxsize=int(os.getenv('INFERENCE_QUEUE_MAXSIZE', '50')),
            enable_auto_scaling=os.getenv('ENABLE_AUTO_SCALING', 'true').lower() == 'true',
            scale_up_threshold=int(os.getenv('SCALE_UP_THRESHOLD', '10')),
            scale_down_threshold=int(os.getenv('SCALE_DOWN_THRESHOLD', '2')),
            scale_down_idle_seconds=int(os.getenv('SCALE_DOWN_IDLE_SECONDS', '60')),
            max_total_threads=int(os.getenv('MAX_TOTAL_THREADS', '8')),
            task_timeout=float(os.getenv('TASK_TIMEOUT', '300.0')),
            queue_wait_timeout=float(os.getenv('QUEUE_WAIT_TIMEOUT', '5.0')),
            metrics_collection_interval=float(os.getenv('METRICS_COLLECTION_INTERVAL', '5.0'))
        )
    
    def get_total_worker_threads(self) -> int:
        """获取总工作线程数"""
        return self.decode_threads + self.preprocess_threads + self.postprocess_threads
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """验证配置有效性"""
        if self.decode_threads < 1:
            return False, "decode_threads must be >= 1"
        if self.preprocess_threads < 1:
            return False, "preprocess_threads must be >= 1"
        if self.postprocess_threads < 1:
            return False, "postprocess_threads must be >= 1"
        if self.max_total_threads < self.get_total_worker_threads():
            return False, f"max_total_threads ({self.max_total_threads}) must be >= total worker threads ({self.get_total_worker_threads()})"
        if self.scale_up_threshold <= self.scale_down_threshold:
            return False, "scale_up_threshold must be > scale_down_threshold"
        return True, None


# 全局配置实例
_pool_config: Optional[ThreadPoolConfig] = None


def get_pool_config() -> ThreadPoolConfig:
    """获取线程池配置（单例模式）"""
    global _pool_config
    if _pool_config is None:
        _pool_config = ThreadPoolConfig.from_env()
        is_valid, error_msg = _pool_config.validate()
        if not is_valid:
            raise ValueError(f"Invalid thread pool config: {error_msg}")
    return _pool_config


def reset_pool_config():
    """重置配置（用于测试）"""
    global _pool_config
    _pool_config = None

