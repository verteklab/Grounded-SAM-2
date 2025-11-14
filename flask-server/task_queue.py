"""
任务队列模块
实现解码队列和推理队列，支持Future模式
"""

import queue
import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional, Any, Callable
from concurrent.futures import Future
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"  # 等待处理
    DECODING = "decoding"  # 正在解码
    PREPROCESSING = "preprocessing"  # 正在预处理
    INFERRING = "inferring"  # 正在推理
    POSTPROCESSING = "postprocessing"  # 正在后处理
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    TIMEOUT = "timeout"  # 超时


@dataclass
class InferenceTask:
    """推理任务数据类"""
    request_id: str
    image_base64: str
    text_prompt: str
    box_threshold: float
    text_threshold: float
    epsilon: float
    
    # 任务状态
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # 中间结果
    decoded_image: Optional[Any] = None  # (image_source, image) tuple
    preprocessed_data: Optional[dict] = None
    inference_result: Optional[dict] = None
    
    # Future对象（用于异步返回结果）
    future: Optional[Future] = None
    
    # 错误信息
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.future is None:
            self.future = Future()
    
    def get_age(self) -> float:
        """获取任务年龄（秒）"""
        return time.time() - self.created_at
    
    def is_timeout(self, timeout: float) -> bool:
        """检查任务是否超时"""
        return self.get_age() > timeout


class TaskQueue:
    """任务队列基类"""
    
    def __init__(self, name: str, maxsize: int = 0):
        """
        初始化任务队列
        
        Args:
            name: 队列名称（用于日志）
            maxsize: 队列最大长度（0表示无界）
        """
        self.name = name
        self.queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._stats = {
            'total_put': 0,
            'total_get': 0,
            'total_rejected': 0,
            'max_size': 0,
            'current_size': 0
        }
    
    def put(self, task: InferenceTask, timeout: Optional[float] = None) -> bool:
        """
        将任务放入队列
        
        Args:
            task: 推理任务
            timeout: 超时时间（秒），None表示阻塞直到成功
        
        Returns:
            True表示成功，False表示超时或队列满
        """
        try:
            self.queue.put(task, timeout=timeout)
            with self._lock:
                self._stats['total_put'] += 1
                self._stats['current_size'] = self.queue.qsize()
                self._stats['max_size'] = max(self._stats['max_size'], self._stats['current_size'])
            logger.debug(f"[{self.name}] 任务入队: {task.request_id}, 队列长度: {self._stats['current_size']}")
            return True
        except queue.Full:
            with self._lock:
                self._stats['total_rejected'] += 1
            logger.warning(f"[{self.name}] 队列已满，拒绝任务: {task.request_id}")
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[InferenceTask]:
        """
        从队列获取任务
        
        Args:
            timeout: 超时时间（秒），None表示阻塞直到有任务
        
        Returns:
            任务对象，超时返回None
        """
        try:
            task = self.queue.get(timeout=timeout)
            with self._lock:
                self._stats['total_get'] += 1
                self._stats['current_size'] = self.queue.qsize()
            logger.debug(f"[{self.name}] 任务出队: {task.request_id}, 队列长度: {self._stats['current_size']}")
            return task
        except queue.Empty:
            return None
    
    def task_done(self):
        """标记任务完成"""
        self.queue.task_done()
    
    def qsize(self) -> int:
        """获取队列当前长度"""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """检查队列是否为空"""
        return self.queue.empty()
    
    def full(self) -> bool:
        """检查队列是否已满"""
        return self.queue.full()
    
    def get_stats(self) -> dict:
        """获取队列统计信息"""
        with self._lock:
            return {
                **self._stats,
                'current_size': self.queue.qsize()
            }


class DecodeQueue(TaskQueue):
    """解码队列：存储待解码的base64图像"""
    
    def __init__(self, maxsize: int = 100):
        super().__init__("DecodeQueue", maxsize=maxsize)


class InferenceQueue(TaskQueue):
    """推理队列：存储待推理的预处理数据"""
    
    def __init__(self, maxsize: int = 50):
        super().__init__("InferenceQueue", maxsize=maxsize)

