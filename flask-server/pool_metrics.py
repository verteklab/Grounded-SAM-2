"""
线程池监控指标模块
提供Prometheus风格的指标收集和暴露
"""

import time
import threading
import logging
from typing import Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class PoolMetrics:
    """线程池指标收集器"""
    
    _instance: Optional['PoolMetrics'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._lock = threading.Lock()
        self._metrics = {
            # 任务统计
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_timeout': 0,
            'tasks_rejected': 0,
            
            # 延迟统计
            'decode_latency_sum': 0.0,
            'preprocess_latency_sum': 0.0,
            'inference_latency_sum': 0.0,
            'total_latency_sum': 0.0,
            'decode_latency_count': 0,
            'preprocess_latency_count': 0,
            'inference_latency_count': 0,
            'total_latency_count': 0,
            
            # 队列统计
            'decode_queue_max_size': 0,
            'inference_queue_max_size': 0,
            'decode_queue_current_size': 0,
            'inference_queue_current_size': 0,
            
            # 线程统计
            'decode_threads_active': 0,
            'preprocess_threads_active': 0,
            
            # 时间戳
            'last_update_time': time.time()
        }
        
        self._initialized = True
    
    def record_task_submitted(self):
        """记录任务提交"""
        with self._lock:
            self._metrics['tasks_submitted'] += 1
    
    def record_task_completed(self, total_latency: float, decode_latency: float = 0.0, 
                             preprocess_latency: float = 0.0, inference_latency: float = 0.0):
        """记录任务完成"""
        with self._lock:
            self._metrics['tasks_completed'] += 1
            self._metrics['total_latency_sum'] += total_latency
            self._metrics['total_latency_count'] += 1
            if decode_latency > 0:
                self._metrics['decode_latency_sum'] += decode_latency
                self._metrics['decode_latency_count'] += 1
            if preprocess_latency > 0:
                self._metrics['preprocess_latency_sum'] += preprocess_latency
                self._metrics['preprocess_latency_count'] += 1
            if inference_latency > 0:
                self._metrics['inference_latency_sum'] += inference_latency
                self._metrics['inference_latency_count'] += 1
    
    def record_task_failed(self):
        """记录任务失败"""
        with self._lock:
            self._metrics['tasks_failed'] += 1
    
    def record_task_timeout(self):
        """记录任务超时"""
        with self._lock:
            self._metrics['tasks_timeout'] += 1
    
    def record_task_rejected(self):
        """记录任务被拒绝"""
        with self._lock:
            self._metrics['tasks_rejected'] += 1
    
    def update_queue_stats(self, decode_queue_size: int, inference_queue_size: int):
        """更新队列统计"""
        with self._lock:
            self._metrics['decode_queue_current_size'] = decode_queue_size
            self._metrics['inference_queue_current_size'] = inference_queue_size
            self._metrics['decode_queue_max_size'] = max(
                self._metrics['decode_queue_max_size'], 
                decode_queue_size
            )
            self._metrics['inference_queue_max_size'] = max(
                self._metrics['inference_queue_max_size'],
                inference_queue_size
            )
    
    def update_thread_stats(self, decode_active: int, preprocess_active: int):
        """更新线程统计"""
        with self._lock:
            self._metrics['decode_threads_active'] = decode_active
            self._metrics['preprocess_threads_active'] = preprocess_active
    
    def get_metrics(self) -> Dict:
        """获取所有指标"""
        with self._lock:
            metrics = self._metrics.copy()
            
            # 计算平均值
            if metrics['total_latency_count'] > 0:
                metrics['avg_total_latency'] = metrics['total_latency_sum'] / metrics['total_latency_count']
            else:
                metrics['avg_total_latency'] = 0.0
            
            if metrics['decode_latency_count'] > 0:
                metrics['avg_decode_latency'] = metrics['decode_latency_sum'] / metrics['decode_latency_count']
            else:
                metrics['avg_decode_latency'] = 0.0
            
            if metrics['preprocess_latency_count'] > 0:
                metrics['avg_preprocess_latency'] = metrics['preprocess_latency_sum'] / metrics['preprocess_latency_count']
            else:
                metrics['avg_preprocess_latency'] = 0.0
            
            if metrics['inference_latency_count'] > 0:
                metrics['avg_inference_latency'] = metrics['inference_latency_sum'] / metrics['inference_latency_count']
            else:
                metrics['avg_inference_latency'] = 0.0
            
            # 计算成功率
            total = metrics['tasks_submitted']
            if total > 0:
                metrics['success_rate'] = metrics['tasks_completed'] / total
                metrics['failure_rate'] = metrics['tasks_failed'] / total
                metrics['timeout_rate'] = metrics['tasks_timeout'] / total
                metrics['rejection_rate'] = metrics['tasks_rejected'] / total
            else:
                metrics['success_rate'] = 0.0
                metrics['failure_rate'] = 0.0
                metrics['timeout_rate'] = 0.0
                metrics['rejection_rate'] = 0.0
            
            metrics['last_update_time'] = time.time()
            return metrics
    
    def reset(self):
        """重置所有指标"""
        with self._lock:
            self.__init__()


# 全局指标实例
_metrics_instance: Optional[PoolMetrics] = None


def get_metrics() -> PoolMetrics:
    """获取指标实例（单例）"""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = PoolMetrics()
    return _metrics_instance

