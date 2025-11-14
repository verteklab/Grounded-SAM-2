"""
工作线程池管理器
负责管理消息队列分发线程、解码线程、推理线程等
架构：请求 -> 消息队列 -> 空闲线程 -> 模型实例 -> 直接返回结果
"""

import threading
import time
import logging
import os
import queue
from typing import Optional, List
from concurrent.futures import Future

from pool_config import get_pool_config, ThreadPoolConfig
from task_queue import InferenceTask, TaskStatus

logger = logging.getLogger(__name__)


class ThreadPoolManager:
    """工作线程池管理器（单例模式，每个Worker进程一个实例）"""
    
    _instance: Optional['ThreadPoolManager'] = None
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
        
        self.config = get_pool_config()
        self.pid = os.getpid()
        
        # 队列架构：消息队列（解码前） -> 推理队列（解码后）
        self.message_queue = queue.Queue(maxsize=self.config.decode_queue_maxsize)  # 待解码任务队列
        self.inference_queue = queue.Queue(maxsize=self.config.inference_queue_maxsize)  # 待推理任务队列
        
        # 线程列表
        self.decode_threads: List[threading.Thread] = []  # 解码线程（I/O密集型，可以并行）
        self.inference_worker_threads: List[threading.Thread] = []  # 推理工作线程（GPU密集型）
        
        # 控制标志
        self._running = False
        self._shutdown_event = threading.Event()
        
        # 模型管理器引用（延迟注入，避免循环导入）
        self._model_manager = None
        
        # 线程统计
        self._thread_stats = {
            'message_queue_size': 0,
            'inference_queue_size': 0,
            'decode_threads_active': 0,
            'inference_workers_active': 0
        }
        
        # 线程同步
        self._stats_lock = threading.Lock()
        
        self._initialized = True
        logger.info(f"[ThreadPool] 线程池管理器初始化完成 (PID={self.pid})")
    
    def set_model_manager(self, model_manager):
        """设置模型管理器（延迟注入）"""
        self._model_manager = model_manager
    
    def start(self):
        """启动线程池"""
        if self._running:
            logger.warning(f"[ThreadPool] 线程池已在运行 (PID={self.pid})")
            return
        
        self._running = True
        self._shutdown_event.clear()
        
        logger.info(f"[ThreadPool] 启动线程池 (PID={self.pid})")
        logger.info(f"  配置: 解码线程={self.config.decode_threads}, 推理工作线程={self.config.preprocess_threads}")
        logger.info(f"  架构: 请求 -> 消息队列 -> 解码线程(并行解码) -> 推理队列 -> 推理线程(并行推理) -> 返回结果")
        
        # 启动解码线程（从消息队列取任务，解码后放入推理队列）
        for i in range(self.config.decode_threads):
            thread = threading.Thread(
                target=self._decode_worker,
                name=f"DecodeWorker-{i}",
                daemon=True
            )
            thread.start()
            self.decode_threads.append(thread)
            logger.info(f"[ThreadPool] 启动解码线程: {thread.name}")
        
        # 启动推理工作线程（从推理队列取任务，执行推理）
        for i in range(self.config.preprocess_threads):
            thread = threading.Thread(
                target=self._inference_worker,
                name=f"InferenceWorker-{i}",
                daemon=True
            )
            thread.start()
            self.inference_worker_threads.append(thread)
            logger.info(f"[ThreadPool] 启动推理工作线程: {thread.name}")
        
        logger.info(f"[ThreadPool] 线程池启动完成 (PID={self.pid})")
    
    def shutdown(self, timeout: float = 30.0):
        """关闭线程池"""
        if not self._running:
            return
        
        logger.info(f"[ThreadPool] 开始关闭线程池 (PID={self.pid})")
        self._running = False
        self._shutdown_event.set()
        
        # 等待所有线程结束
        all_threads = self.decode_threads + self.inference_worker_threads
        
        for thread in all_threads:
            if thread and thread.is_alive():
                thread.join(timeout=timeout / len(all_threads))
        
        logger.info(f"[ThreadPool] 线程池关闭完成 (PID={self.pid})")
    
    def submit_task(self, task: InferenceTask) -> Future:
        """
        提交推理任务到消息队列
        
        Args:
            task: 推理任务（包含request_id, image_base64等）
        
        Returns:
            Future对象，用于获取结果
        """
        if not self._running:
            raise RuntimeError("线程池未启动")
        
        # 记录指标
        try:
            from pool_metrics import get_metrics
            get_metrics().record_task_submitted()
        except:
            pass
        
        # 将任务放入消息队列（统一入口）
        try:
            self.message_queue.put(task, timeout=self.config.queue_wait_timeout)
            with self._stats_lock:
                self._thread_stats['message_queue_size'] = self.message_queue.qsize()
            task.status = TaskStatus.PENDING
            logger.info(f"[ThreadPool] 任务已提交到消息队列: {task.request_id}, 队列长度: {self._thread_stats['message_queue_size']}")
        except queue.Full:
            error_msg = "消息队列已满，任务被拒绝"
            logger.error(f"[ThreadPool] {error_msg}: {task.request_id}")
            task.error = error_msg
            task.status = TaskStatus.FAILED
            task.future.set_exception(RuntimeError(error_msg))
            # 记录拒绝指标
            try:
                from pool_metrics import get_metrics
                get_metrics().record_task_rejected()
            except:
                pass
            return task.future
        
        return task.future
    
    def _decode_worker(self):
        """解码工作线程：从消息队列取任务，进行base64解码，然后放入推理队列"""
        thread_name = threading.current_thread().name
        logger.info(f"[ThreadPool] 解码工作线程启动: {thread_name}")
        
        while self._running:
            try:
                # 从消息队列获取任务（带超时，便于检查shutdown事件）
                try:
                    task = self.message_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if task is None:
                    continue
                
                # 检查任务是否超时
                if task.is_timeout(self.config.task_timeout):
                    logger.warning(f"[ThreadPool] 任务超时（解码阶段）: {task.request_id}")
                    task.status = TaskStatus.TIMEOUT
                    task.error = "任务在解码阶段超时"
                    task.future.set_exception(TimeoutError(task.error))
                    continue
                
                task.status = TaskStatus.DECODING
                task.started_at = time.time()
                
                with self._stats_lock:
                    self._thread_stats['decode_threads_active'] += 1
                    self._thread_stats['message_queue_size'] = self.message_queue.qsize()
                
                try:
                    # 执行解码
                    from grounding_dino.groundingdino.util.inference import load_image_from_base64
                    
                    decode_start = time.time()
                    image_source, image = load_image_from_base64(task.image_base64)
                    decode_time = time.time() - decode_start
                    
                    # 将解码结果保存到任务中
                    task.preprocessed_data = {
                        'image_source': image_source,
                        'image': image,
                        'decode_time': decode_time
                    }
                    
                    logger.debug(
                        f"[ThreadPool] [{task.request_id}] 解码完成 | "
                        f"Size={image_source.shape[1]}x{image_source.shape[0]} | "
                        f"Duration={decode_time:.3f}s | "
                        f"Worker={thread_name}"
                    )
                    
                    # 放入推理队列
                    try:
                        self.inference_queue.put(task, timeout=self.config.queue_wait_timeout)
                    except queue.Full:
                        error_msg = "推理队列已满，任务被拒绝"
                        logger.error(f"[ThreadPool] [{task.request_id}] {error_msg}")
                        task.error = error_msg
                        task.status = TaskStatus.FAILED
                        task.future.set_exception(RuntimeError(error_msg))
                    
                except Exception as e:
                    error_msg = f"解码失败: {str(e)}"
                    logger.error(f"[ThreadPool] [{task.request_id}] {error_msg}", exc_info=True)
                    task.error = error_msg
                    task.status = TaskStatus.FAILED
                    task.future.set_exception(e)
                
                finally:
                    with self._stats_lock:
                        self._thread_stats['decode_threads_active'] = max(
                            self._thread_stats['decode_threads_active'] - 1, 0
                        )
                        self._thread_stats['message_queue_size'] = self.message_queue.qsize()
            
            except Exception as e:
                if self._running:
                    logger.error(f"[ThreadPool] 解码工作线程异常: {thread_name}", exc_info=True)
                time.sleep(0.1)
        
        logger.info(f"[ThreadPool] 解码工作线程退出: {thread_name}")
    
    def _inference_worker(self):
        """推理工作线程：从推理队列取任务，执行模型推理，然后直接返回结果"""
        thread_name = threading.current_thread().name
        worker_id = int(thread_name.split('-')[-1]) if '-' in thread_name else 0
        logger.info(f"[ThreadPool] 推理工作线程启动: {thread_name} (WorkerID={worker_id})")
        
        if self._model_manager is None:
            logger.error(f"[ThreadPool] 模型管理器未设置，推理工作线程无法工作")
            return
        
        while self._running:
            try:
                # 从推理队列获取任务（工作线程直接竞争获取，实现负载均衡）
                try:
                    task = self.inference_queue.get(timeout=1.0)
                except queue.Empty:
                    with self._stats_lock:
                        self._thread_stats['inference_queue_size'] = self.inference_queue.qsize()
                    continue
                
                if task is None:
                    continue
                
                # 检查任务是否超时
                if task.is_timeout(self.config.task_timeout):
                    logger.warning(f"[ThreadPool] 任务超时（推理阶段）: {task.request_id}")
                    task.status = TaskStatus.TIMEOUT
                    task.error = "任务在推理阶段超时"
                    task.future.set_exception(TimeoutError(task.error))
                    self.inference_queue.task_done()
                    continue
                
                # 标记工作线程为活跃
                with self._stats_lock:
                    self._thread_stats['inference_workers_active'] += 1
                    self._thread_stats['inference_queue_size'] = self.inference_queue.qsize()
                
                task.status = TaskStatus.INFERRING
                
                try:
                    # 获取已解码的图像数据
                    preprocessed_data = task.preprocessed_data
                    if preprocessed_data is None:
                        raise RuntimeError(f"任务 {task.request_id} 的预处理数据为空，解码可能失败")
                    
                    image_source = preprocessed_data['image_source']
                    image = preprocessed_data['image']
                    decode_time = preprocessed_data.get('decode_time', 0.0)
                    
                    # 执行推理（使用模型管理器的推理方法，内部有锁保护）
                    logger.debug(f"[ThreadPool] [{task.request_id}] 开始推理: {thread_name}")
                    inference_start = time.time()
                    
                    try:
                        result = self._model_manager._inference_without_lock(
                            image_source=image_source,
                            image=image,
                            text_prompt=task.text_prompt,
                            box_threshold=task.box_threshold,
                            text_threshold=task.text_threshold,
                            epsilon=task.epsilon,
                            request_id=task.request_id  # 确保请求ID正确传递
                        )
                        inference_time = time.time() - inference_start
                    except Exception as inference_error:
                        inference_time = time.time() - inference_start
                        logger.error(f"[ThreadPool] [{task.request_id}] 推理过程异常: {inference_error}", exc_info=True)
                        raise  # 重新抛出异常，让外层catch处理
                    
                    # 验证结果有效性
                    if result is None:
                        error_msg = f"推理返回None，request_id={task.request_id}"
                        logger.error(f"[ThreadPool] [{task.request_id}] {error_msg}")
                        raise RuntimeError(error_msg)
                    
                    if not isinstance(result, dict):
                        error_msg = f"推理返回类型错误，期望dict，得到{type(result)}，request_id={task.request_id}"
                        logger.error(f"[ThreadPool] [{task.request_id}] {error_msg}")
                        raise RuntimeError(error_msg)
                    
                    # 确保结果包含请求ID（双重保险）
                    result['request_id'] = task.request_id
                    
                    # 验证结果基本字段
                    if 'status' not in result:
                        result['status'] = 'success'
                    if 'count' not in result:
                        result['count'] = result.get('results', [])
                        if isinstance(result['count'], list):
                            result['count'] = len(result['count'])
                    
                    task.inference_result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                    
                    total_latency = task.completed_at - task.created_at
                    
                    logger.info(
                        f"[ThreadPool] [{task.request_id}] 推理完成 | "
                        f"InferenceTime={inference_time:.3f}s | "
                        f"TotalTime={total_latency:.3f}s | "
                        f"Detected={result.get('count', 0)} | "
                        f"Status={result.get('status', 'unknown')} | "
                        f"Worker={thread_name}"
                    )
                    
                    # 记录指标
                    try:
                        from pool_metrics import get_metrics
                        get_metrics().record_task_completed(
                            total_latency=total_latency,
                            decode_latency=decode_time,
                            inference_latency=inference_time
                        )
                    except:
                        pass
                    
                    # 设置Future结果（确保结果正确返回）
                    task.future.set_result(result)
                    logger.debug(
                        f"[ThreadPool] [{task.request_id}] ✅ 结果已设置到Future | "
                        f"ResultKeys={list(result.keys())} | "
                        f"Count={result.get('count', 0)}"
                    )
                    
                except Exception as e:
                    error_msg = f"推理失败: {str(e)}"
                    logger.error(f"[ThreadPool] [{task.request_id}] {error_msg}", exc_info=True)
                    task.error = error_msg
                    task.status = TaskStatus.FAILED
                    # 记录失败指标
                    try:
                        from pool_metrics import get_metrics
                        get_metrics().record_task_failed()
                    except:
                        pass
                    task.future.set_exception(e)
                
                finally:
                    with self._stats_lock:
                        self._thread_stats['inference_workers_active'] = max(
                            self._thread_stats['inference_workers_active'] - 1, 0
                        )
                        self._thread_stats['inference_queue_size'] = self.inference_queue.qsize()
                    self.inference_queue.task_done()
            
            except Exception as e:
                if self._running:
                    logger.error(f"[ThreadPool] 推理工作线程异常: {thread_name}", exc_info=True)
                time.sleep(0.1)
        
        logger.info(f"[ThreadPool] 推理工作线程退出: {thread_name}")
    
    def get_stats(self) -> dict:
        """获取线程池统计信息"""
        # 更新队列统计到指标
        try:
            from pool_metrics import get_metrics
            metrics = get_metrics()
            with self._stats_lock:
                metrics.update_queue_stats(
                    decode_queue_size=self._thread_stats['message_queue_size'],
                    inference_queue_size=self._thread_stats['inference_queue_size']
                )
                metrics.update_thread_stats(
                    decode_active=self._thread_stats['decode_threads_active'],
                    preprocess_active=self._thread_stats['inference_workers_active']
                )
        except:
            pass
        
        with self._stats_lock:
            return {
                'pid': self.pid,
                'running': self._running,
                'thread_stats': self._thread_stats.copy(),
                'message_queue_size': self._thread_stats['message_queue_size'],
                'inference_queue_size': self._thread_stats['inference_queue_size'],
                'decode_threads_active': self._thread_stats['decode_threads_active'],
                'inference_workers_active': self._thread_stats['inference_workers_active'],
                'config': {
                    'decode_threads': self.config.decode_threads,
                    'inference_worker_threads': self.config.preprocess_threads
                }
            }


# 全局线程池管理器实例（每个Worker进程一个）
thread_pool_manager: Optional[ThreadPoolManager] = None


def get_thread_pool_manager() -> ThreadPoolManager:
    """获取线程池管理器（单例）"""
    global thread_pool_manager
    if thread_pool_manager is None:
        thread_pool_manager = ThreadPoolManager()
    return thread_pool_manager

