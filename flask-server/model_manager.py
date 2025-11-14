import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
# 获取当前文件的目录（flask-server）
current_dir = Path(__file__).parent
# 获取项目根目录（Grounded-SAM-2）
project_root = current_dir.parent
# 添加到 sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import time
import numpy as np
import threading
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image_from_base64, predict
from torchvision.ops import box_convert
import logging
from convert import convert_masks_to_json

logger = logging.getLogger(__name__)

class ModelManager:
    """最简单的模型管理器 - 全局单例"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.sam2_predictor = None
            self.grounding_model = None
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.models_loaded = False
            self._model_lock = threading.Lock()  # 线程锁（传统模式）
            self._use_thread_pool = False  # 是否使用线程池模式
            self._thread_pool_manager = None  # 线程池管理器引用
            self._initialized = True
    
    def load_models(self):
        """一次性加载模型到内存 - 线程安全版本"""
        if self.models_loaded:
            logger.info("模型已加载，跳过重复加载")
            return
        
        # 添加进程ID日志，方便调试
        logger.info(f"🚀 进程 {os.getpid()} 开始加载模型到 {self.device}...")
        start_time = time.time()
        
        try:
            with self._model_lock:  # 获取锁
                # 双重检查：获取锁后再次确认
                if self.models_loaded:
                    logger.info("模型已被其他线程加载，跳过")
                    return
                
                # 获取项目根目录路径（用于模型权重文件路径）
                # 支持通过环境变量覆盖（Docker 环境）
                project_root = Path(os.getenv('PROJECT_ROOT', Path(__file__).parent.parent))
                
                # 加载SAM2模型
                # 注意：build_sam2 使用 Hydra，config_file 应该是 Hydra 配置名称（相对路径），
                # 而不是绝对文件路径。Hydra 会在其配置搜索路径中查找配置文件。
                # 配置文件实际位置：sam2/configs/sam2.1/sam2.1_hiera_l.yaml
                logger.info("📦 加载SAM2模型...")
                sam2_config_name = os.getenv('SAM2_CONFIG_NAME', "configs/sam2.1/sam2.1_hiera_l.yaml")  # Hydra 配置名称（相对路径）
                sam2_checkpoint_path = Path(os.getenv('SAM2_CHECKPOINT_PATH',
                    str(project_root / "checkpoints" / "sam2.1_hiera_large.pt")))
                
                logger.info(f"📦 SAM2模型路径: {sam2_checkpoint_path}")
                sam2_model = build_sam2(
                    sam2_config_name,  # 使用 Hydra 配置名称，不是绝对路径
                    str(sam2_checkpoint_path),
                    device=self.device
                )
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                
                # 加载GroundingDINO模型
                # GroundingDINO 的 load_model 需要绝对路径
                logger.info("📦 加载GroundingDINO模型...")
                gdino_config_path = Path(os.getenv('GDINO_CONFIG_PATH',
                    str(project_root / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")))
                gdino_checkpoint_path = Path(os.getenv('GDINO_CHECKPOINT_PATH',
                    str(project_root / "gdino_checkpoints" / "groundingdino_swint_ogc.pth")))
                
                logger.info(f"📦 GroundingDINO配置路径: {gdino_config_path}")
                logger.info(f"📦 GroundingDINO模型路径: {gdino_checkpoint_path}")
                
                self.grounding_model = load_model(
                    str(gdino_config_path),
                    str(gdino_checkpoint_path),
                    device=self.device
                )
                
                self.models_loaded = True
                load_time = time.time() - start_time
                logger.info(f"✅ 进程 {os.getpid()} 模型加载完成，耗时: {load_time:.2f}秒")
            
        except Exception as e:
            logger.error(
                f"❌ 进程 {os.getpid()} 模型加载失败: {e}",
                exc_info=True
            )
            raise
    
    def enable_thread_pool(self, thread_pool_manager):
        """启用线程池模式"""
        self._use_thread_pool = True
        self._thread_pool_manager = thread_pool_manager
        logger.info(f"[ModelManager] 线程池模式已启用 (PID={os.getpid()})")
    
    def disable_thread_pool(self):
        """禁用线程池模式，回退到锁模式"""
        self._use_thread_pool = False
        self._thread_pool_manager = None
        logger.info(f"[ModelManager] 线程池模式已禁用，回退到锁模式 (PID={os.getpid()})")
    
    def inference(self, image_base64, text_prompt="road surface.", box_threshold=0.2, text_threshold=0.25, epsilon=1.0, request_id=None):
        """
        执行推理 - 使用 Base64 输入（支持线程池模式和传统锁模式）
        
        Args:
            image_base64: Base64 编码的图像字符串
            text_prompt: 文本提示（需要小写并以点结尾）
            box_threshold: 检测框阈值
            text_threshold: 文本匹配阈值
            epsilon: 多边形简化精度参数（默认: 1.0）
            request_id: 请求ID（用于日志追踪）
        
        Returns:
            推理结果字典（传统模式）或Future对象（线程池模式）
        """
        if not self.models_loaded:
            raise RuntimeError("模型未加载")
        
        request_id = request_id or "unknown"
        
        # 如果启用线程池模式，使用异步方式
        if self._use_thread_pool and self._thread_pool_manager:
            from task_queue import InferenceTask
            from concurrent.futures import Future
            
            # 创建任务
            task = InferenceTask(
                request_id=request_id,
                image_base64=image_base64,
                text_prompt=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                epsilon=epsilon
            )
            
            # 提交任务到线程池
            future = self._thread_pool_manager.submit_task(task)
            
            # 为了保持API兼容性，这里需要同步等待结果
            # 但实际应用中，可以返回Future让调用方异步处理
            try:
                result = future.result(timeout=300.0)  # 300秒超时
                return result
            except Exception as e:
                logger.error(f"[{request_id}] 线程池推理失败: {e}", exc_info=True)
                raise
        
        # 传统锁模式（保持向后兼容）
        thread_id = threading.current_thread().ident
        
        # 在推理时也加锁（特别是写操作）
        lock_start = time.time()
        logger.info(
            f"[{request_id}] 🔒 等待模型锁 | "
            f"PID={os.getpid()} | "
            f"TID={thread_id}"
        )
        
        with self._model_lock:  # 获取锁
            lock_wait_time = time.time() - lock_start
            if lock_wait_time > 0.1:  # 如果等待时间超过100ms，记录警告
                logger.warning(
                    f"[{request_id}] ⚠️ 模型锁等待时间较长 | "
                    f"WaitTime={lock_wait_time:.3f}s | "
                    f"PID={os.getpid()} | "
                    f"TID={thread_id}"
                )
            
            logger.info(
                f"[{request_id}] 🔓 获取模型锁成功 | "
                f"PID={os.getpid()} | "
                f"TID={thread_id}"
            )
            
            # ==============================
            # 阶段 B：读取图像与前处理（参考 grounded_sam2_local_demo.py）
            # ==============================
            stage_start = time.time()
            logger.info(f"[{request_id}] 📸 阶段1: 加载图像 | PID={os.getpid()} | TID={thread_id}")
            
            # 使用 load_image_from_base64 加载图像
            # 返回: (image_source: np.array, image: torch.Tensor)
            image_source, image = load_image_from_base64(image_base64)
            
            # 记录推理前的显存使用
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated() / 1024**2  # MB
                logger.debug(f"[{request_id}] 推理前显存: {mem_before:.1f} MB")
            
            try:
                # 调用无锁版本的推理方法
                result = self._inference_without_lock(
                    image_source=image_source,
                    image=image,
                    text_prompt=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    epsilon=epsilon,
                    request_id=request_id
                )
                
                # 记录推理后的显存使用
                if torch.cuda.is_available():
                    mem_after = torch.cuda.memory_allocated() / 1024**2  # MB
                    logger.debug(f"[{request_id}] 推理后显存: {mem_after:.1f} MB (变化: {mem_after - mem_before:+.1f} MB)")
                
                return result
            finally:
                # 确保image tensor被清理（如果还在GPU上）
                if isinstance(image, torch.Tensor) and image.is_cuda:
                    del image
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
    def _inference_without_lock(self, image_source, image, text_prompt="road surface.", box_threshold=0.2, text_threshold=0.25, epsilon=1.0, request_id=None):
        """
        推理方法（由GPU工作线程调用）
        
        注意：虽然方法名叫"without_lock"，但实际上整个推理流程都在一个锁内执行，
        确保set_image和predict之间的原子性，避免被其他线程打断。
        
        Args:
            image_source: numpy数组格式的图像（RGB）
            image: torch.Tensor格式的图像（已预处理）
            text_prompt: 文本提示
            box_threshold: 检测框阈值
            text_threshold: 文本匹配阈值
            epsilon: 多边形简化精度参数
            request_id: 请求ID
        
        Returns:
            推理结果字典
        """
        request_id = request_id or "unknown"
        thread_id = threading.current_thread().ident
        inference_start = time.time()
        
        image_h, image_w = image_source.shape[:2]
        
        # ==============================
        # 关键修复：将整个推理流程（set_image + GroundingDINO + predict）放在一个锁内
        # 这样可以确保原子性，避免set_image和predict之间被其他线程打断
        # ==============================
        with self._model_lock:
            # ==============================
            # 阶段 B：设置SAM2图像嵌入
            # ==============================
            stage_start = time.time()
            logger.info(f"[{request_id}] 🧠 阶段2: 设置SAM2图像嵌入 | PID={os.getpid()} | TID={thread_id}")
            
            try:
                self.sam2_predictor.set_image(image_source)
                # 验证set_image是否成功
                if not hasattr(self.sam2_predictor, '_is_image_set') or not self.sam2_predictor._is_image_set:
                    raise RuntimeError("set_image执行后_is_image_set仍为False")
            except Exception as e:
                logger.error(f"[{request_id}] ❌ SAM2图像嵌入失败: {e}", exc_info=True)
                raise
            
            embed_time = time.time() - stage_start
            logger.info(
                f"[{request_id}] ✅ SAM2图像嵌入完成 | "
                f"Duration={embed_time:.3f}s"
            )
            
            # ==============================
            # 阶段 C：GroundingDINO 检测（文本→检测框）
            # ==============================
            stage_start = time.time()
            logger.info(
                f"[{request_id}] 🔍 阶段3: GroundingDINO检测 | "
                f"Prompt='{text_prompt}' | "
                f"BoxThresh={box_threshold} | "
                f"TextThresh={text_threshold} | "
                f"PID={os.getpid()} | "
                f"TID={thread_id}"
            )
            
            # GroundingDINO检测
            # 注意：GroundingDINO模型在多线程环境下不是线程安全的
            # 模型的forward方法会调用set_image_tensor/unset_image_tensor，这些操作会修改模型内部状态
            debug_box_threshold = 0.01  # 非常低的阈值，用于查看所有可能的检测
            
            # 添加详细诊断：直接调用模型查看原始输出
            try:
                import torch
                with torch.no_grad():
                    # 直接调用模型，查看原始logits
                    outputs_raw = self.grounding_model(image[None], captions=[text_prompt.lower().strip() + ("." if not text_prompt.endswith(".") else "")])
                    prediction_logits_raw = outputs_raw["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
                    max_logits_per_query = prediction_logits_raw.max(dim=1)[0]  # (nq,)
                    
                    # 记录原始logits统计信息
                    if len(max_logits_per_query) > 0:
                        max_logit_value = float(max_logits_per_query.max())
                        mean_logit_value = float(max_logits_per_query.mean())
                        logger.info(
                            f"[{request_id}] 🔍 模型原始输出诊断: "
                            f"Queries={len(max_logits_per_query)}, "
                            f"MaxLogit={max_logit_value:.6f}, "
                            f"MeanLogit={mean_logit_value:.6f}, "
                            f"MaxLogitAbove0.01={int((max_logits_per_query > 0.01).sum())}, "
                            f"MaxLogitAbove0.001={int((max_logits_per_query > 0.001).sum())}"
                        )
                    else:
                        logger.warning(f"[{request_id}] ⚠️ 模型返回空的logits，可能是模型状态异常")
            except Exception as e:
                logger.warning(f"[{request_id}] ⚠️ 无法获取模型原始输出: {e}")
            
            # 使用debug阈值进行检测
            boxes_debug, confidences_debug, labels_debug = predict(
                model=self.grounding_model,
                image=image,
                caption=text_prompt,
                box_threshold=debug_box_threshold,
                text_threshold=0.01,  # 也降低text_threshold
                device=self.device
            )
            
            # 使用实际阈值进行检测
            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )
            
            # 计算检测耗时（在锁内计算，但日志记录在锁外）
            detect_time = time.time() - stage_start
        
        # 记录原始检测结果（在锁外，避免长时间持有锁）
        if len(boxes_debug) > 0:
            conf_debug_min = float(confidences_debug.min()) if hasattr(confidences_debug, 'min') else float(min(confidences_debug))
            conf_debug_max = float(confidences_debug.max()) if hasattr(confidences_debug, 'max') else float(max(confidences_debug))
            logger.info(
                f"[{request_id}] 🔍 原始检测结果（box_threshold=0.01）: "
                f"Detected={len(boxes_debug)} boxes, "
                f"ConfidenceRange=[{conf_debug_min:.3f}, {conf_debug_max:.3f}]"
            )
        else:
            logger.warning(
                f"[{request_id}] ⚠️ 即使使用极低阈值(box_threshold=0.01)也未检测到任何目标，"
                f"可能是text_prompt='{text_prompt}'与图片内容不匹配，或模型输出logits全部小于0.01"
            )
        
        # 记录详细的检测信息（在锁外，避免长时间持有锁）
        if len(boxes) == 0:
            if len(boxes_debug) > 0:
                # 有原始检测结果但被过滤掉了
                conf_debug_max = float(confidences_debug.max()) if hasattr(confidences_debug, 'max') else float(max(confidences_debug))
                logger.warning(
                    f"[{request_id}] ⚠️ GroundingDINO检测完成但未找到目标 | "
                    f"Detected=0 boxes (过滤后) | "
                    f"原始检测={len(boxes_debug)} boxes, 最高置信度={conf_debug_max:.3f} | "
                    f"Duration={detect_time:.3f}s | "
                    f"提示: 当前box_threshold={box_threshold}太高，建议降低到{max(0.01, conf_debug_max * 0.8):.3f}以下"
                )
            else:
                # 完全没有检测结果
                logger.warning(
                    f"[{request_id}] ⚠️ GroundingDINO检测完成但未找到目标 | "
                    f"Detected=0 boxes | "
                    f"Duration={detect_time:.3f}s | "
                    f"提示: 尝试调整text_prompt (当前='{text_prompt}') 或检查图片内容"
                )
            
            # 提前返回，避免处理空的boxes
            h, w, _ = image_source.shape
            return {
                "status": "success",
                "results": [],
                "count": 0,
                "image_shape": {
                    "width": w,
                    "height": h
                },
                "message": "未检测到任何目标"
            }
        else:
            # 记录置信度范围
            if len(confidences) > 0:
                conf_min = float(confidences.min()) if hasattr(confidences, 'min') else float(min(confidences))
                conf_max = float(confidences.max()) if hasattr(confidences, 'max') else float(max(confidences))
                conf_mean = float(confidences.mean()) if hasattr(confidences, 'mean') else float(sum(confidences) / len(confidences))
                logger.info(
                    f"[{request_id}] ✅ GroundingDINO检测完成 | "
                    f"Detected={len(boxes)} boxes | "
                    f"ConfidenceRange=[{conf_min:.3f}, {conf_max:.3f}], Mean={conf_mean:.3f} | "
                    f"Duration={detect_time:.3f}s"
                )
            else:
                logger.info(
                    f"[{request_id}] ✅ GroundingDINO检测完成 | "
                    f"Detected={len(boxes)} boxes | "
                    f"Duration={detect_time:.3f}s"
                )
        
        # 处理边界框：将相对坐标转换为绝对像素坐标
        h, w, _ = image_source.shape
        # 确保Tensor在正确的设备上
        if isinstance(boxes, torch.Tensor):
            boxes = boxes * torch.Tensor([w, h, w, h]).to(boxes.device)
        else:
            boxes = boxes * torch.Tensor([w, h, w, h])
        
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        # 立即清理GPU tensor（GroundingDINO的boxes可能在GPU上）
        if isinstance(boxes, torch.Tensor):
            del boxes
        if isinstance(confidences, torch.Tensor):
            confidences = confidences.cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 再次检查是否有检测结果（双重保险）
        if len(input_boxes) == 0:
            logger.info(
                f"[{request_id}] ⚠️ 未检测到任何目标 | "
                f"ImageSize={w}x{h}"
            )
            return {
                "status": "success",
                "results": [],
                "count": 0,
                "image_shape": {
                    "width": w,
                    "height": h
                },
                "message": "未检测到任何目标"
            }
        
        # ==============================
        # 阶段 D：SAM2 分割（框→mask）
        # ==============================
        stage_start = time.time()
        logger.info(
            f"[{request_id}] 🎯 阶段4: SAM2分割 | "
            f"Boxes={len(input_boxes)} | "
            f"PID={os.getpid()} | "
            f"TID={thread_id}"
        )
        
        # 启用自动混合精度（bfloat16）
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # SAM2分割
        # 注意：predict必须在锁内执行，确保与set_image的原子性
        # 但是，由于整个推理流程（set_image + GroundingDINO + predict）都在锁内，
        # 这里需要重新获取锁，因为之前的锁已经释放了
        try:
            with self._model_lock:
                # 验证状态（在锁内检查，确保原子性）
                if not hasattr(self.sam2_predictor, '_is_image_set') or not self.sam2_predictor._is_image_set:
                    raise RuntimeError("调用predict前_is_image_set为False，状态异常（可能被其他线程的set_image打断）")
                
                if not hasattr(self.sam2_predictor, '_orig_hw') or self.sam2_predictor._orig_hw is None:
                    raise RuntimeError("调用predict前_orig_hw为None，状态异常（可能被其他线程的set_image打断）")
                
                # 在锁内调用predict，确保与set_image的原子性
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    predict_result = self.sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )
            
            # 检查predict返回值（在锁外检查，避免长时间持有锁）
            if predict_result is None:
                raise RuntimeError("SAM2 predict返回None")
            
            masks, scores, logits = predict_result
            
            # 检查返回值是否有效
            if masks is None:
                raise RuntimeError("SAM2 predict返回的masks为None")
            if scores is None:
                raise RuntimeError("SAM2 predict返回的scores为None")
            if logits is None:
                raise RuntimeError("SAM2 predict返回的logits为None")
                
        except Exception as e:
            logger.error(f"[{request_id}] ❌ SAM2分割失败: {e}", exc_info=True)
            raise
        
        segment_time = time.time() - stage_start
        logger.info(
            f"[{request_id}] ✅ SAM2分割完成 | "
            f"Masks={len(masks) if hasattr(masks, '__len__') else 'N/A'} | "
            f"Duration={segment_time:.3f}s"
        )
        
        # 转换为 numpy 数组（如果是 torch.Tensor）
        # 注意：先转换再删除，避免tensor留在GPU
        masks_numpy = None
        scores_numpy = None
        
        # 立即释放logits（我们不需要logits，只保留masks和scores）
        if isinstance(logits, torch.Tensor):
            del logits
            logits = None
        
        if isinstance(masks, torch.Tensor):
            masks_numpy = masks.cpu().numpy()
            del masks  # 立即删除GPU tensor
            masks = masks_numpy
        if isinstance(scores, torch.Tensor):
            scores_numpy = scores.cpu().numpy()
            del scores  # 立即删除GPU tensor
            scores = scores_numpy
        
        # 清理GPU显存：释放推理过程中的临时tensor
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 先同步，确保所有操作完成
            torch.cuda.empty_cache()  # 再清理缓存
        
        # 处理多mask输出
        if masks.ndim == 4 and masks.shape[1] > 1:
            best = np.argmax(scores, axis=1)
            masks = masks[np.arange(masks.shape[0]), best]
        
        # 转换为 (n, H, W) 格式
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        elif masks.ndim == 2:
            masks = masks[np.newaxis, :, :]
        
        # 确保 masks 是 3维数组 (n, H, W)
        assert masks.ndim == 3, f"masks 应该是3维数组 (n, H, W)，但得到 {masks.ndim} 维，形状: {masks.shape}"
        
        # 确保所有数组长度一致
        n_masks = masks.shape[0]
        n_boxes = len(input_boxes)
        n_labels = len(labels)
        n_confidences = len(confidences)
        
        if not (n_masks == n_boxes == n_labels == n_confidences):
            logger.warning(f"数组长度不一致: masks={n_masks}, boxes={n_boxes}, labels={n_labels}, confidences={n_confidences}")
            min_len = min(n_masks, n_boxes, n_labels, n_confidences)
            masks = masks[:min_len]
            input_boxes = input_boxes[:min_len]
            labels = labels[:min_len]
            confidences = confidences[:min_len]
        
        # ==============================
        # 阶段 E：结果格式化 - 使用convert_masks_to_json转换为多边形格式
        # ==============================
        stage_start = time.time()
        logger.info(f"[{request_id}] 📦 阶段5: 格式化结果（转换为多边形格式） | PID={os.getpid()} | TID={thread_id}")
        
        # 确保masks是numpy数组格式（布尔类型）
        masks_np = []
        for mask in masks:
            if isinstance(mask, np.ndarray):
                # 转换为布尔类型
                mask_bool = mask.astype(bool)
            else:
                mask_bool = np.array(mask, dtype=bool)
            masks_np.append(mask_bool)
        
        # 确保input_boxes是numpy数组格式
        if isinstance(input_boxes, np.ndarray):
            input_boxes_np = input_boxes
        else:
            input_boxes_np = np.array(input_boxes)
        
        # 确保labels是列表格式
        if isinstance(labels, np.ndarray):
            labels_list = labels.tolist()
        else:
            labels_list = list(labels)
        
        # 确保confidences是列表格式（float）
        confidences_list = []
        for conf in confidences:
            if isinstance(conf, (torch.Tensor, np.ndarray)):
                conf = float(conf.item() if hasattr(conf, 'item') else conf)
            else:
                conf = float(conf)
            confidences_list.append(conf)
        
        # 使用convert_masks_to_json转换为多边形格式
        try:
            # 在转换前清理显存（masks_np可能很大）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            result_dict = convert_masks_to_json(
                masks=masks_np,
                input_boxes=input_boxes_np,
                labels=labels_list,
                confidences=confidences_list,
                w=w,
                h=h,
                epsilon=epsilon,
                enable_visualization=False
            )
            
            # 转换后立即清理临时变量
            del masks_np, input_boxes_np, labels_list, confidences_list
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            format_time = time.time() - stage_start
            
            logger.info(
                f"[{request_id}] ✅ 结果格式化完成（多边形格式） | "
                f"Count={result_dict.get('count', 0)} | "
                f"FormatTime={format_time:.3f}s | "
                f"PID={os.getpid()} | "
                f"TID={thread_id}"
            )
        except Exception as e:
            logger.error(
                f"[{request_id}] ❌ 结果格式化失败: {e}",
                exc_info=True
            )
            # 如果转换失败，返回原始格式作为降级方案
            format_time = time.time() - stage_start
            masks_list = []
            for mask in masks:
                masks_list.append(mask.astype(int).tolist())
            
            input_boxes_list = input_boxes.tolist() if isinstance(input_boxes, np.ndarray) else list(input_boxes)
            
            result_dict = {
                "status": "error",
                "message": f"格式转换失败，返回原始格式: {str(e)}"
            }
            logger.warning(f"[{request_id}] 使用原始格式作为降级方案")
        
        # 最终清理：释放所有临时变量和GPU显存
        # 注意：按顺序删除，先删除大的tensor
        if torch.cuda.is_available():
            mem_before_cleanup = torch.cuda.memory_allocated() / 1024**2  # MB
            
            # 清理所有可能留在GPU的tensor
            if 'logits' in locals() and logits is not None:
                if isinstance(logits, torch.Tensor):
                    del logits
                logits = None
            if 'masks' in locals() and isinstance(masks, torch.Tensor):
                del masks
            if 'scores' in locals() and isinstance(scores, torch.Tensor):
                del scores
            if 'image' in locals() and isinstance(image, torch.Tensor):
                if image.is_cuda:
                    del image
            if 'boxes' in locals() and isinstance(boxes, torch.Tensor):
                if boxes.is_cuda:
                    del boxes
            
            # 清理SAM2 predictor的图像嵌入缓存（重要！）
            # 这会释放set_image时创建的图像嵌入特征
            try:
                self.sam2_predictor.reset_predictor()
            except Exception as e:
                logger.warning(f"[{request_id}] 清理SAM2 predictor缓存失败: {e}")
            
            # 强制同步并清理缓存
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            mem_after_cleanup = torch.cuda.memory_allocated() / 1024**2  # MB
            freed_memory = mem_before_cleanup - mem_after_cleanup
            if freed_memory > 10:  # 如果释放了超过10MB，记录日志
                logger.info(f"[{request_id}] 🧹 显存清理完成，释放了 {freed_memory:.1f} MB (清理前: {mem_before_cleanup:.1f} MB, 清理后: {mem_after_cleanup:.1f} MB)")
        
        # 清理CPU变量
        if 'image_source' in locals():
            del image_source
        if 'input_boxes' in locals():
            del input_boxes
        if 'labels' in locals():
            del labels
        if 'confidences' in locals():
            del confidences
        if 'masks' in locals():
            del masks
        if 'scores' in locals():
            del scores
        if 'logits' in locals():
            logits = None
            
        total_inference_time = time.time() - inference_start
        
        logger.info(
            f"[{request_id}] ✅ 推理完成 | "
            f"TotalTime={total_inference_time:.3f}s | "
            f"PID={os.getpid()} | "
            f"TID={thread_id}"
        )
        
        return result_dict
#         #——————————————————————————————————————————return______________________________________________
#         # ==============================
#         # 阶段 F：结果处理与格式化
#         # ==============================
#         # 转换为前端格式（参考 grounded_sam2_local_demo.py 的多边形格式）
#         results = []
#         for idx in range(len(masks)):
#             mask = masks[idx]
#             box = input_boxes[idx]
#             label = labels[idx]
#             confidence = confidences[idx]
            
#             # 确保 mask 是布尔类型或可以转换为布尔类型
#             if isinstance(confidence, torch.Tensor):
#                 confidence = confidence.item() if confidence.numel() == 1 else float(confidence)
#             else:
#                 confidence = float(confidence)
            
#             # 提取多边形轮廓（使用与 grounded_sam2_local_demo.py 相同的方法）
#             polygon = self._mask_to_polygon_json(mask, box, label, confidence, idx + 1, epsilon=epsilon)
            
#             if polygon is not None:
#                 results.append({
#                     "id": polygon.get("id"),
#                     "type": polygon.get("type"),
#                     "points": polygon.get("points"),
#                     "label": polygon.get("label"),
#                     "score": polygon.get("score"),
#                     "order": polygon.get("order"),
#                     "bbox": box.tolist() if isinstance(box, np.ndarray) else list(box)
#                 })
        
#         return {
#             "status": "success",
#             "results": results,
#             "count": len(results),
#             "image_shape": {
#                 "width": w,
#                 "height": h
#             }
#         }
        
#     except Exception as e:
#         logger.error(f"推理失败: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return {"status": "error", "message": str(e)}

# def _mask_to_polygon_json(self, mask, box, label, score, order, epsilon=1.0):
#     """
#     将 mask 转换为前端多边形 JSON 格式（参考 grounded_sam2_local_demo.py）
    
#     Args:
#         mask: 全图尺寸的布尔 mask (H, W)
#         box: 边界框 (x1, y1, x2, y2) 原图坐标系
#         label: 类别标签
#         score: 检测框置信度分数
#         order: 顺序编号
#         epsilon: 多边形简化精度参数
    
#     Returns:
#         polygon_json: 前端多边形 JSON 对象，如果没有有效轮廓则返回 None
#     """
#     try:
#         import cv2
#         import random
#         import string
        
#         def generate_random_id():
#             """生成随机 ID"""
#             chars = string.digits + string.ascii_lowercase
#             return ''.join(random.choices(chars, k=11))
        
#         def extract_mask_contour_from_box(mask, box):
#             """从全图 mask 中提取框内区域的轮廓"""
#             # 确保 mask 是 numpy 数组
#             if not isinstance(mask, np.ndarray):
#                 mask = np.array(mask)
            
#             # 确保 mask 是2维数组
#             if mask.ndim != 2:
#                 if mask.ndim == 3:
#                     mask = mask.squeeze(0)
#                 else:
#                     raise ValueError(f"mask 应该是2维数组 (H, W)，但得到 {mask.ndim} 维，形状: {mask.shape}")
            
#             # 确保 box 是 numpy 数组并转换为整数
#             if not isinstance(box, np.ndarray):
#                 box = np.array(box)
#             x1, y1, x2, y2 = box.astype(int)
            
#             # 确保坐标在有效范围内
#             mask_h, mask_w = mask.shape
#             x1_actual = max(0, min(x1, mask_w - 1))
#             y1_actual = max(0, min(y1, mask_h - 1))
#             x2_actual = max(x1_actual + 1, min(x2, mask_w))
#             y2_actual = max(y1_actual + 1, min(y2, mask_h))
            
#             actual_box = np.array([x1_actual, y1_actual, x2_actual, y2_actual])
            
#             # 检查裁剪区域是否有效
#             if x2_actual <= x1_actual or y2_actual <= y1_actual:
#                 return [], actual_box
            
#             # 裁剪到框内区域（注意：mask 索引是 [y, x] 顺序）
#             box_mask = mask[y1_actual:y2_actual, x1_actual:x2_actual].astype(np.uint8) * 255
            
#             if box_mask.sum() == 0:
#                 return [], actual_box
            
#             # 提取轮廓（只提取外部轮廓）
#             contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
#             contour_list = []
#             for contour in contours:
#                 if len(contour) >= 3:
#                     contour_2d = contour.reshape(-1, 2).astype(float)
#                     contour_list.append(contour_2d)
            
#             return contour_list, actual_box
        
#         def simplify_polygon(contour, epsilon=2.0):
#             """使用 Douglas-Peucker 算法简化多边形"""
#             # 确保 contour 是 numpy 数组
#             if not isinstance(contour, np.ndarray):
#                 contour = np.array(contour)
            
#             # 检查 contour 是否有效
#             if len(contour) < 3:
#                 return contour
            
#             # 确保 contour 是2维数组 (N, 2)
#             if contour.ndim == 1:
#                 if len(contour) == 2:
#                     contour = contour.reshape(1, 2)
#                 else:
#                     raise ValueError(f"contour 形状无效: {contour.shape}")
#             elif contour.ndim == 2:
#                 if contour.shape[1] != 2:
#                     raise ValueError(f"contour 应该是 (N, 2) 形状，但得到 {contour.shape}")
#             else:
#                 raise ValueError(f"contour 应该是2维数组，但得到 {contour.ndim} 维")
            
#             # 转换为整数类型（OpenCV 要求）
#             contour_int = contour.astype(np.int32)
            
#             # 转换为 OpenCV 要求的格式 (N, 1, 2)
#             if contour_int.ndim == 2:
#                 contour_int = contour_int.reshape(-1, 1, 2)
            
#             # 计算简化参数
#             epsilon_val = epsilon * cv2.arcLength(contour_int, closed=True) / 100.0
            
#             # 执行多边形简化
#             simplified = cv2.approxPolyDP(contour_int, epsilon_val, closed=True)
            
#             # 转换回 (N, 2) 格式并返回浮点类型
#             if simplified.shape[0] == 0:
#                 return contour  # 如果简化后为空，返回原始轮廓
            
#             return simplified.reshape(-1, 2).astype(float)
        
#         def local_to_global_coords(local_points, box, actual_box):
#             """将框局部坐标系转换为原图全局坐标系"""
#             # 确保输入是 numpy 数组
#             if not isinstance(local_points, np.ndarray):
#                 local_points = np.array(local_points)
#             if not isinstance(box, np.ndarray):
#                 box = np.array(box)
#             if not isinstance(actual_box, np.ndarray):
#                 actual_box = np.array(actual_box)
            
#             # 确保 local_points 是2维数组 (N, 2)
#             if local_points.ndim == 1:
#                 local_points = local_points.reshape(1, -1)
#             if local_points.shape[1] != 2:
#                 raise ValueError(f"local_points 应该是 (N, 2) 形状，但得到 {local_points.shape}")
            
#             x1_actual, y1_actual, x2_actual, y2_actual = actual_box.astype(float)
#             actual_box_w = x2_actual - x1_actual
#             actual_box_h = y2_actual - y1_actual
            
#             x1_orig, y1_orig, x2_orig, y2_orig = box.astype(float)
#             orig_box_w = x2_orig - x1_orig
#             orig_box_h = y2_orig - y1_orig
            
#             # 创建副本以避免修改原始数据
#             global_points = local_points.copy()
            
#             if actual_box_w != orig_box_w or actual_box_h != orig_box_h:
#                 scale_x = orig_box_w / actual_box_w if actual_box_w > 0 else 1.0
#                 scale_y = orig_box_h / actual_box_h if actual_box_h > 0 else 1.0
#                 # 先缩放
#                 global_points[:, 0] = local_points[:, 0] * scale_x
#                 global_points[:, 1] = local_points[:, 1] * scale_y
#                 # 再平移
#                 global_points[:, 0] = x1_orig + global_points[:, 0]
#                 global_points[:, 1] = y1_orig + global_points[:, 1]
#             else:
#                 # 直接平移
#                 global_points[:, 0] = x1_orig + local_points[:, 0]
#                 global_points[:, 1] = y1_orig + local_points[:, 1]
            
#             return global_points
        
#         # 步骤1: 从 mask 提取框内轮廓
#         contours, actual_box = extract_mask_contour_from_box(mask, box)
        
#         if not contours:
#             return None
        
#         # 选择最大的轮廓作为主要轮廓
#         main_contour = max(contours, key=len)
        
#         # 步骤2: 简化轮廓
#         simplified_contour = simplify_polygon(main_contour, epsilon=epsilon)
        
#         # 步骤3: 局部坐标转换为全局坐标
#         global_points = local_to_global_coords(simplified_contour, box, actual_box)
        
#         # 步骤4: 组装前端 JSON
#         polygon_id = generate_random_id()
#         points = [
#             {
#                 "id": generate_random_id(),
#                 "x": float(x),
#                 "y": float(y)
#             }
#             for x, y in global_points
#         ]
        
#         polygon_json = {
#             "id": polygon_id,
#             "type": "line",
#             "points": points,
#             "label": label,
#             "score": float(score),
#             "order": int(order)
#         }
        
#         return polygon_json
        
#     except Exception as e:
#         logger.error(f"多边形转换失败: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return None

# 全局模型管理器实例
model_manager = ModelManager()