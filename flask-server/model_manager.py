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
            self._model_lock = threading.Lock()  # 线程锁
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
                project_root = Path(__file__).parent.parent
                
                # 加载SAM2模型
                # 注意：build_sam2 使用 Hydra，config_file 应该是 Hydra 配置名称（相对路径），
                # 而不是绝对文件路径。Hydra 会在其配置搜索路径中查找配置文件。
                # 配置文件实际位置：sam2/configs/sam2.1/sam2.1_hiera_l.yaml
                logger.info("📦 加载SAM2模型...")
                sam2_config_name = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Hydra 配置名称（相对路径）
                sam2_checkpoint_path = project_root / "checkpoints" / "sam2.1_hiera_large.pt"
                
                sam2_model = build_sam2(
                    sam2_config_name,  # 使用 Hydra 配置名称，不是绝对路径
                    str(sam2_checkpoint_path),
                    device=self.device
                )
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                
                # 加载GroundingDINO模型
                # GroundingDINO 的 load_model 需要绝对路径
                logger.info("📦 加载GroundingDINO模型...")
                gdino_config_path = project_root / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
                gdino_checkpoint_path = project_root / "gdino_checkpoints" / "groundingdino_swint_ogc.pth"
                
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
    
    def inference(self, image_base64, text_prompt="road surface.", box_threshold=0.2, text_threshold=0.25, epsilon=1.0, request_id=None):
        """
        执行推理 - 使用 Base64 输入（线程安全版本）
        
        Args:
            image_base64: Base64 编码的图像字符串
            text_prompt: 文本提示（需要小写并以点结尾）
            box_threshold: 检测框阈值
            text_threshold: 文本匹配阈值
            epsilon: 多边形简化精度参数（默认: 1.0）
            request_id: 请求ID（用于日志追踪）
        
        Returns:
            推理结果字典
        """
        if not self.models_loaded:
            raise RuntimeError("模型未加载")
        
        request_id = request_id or "unknown"
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
            image_h, image_w = image_source.shape[:2]
            
            load_time = time.time() - stage_start
            logger.info(
                f"[{request_id}] ✅ 图像加载完成 | "
                f"Size={image_w}x{image_h} | "
                f"Duration={load_time:.3f}s"
            )
            
            # 设置SAM2图像（计算图像嵌入）
            stage_start = time.time()
            logger.info(f"[{request_id}] 🧠 阶段2: 设置SAM2图像嵌入 | PID={os.getpid()} | TID={thread_id}")
            
            self.sam2_predictor.set_image(image_source)
            
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
            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )
            
            detect_time = time.time() - stage_start
            logger.info(
                f"[{request_id}] ✅ GroundingDINO检测完成 | "
                f"Detected={len(boxes)} boxes | "
                f"Duration={detect_time:.3f}s"
            )
            
            # 处理边界框：将相对坐标转换为绝对像素坐标
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            
            # 检查是否有检测结果
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
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                masks, scores, logits = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
            
            segment_time = time.time() - stage_start
            logger.info(
                f"[{request_id}] ✅ SAM2分割完成 | "
                f"Masks={len(masks) if hasattr(masks, '__len__') else 'N/A'} | "
                f"Duration={segment_time:.3f}s"
            )
            
            # 转换为 numpy 数组（如果是 torch.Tensor）
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            
            # 清理GPU显存：释放推理过程中的临时tensor
            # 注意：不要清理SAM2的_features，因为下次推理还需要使用
            # 只需要清理PyTorch的缓存即可
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 清理PyTorch的未使用缓存
            
            # 处理多mask输出（参考 grounded_sam2_local_demo.py）
            # 如果 multimask_output=True，masks 形状为 (n, 3, H, W)，需要选择最佳mask
            if masks.ndim == 4 and masks.shape[1] > 1:
                # 多mask输出情况：选择最佳mask
                best = np.argmax(scores, axis=1)
                masks = masks[np.arange(masks.shape[0]), best]
            
            # 转换为 (n, H, W) 格式
            # 如果还有多余的维度，使用 squeeze
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            elif masks.ndim == 2:
                # 如果只有一个mask，添加batch维度
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
                # 取最小长度，确保不会索引越界
                min_len = min(n_masks, n_boxes, n_labels, n_confidences)
                masks = masks[:min_len]
                input_boxes = input_boxes[:min_len]
                labels = labels[:min_len]
                confidences = confidences[:min_len]
            # return {"masks": masks, "input_boxes": input_boxes, "labels": labels, "confidences": confidences}
            # ==============================
            # 阶段 E：结果格式化
            # ==============================
            stage_start = time.time()
            logger.info(f"[{request_id}] 📦 阶段5: 格式化结果 | PID={os.getpid()} | TID={thread_id}")
            
            # 转换为可JSON序列化的格式
            # masks: (n, H, W) -> 转换为列表，每个mask是布尔数组的列表
            masks_list = []
            for mask in masks:
                # 将布尔数组转换为整数列表（0和1），便于传输
                masks_list.append(mask.astype(int).tolist())
            
            # input_boxes: (n, 4) -> 转换为列表
            input_boxes_list = input_boxes.tolist() if isinstance(input_boxes, np.ndarray) else list(input_boxes)
            
            # labels: 转换为列表
            labels_list = labels.tolist() if isinstance(labels, np.ndarray) else list(labels)
            
            # confidences: 转换为列表（确保是Python float类型）
            confidences_list = []
            for conf in confidences:
                if isinstance(conf, (torch.Tensor, np.ndarray)):
                    conf = float(conf.item() if hasattr(conf, 'item') else conf)
                else:
                    conf = float(conf)
                confidences_list.append(conf)
            
            format_time = time.time() - stage_start
            
            # 最终清理：释放所有临时变量
            del image_source, image, boxes, input_boxes, labels, confidences
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 再次清理缓存
            
            total_inference_time = time.time() - lock_start
            
            logger.info(
                f"[{request_id}] ✅ 结果格式化完成 | "
                f"Count={len(masks_list)} | "
                f"FormatTime={format_time:.3f}s | "
                f"TotalTime={total_inference_time:.3f}s | "
                f"PID={os.getpid()} | "
                f"TID={thread_id}"
            )
            
            return {
                "status": "success",
                "masks": masks_list,
                "input_boxes": input_boxes_list,
                "labels": labels_list,
                "confidences": confidences_list,
                "count": len(masks_list),
                "image_shape": {
                    "width": w,
                    "height": h
                }
            }
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