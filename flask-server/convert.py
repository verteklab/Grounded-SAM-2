import numpy as np
import logging

logger = logging.getLogger(__name__)
# import tempfile
import requests
# import urllib.parse
# from PIL import Image, ImageDraw, ImageFont
import random
# from django.conf import settings


import os
import time
import random

def convert_masks_to_json(
    masks,
    input_boxes,
    labels,
    confidences,
    w,
    h,
    epsilon=1.0,
    enable_visualization=False,
    original_image_path=None,
):
    """
    将 numpy 格式的分割结果转换为前端需要的 JSON 格式。

    只支持 numpy.ndarray：

    Args:
        masks:       list[np.ndarray]，每个元素为 mask (H, W)，dtype=bool 或 0/1
        input_boxes: list[np.ndarray] 或 list[list/tuple]，每个为 (x1, y1, x2, y2)
        labels:      list[str] 或 list[int]，类别标签
        confidences: list[float]，置信度
        w:           图像宽度
        h:           图像高度
        epsilon:     多边形简化精度参数
        enable_visualization: 是否生成可视化图片
        original_image_path: 原始图片路径（支持本地文件路径和HTTP URL，可视化时需要）

    Returns:
        dict: 用于前端的 JSON 对象，包含可视化图片路径（如果启用）
    """
    try:
        results = []
        visualization_path = None
        skipped_count = 0  # 记录跳过的数量
        skipped_reasons = {}  # 记录跳过原因统计

        # 如果启用可视化，先准备可视化环境
        if enable_visualization:
            # visualization_path = _generate_visualization(
            #     masks, input_boxes, labels, confidences, w, h, original_image_path
            # )
            pass  # 暂时禁用可视化功能

        for idx in range(len(masks)):
            mask = masks[idx]
            box = input_boxes[idx]
            label = labels[idx]
            confidence = confidences[idx]

            # 强制只支持 numpy
            if not isinstance(mask, np.ndarray):
                raise TypeError(f"mask 只支持 numpy.ndarray，收到类型: {type(mask)}")

            # 置信度转 float
            confidence = float(confidence)

            # 跳过空标签的检测结果
            # 将label转换为字符串并检查是否为空
            label_str = str(label).strip() if label is not None else ""
            if not label_str:
                skipped_count += 1
                skipped_reasons['empty_label'] = skipped_reasons.get('empty_label', 0) + 1
                logger.debug(f"跳过结果 {idx+1}: 空标签")
                continue  # 跳过空标签的结果
            
            # 提取多边形轮廓
            try:
                polygon = _mask_to_polygon_json(
                    mask=mask,
                    box=box,
                    label=label,
                    score=confidence,
                    order=idx + 1,
                    epsilon=float(epsilon),
                )
            except Exception as e:
                skipped_count += 1
                skipped_reasons['polygon_conversion_error'] = skipped_reasons.get('polygon_conversion_error', 0) + 1
                logger.warning(f"结果 {idx+1} (label={label_str}) 多边形转换失败: {e}")
                continue

            if polygon is not None:
                # 再次检查polygon中的label是否为空（双重保险）
                polygon_label = polygon.get("label", "")
                if not polygon_label or not str(polygon_label).strip():
                    skipped_count += 1
                    skipped_reasons['polygon_empty_label'] = skipped_reasons.get('polygon_empty_label', 0) + 1
                    logger.debug(f"跳过结果 {idx+1}: polygon中label为空")
                    continue  # 跳过空标签的结果
                
                # 保证 bbox 为 list
                bbox = box.tolist() if isinstance(box, np.ndarray) else list(box)

                results.append(
                    {
                        "id": polygon.get("id"),
                        "type": polygon.get("type"),
                        "points": polygon.get("points"),
                        "label": polygon.get("label"),
                        "score": polygon.get("score"),
                        "order": polygon.get("order"),
                        "bbox": bbox,
                    }
                )
            else:
                skipped_count += 1
                skipped_reasons['no_contour'] = skipped_reasons.get('no_contour', 0) + 1
                logger.debug(f"跳过结果 {idx+1} (label={label_str}): 未找到有效轮廓")
        
        # 记录转换统计信息
        if skipped_count > 0:
            logger.warning(
                f"多边形转换统计: 总输入={len(masks)}, 成功={len(results)}, 跳过={skipped_count}, "
                f"跳过原因={skipped_reasons}"
            )

        result_dict = {
            "status": "success",
            "results": results,
            "count": len(results),
            "image_shape": {
                "width": w,
                "height": h,
            },
        }

        # 如果有可视化图片，添加到结果中
        if visualization_path:
            result_dict["visualization_path"] = visualization_path

        return result_dict

    except Exception as e:
        raise Exception(f"convert_masks_to_json error: {e}")


# def _generate_visualization(
#     masks, input_boxes, labels, confidences, w, h, original_image_path=None
# ):
#     """
#     生成SAM2分割结果的可视化图片

#     Args:
#         masks: 分割掩码列表
#         input_boxes: 边界框列表
#         labels: 标签列表
#         confidences: 置信度列表
#         w: 图像宽度
#         h: 图像高度
#         original_image_path: 原始图片路径（支持本地文件路径和HTTP URL）

#     Returns:
#         str: 可视化图片保存路径
#     """
#     try:
#         # 生成随机颜色
#         colors = [
#             (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#             for _ in range(len(masks))
#         ]

#         # 创建基础图像，支持HTTP URL和本地文件路径
#         if original_image_path:
#             try:
#                 base_image = _load_base_image(original_image_path, w, h)
#             except Exception as e:
#                 base_image = Image.new("RGB", (w, h), "white")
#         else:
#             # 创建白色背景
#             base_image = Image.new("RGB", (w, h), "white")

#         # 创建绘图对象
#         draw = ImageDraw.Draw(base_image)

#         # 绘制掩码和边界框
#         for idx, (mask, box, label, confidence, color) in enumerate(
#             zip(masks, input_boxes, labels, confidences, colors)
#         ):
#             # 绘制掩码
#             if isinstance(mask, np.ndarray):
#                 # 确保掩码是2D
#                 if mask.ndim == 3:
#                     mask = mask.squeeze()

#                 # 创建掩码图像
#                 mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
#                 mask_image = mask_image.resize((w, h), Image.Resampling.NEAREST)

#                 # 创建彩色掩码
#                 colored_mask = Image.new("RGB", (w, h), color)
#                 # 应用透明度
#                 mask_alpha = Image.fromarray((mask * 128).astype(np.uint8), mode="L")
#                 mask_alpha = mask_alpha.resize((w, h), Image.Resampling.NEAREST)

#                 # 合并掩码到基础图像
#                 base_image = Image.composite(colored_mask, base_image, mask_alpha)

#             # 绘制边界框
#             if isinstance(box, np.ndarray):
#                 box = box.tolist()
#             x1, y1, x2, y2 = map(float, box)

#             # 绘制矩形边框
#             draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

#             # 绘制标签文本
#             label_text = f"{label}: {confidence:.2f}"
#             try:
#                 # 尝试使用更好的字体
#                 font = ImageFont.truetype(
#                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
#                 )
#             except:
#                 # 如果没有找到字体，使用默认字体
#                 font = ImageFont.load_default()

#             # 计算文本大小
#             text_bbox = draw.textbbox((0, 0), label_text, font=font)
#             text_width = text_bbox[2] - text_bbox[0]
#             text_height = text_bbox[3] - text_bbox[1]

#             # 绘制文本背景
#             text_bg_x1 = x1
#             text_bg_y1 = y1 - text_height - 5
#             text_bg_x2 = x1 + text_width + 10
#             text_bg_y2 = y1

#             draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], fill=color)
#             draw.text((x1 + 5, y1 - text_height), label_text, fill="white", font=font)

#         # 保存可视化图片
#         temp_dir = getattr(settings, "AI_ANNOTATION_TEMP_DIR", tempfile.gettempdir())
#         os.makedirs(temp_dir, exist_ok=True)

#         vis_filename = (
#             f"sam2_visualization_{int(time.time())}_{random.randint(1000, 9999)}.jpg"
#         )
#         vis_path = os.path.join(temp_dir, vis_filename)

#         base_image.save(vis_path, "JPEG", quality=85)

#         return vis_path

#     except Exception as e:
#         return None


def _mask_to_polygon_json(mask, box, label, score, order, epsilon=1.0):
    """
    将 mask 转换为前端多边形 JSON 格式（参考 grounded_sam2_local_demo.py）

    只支持 numpy.ndarray 的 mask 和 box。

    Args:
        mask:   全图尺寸的布尔 mask (H, W)，numpy.ndarray
        box:    边界框 (x1, y1, x2, y2) 原图坐标系，numpy.ndarray 或 list/tuple
        label:  类别标签
        score:  检测框置信度分数
        order:  顺序编号
        epsilon: 多边形简化精度参数

    Returns:
        dict | None: 前端多边形 JSON 对象，如果没有有效轮廓则返回 None
    """
    try:
        import cv2
        import random
        import string

        def generate_random_id():
            """生成随机 ID（11 位，小写字母 + 数字）"""
            chars = string.digits + string.ascii_lowercase
            return "".join(random.choices(chars, k=11))

        def extract_mask_contour_from_box(mask_, box_):
            """从全图 mask 中提取框内区域的轮廓（只支持 numpy）"""
            if not isinstance(mask_, np.ndarray):
                raise TypeError(f"mask 只支持 numpy.ndarray，收到类型: {type(mask_)}")

            # 确保 mask 是 2 维数组
            if mask_.ndim != 2:
                if mask_.ndim == 3:
                    mask_ = mask_.squeeze(0)
                else:
                    raise ValueError(
                        f"mask 应该是2维数组 (H, W)，但得到 {mask_.ndim} 维，形状: {mask_.shape}"
                    )

            # box 也转成 numpy
            if not isinstance(box_, np.ndarray):
                box_ = np.array(box_)
            x1, y1, x2, y2 = box_.astype(int)

            # 确保坐标在有效范围内
            mask_h, mask_w = mask_.shape
            x1_actual = max(0, min(x1, mask_w - 1))
            y1_actual = max(0, min(y1, mask_h - 1))
            x2_actual = max(x1_actual + 1, min(x2, mask_w))
            y2_actual = max(y1_actual + 1, min(y2, mask_h))

            actual_box_ = np.array([x1_actual, y1_actual, x2_actual, y2_actual])

            # 检查裁剪区域是否有效
            if x2_actual <= x1_actual or y2_actual <= y1_actual:
                return [], actual_box_

            # 裁剪到框内区域（注意：mask 索引是 [y, x] 顺序）
            box_mask = (
                mask_[y1_actual:y2_actual, x1_actual:x2_actual].astype(np.uint8) * 255
            )

            if box_mask.sum() == 0:
                return [], actual_box_

            # 提取轮廓（只提取外部轮廓）
            contours_, _ = cv2.findContours(
                box_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )

            contour_list_ = []
            for contour in contours_:
                if len(contour) >= 3:
                    contour_2d = contour.reshape(-1, 2).astype(float)
                    contour_list_.append(contour_2d)

            return contour_list_, actual_box_

        def simplify_polygon(contour, epsilon_=2.0):
            """使用 Douglas-Peucker 算法简化多边形"""
            if not isinstance(contour, np.ndarray):
                contour = np.array(contour)

            if len(contour) < 3:
                return contour

            # 确保 (N, 2)
            if contour.ndim == 1:
                if len(contour) == 2:
                    contour = contour.reshape(1, 2)
                else:
                    raise ValueError(f"contour 形状无效: {contour.shape}")
            elif contour.ndim == 2:
                if contour.shape[1] != 2:
                    raise ValueError(
                        f"contour 应该是 (N, 2) 形状，但得到 {contour.shape}"
                    )
            else:
                raise ValueError(f"contour 应该是2维数组，但得到 {contour.ndim} 维")

            contour_int = contour.astype(np.int32)

            if contour_int.ndim == 2:
                contour_int = contour_int.reshape(-1, 1, 2)

            epsilon_val = epsilon_ * cv2.arcLength(contour_int, closed=True) / 100.0

            simplified = cv2.approxPolyDP(contour_int, epsilon_val, closed=True)

            if simplified.shape[0] == 0:
                return contour

            return simplified.reshape(-1, 2).astype(float)

        def local_to_global_coords(local_points, box_, actual_box_):
            """将框局部坐标系转换为原图全局坐标系"""
            if not isinstance(local_points, np.ndarray):
                local_points = np.array(local_points)
            if not isinstance(box_, np.ndarray):
                box_ = np.array(box_)
            if not isinstance(actual_box_, np.ndarray):
                actual_box_ = np.array(actual_box_)

            if local_points.ndim == 1:
                local_points = local_points.reshape(1, -1)
            if local_points.shape[1] != 2:
                raise ValueError(
                    f"local_points 应该是 (N, 2) 形状，但得到 {local_points.shape}"
                )

            x1_actual, y1_actual, x2_actual, y2_actual = actual_box_.astype(float)
            actual_box_w = x2_actual - x1_actual
            actual_box_h = y2_actual - y1_actual

            x1_orig, y1_orig, x2_orig, y2_orig = box_.astype(float)
            orig_box_w = x2_orig - x1_orig
            orig_box_h = y2_orig - y1_orig

            global_points = local_points.copy()

            if actual_box_w != orig_box_w or actual_box_h != orig_box_h:
                scale_x = orig_box_w / actual_box_w if actual_box_w > 0 else 1.0
                scale_y = orig_box_h / actual_box_h if actual_box_h > 0 else 1.0

                global_points[:, 0] = local_points[:, 0] * scale_x
                global_points[:, 1] = local_points[:, 1] * scale_y

                global_points[:, 0] = x1_orig + global_points[:, 0]
                global_points[:, 1] = y1_orig + global_points[:, 1]
            else:
                global_points[:, 0] = x1_orig + local_points[:, 0]
                global_points[:, 1] = y1_orig + local_points[:, 1]

            return global_points

        # 步骤1: 从 mask 提取框内轮廓
        contours, actual_box = extract_mask_contour_from_box(mask, box)

        if not contours:
            # 记录为什么没有找到轮廓（用于调试）
            mask_sum = mask.sum() if isinstance(mask, np.ndarray) else 0
            box_str = f"[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]" if isinstance(box, (list, np.ndarray)) and len(box) >= 4 else str(box)
            logger.debug(
                f"未找到有效轮廓: mask_sum={mask_sum}, box={box_str}, "
                f"mask_shape={mask.shape if isinstance(mask, np.ndarray) else 'N/A'}"
            )
            return None

        # 选择最大的轮廓作为主要轮廓
        main_contour = max(contours, key=len)

        # 步骤2: 简化轮廓
        simplified_contour = simplify_polygon(main_contour, epsilon_=epsilon)

        # 步骤3: 局部坐标转换为全局坐标
        global_points = local_to_global_coords(simplified_contour, box, actual_box)

        # 步骤4: 组装前端 JSON
        polygon_id = generate_random_id()
        points = [
            {
                "id": generate_random_id(),
                "x": float(x),
                "y": float(y),
            }
            for x, y in global_points
        ]

        polygon_json = {
            "id": polygon_id,
            "type": "line",
            "points": points,
            "label": label,
            "score": float(score),
            "order": int(order),
        }

        return polygon_json

    except Exception as e:
        raise Exception(f"convert_masks_to_json error: {e}")