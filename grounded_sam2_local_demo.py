import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, load_image_from_url,load_image_from_base64 , predict
import time
import random
import string



import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import base64

# def load_image_from_url(url: str):
#     # 请求网页图片
#     resp = requests.get(url)
#     resp.raise_for_status()  # 如果失败会抛异常

#     # 用 BytesIO 包装成文件对象，直接交给 PIL
#     image_source = Image.open(BytesIO(resp.content)).convert("RGB")
#     return image_source
#     # # 原始 numpy 格式
#     # image = np.asarray(image_source)

#     # # 预处理成模型输入
#     # transform = T.Compose([
#     #     T.Resize(800, max_size=1333),
#     #     T.ToTensor(),
#     #     T.Normalize([0.485, 0.456, 0.406],
#     #                 [0.229, 0.224, 0.225]),
#     # ])
#     # image_transformed, _ = transform(image_source, None)

#     # return image, image_transformed

# # 使用示例
# url = "http://36.134.41.199:9000/test/thumbs/2025/11/c9a25f270d.webp"
# image_source = load_image_from_url(url)
# # image_np, image_tensor = load_image_from_url(url)
# # print(image_np.shape, image_tensor.shape)
# print(image_source)


"""
辅助函数：读取base64文件
"""
def load_base64_from_file(file_path: str) -> str:
    """
    从文件中读取base64编码的字符串
    
    Args:
        file_path: base64文件路径
        
    Returns:
        base64字符串
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        base64_str = f.read().strip()
    return base64_str

"""
Hyper parameters
"""
t0 = time.perf_counter()

TEXT_PROMPT = "road surface."
# 支持三种输入方式：
# 1. 本地图片文件路径：如 "/path/to/image.png"
# 2. URL路径：如 "http://example.com/image.jpg"
# 3. Base64文件路径：如 "/path/to/image.b64" (文件扩展名为.b64或路径包含base64)
# IMG_PATH = "/home/baiqiliu/Grounded-SAM-2/test_GSAM/G2 K529+500 上行_20250428_150000_frame_1.png"
IMG_PATH = "/home/baiqiliu/Grounded-SAM-2/test_GSAM/Xinzhuang_Flyover_in_Nanjing.jpg"
# IMG_PATH = "http://36.134.41.199:9000/test/thumbs/2025/11/c9a25f270d.webp"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.16
TEXT_THRESHOLD = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo")
DUMP_JSON_RESULTS = True
MULTIMASK_OUTPUT = False

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16
# ==============================
# 阶段 A：构建/加载模型（GroundingDINO + SAM2）
# ==============================
tA = time.perf_counter()
# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT# SAM2模型权重路径
model_cfg = SAM2_MODEL_CONFIG# SAM2模型配置路径
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)# 构建SAM2模型实例
sam2_predictor = SAM2ImagePredictor(sam2_model)# 构建SAM2图像预测器实例

# build grounding dino model
# 构建Grounding DINO模型
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)
tA_end = time.perf_counter()
print(f"[计时] 阶段A 模型加载耗时: {tA_end - tA:.3f} s")
# ==============================
# 阶段 B：读取图像与前处理
# ==============================
tB = time.perf_counter()
# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT
img_path = IMG_PATH
#(H, W, C)->(C, H, W)
#[R1, G1, B1, R2, G2, B2, ..., Rn, Gn, Bn]->[R1, R2, ..., Rn, G1, G2, ..., Gn, B1, B2, ..., Bn]
# 根据输入类型选择相应的加载方式
print(f"输入路径: {img_path}")
if img_path.startswith(('http://', 'https://')):
    # URL路径
    print("检测到URL，使用load_image_from_url加载")
    image_source, image = load_image_from_url(img_path)
elif img_path.lower().endswith('.b64') or 'base64' in img_path.lower():
    # Base64文件路径
    print("检测到base64文件，使用load_image_from_base64加载")
    base64_str = load_base64_from_file(img_path)
    image_source, image = load_image_from_base64(base64_str)
else:
    # 本地图片文件路径
    print("检测到本地图片文件，使用load_image加载")
    image_source, image = load_image(img_path)

sam2_predictor.set_image(image_source)# 为SAM2预测器设置图像（计算图像嵌入）

tB_end = time.perf_counter()
print(f"[计时] 阶段B 图像读取与设置耗时: {tB_end - tB:.3f} s")
# ==============================
# 阶段 C：GroundingDINO 检测（文本→检测框）
# ==============================
tC = time.perf_counter()

boxes, confidences, labels = predict(# 使用GroundingDINO模型进行预测
    model=grounding_model,
    image=image,
    caption=text,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=DEVICE
)

# process the box prompt for SAM 2
h, w, _ = image_source.shape# 为SAM2处理边界框提示
boxes = boxes * torch.Tensor([w, h, w, h])# 将相对坐标转换为绝对像素坐标（乘以图像宽高）
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
# 将边界框格式从(cx, cy, w, h)转换为(x1, y1, x2, y2)

tC_end = time.perf_counter()
print(f"[计时] 阶段C GroundingDINO检测耗时: {tC_end - tC:.3f} s")
# ==============================
# 阶段 D：SAM2 分割（框→mask）
# ==============================
tD = time.perf_counter()

# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()# 启用自动混合精度（bfloat16），可提高计算效率并减少内存使用
# 如果CUDA可用且GPU为Ampere架构或更高，启用TF32计算
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
# 使用SAM2预测器进行分割预测
masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,# 边界框提示（来自GroundingDINO的检测结果）
    multimask_output=MULTIMASK_OUTPUT,
)

"""
Sample the best mask according to the score，根据分数选择最佳掩码
"""
if MULTIMASK_OUTPUT:
    best = np.argmax(scores, axis=1)                     
    masks = masks[np.arange(masks.shape[0]), best]       

"""
Post-process the output of the model to get the masks, scores, and logits for visualization
"""
# convert the shape to (n, H, W)
# 将形状转换为(n, H, W) - n个目标，每个有高度H和宽度W
if masks.ndim == 4:
    masks = masks.squeeze(1)

# 将置信度从张量转换为numpy数组再转换为列表
confidences = confidences.numpy().tolist()
class_names = labels
# 生成类别ID数组[0, 1, 2, ..., n-1]
class_ids = np.array(list(range(len(class_names))))
# 创建标签列表，格式为"类别名 置信度"
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]



tD_end = time.perf_counter()
print(f"[计时] 阶段D SAM2分割耗时: {tD_end - tD:.3f} s")
tE = time.perf_counter()
# ==============================
# 阶段 E：可视化与图片写出（两张图）
# ==============================
"""
Visualize image with supervision useful API
"""
# 判断输入类型，使用相应的方式读取图像
if img_path.startswith(('http://', 'https://')):
    # URL路径：使用已加载的image_source（RGB格式），转换为BGR格式
    img = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
elif img_path.lower().endswith('.b64') or 'base64' in img_path.lower():
    # Base64文件路径：使用已加载的image_source（RGB格式），转换为BGR格式
    img = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
else:
    # 本地图片文件路径：使用cv2.imread读取
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
img = img.copy()# 创建图像副本以避免修改原始数据
detections = sv.Detections(# 创建检测结果对象，包含边界框、掩码和类别ID
    xyxy=input_boxes,  # 边界框坐标，(n, 4)格式
    mask=masks.astype(bool),  # 分割掩码，转换为布尔类型，(n, h, w)
    class_id=class_ids# 类别ID数组
)

box_annotator = sv.BoxAnnotator()# 创建边界框注释器
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)# 注释图像，添加边界框

label_annotator = sv.LabelAnnotator()# 创建标签注释器
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)# 注释图像，添加标签
cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)# 将注释后的图像保存到输出目录

mask_annotator = sv.MaskAnnotator()# 创建掩码注释器
masked_only_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)# 注释图像，添加掩码
cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), masked_only_frame)
# 将注释后的图像保存到输出目录  

tE_end = time.perf_counter()
print(f"[计时] 阶段E 可视化与写图耗时: {tE_end - tE:.3f} s")
# ==============================
# 阶段 F：JSON 写出（可选）
# ==============================
tF = time.perf_counter()
"""
Dump the results in standard format and save as json files
"""
# 定义函数：将单个掩码转换为RLE（Run-Length Encoding）格式
def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

# ==============================
# Mask 到前端多边形 JSON 转换函数
# ==============================
def generate_random_id():
    """
    生成随机 ID，等价于 JavaScript 的 Math.random().toString(36).slice(2)
    Python 实现：生成 base36 编码的随机字符串
    """
    # base36: 0-9, a-z
    chars = string.digits + string.ascii_lowercase
    # 生成 11 位随机字符串（去掉前2位相当于 slice(2)）
    return ''.join(random.choices(chars, k=11))

def extract_mask_contour_from_box(mask, box):
    """
    从全图 mask 中提取框内区域的轮廓（框局部坐标系）
    
    Args:
        mask: 全图尺寸的布尔 mask (H, W)，numpy.ndarray
        box: 边界框 (x1, y1, x2, y2) 原图坐标系，numpy.ndarray 或 list/tuple
    
    Returns:
        tuple: (contours, actual_box) 
            - contours: 轮廓列表，每个轮廓是 (N, 2) 的数组，坐标在框局部坐标系
            - actual_box: 实际裁剪区域的边界框 (x1, y1, x2, y2)，在图像范围内
    """
    # 类型检查：确保 mask 是 numpy.ndarray
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask 只支持 numpy.ndarray，收到类型: {type(mask)}")
    
    # 确保 mask 是 2 维数组
    if mask.ndim != 2:
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        else:
            raise ValueError(
                f"mask 应该是2维数组 (H, W)，但得到 {mask.ndim} 维，形状: {mask.shape}"
            )
    
    # box 也转成 numpy
    if not isinstance(box, np.ndarray):
        box = np.array(box)
    x1, y1, x2, y2 = box.astype(int)
    
    # 确保坐标在有效范围内（与convert.py保持一致）
    mask_h, mask_w = mask.shape
    x1_actual = max(0, min(x1, mask_w - 1))
    y1_actual = max(0, min(y1, mask_h - 1))
    x2_actual = max(x1_actual + 1, min(x2, mask_w))
    y2_actual = max(y1_actual + 1, min(y2, mask_h))
    
    actual_box = np.array([x1_actual, y1_actual, x2_actual, y2_actual])
    
    # 检查裁剪区域是否有效
    if x2_actual <= x1_actual or y2_actual <= y1_actual:
        return [], actual_box
    
    # 裁剪到框内区域（注意：mask 索引是 [y, x] 顺序）
    box_mask = mask[y1_actual:y2_actual, x1_actual:x2_actual].astype(np.uint8) * 255
    
    # 如果框内没有 mask，返回空列表
    if box_mask.sum() == 0:
        return [], actual_box
    
    # 提取轮廓（只提取外部轮廓，忽略内部洞）
    contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # 转换为 (N, 2) 格式，坐标在框局部坐标系（相对于 actual_box）
    contour_list = []
    for contour in contours:
        if len(contour) >= 3:  # 至少需要3个点构成多边形
            contour_2d = contour.reshape(-1, 2).astype(float)
            contour_list.append(contour_2d)
    
    return contour_list, actual_box

def simplify_polygon(contour, epsilon=2.0):
    """
    使用 Douglas-Peucker 算法简化多边形
    
    Args:
        contour: 轮廓点 (N, 2)，可以是浮点或整数类型
        epsilon: 简化精度参数，值越大简化越多
    
    Returns:
        simplified_contour: 简化后的轮廓点 (M, 2)，M <= N
    """
    # 类型检查和维度处理（与convert.py保持一致）
    if not isinstance(contour, np.ndarray):
        contour = np.array(contour)
    
    if len(contour) < 3:
        return contour
    
    # 确保 (N, 2) 格式
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
    
    # 确保轮廓点是整数类型（OpenCV 要求）
    contour_int = contour.astype(np.int32)
    
    # 确保是 (N, 1, 2) 格式，OpenCV 标准格式
    if contour_int.ndim == 2:
        contour_int = contour_int.reshape(-1, 1, 2)
    
    # 使用 cv2.approxPolyDP 进行多边形近似
    # epsilon 参数控制简化程度
    epsilon_val = epsilon * cv2.arcLength(contour_int, closed=True) / 100.0
    simplified = cv2.approxPolyDP(contour_int, epsilon_val, closed=True)
    
    # 检查简化结果是否为空
    if simplified.shape[0] == 0:
        return contour
    
    # 转换回 (M, 2) 格式并返回浮点类型
    return simplified.reshape(-1, 2).astype(float)

def local_to_global_coords(local_points, box, actual_box):
    """
    将框局部坐标系转换为原图全局坐标系
    
    Args:
        local_points: 局部坐标点 (N, 2)，框局部坐标系（相对于 actual_box）
        box: 原始边界框 (x1, y1, x2, y2) 原图坐标系（可能超出图像边界）
        actual_box: 实际裁剪区域的边界框 (x1, y1, x2, y2) 原图坐标系（在图像范围内）
    
    Returns:
        global_points: 全局坐标点 (N, 2)，原图坐标系
    """
    # 使用实际裁剪区域的左上角作为偏移
    x1_actual, y1_actual, x2_actual, y2_actual = actual_box
    actual_box_w = x2_actual - x1_actual
    actual_box_h = y2_actual - y1_actual
    
    # 计算原始 box 的尺寸
    x1_orig, y1_orig, x2_orig, y2_orig = box
    orig_box_w = x2_orig - x1_orig
    orig_box_h = y2_orig - y1_orig
    
    # 如果实际裁剪区域与原始 box 尺寸不同（box 超出边界），需要缩放
    if actual_box_w != orig_box_w or actual_box_h != orig_box_h:
        scale_x = orig_box_w / actual_box_w if actual_box_w > 0 else 1.0
        scale_y = orig_box_h / actual_box_h if actual_box_h > 0 else 1.0
        # 先缩放到原始 box 坐标系
        scaled_points = local_points.copy()
        scaled_points[:, 0] = local_points[:, 0] * scale_x
        scaled_points[:, 1] = local_points[:, 1] * scale_y
        # 然后平移到原图坐标系
        global_points = scaled_points.copy()
        global_points[:, 0] = x1_orig + scaled_points[:, 0]
        global_points[:, 1] = y1_orig + scaled_points[:, 1]
    else:
        # 如果尺寸一致，直接平移
        global_points = local_points.copy()
        global_points[:, 0] = x1_orig + local_points[:, 0]
        global_points[:, 1] = y1_orig + local_points[:, 1]
    
    return global_points

def mask_to_polygon_json(mask, box, label, score, order, epsilon=2.0):
    """
    将 mask 转换为前端多边形 JSON 格式
    
    Args:
        mask: 全图尺寸的布尔 mask (H, W)
        box: 边界框 (x1, y1, x2, y2) 原图坐标系
        label: 类别标签
        score: 检测框置信度分数
        order: 顺序编号
        epsilon: 多边形简化精度参数
    
    Returns:
        polygon_json: 前端多边形 JSON 对象，如果没有有效轮廓则返回 None
    """
    # 步骤1: 从 mask 提取框内轮廓（框局部坐标系）
    contours, actual_box = extract_mask_contour_from_box(mask, box)
    
    if not contours:
        return None
    
    # 选择最大的轮廓作为主要轮廓
    main_contour = max(contours, key=len)
    
    # 步骤2: 简化轮廓
    simplified_contour = simplify_polygon(main_contour, epsilon=epsilon)
    
    # 步骤3: 局部坐标转换为全局坐标
    global_points = local_to_global_coords(simplified_contour, box, actual_box)
    
    # 步骤4: 组装前端 JSON
    polygon_id = generate_random_id()
    points = [
        {
            "id": generate_random_id(),
            "x": float(x),
            "y": float(y)
        }
        for x, y in global_points
    ]
    
    polygon_json = {
        "id": polygon_id,
        "type": "line",
        "points": points,
        "label": label,
        "score": float(score),
        "order": int(order)
    }
    
    return polygon_json

def visualize_polygon_from_json(image_path, json_path, output_dir):
    """
    根据 JSON 文件中的多边形点进行可视化标注
    
    Args:
        image_path: 原始图像路径
        json_path: 多边形 JSON 文件路径
        output_dir: 输出目录
    """
    # 读取原始图像
    if image_path.startswith(('http://', 'https://')):
        # URL路径：使用load_image_from_url加载
        try:
            image_source_rgb, _ = load_image_from_url(image_path)
            img = cv2.cvtColor(image_source_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[警告] 无法从URL读取图像: {image_path}, 错误: {e}")
            return
    elif image_path.lower().endswith('.b64') or 'base64' in image_path.lower():
        # Base64文件路径：使用load_image_from_base64加载
        try:
            base64_str = load_base64_from_file(image_path)
            image_source_rgb, _ = load_image_from_base64(base64_str)
            img = cv2.cvtColor(image_source_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[警告] 无法从base64文件读取图像: {image_path}, 错误: {e}")
            return
    else:
        # 本地图片文件路径：使用cv2.imread读取
        img = cv2.imread(image_path)
        if img is None:
            print(f"[警告] 无法读取图像: {image_path}")
            return
    
    img_copy = img.copy()
    
    # 读取 JSON 文件
    with open(json_path, "r") as f:
        polygon_data = json.load(f)
    
    polygon_list = polygon_data.get("polygon", [])
    if not polygon_list:
        print("[警告] JSON 文件中没有多边形数据")
        return
    
    # 定义颜色列表（用于区分不同多边形）
    colors = [
        (0, 255, 0),      # 绿色
        (255, 0, 0),      # 蓝色
        (0, 0, 255),      # 红色
        (255, 255, 0),    # 青色
        (255, 0, 255),    # 洋红
        (0, 255, 255),    # 黄色
        (128, 0, 128),    # 紫色
        (255, 165, 0),    # 橙色
    ]
    
    # 绘制每个多边形
    for idx, polygon in enumerate(polygon_list):
        points_data = polygon.get("points", [])
        if len(points_data) < 3:
            continue
        
        # 提取点坐标
        points = np.array([[p["x"], p["y"]] for p in points_data], dtype=np.int32)
        
        # 选择颜色
        color = colors[idx % len(colors)]
        
        # 绘制多边形轮廓
        cv2.polylines(img_copy, [points], isClosed=True, color=color, thickness=2)
        
        # 绘制多边形填充（半透明）
        overlay = img_copy.copy()
        cv2.fillPoly(overlay, [points], color=color)
        cv2.addWeighted(overlay, 0.3, img_copy, 0.7, 0, img_copy)
        
        # 绘制顶点
        for point in points:
            cv2.circle(img_copy, tuple(point), 3, color, -1)
        
        # 获取标签和分数
        label = polygon.get("label", "")
        score = polygon.get("score", 0.0)
        order = polygon.get("order", idx + 1)
        
        # 计算标签位置（多边形中心点）
        if len(points) > 0:
            center_x = int(points[:, 0].mean())
            center_y = int(points[:, 1].mean())
            
            # 构建标签文本
            label_text = label if label else f"Object {order}"
            score_text = f"{score:.2f}"
            full_text = f"{label_text} ({score_text})"
            
            # 获取文本尺寸
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                full_text, font, font_scale, thickness
            )
            
            # 绘制文本背景（白色矩形）
            cv2.rectangle(
                img_copy,
                (center_x - text_width // 2 - 5, center_y - text_height - baseline - 5),
                (center_x + text_width // 2 + 5, center_y + baseline + 5),
                (255, 255, 255),
                -1
            )
            
            # 绘制文本
            cv2.putText(
                img_copy,
                full_text,
                (center_x - text_width // 2, center_y),
                font,
                font_scale,
                color,
                thickness
            )
    
    # 保存可视化结果
    output_path = os.path.join(output_dir, "grounded_sam2_polygon_visualization.jpg")
    cv2.imwrite(output_path, img_copy)
    print(f"[信息] 多边形可视化结果已保存至: {output_path}")

# 如果启用JSON结果输出
if DUMP_JSON_RESULTS:
    # ==============================
    # 原有格式：RLE 格式输出（保留）
    # ==============================
    # 将掩码转换为RLE格式
    mask_rles = [single_mask_to_rle(mask) for mask in masks]
    # 将边界框坐标转换为列表
    input_boxes_list = input_boxes.tolist()
    scores_list = scores.tolist()
    # save the results in standard format
    # 以标准格式保存结果
    results = {
        "image_path": img_path,# 图像路径
        "annotations" : [# 注释列表
            {
                "class_name": class_name,# 类别名称
                "bbox": box,# 边界框坐标
                "segmentation": mask_rle,# RLE格式的分割掩码
                "score": score,# 置信度分数
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes_list, mask_rles, scores_list)
        ],# 遍历所有检测结果
        "box_format": "xyxy",# 边界框格式说明
        "img_width": w,# 图像宽度
        "img_height": h,# 图像高度
    }
    # 将结果写入JSON文件
    with open(os.path.join(OUTPUT_DIR, "grounded_sam2_local_image_demo_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # ==============================
    # 新增格式：前端多边形 JSON 格式输出
    # ==============================
    polygon_list = []
    for idx, (mask, box, label, confidence) in enumerate(zip(masks, input_boxes, class_names, confidences)):
        polygon_json = mask_to_polygon_json(
            mask=mask,
            box=box,
            label=label,
            score=confidence,
            order=idx + 1,
            epsilon=1.0  # 多边形简化精度参数，可根据需要调整
        )
        if polygon_json is not None:
            polygon_list.append(polygon_json)
    
    # 保存前端多边形格式 JSON
    frontend_polygon_results = {
        "polygon": polygon_list
    }
    polygon_json_path = os.path.join(OUTPUT_DIR, "grounded_sam2_frontend_polygon_results.json")
    with open(polygon_json_path, "w") as f:
        json.dump(frontend_polygon_results, f, indent=2)
    
    # ==============================
    # 根据 JSON 多边形点进行可视化标注
    # ==============================
    if polygon_list:
        visualize_polygon_from_json(img_path, polygon_json_path, OUTPUT_DIR)


tF_end = time.perf_counter()
print(f"[计时] 阶段F JSON写出耗时: {tF_end - tF:.3f} s")

# ==============================
# 总耗时统计
# ==============================
t1 = time.perf_counter()
print(f"[计时] 总耗时（启动→图片/JSON写出完成）: {t1 - t0:.3f} s")
print(f"[细分] A模型加载: {tA_end - tA:.3f}s | B图像读取: {tB_end - tB:.3f}s | C检测: {tC_end - tC:.3f}s | D分割: {tD_end - tD:.3f}s | E可视化: {tE_end - tE:.3f}s | F写JSON: {tF_end - tF:.3f}s")