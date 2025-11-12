from flask import Flask, request, jsonify, g
import logging
from logging.handlers import RotatingFileHandler
import psutil  # 系统监控
import torch   # GPU监控
import os      # 进程信息
import time    # 时间戳
import uuid    # 请求ID生成
import threading  # 线程信息
import traceback  # 异常追踪
from datetime import datetime
from pathlib import Path
from model_manager import model_manager

# 创建logs目录
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(exist_ok=True)

# 配置详细的日志系统
# 使用RotatingFileHandler支持日志轮转（每个文件10MB，保留5个备份）
file_handler = RotatingFileHandler(
    logs_dir / "app.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 统一的日志格式
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | PID:%(process)d | TID:%(thread)d | %(name)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 配置根日志记录器
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# 获取应用日志记录器
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 请求统计
_request_stats = {
    'total_requests': 0,
    'success_requests': 0,
    'error_requests': 0,
    'lock': threading.Lock()
}

def load_models_on_startup():
    """在启动时加载模型"""
    logger.info("🔄 启动时加载模型...")
    try:
        model_manager.load_models()
        logger.info("✅ 模型加载完成，服务就绪")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
        raise

@app.before_request
def before_request():
    """请求前处理：生成请求ID并记录请求信息"""
    # 生成唯一请求ID
    g.request_id = str(uuid.uuid4())[:8]
    g.start_time = time.time()
    g.thread_id = threading.current_thread().ident
    
    # 记录请求信息
    logger.info(
        f"[{g.request_id}] 📥 收到请求 | "
        f"Method={request.method} | "
        f"Path={request.path} | "
        f"Remote={request.remote_addr} | "
        f"PID={os.getpid()} | "
        f"TID={g.thread_id} | "
        f"User-Agent={request.headers.get('User-Agent', 'N/A')[:50]}"
    )
    
    # 更新统计
    with _request_stats['lock']:
        _request_stats['total_requests'] += 1

@app.after_request
def after_request(response):
    """请求后处理：记录响应信息和耗时"""
    # 计算处理时间
    duration = time.time() - g.start_time
    
    # 获取响应大小
    response_size = len(response.get_data()) if hasattr(response, 'get_data') else 0
    
    # 记录响应信息
    log_level = logging.INFO if response.status_code < 400 else logging.WARNING
    logger.log(
        log_level,
        f"[{g.request_id}] 📤 响应完成 | "
        f"Status={response.status_code} | "
        f"Duration={duration:.3f}s | "
        f"Size={response_size} bytes | "
        f"PID={os.getpid()} | "
        f"TID={g.thread_id}"
    )
    
    # 更新统计
    with _request_stats['lock']:
        if response.status_code < 400:
            _request_stats['success_requests'] += 1
        else:
            _request_stats['error_requests'] += 1
    
    # 添加请求ID到响应头（便于追踪）
    response.headers['X-Request-ID'] = g.request_id
    response.headers['X-Process-ID'] = str(os.getpid())
    response.headers['X-Thread-ID'] = str(g.thread_id)
    
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口 - 增强版"""
    try:
        # 检查模型状态
        model_status = model_manager.models_loaded
        
        # 检查GPU状态
        gpu_available = torch.cuda.is_available()
        gpu_memory_used = 0
        if gpu_available:
            gpu_memory_used = torch.cuda.memory_allocated() // 1024**2  # MB
        
        # 检查内存使用
        memory = psutil.virtual_memory()
        
        return jsonify({
            "status": "healthy" if model_status else "unhealthy",
            "models_loaded": model_status,
            "device": model_manager.device,
            "service": "Grounded-SAM2 Flask Service",
            "pid": os.getpid(),  # 当前进程ID
            "gpu": {
                "available": gpu_available,
                "memory_used_mb": gpu_memory_used
            },
            "memory": {
                "percent": memory.percent,
                "used_mb": memory.used // 1024**2
            }
        }), 200 if model_status else 503
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/semantic-segmentation', methods=['POST'])
def inference():
    """推理接口 - 仅支持 Base64 输入格式"""
    request_id = getattr(g, 'request_id', 'unknown')
    
    # 检查模型状态
    if not model_manager.models_loaded:
        logger.warning(f"[{request_id}] ⚠️ 模型未加载，返回503")
        return jsonify({"error": "模型未加载完成", "request_id": request_id}), 503
    
    try:
        # 获取请求数据（支持 JSON 和 form-data）
        if request.is_json:
            data = request.json
            base64_str = data.get('image_base64')
            text_prompt = data.get('text_prompt', 'road surface.')
            box_threshold = float(data.get('box_threshold', 0.01))
            text_threshold = float(data.get('text_threshold', 0.25))
            epsilon = float(data.get('epsilon', 1.0))
        else:
            base64_str = request.form.get('image_base64')
            text_prompt = request.form.get('text_prompt', 'road surface.')
            box_threshold = float(request.form.get('box_threshold', 0.01))
            text_threshold = float(request.form.get('text_threshold', 0.25))
            epsilon = float(request.form.get('epsilon', 1.0))
        
        # 记录请求参数
        base64_len = len(base64_str) if base64_str else 0
        logger.info(
            f"[{request_id}] 📋 请求参数 | "
            f"text_prompt='{text_prompt}' | "
            f"box_threshold={box_threshold} | "
            f"text_threshold={text_threshold} | "
            f"epsilon={epsilon} | "
            f"base64_length={base64_len}"
        )
        
        # 验证 base64 输入
        if not base64_str:
            logger.warning(f"[{request_id}] ⚠️ 缺少image_base64参数")
            return jsonify({"error": "请提供 image_base64 参数（Base64 编码的图像数据）", "request_id": request_id}), 400
        
        # 记录推理开始
        inference_start = time.time()
        logger.info(f"[{request_id}] 🚀 开始推理 | PID={os.getpid()} | TID={threading.current_thread().ident}")
        
        # 执行推理
        result = model_manager.inference(
            image_base64=base64_str,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            epsilon=epsilon,
            request_id=request_id  # 传递请求ID用于日志追踪
        )
        
        # 记录推理完成
        inference_duration = time.time() - inference_start
        result_count = result.get('count', 0)
        logger.info(
            f"[{request_id}] ✅ 推理完成 | "
            f"Duration={inference_duration:.3f}s | "
            f"Detected={result_count} objects | "
            f"Status={result.get('status', 'unknown')}"
        )
        
        # 添加请求ID到响应
        result['request_id'] = request_id
        result['inference_time'] = round(inference_duration, 3)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(
            f"[{request_id}] ❌ 推理请求处理失败 | "
            f"Error={str(e)} | "
            f"PID={os.getpid()} | "
            f"TID={threading.current_thread().ident}",
            exc_info=True
        )
        return jsonify({"error": str(e), "request_id": request_id}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """请求统计接口"""
    with _request_stats['lock']:
        stats_copy = _request_stats.copy()
    stats_copy.pop('lock', None)  # 移除锁对象
    return jsonify(stats_copy)

@app.route('/metrics', methods=['GET'])
def metrics():
    """性能监控接口 - 提供详细系统状态"""
    try:
        # 系统层面
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU层面
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_count": torch.cuda.device_count(),
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() // 1024**2,
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() // 1024**2,
                "gpu_memory_total_mb": torch.cuda.get_device_properties(0).total_memory // 1024**2,
                "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            }
        
        # 进程层面
        process = psutil.Process()
        process_info = {
            "pid": process.pid,
            "threads": process.num_threads(),
            "memory_mb": process.memory_info().rss // 1024**2,
            "cpu_percent": process.cpu_percent()
        }
        
        # 模型层面
        model_info = {
            "loaded": model_manager.models_loaded,
            "device": model_manager.device,
            "sam2_predictor": model_manager.sam2_predictor is not None,
            "grounding_model": model_manager.grounding_model is not None
        }
        
        return jsonify({
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used // 1024**2,
                "memory_total_mb": memory.total // 1024**2,
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free // 1024**3
            },
            "gpu": gpu_info,
            "process": process_info,
            "model": model_info,
            "timestamp": time.time(),
            "request_stats": {
                "total": _request_stats['total_requests'],
                "success": _request_stats['success_requests'],
                "error": _request_stats['error_requests']
            }
        })
        
    except Exception as e:
        logger.error(f"监控数据获取失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """全局异常处理器 - 记录所有未捕获的异常"""
    request_id = getattr(g, 'request_id', 'unknown')
    
    # 记录详细的异常信息
    logger.error(
        f"[{request_id}] ❌ 未捕获的异常 | "
        f"Error={str(e)} | "
        f"Type={type(e).__name__} | "
        f"PID={os.getpid()} | "
        f"TID={threading.current_thread().ident}",
        exc_info=True
    )
    
    # 返回错误响应
    return jsonify({
        "error": "服务器内部错误",
        "error_type": type(e).__name__,
        "error_message": str(e),
        "request_id": request_id
    }), 500

@app.route('/')
def index():
    """首页"""
    return jsonify({
        "message": "Grounded-SAM2推理服务",
        "endpoints": {
            "健康检查": "/health",
            "性能监控": "/metrics",
            "请求统计": "/stats",
            "推理接口": "/semantic-segmentation(POST)",
            "输入格式": "Base64 编码的图像数据",
            "参数": {
                "image_base64": "必需，Base64 编码的图像字符串",
                "text_prompt": "可选，文本提示（默认: 'road surface.'）",
                "box_threshold": "可选，检测框阈值（默认: 0.01）",
                "text_threshold": "可选，文本匹配阈值（默认: 0.25）",
                "epsilon": "可选，多边形简化精度参数（默认: 1.0）"
            }
        }
    })

if __name__ == '__main__':
    # 仅用于直接调试，生产环境通过Gunicorn启动
    app.run(
        host='0.0.0.0',
        port=6155,
        debug=False,  # 生产环境设为False
        threaded=True  # 支持并发
    )