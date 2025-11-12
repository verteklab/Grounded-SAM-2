"""
Gunicorn启动入口文件
注意：由于CUDA与fork()不兼容，不能使用preload_app=True
模型会在每个worker进程的post_fork回调中加载（见gunicorn.conf.py）
"""
from app import app
import logging

# 配置Gunicorn日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 注意：模型加载在gunicorn.conf.py的post_fork回调中进行
# 不在模块级别加载，避免CUDA fork问题

if __name__ == '__main__':
    # 直接运行时也预加载模型（用于测试）
    from model_manager import model_manager
    logger.info("🚀 直接运行wsgi.py，正在预加载模型...")
    model_manager.load_models()
    logger.info("✅ 模型预加载完成")

