# 开发环境Gunicorn配置

# 减少进程数，便于调试
workers = 2

# 禁用守护进程，日志输出到控制台
daemon = False

# 调试模式
loglevel = "debug"

# 绑定到本地
bind = "127.0.0.1:6155"

# 启用自动重载（代码修改后自动重启）
reload = True

# 超时时间缩短
timeout = 60
