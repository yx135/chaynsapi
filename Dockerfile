# 使用Python 3.9作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装依赖和curl（用于健康检查）
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir fastapi uvicorn httpx pydantic

# 创建日志目录
RUN mkdir -p /app/logs

# 复制应用文件
COPY . .

# 确保配置文件存在
RUN if [ ! -f config.json ]; then cp config.example.json config.json; fi

# 设置启动脚本权限
RUN chmod +x docker-entrypoint.sh

# 设置环境变量默认值
ENV USER_TOBIT_ID=0 \
    BEARER_TOKEN=YOUR_BEARER_TOKEN_HERE \
    PORT=5555 \
    DEBUG=true

# 暴露端口（使用环境变量）
EXPOSE ${PORT}

# 使用启动脚本
ENTRYPOINT ["./docker-entrypoint.sh"] 