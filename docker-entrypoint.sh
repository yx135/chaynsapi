#!/bin/bash

# 从环境变量生成配置文件
cat > config.json << EOF
{
    "api": {
        "user_tobit_id": ${USER_TOBIT_ID:-0},
        "bearer_token": "${BEARER_TOKEN:-YOUR_BEARER_TOKEN_HERE}"
    },
    "server": {
        "host": "0.0.0.0",
        "port": ${PORT:-5555},
        "debug": ${DEBUG:-true}
    },
    "session": {
        "timeout": 3600,
        "retry": {
            "max_attempts": ${MAX_ATTEMPTS:-30},
            "delay": ${Delay:-1}
        }
    },
    "logging": {
        "level": "${LOG_LEVEL:-INFO}",
        "format": "%(asctime)s - %(levelname)s - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S"
    }
}
EOF

# 显示生成的配置文件内容（用于调试）
echo "Generated config.json:"
cat config.json

# 启动应用
exec uvicorn chaynsapi:app --host 0.0.0.0 --port ${PORT:-5555} 