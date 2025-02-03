import time
import json
import uuid
import asyncio
import httpx
import logging
import os
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """加载配置文件"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "config.json")
            
            if not os.path.exists(config_path):
                example_path = os.path.join(current_dir, "config.example.json")
                if os.path.exists(example_path):
                    raise FileNotFoundError(
                        f"配置文件不存在: {config_path}\n"
                        f"请复制 {example_path} 到 {config_path} 并填写正确的配置信息"
                    )
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 设置配置属性
            self.user_tobit_id = config["api"]["user_tobit_id"]
            self.bearer_token = config["api"]["bearer_token"]
            self.server_host = config["server"]["host"]
            self.server_port = config["server"]["port"]
            self.server_debug = config["server"]["debug"]
            self.session_timeout = config["session"]["timeout"]
            self.retry_max_attempts = config["session"]["retry"]["max_attempts"]
            self.retry_delay = config["session"]["retry"]["delay"]
            
            # 配置日志
            logging.basicConfig(
                level=getattr(logging, config["logging"]["level"]),
                format=config["logging"]["format"],
                datefmt=config["logging"]["date_format"]
            )
            
            self.logger = logging.getLogger(__name__)
            self.logger.info("配置加载成功")
            
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            raise

# 创建配置实例
config = Config()

# 模型配置
models_data = {"models": []}

async def load_models():
    """从API加载模型配置"""
    global models_data
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://cube.tobit.cloud/ai-proxy/v1/models")
            if response.status_code == 200:
                api_models = response.json()
                # 过滤可用的模型并转换格式
                available_models = []
                for model in api_models:
                    if model.get("isAvailable", False):
                        available_models.append({
                            "id": model.get("modelName"),
                            "name": model.get("showName"),
                            "description": f"{model.get('showName')} model",
                            "context_length": 32000,  # 默认上下文长度
                            "tobit_id": model.get("tobitId")
                        })
                models_data = {"models": available_models}
                config.logger.info(f"成功从API加载 {len(available_models)} 个模型配置")
            else:
                config.logger.error(f"加载模型配置失败，状态码: {response.status_code}")
    except Exception as e:
        config.logger.error(f"加载模型配置时发生错误: {str(e)}")

# FastAPI 应用实例
app = FastAPI()

# Pydantic 模型
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False

class ChatSession:
    def __init__(self):
        self.thread_id: str = ""
        self.user_message_id: str = ""
        self.creation_time: str = ""
        self.last_active: float = time.time()
        self.model_tobit_id: Optional[int] = None  # 添加模型ID存储

# 会话存储
sessions: Dict[str, ChatSession] = {}

def generate_client_id(request: Request) -> str:
    """生成客户端 ID"""
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "")
    # 移除时间戳，使用固定的标识
    return f"{client_ip}_{user_agent}"

def get_session(client_id: str, model_tobit_id: Optional[int] = None, messages: Optional[List[Message]] = None) -> ChatSession:
    """获取或创建会话"""
    need_new_session = False
    
    # 判断是否需要创建新会话
    if messages:
        if len(messages) == 1:
            need_new_session = True
            config.logger.info("消息数量为1，需要新会话")
        elif len(messages) == 2 and messages[1].role == "user":
            need_new_session = True
            config.logger.info("消息数量为2且第二条是用户消息，需要新会话")
    
    # 如果已存在会话
    if client_id in sessions:
        session = sessions[client_id]
        session.last_active = time.time()
        config.logger.info(f"已存在会话: {session.thread_id}")
        
        # 如果需要新会话，清除旧的thread_id
        if need_new_session:
            config.logger.info("清除现有会话")
            session.thread_id = ""
        return session
    
    # 创建新会话
    session = ChatSession()
    if model_tobit_id is not None:
        session.model_tobit_id = model_tobit_id
    sessions[client_id] = session
    return session

def clean_expired_sessions():
    """清理过期会话"""
    now = time.time()
    expired = [k for k, v in sessions.items() if now - v.last_active > config.session_timeout]
    for k in expired:
        del sessions[k]

async def begin_chat(session: ChatSession, model_tobit_id: int) -> bool:
    """创建新的聊天线程"""
    config.logger.info("开始创建新的聊天线程")
    url = "https://intercom.tobit.cloud/api/thread"
    headers = {
        "Content-Type": "application/json",
        "Authorization": config.bearer_token,
        "Accept": "*/*"
    }
    data = {
        "forceCreate": True,
        "tag": "Agent",
        "thread": {
            "anonymizationForAI": False,
            "members": [
                {"tobitId": model_tobit_id},
                {"tobitId": config.user_tobit_id}
            ],
            "typeId": 8
        }
    }
    
    config.logger.debug(f"发送请求到 {url}")
    config.logger.debug(f"请求数据: {json.dumps(data, indent=2)}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            config.logger.info(f"创建线程响应状态码: {response.status_code}")
            config.logger.debug(f"响应内容: {response.text}")
            
            if response.status_code == 201:
                resp_data = response.json()
                session.thread_id = resp_data["thread"]["id"]
                if len(resp_data["thread"]["members"]) > 1:
                    session.user_message_id = resp_data["thread"]["members"][1]["id"]
                config.logger.info(f"成功创建线程，ID: {session.thread_id},用户消息ID: {session.user_message_id}")
                return True
            else:
                config.logger.error(f"创建线程失败，状态码: {response.status_code}")
                config.logger.error(f"错误响应: {response.text}")
        except Exception as e:
            config.logger.error(f"创建线程时发生异常: {str(e)}", exc_info=True)
    return False

async def send_chat_message(session: ChatSession, message: str) -> bool:
    """发送聊天消息"""
    config.logger.info(f"开始发送消息到线程 {session.thread_id}")
    url = f"https://intercom.tobit.cloud/api/thread/{session.thread_id}/message"
    headers = {
        "Content-Type": "application/json",
        "Authorization": config.bearer_token,
        "Accept": "*/*"
    }
    data = {
        "author": {"tobitId": config.user_tobit_id},
        "message": {
            "guid": str(uuid.uuid4()),
            "meta":{},
            "text": message,
            "typeId": 1
        }
    }
    
    config.logger.debug(f"发送请求到 {url}")
    config.logger.debug(f"请求数据: {json.dumps(data, indent=2)}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            config.logger.info(f"发送消息响应状态码: {response.status_code}")
            config.logger.debug(f"响应内容: {response.text}")
            
            if response.status_code == 201:
                resp_data = response.json()
                session.creation_time = resp_data["message"]["creationTime"]
                config.logger.info(f"消息发送成功，创建时间: {session.creation_time}")
                return True
            else:
                config.logger.error(f"发送消息失败，状态码: {response.status_code}")
                config.logger.error(f"错误响应: {response.text}")
        except Exception as e:
            config.logger.error(f"发送消息时发生异常: {str(e)}", exc_info=True)
    return False

async def get_chat_message(session: ChatSession) -> Optional[str]:
    """获取聊天响应"""
    config.logger.info(f"开始获取线程 {session.thread_id} 的消息")
    url = f"https://intercom.tobit.cloud/api/thread/{session.thread_id}/messages"
    headers = {
        "Content-Type": "application/json",
        "Authorization": config.bearer_token,
        "Accept": "*/*"
    }
    # 不对日期进行URL编码
    url = f"{url}?memberId={session.user_message_id}&date={session.creation_time}"
    
    config.logger.debug(f"请求URL: {url}")
    
    async with httpx.AsyncClient() as client:
        for retry in range(config.retry_max_attempts):
            try:
                response = await client.get(url, headers=headers)
                config.logger.info(f"获取消息响应状态码: {response.status_code} (重试 {retry + 1}/{config.retry_max_attempts})")
                config.logger.debug(f"响应内容: {response.text}")
                
                if response.status_code == 204:
                    config.logger.info("消息尚未准备好，等待重试")
                    await asyncio.sleep(config.retry_delay)
                    continue
                    
                if response.status_code == 200:
                    resp_data = response.json()
                    if resp_data.get("messages") and len(resp_data["messages"]) > 0:
                        message = resp_data["messages"][0]["text"]
                        config.logger.info("成功获取消息响应")
                        config.logger.debug(f"消息内容: {message}")
                        return message
                    else:
                        config.logger.warning("响应中没有消息内容")
                else:
                    config.logger.error(f"获取消息失败，状态码: {response.status_code}")
                    config.logger.error(f"错误响应: {response.text}")
            except Exception as e:
                config.logger.error(f"获取消息时发生异常: {str(e)}", exc_info=True)
                await asyncio.sleep(config.retry_delay)
    
    config.logger.error(f"在 {config.retry_max_attempts} 次重试后仍未获取到消息")
    return None

@app.get("/chaynsapi/v1/models")
async def get_models():
    """获取可用模型列表"""
    return {
        "object": "list",
        "data": models_data["models"]
    }

@app.post("/chaynsapi/v1/chat/completions")
async def chat(request: Request, chat_request: ChatRequest):
    """处理聊天请求"""
    config.logger.info("收到新的聊天请求")
    config.logger.debug(f"请求内容: {chat_request.dict()}")
    
    try:
        # 验证模型是否存在
        model_exists = False
        model_tobit_id = None
        for model in models_data["models"]:
            if model["id"] == chat_request.model:
                model_exists = True
                model_tobit_id = model["tobit_id"]
                break
        
        if not model_exists:
            config.logger.error(f"请求的模型不存在: {chat_request.model}")
            raise HTTPException(status_code=400, detail=f"Model {chat_request.model} not found")
        
        # 清理过期会话
        clean_expired_sessions()
        
        # 获取或创建会话
        client_id = generate_client_id(request)
        config.logger.info(f"客户端ID: {client_id}")
        session = get_session(client_id, model_tobit_id, chat_request.messages)
        
        # 获取用户消息
        user_message = None
        for msg in reversed(chat_request.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            config.logger.error("未找到用户消息")
            raise HTTPException(status_code=400, detail="No user message found")
        
        config.logger.info(user_message[:100] + "..." if len(user_message) > 100 else user_message)
        
        # 如果没有thread_id，创建新的对话线程
        if not session.thread_id:
            config.logger.info("创建新的对话线程")
            if not await begin_chat(session, model_tobit_id):
                config.logger.error("创建聊天线程失败")
                raise HTTPException(status_code=500, detail="Failed to create chat thread")
        else:
            config.logger.info(f"继续使用现有对话线程: {session.thread_id}")
        
        # 发送消息
        if not await send_chat_message(session, user_message):
            config.logger.error("发送消息失败")
            raise HTTPException(status_code=500, detail="Failed to send message")
        
        # 获取响应
        response_message = await get_chat_message(session)
        if not response_message:
            config.logger.error("获取响应失败")
            raise HTTPException(status_code=500, detail="Failed to get response")
        
        config.logger.info("成功获取响应")
        config.logger.debug(f"响应内容: {response_message}")
        
        # 处理流式响应
        if chat_request.stream:
            async def generate_stream():
                # 将响应分成小块
                chunk_size = 3  # UTF-8中文字符通常是3字节
                start = 0
                while start < len(response_message):
                    # 确保不会切断UTF-8字符
                    chunk_length = 0
                    i = 0
                    while i < chunk_size and (start + chunk_length) < len(response_message):
                        c = response_message[start + chunk_length].encode('utf-8')
                        chunk_length += len(c)
                        i += 1
                    
                    chunk = response_message[start:start + chunk_length]
                    
                    # 构造SSE消息
                    data = {
                        "id": f"chatcmpl-{session.user_message_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "choices": [{
                            "delta": {"content": chunk},
                            "finish_reason": None,
                            "index": 0
                        }]
                    }
                    
                    yield f"data: {json.dumps(data)}\n\n"
                    await asyncio.sleep(0.1)  # 100ms 延迟
                    start += chunk_length
                
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
        
        # 非流式响应
        return {
            "id": f"chatcmpl-{session.user_message_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [{
                "message": {
                    "content": response_message,
                    "role": "assistant"
                },
                "finish_reason": "stop",
                "index": 0
            }]
        }
    
    except Exception as e:
        config.logger.error(f"处理请求时发生异常: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 在应用启动时加载模型
@app.on_event("startup")
async def startup_event():
    await load_models()
    

if __name__ == "__main__":
    import uvicorn
    config.logger.info("启动服务器...")
    uvicorn.run(app, host=config.server_host, port=config.server_port)
