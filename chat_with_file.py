import os
import uvicorn
import json # 用于解析 WebSocket 接收到的 JSON 字符串
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect # WebSocket 相关导入
from starlette.websockets import WebSocketState # 用于检查 WebSocket 状态
from pydantic import BaseModel, ValidationError # 用于数据模型定义和验证
from typing import List, Dict, Optional, AsyncGenerator # 类型提示
import base64
import httpx
from openai import OpenAI, APIError # OpenAI 客户端库及API错误类型
from llama_index.readers.file import FlatReader
from pathlib import Path
from langchain_core.runnables import Runnable,RunnableConfig
from langchain_core.messages import HumanMessage
from chatbot_langchain_client import init_chat_model_with_memory

from fileparser import parse_document_PyPDF
from chatBotUtils import user_input_paser

import logging

logging.basicConfig(level=logging.DEBUG,filename='chat_with_file.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', # 设置日志时间格式
                    filemode='a', # 设置日志记录器，追加模式
                    encoding='utf8')  # 设置日志文件编码为utf8
# 设置日志记录器
logger = logging.getLogger(__name__)
# --- 全局变量 ---
# 存储文件解析器实例

base_path = "G:/FTPFIle"
async def parse_document(file_path: str):
    reader = FlatReader()
    # 拼接文件路径
    parsed_data = reader.load_data(Path().joinpath(base_path, file_path))
    return parsed_data


# --- 聊天记录存储目录 ---
CHAT_HISTORY_DIR = "chat_history"
# 应用启动时创建聊天记录目录 (如果不存在)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

filetypes = set(['.txt', '.md', '.json', '.py', '.js', '.html','.css','.png','.jpg',".pdf"])  # 支持的文件类型
# --- Pydantic 模型定义 (用于校验 WebSocket 接收的数据) ---
class WebSocketChatRequest(BaseModel):
    user_input: str # 用户输入的内容
    session_id: str # 会话ID，用于区分不同用户的对话

# session_conversations: 存储每个会话的对话历史 (从文件加载或实时更新)
# 格式: {session_id: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
session_conversations: Dict[str, List[Dict[str, str]]] = {}

# session_configs: 存储每个会话的配置信息 (虽然现在是固定的，但结构保留)
# 格式: {session_id: {"api_key": "...", "base_url": "...", "context_rounds": ..., "model_name": "..."}}
session_configs: Dict[str, Dict] = {}

# --- FastAPI 应用初始化 ---
app = FastAPI()

# --- 服务端定义的默认配置 ---
# DEFAULT_API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-goqxmtrrniobhivswxoltmoprlwwtccrfkezefxulbccppea") # API 密钥
# DEFAULT_API_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1") # API 基础 URL

DEFAULT_API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-9d664c6305984a30899902d5ce56e468") # API 密钥
DEFAULT_API_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1") # API 基础 URL

try:
    DEFAULT_CONTEXT_ROUNDS = int(os.getenv("CONTEXT_ROUNDS", 3)) # 上下文对话轮数
except ValueError:
    print("警告: CONTEXT_ROUNDS 环境变量不是一个有效的整数，将使用默认值 5。")
    DEFAULT_CONTEXT_ROUNDS = 5
# DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-32B-Instruct" # 默认使用的模型名称
DEFAULT_MODEL_NAME = "qwen-vl-plus" # 默认使用的模型名称


async def get_openai_client(session_id: str) -> OpenAI:
    """
    根据会话ID获取其在服务端定义的配置，并返回一个OpenAI客户端实例。
    """
    config = session_configs.get(session_id)
    if not config: #理论上在主逻辑中会先初始化config
        print(f"警告: 在 get_openai_client 中未找到会话 {session_id} 的配置。将创建临时默认配置。")
        # 这种回退逻辑应该尽量避免，确保主调用流程中 session_configs[session_id] 已存在
        config = {
            "api_key": DEFAULT_API_KEY,
            "base_url": DEFAULT_API_BASE_URL,
            "context_rounds": DEFAULT_CONTEXT_ROUNDS,
            "model_name": DEFAULT_MODEL_NAME
        }
        session_configs[session_id] = config

    api_key = config.get("api_key")
    base_url = config.get("base_url")

    if not api_key:
        # 此处错误应由调用方捕获并通过WebSocket发送给客户端
        raise HTTPException(status_code=500, detail="服务器配置错误: API 密钥缺失。")
    if not base_url:
        # 此处错误应由调用方捕获并通过WebSocket发送给客户端
        raise HTTPException(status_code=500, detail="服务器配置错误: API 基础 URL 缺失。")

    return OpenAI(api_key=api_key, base_url=base_url)

#  Base64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

async def stream_model_response_ws(
    client: OpenAI, # OpenAI 客户端实例
    model_name: str, # 使用的模型名称
    messages_for_api: List[Dict[str, str]], # 发送给模型的包含上下文的消息列表
    session_id: str, # 当前会话ID
    websocket: WebSocket # WebSocket 连接对象
):
    """
    以流式方式获取模型的响应，更新内存中的对话历史，
    并将当前轮次的对话追加到文件，通过WebSocket将响应块发送给客户端。
    """
    global session_conversations # 引用全局对话历史变量
    full_response_content = "" # 用于拼接完整的模型回复内容
    current_turn_user_message_obj = None

    # messages_for_api 已经包含了当前用户的输入，它是最后一个元素
    if messages_for_api and messages_for_api[-1]["role"] == "user":
        current_turn_user_message_obj = messages_for_api[-1]

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages_for_api,
            stream=True
        )
        for chunk in stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    await websocket.send_text(content) # 通过 WebSocket 发送文本块 (原始文本，非JSON)
                    full_response_content += content
        
        # 流结束后，将完整的助手回复添加到内存历史
        if session_id in session_conversations:
            assistant_message_obj = {"role": "assistant", "content": full_response_content or ""}
            # print(f"会话 {session_id} (流式) 完整助手回复: {assistant_message_obj}")
            session_conversations[session_id].append(assistant_message_obj)

            # 将用户输入和完整的助手回复写入文件
            if current_turn_user_message_obj: # 确保用户消息对象存在
                try:
                    filepath = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.txt")
                    with open(filepath, "a", encoding="utf-8") as f:
                        # 用户消息已由调用方写入或在此处根据messages_for_api[-1]写入
                        # 为了统一，调用方（websocket_chat_endpoint）添加用户消息到内存
                        # 此函数负责添加助手消息到内存并写入用户+助手到文件
                        f.write(f"User: {current_turn_user_message_obj['content']}\n") # 添加换行符
                        f.write(f"Assistant: {assistant_message_obj['content']}\n") # 添加换行符
                except Exception as e:
                    print(f"警告: 无法将会话 {session_id} 的当前轮次写入文件 {filepath}: {e}")
        # 可选：发送一个流结束的特殊标记
        # await websocket.send_text(json.dumps({"type": "stream_end", "session_id": session_id}, ensure_ascii=False))

    except APIError as e:
        # 如果API调用失败，从内存中移除之前添加的当前用户输入
        if session_id in session_conversations and \
           session_conversations[session_id] and \
           session_conversations[session_id][-1]["role"] == "user":
            session_conversations[session_id].pop()
        error_message = f"API 错误: {e.status_code} - {e.message if e.message else e.body}"
        print(f"会话 {session_id} 发生错误: {error_message}")
        await websocket.send_text(json.dumps({"error": error_message, "type":"error", "session_id": session_id}, ensure_ascii=False))
    except Exception as e:
        # 其他错误，同样移除用户输入
        if session_id in session_conversations and \
           session_conversations[session_id] and \
           session_conversations[session_id][-1]["role"] == "user":
            session_conversations[session_id].pop()
        error_message = f"发生意外错误: {str(e)}"
        print(f"会话 {session_id} 发生错误: {error_message}")
        await websocket.send_text(json.dumps({"error": error_message, "type":"error", "session_id": session_id}, ensure_ascii=False))


async def get_model_response_non_stream(
    client: OpenAI,
    model_name: str,
    messages_for_api: List[Dict[str, str]],
    session_id: str
) -> str:
    """
    获取模型的单个、非流式响应。
    更新内存中的对话历史，并将当前轮次的对话追加到文件。
    如果发生错误，则从内存历史中移除当前的用户输入并重新引发异常。
    """
    global session_conversations
    full_response_content = ""
    current_turn_user_message_obj = None

    if messages_for_api and messages_for_api[-1]["role"] == "user":
        current_turn_user_message_obj = messages_for_api[-1]
    else:
        # This case should ideally not happen if called correctly
        print(f"警告: get_model_response_non_stream 未在 messages_for_api 中找到用户消息。会话: {session_id}")

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages_for_api,
            stream=False # 关键：非流式请求
        )
        if completion.choices and completion.choices[0].message:
            full_response_content = completion.choices[0].message.content or ""
        
        # 将助手回复添加到内存历史
        # 用户消息已由调用此函数的端点添加到内存历史
        if session_id in session_conversations:
            assistant_message_obj = {"role": "assistant", "content": full_response_content}
            print(f"会话 {session_id} (非流式) 完整助手回复: {assistant_message_obj}")
            session_conversations[session_id].append(assistant_message_obj)

            # 将用户输入和助手回复写入文件
            if current_turn_user_message_obj: # 确保用户消息对象存在
                try:
                    filepath = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.txt")
                    with open(filepath, "a", encoding="utf-8") as f:
                        f.write(f"User: {current_turn_user_message_obj['content']}\n")
                        f.write(f"Assistant: {assistant_message_obj['content']}\n")
                except Exception as e:
                    print(f"警告: 无法将会话 {session_id} 的当前轮次写入文件 {filepath} (非流式): {e}")
        
        return full_response_content

    except (APIError, Exception) as e:
        # 如果API调用或后续处理失败，从内存中移除之前由调用方添加的当前用户输入
        if session_id in session_conversations and \
           session_conversations[session_id] and \
           session_conversations[session_id][-1]["role"] == "user": # 确保最后一个是刚添加的用户消息
            # This assumes the user message that triggered this call is the last one.
            # This is true because the websocket endpoint adds it right before calling this.
            session_conversations[session_id].pop()
            print(f"信息: 已从会话 {session_id} 的内存历史中回滚用户输入，原因: {type(e).__name__}")

        # 重新引发异常，以便WebSocket端点可以捕获它并向客户端发送错误消息
        raise

async def dt_get_model_response_non_stream(
    client: Runnable,
    model_name: str,
    messages_for_api,
    session_id: str,
    websocket: Optional[WebSocket]=None, # 如果需要通过WebSocket发送响应
    stream: bool = False
) -> str:
    """
    获取模型的单个、非流式响应。
    更新内存中的对话历史，并将当前轮次的对话追加到文件。
    如果发生错误，则从内存历史中移除当前的用户输入并重新引发异常。
    """
    import uuid
    global session_conversations



    # config = {"configurable": {"session_id": session_id}}
    config = RunnableConfig(configurable={"session_id": session_id})
    logger.debug(f"会话 {session_id} 正在处理请求: {messages_for_api=}")

    try:
        if stream:
            # 如果需要流式响应，使用异步流式调用
            msg_id = uuid.uuid4()  # 生成唯一的消息ID
            async for result in client.astream({"input": str(messages_for_api)}, config=config):
                full_response_content += result.content
                # 这里可以选择将每个块发送到 WebSocket 客户端
                # await websocket.send_text(result.content) # 如果需要实时发送
                await websocket.send_text(json.dumps({
                    "type": "stream_response_chunk", # 标记为流式响应块
                    "msg_id": str(msg_id), # 发送消息ID
                    "response": result.content, 
                    "session_id": session_id,
                    "type": "stream_response"
                }, ensure_ascii=False)) # <--- 修改点
        else:
            # print(f"会话 {session_id} (非流式) 正在处理请求: {config=}")
            # logger.debug(f"会话 {session_id} (非流式) 正在处理请求: {config=}")
            result = client.invoke({"input": str(messages_for_api)}, config=config)
            full_response_content = result.content
            
        return full_response_content
    except (APIError, Exception) as e:
        # 如果API调用或后续处理失败，从内存中移除之前由调用方添加的当前用户输入
        # if session_id in session_conversations and \
        #    session_conversations[session_id] and \
        #    session_conversations[session_id][-1]["role"] == "user": # 确保最后一个是刚添加的用户消息
        #     # This assumes the user message that triggered this call is the last one.
        #     # This is true because the websocket endpoint adds it right before calling this.
        #     session_conversations[session_id].pop()
        #     print(f"信息: 已从会话 {session_id} 的内存历史中回滚用户输入，原因: {type(e).__name__}")

        # 重新引发异常，以便WebSocket端点可以捕获它并向客户端发送错误消息
        raise



# --- WebSocket 端点定义 ---
@app.websocket("/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id_local = None 
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                request_data_dict = json.loads(data)
                request_data = WebSocketChatRequest(**request_data_dict)
                user_input = request_data.user_input
                session_id_local = request_data.session_id 
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "无效的JSON格式", "type":"error", "session_id": session_id_local}, ensure_ascii=False))
                continue
            except ValidationError as e:
                await websocket.send_text(json.dumps({"error": f"请求数据校验失败: {e.errors()}", "type":"error", "session_id": session_id_local}, ensure_ascii=False))
                continue
            except Exception as e: 
                await websocket.send_text(json.dumps({"error": f"解析请求数据时出错: {str(e)}", "type":"error", "session_id": session_id_local}, ensure_ascii=False))
                continue

            if session_id_local not in session_configs:
                session_configs[session_id_local] = {
                    "api_key": DEFAULT_API_KEY,
                    "base_url": DEFAULT_API_BASE_URL,
                    "context_rounds": DEFAULT_CONTEXT_ROUNDS,
                    "model_name": DEFAULT_MODEL_NAME
                }
                print(f"新会话配置已为 WebSocket (流式) 初始化: {session_id_local}")
                session_conversations[session_id_local] = []
                filepath = os.path.join(CHAT_HISTORY_DIR, f"{session_id_local}.txt")
                if os.path.exists(filepath):
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line.startswith("User: "):
                                    session_conversations[session_id_local].append({"role": "user", "content": line[len("User: "):]})
                                elif line.startswith("Assistant: "):
                                    session_conversations[session_id_local].append({"role": "assistant", "content": line[len("Assistant: "):]})
                        print(f"已从文件 {filepath} 加载会话 {session_id_local} 的历史记录 (流式)。")
                    except Exception as e:
                        print(f"警告: 无法从文件 {filepath} 加载会话 {session_id_local} 的历史记录 (流式): {e}")
                        session_conversations[session_id_local] = [] 
                else:
                    print(f"新会话 (历史文件将创建 - 流式): {session_id_local}")
            
            session_conversations[session_id_local].append({"role": "user", "content": user_input})

            current_session_history = session_conversations[session_id_local]
            context_rounds = session_configs[session_id_local].get("context_rounds", DEFAULT_CONTEXT_ROUNDS)
            # +1 for current user input, *2 for user/assistant pairs
            max_history_for_api = context_rounds * 2 + 1 

            messages_for_api = current_session_history[-max_history_for_api:] if len(current_session_history) > max_history_for_api else current_session_history[:]
            
            try:
                client = await get_openai_client(session_id_local)
                model_name = session_configs[session_id_local].get("model_name", DEFAULT_MODEL_NAME)
                
                await stream_model_response_ws(client, model_name, messages_for_api, session_id_local, websocket)

            except HTTPException as he: 
                print(f"会话 {session_id_local} (流式) 处理时发生HTTP配置错误: {he.detail}")
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.send_text(json.dumps({"error": f"服务器配置错误: {he.detail}", "type":"error", "session_id": session_id_local}, ensure_ascii=False))
                # 回滚用户输入 (如果 get_openai_client 失败)
                if session_id_local in session_conversations and \
                   session_conversations[session_id_local] and \
                   session_conversations[session_id_local][-1]["role"] == "user":
                    session_conversations[session_id_local].pop()
            except Exception as e: 
                print(f"处理会话 {session_id_local} (流式) 的聊天请求失败: {e}")
                # stream_model_response_ws 内部会处理其特定错误并回滚
                # 此处的捕获是针对 stream_model_response_ws 之外的错误，或它未捕获的错误
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.send_text(json.dumps({"error": f"发生内部服务器错误: {str(e)}", "type":"error", "session_id": session_id_local}, ensure_ascii=False))
                # 如果错误发生在 stream_model_response_ws 外部，但用户消息已添加，则在此处回滚
                # stream_model_response_ws 已经有自己的回滚逻辑，所以这里可能不需要重复
                # 但为了安全，如果 stream_model_response_ws 抛出未处理的异常，确保回滚
                if session_id_local in session_conversations and \
                   session_conversations[session_id_local] and \
                   session_conversations[session_id_local][-1]["role"] == "user":
                    # 检查错误是否来自 stream_model_response_ws 内部，避免双重 pop
                    # 简单起见，如果 stream_model_response_ws 失败，它应该已经 pop 了
                    pass 


    except WebSocketDisconnect:
        print(f"客户端断开连接 (流式): {websocket.client} (会话: {session_id_local or '未知'})")
    except Exception as e: 
        print(f"WebSocket (流式) 通信发生严重错误 (会话: {session_id_local or '未知'}): {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close(code=1011) 
            except RuntimeError: 
                pass


# @app.websocket("/chatWithFile_d")
# async def websocket_chat_endpoint_once(websocket: WebSocket):
#     await websocket.accept()
#     session_id_local = None
#     try:
#         while True: # 保持连接以处理多个单次请求
#             data = await websocket.receive_text()
            
#             try:
#                 request_data_dict = json.loads(data)
#                 print(f'{request_data_dict=}')
#                 # request_data = WebSocketChatRequest(**request_data_dict)
#                 user_input = request_data_dict['user_input']
#                 session_id_local = request_data_dict['session_id']
#                 file_msg = request_data_dict['files']
#             except json.JSONDecodeError:
#                 await websocket.send_text(json.dumps({"error": "无效的JSON格式", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
#                 continue
#             except ValidationError as e:
#                 await websocket.send_text(json.dumps({"error": f"请求数据校验失败: {e.errors()}", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
#                 continue
#             except Exception as e:
#                 await websocket.send_text(json.dumps({"error": f"解析请求数据时出错: {str(e)}", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
#                 continue

#             # 会话初始化和历史加载逻辑 (与 /chat 类似)
#             if session_id_local not in session_configs:
#                 session_configs[session_id_local] = {
#                     "api_key": DEFAULT_API_KEY,
#                     "base_url": DEFAULT_API_BASE_URL,
#                     "context_rounds": DEFAULT_CONTEXT_ROUNDS,
#                     "model_name": DEFAULT_MODEL_NAME
#                 }
#                 print(f"新会话配置已为 WebSocket (单次) 初始化: {session_id_local}")
#                 session_conversations[session_id_local] = []
#                 filepath = os.path.join(CHAT_HISTORY_DIR, f"{session_id_local}.txt")
#                 if os.path.exists(filepath):
#                     try:
#                         with open(filepath, "r", encoding="utf-8") as f:
#                             for line in f:
#                                 line = line.strip()
#                                 if line.startswith("User: "):
#                                     session_conversations[session_id_local].append({"role": "user", "content": line[len("User: "):]})
#                                 elif line.startswith("Assistant: "):
#                                     session_conversations[session_id_local].append({"role": "assistant", "content": line[len("Assistant: "):]})
#                         print(f"已从文件 {filepath} 加载会话 {session_id_local} 的历史记录 (单次)。")
#                     except Exception as e:
#                         print(f"警告: 无法从文件 {filepath} 加载会话 {session_id_local} 的历史记录 (单次): {e}")
#                         session_conversations[session_id_local] = []
#                 else:
#                     print(f"新会话 (历史文件将创建 - 单次): {session_id_local}")
            
#             # 将当前用户输入添加到内存对话历史
#             session_conversations[session_id_local].append({"role": "user", "content": user_input})

#             current_session_history = session_conversations[session_id_local]
#             context_rounds = session_configs[session_id_local].get("context_rounds", DEFAULT_CONTEXT_ROUNDS)
#             max_history_for_api = context_rounds*2 + 1 

#             messages_for_api = current_session_history[-max_history_for_api:] if len(current_session_history) > max_history_for_api else current_session_history[:]

#             # for msg in msg_contents:
#             user_text = user_input
#             print(f"会话 {session_id_local} (单次) 收到用户输入: {user_text}, 文件: {file_msg}")

#             if file_msg:
#                 file = file_msg[0]

#                 #判断文件类型
#                 file_extension = os.path.splitext(file)[1].lower()
#                 if file_extension not in filetypes:
#                     await websocket.send_text(json.dumps({"error": f"不支持的文件类型: {file_extension}", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
#                     continue
#                 # 处理文件内容，假设文件内容是图片路径
#                 # 这里假设文件内容是图片的相对路径或URL
#                 # 例如: file['content'] = "example.png"
#                 if file_extension in ['.png', '.jpg', '.jpeg']:
#                     # 如果是图片文件，使用 base64 编码
#                     image_url = f"G:/FTPFIle/{file}"

#                     base64_image = encode_image(image_url)
#                     # image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

#                     current_msg = {
#                         "role": "user",
#                         "content": [
#                             {
#                                 "type": "image_url",
#                                 "image_url": {
#                                     "url": f"data:image/png;base64,{base64_image}"
#                                 },
#                             },
#                             {"type": "text", 
#                                 "text": f"{user_text}"
#                             }
#                         ],
#                     }
#                 elif file_extension in ['.txt', '.md', '.json', '.py', '.js', '.html', '.css']:
#                     # 如果是文本文件，直接使用文本内容
#                     # 解析文件内容
#                     parsed_files = await parse_document(file)
#                     current_msg = {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": f"{user_text}"},
#                             {"type": "text", "text": parsed_files[0].text}
#                         ]
#                     }
#                 elif file_extension in ['.pdf']:
#                     parsed_text = await parse_document_PyPDF(file)
#                     current_msg = {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": f"{user_text}"},
#                             {"type": "text", "text": parsed_text}
#                         ]
#                     }

#                 else:
#                     # 如果是其他类型的文件，发送错误消息
#                     await websocket.send_text(json.dumps({"error": f"不支持的文件类型: {file_extension}", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
#                     continue
#                 messages_for_api.append(current_msg)

#             else:
#                 # 如果没有文件，直接使用用户输入
#                 messages_for_api.append({
#                     "role": "user",
#                     "content": user_text
#                 })

#             messages_for_api.append({
#                 "role": "system",
#                 "content": "你是简乐互动的智能助手，你的回答应该尽量简洁明了，避免冗长的解释。"
#             })

#             try:
#                 client = await get_openai_client(session_id_local)
#                 model_name = session_configs[session_id_local].get("model_name", DEFAULT_MODEL_NAME)
                
#                 assistant_response = await get_model_response_non_stream(
#                     client, model_name, messages_for_api, session_id_local
#                 )
                
#                 # 在这里修改，确保中文不被转码
#                 await websocket.send_text(json.dumps({
#                     "response": assistant_response, 
#                     "session_id": session_id_local,
#                     "type": "single_response"
#                 }, ensure_ascii=False)) # <--- 修改点

#                 # print(f"会话 {session_id_local} (单次) 发送助手回复: {assistant_response}")

#             except (APIError, HTTPException) as he_or_ae: #捕获 get_openai_client 或 get_model_response_non_stream 的错误
#                 error_detail = he_or_ae.detail if isinstance(he_or_ae, HTTPException) else \
#                                (f"{he_or_ae.status_code} - {he_or_ae.message if he_or_ae.message else he_or_ae.body}" if isinstance(he_or_ae, APIError) else str(he_or_ae))
#                 error_message_to_send = f"服务器错误: {error_detail}" if isinstance(he_or_ae, HTTPException) else f"API 错误: {error_detail}"
                
#                 print(f"会话 {session_id_local} (单次) 发生错误: {error_message_to_send}")
#                 if websocket.client_state != WebSocketState.DISCONNECTED:
#                     await websocket.send_text(json.dumps({"error": error_message_to_send, "type":"error", "session_id": session_id_local}, ensure_ascii=False))
#                 # get_model_response_non_stream 内部会处理用户消息的回滚
#                 # 如果是 get_openai_client 抛出的 HTTPException，且用户消息已添加，则需要在这里回滚
#                 if isinstance(he_or_ae, HTTPException):
#                      if session_id_local in session_conversations and \
#                         session_conversations[session_id_local] and \
#                         session_conversations[session_id_local][-1]["role"] == "user":
#                           session_conversations[session_id_local].pop()
#                           print(f"信息: 由于 HTTPException，已从会话 {session_id_local} 的内存历史中回滚用户输入。")


#             except Exception as e:
#                 # get_model_response_non_stream 会处理其内部错误并回滚，然后重新引发
#                 # 此处捕获的是 get_model_response_non_stream 重新引发的错误或其他意外错误
#                 print(f"处理会话 {session_id_local} (单次) 的聊天请求时发生意外失败: {e}")
#                 if websocket.client_state != WebSocketState.DISCONNECTED:
#                     await websocket.send_text(json.dumps({"error": f"发生内部服务器错误: {str(e)}", "type":"error", "session_id": session_id_local}, ensure_ascii=False))
#                 # 用户消息的回滚已在 get_model_response_non_stream 中处理

#     except WebSocketDisconnect:
#         print(f"客户端断开连接 (单次): {websocket.client} (会话: {session_id_local or '未知'})")
#     except Exception as e:
#         print(f"WebSocket (单次) 通信发生严重错误 (会话: {session_id_local or '未知'}): {e}")
#         if websocket.client_state != WebSocketState.DISCONNECTED:
#             try:
#                 await websocket.close(code=1011)
#             except RuntimeError:
#                 pass


@app.websocket("/chatWithFile_o")
async def websocket_chat_endpoint_once_v1(websocket: WebSocket):
    await websocket.accept()
    session_id_local = None
    try:
        while True: # 保持连接以处理多个单次请求
            data = await websocket.receive_text()
            
            try:
                request_data_dict = json.loads(data)
                print(f'{request_data_dict=}')
                # request_data = WebSocketChatRequest(**request_data_dict)
                user_input = request_data_dict['user_input']
                session_id_local = request_data_dict['session_id']
                file_msg = request_data_dict['files']
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "无效的JSON格式", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
                continue
            except ValidationError as e:
                await websocket.send_text(json.dumps({"error": f"请求数据校验失败: {e.errors()}", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
                continue
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"解析请求数据时出错: {str(e)}", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
                continue

            # 会话初始化和历史加载逻辑 (与 /chat 类似)
            if session_id_local not in session_configs:
                session_configs[session_id_local] = {
                    "api_key": DEFAULT_API_KEY,
                    "base_url": DEFAULT_API_BASE_URL,
                    "context_rounds": DEFAULT_CONTEXT_ROUNDS,
                    "model_name": DEFAULT_MODEL_NAME
                }
                print(f"新会话配置已为 WebSocket (单次) 初始化: {session_id_local}")
                session_conversations[session_id_local] = []
                filepath = os.path.join(CHAT_HISTORY_DIR, f"{session_id_local}.txt")
                if os.path.exists(filepath):
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line.startswith("User: "):
                                    session_conversations[session_id_local].append({"role": "user", "content": line[len("User: "):]})
                                elif line.startswith("Assistant: "):
                                    session_conversations[session_id_local].append({"role": "assistant", "content": line[len("Assistant: "):]})
                        print(f"已从文件 {filepath} 加载会话 {session_id_local} 的历史记录 (单次)。")
                    except Exception as e:
                        print(f"警告: 无法从文件 {filepath} 加载会话 {session_id_local} 的历史记录 (单次): {e}")
                        session_conversations[session_id_local] = []
                else:
                    print(f"新会话 (历史文件将创建 - 单次): {session_id_local}")
            
            # 将当前用户输入添加到内存对话历史
            session_conversations[session_id_local].append({"role": "user", "content": user_input})

            current_session_history = session_conversations[session_id_local]
            context_rounds = session_configs[session_id_local].get("context_rounds", DEFAULT_CONTEXT_ROUNDS)
            max_history_for_api = context_rounds*2 + 1 

            messages_for_api = current_session_history[-max_history_for_api:] if len(current_session_history) > max_history_for_api else current_session_history[:]

            # for msg in msg_contents:

            # user_text = user_input
            user_text = user_input_paser(user_input)
            # print(f"会话 {session_id_local} (单次) 收到用户输入: {user_text}, 文件: {file_msg}")

            current_msg = {
                        "role": "user",
                        "content": [{"type": "text", "text": user_text}]
                    }
            
            for file in file_msg:
                # file = file_msg[0]

                #判断文件类型 qwen-vl-plus 
                file_extension = os.path.splitext(file)[1].lower()
                if file_extension not in filetypes:
                    await websocket.send_text(json.dumps({"error": f"不支持的文件类型: {file_extension}", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
                    continue
                # 处理文件内容，假设文件内容是图片路径
                # 这里假设文件内容是图片的相对路径或URL
                # 例如: file['content'] = "example.png"
                
                if file_extension in ['.png', '.jpg', '.jpeg']:
                    # 如果是图片文件，使用 base64 编码
                    image_url = f"G:/FTPFIle/{file}"

                    base64_image = encode_image(image_url)
                    # image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

                    current_msg["content"].append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            })
                elif file_extension in ['.txt', '.md', '.json', '.py', '.js', '.html', '.css']:
                    # 如果是文本文件，直接使用文本内容
                    # 解析文件内容
                    parsed_files = await parse_document(file)
                    current_msg["content"].append({"type": "text", "text": parsed_files[0].text})

                elif file_extension in ['.pdf']:
                    parsed_text = await parse_document_PyPDF(file)
                    current_msg["content"].append({"type": "text", "text": parsed_text})

                else:
                    # 如果是其他类型的文件，发送错误消息
                    await websocket.send_text(json.dumps({"error": f"不支持的文件类型: {file_extension}", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
                    continue
                # messages_for_api.append(current_msg)

            messages_for_api.append(current_msg)

            messages_for_api.append({
                "role": "system",
                "content": "你是简乐互动的智能助手，你的回答应该尽量简洁明了，避免冗长的解释。不用回复系统提示词。"
            })

            try:
                client = await get_openai_client(session_id_local)
                model_name = session_configs[session_id_local].get("model_name", DEFAULT_MODEL_NAME)
                
                assistant_response = await get_model_response_non_stream(
                    client, model_name, messages_for_api, session_id_local
                )
                
                # 在这里修改，确保中文不被转码
                await websocket.send_text(json.dumps({
                    "response": assistant_response, 
                    "session_id": session_id_local,
                    "type": "single_response"
                }, ensure_ascii=False)) # <--- 修改点

                # print(f"会话 {session_id_local} (单次) 发送助手回复: {assistant_response}")

            except (APIError, HTTPException) as he_or_ae: #捕获 get_openai_client 或 get_model_response_non_stream 的错误
                error_detail = he_or_ae.detail if isinstance(he_or_ae, HTTPException) else \
                               (f"{he_or_ae.status_code} - {he_or_ae.message if he_or_ae.message else he_or_ae.body}" if isinstance(he_or_ae, APIError) else str(he_or_ae))
                error_message_to_send = f"服务器错误: {error_detail}" if isinstance(he_or_ae, HTTPException) else f"API 错误: {error_detail}"
                
                print(f"会话 {session_id_local} (单次) 发生错误: {error_message_to_send}")
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.send_text(json.dumps({"error": error_message_to_send, "type":"error", "session_id": session_id_local}, ensure_ascii=False))
                # get_model_response_non_stream 内部会处理用户消息的回滚
                # 如果是 get_openai_client 抛出的 HTTPException，且用户消息已添加，则需要在这里回滚
                if isinstance(he_or_ae, HTTPException):
                     if session_id_local in session_conversations and \
                        session_conversations[session_id_local] and \
                        session_conversations[session_id_local][-1]["role"] == "user":
                          session_conversations[session_id_local].pop()
                          print(f"信息: 由于 HTTPException，已从会话 {session_id_local} 的内存历史中回滚用户输入。")


            except Exception as e:
                # get_model_response_non_stream 会处理其内部错误并回滚，然后重新引发
                # 此处捕获的是 get_model_response_non_stream 重新引发的错误或其他意外错误
                print(f"处理会话 {session_id_local} (单次) 的聊天请求时发生意外失败: {e}")
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.send_text(json.dumps({"error": f"发生内部服务器错误: {str(e)}", "type":"error", "session_id": session_id_local}, ensure_ascii=False))
                # 用户消息的回滚已在 get_model_response_non_stream 中处理

    except WebSocketDisconnect:
        print(f"客户端断开连接 (单次): {websocket.client} (会话: {session_id_local or '未知'})")
    except Exception as e:
        print(f"WebSocket (单次) 通信发生严重错误 (会话: {session_id_local or '未知'}): {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close(code=1011)
            except RuntimeError:
                pass

@app.websocket("/chatWithFile")
async def websocket_chat_endpoint_once_langchianMeory(websocket: WebSocket):
    await websocket.accept()
    session_id_local = None
    try:
        while True: # 保持连接以处理多个单次请求
            data = await websocket.receive_text()
            
            try:
                request_data_dict = json.loads(data)
                # print(f'{request_data_dict=}')
                logger.debug(f'{request_data_dict=}')
                # request_data = WebSocketChatRequest(**request_data_dict)
                user_input = request_data_dict['user_input']
                session_id_local = request_data_dict['session_id']
                file_msg = request_data_dict['files']
            except json.JSONDecodeError:
                logger.error(f"无效的JSON格式: {data}")
                await websocket.send_text(json.dumps({"error": "无效的JSON格式", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
                continue
            except ValidationError as e:
                logger.error(f"请求数据校验失败: {e.errors()}")
                await websocket.send_text(json.dumps({
                    "error": f"请求数据校验失败: {e.errors()}", 
                    "type": "error",
                    "session_id": session_id_local},
                    ensure_ascii=False))
                continue
            except Exception as e:
                logger.error(f"解析请求数据时出错: {str(e)}")
                await websocket.send_text(json.dumps({"error": f"解析请求数据时出错: {str(e)}", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
                continue

            # 会话初始化和历史加载逻辑 (与 /chat 类似)
            if session_id_local not in session_configs:
                session_configs[session_id_local] = {
                    "api_key": DEFAULT_API_KEY,
                    "base_url": DEFAULT_API_BASE_URL,
                    "context_rounds": DEFAULT_CONTEXT_ROUNDS,
                    "model_name": DEFAULT_MODEL_NAME
                }
                # print(f"新会话配置已为 WebSocket (单次) 初始化: {session_id_local}")
                logger.debug(f"新会话配置已为 WebSocket (单次) 初始化: {session_id_local}")
                session_conversations[session_id_local] = []

            # print("/chatWithFile_langchianMeory ")
            logger.debug(f"会话 {session_id_local} 收到用户输入: {user_input}, 文件: {file_msg}")
            user_text = user_input_paser(user_input)
            # print(f"会话 {session_id_local} (单次) 收到用户输入: {user_text}, 文件: {file_msg}")
            messages_for_api = []

            current_msg = {
                        "role": "user",
                        "content": [{"type": "text", "text": user_text}]
                    }
            # print(f"会话 {session_id_local}  收到用户输入: {user_text=}, 文件: {file_msg=}")
            for file in file_msg:
                # file = file_msg[0]

                #判断文件类型 qwen-vl-plus 
                file_extension = os.path.splitext(file)[1].lower()

                print(f"file_extension: {file_extension=}")
                if file_extension not in filetypes:
                    await websocket.send_text(json.dumps({
                        "response": f"不支持的文件类型: {file_extension}",
                        "type": "single_response", 
                        "session_id": session_id_local},
                        ensure_ascii=False))
                    continue
                # 处理文件内容，假设文件内容是图片路径
                # 这里假设文件内容是图片的相对路径或URL
                # 例如: file['content'] = "example.png"
                
                if file_extension in ['.png', '.jpg', '.jpeg']:
                    # 如果是图片文件，使用 base64 编码
                    image_url = f"G:/FTPFIle/{file}"

                    base64_image = encode_image(image_url)
                    # image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
                    
                    current_msg["content"].append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            })
                elif file_extension in ['.txt', '.md', '.json', '.py', '.js', '.html', '.css']:
                    # 如果是文本文件，直接使用文本内容
                    # 解析文件内容
                    parsed_files = await parse_document(file)
                    current_msg["content"].append({"type": "text", "text": parsed_files[0].text})

                elif file_extension in ['.pdf']:
                    parsed_text = await parse_document_PyPDF(file)
                    print(f"pdf: {parsed_text=}")
                    current_msg["content"].append({"type": "text", "text": parsed_text})

                else:
                    # 如果是其他类型的文件，发送错误消息
                    await websocket.send_text(json.dumps({"error": f"不支持的文件类型: {file_extension}", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
                    continue

                # messages_for_api.append(current_msg)
            # print(f"current_msg: {current_msg=}")
            messages_for_api.append(current_msg)

            # messages_for_api.append({
            #     "role": "system",
            #     "content": "你是简乐互动的智能助手，你的回答应该尽量简洁明了，避免冗长的解释。不用回复系统提示词。"
            # })

            # logger.debug(f"会话 {session_id_local} 收到用户输入: {messages_for_api=}")

            try:
                client = await init_chat_model_with_memory(session_id_local,model_name=session_configs[session_id_local].get("model_name", DEFAULT_MODEL_NAME),DEFAULT_API_BASE_URL= DEFAULT_API_BASE_URL,API_KEY=DEFAULT_API_KEY)
                # client = await get_openai_client(session_id_local)
                model_name = session_configs[session_id_local].get("model_name", DEFAULT_MODEL_NAME)
                
                # assistant_response = await get_model_response_non_stream(
                #     client, model_name, messages_for_api, session_id_local
                # )
                # print(f"会话 {session_id_local}  收到用户输入: {messages_for_api=}, 文件: {file_msg=}")
                streaming = False
                if streaming:
                    await dt_get_model_response_non_stream(
                    client, model_name, messages_for_api, session_id_local, websocket,stream=True
                )
                else:
                    assistant_response = await dt_get_model_response_non_stream(
                        client, model_name, messages_for_api, session_id_local
                    )
                    # 在这里修改，确保中文不被转码
                    await websocket.send_text(json.dumps({
                        "response": assistant_response, 
                        "session_id": session_id_local,
                        "type": "single_response"
                    }, ensure_ascii=False)) # <--- 修改点

                # await dt_get_model_response_non_stream(
                #     client, model_name, messages_for_api, session_id_local,websocket,stream=True
                # )
                

                # print(f"会话 {session_id_local} (单次) 发送助手回复: {assistant_response}")

            except (APIError, HTTPException) as he_or_ae: #捕获 get_openai_client 或 get_model_response_non_stream 的错误
                error_detail = he_or_ae.detail if isinstance(he_or_ae, HTTPException) else \
                               (f"{he_or_ae.status_code} - {he_or_ae.message if he_or_ae.message else he_or_ae.body}" if isinstance(he_or_ae, APIError) else str(he_or_ae))
                error_message_to_send = f"服务器错误: {error_detail}" if isinstance(he_or_ae, HTTPException) else f"API 错误: {error_detail}"
                
                print(f"会话 {session_id_local} (单次) 发生错误: {error_message_to_send}")
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.send_text(json.dumps({"error": error_message_to_send, "type":"error", "session_id": session_id_local}, ensure_ascii=False))
                # get_model_response_non_stream 内部会处理用户消息的回滚
                # 如果是 get_openai_client 抛出的 HTTPException，且用户消息已添加，则需要在这里回滚
                if isinstance(he_or_ae, HTTPException):
                    if session_id_local in session_conversations and \
                        session_conversations[session_id_local] and \
                        session_conversations[session_id_local][-1]["role"] == "user":
                          session_conversations[session_id_local].pop()
                          logger.info(f"信息: 由于 HTTPException，已从会话 {session_id_local} 的内存历史中回滚用户输入。")
                        #   print(f"信息: 由于 HTTPException，已从会话 {session_id_local} 的内存历史中回滚用户输入。")


            except Exception as e:
                # get_model_response_non_stream 会处理其内部错误并回滚，然后重新引发
                # 此处捕获的是 get_model_response_non_stream 重新引发的错误或其他意外错误
                logger.error(f"处理会话 {session_id_local} (单次) 的聊天请求时发生意外失败: {e}")
                # print(f"处理会话 {session_id_local} (单次) 的聊天请求时发生意外失败: {e}")
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.send_text(json.dumps({"error": f"发生内部服务器错误: {str(e)}", "type":"error", "session_id": session_id_local}, ensure_ascii=False))
                # 用户消息的回滚已在 get_model_response_non_stream 中处理

    except WebSocketDisconnect:
        logger.info(f"客户端断开连接 (单次): {websocket.client} (会话: {session_id_local or '未知'})")
        # print(f"客户端断开连接 (单次): {websocket.client} (会话: {session_id_local or '未知'})")
    except Exception as e:
        logger.error(f"WebSocket (单次) 通信发生严重错误 (会话: {session_id_local or '未知'}): {e}")
        # print(f"WebSocket (单次) 通信发生严重错误 (会话: {session_id_local or '未知'}): {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close(code=1011)
            except RuntimeError:
                pass


@app.websocket("/chatWithFileStream")
async def websocket_chat_endpoint_stream(websocket: WebSocket):
    await websocket.accept()
    session_id_local = None
    try:
        while True: # 保持连接以处理多个单次请求
            data = await websocket.receive_text()
            
            try:
                request_data_dict = json.loads(data)
                print(f'{request_data_dict=}')
                # request_data = WebSocketChatRequest(**request_data_dict)
                user_input = request_data_dict['user_input']
                session_id_local = request_data_dict['session_id']
                file_msg = request_data_dict['files']
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "无效的JSON格式", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
                continue
            except ValidationError as e:
                await websocket.send_text(json.dumps({"error": f"请求数据校验失败: {e.errors()}", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
                continue
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"解析请求数据时出错: {str(e)}", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
                continue

            # 会话初始化和历史加载逻辑 (与 /chat 类似)
            if session_id_local not in session_configs:
                session_configs[session_id_local] = {
                    "api_key": DEFAULT_API_KEY,
                    "base_url": DEFAULT_API_BASE_URL,
                    "context_rounds": DEFAULT_CONTEXT_ROUNDS,
                    "model_name": DEFAULT_MODEL_NAME
                }
                print(f"新会话配置已为 WebSocket (单次) 初始化: {session_id_local}")
                session_conversations[session_id_local] = []
                filepath = os.path.join(CHAT_HISTORY_DIR, f"{session_id_local}.txt")
                if os.path.exists(filepath):
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line.startswith("User: "):
                                    session_conversations[session_id_local].append({"role": "user", "content": line[len("User: "):]})
                                elif line.startswith("Assistant: "):
                                    session_conversations[session_id_local].append({"role": "assistant", "content": line[len("Assistant: "):]})
                        print(f"已从文件 {filepath} 加载会话 {session_id_local} 的历史记录 (单次)。")
                    except Exception as e:
                        print(f"警告: 无法从文件 {filepath} 加载会话 {session_id_local} 的历史记录 (单次): {e}")
                        session_conversations[session_id_local] = []
                else:
                    print(f"新会话 (历史文件将创建 - 单次): {session_id_local}")
            
            # 将当前用户输入添加到内存对话历史
            session_conversations[session_id_local].append({"role": "user", "content": user_input})

            current_session_history = session_conversations[session_id_local]
            context_rounds = session_configs[session_id_local].get("context_rounds", DEFAULT_CONTEXT_ROUNDS)
            max_history_for_api = context_rounds*2 + 1 

            messages_for_api = current_session_history[-max_history_for_api:] if len(current_session_history) > max_history_for_api else current_session_history[:]

            # for msg in msg_contents:
            user_text = user_input
            # print(f"会话 {session_id_local} (单次) 收到用户输入: {user_text}, 文件: {file_msg}")

            current_msg = {
                        "role": "user",
                        "content": [{"type": "text", "text": user_text}]
                    }
            
            for file in file_msg:
                # file = file_msg[0]

                #判断文件类型
                file_extension = os.path.splitext(file)[1].lower()
                if file_extension not in filetypes:
                    await websocket.send_text(json.dumps({"error": f"不支持的文件类型: {file_extension}", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
                    continue
                # 处理文件内容，假设文件内容是图片路径
                # 这里假设文件内容是图片的相对路径或URL
                # 例如: file['content'] = "example.png"
                
                if file_extension in ['.png', '.jpg', '.jpeg']:
                    # 如果是图片文件，使用 base64 编码
                    image_url = f"G:/FTPFIle/{file}"

                    base64_image = encode_image(image_url)
                    # image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

                    current_msg["content"].append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            })
                elif file_extension in ['.txt', '.md', '.json', '.py', '.js', '.html', '.css']:
                    # 如果是文本文件，直接使用文本内容
                    # 解析文件内容
                    parsed_files = await parse_document(file)
                    current_msg["content"].append({"type": "text", "text": parsed_files[0].text})

                elif file_extension in ['.pdf']:
                    parsed_text = await parse_document_PyPDF(file)
                    current_msg["content"].append({"type": "text", "text": parsed_text})

                else:
                    # 如果是其他类型的文件，发送错误消息
                    await websocket.send_text(json.dumps({"error": f"不支持的文件类型: {file_extension}", "type": "error", "session_id": session_id_local}, ensure_ascii=False))
                    continue
                # messages_for_api.append(current_msg)

            messages_for_api.append(current_msg)

            messages_for_api.append({
                "role": "system",
                "content": "你是简乐互动的智能助手，你的回答应该尽量简洁明了，避免冗长的解释。"
            })

            try:
                client = await get_openai_client(session_id_local)
                model_name = session_configs[session_id_local].get("model_name", DEFAULT_MODEL_NAME)

                await stream_model_response_ws(client, model_name, messages_for_api, session_id_local, websocket)


            except (APIError, HTTPException) as he_or_ae: #捕获 get_openai_client 或 get_model_response_non_stream 的错误
                error_detail = he_or_ae.detail if isinstance(he_or_ae, HTTPException) else \
                               (f"{he_or_ae.status_code} - {he_or_ae.message if he_or_ae.message else he_or_ae.body}" if isinstance(he_or_ae, APIError) else str(he_or_ae))
                error_message_to_send = f"服务器错误: {error_detail}" if isinstance(he_or_ae, HTTPException) else f"API 错误: {error_detail}"
                
                print(f"会话 {session_id_local} (单次) 发生错误: {error_message_to_send}")
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.send_text(json.dumps({"error": error_message_to_send, "type":"error", "session_id": session_id_local}, ensure_ascii=False))
                # get_model_response_non_stream 内部会处理用户消息的回滚
                # 如果是 get_openai_client 抛出的 HTTPException，且用户消息已添加，则需要在这里回滚
                if isinstance(he_or_ae, HTTPException):
                     if session_id_local in session_conversations and \
                        session_conversations[session_id_local] and \
                        session_conversations[session_id_local][-1]["role"] == "user":
                          session_conversations[session_id_local].pop()
                          print(f"信息: 由于 HTTPException，已从会话 {session_id_local} 的内存历史中回滚用户输入。")


            except Exception as e:
                # get_model_response_non_stream 会处理其内部错误并回滚，然后重新引发
                # 此处捕获的是 get_model_response_non_stream 重新引发的错误或其他意外错误
                print(f"处理会话 {session_id_local} (单次) 的聊天请求时发生意外失败: {e}")
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.send_text(json.dumps({"error": f"发生内部服务器错误: {str(e)}", "type":"error", "session_id": session_id_local}, ensure_ascii=False))
                # 用户消息的回滚已在 get_model_response_non_stream 中处理

    except WebSocketDisconnect:
        print(f"客户端断开连接 (单次): {websocket.client} (会话: {session_id_local or '未知'})")
    except Exception as e:
        print(f"WebSocket (单次) 通信发生严重错误 (会话: {session_id_local or '未知'}): {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close(code=1011)
            except RuntimeError:
                pass


def get_lan_ip_v1():
    import socket
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except socket.gaierror:
        return "无法获取IP地址"
    
# --- 主程序执行入口 (用于直接运行脚本启动服务) ---
if __name__ == "__main__":
    print("正在启动 FastAPI 应用 (WebSocket, 固定配置, 文件历史记录)...")
    ip = get_lan_ip_v1()
    print(f"流式 WebSocket 服务将在 ws://{ip}:{8999}/chat 上可用")
    print(f"单次响应 WebSocket 服务将在 ws://{ip}:{8999}/chat1 上可用")
    print(f"文件聊天 WebSocket 服务将在 ws://{ip}:{8999}/chatWithFile 上可用")
    print(f"聊天记录将存储在数据库: ./chat_history.db")
    api_key_display = DEFAULT_API_KEY
    if api_key_display and len(api_key_display) > 4:
        api_key_display = '*' * (len(api_key_display) - 4) + api_key_display[-4:]
    else:
        api_key_display = '未设置或过短'
    print(f"使用 API 密钥: {api_key_display}")
    print(f"使用 API 基础 URL: {DEFAULT_API_BASE_URL}")
    print(f"使用上下文轮数: {DEFAULT_CONTEXT_ROUNDS}")
    uvicorn.run(app, host="0.0.0.0", port=8999)
