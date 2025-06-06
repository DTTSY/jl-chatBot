import os
import uvicorn
import json # 用于解析 WebSocket 接收到的 JSON 字符串
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect # WebSocket 相关导入
from starlette.websockets import WebSocketState # 用于检查 WebSocket 状态
from pydantic import BaseModel, ValidationError # 用于数据模型定义和验证
from typing import List, Dict, Optional, AsyncGenerator # 类型提示

from openai import OpenAI, APIError # OpenAI 客户端库及API错误类型

from chatbot_langchain_client import init_chat_model_with_memory


# --- 聊天记录存储目录 ---
CHAT_HISTORY_DIR = "chat_history"
# 应用启动时创建聊天记录目录 (如果不存在)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

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
DEFAULT_API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-xdmjxqwafrkyukzjeigetqnppifdyukszhehtyxiffljjlus") # API 密钥
DEFAULT_API_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1") # API 基础 URL
try:
    DEFAULT_CONTEXT_ROUNDS = int(os.getenv("CONTEXT_ROUNDS", 10)) # 上下文对话轮数
except ValueError:
    print("警告: CONTEXT_ROUNDS 环境变量不是一个有效的整数，将使用默认值 5。")
    DEFAULT_CONTEXT_ROUNDS = 5
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" # 默认使用的模型名称


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
                    await websocket.send_text(content) # 通过 WebSocket 发送文本块
                    full_response_content += content
        
        # 流结束后，将完整的助手回复添加到内存历史
        if session_id in session_conversations:
            assistant_message_obj = {"role": "assistant", "content": full_response_content or ""}
            session_conversations[session_id].append(assistant_message_obj)

            # 将用户输入和完整的助手回复写入文件
            if current_turn_user_message_obj: # 确保用户消息对象存在
                try:
                    filepath = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.txt")
                    with open(filepath, "a", encoding="utf-8") as f:
                        # 用户消息已由调用方写入或在此处根据messages_for_api[-1]写入
                        # 为了统一，调用方（websocket_chat_endpoint）添加用户消息到内存
                        # 此函数负责添加助手消息到内存并写入用户+助手到文件
                        f.write(f"User: {current_turn_user_message_obj['content']}")
                        f.write(f"Assistant: {assistant_message_obj['content']}")
                except Exception as e:
                    print(f"警告: 无法将会话 {session_id} 的当前轮次写入文件 {filepath}: {e}")
        # 可选：发送一个流结束的特殊标记
        # await websocket.send_text(json.dumps({"type": "stream_end", "session_id": session_id}))

    except APIError as e:
        # 如果API调用失败，从内存中移除之前添加的当前用户输入
        if session_id in session_conversations and \
           session_conversations[session_id] and \
           session_conversations[session_id][-1]["role"] == "user":
            session_conversations[session_id].pop()
        error_message = f"API 错误: {e.status_code} - {e.message if e.message else e.body}"
        print(f"会话 {session_id} 发生错误: {error_message}")
        await websocket.send_text(json.dumps({"error": error_message, "type":"error", "session_id": session_id}))
    except Exception as e:
        # 其他错误，同样移除用户输入
        if session_id in session_conversations and \
           session_conversations[session_id] and \
           session_conversations[session_id][-1]["role"] == "user":
            session_conversations[session_id].pop()
        error_message = f"发生意外错误: {str(e)}"
        print(f"会话 {session_id} 发生错误: {error_message}")
        await websocket.send_text(json.dumps({"error": error_message, "type":"error", "session_id": session_id}))


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
    client,
    model_name: str,
    messages_for_api: str,
    session_id: str
) -> str:
    """
    获取模型的单个、非流式响应。
    更新内存中的对话历史，并将当前轮次的对话追加到文件。
    如果发生错误，则从内存历史中移除当前的用户输入并重新引发异常。
    """
    global session_conversations
    full_response_content = ""

    config = {"configurable": {"session_id": session_id}}
    try:
        result = client.invoke({"input": messages_for_api}, config=config)
        
        return result.content

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
                await websocket.send_text(json.dumps({"error": "无效的JSON格式", "type":"error", "session_id": session_id_local}))
                continue
            except ValidationError as e:
                await websocket.send_text(json.dumps({"error": f"请求数据校验失败: {e.errors()}", "type":"error", "session_id": session_id_local}))
                continue
            except Exception as e: 
                await websocket.send_text(json.dumps({"error": f"解析请求数据时出错: {str(e)}", "type":"error", "session_id": session_id_local}))
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
                    await websocket.send_text(json.dumps({"error": f"服务器配置错误: {he.detail}", "type":"error", "session_id": session_id_local}))
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
                    await websocket.send_text(json.dumps({"error": f"发生内部服务器错误: {str(e)}", "type":"error", "session_id": session_id_local}))
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


@app.websocket("/chat1")
async def websocket_chat_endpoint_once(websocket: WebSocket):
    await websocket.accept()
    session_id_local = None
    try:
        while True: # 保持连接以处理多个单次请求
            data = await websocket.receive_text()
            
            try:
                request_data_dict = json.loads(data)
                request_data = WebSocketChatRequest(**request_data_dict)
                user_input = request_data.user_input
                session_id_local = request_data.session_id
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "无效的JSON格式", "type": "error", "session_id": session_id_local}))
                continue
            except ValidationError as e:
                await websocket.send_text(json.dumps({"error": f"请求数据校验失败: {e.errors()}", "type": "error", "session_id": session_id_local}))
                continue
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"解析请求数据时出错: {str(e)}", "type": "error", "session_id": session_id_local}))
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
            max_history_for_api = context_rounds * 2 + 1 

            messages_for_api = current_session_history[-max_history_for_api:] if len(current_session_history) > max_history_for_api else current_session_history[:]
            
            try:
                client = await get_openai_client(session_id_local)
                model_name = session_configs[session_id_local].get("model_name", DEFAULT_MODEL_NAME)
                
                assistant_response = await get_model_response_non_stream(
                    client, model_name, messages_for_api, session_id_local
                )
                
                await websocket.send_text(json.dumps({
                    "response": assistant_response, 
                    "session_id": session_id_local,
                    "type": "single_response"
                }))

            except (APIError, HTTPException) as he_or_ae: #捕获 get_openai_client 或 get_model_response_non_stream 的错误
                error_detail = he_or_ae.detail if isinstance(he_or_ae, HTTPException) else \
                               (f"{he_or_ae.status_code} - {he_or_ae.message if he_or_ae.message else he_or_ae.body}" if isinstance(he_or_ae, APIError) else str(he_or_ae))
                error_message_to_send = f"服务器错误: {error_detail}" if isinstance(he_or_ae, HTTPException) else f"API 错误: {error_detail}"
                
                print(f"会话 {session_id_local} (单次) 发生错误: {error_message_to_send}")
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.send_text(json.dumps({"error": error_message_to_send, "type":"error", "session_id": session_id_local}))
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
                    await websocket.send_text(json.dumps({"error": f"发生内部服务器错误: {str(e)}", "type":"error", "session_id": session_id_local}))
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


@app.websocket("/dtchat1")
async def websocket_chat_endpoint_once_dt(websocket: WebSocket):
    await websocket.accept()
    session_id_local = None
    try:
        while True: # 保持连接以处理多个单次请求
            data = await websocket.receive_text()
            
            try:
                request_data_dict = json.loads(data)
                request_data = WebSocketChatRequest(**request_data_dict)
                user_input = request_data.user_input
                session_id_local = request_data.session_id
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "无效的JSON格式", "type": "error", "session_id": session_id_local}))
                continue
            except ValidationError as e:
                await websocket.send_text(json.dumps({"error": f"请求数据校验失败: {e.errors()}", "type": "error", "session_id": session_id_local}))
                continue
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"解析请求数据时出错: {str(e)}", "type": "error", "session_id": session_id_local}))
                continue

            # 会话初始化和历史加载逻辑 (与 /chat 类似)
            config = {"configurable": {"session_id": session_id_local}}

            if session_id_local not in session_configs:
                session_configs[session_id_local] = {
                    "api_key": DEFAULT_API_KEY,
                    "base_url": DEFAULT_API_BASE_URL,
                    "context_rounds": DEFAULT_CONTEXT_ROUNDS,
                    "model_name": DEFAULT_MODEL_NAME
                }
                print(f"新会话配置已为 WebSocket (单次) 初始化: {session_id_local}")
                session_conversations[session_id_local] = []
                            
            try:
                client = await get_openai_client(session_id_local)
                client = init_chat_model_with_memory(session_id_local)
                model_name = session_configs[session_id_local].get("model_name", DEFAULT_MODEL_NAME)
                
                assistant_response = await dt_get_model_response_non_stream(
                    client, model_name, user_input, session_id_local
                )
                
                await websocket.send_text(json.dumps({
                    "response": assistant_response, 
                    "session_id": session_id_local,
                    "type": "single_response"
                }))

            except (APIError, HTTPException) as he_or_ae: #捕获 get_openai_client 或 get_model_response_non_stream 的错误
                error_detail = he_or_ae.detail if isinstance(he_or_ae, HTTPException) else \
                               (f"{he_or_ae.status_code} - {he_or_ae.message if he_or_ae.message else he_or_ae.body}" if isinstance(he_or_ae, APIError) else str(he_or_ae))
                error_message_to_send = f"服务器错误: {error_detail}" if isinstance(he_or_ae, HTTPException) else f"API 错误: {error_detail}"
                
                print(f"会话 {session_id_local} (单次) 发生错误: {error_message_to_send}")
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.send_text(json.dumps({"error": error_message_to_send, "type":"error", "session_id": session_id_local}))
                # get_model_response_non_stream 内部会处理用户消息的回滚
                # 如果是 get_openai_client 抛出的 HTTPException，且用户消息已添加，则需要在这里回滚
                # if isinstance(he_or_ae, HTTPException):
                #      if session_id_local in session_conversations and \
                #         session_conversations[session_id_local] and \
                #         session_conversations[session_id_local][-1]["role"] == "user":
                #          session_conversations[session_id_local].pop()
                #          print(f"信息: 由于 HTTPException，已从会话 {session_id_local} 的内存历史中回滚用户输入。")


            except Exception as e:
                # get_model_response_non_stream 会处理其内部错误并回滚，然后重新引发
                # 此处捕获的是 get_model_response_non_stream 重新引发的错误或其他意外错误
                print(f"处理会话 {session_id_local} (单次) 的聊天请求时发生意外失败: {e}")
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.send_text(json.dumps({"error": f"发生内部服务器错误: {str(e)}", "type":"error", "session_id": session_id_local}))
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
    print(f"流式 WebSocket 服务将在 ws://192.168.0.141:{8999}/chat 上可用")
    print(f"单次响应 WebSocket 服务将在 ws://{get_lan_ip_v1()}:{8999}/chat1 上可用")
    print
    print(f"聊天记录将存储在目录: ./{CHAT_HISTORY_DIR}/")
    api_key_display = DEFAULT_API_KEY
    if api_key_display and len(api_key_display) > 4:
        api_key_display = '*' * (len(api_key_display) - 4) + api_key_display[-4:]
    else:
        api_key_display = '未设置或过短'
    print(f"使用 API 密钥: {api_key_display}")
    print(f"使用 API 基础 URL: {DEFAULT_API_BASE_URL}")
    print(f"使用上下文轮数: {DEFAULT_CONTEXT_ROUNDS}")
    uvicorn.run(app, host="0.0.0.0", port=8999)
