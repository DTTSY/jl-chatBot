import asyncio
import websockets
import json
import uuid # 用于生成唯一的会话ID



async def chat_client():
    uri = "ws://192.168.0.141:8999/chat1"  # WebSocket 服务器的地址
    # 为此客户端实例生成一个唯一的会话 ID
    session_id = str(uuid.uuid4())
    print(f"正在连接到 {uri}，会话ID: {session_id}")

    try:
        # 异步连接到 WebSocket 服务器
        async with websockets.connect(uri) as websocket:
            print("成功连接到 WebSocket 服务器。")
            print("请输入消息并按 Enter 发送。输入 'quit' 或 'exit' 断开连接。")

            while True:  # 无限循环以持续发送和接收消息
                try:
                    # 从用户获取输入。使用 asyncio.to_thread 以避免阻塞 asyncio 事件循环。
                    user_input = await asyncio.to_thread(input, f"您 (会话: {session_id[:8]}): ")

                    if user_input.lower() in ["quit", "exit"]:  # 如果用户输入 'quit' 或 'exit'
                        print("正在断开连接...")
                        break  # 跳出循环以关闭连接

                    # 准备要发送给服务器的消息体
                    message_to_send = {
                        "user_input": user_input,
                        "session_id": session_id
                    }

                    # 将消息体转换为 JSON 字符串并发送
                    await websocket.send(json.dumps(message_to_send))
                    # print(f"-> 已发送: {message_to_send}") # 可选: 打印已发送的内容

                    # 等待并接收服务器的响应
                    response_str = await websocket.recv()
                    # print(f"<- 接收到原始数据: {response_str}") # 可选: 打印接收到的原始字符串

                    try:
                        #尝试将接收到的字符串解析为 JSON 对象
                        response_json = json.loads(response_str)
                        # 以格式化的方式打印 JSON 响应，确保中文字符正确显示
                        print(f"{json.dumps(response_json, indent=2, ensure_ascii=False)}")

                    except json.JSONDecodeError:
                        # 如果响应不是有效的 JSON 格式
                        print(f"接收到非JSON格式的响应: {response_str}")

                except websockets.exceptions.ConnectionClosedOK:
                    # 如果连接被服务器正常关闭
                    print("连接已被服务器关闭。")
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    # 如果连接因错误而关闭
                    print(f"连接因错误关闭: {e}")
                    break
                except Exception as e:
                    # 捕获其他意外错误
                    print(f"发生意外错误: {e}")
                    break

    except ConnectionRefusedError:
        # 如果连接被拒绝 (例如，服务器未运行)
        print(f"连接被拒绝。服务器是否正在 {uri} 运行？")
    except Exception as e:
        # 捕获连接过程中的其他错误
        print(f"连接失败或发生错误: {e}")

if __name__ == "__main__":
    try:
        # 运行异步客户端函数
        asyncio.run(chat_client())
    except KeyboardInterrupt:
        # 处理用户通过 Ctrl+C 中断程序的情况
        print("\n 正在退出客户端。")
