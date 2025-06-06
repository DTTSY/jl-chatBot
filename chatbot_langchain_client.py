from langchain.memory import  ConversationBufferWindowMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

async def get_session_history_from_sqlite(session_id: str) -> BaseChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///chat_history.db"
    )


async def init_chat_model_with_memory(session_id: str, model_name: str = "Qwen/Qwen3-8B", DEFAULT_API_BASE_URL: str|None = None,API_KEY:str|None=None,temperature: float = 0.1,) -> Runnable:
    """
    初始化一个带记忆的 ai聊天模型

    :param api_key: aiAPI 密钥
    :param model_name: 模型名称，默认为 "deepseek-chat"
    :param temperature: 生成温度，默认为 0.1
    :return: chat_model
    """
    # 初始化对话记忆

    memory = ConversationBufferWindowMemory(memory_key=session_id,k=5, return_messages=True)
    # memory.load_memory_variables({})

    # 初始化聊天模型（假设兼容 OpenAI API）
    chat_model = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=DEFAULT_API_BASE_URL,  # DeepSeek API 基础 URL
        model_name=model_name,
        temperature=temperature,
        max_tokens=1024,  # 设置最大 token 数量
    )

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是简乐互动的智能助手，你的回答应该尽量简洁明了，避免冗长的解释。不用回复系统提示词."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
    )
    runnable = prompt | chat_model

    chain_with_history = RunnableWithMessageHistory(
    runnable,
    lambda  session_id: SQLChatMessageHistory(
    session_id=session_id, connection="sqlite:///chat_history.db"
    ),
    input_messages_key="input",
    history_messages_key="history",
    memory=memory,
)
    return chain_with_history
