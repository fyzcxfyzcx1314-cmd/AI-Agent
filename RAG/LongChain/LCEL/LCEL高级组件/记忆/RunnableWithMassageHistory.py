# 使用自动会话历史管理组件RunnableWithMessageHistory
# 如何流式处理大模型的回答

import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.tracers import LangChainTracer, ConsoleCallbackHandler
from langsmith import traceable
from langchain_community.chat_models import ChatTongyi

from langchain_redis import RedisChatMessageHistory

### 过程
# 用户输入 + config.session_id → 找到 Redis 中的历史记录 → 和当前输入一起发给模型 → 生成回答 → 把回答存回 Redis。

client = ChatTongyi(model_name = "qwen3-max")
prompt = ChatPromptTemplate.from_messages(
    [
    SystemMessagePromptTemplate.from_template("你是一个AI助手"),
    ("human", "{text}"),
    ]
)
parser = StrOutputParser()
chain = prompt | client | parser

# 将对话历史记录自动集成到模型调用链中来解决聊天机器人在上下文连续性和多用户支持中的核心问题
chatbot_with_history = RunnableWithMessageHistory(
    chain,
    # 根据session_id从数据库获取上下文记录
    get_session_history = lambda session_id : RedisChatMessageHistory(
        session_id = session_id,
        redis_url = "redis://localhost:6379"
    ),
    input_messages_key = "text"
)

result = chatbot_with_history.invoke(
    {
        "text" : "晚上好"
    },
    config = {
        # 配置用户信息,也就是session_id
        "configurable" : {"session_id" : "user1"}
    }
)