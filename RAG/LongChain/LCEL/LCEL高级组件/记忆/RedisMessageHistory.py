### RedisChatMessageHistory与ChatMessageHistory的使用基本相同
### ChatMessageHistory其实就是InMemoryChatMessageHistory
### 它们之间知识将上下文存储的位置不同

from gradio.themes.builder_app import history
from langchain_redis import RedisChatMessageHistory

# session_id识别身份, redis_url表示访问路径
history = RedisChatMessageHistory(
    session_id = "my_session_id",
    redis_url = "redis://localhost:6379"
)
history.add_user_message("一个问题")
result = chain.invoke(history.messages)
history.add_ai_message(result)