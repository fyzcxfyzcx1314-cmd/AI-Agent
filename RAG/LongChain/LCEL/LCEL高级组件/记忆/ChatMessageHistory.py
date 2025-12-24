### ChatMessageHistory是存储上下文的一种方式

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("你是人工智能助手"),
        HumanMessagePromptTemplate.from_template("text"),
        #上下文信息就存储在message中
        MessagesPlaceholder(variable_name = "message")
    ]
)

client = ChatTongyi(model_name = "qwen2-max")

parser = StrOutputParser()
chain = chat_prompt | client | parser

### 创建信息历史记录
chat_history = ChatMessageHistory()

while True:
    user_input = input("用户: ")
    if user_input == "exit":
        break
    # 添加用户输入
    chat_history.add_user_message(user_input)
    # 将所有历史信息作为提示词交给model
    result = chain.invoke({'message' : chat_history.messages})
    print(result)
    # 将大模型的输出也作为上下文历史记录存储起来
    chat_history.add_ai_message(result)