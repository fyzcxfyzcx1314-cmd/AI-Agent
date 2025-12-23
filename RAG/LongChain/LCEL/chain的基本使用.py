from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("用五句话介绍自己{topic}") |
    ChatTongyi(model = "qwen3-max", streaming = True) |
    StrOutputParser()
)

### 直接使用 | 通道
### 1. invoke（完整的数据输出）

### 2. stream（流式输出）
topic = "人生"
print(f"开始生成{topic}介绍", topic)
for chunk in chain.stream({"topic" : topic}):
    print(chunk, end = "", flush = True)
print("="*50)
### 3. batch 批量输出（利用多线程并行运算输出）
topics = ["人生", "我的人生", "余畅的人生"]
#### 必须先将所有输入转换为字典的列表形式
inputs = [{"topic" : topic} for topic in topics]

batch_results = chain.batch(inputs)
for i, res in enumerate(batch_results):
    print(f"{topics[i]} : {res}")


#### 利用RunnableSequence实现chain
from langchain_core.runnables import RunnableSequence
chain = RunnableSequence(prompt, model,parser)