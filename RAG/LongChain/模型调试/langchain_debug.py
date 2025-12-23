import langchain
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate,      SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

### langchain自带的调试工具, 给出链中的每一步的输入输出
langchain.debug = True

model3 = ChatTongyi(model_name = "qwen3-max")

chatPrompt = ChatPromptTemplate.from_messages([
    # ("system", "你是一个翻译模型，你需要将输入的句子翻译成{language}"),
    SystemMessagePromptTemplate.from_template("你是一个翻译模型，你需要将输入的句子翻译成{language}"),
    # ("user", "{text}"),
    HumanMessagePromptTemplate.from_template("{text}"),
    # ("assistant", "我非常抱歉，但是这个任务无法完成。"),
    AIMessagePromptTemplate.from_template("我非常抱歉，但是这个任务无法完成。")
])
parser = StrOutputParser()

chain = chatPrompt | model3 | parser
result = chain.invoke({"language": "英文", "text": "我喜欢编程"})