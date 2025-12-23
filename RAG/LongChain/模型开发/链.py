from langchain_community.chat_models import ChatTongyi
model3 = ChatTongyi(model_name = "qwen3-max")

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate,      SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

chatPrompt = ChatPromptTemplate.from_messages([
    # ("system", "你是一个翻译模型，你需要将输入的句子翻译成{language}"),
    SystemMessagePromptTemplate.from_template("你是一个翻译模型，你需要将输入的句子翻译成{language}"),
    # ("user", "{text}"),
    HumanMessagePromptTemplate.from_template("{text}"),
    # ("assistant", "我非常抱歉，但是这个任务无法完成。"),
    AIMessagePromptTemplate.from_template("我非常抱歉，但是这个任务无法完成。")
])

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

# 链
chain = chatPrompt | model3 | parser
result = chain.invoke({"language": "英文", "text": "我喜欢编程"})