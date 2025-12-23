### LangSmith调试
import langchain
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.tracers import LangChainTracer, ConsoleCallbackHandler
from langchain_openai import ChatOpenAI

# 环境变量添加api_key

# 添加 LangSmith 跟踪器
tracer = LangChainTracer()
tracer.api_url = "https://api.langsmith.com"  # 强制使用正确域名
tracer.project_name = "langchaindemo2"    # 设置项目名
console_handler = ConsoleCallbackHandler()

client = ChatOpenAI(
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model = "qwen3-max",
    temperature = 0.7
)

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("请将以下的内容翻译成{language}"),
        HumanMessagePromptTemplate.from_template("{text}") 
    ]
)

parser = StrOutputParser()

chain = chat_template | client | parser

result = chain.invoke({'language': '意大利文', 'text': '朋友啊再见！'}, config={'callbacks': [console_handler, tracer]})

print(result)