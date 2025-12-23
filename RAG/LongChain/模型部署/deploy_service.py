### 利用LangChain部署我们的应用成为WEB服务
import os
from fastapi import FastAPI
from langchain_community.chat_models import ChatTongyi

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langserve import add_routes
from langchain_openai import ChatOpenAI

### 1. 定义链
model = ChatOpenAI(
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model = "qwen3-max",
    temperature = 0.7
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("请将下面的文字翻译成{language}"),
        HumanMessagePromptTemplate.from_template("{text}")
    ]
)

parser = StrOutputParser()

chain = prompt_template | model | parser

### 2. 使用fastapi部署服务
app = FastAPI(title = "基于Langchain的服务", version = "V1.5", description = "翻译服务")
add_routes(app, chain, path = "/langchainservice")

if __name__ == "__main___" :
    import uvicorn
    uvicorn.run(app, host = "localhost", port = "8000")