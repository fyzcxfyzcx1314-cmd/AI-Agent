### 构建模型包装的几种方式

import os
import langchain

### 1. 方法一ChatOpenAI
from langchain_openai import ChatOpenAI
model1 = ChatOpenAI(
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    model_name = "qwen3-max",
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

### 2. 方法二init_chat_model（最常用）
from langchain.chat_models import init_chat_model
model2 = init_chat_model("qwen3-max", model_provider = "aliyun")

### 3. 方法三采用对应model的社区库
from langchain_community.chat_models import ChatTongyi
model3 = ChatTongyi(model_name = "qwen3-max")