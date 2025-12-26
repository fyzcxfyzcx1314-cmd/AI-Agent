import datetime
import os
import webbrowser

from langchain_openai import ChatOpenAI
from langchain.chat_models import  init_chat_model
from langchain_core.prompts import  PromptTemplate
from langchain_core.tools import tool
from langchain.tools import tool

model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    model="qwen-max", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 工具的基本使用
@tool
def get_data():
    """
    获取现在的时间
    """
    return datetime.date.today.strftime("%Y-%m-%d")

@tool
def open_browser(url, browser_name = None):
    """
    获取网页地址，打开网站
    """
    if browser_name:
        browser = webbrowser.get(browser_name)
    else:
        browser = webbrowser
    browser.open(url)

all_tool = {
    "get_data" : get_data,
    "open_browser" : open_browser
}

# 大模型绑定工具
tool_model = model.bind_tools([get_data, open_browser])
response = tool_model.invoke("帮我打开BiliBili网页")

print(response)

# 大模型给函数调用的筛选结果，并没有直接调用工具
# 手动执行调用函数的过程
if response.tool_calls:
    for tool_call in response.tool_calls:
        tool = tool_call["name"]
        print(tool)
        print(tool_call["args"])
        # 从字典中获取工具函数体
        selected_tool = all_tool.get(tool)
        # 手动执行函数
        result = selected_tool.invoke(tool_call["args"])
        print(result)
