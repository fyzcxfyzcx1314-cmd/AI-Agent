import gradio as gr
import os
import json
import random
import subprocess
import webbrowser
from datetime import datetime
from http import HTTPStatus
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import MessagesPlaceholder
from langchain.agents.agent import AgentOutputParser
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler

# 1. 初始化历史消息
INIT_MESSAGE = SystemMessage(content="你是由通义千问提供的人工智能助手，你是百科全书。")

# 2. 定义工具函数
def get_data(input : str = "") -> str:
    """获取当前时间"""
    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    result = f"当前时间 : {formatted_time}。"
    return result
def recom_drink(input: str = "") -> str:
    """推荐附近的饮品店"""
    result = '''距离您500米内有如下饮料店：\n
    1、蜜雪冰城\n
    2、茶颜悦色\n
    另外距离您200米内有惠民便利店，里面应该有矿泉水或其他饮品'''
    return result
def open_calc(input: str = "") -> str:
    """打开计算器"""
    try:
        subprocess.Popen(['calc.exe'])
        return "计算器已打开"
    except Exception as e:
        return f"打开计算器失败: {str(e)}"
def open_browser(url: str) -> str:
    """打开浏览器访问指定网址"""
    try:
        webbrowser.open(url)
        return f"已打开浏览器访问 {url}"
    except Exception as e:
        return f"打开浏览器失败: {str(e)}"
    
# 3. 用langchain包装函数
tools = [
    Tool(
        name = "get_current_time",
        func = get_data,
        description = "当你想知道现在的时间时非常有用。"
    ),
    Tool(
        name="recom_drink",
        func=recom_drink,
        description="为您推荐附近的饮品店"
    ),
    Tool(
        name="open_calc",
        func=open_calc,
        description="打开本地计算机上的计算器。"
    ),
    Tool(
        name="open_browser",
        func=lambda url: open_browser(url),
        description="打开本地计算机上的网页浏览器，并接受网站的url作为参数。"
    )
]

# 4. 定义模型
client = ChatOpenAI(
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    model_name = "qwen3-max",
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 5. 创建记忆存储
### 创建记忆 ConversationBufferWindowMemory 用于管理对话历史的记忆机制，它通过滑动窗口（Sliding Window） 的方式只保留最近若干轮的对话内容
### ConversationBufferMemory 完整记忆所有历史
memory = ConversationBufferWindowMemory(
    memory_key = "chat_history",
    return_messages = True, # 适配聊天模型,将历史记录转换为聊天模型可读的格式（如列表形式），否则模型无法正确加载上下文
    k = 10 # 仅保留最近的 k 轮人类与AI的互动内容
)

# 6. 初始化记忆
memory.chat_memory.add_message(INIT_MESSAGE)

# 7. 初始化Agent
agent_kwargs = {
    "extra_prompt_messages" : [MessagesPlaceholder(variable_name = "chat_history")]
}
agent = initialize_agent(
    tools,
    client,
    agent = AgentType.OPENAI_FUNCTIONS,
    verbose = True,
    memory = memory,
    agent_kwargs = agent_kwargs,
    handle_parsing_errors = True
)

# 8. 处理模型响应
def process_llm_response(query, show_history):
    if len(query) == 0:
        return show_history + [("", "")]
    try:
        # 显示用户查询和等待提示
        yield show_history + [(query, "正在查询大模型...")]

        # 使用Agent处理查询
        response = agent.run(query)

        # 返回结果
        yield show_history + [(query, response)]
    except Exception as e:
        print(f"Error: {e}")
        yield show_history + [(query, "AI助手出错，请重试或者检查")]

### 前端页面展示
with gr.Blocks(title="大模型中Function Call演示") as demo:
    gr.HTML('<center><h1>欢迎大模型中Function Call演示</h1></center>')

    with gr.Row():
        with gr.Column(scale=10):
            chatbot = gr.Chatbot(value=[["hello", "很高兴见到您！我是AI助手小可爱，希望可以帮到你"]], height=650)

    with gr.Row():
        msg = gr.Textbox(label="输入", placeholder="您想了解什么呢？")

    with gr.Row():
        examples = gr.Examples(
            examples=[
                '请问如何做红烧牛肉？',
                '料酒可以换成白酒吗？',
                '帮我打开计算器',
                '现在几点了？',
                '帮我访问淘宝网',
                '我渴了'
            ],
            inputs=[msg]
        )

    clear = gr.ClearButton([chatbot, msg])
    msg.submit(process_llm_response, [msg, chatbot], [chatbot])

if __name__ == '__main__':
    demo.launch(server_port=7778)