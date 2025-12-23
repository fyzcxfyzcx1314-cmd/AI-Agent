### 提示词构建

### 方法一，直接构建
prompt1 = "请帮我将一下内容翻译成英文：我喜欢编程。"

### 方法二，提示词模板
from langchain_core.prompts import PromptTemplate
prompt2 = PromptTemplate.from_template("请将一下内容翻译成 {language} : {text}")

factprompt = prompt2.format(language="英文", text="我喜欢编程")

### 方法三，利用jinjia2或者freemarker设计带有逻辑的prompt

### 方法四,提供角色设置  System   user  人类  assistant大模型回复
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate,      SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

chatPrompt = ChatPromptTemplate.from_messages([
    # ("system", "你是一个翻译模型，你需要将输入的句子翻译成{language}"),
    SystemMessagePromptTemplate.from_template("你是一个翻译模型，你需要将输入的句子翻译成{language}"),
    # ("user", "{text}"),
    HumanMessagePromptTemplate.from_template("{text}"),
    # ("assistant", "我非常抱歉，但是这个任务无法完成。"),
    AIMessagePromptTemplate.from_template("我非常抱歉，但是这个任务无法完成。")
])
factPrompt = chatPrompt.format(language="英文", text="我喜欢编程")