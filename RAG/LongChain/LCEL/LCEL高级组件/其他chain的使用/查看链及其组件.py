#查看链及其组件
#需要安装 pip install grandalf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_community.chat_models import ChatTongyi

#获得访问大模型客户端
client = ChatTongyi(model_name = "qwen3-max")

#使用ChatPromptTemplate，它是个大模版，可以组合各种角色的消息模板
#定义提示模版
'''LangChain提供不同类型的角色的消息模板.
最常用的是AIMessagePromptTemplate、 SystemMessagePromptTemplate和HumanMessagePromptTemplate，
分别创建代表人工智能应答消息、系统消息（提示词中的角色、任务、参考案例、约束项等等）和人类消息（具体的任务）。'''
chat_template = ChatPromptTemplate.from_messages(
    [
        #改为 ('system','请将以下的内容翻译成{language}') 也可以
        SystemMessagePromptTemplate.from_template("请将以下的内容翻译成{language}"),
        # 改为 HumanMessagePromptTemplate.from_template("{text}") 也可以
        ('human', '{text}')
    ]
)

# #2、解析返回结果
parser = StrOutputParser()

chain = chat_template | client | parser
# print(parser.input_schema.model_json_schema())
# print(parser.output_schema.model_json_schema())
print(chain.input_schema.model_json_schema())
print(chain.output_schema.model_json_schema())
print(chain.get_prompts())
#打印出链的结构图，该功能需要额外安装 pip install grandalf
chain.get_graph().print_ascii()
# print(chain.invoke({'language': '意大利文', 'text': '朋友啊再见！'}))