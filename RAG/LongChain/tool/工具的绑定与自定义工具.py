import os

from langchain_community.chat_models import ChatTongyi
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

client = ChatTongyi(model_name = "qwen3-max")

#数据库配置
HOSTNAME ='127.0.0.1'
PORT ='3306'
DATABASE = 'world'
USERNAME = 'root'
PASSWORD ='1234'
MYSQL_URI ='mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME,PASSWORD,HOSTNAME,PORT,DATABASE)
db = SQLDatabase.from_uri(MYSQL_URI)

# 定义一个函数，用于查询表中的所有表名
def get_table_names():
    return db.get_table_names
# 包装为tool工具
get_table_names_tool = Tool(
    name = "获取表名",
    func = get_table_names, # 所绑定的函数
    description = "获取数据库中的所有表名"# 就是函数功能的描述
)

# 将工具绑定到大模型中
client_with_tool = client.bind_tools([get_table_names_tool])
response = client_with_tool.invoke([HumanMessage(content = "请从国家表中查询出China的所有数据")])