import os
import re
from operator import itemgetter

import langchain
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

client = ChatOpenAI(api_key = os.getenv("DASHSCOPE_API_KEY"),
    model_name = "qwen3-max",
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1")

#数据库配置
HOSTNAME ='127.0.0.1'
PORT ='3306'
DATABASE = 'world'
USERNAME = 'root'
PASSWORD ='1234'
MYSQL_URI ='mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME,PASSWORD,HOSTNAME,PORT,DATABASE)
db = SQLDatabase.from_uri(MYSQL_URI)

"""
    对于一个数据库查询问题，我们需要的步骤：
    1. 大模型要利用get_table_names工具得到数据库中所有表名与字段名
    2. 工具得到数据库信息后，交给大模型
    3. 大模型得到数据库基本信息，根据问题得到SQL查询语句
    4. 大模型利用工具执行SQL语句，得到查询信息
    5. 将上面得到的信息交给大模型后，得到最终的结果
"""

# 1. 用LangChain内置链create_sql_query_chain将大模型和数据库结合，会产生sql而不会执行sql
# 通过create_sql_query_chain将步骤中的1、2、3合起来一起做了
# sql_make_chain = create_sql_query_chain(client, db)
# resp = sql_make_chain.invoke({"question":"请从国家表中查询出China的所有数据"})
# print("产生的SQL语句：",resp)
# print("**"*15)

# 2. 因为实际产生的sql是形如```sql....```的，无法直接执行，所以需要清理
# 自定义一个输出解析器SQLCleaner,得到能够直接执行的SQL语句
class SQLCleaner(StrOutputParser):
    def parse(self, text: str) -> str:
        pattern = r'```sql(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sql = match.group(1).strip()
            # 某些大模型还会产生类似'SQLQuery:'前缀，必须去除
            sql = re.sub(r'^SQLQuery:', '', sql).strip()
            return sql
        # 某些大模型还会产生类似'SQLQuery:'前缀，必须去除
        text = re.sub(r'^SQLQuery:', '', text).strip()
        return text

sql_make_chain = create_sql_query_chain(client, db) | SQLCleaner

# 3. 构造提示词
answer_prompt = PromptTemplate.from_template(
    """
        给定一下用户问题，可能的SQL语句以及SQL的执行结果，回答用户问题：
        Question : {question}
        SQL Query : {query}
        SQL Result : {result}
    """
)

# 4. 创建一个能够执行SQL语句的工具
execute_sql_tools = QuerySQLDatabaseTool(db = db)

chain = (RunnablePassthrough.assign(query = sql_make_chain).assign(result = itemgetter('query') | execute_sql_tools) | answer_prompt | client | StrOutputParser())

final_result = chain.invoke(input = {"question" : "请从国家表中查询出China的相关数据"})