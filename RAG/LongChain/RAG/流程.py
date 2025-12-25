### LangChain中使用RAG流程

from langchain_community.chat_models import ChatTongyi
import os
import bs4
import langchain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.runnables import RunnableLambda

### 1. 获得大模型
client = ChatTongyi(model_name = "qwen3-max")

### 2. 加载文档
#### 网页
loader1 = WebBaseLoader(
    web_path = ["https://www.news.cn/fortune/20250212/895ac6738b7b477db8d7f36c315aae22/c.html"],
    bs_kwargs = dict(parse_only = bs4.SoupStrainer(class_=("main-left left","title")))
)
#### docx文件
loader2 = Docx2txtLoader("人事管理流程.docx")

### 3. 将数据转换为document形式
docs1 = loader1.load()
docs2 = loader2.load()

### 4. document的切割
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)
documents = splitter.split_documents(docs1)

### 5. 嵌入模型的实例化
llm_embedding = DashScopeEmbeddings(
    model = "text-embedding-v4",
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
)

### 6. 向量数据库的实例化
db = Chroma.from_documents(documents = documents, embedding = llm_embedding)

### 7. 检索器
retriever = db.as_retriever()
#### 检索器的使用
# 1. 相似度的查询
db.similarity_search()
## 从向量数据库检索，使用chroma原始API查询   bind(k=1)表示返回相似度最高的第一个
docs_find = RunnableLambda(db.similarity_search).bind(k = 1)
# 2. 检索器不仅可以实现相似度的查询，还可以对检索到的文档进行后处理（重排，父子查询）
#### 检索器的默认查询方式也是相似度查询（l2，余弦相似度），但是可以通过allowed_search_type设置检索方式
#### 其中mmr（最大边界相似度检索）检索方式也比较常用
##### 1. mmr先进行相似度检索
##### 2. 根据相似度得分进行过滤，与已选择的结果进行相似度匹配，可以去除冗余，鼓励多样性

### 8. 提示词
system_prompt = """
您是问答任务的助理。使用以下的上下文来回答问题，
上下文：<{context}>
如果你不知道答案，不要其他渠道去获得答案，就说你不知道。
"""
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

### 9. 创建链
chain1 = create_stuff_documents_chain(client, prompt_template)
chain2 = create_retrieval_chain(retriever, chain1)

resp = chain2.invoke({"input" : "张成刚说了什么？", "context" : documents[1 : ]})

print(resp)