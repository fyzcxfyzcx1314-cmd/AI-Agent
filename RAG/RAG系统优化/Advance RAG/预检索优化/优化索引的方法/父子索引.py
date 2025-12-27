import os
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from  langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

# 对于类似操作手册，既有文档也有表格的半结构化文档

# 1. 初始化模型
client = init_chat_model(model_name = "qwen3-max")
embedding_model = DashScopeEmbeddings(
    model = "text-embedding-v4",
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
)

# 2. 记载数据
loader = TextLoader()
docs = loader.load()

# 3. 构建文本切割器
### 父文本切割
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1024
)
### 子文本切割
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 256
)

# 3. 构建向量数据库和存储
vectordb = Chroma(collection_name = "split_parents", embedding_function = embedding_model)
store = InMemoryStore()

# 4. 创建父子文档检索器，langchain已经定义好了父子对应关系
retriever = ParentDocumentRetriever(
    vectorstore = vectordb,
    docstore = store,
    child_splitter = child_splitter,
    parent_splitter = parent_splitter,
    search_kwargs = {"k" : 1} # top k :1,相似度最高的子文档块
    #top k :2,如果是同一个父文档的两个子块，召回不会去重，所以通常设置成k=1
)
retriever.add_documents(docs)

# 5. 提示词构建
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 6. chain
chain = RunnableMap({
    "context" : lambda x : retriever.invoke(x["question"]),
    "question" : lambda x : x["question"]
}) | prompt | client | StrOutputParser()

response = chain.invoke({"question" : "deepseek"})