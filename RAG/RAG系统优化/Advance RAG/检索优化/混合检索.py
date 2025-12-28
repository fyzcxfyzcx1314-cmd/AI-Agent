import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain.chat_models import init_chat_model

"""
混合索引：
    1. 利用全文搜索和相似度检索
    2. 适合异构数据场景：处理多类型、多格式数据，复杂查询场景：兼顾精确匹配与语义理解
"""

# 1. 初始化模型（对话模型和嵌入模型）
client = init_chat_model(model_name = "qwen3-max")
embedding_model = DashScopeEmbeddings(
    model = "text-embedding-v4",
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
)
question = "相关评价"

# 2. 加载并分割文档
loader = TextLoader()
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap = 50
)
documents = splitter.split_documents(docs)

# 3. 初始化向量数据库
vectordb = Chroma.from_documents(documents = documents, embedding = embedding_model)

# 4. 向量检索器和全文检索器
### 向量检索器
vector_retriever = vectordb.as_retriever(search_keargs = {"k" : 3})
### 全文检索器
BM25_retriever = BM25Retriever.from_documents(documents)
BM25_retriever.k = 3

# 5. 混合索引
ensembleRetriever = EnsembleRetriever(retrievers = [vector_retriever, BM25_retriever], weights = [0.5, 0.5])
retriever_docs = ensembleRetriever.invoke(question)

# 6. 提示词模板
template = """
根据下面的上下文回答问题：
{context}
问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 7. chain
chain = RunnableMap({
    "context" : lambda x : ensembleRetriever(x["question"]),
    "question" : lambda x : x["question"]
}) | prompt | client | StrOutputParser()

response = chain.invoke({"question" : question})