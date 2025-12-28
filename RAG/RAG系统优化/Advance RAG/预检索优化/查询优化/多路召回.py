from operator import itemgetter
import os
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import BaichuanTextEmbeddings, DashScopeEmbeddings, HunyuanEmbeddings

"""
多路召回
    1. 当用户没有正确书写查询语句，或者LLM不能够正确理解用户查询语句的含义时，此时LLM生成的答案可能就不够完整和全面。
    2. 我们口语要求大模型在用户提出问题的基础上提出更多的问题。
    3. 利用这些问题检索文档。
"""

# 1. 初始化模型
client = init_chat_model(model_name = "qwen3-max")
embedding_model = DashScopeEmbeddings(
    model = "text-embedding-v4",
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
)

# 2. 加载并分割文档
loader = TextLoader()
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 100)
documents = splitter.split_documents(docs)

# 3.初始化向量数据库和检索器
vectordb = Chroma.from_documents(documents = documents, embedding = embedding_model)
retriever = vectordb.as_retriever()

# 4. 使用langchain的MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever(
    retriever = retriever,
    llm = client
)
# 多个问题检索到的文档，默认是生成3个另外的问题
unique_docs = retriever_from_llm.invoke({"question":'deepseek的应用场景'})