### 初始化向量数据库

import os
import chromadb
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def init_chroma():
    embedding = DashScopeEmbeddings(
        model = "text-embedding-v4",
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    )
    loader = PyMuPDFLoader("deepseek-v3-1-4.pdf")
    pages = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 200,
        separators = ["\n\n", "\n", "。", ";"]
    )
    texts = splitter.create_documents([page.page_content for page in pages])
    return Chroma.from_documents(documents = texts, embedding = embedding)