import uuid

import chromadb
from chromadb.config import Settings
from functools import wraps
from pypinyin import pinyin, Style
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_openai import ChatOpenAI
from openai import OpenAI
import inspect
from langchain_community.embeddings import DashScopeEmbeddings, HunyuanEmbeddings

# 向量数据库
class MyVectorDBConnector:

    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path = "./chroma")
        self.client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), 
                             base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1")

    def get_embeddings_batch(self,texts, model="text-embedding-v4", batch_size=10):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_text = texts[i:i + batch_size]
            data = self.client.embeddings.create(input=batch_text, model=model).data
            all_embeddings.extend([x.embedding for x in data])
        return all_embeddings


    def add_documents_and_embeddings(self, documents, collection_name='demo'):
        print('add_documents: collection_name:', collection_name)
        # 创建or获取一个 collection
        collection = self.chroma_client.get_or_create_collection(name=collection_name)

        batch_size = 10
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i: i + batch_size]
            collection.add(
                embeddings=self.get_embeddings(batch_docs),
                documents=batch_docs,
                ids=[str(uuid.uuid4()) for _ in batch_docs]
            )

    def search(self, query, collection_name='demo', n_results=5):
        '''检索向量数据库'''
        print('search: collection_name:', collection_name)
        collection = self.chroma_client.get_or_create_collection(name=collection_name)
        results = collection.query(
            query_embeddings=self.get_embeddings([query]),  
            n_results=n_results 
        )
        return results
    
# 读取文档
def extract_text_from_docs(filename):
    full_text = ''
    # 读取文档
    doc = Document(filename)
    for para in doc.paragraphs:
        if para.text.strip():
            full_text += para.text + '\n'
    
    splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 50)
    documents = splitter.split_text(full_text)

    return documents

# 访问大模型
def get_completion(prompt, model = 'qwen3-max'):
    client = OpenAI(
        api_key = os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    message = [{"role" : "user", "content" : prompt}]
    response = client.chat.completions.create(
        model = model,
        messages = message
    )
    return response.choices[0].message.content

# 装饰器：中文->英文
def to_pinyin(fn):
    @wraps(fn)
    def chinese_to_pinyin(*args, **kwargs):
        chinese_name = kwargs['collection_name']
        # 把.去掉
        chinese_name = chinese_name.replace('.', '')

        # 中文->拼音
        pinyin_list = pinyin(chinese_name, style=Style.NORMAL, heteronym=False)
        # 将拼音列表转换为字符串
        pinyin_str = ''.join([word[0].lower() for word in pinyin_list])
        kwargs['collection_name'] = pinyin_str
        return fn(*args, **kwargs)
    return chinese_to_pinyin