### ChromaDB的使用（QA数据集）

import chromadb
from chromadb.config import Settings
import json
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_openai import ChatOpenAI
from openai import OpenAI
import inspect
from langchain_community.embeddings import DashScopeEmbeddings, HunyuanEmbeddings

### 1. 加载并处理数据
with open('../Data/train.json', 'r', encoding = 'utf-8') as f:
    data = [json.loads(line) for line in f.readlines()]

# 提取instruction和output
instructions = [d['instruction'] for d in data]
outputs = [d['output'] for d in data]

###  2. 定义并且创建向量库对象
class MyvectorDBConnector:
    def __init__(self, collection_name):
        # 利用持久化处理
        chroma_client = chromadb.PersistentClient(path = './chroma_data')
        self.collection = chroma_client.get_or_create_collection(name = collection_name)
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"), 
            base_url ="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    def get_embeddings(self, texts, model = "text-embedding-v4", batch_size = 10):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            data = self.client.embeddings.create(
                input = batch_texts,
                model = model
            ).data
            all_embeddings.extend([x.embedding for x in data])
        return all_embeddings
    
    # embeddings部分用instructions的向量化，对应的documents用outputs
    def add_documents_and_embeddings(self, instructions, outputs):
        embeddings = self.get_embeddings(instructions)

        self.collection.add(
            documents = outputs,
            embeddings = embeddings,
            ids = [f"id{i}" for i in range(len(instructions))]
        )
        print("self.collection.count():", self.collection.count())
    def search(self, query, top_k):
        results = self.collection.query(
            query_embeddings = self.get_embeddings([query]),
            n_results = top_k
        )
        return results
vector_db = MyvectorDBConnector("demo")

### 3. 添加数据
vector_db.add_documents_and_embeddings(instructions, outputs)

### 4. 查询相关向量块
user_query = "阳痿怎么办"
results = vector_db.search(user_query, 2)

print(results)