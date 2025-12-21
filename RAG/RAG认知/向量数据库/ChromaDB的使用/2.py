### ChromaDB的使用（test文本数据集）

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_openai import ChatOpenAI
from openai import OpenAI
import inspect
from langchain_community.embeddings import DashScopeEmbeddings, HunyuanEmbeddings

### 1. 加载数据
with open('../Data/deepseek百度百科.txt', 'r', encoding='utf-8') as f:
    content = f.read()
print(len(content))

### 2. 定义并且创建向量库对象
class MyVectorDBConnector:
    # 初始化
    def __init__(self, collection_name):
        ### 加载到内存当中
        chroma_client = chromadb.Client()
        ### 持久化处理
        ### chroma_client=chromadb.PersistentClient(path="./chroma_data")

        ### 创建集合
        self.collection = chroma_client.get_or_create_collection(name = collection_name)

        ### 使用在线向量化
        self.client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url = 
                             "https://dashscope.aliyuncs.com/compatible-mode/v1")
        
    # 分批次向量化（当前通义千问的模型最多支持10个文件同时向量化）
    def get_embeddings_batch(self, texts, model = "text-embedding-v4", batch_size = 10):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            data = self.client.embeddings.create(input = batch_texts, model = model).data
            all_embeddings.extend([x.embedding for x in data])
        return all_embeddings
    
    # 集合中文档，向量和id的添加
    def add_documents_and_embeddings(self, documents):
        embeddings = self.get_embeddings_batch(documents)

        self.collection.add(
            embeddings = embeddings,
            documents = documents,
            ids = [f"id{i}" for i in range(len(documents))]
        )
        print("self.collection.count():", self.collection.count())

    # 检索向量数据库
    def search(self, query, top_k):
        results = self.collection.query(
            query_embeddings = self.get_embeddings_batch([query]),
            n_results = top_k
        )
        return results
vector_db = MyVectorDBConnector("demo")

### 3. 文档切分，利用longchain的函数
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    separators = ["\n\n", "\n", "。", "?", "，", ""]
)
texts = splitter.split_text(content)

# 4. 将文档（不必须），embeddings（必须），ids（必须）加入向量数据库的一个集合中
vector_db.add_documents_and_embeddings(texts)

# 5. 开始检索
user_query = "deepseek的发展历程"
query_results = vector_db.search(user_query, 5)
### 将查询到的五个相关结果合并
contents = '\n'.join(query_results['documents'][0])

# 6. 构建提示词
prompt = f"""
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回复完全依据下述已知信息。不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

已知信息:
{contents}

----
用户问：
{user_query}

请用中文回答用户问题。
"""

# 7. 调用大模型api
def get_completion(prompt, model = "qwen3-max"):
    client_model = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    message = [{"role" : "user", "content" : prompt}]
    response = client_model.chat.completions.create(
        model = model,
        messages = message,
        temperature = 0
    )
    return response.choices[0].message.content

print(get_completion(prompt))