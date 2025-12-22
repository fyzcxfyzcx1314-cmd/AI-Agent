import numpy as np
from rank_bm25 import BM25Okapi
import jieba
import json
import chromadb
from chromadb.config import Settings
import os
from langchain_openai import ChatOpenAI
from openai import OpenAI
import inspect
from langchain_community.embeddings import DashScopeEmbeddings, HunyuanEmbeddings

### 1. 读取文件
with open('../Data/train.json', 'r', encoding = 'utf-8') as f:
    data = [json.loads(line) for line in f.readlines()]

instructions = [d['instruction'] for d in data]
outputs = [d['output'] for d in data]

### 2. 使用bm25进行全文检索，得到全文检索的，经过归一化处理后的相似度分数
def bm25_search(query):
    corpus_tokenizer = [jieba.lcut(doc) for doc in instructions]
    bm25 = BM25Okapi(corpus_tokenizer)

    query_tokenizer = jieba.lcut(query)
    scorces = np.array(bm25.get_scores(query_tokenizer))

    max_score = scorces.max()
    min_score = scorces.min()
    bm25_scores_normalized = (scorces - min_score) / (max_score - min_score)

    print("bm25_scores_normalized:", bm25_scores_normalized)
    return bm25_scores_normalized

### 3. 使用向量数据库进行向量数据库检索
class MyVectorConnector:
    def __init__(self, collection_name):
        chroma_client = chromadb.Client()
        self.collection = chroma_client.get_or_create_collection(name = collection_name)
        self.client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), 
                             base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    def get_embeddings_batch(self, texts, model = "text-embedding-v4", batch_size = 10):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            data = self.client.embeddings.create(
                input = batch_texts,
                model = model
            ).data
            all_embeddings.extend([x.embedding for x in data])
        return all_embeddings
    
    def add_documents_and_embeddings(self, instructions, outputs):
        embeddings = self.get_embeddings_batch(instructions)

        self.collection.add(
            embeddings = embeddings,
            documents = outputs,
            ids = [f"id{i}" for i in range(len(instructions))]
        )

    def search(self, query, top_k):
        results = self.collection.query(
            query_embeddings = self.get_embeddings_batch([query]),
            n_results = top_k
        )
        return results
    
def vector_search(query):
    vector_db = MyVectorConnector("demo")

    query_embedding = np.array(vector_db.get_embeddings_batch(query))
    doc_embedding = np.array(vector_db.get_embeddings_batch(instructions))

    vector_scorces = np.linalg.norm(query_embedding - doc_embedding, axis = 1)

    max_scorce = np.max(vector_scorces)
    min_scorce = np.min(vector_scorces)
    vector_scorces_normalized = 1 - (vector_scorces - min_scorce) / (max_scorce - min_scorce)
    print("vector_scores_normalized:", vector_scorces_normalized)

    return vector_scorces_normalized

### 4. 使用混合检索
def hybrid_search(query, top_k = 3, bm25_weight = 0.5):
    # 混合相似度分数
    bm25_scorces_normalized = bm25_search(query)
    vector_scorces_normalized = vector_search(query)
    hybrid_scorces = bm25_weight * bm25_scorces_normalized + (1 - bm25_weight) * vector_scorces_normalized

    # 按序号降序排序
    top_indexs = hybrid_scorces.argsort()[::-1]
    print("top_index:", top_indexs)

    # 输出结果
    hybrid_outputs = [outputs[i] for i in top_indexs[:top_k]]
    return hybrid_outputs

if __name__ == '__main__':
    query = "有病了怎么办"

    print(hybrid_search(query))