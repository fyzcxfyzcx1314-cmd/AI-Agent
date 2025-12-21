### RAG向量的相似度计算

import numpy as np
from numpy import dot
from numpy.linalg import norm
import os
from langchain_openai import ChatOpenAI
from openai import OpenAI
import inspect
from langchain_community.embeddings import DashScopeEmbeddings, HunyuanEmbeddings

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 余弦相似度
def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# 欧氏距离
def l2(a, b):
    x = np.asarray(a) - np.asarray(b)
    return norm(x)

def get_embeddings(texts, model = "text-embedding-v4"):
    data = client.embeddings.create(input = texts, model = model).data
    return [x.embedding for x in data]

query = "国际争端"
documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
]

query_vec = get_embeddings([query])[0]
doc_vec = get_embeddings(documents)

print("cos相似度")
for vec in doc_vec:
    print(cos_sim(query_vec, vec))

print("欧氏距离")
for vec in doc_vec:
    print(l2(query_vec, vec))