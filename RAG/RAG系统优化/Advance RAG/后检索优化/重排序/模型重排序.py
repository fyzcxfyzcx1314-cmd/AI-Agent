from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank
import os

# 1. 初始化模型（对话模型和嵌入模型）
client = init_chat_model(model_name = "qwen3-max")
embedding_model = DashScopeEmbeddings(
    model = "text-embedding-v4",
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
)

query = "孕妇感冒了怎么办"
documents = [
    "感冒应该吃999感冒灵",
    "高血压患者感冒了吃什么",
    "感冒了可以吃感康，但是孕妇禁用",
    "感冒了可以咨询专业医生"
]

### 重排序模型
top_n = 3
reranker = DashScopeRerank(
        model = "text-embedding-v3",
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY"),
        top_n = top_n
)

### 重排序的得分
score = reranker.rerank(documents, query)
