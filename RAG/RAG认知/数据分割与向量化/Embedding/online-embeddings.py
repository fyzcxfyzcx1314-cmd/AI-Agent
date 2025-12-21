import os
from langchain_openai import ChatOpenAI
from openai import OpenAI
import inspect
from langchain_community.embeddings import DashScopeEmbeddings, HunyuanEmbeddings

# 创建 client: 默认通义千问
#   通义千问
client_qwen = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1")


# 嵌入模型
def get_embeddings(texts, model, dimensions=1024):
    data = client_qwen.embeddings.create(input=texts, model=model, dimensions=dimensions).data
    print(data)
    print("-" * 100)
    return [x.embedding for x in data]


test_query = ["我爱你",
              "由此上溯到一千八百四十年，从那时起，为了反对内外敌人，争取民族独立和人民自由幸福，在历次斗争中牺牲的人民英雄们永垂不朽！"]

vec = get_embeddings(test_query, model = "text-embedding-v4", dimensions = 512)
print(vec)
print("=" * 100)
print(vec[0])
print("第1句话的维度:", len(vec[0]))
print("第2句话的维度:", len(vec[1]))

