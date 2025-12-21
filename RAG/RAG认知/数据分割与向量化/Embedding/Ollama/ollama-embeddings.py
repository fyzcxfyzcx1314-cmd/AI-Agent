### 利用Ollama部署本地的嵌入模型

from ollama import Client
import ollama

#### 方法一，创建Ollama客户端实例
client = Client(host = "http://127.0.0.1:11434")
# 获取模型列表
models = client.list()
for model in models:
    print(model)
print("=" * 50)

def get_embedding1(text, model_name = "bge-m3"):
    # 使用ollama库获取嵌入向量
    response = client.embed(model_name, text)
    embedding = response['embeddings']
    return embedding

text = ["我爱你", 'hello我是Jeff，I am a goodman']

vec1 = get_embedding1(text)
print(vec1)
print("维度:",len(vec1))
print("维度:",len(vec1[0]))
print("="*50)

#### 方法二，直接利用ollama库
def get_embedding2(text, model_name = "bge-m3"):
    response = ollama.embed(model_name, text)
    embedding = response['embeddings']
    return embedding

vec2 = get_embedding2(text)
print(vec2[0])
print("维度:",len(vec2[0]))