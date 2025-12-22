### 使用bm25库来实现
from rank_bm25 import BM25Okapi
import jieba

corpus = [
    "这是第一个文档",
    "这是第二个文档",
    "这是第三个文档"
]

### 1. 对文档进行分词操作
corpus_tokenizer = [jieba.lcut(doc) for doc in corpus]

### 2. 将分词后的语料库初始化BM25OKapi对象
bm25 = BM25Okapi(corpus_tokenizer)

### 3. 查询
query = "第一个文档"
query_tokenizer = jieba.lcut(query)

# 计算相似度分数
scores = bm25.get_scores(query_tokenizer)
print(scores)
print("=" * 50)
# 直接得到相似度最高的前top_K条数据
n_data = bm25.get_top_n(query_tokenizer, corpus, n = 1)
print(n_data)