from langchain_community.document_transformers import LongContextReorder
# 5,4,3,2,1
# 倒排：1,2,3,4,5
# 前一个，后一个：5,3,1,2,4

# 按相关性排序5，4，3，2，1，5是最相关的，相关性依次递减
documents = [
    "相关性:5",
    "相关性:4",
    "相关性:3",
    "相关性:2",
    "相关性:1",
]

### 重排序算法，langchain已经定义好了
reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(documents)

print(reordered_docs)