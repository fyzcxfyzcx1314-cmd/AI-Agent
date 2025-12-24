# RunnablePassthrough的使用

from langchain_core.runnables import RunnablePassthrough, RunnableParallel

### 1. 作用一 : 原样进行数据传送
runnable = RunnableParallel(
    passed = RunnablePassthrough(),
    modified = lambda x : x["num"] + 1
)
print(runnable.invoke({"num" : 1}))

### 作用二 ： 对数据增强后传递
#### RunnablePassthrough.assign会创建一个新的字典，包含原始所有字段以及新指定的字段
chain = RunnableParallel(
    passed = RunnablePassthrough.assign(query = lambda x : x["num"] + 2),
    modified = lambda x : x["num"] + 1
)
print(chain.invoke({"num" : 1}))