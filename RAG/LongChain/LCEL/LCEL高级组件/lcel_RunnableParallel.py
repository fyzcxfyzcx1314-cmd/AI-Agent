### RunnableParallel的使用
### 作用是：RunnableLambad或者 | 的直接使用，其实是串行计算的。RunnableParallel支持并行运算

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableMap

### 1. 定义三个函数
def add1(a):
    return a + 1

def add2(a):
    return a - 1

def add3(a):
    return a

### 单独使用
chain1 = RunnableParallel(
    b = add1,
    a = add2
)
print(chain1.invoke(2))
### 混合使用
chain2 = RunnableLambda(add1) | RunnableParallel(
    x = add2,
    y = add3
)
### RunnableMap的使用，类似与RunnanleParallel