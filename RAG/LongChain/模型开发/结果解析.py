### 结果解析，将模型输出转换为字符串或者json格式

from langchain_core.output_parsers import StrOutputParser
result = 0
### 1. 创建字符串解析器
parser = StrOutputParser()
### 2. 提取
invoke = parser.invoke(result)