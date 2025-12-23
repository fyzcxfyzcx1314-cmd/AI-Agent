### 结果解析，将模型输出转换为字符串或者json格式

from langchain_core.output_parsers import StrOutputParser
result = 0
### 1. 创建字符串解析器
parser = StrOutputParser()
### 2. 提取
invoke = parser.invoke(result)


### 其他类型的解析器

#### 1. CSV解析器
from langchain_core.output_parsers import CommaSeparatedListOutputParser
csv_parser = CommaSeparatedListOutputParser()

#### 2. json解析器
from langchain_core.output_parsers import JsonOutputParser
json_parser = JsonOutputParser()

#### 3. 日期时间解析器
from langchain.output_parsers import DatetimeOutputParser
data_parser = DatetimeOutputParser()