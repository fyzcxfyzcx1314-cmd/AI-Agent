# RunnableLambda的使用
# 作用就是能够让普通函数也能在langchain中的chain中使用，一般使用在|通道中
from operator import itemgetter
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain
from langchain_core.runnables import RunnableLambda

### 1. 定义prompt，model，parser
model = ChatTongyi(model = "qwen3-max")
prompt = ChatPromptTemplate.from_template("{a} + {b}是多少？")
parser = StrOutputParser()

### 2. 定义一般函数
def length(text):
    return len(text)

def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)

@chain
def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])

### 3. chain
pre_chain = prompt | model | parser
chain = (
    {
        "a" : itemgetter("foo") | RunnableLambda(length),
        "b" : {"text1" : itemgetter("foo"), "text2" : itemgetter("bar")} | multiple_length_function
    }
    | pre_chain
)

print(chain.invoke({"foo" : "abc", "bar" : "abcd"}))