### 文档链的使用
### 作用是可以将多个文档合并为一个交给prompt

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatTongyi

client = ChatTongyi(model_name = "qwen3-max")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "根据提供的上下文: {context} \n\n 回答问题"),
        ("human", "问题：{input}")
    ]
)

# 构建链， 这个链将文档作为输入，并使用之前定义的提示词和初始化的大模型生成答案
# 链要求输入必须是一个字典，必须包含context，默认将context中的内容作为文档交给大模型
chain = create_stuff_documents_chain(client, prompt)

# 加载文档 bs:就是一个分割器
# 后面在langchain-RAG中会有各种加载器和分割器的说明
loader = WebBaseLoader(
    web_path = "https://www.gov.cn/xinwen/2020-06/01/content_5516649.htm",
    bs_kwargs = {"parse_only":bs4.SoupStrainer(id="UCAP-CONTENT")}
)

docs = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 50
)
document = text_splitter.split_documents(docs)

result = chain.invoke({"input" : "民事法律行为?", "context" : document[0 : 5]})
print(result)