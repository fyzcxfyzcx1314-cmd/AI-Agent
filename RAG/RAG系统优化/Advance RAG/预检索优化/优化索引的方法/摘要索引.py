import os
from langchain.storage import InMemoryByteStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import MultiVectorRetriever
import uuid  
from langchain_core.documents import Document  
from langchain_core.output_parsers import StrOutputParser  
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.runnables import RunnableMap
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings

"""
步骤：
    1. 将文档提取，分割成比较大的块（一般针对结构化的文档，表格之类，还有类似法律条款，不能分割为小块）
    2. 利用大模型将分割的块生成对应的摘要（摘要也要相对详细一点，太短会失去大部分语义信息）
    3. 将摘要进行向量化存储到向量数据库中，并将摘要与原始文档进行id对应，同时原始文档也存储到数据库中（不是向量数据库）
    4. 检索，匹配摘要，返回原始文档块
"""
"""
注意：
    1. 摘要质量非常重要，不能太简略
    2. 使用摘要索引的成本更大，不仅需要存储两部分信息，还需要利用大模型生成摘要
    3. 一致性问题，一般使用uuid生成唯一的id将摘要与原始文档对齐
"""

# 1. 初始化模型（对话模型和嵌入模型）
client = init_chat_model(model_name = "qwen3-max")
embedding_model = DashScopeEmbeddings(
    model = "text-embedding-v4",
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
)

# 2. 文档加载与切割
loader = TextLoader("")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1024,
    chunk_overlpa = 200
)
documents = splitter.split_documents(docs)

# 3. 生成摘要
### x.page_content 就是这个文档块的纯文本内容（不带 metadata）
### doc.page_content	属性（Attribute）	直接获取存储的文本内容
### doc.some_method()	方法（Method）	需要加 ()，会执行一段逻辑
chain = (
    {"doc" : lambda x : x.page_content}
    | ChatPromptTemplate.from_template("总结下面的文档:\n\n{doc}")
    | client
    | StrOutputParser()
)
### 利用batc并发执行链
summaries = chain.batch(documents, {"max_concurrency" : 5})

# 4. 初始化向量数据库
vectordb = Chroma(
    collection_name = "summaries",
    embedding_function = embedding_model
)

# 5. 初始化内存存储（存储原始文档）
store = InMemoryByteStore()

# 6. 摘要对齐
docs_id = [str(uuid.uuid4()) for _ in documents]

# 7. 使用多向量检索器进行检索（多向量检索器可以自动匹配并查询原始文档，由retriever直接返回chunk）
retriever = MultiVectorRetriever(
    vectorstore = vectordb,
    byte_store = store,
    id_key = "docs_id"
)

# 8. 将摘要转换为document形式
summary_docs = [
    Document(page_content = doc, metadata = {id : docs_id[i]})
    for i, doc in enumerate(summaries)
]

# 9. 存储数据库
retriever.vectorstore.add_documents(summary_docs)
### 将原始文档存储到字节存储（使用ID关联）
### 把 doc_ids（UUID 列表）和 docs（原始 chunk 列表）一一配对,存入 byte_store（即 InMemoryByteStore 实例）
retriever.docstore.mset(list(zip(docs_id, documents)))

# 10. 构建提问提示词
prompt =  ChatPromptTemplate.from_template("根据下面的文档回答问题:\n\n{doc}\n\n问题: {question}") 

# 11. chain
chain = RunnableMap({
    "doc" : lambda x : retriever.invoke(x["question"]),
    "question" : lambda x : x["question"]
}) | prompt | client | StrOutputParser()