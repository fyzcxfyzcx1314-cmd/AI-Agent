import langchain
from typing import List
from langchain.storage import InMemoryByteStore
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import MultiVectorRetriever
import uuid
from langchain_core.documents import Document  
from langchain_core.output_parsers import StrOutputParser  
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.runnables import RunnableMap, RunnableParallel
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
import json
import os

"""
应用场景:
    1. 用户的表述性差异导致用户用词不等于文档用词，这种通过模型生成多种提问，覆盖同义，转述，反问等表达
    2. 问答视角不一样，文档是陈述句，用户是疑问句，可以主动将陈述转换为疑问，对齐用户输入
    3. 隐含信息无法检索，可以生成推理型问题，暴露隐含关系
    4. 检索召回率低，可以扩展向量空间的”查询锚点“ ，提高命中率
"""
# 1. 初始化模型
client = init_chat_model(model_name = "qwen3-max")
embedding_model = DashScopeEmbeddings(
    model = "text-embedding-v4",
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
)

# 2. 文档加载
loader = TextLoader()
docs = loader.load()

# 3. 初始化存储
vectordb = Chroma(collection_name = "hypo-question", embedding_function = embedding_model)
store = InMemoryByteStore()

# 4. 建立唯一索引
doc_ids = [str(uuid.uuid4()) for _ in docs]

# 5. 构建检索器
retriever = MultiVectorRetriever(
    vectorstore = vectordb,
    byte_store = store,
    id_key = "doc_ids"
)

# 6. 定义约束类来限制大模型生成问题的格式
class HypotheticalQuestion(BaseModel):
    """约束生成假设性问题的格式"""
    question: List[str] = Field(..., description = "List of question")

# 7. 构建问题生成的提示词
prompt = ChatPromptTemplate.from_template(
    """
        请基于以下文档生成3个假设性问题（必须使用JSON格式）:
        {doc}
        要求：
        1. 输出必须为合法JSON格式，包含questions字段
        2. questions字段的值是包含3个问题的数组
        3. 使用中文提问
        示例格式：
        {{
            "questions": ["问题1", "问题2", "问题3"]
        }}
    """
)

# 8. 构造提问链
'''
其中的client.with_structured_output可以理解为输出解析器的一种更高级用法
将大模型的输出转换为HypotheticalQuestions所限定的格式，
而HypotheticalQuestions要求的格式是：
定义了一个字段 questions，它具有以下特性：
类型注解：List[str] 表示 questions 字段应该是一个字符串列表。
必需性：Field(...) 中的省略号 ... 表示这个字段是必需的。
描述信息：description="List of questions" 为该字段添加了描述，这对于生成文档或帮助理解模型结构很有用。
'''
question_chain = (
    {"doc" : lambda x : x.page_content}
    | prompt
    | client.with_structured_output(
        HypotheticalQuestion
    )
    # 提取问题列表
    | (lambda x : x.questions)
)
# 假设性问题列表
hypothetical_questions = question_chain.batch(docs, {"max_concurrency" : 5})
question_docs = []
for i, question_list in enumerate(hypothetical_questions):
    question_docs.extend(
        {Document(page_content = s, metadata = {id : doc_ids[i]}) for s in question_list}
    )

# 9. 文档加入存储
retriever.vectorstore.add_documents(question_docs)
retriever.byte_store.mset(list(zip(doc_ids, docs)))

# 10. 提示词
final_prompt = ChatPromptTemplate.from_template("根据下面的文档回答问题:\n\n{doc}\n\n问题: {question}")

# 11. chain
chain = RunnableMap({
    "doc" : lambda x : retriever.invoke(x["question"]),
    "question" : lambda x : x["question"]
}) | final_prompt | client | StrOutputParser()

response = chain.invoke({"question" : "deepseek"})