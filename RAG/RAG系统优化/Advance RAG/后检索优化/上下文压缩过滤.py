import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter

"""
上下文压缩过滤：
    1. 文档块过大，实际需要的文档只占其中的一部分，我们可以对文档快进行压缩过滤，去除不需要的文档部分，保留与问题最相关的
"""

# 1. 初始化模型（对话模型和嵌入模型）
client = init_chat_model(model_name = "qwen3-max")
embedding_model = DashScopeEmbeddings(
    model = "text-embedding-v4",
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
)
### 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
query = "deepseek的发展历程"

# 2. 加载文档
loader = TextLoader()
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 100)
documents = splitter.split_documents(docs)

# 3. 初始化数据库和检索器
vectordb = Chroma.from_documents(documents = documents, embedding = embedding_model)
retriever = vectordb.as_retriever()

"""-------------------第一种：LLMChainExtractor压缩------------------(不推荐)"""
# 使用上下文压缩检索器
compressor = LLMChainExtractor.from_llm(client)
compression_retriever = ContextualCompressionRetriever(
    base_compressor = compressor,
    base_retriever = retriever
)
### 压缩后的文档
compressed_docs = compression_retriever.invoke(query)

"""-------------------第二种：LLMChainFilter压缩------------------(不推荐)"""
### LLMChainFilter 是稍微简单但更强大的压缩器
_filter = LLMChainExtractor.from_llm(client)
compression_retriever = ContextualCompressionRetriever(
    base_compressor = _filter,
    base_retriever = retriever
)
compressed_docs = compression_retriever.invoke(query)

"""-------------------第三种：EmbeddingsFilter压缩------------------(推荐)"""
### 上面的方法对每个检索到的文档都需要额外的 LLM 调用，所以既昂贵又缓慢。
### EmbeddingsFilter 通过嵌入文档和查询并仅返回那些与查询具有足够相似嵌入的文档来提供更便宜且更快的选项
### similarity_threshold = 0.6 表示只保留查询到的相似度大于等于0.6的
embedding_filter = EmbeddingsFilter(embeddings = embedding_model, similarity_threshold = 0.6)
compression_retriever = ContextualCompressionRetriever(
    base_compressor = embedding_filter,
    base_retriever = retriever
)
compressed_docs = compression_retriever.invoke(query)

"""-------------------第四种：组合压缩------------------(推荐)"""
### DocumentCompressorPipeline轻松地按顺序组合多个压缩器
'''
压缩过程：
    1.首先TextSplitters可以用作文档转换器，将文档分割成更小的块，
    2.然后EmbeddingsRedundantFilter 根据文档之间嵌入的相似性来过滤掉冗余文档，该过滤操作以文本的嵌入向量为依据，也就是借助余弦相似度来衡量文本之间的相似程度，进而判定是否存在冗余，它会把文本列表转化成对应的嵌入向量，然后计算每对文本之间的余弦相似度。一旦相似度超出设定的阈值，就会将其中一个文本判定为冗余并过滤掉。
    3.最后 EmbeddingsFilter 根据与查询的相关性进行过滤。
'''
splitter = CharacterTextSplitter(chunk_size = 300, chunk_overlap = 0, separator = ". ")
### EmbeddingsRedundantFilter 去除重复的文档块（用同一个嵌入模型为每个文档生成向量，
### 判断每两个文档块之间计算余弦相似度值是否相近，余弦相似度高（大于0.95）且相近，排名第二位的文档块过滤掉）
redundant_filter = EmbeddingsRedundantFilter(embeddings = embedding_model)
### EmbeddingsFilter 过滤掉问题和文档快召回余弦相似度小于0.6的文档块
relevant_filter = EmbeddingsFilter(embeddings = embedding_model, similarity_threshold = 0.6)
### 组合压缩
pipeline_compressor = DocumentCompressorPipeline(
    transformers = [splitter, redundant_filter, relevant_filter]
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor = pipeline_compressor,
    base_retriever = retriever
)
compressed_docs = compression_retriever.invoke(query)