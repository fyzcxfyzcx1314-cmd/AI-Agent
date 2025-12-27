import os
from langchain_core.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain.chat_models import init_chat_model

# 1. 初始化模型（对话模型和嵌入模型）
client = init_chat_model(model_name = "qwen3-max")
embedding_model = DashScopeEmbeddings(
    model = "text-embedding-v4",
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
)

### 文档数据
#### metadata部分可以由人工构筑，也可以利用大模型生成
docs = [
    Document(
        page_content="作者A团队开发出基于人工智能的自动驾驶决策系统，在复杂路况下的响应速度提升300%",
        metadata={"year": 2024, "rating": 9.2, "genre": "AI", "author": "A"},
    ),
    Document(
        page_content="区块链技术成功应用于跨境贸易结算，作者B主导的项目实现交易确认时间从3天缩短至30分钟",
        metadata={"year": 2023, "rating": 9.8, "genre": "区块链", "author": "B"},
    ),
    Document(
        page_content="云计算平台实现量子计算模拟突破，作者C构建的新型混合云架构支持百万级并发计算",
        metadata={"year": 2022, "rating": 8.6, "genre": "云", "author": "C"},
    ),
    Document(
        page_content="大数据分析预测2024年全球经济趋势，作者A团队构建的模型准确率超92%",
        metadata={"year": 2023, "rating": 8.9, "genre": "大数据", "author": "A"},
    ),
    Document(
        page_content="人工智能病理诊断系统在胃癌筛查中达到三甲医院专家水平，作者B获医疗科技创新奖",
        metadata={"year": 2024, "rating": 7.1, "genre": "AI", "author": "B"},
    ),
    Document(
        page_content="基于区块链的数字身份认证系统落地20省市，作者C设计的新型加密协议通过国家级安全认证",
        metadata={"year": 2022, "rating": 8.7, "genre": "区块链", "author": "C"},
    ),
    Document(
        page_content="云计算资源调度算法重大突破，作者A研发的智能调度器使数据中心能效提升40%",
        metadata={"year": 2023, "rating": 8.5, "genre": "云", "author": "A"},
    ),
    Document(
        page_content="大数据驱动城市交通优化系统上线，作者B团队实现早晚高峰通行效率提升25%",
        metadata={"year": 2024, "rating": 7.4, "genre": "大数据", "author": "B"},
    )
]

# 2. 元数据字段定义(对metadata中的数据进行说明，指导LLM解析查询条件)
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="文章的技术领域，选项:['AI '，'区块链'，'云'，'大数据']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="文章的出版年份",
        type="integer",
    ),
    AttributeInfo(
        name="author",
        description="署名文章的作者姓名",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="技术价值评估得分（1-10分）",
        type="float"
    )
]

# 3. 初始化数据库
vectordb = Chroma.from_documents(docs, embedding_model)

### 文档内容描述（指导LLM理解文档内容）
document_content_description = "技术文章简述"

# 4. 创建自查询检索器（核心组件）
'''SelfQueryRetriever 是 langchain 库中的一个工具，其主要功能是把自然语言查询转变为结构化查询，
以此提升检索的精准度。它整合了检索器和语言模型，能依据查询内容自动推断出筛选条件，还能识别出相关的元数据字段。'''
retriever = SelfQueryRetriever.from_llm(
    client,
    vectordb,
    document_content_description,# 整个文档说明
    metadata_field_info,
)

response = retriever.invoke()

#### 原理
# 构建查询解析器（分析内部工作机制用）
'''构建查询提示模板
document_content_description：对文档内容的概括性描述，例如 "有关各种主题的文章"。
metadata_field_info：元数据字段的详细描述，涵盖字段名称、类型以及描述。
此函数会生成一个提示模板，其用途是指导语言模型如何将自然语言查询转换为结构化查询。'''