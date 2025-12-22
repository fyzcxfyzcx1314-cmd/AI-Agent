from function_tools import *

### 1. 上传文件
vector_db = MyVectorDBConnector

# 上传到向量库
@to_pinyin
def save_to_db(filepath, collection_name = "demo"):
    documents = ''
    if filepath.endswith('.docx') or filepath.endswith('.doc'):
        documents = extract_text_from_docs(filepath)
    
    if not documents:
        return '读取文件内容为空'
    # 存入向量数据库
    vector_db.add_documents_and_embeddings(documents, collection_name = collection_name)

### 2. 聊天
@to_pinyin
def rag_chat(user_query, collection_name, n_results = 5):
    # 检索
    search_results = vector_db.search(user_query, collection_name = collection_name, n_results = n_results)
    # 构建prompt
    info = '\n'.join(search_results['decumemts'][0])
    query = user_query

    prompt = f"""
    你是一个问答机器人。
    你的任务是根据下述给定的已知信息回答用户问题。
    确保你的回复完全依据下述已知信息。不要编造答案。
    如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

    已知信息:
    {info}
    
    ----
    用户问：
    {query}

    请用中文回答用户问题。
    """
    # 调用大模型
    response = get_completion(prompt)
    return response

if __name__ == '__main__':
    save_to_db(filepath='../Data/人事管理流程.docx', collection_name='人事管理流程.docx')
    print('-' * 100)

    user_query = "视为不符合录用条件的情形有哪些?"
    response = rag_chat(user_query, collection_name='人事管理流程.docx', n_results=5)
    print("response:", response)