from openai import OpenAI
import os


# 一个样本（One-Shot）
# 少样本提示（Few-Shot）
'''
少样本提示（Few-Shot）
使用场景： 
    1. 提供少量示例，引导模型理解任务模式，
    2. 适用于零样本无法准确回答，或回答格式不符合要求。
'''

def get_completion(prompt):
    messages = [{"role": "user", "content": prompt}]

    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 获取模型返回结果
    response = client.chat.completions.create(
        model='qwen3-max',
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


prompt = """
1. 生成文本：ChatGPT可以生成与给定主题相关的文章、新闻、博客、推文等等。
2. 语言翻译：ChatGPT可以将一种语言的文本翻译成另一种语言。
3. 问答系统：ChatGPT可以回答您提出的问题，无论是事实性的问题、主观性的问题还是开放性的问题。
4. 对话系统：ChatGPT可以进行对话，您可以与ChatGPT聊天，让它回答您的问题或就某个话题进行讨论。
5. 摘要生成：ChatGPT可以从较长的文本中生成摘要，帮助您快速了解文章的主要内容。
6. 文本分类：ChatGPT可以将一些给定的文本分类到不同的类别中，例如新闻、体育、科技等等。
7. 文本纠错：ChatGPT可以自动纠正文本中的拼写错误和语法错误，提高文本的准确性。
请把上面7段话各自的开头几个词，翻译成英文，并按序号输出。
例如：第1段话的开头是"生成文本"，那么就输出"generate text"
"""

prompt2 = """
将句子改写成正式商务邮件风格，参考示例：  
示例1: "快点把报告发我" → "请您尽快将报告发送给我，谢谢。"  
示例2: "这方案不行" → "该方案目前存在一些不足之处。"  
现在改写："明天别迟到！"  
"""

prompt3 = """
写一首关于春天的诗句

示例1（关于冬天）：
孤枝覆新雪，
寒鸦独立悄，
冬寂入心境。

示例2（关于夏天）：
蝉鸣扰午梦，
蜻蜓点水涟，
烈日灼草青。
"""

print(get_completion(prompt))
print('-' * 100)
print(get_completion(prompt2))
print('-' * 100)
print(get_completion(prompt3))
