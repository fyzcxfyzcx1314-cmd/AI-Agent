from openai import OpenAI
import os

# 零样本提示（Zero-Shot）：
'''
零样本提示（Zero-Shot）
应用场景：
    1. 大模型可以直接准确回答
    2. 适用于简单通用的问答，如：翻译，简单计算
'''
# 访问大模型，返回结果
def get_completion(prompt):
    messages = [{"role": "user", "content": prompt}]

    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 获取模型返回结果
    response = client.chat.completions.create(
        model='qwen3-max',
        messages=messages,
        temperature=0,
        # temperature采样温度：取值范围[0, 2)，表示模型输出的随机性，0表示随机性最小
        #   如果是要求精确结果的，则设置较小的temperature, 比如：数学运算，代码生成
        #   如果是要求多样性或创造性的，则设置较大的temperature, 比如：文案生成，故事编写
    )
    return response.choices[0].message.content

# 零样本提示词
prompt = """
判断以下句子是正面情绪、负面情绪还是中性：
"刚收到礼物，非常惊喜！"  
"""

prompt2 = """
将以下英文句子翻译成中文：  
"The rapid development of AI technology is reshaping industries."  
"""

prompt3 = """
珠穆朗玛峰的高度是多少？  
"""

print(get_completion(prompt))
print('-' * 100)
print(get_completion(prompt2))
print('-' * 100)
print(get_completion(prompt3))

