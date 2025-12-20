from openai import OpenAI
import os
from collections import Counter


# 自我一致性(自洽性，Self-Consistency)
'''
自我一致性(自洽性，Self-Consistency)
使用场景：
    1. 生成多个推理路径或答案，选择最一致或最频繁出现的结果，减少随机性
    2. 适用于高可靠性要求任务：如科学计算、法律判断、医疗诊断。
    3. 可以与思维链结合：生成多条CoT路径，选择最优解
    4. 缺点：响应时间增加，计算成本更高。
'''

def get_completion(prompt):
    messages = [{"role": "user", "content": prompt}]

    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 获取模型返回结果
    response = client.chat.completions.create(
        model='qwen-plus-2025-07-28',
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


prompt = """
现在我70岁了，当我6岁时，我的妹妹是我的年龄的一半。现在我的妹妹多大？请逐步思考，并将答案写在括号中《》
"""


# 请求多次: n=5
def get_multiple_completions(prompt, n=1):
    responses = []
    for _ in range(n):
        responses.append(get_completion(prompt).strip())
    return responses

# 票数统计的函数
def majority_vote(responses):
    print("开始投票：")
    # 提取《》中的内容，并用Counter自动计数
    counter = Counter([res.rsplit('《')[-1].split("》")[0] for res in responses])
    print(counter)
    # 第二个值是次数，我们这里不需要，所以取第一个值
    #  most_common(1):包含前1个最常见的元素及其出现次数
    most_common_answer = counter.most_common(1)[0]
    print("投票结果：", most_common_answer)
    return most_common_answer

def get_completion_with_self_consistency(prompt, n=5):
    # 调用5次大模型，获取5个结果，拼接结果到列表
    responses = get_multiple_completions(prompt, n)

    print("Generated Responses:")
    for i, response in enumerate(responses):
        print(f"Response {i+1}: {response}")
        print("=" * 100)

    # 调用统计投票的函数
    final_answer = majority_vote(responses)
    return final_answer


# 测试
print(get_completion_with_self_consistency(prompt))
print('-' * 100)


