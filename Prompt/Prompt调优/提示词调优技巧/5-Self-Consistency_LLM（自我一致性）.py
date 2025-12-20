from openai import OpenAI
import os

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


# 生成多个样本数量。
def get_multiple_completions(prompt, n):
    responses = []
    for _ in range(n):
        responses.append(get_completion(prompt).strip())
    return responses

# 使用大模型来统计和选择最一致的答案
def get_most_common_answer_via_model(responses):
    responses_str = "\n".join(responses)
    prompt_for_model = f"""请根据以下生成的多个答案：{responses_str}，\n选择出现次数最多的答案作为最终答案。最终答案是："""
    # 将前面5个结果拼接成一个提示词，交给大模型进行次数统计，返回最终答案
    final_answer = get_completion(prompt_for_model).strip()
    return final_answer

# 调度函数，并打印生成的所有响应，以便观察和调试。
def get_completion_with_self_consistency(prompt, n=5):
    responses = get_multiple_completions(prompt, n)

    print("Generated Responses:")
    for i, response in enumerate(responses):
        print('=' * 100)
        print(f"Response {i+1}: {response}")

    final_answer = get_most_common_answer_via_model(responses)
    print("===============经过模型的判断，最终答案如下:================")

    return final_answer


prompt = """
现在我70岁了，当我6岁时，我的妹妹是我的年龄的一半。现在我的妹妹多大？请逐步思考
"""


# 测试
print(get_completion_with_self_consistency(prompt))
