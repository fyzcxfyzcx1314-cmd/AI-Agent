from openai import OpenAI
import os

# 思维树(Tree-of-thought, ToT)
'''
思维树(Tree-of-thought, ToT)
使用场景：
    1. 把问题思路设计为树结构，探索多种推理路径，最终综合选择最优解。
    2. 适用于高度复杂的决策问题，数学证明、编程调试、多路径决策
    3. 缺点：实现复杂，计算开销极大，对模型能力要求高。
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
小明100米跑成绩：10.5秒，1500米跑成绩：3分20秒，铅球成绩：12米。他适合参加哪些搏击运动训练?
请按下面步骤思考：
1.请根据以上成绩，分析候选人在速度、耐力、力量三方面素质的分档。分档包括：强（3），中（2），弱（1）三档
2.根据小明的速度、耐力、力量的分档结果，分别给小明从3个维度来推荐运动
   - 需要根据速度强度来推荐运动有哪些，给出10个例子，
   - 需要根据耐力强度来推荐运动有哪些，给出10个例子，
   - 需要根据力量强度来推荐运动有哪些，给出10个例子。
3.分别分析上面给的10个运动对速度、耐力、力量方面素质的要求: 强（3），中（2），弱（1）
根据上面的分析：生成一篇小明适合那种运动训练的分析报告, 请将思维树的步骤用树形图画出来
"""

# 运行可能需要几十秒
print(get_completion(prompt))

