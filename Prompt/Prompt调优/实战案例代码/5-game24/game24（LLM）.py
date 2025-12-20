from openai import OpenAI
import os
from game24_prompt import *


# 实战项目5： 基于提示工程 TOT Paper 24点计算实战


# 访问大模型:通义千问
def get_completion(prompt):
    messages = [{"role": "user", "content": prompt}]

    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    response = client.chat.completions.create(
        model="qwen-plus-2025-07-14", # 可以尝试其他模型
        messages=messages,
    )
    return response.choices[0].message.content

# 思路：
'''
# 4 9 10 13
# 第1轮：计算
#  4 + 9 = 13 (left:13 10 13)
#  4 - 9 = -5 (left:-5 10 13)
#  10- 4 = 6  (left:6 9 13)
#  ....

# 第1轮验证：对上面计算的结果做一次验证(sure(20)/likely(1)/impossible(0.001))
#  13 10 13  => Sure  20
#  -5 10 13  => impossible  0.001
#  6 9 13 => likely  1
#  ...


# 第2轮计算：第1轮不为impossible的数据（论文中是取最可能的前5个结果）
# 6 9 13
#  6 + 9 = 15 (left:15 13)
#  6 - 9 = -3 (left:-3 13)
#  6 * 9 = 54  (left:54 13)
#  13 -9 = 4  (left:4 6)
#  ...

# 第2轮验证： 对上面计算的结果做一次验证(sure(20)/likely(1)/impossible(0.001))
# 15 13  => impossible(0.001)
# 4 6  => sure(20)
#  ...

# 第3轮：计算
#  15 13
#  4 6 : 4*6 = 24
#  ...
'''


# 第一轮：计算   input = "4 9 10 13"
def first_think(input):
    # 将输入的字符串以'\n'分割转换为列表
    propose_list = get_completion(propose_prompt.format(input=input)).split('\n')
    propose_list2 = list(filter(lambda i: "left" in i or "剩余" in i, propose_list))
    propose_list3 = list(filter(lambda i: '≈' not in i, propose_list2))

    for p in propose_list3:
        print(p)
    print('-' * 100)

    return propose_list3

# 第一轮：验证 (非常考验模型能力)
def first_evaluate(propose_list):
    results = []
    print('第一轮结果数共：', len(propose_list))

    for propose in propose_list:
        # propose = "4 + 9 = 13 (left: 10 13 13)"
        print("第一轮：验证; propose:", propose)

        # 提取left之后的3个数字 : "10 13 13"
        propose_numbers = propose.strip().split(': ')[-1].split(')')[0]
        print('第一轮：验证; propose_numbers:', propose_numbers)

        # 不考虑小数
        is_float = False
        for num_str in propose_numbers:
            if '.' in num_str:
                is_float = True
        if is_float: continue


        # 大模型来验证left剩下数有没有可能达到24
        output = get_completion(value_prompt1.format(input=propose_numbers))
        print("第一轮，验证结果; output:", output)

        # 获取得分
        value = output.split()[0]
        if "left" in output:
            value = output.split('left:')[-1]
        print('value:', value)

        # 将所有结果拼接：排除0.001 impossible的情况
        # if value in ['20', 'sure(20)']:
        if value in ['20', '1', 'sure(20)', 'likely(1)']:
            results.append({'value': value,
                            'propose_numbers': propose_numbers,
                            'propose': propose})

        print('*' * 100)

    # 第一轮：获取前5个，减少第二轮访问LLM次数（准确率可能降低）
    # results = sorted(results, key=lambda i: i['value'], reverse=True)[:5]
    propose_numbers_list = [res['propose_numbers'] for res in results]

    print("第一轮验证后的成功数量:", len(results))
    return propose_numbers_list, results


# 第二轮：计算
def second_think(input):
    print("第二轮 input:", input)
    # 将输入的字符串以'\n'分割转换为列表
    propose_list = get_completion(propose_prompt2.format(input=input)).split('\n')
    propose_list2 = list(filter(lambda i: "left" in i or "剩余" in i, propose_list))
    propose_list3 = list(filter(lambda i: '≈' not in i, propose_list2))

    for p in propose_list3:
        print(p)
    print('-' * 100)

    return propose_list3

# 第二轮：验证
def second_evaluate(propose_list):
    results = []
    print('第二轮结果数共：', len(propose_list))
    print('-' * 100)

    for propose in propose_list:
        # propose = "4 + 9 = 13 (left: 10 13)"
        print("第二轮：验证; propose:", propose)

        # 提取left之后的2个数字
        propose_numbers = propose.strip().split(': ')[-1].split(')')[0]
        print('第二轮：验证; propose_numbers:', propose_numbers)

        # 不考虑小数
        is_float = False
        for num_str in propose_numbers:
            if '.' in num_str:
                is_float = True
        if is_float: continue

        # 大模型来验证left剩下数有没有可能达到24
        output = get_completion(value_prompt2.format(input=propose_numbers))
        print("第二轮，验证结果; output:", output)

        value = output.split()[0]
        if "left" in output:
            value = output.split('left:')[-1]
        print('value:', value)

        # 将所有结果拼接
        # if value in ['20', 'sure(20)']:
        if value in ['20', '1', 'sure(20)', 'likely(1)']:
            results.append({'value': value,
                            'propose_numbers': propose_numbers,
                            'propose': propose})
        print('*' * 100)

    # 第二轮：获取前5个，减少第二轮访问LLM次数（准确率可能降低）
    # results = sorted(results, key=lambda i: i['value'], reverse=True)[:5]
    propose_numbers_list = [res['propose_numbers'] for res in results]

    print("第二轮验证后的成功数量:", len(results))
    return propose_numbers_list, results

# 第三轮：计算
def third_think(input):
    print("第三轮 input:", input)
    print('-' * 100)

    # 将输入的字符串以'\n'分割转换为列表
    propose_list = get_completion(propose_prompt3.format(input=input)).split('\n')
    propose_list2 = list(filter(lambda i: "left" in i or "剩余" in i, propose_list))

    has_24 = False
    for p in propose_list2:
        propose_number = p.strip().split(': ')[-1].split(')')[0]
        print(f'p:{p} ==> {propose_number}')
        if propose_number == '24':
            print(f"--- 恭喜您！{p} 得到结果{propose_number}了！")
            has_24 = True
    if not has_24:
        print("没有找到24！")

    print('-' * 100)

    return propose_list2


# 程序主入口
if __name__ == '__main__':
    import datetime
    print('start_time:', datetime.datetime.now())

    input = "4 9 10 13"
    print("第一轮：计算中......")
    propose_list = first_think(input)
    print('第一轮：计算结束:', datetime.datetime.now())

    print("第一轮：验证结果中......")
    propose_numbers_list1, results1 = first_evaluate(propose_list)
    print('第一轮：验证结果结束:', datetime.datetime.now())
    print('*' * 100, end='\n\n')

    print("第二轮：计算中......")
    propose_list2 = second_think(propose_numbers_list1)
    print('第二轮：计算结束:', datetime.datetime.now())

    print("第二轮：验证结果中......")
    propose_numbers_list2, results2 = second_evaluate(propose_list2)
    print('第二轮：验证结果结束:', datetime.datetime.now())
    print('*' * 100, end='\n\n')

    print("第三轮：计算中......")
    propose_list3 = third_think(propose_numbers_list2)

    print('end_time:', datetime.datetime.now())
