import itertools

# game24: Python实现
def game24(*nums, history=None):
    # history:保持计算路径
    if history is None:
        history = []

    # 递归终止条件：如果只有1个数则不继续递归
    if len(nums) <= 1:
        if nums[0] == 24:  # 找到答案，打印history
            print("找到解：")
            for step in history:
                print(f"{step[0]} {step[1]} {step[2]} = {step[3]}")
            print()
        return

    # 使用排列（permutations）尽量包含所有情况：2个数排列的所有情况
    # 排列：4, 9, 10, 13
    permutations = list(itertools.permutations(nums, 2))
    # print(permutations)
    ops = ['+', '-', '*', '/']  # 操作符

    # 遍历排列中的 每个组合
    for a, b in permutations:
        # 删除要参与计算的a,b，得到剩下的数
        left_list = list(nums).copy()
        left_list.remove(a)
        left_list.remove(b)

        # 遍历4个操作符，让a,b有4次运算
        for op in ops:
            # 避免加法 (+) 和乘法 (*) 的重复计算：只考虑a<=b的计算
            if (op == '+' or op == '*') and a > b:
                continue  # 跳过 a + b 和 b + a 的重复情况

            # 计算：防止除法中分母有0的情况
            try:
                c = eval(f'{a}{op}{b}')  # eval( '4 + 9' )  4 + 9 = 13
            except ZeroDivisionError:
                continue

            # 记录当前运算步骤
            new_history = history.copy()
            new_history.append((a, op, b, c))

            # 构建新的数字列表
            new_nums = left_list.copy()
            new_nums.append(c)

            # 递归调用
            game24(*new_nums, history=new_history)


if __name__ == '__main__':
    game24(4, 9, 10, 13)


'''
找到解：
4 - 10 = -6
9 - 13 = -4
-6 * -4 = 24

找到解：
9 + 10 = 19
19 - 13 = 6
4 * 6 = 24

找到解：
9 - 13 = -4
4 - 10 = -6
-6 * -4 = 24

找到解：
9 - 13 = -4
-4 + 10 = 6
4 * 6 = 24

找到解：
10 - 4 = 6
13 - 9 = 4
4 * 6 = 24

找到解：
10 - 13 = -3
-3 + 9 = 6
4 * 6 = 24

找到解：
13 - 9 = 4
10 - 4 = 6
4 * 6 = 24

找到解：
13 - 9 = 4
10 - 4 = 6
4 * 6 = 24

找到解：
13 - 10 = 3
9 - 3 = 6
4 * 6 = 24
'''

