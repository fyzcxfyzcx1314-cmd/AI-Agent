# 安装gradio：pip install gradio
from openai import OpenAI
import gradio as gr  # python中的写界面的包

import os

# 应用实战2：基于提示工程的旅游攻略系统实现

def get_completion(messages):
    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    response = client.chat.completions.create(
        model='qwen-plus-2025-07-14',
        messages=messages,
    )
    return response.choices[0].message.content


# 按钮绑定了函数，将用户提示和对应内容传进去
def start_plan(qa1, qa2, qa3, qa4, qa5):
    instruction = """
    # 角色：你是一位专业的旅游助手
    ## 技能：非常擅长制定1-15天的旅游规划，有介绍酒店和餐厅的特色、价格和评分的经验。
    # 任务：分析并根据用户输入的要求，结合你总结出来的，设计一个详细到每天的旅游计划，并说明每天的旅游景点，时间，餐饮，住宿情况。 
    # 要求：总费用可以在预算的10%左右浮动
    """
    examples = """
        # 示例：
        Q：您的旅游目的地是哪个城市？
        A：深圳
        Q：您计划在该城市玩几天？
        A：3天
        Q：您一共有几个人一同出行？
        A：3个人，2大1小
        Q：有没有一定要去的景点？是什么？
        A：大梅沙海滩、欢乐谷
        Q：您的预算大概是多少钱?
        A：5000元

        回复: 圳3天2晚家庭游（2大1小）行程规划
            总预算：5000元（实际预估：5100元，浮动+2%）
            行程特点：海滨休闲+主题乐园+城市观光，兼顾亲子需求
            
            Day 1：大梅沙海滨休闲日
            上午
            景点：大梅沙海滨公园（免费）
            时间：9:00-12:00
            活动：沙滩玩水、堆沙堡、拍照，儿童可租用沙滩玩具（约20元/套）。
            提示：建议早到避开人流，带好防晒用品。
            午餐
            餐厅：五谷芳乳鸽王（大梅沙店）
            人均：80元
            推荐菜：红烧乳鸽、海鲜蒸锅，适合家庭口味。
            总费用：240元
            下午
            景点：大梅沙奥特莱斯（免费观光）
            时间：14:00-16:00
            活动：购物或散步，品牌折扣店可淘货。
            晚餐
            餐厅：盐田海鲜街（推荐师公会海鲜酒家）
            人均：100元
            推荐菜：椒盐皮皮虾、蒜蓉粉丝蒸扇贝。
            总费用：300元
            住宿
            酒店：京基海湾大酒店（大梅沙区域）
            房型：豪华海景房（含早）
            价格：600元/晚
            评分：4.6/5（亲子友好，步行5分钟到沙滩）
            总费用：600元
            
            Day 2：欢乐谷狂欢日
            全天
            景点：深圳欢乐谷（门票）
            时间：10:00-18:00
            费用：成人230元/人，儿童120元（1.2米以下免费），总580元。            
            推荐项目：魔幻城堡（儿童区）、飓风湾漂流。            
            午餐            
            餐厅：欢乐谷内“老船长餐厅”            
            人均：60元            
            推荐：儿童套餐、意面，方便快捷。            
            总费用：180元
            晚餐
            餐厅：福田区“润园四季椰子鸡”
            人均：90元
            推荐菜：竹荪椰子鸡、煲仔饭，清淡适合孩子。
            总费用：270元
            住宿
            酒店：深圳威尼斯睿途酒店（欢乐谷附近）
            房型：亲子主题房（含早）
            价格：700元/晚
            评分：4.7/5（泳池+儿童乐园）
            总费用：700元
            
            Day 3：城市文化探索
            上午
            景点：深圳野生动物园
            时间：9:00-11:30
            费用：成人240元/人，儿童140元，总620元。
            亮点：猛兽谷投喂、海洋动物表演。
            午餐
            餐厅：南山区“农耕记湖南土菜”
            人均：70元
            推荐菜：土鸡汤、农家小炒肉。
            总费用：210元
            下午
            景点：华侨城创意园（免费）
            时间：14:00-16:00
            活动：文创小店拍照、下午茶休憩。
            返程
            交通建议：打车至机场/高铁站（约50元）。
            
            费用明细
            住宿：600+700=1300元
            餐饮：240+300+180+270+210=1200元
            门票：580（欢乐谷）+620（动物园）=1200元
            交通：市内打车+地铁约200元
            其他：沙滩玩具+零食约200元
            总计：1300+1200+1200+200+200=5100元
    """
    user_input = f"""
    Q：您的旅游目的地是哪个城市？
    A：{qa1}
    Q：您计划在该城市玩几天？
    A：{qa2}
    Q：您一共有几个人一同出行？
    A：{qa3}
    Q：有没有一定要去的景点？是什么？
    A：{qa4}
    Q：您的预算大概是多少钱?
    A：{qa5}

    """
    # 提示词
    prompt = f"""
        {instruction}
        {examples}
        用户输入：
        {user_input}
    """
    # print(prompt)
    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages)
    print(response)
    return response


# 创建一个 Gradio 界面
def show_gradio():

    with gr.Blocks() as demo:
        # 设置标题
        gr.Markdown("# 基于提示工程的旅游攻略系统")
        gr.Markdown("### 为了给您制作一个更好的旅游攻略，请根据如下问题准确进行回答！")
        question_list = [
            "1、您的旅游目的地是哪个城市？",
            "2、您计划在该城市玩几天？",
            "3、您一共有几个人一同出行？",
            "4、有没有一定要去的景点？是什么？",
            "5、您的预算大概是多少钱？"
        ]
        # 创建输入框，用户可以输入5个问题的答案
        qa1_input = gr.Textbox(label=question_list[0])
        qa2_input = gr.Textbox(label=question_list[1])
        qa3_input = gr.Textbox(label=question_list[2])
        qa4_input = gr.Textbox(label=question_list[3])
        qa5_input = gr.Textbox(label=question_list[4])

        # 按钮
        submit = gr.Button("开始计划")

        # 创建输出框，显示结果
        result = gr.Textbox(label="旅游计划结果", placeholder="点击按钮后显示结果", lines=20)

        # 点击，执行函数start_plan; input是函数的输入参数，outputs是函数的返回结果
        submit.click(start_plan,
                     inputs=[qa1_input, qa2_input, qa3_input, qa4_input, qa5_input],
                     outputs=result)

    # 启动应用
    demo.launch()

if __name__ == '__main__':
    show_gradio()

