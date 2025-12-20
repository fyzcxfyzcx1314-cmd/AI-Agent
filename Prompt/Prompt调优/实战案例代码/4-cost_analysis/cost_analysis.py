import gradio as gr
from openai import OpenAI
import os
import datetime
import threading

# 应用实战3：企业运营成本分析核算

# 提示：本案例运行消耗的Token较多,请选择合适的模型
# 浏览器中访问：http://127.0.0.1:7860，需要填入的数据可从三一重工.json和中联重科.json中获得

class CostAnalysisPipeline:
    def __init__(self, name):
        self.name = name

    # 调用大模型API
    @classmethod
    def get_completion(cls, messages):
        client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

        response = client.chat.completions.create(
            model='qwen-plus-2025-07-14',
            messages=messages,
            temperature=0,  # 有数字计算，给较低的temperature
        )
        return response

    # 使用大模型进行问答
    @classmethod
    def LLM_QA(cls, llm_q):
        # 构建问答模板
        qa_template = f"""你是一个企业业绩分析专家，请回答以下问题：
        {llm_q}
        请给出详细和精确的分析。"""
        try:
            # 打印开始询问大模型的时间
            messages = [{"role": "user", "content": qa_template}]
            # 调用OpenAI API获取回答
            response = cls.get_completion(messages)
            # 提取回答内容
            answer = response.choices[0].message.content.strip() if response.choices else "无法提供答案"
            # print(f"完整回答: {answer}")
            return answer
        except Exception as e:
            # 捕获并打印API调用异常
            print(f"{'*'*100}\nAPI调用失败: {e}\n{'*'*100}")
            return "无法提供答案"

    # 1.提取财务指标
    def extract_financial_data(self, report_text):
        print(f'{self.name}: 从财务报表中提取关键指标..')

        # 构建提取关键指标的提示
        prompt = (
            f"请从以下财务报表中提取关键指标：营业收入、净利润、每股收益、可以适当精简,但分析后需要保留发表日期时间信息"
            f"""总资产收益率、毛利率、净利率。\n\n,如果有多年数据，需要分析趋势
                输入数据：{report_text}"""
        )
        # 调用LLM_QA方法获取回答
        return self.LLM_QA(prompt)

    # 2.分析财务指标
    def analyze_financial_indicators(self, report_text):
        print(f'{self.name}: 分析财务指标..')

        # 构建分析财务指标的提示
        prompt = (
            f"根据以下财务数据，计算营业收入同比增长率和净利润同比增长率，可以适当精简，但分析后需要保留发表日期时间信息"
            f"并生成年度对比分析。如果有多年数据，需要分析趋势\n\n输入数据：{report_text}"
        )
        # 调用LLM_QA方法获取回答
        return self.LLM_QA(prompt)

    # 3.预测未来趋势
    def predict_future_trends(self, report_text):
        print(f'{self.name}: 预测未来趋势..')

        # 构建预测未来趋势的提示
        prompt = (
            f"根据以下当前财务数据，预测未来三年的营业收入和净利润趋势。但分析后需要保留发表日期时间信息\n\n"
            f"输入数据：{report_text}"
        )
        # 调用LLM_QA方法获取回答
        return self.LLM_QA(prompt)

    # 4.优化成本
    def optimize_costs(self, report_text):
        print(f'{self.name}: 优化成本..')

        # 构建优化成本的提示
        prompt = (
            f"根据以下业绩信息，提出业绩优化建议，可以适当精简,但分析后需要保留发表日期时间信息"
            f"如何提高企业的业绩效益。\n\n输入数据：{report_text}"
        )
        # 调用LLM_QA方法获取回答
        return self.LLM_QA(prompt)

    # 生成最终报告
    def generate_final_report(self, extracted_data, analysis_result, prediction_result, optimization_result):
        print(f'{self.name}: 生成最终报告..')

        # 构建生成最终报告的提示
        prompt = (
            f"根据以下数据，生成一份企业业绩分析报告，"
            f"报告包含财务分析、趋势预测和业绩优化建议。可以适当精简,但分析后需要保留发表日期时间信息\n\n"
            f"提取的财务数据：{extracted_data}\n\n"
            f"财务分析：{analysis_result}\n\n"
            f"趋势预测：{prediction_result}\n\n"
            f"业绩优化建议：{optimization_result}"
        )
        # 调用LLM_QA方法获取回答
        return self.LLM_QA(prompt)

    # 分析数据
    def analyze_data(self, report_text, res):
        print(f'{"*"*80}\n{self.name}: 开始分析数据..: {datetime.datetime.now()}')
        results = []

        # 1.提取财务指标
        extracted_data = self.extract_financial_data(report_text)
        results.append(f"提取的财务指标:\n{extracted_data}\n")

        # 2.分析财务指标
        analysis_result = self.analyze_financial_indicators(report_text)
        results.append(f"财务指标分析:\n{analysis_result}\n")

        # 3.预测未来趋势
        prediction_result = self.predict_future_trends(report_text)
        results.append(f"未来趋势预测:\n{prediction_result}\n")

        # 4.优化业绩
        optimization_result = self.optimize_costs(report_text)
        results.append(f"业绩优化建议:\n{optimization_result}\n")

        # 生成最终报告
        final_report = self.generate_final_report(
            extracted_data, analysis_result, prediction_result, optimization_result
        )
        results.append(f"企业业绩分析报告:\n{final_report}\n")

        res.extend(["\n".join(results), final_report])


# 比较两个报告
def compare_reports(report1, report2):
    # 构建比较报告的提示
    prompt = (
        f"以下是两家公司或者同一个公司不同时期财务分析报告，请先判断这是一个公司的不同时间报告还是不同公司的报告，然后对它们进行比较,需要关注财报的发表日期。可以适当精简\n\n"
        f"数据1的报告：{report1}\n\n"
        f"数据2的报告：{report2}"
    )
    # 调用LLM_QA方法获取回答
    return CostAnalysisPipeline.LLM_QA(prompt)

# 比较两个报告
def compare(input1, input2):

    # 创建CostAnalysisPipeline实例
    pipeline1 = CostAnalysisPipeline('report1')
    pipeline2 = CostAnalysisPipeline('report2')

    # 使用多线程：让2个报告可以同时分析（节省时间）
    res1 = []  # 用来接收报告1的结果
    res2 = []  # 用来接收报告2的结果
    t1 = threading.Thread(target=pipeline1.analyze_data, args=(input1, res1))
    t2 = threading.Thread(target=pipeline2.analyze_data, args=(input2, res2))
    t1.start()  # 启动线程
    t2.start()
    t1.join()  # 等待让前面2个线程都结束
    t2.join()

    # 把多线程分析得到的报告用result1,result2,final_report1
    result1, final_report1 = res1
    result2, final_report2 = res2
    print(f"report1完成：{datetime.datetime.now()}\n-{'#'*80}")
    print(f"report2完成：{datetime.datetime.now()}\n-{'#'*80}")


    # 将报告1和报告2的结果写入md文件
    with open('result/analysis_result1.md', 'w', encoding='utf-8') as f:
        f.write(result1)
    with open('result/analysis_result2.md', 'w', encoding='utf-8') as f:
        f.write(result2)
    with open('result/final_report1.md', 'w', encoding='utf-8') as f:
        f.write(final_report1)
    with open('result/final_report2.md', 'w', encoding='utf-8') as f:
        f.write(final_report2)


    # 开始比较2个报告，并得到比较结果
    compare_result = compare_reports(result1, result2)
    print(f"compare报告比较完成：{datetime.datetime.now()}\n-{'#'*80}")

    # 将2个报告的比较结果，写入本地md文件
    with open('result/compare_result.md', 'w', encoding='utf-8') as f:
        f.write(compare_result)

    return compare_result


# 创建Gradio界面
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## 企业业绩分析")
        with gr.Row():
            with gr.Column():
                # 输入框1，用于输入财务报表文本
                input1 = gr.Textbox(label="输入数据 1 (财务报表文本)", placeholder="请输入财务报表内容...")

            with gr.Column():
                # 输入框2，用于输入财务报表文本
                input2 = gr.Textbox(label="输入数据 2 (财务报表文本)", placeholder="请输入财务报表内容...")

        # 比较按钮
        compare_button = gr.Button("比较报告")
        # 输出框，用于显示比较结果
        compare_output = gr.Markdown(label="比较结果")
        # 绑定比较按钮到compare函数
        compare_button.click(compare, inputs=[input1, input2], outputs=compare_output)

    demo.launch()


if __name__ == "__main__":
    main()

