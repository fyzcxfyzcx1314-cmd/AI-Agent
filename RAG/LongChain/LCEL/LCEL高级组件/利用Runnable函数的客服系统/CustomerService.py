import os
import re
import json
import time

import langchain
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import chain

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse

import uvicorn

# 业务场景：电商客户反馈处理系统
# 需求描述
# 某电商平台需要自动处理客户反馈，实现以下功能：
#
# 情感分析：判断用户反馈的情感倾向
#
# 问题分类：识别反馈中的问题类型
#
# 紧急程度评估：根据内容判断处理优先级
#
# 生成回复草稿：根据分析结果生成初步回复

### 1. 调用大模型
qwen = ChatTongyi(
    model_name = "qwen3-max",
    temperature = 0.2,
    max_tokens = 2000,
    streaming = False,
    enable_search = True #启用联网搜索增强
)

### 2. 带有重试次数以及时间间隔的模型调用
def call_qwen_with_retry(prompt, max_retries = 3, retry_delay = 2):
    for attempt in range(max_retries):
        try:
            response = qwen.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"模型调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(retry_delay)
    return "模型服务暂时不可用，请稍后再试。"

### 3. 提取订单号
def extract_order_id(text : str) -> dict:
    prompt = f"""
    你是一个电商订单处理工具，从一下客户的反馈中提取订单号：
    {text}
    要求：
        1. 订单号通常是以"ORD"开头的10位数字组合。
        2. 如果为发现订单号，返回"NOT FOUND"。
        3. 严格按照JSON格式返回数据结果：{{"order_id : "提取结果"}} 
    """
    ### 先使用正则提取，后面会使用AI Agent调用外部工具实现
    try:
        match = re.search(r'ORD\d{10}', text)
        return {"order_id" : match.group(0) if match else "NOT_FOUND"}
    except:
        result = call_qwen_with_retry(prompt)
        return json.loads(result.strip())
    
### 4. 情感分析
def analyze_sentiment(text : str) -> dict:
    promot = f"""
    请分析以下客户反馈的情感倾向：
    「{text}」

    要求：
    1. 判断情感类型：POSITIVE(积极)/NEUTRAL(中性)/NEGATIVE(消极)
    2. 评估置信度(0.0-1.0)
    3. 提取3个关键短语

    返回JSON格式：
    {{
        "sentiment": "情感类型",
        "confidence": 置信度,
        "key_phrases": ["短语1", "短语2", "短语3"]
    }}
    """
    try:
        result = call_qwen_with_retry(promot)
        parser = JsonOutputParser()
        result = parser.parse(result)
        return result
    except Exception as e:
        print("情感分析失败")
        return {
            "sentiment": "NEUTRAL",
            "confidence": 0.7,
            "key_phrases": []
        }
    
### 5. 问题分类
def classify_issue(text : str) -> dict:
    prompt = f"""
    作为电商客服专家，请对以下客户反馈进行分类：
    「{text}」

    分类选项：
    - 物流问题：配送延迟、物流损坏等
    - 产品质量：商品瑕疵、功能故障等
    - 客户服务：客服态度、响应速度等
    - 支付问题：扣款异常、退款延迟等
    - 退货退款：退货流程、退款金额等
    - 其他：无法归类的反馈

    要求：
    1. 选择最相关的1-2个分类
    2. 按相关性排序

    返回JSON格式：{{"categories": ["分类1", "分类2"]}}
    """
    try:
        parser = JsonOutputParser()
        result = parser.parse(call_qwen_with_retry(prompt))
        return result
    except Exception as e:
        print("问题分类失败")
        return {"categories" : ["其他"]}
    
### 6. 紧急程度分析
def asses_urgency(text : str) -> dict:
    prompt = f"""
    作为客服主管，请评估以下客户反馈的紧急程度：
    「{text}」

    评估标准：
    - HIGH(高)：包含"紧急"、"立刻"、"马上"或威胁投诉
    - MEDIUM(中)：表达强烈不满但无立即行动要求
    - LOW(低)：一般反馈或建议

    返回JSON格式：
    {{
        "urgency": "紧急级别",
        "sla_hours": 响应时限(小时),
        "reason": "评估理由"
    }}
    """
    try:
        parser = JsonOutputParser()
        result = parser.parse(call_qwen_with_retry(prompt))
        result["sla_hours"] = int(result["sla_hours"])
        return result
    except Exception as e:
        print("紧急评估失败")
        return {
            "urgency": "NONE",
            "sla_hours": 24,
            "reason": "评估失败"
        }
    
### 7. 总结上面所用信息，给出客服回复
@chain
def generate_response(data : str) -> dict:
    print(data)
    """使用千问模型生成定制化回复"""
    prompt_template = """
    你是一名资深电商客服专家，请根据以下分析结果生成客户回复：

    ### 客户反馈原文：
    {feedback}

    ### 分析结果：
    - 订单ID：{order_id}
    - 情感倾向：{sentiment} (置信度：{confidence:.2f})
    - 问题类型：{categories}
    - 紧急程度：{urgency} (需在{sla_hours}小时内响应)
    {key_phrases_section}

    ### 回复要求：
    1. 根据情感倾向调整语气：
       - 积极反馈：表达感谢，适当赞美
       - 消极反馈：诚恳道歉，明确解决方案
    2. 包含订单ID和问题分类
    3. 明确说明处理时限和后续步骤
    4. 长度100-150字，使用自然口语
    5. 结尾询问是否还有其他问题

    请直接输出回复内容，不需要额外说明。
    """
    # 构建关键短语部分
    key_phrases = data.get("key_phrases", [])
    if key_phrases:
        key_phrases_section = "- 关键要点：" + "，".join(key_phrases[:3])
    else:
        key_phrases_section = ""

    # 填充模板
    prompt = prompt_template.format(
        feedback=data["original_feedback"],
        order_id=data["order_id"],
        sentiment=data["sentiment"],
        confidence=data.get("confidence", 0.8),
        categories="、".join(data["categories"]),
        urgency=data["urgency"],
        sla_hours=data["sla_hours"],
        key_phrases_section=key_phrases_section
    )
    try:
        response = call_qwen_with_retry(prompt)

        # 添加紧急标识
        if data["urgency"] == "HIGH":
            response = f"[紧急] {response}"

        return {
            "final_response": response,
            "assigned_team": data["categories"][0] if data["categories"] else "General",
            "result":data
        }

    except Exception as e:
        print(f"回复生成失败: {e}")
        return {
            "final_response": "感谢您的反馈，我们的团队将尽快处理您的问题。",
            "assigned_team": "General"
        }
    
### 8. 使用Runnable对提取订单号函数处理，保留基本反馈信息
extract_chain = RunnableParallel(
    order_id = RunnableLambda(extract_order_id),
    original_feedback = lambda x : x
)
"""
extrct_chain返回的数据格式是:
{
  "order_id": {
    "order_id": "ORD2024071501"  # 或 "NOT_FOUND"
  },
  "original_feedback": "原始反馈文本"
}
"""

### 9. 情感分析，问题分类，紧急程度并行执行
analysis_chain = RunnableParallel(
    sentiment=RunnableLambda(analyze_sentiment),
    categories=RunnableLambda(classify_issue),
    urgency=RunnableLambda(asses_urgency)
)
"""
analyze_chain返回的数据格式是:
{
  "sentiment": {
    "sentiment": "NEGATIVE",  # 情感类型
    "confidence": 0.92,       # 置信度
    "key_phrases": ["物流太慢", "承诺三天", "实际七天"]  # 关键短语
  },
  "categories": {
    "categories": ["物流问题"]  # 问题分类
  },
  "urgency": {
    "urgency": "HIGH",        # 紧急程度
    "sla_hours": 4,           # 响应时限(小时)
    "reason": "包含紧急处理要求"  # 评估理由
  }
}
"""

### 10. 完整流程
process_chain = (
    extract_chain
    | RunnablePassthrough.assign(
        analysis = lambda x : analysis_chain.invoke(x["original_feedback"])
    )
    | {
        "original_feedback": lambda x: x["original_feedback"],
        "order_id": lambda x: x["order_id"]["order_id"],
        "sentiment": lambda x: x["analysis"]["sentiment"].get("sentiment", "NEUTRAL"),
        "confidence": lambda x: x["analysis"]["sentiment"].get("confidence", 0.8),
        "key_phrases": lambda x: x["analysis"]["sentiment"].get("key_phrases", []),
        "categories": lambda x: x["analysis"]["categories"]["categories"],
        "urgency": lambda x: x["analysis"]["urgency"]["urgency"],
        "sla_hours": lambda x: x["analysis"]["urgency"]["sla_hours"],
        "urgency_reason": lambda x: x["analysis"]["urgency"].get("reason", "")
    }
    | generate_response
)
"""
经过extract_chain
    | RunnablePassthrough.assign(
        analysis = lambda x : analyze_chain.invoke(x["original_feedback"])
    )操作后的数据格式是:
{
    "order_id": {
        "order_id": "ORD2024071501"  # 或 "NOT_FOUND"
    },
    "original_feedback": "原始反馈文本"
    "analysis" : 
    {
    "sentiment": {
        "sentiment": "NEGATIVE",  # 情感类型
        "confidence": 0.92,       # 置信度
        "key_phrases": ["物流太慢", "承诺三天", "实际七天"]  # 关键短语
    },
    "categories": {
        "categories": ["物流问题"]  # 问题分类
    },
    "urgency": {
        "urgency": "HIGH",        # 紧急程度
        "sla_hours": 4,           # 响应时限(小时)
        "reason": "包含紧急处理要求"  # 评估理由
    }
    }
}
"""

### 11. 模型部署为API服务
app = FastAPI(title = "电商客服平台")

class FeedbackRequest(BaseModel):
    content : str
    user_id : str = "anonymous"

@app.get("/")
async def read_index():
    return FileResponse("index.html")
@app.post("/process-feedback")
async def process_feedback(request : FeedbackRequest):
    try:
        start = time.time()
        result = process_chain.invoke(request.content)
        elapsed = time.time() - start

        return {
            "success": True,
            "processing_time": f"{elapsed:.2f}s",
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理失败: {str(e)}"
        )

if __name__ == "__main__":
    print("当前目录:", os.getcwd())
    print("脚本路径:", __file__)
    print("HTML文件是否存在:", os.path.exists("index.html"))
    uvicorn.run(app, host = "localhost", port = 8000)