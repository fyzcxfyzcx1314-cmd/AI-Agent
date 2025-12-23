### 少样本的提示词

from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_openai import ChatOpenAI

### 1. 创建示例
example = [
    {"sinput": "2+2", "soutput": "4", "sdescription": "加法运算"},
    {"sinput": "5-2", "soutput": "3", "sdescription": "减法运算"},
]
### 2. 配置一个提示模板，用来示例格式化
example_prompt_txt = "算式： {sinput} 值： {soutput} 类型： {sdescription} "

### 3. 构建提示词模板示例
prompt_sample = PromptTemplate.from_template(example_prompt_txt)
### 4. 创建少样本示例的对象
prompt = FewShotPromptTemplate(
    example = example,
    example_prompt = prompt_sample,
    prefix="你是一个数学专家, 能够准确说出算式的类型，",
    suffix="现在给你算式: {input} ， 值: {output} ，告诉我类型：",
    input_variables=["input", "output"]
)