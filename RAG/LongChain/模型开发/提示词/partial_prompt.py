from langchain.prompts import PromptTemplate
import datetime

def get_datatime():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

prompt_txt = "工具现在的时间{data},简述今天有什么事件{story}"

prompt = PromptTemplate(
    template = prompt_txt,
    input_variables=["date", "story_type"]
)

half_prompt = prompt.partial(data = get_datatime)