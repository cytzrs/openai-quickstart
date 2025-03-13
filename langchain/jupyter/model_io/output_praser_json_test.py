from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class SynopsisOutput(BaseModel):
    synopsis: str = Field(description="生成的剧情简介文本")

class ReviewOutput(BaseModel):
    review: str = Field(description="生成的戏剧评论文本")


synopsis_parser = PydanticOutputParser(pydantic_object=SynopsisOutput)
review_parser = PydanticOutputParser(pydantic_object=ReviewOutput)

llm = OpenAI(temperature=.7, max_tokens=1000)

# # 这是一个 LLMChain，根据剧名和设定的时代来撰写剧情简介。
synopsis_template = """你是一位剧作家。根据戏剧的标题和设定的时代，你的任务是为该标题写一个简介。

标题：{title}
时代：{era}
剧作家：以下是对上述戏剧的简介："""




synopsis_prompt = PromptTemplate(
    input_variables=["title", "era"],
    template=synopsis_template,
    partial_variables={"format_instructions": synopsis_parser.get_format_instructions()}
)
# output_key
synopsis_chain = LLMChain(llm=llm, prompt=synopsis_prompt, 
                          output_key="synopsis", verbose=True)

# 这是一个LLMChain，用于根据剧情简介撰写一篇戏剧评论。

review_template = """你是《纽约时报》的戏剧评论家。根据该剧的剧情简介，你需要撰写一篇关于该剧的评论。

剧情简介：
{synopsis}

来自《纽约时报》戏剧评论家对上述剧目的评价："""

prompt_template = PromptTemplate(input_variables=["synopsis"], 
                                 template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template, 
                        output_key="review", 
                        verbose=True)

from langchain.chains import SequentialChain

m_overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["era", "title"],
    # Here we return multiple variables
    output_variables=["synopsis", "review"],
    verbose=True)
    
output = m_overall_chain.invoke({"title":"三体人不是无法战胜的", "era": "二十一世纪的新中国"})
    
# regular_output = output_parser.parse(output)
print(output)
