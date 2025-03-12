from langchain.output_parsers import DatetimeOutputParser
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate


output_parser = DatetimeOutputParser()
template = """Answer the users question:

{question}

{format_instructions}"""

prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

llm = OpenAI()
chain = prompt | llm

output = chain.invoke("around when was bitcoin founded?")
print(output)
regular_output = output_parser.parse(output)
print(regular_output)