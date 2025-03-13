from langchain.chains import TransformChain


with open("langchain/jupyter/tests/the_old_man_and_the_sea.txt", encoding='utf-8') as f:
    novel_text = f.read()

    # 定义一个转换函数，输入是一个字典，输出也是一个字典。


def transform_func(inputs: dict) -> dict:
    # 从输入字典中获取"text"键对应的文本。
    text = inputs["text"]
    # 使用split方法将文本按照"\n\n"分隔为多个段落，并只取前三个，然后再使用"\n\n"将其连接起来。
    shortened_text = "\n\n".join(text.split("\n\n")[:3])
    # 返回裁剪后的文本，用"output_text"作为键。
    return {"output_text": shortened_text}


# 使用上述转换函数创建一个TransformChain对象。
# 定义输入变量为["text"]，输出变量为["output_text"]，并指定转换函数为transform_func。


transform_chain = TransformChain(
    input_variables=["text"], 
    output_variables=["output_text"],
    transform=transform_func
)
# d = transform_chain({"text": novel_text})
# print(transform_chain({"text": novel_text}))

transformed_novel = transform_chain.invoke(novel_text)

print(transformed_novel['text'])
# print(transformed_novel["output_text"])
