from langchain.chains import TransformChain

# 定义转换函数（输入字典 → 输出字典）


def transform_func(inputs: dict) -> dict:
    text = inputs["text"]
    # 示例：将文本转为大写并统计字符数
    return {
        "uppercase_text": text.upper(),
        "char_count": len(text)
    }


# 创建 TransformChain
transform_chain = TransformChain(
    input_variables=["text"], 
    output_variables=["uppercase_text", "char_count"],
    transform=transform_func  # 绑定转换函数
)

# 执行转换
result = transform_chain({"text": "hello world"})
# result = transform_chain.invoke("hello world")

print(result)
# 输出: {'uppercase_text': 'HELLO WORLD', 'char_count': 11}