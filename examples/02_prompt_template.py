"""
═══════════════════════════════════════════════════════════════════
示例 02：提示词模板（Prompt Template）
═══════════════════════════════════════════════════════════════════
目的：学习如何用模板构造结构化的提示词。
提示词不再是一句固定的话，而是可以动态填充的"模版"。

核心概念：
  - SystemMessage：设定 AI 的角色和规则（系统指令）
  - HumanMessage：用户说的话
  - ChatPromptTemplate：把上面的消息组合起来，支持 {变量} 动态填充

运行方式：同 01，先设置 OPENAI_API_KEY
═══════════════════════════════════════════════════════════════════
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ────────────────────────────────────────
# 第 1 步：定义提示词模板
# ────────────────────────────────────────
# SystemMessage     →  告诉 AI "你是谁、要做什么"
# HumanMessage      →  用户的实际输入，用 {变量名} 做占位符
# ChatPromptTemplate.from_messages() 把这些消息组合成一个模板
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一位{role}专家，请用{style}风格回答问题。"),  # ← 角色 + 风格占位
    ("human", "{question}"),                                      # ← 用户问题占位
])

# 打印一下模板结构，帮忙理解
print("【模板结构】")
print(f"模板包含的消息角色：system、human")
print(f"模板变量：role, style, question\n")

# ────────────────────────────────────────
# 第 2 步：用 .invoke() 给变量赋值并生成最终消息
# ────────────────────────────────────────
# .invoke({"变量1": "值1", ...}) 把占位符全部替换
# 返回值是一个消息列表，传给 LLM 即可
filled_messages = prompt_template.invoke({
    "role": "Python 编程",
    "style": "通俗易懂",
    "question": "Python 中的装饰器是如何工作的？",
})

print("【填充后的消息】")
for msg in filled_messages.messages:
    print(f"  [{msg.__class__.__name__}] {msg.content}")
print()

# ────────────────────────────────────────
# 第 3 步：将模板与 LLM 组合调用
# ────────────────────────────────────────
# 这里开始接触 LangChain 的灵魂：用 | 串起"模板 → LLM"
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

chain = prompt_template | llm  # ← LCEL（LangChain 表达式语言）：用管道符串联组件

resposta = chain.invoke({
    "role": "历史",
    "style": "讲故事",
    "question": "请讲讲唐朝长安城的一天",
})

print("【LLM 回复】")
print(resposta.content)
