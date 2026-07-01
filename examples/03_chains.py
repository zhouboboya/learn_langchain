"""
═══════════════════════════════════════════════════════════════════
示例 03：链式调用（Chains + LCEL）
═══════════════════════════════════════════════════════════════════
目的：用管道符 | 将多个组件串联起来，形成一条处理流水线。
这是 LangChain 最核心的心智模型。

核心概念：
  - LCEL（LangChain Expression Language）：用 | 运算符串联组件
  - StrOutputParser：把 LLM 返回的 AIMessage 转成纯字符串
  - RunnableParallel：让多个子链并行执行（节省时间）

示意：
  提示词模板 →  LLM  →  输出解析器
  (Prompt)     (Model)    (Parser)
═══════════════════════════════════════════════════════════════════
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ────────────────────────────────────────
# 第 1 步：最简单的链 —— 三件套
# ────────────────────────────────────────
# 模板 + LLM + 解析器，用 | 串成一条链
# StrOutputParser 负责把 AIMessage 对象转成纯字符串

prompt = ChatPromptTemplate.from_template(
    "用{language}写一个{task}的代码示例，只输出代码不要解释。"
)

chain = prompt | llm | StrOutputParser()  # ← 三个组件首尾相连

result = chain.invoke({"language": "Python", "task": "快速排序"})

print("【基础链输出】")
print(result)
print()

# ────────────────────────────────────────
# 第 2 步：并行执行 —— RunnableParallel
# ────────────────────────────────────────
# 有时候你想同时向 LLM 提多个问题，不用等第一个结束再发第二个
# RunnableParallel 让多条子链同时执行

# 先定义一条"公用"链：接收一个主题，生成一句话介绍
base_chain = ChatPromptTemplate.from_template(
    "用一句话介绍{topic}的核心概念。"
) | llm | StrOutputParser()

# 用 RunnableParallel 并行跑三个主题
# 关键：itemgetter 从外层 dict 中取出各分支自己的数据，再传给 base_chain
from langchain_core.runnables import RunnableParallel
from operator import itemgetter

parallel_chain = RunnableParallel(
    langchain=itemgetter("langchain") | base_chain,
    docker=itemgetter("docker") | base_chain,
    kubernetes=itemgetter("kubernetes") | base_chain,
)

# 一次性传入所有主题（外层 key → 内层 {topic: ...}）
results = parallel_chain.invoke({
    "langchain": {"topic": "LangChain"},
    "docker": {"topic": "Docker"},
    "kubernetes": {"topic": "Kubernetes"},
})

print("【并行输出】")
for key, value in results.items():
    print(f"  {key}: {value}")
print()

# ────────────────────────────────────────
# 第 3 步：条件分支 —— RunnableBranch
# ────────────────────────────────────────
# 根据输入内容，选择不同的处理分支
from langchain_core.runnables import RunnableBranch

# 三条针对不同问题类型的提示词
code_prompt = ChatPromptTemplate.from_template("生成{question}的代码，只输出代码。")
explain_prompt = ChatPromptTemplate.from_template("详细解释{question}。")
other_prompt = ChatPromptTemplate.from_template("请回答以下问题：{question}")

# 定义分支规则：[ (条件, 分支), ... ]
branch = RunnableBranch(
    # 如果问题中包含"代码"→走代码分支
    (lambda x: "代码" in x["question"],
     code_prompt | llm | StrOutputParser()),
    # 如果问题中包含"解释"或"原理"→走解释分支
    (lambda x: any(w in x["question"] for w in ["解释", "原理"]),
     explain_prompt | llm | StrOutputParser()),
    # 兜底分支（都不匹配时走这个）
    other_prompt | llm | StrOutputParser(),
)

print("【条件分支测试】")
print("问'写代码':", branch.invoke({"question": "写一个二分查找的代码"}))
