"""
═══════════════════════════════════════════════════════════════════
示例 12：Few-shot 示例学习 —— 给 LLM 看几个"标准答案"
═══════════════════════════════════════════════════════════════════
目的：在 prompt 里放几个"问题→答案"的范例，LLM 模仿范例的风格和格式回答。
这是最简单但最有效的 prompt 技巧之一。

核心概念：
  - Few-shot：给 LLM 看少量（few）范例（shot），让它"照着写"
  - 场景：标准化输出格式、客服话术、分类任务
  - 对比：零样本（zero-shot）= 不给范例直接问，少样本（few-shot）= 给几个范例再问

为什么有效？
  LLM 的"模仿能力"极强——只要你给 2~3 个范例，它就能准确复制你的
  格式、语气、甚至标点习惯。

运行方式：设置了 DEEPSEEK_API_KEY 即可
═══════════════════════════════════════════════════════════════════
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,  # 格式类任务温度低一点，输出更稳定
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ────────────────────────────────────────
# 第 1 步：准备范例（Example）
# ────────────────────────────────────────
# 每个范例 = {输入描述, 期望输出}
# 对于分类任务：input=用户话术，output=分类结果

examples = [
    {
        "input": "这个东西怎么用不了啊？？",
        "output": "分类：投诉-功能故障 | 情绪：愤怒 | 优先级：高",
    },
    {
        "input": "还不错，但希望能加个夜间模式",
        "output": "分类：建议-功能需求 | 情绪：中性 | 优先级：低",
    },
    {
        "input": "客服什么时候上班？周末有人吗",
        "output": "分类：咨询-工作时间 | 情绪：中性 | 优先级：低",
    },
]

# ────────────────────────────────────────
# 第 2 步：构建 Few-shot Prompt
# ────────────────────────────────────────
# LangChain 提供了 FewShotChatMessagePromptTemplate
# 自动把范例格式化成 Human + AI 的对话格式

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

# 最终 prompt = 范例们 + 新的用户问题
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个客服工单分析助手。请对每条用户反馈进行分类、情绪判断和优先级判断。按格式回复：分类：xxx | 情绪：xxx | 优先级：xxx"),
    few_shot_prompt,               # ← 这里插入 3 组范例
    ("human", "{input}"),          # ← 真实问题放在最后
])

chain = final_prompt | llm | StrOutputParser()

# ────────────────────────────────────────
# 第 3 步：测试 —— 看 LLM 是否"学会"了格式
# ────────────────────────────────────────
print("【Few-shot 客服工单分类】")
print()

test_cases = [
    "你们这个APP每次更新完都闪退，还让不让人用了！",
    "想请问一下退款的流程是怎样的，需要多长时间到账？",
    "要是有个深色模式就好了，晚上用太刺眼",
]

# 对比：不給范例（zero-shot）
zero_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个客服工单分析助手。请对每条用户反馈进行分类、情绪判断和优先级判断。按格式回复：分类：xxx | 情绪：xxx | 优先级：xxx"),
    ("human", "{input}"),
])
zero_shot_chain = zero_shot_prompt | llm | StrOutputParser()

print("─" * 60)
print("「Zero-shot（无范例）」vs「Few-shot（3个范例）」对比")
print("─" * 60)

# 只对比第一个测试用例
zs_result = zero_shot_chain.invoke({"input": test_cases[0]})
fs_result = chain.invoke({"input": test_cases[0]})

print(f"\n用户输入：{test_cases[0]}\n")
print(f"【Zero-shot 输出】\n{zs_result}\n")
print(f"【Few-shot  输出】\n{fs_result}")

print()
print("─" * 60)
print("「Few-shot 剩余测试」")
print("─" * 60)

for case in test_cases[1:]:
    result = chain.invoke({"input": case})
    print(f"\n输入：{case}")
    print(f"输出：{result}")
