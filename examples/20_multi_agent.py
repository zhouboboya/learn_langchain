"""
═══════════════════════════════════════════════════════════════════
示例 20：多 Agent 协作 ——"研究员" + "写手" 分工合作
═══════════════════════════════════════════════════════════════════
目的：让多个不同角色的 Agent 串行协作，各司其职。
一个人干不了所有事，一个 Agent 也一样。

核心概念：
  - Supervisor（监督者）：接收任务 → 分配给 Specialist → 审核 → 最终输出
  - Specialist（专家 Agent）：各自擅长一个领域
  - 场景：写报告（研究员→写手→校对）、客服（分类→处理→总结）

本示例流程：
  用户需求 → 研究员收集资料 → 写手写文章 → 用户拿到最终结果

运行方式：设置了 DEEPSEEK_API_KEY 即可
═══════════════════════════════════════════════════════════════════
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

base_llm = lambda temp: ChatOpenAI(
    model="deepseek-chat", temperature=temp,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ────────────────────────────────────────
# 第 1 步：定义三个角色 Agent
# ────────────────────────────────────────
# 每个 Agent = 一个特定的 prompt + LLM，有自己的"人设"

# 研究员：负责收集和整理信息
researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个资深技术研究员。你的任务是根据主题收集整理关键信息。

要求：
1. 列出3~5个核心知识点
2. 每个知识点用1~2句话解释
3. 如果是新技术，标注其成熟度（成熟/增长/早期）
4. 只输出整理好的资料，不要加个人观点"""),
    ("human", "请研究以下主题：{topic}"),
])
researcher = researcher_prompt | base_llm(0.3) | StrOutputParser()

# 写手：根据研究资料写出文章
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个技术文章写手。根据研究资料写一篇通俗易懂的科普文章。

要求：
1. 标题吸引人，开头有 hook（钩子，勾起兴趣的一句话）
2. 用类比或生活中的例子解释复杂概念
3. 分段清晰，每段有小标题
4. 300~500字，适合公众号阅读"""),
    ("human", """主题：{topic}

研究资料：
{research_material}

请根据以上资料写一篇文章。"""),
])
writer = writer_prompt | base_llm(0.7) | StrOutputParser()

# 标题优化师：给文章起几个备选标题
title_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个公众号标题优化师。为一篇技术文章起5个吸引人的标题。

要求：
1. 不标题党，准确反映内容
2. 包含至少1个"问句式"标题和1个"数字式"标题
3. 每个标题20字以内"""),
    ("human", "文章内容：\n{article}\n\n请起5个备选标题。"),
])
title_writer = title_prompt | base_llm(0.8) | StrOutputParser()


# ────────────────────────────────────────
# 第 2 步：定义协作流程（Pipeline）
# ────────────────────────────────────────
def multi_agent_pipeline(topic: str):
    """
    三人协作流水线：
    研究员 → 写手 → 标题优化师
    """
    print(f"📋 任务主题：{topic}\n")

    # 阶段1：研究员工作
    print("─" * 50)
    print("🔍 [研究员] 收集中...\n{research_material}")
    research = researcher.invoke({"topic": topic})

    # 阶段2：写手基于研究资料写文章
    print("✍️  [写手] 写文章中...")
    article = writer.invoke({
        "topic": topic,
        "research_material": research,
    })

    # 阶段3：标题优化
    print("🎯 [标题师] 优化标题中...")
    titles = title_writer.invoke({"article": article})

    return {
        "research": research,
        "article": article,
        "titles": titles,
    }


# ────────────────────────────────────────
# 第 3 步：运行
# ────────────────────────────────────────
result = multi_agent_pipeline("向量数据库的原理与应用")

print(result["article"])
print()
print("📰 备选标题：")
print(result["titles"])
