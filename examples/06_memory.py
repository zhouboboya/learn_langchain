"""
═══════════════════════════════════════════════════════════════════
示例 06：对话记忆（Memory） —— 让 LLM 记住上下文
═══════════════════════════════════════════════════════════════════
目的：默认情况下每次调用 LLM 都是"失忆"的——它不记得上一轮说了什么。
Memory 组件负责保存对话历史，让多轮对话成为可能。

核心概念：
  - InMemorySaver：把对话历史存在内存中（还有 SQLite、Redis 等持久化方案）
  - 场景：客服机器人、编程助手、角色扮演
  - 原理：每次调用时把"历史消息"一起发给 LLM

LangGraph 的 checkpointer 承担了记忆的角色，比旧版 LangChain 的 Memory 更灵活。

运行方式：同 01，设置 OPENAI_API_KEY 即可
═══════════════════════════════════════════════════════════════════
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver  # ← 内存中的"记忆"
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# ────────────────────────────────────────
# 第 1 步：理解 MessagesPlaceholder —— 历史消息的"预留位置"
# ────────────────────────────────────────
# ChatPromptTemplate 中的 MessagesPlaceholder
# 就是一个"历史消息会插入到这儿"的标记
# 每次调用时，之前的 Human + AI 消息会被填入这个位置

prompt_with_memory = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的聊天助手，请记住用户之前说过的话。"),
    MessagesPlaceholder(variable_name="history"),  # ← 历史消息插在这里
    ("human", "{input}"),
])

print("【模板结构（有记忆 vs 无记忆）】")
print("  无记忆：只有 system + human")
print("  有记忆：system + 历史消息 + human\n")


@tool
def remember_user_name(name: str) -> str:
    """记住用户的名字并返回问候语。"""
    return f"已记住你的名字：{name}，欢迎回来！"


# ────────────────────────────────────────
# 第 2 步：用 LangGraph + Checkpointer 实现记忆
# ────────────────────────────────────────
# LangGraph 的 InMemorySaver（检查点保存器）
# 会在每次对话后自动保存状态 实现了记忆功能

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)
memory = InMemorySaver()  # ← 创建记忆（存在内存中，程序结束就消失）

agent = create_react_agent(
    llm,
    [remember_user_name],
    checkpointer=memory,  # ← 把记忆注入 Agent
)

# ────────────────────────────────────────
# 第 3 步：多轮对话
# ────────────────────────────────────────
# config 中的 thread_id 是"会话线程 ID"
# 相同 thread_id 的请求会共享记忆
# 不同 thread_id 之间完全隔离——就像不同的聊天窗口

config = {"configurable": {"thread_id": "session_001"}}

print("【多轮对话演示】")
print("-" * 50)

# 第 1 轮
response = agent.invoke(
    {"messages": [HumanMessage(content="你好！我叫小明。")]},
    config=config,
)
print(f"🤖 第1轮: {response['messages'][-1].content}")

# 第 2 轮 —— Agent 应该"记得"对话历史
response = agent.invoke(
    {"messages": [HumanMessage(content="我爱好编程和跑步。")]},
    config=config,
)
print(f"🤖 第2轮: {response['messages'][-1].content}")

# 第 3 轮 —— Agent 需要用到前两轮的"记忆"来回答
response = agent.invoke(
    {"messages": [HumanMessage(content="请回忆一下，我叫什么名字？我的爱好是什么？")]},
    config=config,
)
print(f"🤖 第3轮: {response['messages'][-1].content}")

print("-" * 50)

# ────────────────────────────────────────
# 第 4 步：不同 thread_id 之间的隔离
# ────────────────────────────────────────
config_2 = {"configurable": {"thread_id": "session_002"}}

response = agent.invoke(
    {"messages": [HumanMessage(content="你还记得我叫什么名字吗？")]},
    config=config_2,
)
print(f"🤖 新会话(无记忆): {response['messages'][-1].content}")
print()
print("💡 提示：不同 thread_id 之间完全隔离，就像开了两个聊天窗口。")
