"""
═══════════════════════════════════════════════════════════════════
示例 05：Agent（智能体）—— 让 LLM 学会使用"工具"
═══════════════════════════════════════════════════════════════════
目的：LLM 自身不会算数学、不会查天气、不会搜网页。
Agent（智能体）通过调用"工具"（Tool）来弥补这些短板。

核心概念：
  - Tool（工具）：一个可以被 LLM 调用的函数（计算器、搜索、API 等）
  - Agent（智能体）：能自主决定"该用哪个工具、什么时候用"的 LLM
  - 流程：用户提问 → LLM 判断需要工具 → 调用工具 → 拿到结果 → 继续思考/回答

运行方式：export OPENAI_API_KEY="sk-xxx"; cd examples && python3 05_agent.py
═══════════════════════════════════════════════════════════════════
"""

import math
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool  # ← 装饰器：把普通函数变成 Tool

# ────────────────────────────────────────
# 第 1 步：定义工具（Tool）
# ────────────────────────────────────────
# 用 @tool 装饰器，给函数加上描述，LLM 会根据描述判断何时调用
# 函数必须写 docstring —— LLM 就是靠这个来理解工具的用途！


@tool
def calculator(expression: str) -> str:
    """计算数学表达式的结果。支持加减乘除、幂运算、三角函数等。
    示例输入：'2 + 3 * 4'、'sqrt(144)'、'sin(30 * 3.14159 / 180)'
    """
    try:
        # 使用受限的 eval，只允许数学函数，保证安全
        allowed_names = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "sqrt": math.sqrt, "pi": math.pi, "abs": math.abs,
            "pow": math.pow, "log": math.log,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算出错：{e}"


@tool
def word_counter(text: str) -> str:
    """统计一段文本的字数、词数、行数。输入一段文本字符串。"""
    lines = text.count("\n") + 1
    words = len(text.split())
    chars = len(text)
    return f"行数：{lines}，词数：{words}，字符数：{chars}"


@tool
def reverse_string(text: str) -> str:
    """将输入的文本反转（倒序排列）。"""
    return text[::-1]


# 收集所有工具
tools = [calculator, word_counter, reverse_string]

# ────────────────────────────────────────
# 第 2 步：创建 Agent
# ────────────────────────────────────────
# create_react_agent 是 LangGraph 提供的标准 Agent 实现
# "ReAct" = Reasoning（推理）+ Acting（行动），是目前最主流的 Agent 模式
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

agent = create_react_agent(llm, tools)

# ────────────────────────────────────────
# 补充知识：工具没有数量上限，但有实际约束
# ────────────────────────────────────────
# 1. 上下文窗口：每个工具约 200~500 tokens，deepseek-chat 窗口 64K
# 2. 选工具准确率：工具越多越容易选错，5~20 个是最佳区间
# 3. 扩展方案：工具超过 50 个时，用"路由 Agent"分类后再选
# 4. 每个响应里查看 token 用量：
#    response["messages"][-1].response_metadata["token_usage"]

# ────────────────────────────────────────
# 第 3 步：让 Agent 自主回答问题
# ────────────────────────────────────────
# Agent 收到问题后，会自动判断是否需要用工具
# 你不需要手动调用工具——Agent 全部自己搞定

print("【Agent 测试——数学计算】")
# 这里需要同时用计算器和字数统计，Agent 会自己协调
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "计算 234 * 567 的结果，然后统计这个结果有几位数。"}
    ]
})
# Agent 返回的是一个消息列表，最后一条是最终回复
print(response["messages"][-1].content)
print()

print("【Agent 测试——文本工具】")
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "请把 'Hello LangChain' 反转过来"}
    ]
})
print(response["messages"][-1].content)
