"""
═══════════════════════════════════════════════════════════════════
示例 22：LangGraph 本质 —— 状态图（StateGraph）
═══════════════════════════════════════════════════════════════════
目的：理解 LangGraph 的核心抽象——把 LLM 应用建模成"节点+边"的有向图。

LangGraph 是什么？
  一个 Python 库，让你用"图"的方式编排 LLM 的调用流程。

  节点 = 函数（调用 LLM、查数据、人工审批...）
  边   = 流程方向（正常流转 / 条件分支 / 循环回退）

和 LCEL 的关系：
  LCEL (|)     → 适合"一次性"任务（问→答）
  LangGraph    → 适合"多步循环"任务（问→思考→查资料→再思考→回答）

三个核心概念：
  1. State（状态）：在节点间流转的数据（dict / Pydantic 对象）
  2. Nodes（节点）：接收 State → 返回修改后的 State  ← 就是普通 Python 函数
  3. Edges（边）：定义节点之间的流转规则

运行方式：设置了 DEEPSEEK_API_KEY 即可
═══════════════════════════════════════════════════════════════════
"""

import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOpenAI(
    model="deepseek-chat", temperature=0,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ────────────────────────────────────────
# 第 1 步：定义 State（在节点间流转的数据）
# ────────────────────────────────────────
# State 就是一张"工作台"，每个节点在上面读写数据
# TypedDict 定义字段，保证类型安全

class QAState(TypedDict):
    question: str          # 用户问题
    answer: str            # 最终答案（可能为空）
    self_check: str        # 自我检查结果
    retry_count: int       # 重试次数

# ────────────────────────────────────────
# 第 2 步：定义节点（Node）
# ────────────────────────────────────────
# 每个节点 = 一个函数，参数和返回值都是 State

def answer_question(state: QAState) -> QAState:
    """节点1：回答问题"""
    response = llm.invoke(f"请回答以下问题：{state['question']}")
    return {"answer": response.content, "retry_count": state.get("retry_count", 0)}

def self_check(state: QAState) -> QAState:
    """节点2：自我审查——回答是否准确、完整？"""
    prompt = f"""请审查以下回答是否准确完整：

问题：{state['question']}
回答：{state['answer']}

判断标准：
1. 是否回答了问题的核心？
2. 是否有事实性错误？
3. 是否够简洁（不啰嗦）？

只回复 PASS（通过）或 FAIL（不通过），不要其他内容。"""

    result = llm.invoke(prompt)
    return {"self_check": result.content.strip()}

def improve_answer(state: QAState) -> QAState:
    """节点3：改进答案"""
    prompt = f"""你之前的回答没有通过审核（{state.get('self_check', 'FAIL')}）。
请重新回答，这次更准确、更简洁：

问题：{state['question']}
之前的回答：{state['answer']}"""

    response = llm.invoke(prompt)
    new_count = state.get("retry_count", 0) + 1
    return {"answer": response.content, "retry_count": new_count}

# ────────────────────────────────────────
# 第 3 步：定义条件路由（Conditional Edge）
# ────────────────────────────────────────
# 根据 State 决定下一步走哪个节点

def route_after_check(state: QAState) -> Literal["improve", "finish"]:
    """自检通过 → 结束；不通过 → 改进；重试超过2次 → 也结束"""
    if state.get("retry_count", 0) >= 2:
        return "finish"  # 别死循环
    if "PASS" in state.get("self_check", ""):
        return "finish"
    return "improve"

# ────────────────────────────────────────
# 第 4 步：构建图
# ────────────────────────────────────────
# 节点 + 边 = 图

graph = StateGraph(QAState)

# 添加节点
graph.add_node("answer", answer_question)     # 回答
graph.add_node("check", self_check)           # 自检
graph.add_node("improve", improve_answer)     # 改进

# 添加边
graph.set_entry_point("answer")               # 入口：先回答问题
graph.add_edge("answer", "check")             # 回答完 → 自检
graph.add_conditional_edges(                  # 自检完 → 条件路由
    "check",
    route_after_check,
    {"improve": "improve", "finish": END},
)
graph.add_edge("improve", "check")            # 改进完 → 再自检（循环）

# 编译（加 checkpoint 后就支持 Memory）
app = graph.compile()

# ────────────────────────────────────────
# 第 5 步：运行
# ────────────────────────────────────────
print("【LangGraph 状态图演示】")
print("流程：answer → check → (PASS→结束 / FAIL→improve→check)")
print()

questions = [
    "Python 中怎么读一个 CSV 文件？",
    "什么是闭包？（只回答是否准确即可，不要说废话）",
]

for q in questions:
    print(f"🤔 {q}")
    result = app.invoke({"question": q, "answer": "", "self_check": "", "retry_count": 0})
    print(f"   ✅ 最终回答（{result['retry_count']}次改进后）：{result['answer'][:100]}...\n")
