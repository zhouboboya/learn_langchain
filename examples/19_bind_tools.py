"""
═══════════════════════════════════════════════════════════════════
示例 19：直接工具调用（bind_tools）—— 不依赖 Agent，自己控制流程
═══════════════════════════════════════════════════════════════════
目的：用 .bind_tools() 让 LLM "知道" 有哪些工具可用，但你**自己决定**
是否执行、何时执行。比 Agent（示例05）更灵活。

对比：
  示例05 Agent：  用户问 → Agent 自动调用工具 → 自动回答（全自动，黑盒）
  本示例 bind_tools： 用户问 → LLM 说"我需要调用计算器" →
                      你决定是否执行 → 你把结果喂回 LLM → 得到答案
                      （半自动，白盒，每一步都可控）

使用场景：
  - 需要审核 LLM 要调用什么工具（安全审查）
  - 工具执行需要人工确认（发邮件、扣款等敏感操作）
  - 需要记录每个工具调用的中间结果

运行方式：设置了 DEEPSEEK_API_KEY 即可
═══════════════════════════════════════════════════════════════════
"""

import os, math, json
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

llm = ChatOpenAI(
    model="deepseek-chat", temperature=0,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ────────────────────────────────────────
# 第 1 步：定义工具（同示例05）
# ────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。示例：'2+3*4'、'sqrt(144)'、'2**10'"""
    try:
        allowed = {"sin": math.sin, "cos": math.cos, "sqrt": math.sqrt,
                   "pi": math.pi, "abs": abs}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算出错：{e}"

@tool
def get_weather(city: str) -> str:
    """查询城市天气（模拟）。输入城市名称。"""
    weather_data = {
        "北京": "晴天，25°C，湿度 40%",
        "上海": "多云，28°C，湿度 65%",
        "杭州": "小雨，22°C，湿度 80%",
    }
    return weather_data.get(city, f"未找到{city}的天气数据")

tools = [calculator, get_weather]
tool_map = {t.name: t for t in tools}  # 名字 → 工具函数的映射表

# ────────────────────────────────────────
# 第 2 步：bind_tools —— 告诉 LLM 工具有哪些
# ────────────────────────────────────────
# 跟 Agent 的区别：这里 LLM 只负责"建议"用什么工具
# 实际执行权在你手里
llm_with_tools = llm.bind_tools(tools)

# ────────────────────────────────────────
# 第 3 步：手动工具调用循环
# ────────────────────────────────────────
def manual_tool_loop(user_input: str):
    """
    自己控制工具调用流程：
    1. 把用户输入发给 LLM
    2. 看 LLM 的回复里有没有 tool_calls（想调工具）
    3. 如果有，执行工具，把结果喂回 LLM，回到步骤2
    4. 如果没有，说明 LLM 已经给出最终回答
    """
    messages = [HumanMessage(content=user_input)]
    iteration = 0

    while iteration < 5:  # 最多循环 5 次，防止死循环
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # 检查 LLM 是否想调工具
        if response.tool_calls:
            print(f"  🔧 LLM 想调用 {len(response.tool_calls)} 个工具：")
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                print(f"     └─ {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

                # 你来决定：是否真正执行？（这里演示自动执行）
                tool_func = tool_map[tool_name]
                result = tool_func.invoke(tool_args)

                # 把工具执行结果作为 ToolMessage 加入对话
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tc["id"],
                ))
                print(f"        → 结果：{result}")
            print()
        else:
            # 没有 tool_calls，说明 LLM 已经给出最终答案
            return response.content

        iteration += 1

    return "超过最大循环次数"


# ────────────────────────────────────────
# 第 4 步：测试
# ────────────────────────────────────────
print("【bind_tools 手动工具调用】")
print("─" * 60)

test_questions = [
    "计算 123 * 456 的结果",
    "杭州今天天气怎么样？",
    "北京比上海热吗？如果北京 25 度上海 28 度，温度差是多少？",
]

for q in test_questions:
    print(f"🤔 {q}")
    answer = manual_tool_loop(q)
    print(f"🤖 {answer}\n")

print("─" * 60)
print("💡 对比 Agent：你可以在这里加入审批逻辑")
print("   比如：if tool_name == 'send_email': 需要人工确认")
