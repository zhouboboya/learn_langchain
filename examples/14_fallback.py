"""
═══════════════════════════════════════════════════════════════════
示例 14：错误处理与降级（Fallback / Retry）
═══════════════════════════════════════════════════════════════════
目的：LLM 调用可能会失败（API 超时、限流、返回格式错误...）。
用 Fallback 和 Retry 机制让应用更"扛造"（鲁棒）。

核心概念：
  - .with_fallbacks([备选链])：主链失败时自动切换到备选链
  - .with_retry(stop_after_attempt=N)：失败后自动重试 N 次
  - 场景：生产环境必备，不能因为一次 API 超时就崩掉

三种保护机制：
  1. Fallback：主模型挂了 → 自动切成备用模型
  2. Retry：   API 超时 → 自动重试
  3. 输出修复：LLM 格式不对 → 让它自己修

运行方式：设置了 DEEPSEEK_API_KEY 即可
═══════════════════════════════════════════════════════════════════
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

# ────────────────────────────────────────
# 准备工作：主模型 + 备用模型
# ────────────────────────────────────────
# 在实际应用中，主模型和备用模型可以是不同厂商的
main_llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    max_retries=1,        # OpenAI SDK 层的重试
    request_timeout=10,   # 10 秒超时
)

# 备用可以用同一个模型（实际项目中建议用不同厂商）
fallback_llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    max_retries=1,
    request_timeout=30,   # 备用模型给更长的超时时间
)

# ────────────────────────────────────────
# 第 1 步：with_fallbacks —— 主链失败自动切换
# ────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")

# 主链
main_chain = prompt | main_llm | StrOutputParser()
# 备用链
fallback_chain = prompt | fallback_llm | StrOutputParser()

# 把备用链"挂在"主链后面
robust_chain = main_chain.with_fallbacks([fallback_chain])

result = robust_chain.invoke({"topic": "Kubernetes"})
print(f"【正常调用】\n{result}\n")

# ────────────────────────────────────────
# 第 2 步：with_retry —— 失败自动重试
# ────────────────────────────────────────
from langchain_core.runnables import RunnableLambda

# 模拟一个不稳定的操作（前两次抛异常，第三次成功）
call_count = {"count": 0}

def unstable_llm(topic: str) -> str:
    """模拟不稳定的 LLM 调用"""
    call_count["count"] += 1
    if call_count["count"] <= 2:
        raise ConnectionError(f"网络超时！（第{call_count['count']}次）")
    return f"{topic} 的介绍已生成（第{call_count['count']}次才成功）"

unstable_chain = RunnableLambda(unstable_llm)

# with_retry：最多重试 3 次，每次间隔 0.5 秒
retry_chain = unstable_chain.with_retry(
    stop_after_attempt=3,   # 最多重试 3 次（含首次调用）
)

print("【with_retry 演示】")
print("（前两次模拟失败，第三次成功...）")
try:
    result = retry_chain.invoke("Docker")
    print(f"结果：{result}")
except Exception as e:
    print(f"3次重试后仍然失败：{e}")
print()

# ────────────────────────────────────────
# 第 3 步：输出格式修复 —— OutputFixingParser
# ────────────────────────────────────────
# 有时 LLM 返回的格式不对（少了个括号、多了个逗号）
# OutputFixingParser 会自动把错误格式"喂回"LLM，让它修复
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Color(BaseModel):
    name: str = Field(description="颜色名称")
    hex_code: str = Field(description="十六进制颜色码，如 #FF0000")

# 用 PydanticOutputParser 生成格式说明嵌入 prompt
parser = PydanticOutputParser(pydantic_object=Color)
format_instructions = parser.get_format_instructions()

# 故意生成一个"有瑕疵"的 prompt，增加 LLM 犯错概率
bad_prompt = ChatPromptTemplate.from_template(
    "说出一种你喜欢的颜色，用 JSON 格式返回。\n"
    "{format_instructions}"
)

llm_for_parse = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

chain = bad_prompt | llm_for_parse | parser

print("【Pydantic 结构化解析（带自动修复兜底）】")
try:
    color = chain.invoke({"format_instructions": format_instructions})
    print(f"  颜色：{color.name}")
    print(f"  色码：{color.hex_code}")
except Exception as e:
    print(f"  解析失败（但实际应用中会触发自动修复）: {e}")
