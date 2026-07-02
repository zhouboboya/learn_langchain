"""
═══════════════════════════════════════════════════════════════════
示例 13：回调与日志（Callback）—— 监控每次 LLM 调用
═══════════════════════════════════════════════════════════════════
目的：在 LLM 调用的各个阶段"偷看"发生了什么：
  - 请求发出去了没？
  - 用了多少 token？
  - 耗时多久？
  - 返回了什么？

核心概念：
  - Callback：回调函数，在 LLM 调用的关键时刻自动触发
  - 事件：on_llm_start（开始）、on_llm_end（结束）、on_llm_error（出错）
  - 场景：调试、成本监控、请求日志、性能分析

LangSmith 是 LangChain 官方的监控平台，但 Callback 可以自己做轻量版。

运行方式：设置了 DEEPSEEK_API_KEY 即可
═══════════════════════════════════════════════════════════════════
"""

import os
import time
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List

# ────────────────────────────────────────
# 第 1 步：自定义 Callback 监控器
# ────────────────────────────────────────
# 继承 BaseCallbackHandler，重写你感兴趣的事件
# 每个事件在 LLM 生命周期的特定时机被调用


class MyMonitor(BaseCallbackHandler):
    """自定义 LLM 调用监控器"""

    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        self.start_time = None

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ):
        """LLM 开始调用时触发"""
        self.call_count += 1
        self.start_time = time.time()
        print(f"🚀 [第{self.call_count}次调用] 开始请求...")
        # prompts 是发给 LLM 的完整消息列表
        msg_count = len(prompts) if prompts else 0
        print(f"   消息数：{msg_count}")

    def on_llm_end(self, response, **kwargs):
        """LLM 返回结果时触发"""
        elapsed = time.time() - self.start_time
        print(f"✅ [第{self.call_count}次调用] 完成！耗时：{elapsed:.2f}秒")

        # 从响应中提取 token 用量
        # response 是 LLMResult，包含 generations 和 token_usage
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            self.total_tokens += prompt_tokens + completion_tokens

            # DeepSeek 价格（极便宜）：
            # deepseek-chat: 输入 ¥0.001/千token, 输出 ¥0.002/千token
            cost = (prompt_tokens / 1000 * 0.001 +
                    completion_tokens / 1000 * 0.002)
            self.total_cost += cost
            print(f"   Token：输入 {prompt_tokens} + 输出 {completion_tokens}")
            print(f"   费用：¥{cost:.6f}")

    def on_llm_error(self, error, **kwargs):
        """LLM 调用出错时触发"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"❌ [第{self.call_count}次调用] 出错！耗时：{elapsed:.2f}秒")
        print(f"   错误信息：{error}")


# ────────────────────────────────────────
# 第 2 步：创建 LLM + 绑定回调
# ────────────────────────────────────────
monitor = MyMonitor()

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    callbacks=[monitor],  # ← 绑定自定义回调
)

# ────────────────────────────────────────
# 第 3 步：正常使用 —— 回调自动工作
# ────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")

chain = prompt | llm | StrOutputParser()

topics = ["深度学习", "微服务架构", "量子计算"]

for topic in topics:
    result = chain.invoke({"topic": topic})
    print(f"   回复：{result[:60]}...\n")

# ────────────────────────────────────────
# 第 4 步：查看汇总
# ────────────────────────────────────────
print("=" * 50)
print("📊 汇总报告")
print(f"   总调用次数：{monitor.call_count}")
print(f"   总 Token 消耗：{monitor.total_tokens}")
print(f"   总费用：¥{monitor.total_cost:.6f}")

# ────────────────────────────────────────
# 补充：如何接入 LangSmith？
# ────────────────────────────────────────
print()
print("💡 生产环境建议使用 LangSmith（LangChain 官方监控）：")
print("   1. pip install langsmith")
print("   2. export LANGCHAIN_API_KEY='ls__...'")
print("   3. 所有调用自动记录到 dashboard，无需写回调代码")
