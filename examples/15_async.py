"""
═══════════════════════════════════════════════════════════════════
示例 15：异步与并发（Async / Await）
═══════════════════════════════════════════════════════════════════
目的：同时发起多个 LLM 请求，不用等一个结束再发下一个。
串行 vs 并行的区别：

  串行：A → 等3秒 → B → 等3秒 → C → 等3秒 = 总耗时 9 秒
  并行：A ↘
        B → 同时发 → 等 3 秒 → A B C 一起回来 = 总耗时 3 秒
        C ↗

核心概念：
  - .ainvoke()：invoke 的异步版本，返回 awaitable 对象
  - asyncio.gather()：同时启动多个异步任务，等全部完成后一起拿结果
  - 场景：批量翻译、批量分类、同时查多个数据源

注意：Python 异步需要理解 async/await 语法，但 LangChain
      把它封装得很简单，你只需要知道 .ainvoke() 就够了

运行方式：设置了 DEEPSEEK_API_KEY 即可
═══════════════════════════════════════════════════════════════════
"""

import os
import time
import asyncio
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

prompt = ChatPromptTemplate.from_template("用一句中文介绍{topic}的核心概念，20字以内。")
chain = prompt | llm

# ────────────────────────────────────────
# 第 1 步：串行方式（invoke）
# ────────────────────────────────────────
# 一个接一个，每个等 2~3 秒
topics = ["Docker", "Redis", "Kafka", "GraphQL", "Elasticsearch"]

print("【方式一：串行 invoke（逐个调用）】")
start = time.time()
for topic in topics:
    result = chain.invoke({"topic": topic})
    print(f"  {topic}: {result.content}")
serial_time = time.time() - start
print(f"  总耗时：{serial_time:.1f}秒\n")

# ────────────────────────────────────────
# 第 2 步：并行方式（ainvoke + asyncio.gather）
# ────────────────────────────────────────
# 同时发 5 个请求，等最慢的那个完成

async def query_async(topic: str) -> str:
    """异步查询单个主题"""
    result = await chain.ainvoke({"topic": topic})
    return f"  {topic}: {result.content}"

async def main():
    """同时发送所有请求"""
    # asyncio.gather 同时启动 5 个协程
    results = await asyncio.gather(
        *[query_async(t) for t in topics]
    )
    return results

print("【方式二：并行 ainvoke（同时调用）】")
start = time.time()
results = asyncio.run(main())
for r in results:
    print(r)
async_time = time.time() - start
print(f"  总耗时：{async_time:.1f}秒")

# ────────────────────────────────────────
# 对比
# ────────────────────────────────────────
print(f"\n{'='*50}")
print(f"📊 性能对比")
print(f"  串行耗时：{serial_time:.1f}秒")
print(f"  并行耗时：{async_time:.1f}秒")
if async_time > 0:
    print(f"  提速：{serial_time / async_time:.1f}倍！")

# ────────────────────────────────────────
# 补充：什么时候用串行 vs 并行？
# ────────────────────────────────────────
print(f"\n💡 使用建议")
print(f"  串行 .invoke()： 只有一个请求、或后一个请求依赖前一个的结果")
print(f"  并行 .ainvoke()： 批量处理、互不依赖的请求、需要节约总时间")
print(f"  注意：          太并发的请求可能触发 API 限流（Rate Limit）")
