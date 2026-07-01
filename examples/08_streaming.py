"""
═══════════════════════════════════════════════════════════════════
示例 08：流式输出（Streaming）
═══════════════════════════════════════════════════════════════════
目的：LLM 生成一个字就输出一个字，不用等全部生成完。
体验过 ChatGPT 逐字蹦出来的效果吗？这就是 Streaming。

核心概念：
  - .stream()：返回一个迭代器，每生成一个 token 就 yield 一次
  - .astream()：异步版本，适合 FastAPI 等异步框架
  - 场景：聊天界面、长时间生成、提升用户体验

对比：
  非流式：invoke() → 等 10 秒 → 一次性返回全部文字
  流式：  stream() → 立刻开始 → 逐字蹦出来，像打字一样

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

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ────────────────────────────────────────
# 第 1 步：对比 invoke 和 stream
# ────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("写一首关于{topic}的五言绝句")

chain = prompt | llm | StrOutputParser()

# 方式一：invoke —— 一口气返回
print("【方式一：invoke（一次性返回）】")
start = time.time()
result = chain.invoke({"topic": "春天"})
print(f"结果：{result}")
print(f"耗时：{time.time() - start:.2f}秒\n")

# 方式二：stream —— 一个字一个字蹦出来
print("【方式二：stream（流式输出）】")
start = time.time()
print("结果：", end="", flush=True)
for chunk in chain.stream({"topic": "秋天"}):
    print(chunk, end="", flush=True)  # ← chunk 就是新生成的字，立刻打印
print()
print(f"耗时：{time.time() - start:.2f}秒\n")

# ────────────────────────────────────────
# 第 2 步：流式输出时做实时处理
# ────────────────────────────────────────
# 实际应用中，可以在每收到一个 token 时做处理：
#   - 推送给 WebSocket
#   - 写入文件
#   - 统计字数
#   - 检测敏感词

print("【实时处理演示】")
word_count = 0

for chunk in chain.stream({"topic": "编程"}):
    # 每收到一个 chunk 都可以做点事
    word_count += len(chunk)
    print(chunk, end="", flush=True)

print(f"\n总共收到 {word_count} 个字符")

# ────────────────────────────────────────
# 第 3 步：不使用 StrOutputParser，看原始流式输出
# ────────────────────────────────────────
# 去掉解析器，能看到 LLM 返回的原始 AIMessageChunk
# 这对调试和理解内部机制很有用
print("\n【原始流式数据（去掉解析器）】")
raw_chain = prompt | llm

i = 0
for chunk in raw_chain.stream({"topic": "夏天"}):
    # chunk 是 AIMessageChunk，包含 token 和元数据
    i += 1
    if i <= 3:  # 只看前 3 个 chunk 的结构
        print(f"  Chunk {i}: {chunk}")
    if i == 3:
        print(f"  ...（后续省略）")
        break
