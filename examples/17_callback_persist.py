"""
═══════════════════════════════════════════════════════════════════
示例 17：自建调用记录系统（Callback + SQLite 持久化）
═══════════════════════════════════════════════════════════════════
目的：自己写一个轻量版"LangSmith"，统计每次 LLM 调用并存入 SQLite。
程序关了数据不丢，随时可以回顾历史。

对比：
  LangSmith     → 外部平台、有免费额度、功能全
  本示例        → 自己代码、数据在本地、零依赖外部服务

核心思路：
  Callback 拦截每次调用 → 写入 SQLite → 随时查询统计

运行方式：设置了 DEEPSEEK_API_KEY 即可
═══════════════════════════════════════════════════════════════════
"""

import os
import time
import sqlite3
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List

# ────────────────────────────────────────
# 第 1 步：数据库初始化
# ────────────────────────────────────────
DB_PATH = "/Users/zhouningbo/Desktop/langchain/llm_logs.db"

def init_db():
    """创建日志表"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_calls (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            call_time   TEXT    NOT NULL,         -- 调用时间
            model       TEXT    NOT NULL,         -- 模型名
            duration_ms INTEGER NOT NULL,         -- 耗时（毫秒）
            prompt_tokens INTEGER DEFAULT 0,      -- 输入 Token
            completion_tokens INTEGER DEFAULT 0,  -- 输出 Token
            cost_yuan   REAL    DEFAULT 0,        -- 费用（元）
            input_text  TEXT,                     -- 输入内容
            output_text TEXT,                     -- 输出内容
            error       TEXT,                     -- 错误信息
            status      TEXT    DEFAULT 'success' -- success / error
        )
    """)
    conn.commit()
    conn.close()


class PersistentMonitor(BaseCallbackHandler):
    """自定义回调：记录每次调用并存到 SQLite"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._start_time = None
        self._current_input = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        """调用开始：记录时间和输入"""
        self._start_time = time.time()
        # prompts 是发给 LLM 的完整消息列表
        self._current_input = json.dumps(prompts, ensure_ascii=False)[:2000]

    def on_llm_end(self, response, **kwargs):
        """调用结束：计算统计并写入数据库"""
        duration_ms = int((time.time() - self._start_time) * 1000)
        output_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        cost = 0

        if response.generations and response.generations[0]:
            first_gen = response.generations[0][0]
            output_text = first_gen.text[:2000] if first_gen.text else ""

        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            # DeepSeek 定价
            cost = (prompt_tokens / 1000 * 0.001 +
                    completion_tokens / 1000 * 0.002)

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO llm_calls
               (call_time, model, duration_ms, prompt_tokens,
                completion_tokens, cost_yuan, input_text, output_text, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'success')""",
            (datetime.now().isoformat(), self.model_name, duration_ms,
             prompt_tokens, completion_tokens, round(cost, 8),
             self._current_input, output_text)
        )
        conn.commit()
        conn.close()

        # 实时打印简要信息
        print(f"  ✅ {duration_ms}ms | 入{prompt_tokens}+出{completion_tokens} token | ¥{cost:.6f}")

    def on_llm_error(self, error, **kwargs):
        """调用失败：记录错误"""
        duration_ms = int((time.time() - self._start_time) * 1000) if self._start_time else 0
        error_msg = str(error)[:500]

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO llm_calls
               (call_time, model, duration_ms, input_text, error, status)
               VALUES (?, ?, ?, ?, ?, 'error')""",
            (datetime.now().isoformat(), self.model_name, duration_ms,
             self._current_input, error_msg)
        )
        conn.commit()
        conn.close()
        print(f"  ❌ {error_msg}")


# ────────────────────────────────────────
# 第 2 步：初始化 + 绑定回调
# ────────────────────────────────────────
init_db()
monitor = PersistentMonitor(model_name="deepseek-chat")

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    callbacks=[monitor],
)

# ────────────────────────────────────────
# 第 3 步：正常使用 —— 数据自动记录
# ────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")
chain = prompt | llm | StrOutputParser()

topics = ["Python", "Docker", "Redis"]

for topic in topics:
    result = chain.invoke({"topic": topic})
    print(f"  {topic}: {result[:50]}...")

# ────────────────────────────────────────
# 第 4 步：查询历史记录（持久化带来的价值）
# ────────────────────────────────────────
def query_history():
    """看看库里存了什么"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("""
        SELECT call_time, model, duration_ms, prompt_tokens,
               completion_tokens, cost_yuan, status
        FROM llm_calls
        ORDER BY id DESC
        LIMIT 10
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

print("\n" + "=" * 60)
print("📊 SQLite 持久化历史记录")
print("=" * 60)

rows = query_history()
for r in rows:
    status_icon = "✅" if r["status"] == "success" else "❌"
    print(f"  {status_icon} {r['call_time'][:19]} | {r['model']}")
    print(f"     耗时 {r['duration_ms']}ms | Token {r['prompt_tokens']}+{r['completion_tokens']} | ¥{r['cost_yuan']:.6f}")

# 汇总统计
conn = sqlite3.connect(DB_PATH)
total = conn.execute("""
    SELECT
        COUNT(*) as total_calls,
        SUM(prompt_tokens) as total_prompt,
        SUM(completion_tokens) as total_completion,
        SUM(cost_yuan) as total_cost,
        AVG(duration_ms) as avg_ms
    FROM llm_calls
    WHERE status = 'success'
""").fetchone()
conn.close()

print(f"\n📈 汇总")
print(f"  总调用：{total[0]} 次")
print(f"  总 Token：{total[1]}+{total[2]}")
print(f"  总费用：¥{total[3]:.6f}")
print(f"  平均耗时：{total[4]:.0f}ms")
print(f"\n💾 数据文件：{DB_PATH}")
print(f"   数据永不过期，关掉程序重新运行也能查到历史记录")
