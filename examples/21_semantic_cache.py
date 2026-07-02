"""
═══════════════════════════════════════════════════════════════════
示例 21：语义缓存（Semantic Cache）—— 相似问题不重复调 LLM
═══════════════════════════════════════════════════════════════════
目的：如果用户问的问题和之前某次**意思一样**（只是措辞不同），直接
返回缓存答案，不再调用 LLM。省时间 + 省钱。

核心概念：
  - 普通缓存：key 完全匹配才命中（"你好" vs "你好吗"→ 不命中）
  - 语义缓存：意思相近就命中（"你好" vs "嗨" → 命中！）
  - 原理：用 Embedding 把问题和答案都向量化，新问题来了找最相似的

使用场景：
  - 客服 FAQ：用户问的内容高度重复
  - 产品介绍：大量用户问同样的问题只是措辞不同
  - 任何高频调用的 LLM 应用

运行方式：设置了 DEEPSEEK_API_KEY 即可（Embedding 用本地模型）
═══════════════════════════════════════════════════════════════════
"""

import os
import time
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatOpenAI(
    model="deepseek-chat", temperature=0,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ────────────────────────────────────────
# 第 1 步：实现语义缓存
# ────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class SemanticCache:
    """
    语义缓存：存储 (问题向量, 答案) 的配对
    新问题来了 → 向量化 → 找最相似的历史问题 → 相似度够高就复用答案
    """

    def __init__(self, threshold: float = 0.85):
        self.cache = []          # [(向量, 问题文本, 答案), ...]
        self.threshold = threshold  # 相似度阈值（0~1），越高越严格
        self.hits = 0
        self.misses = 0

    def _cosine_similarity(self, a, b):
        """计算两个向量的余弦相似度（0~1，1=完全相同）"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def lookup(self, question: str):
        """
        查找缓存：返回 (答案, 相似度) 或 (None, 0)
        """
        q_vec = np.array(embeddings.embed_query(question))

        best_score = 0
        best_answer = None
        best_question = None

        for cached_vec, cached_q, cached_a in self.cache:
            score = self._cosine_similarity(q_vec, np.array(cached_vec))
            if score > best_score:
                best_score = score
                best_answer = cached_a
                best_question = cached_q

        if best_score >= self.threshold and best_answer:
            self.hits += 1
            return best_answer, best_score, best_question

        self.misses += 1
        return None, best_score, None

    def store(self, question: str, answer: str):
        """存入缓存"""
        q_vec = embeddings.embed_query(question)
        self.cache.append((q_vec, question, answer))

    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return {
            "缓存条目": len(self.cache),
            "命中": self.hits,
            "未命中": self.misses,
            "命中率": f"{hit_rate:.0f}%",
            "节省调用": self.hits,
        }


# ────────────────────────────────────────
# 第 2 步：创建带缓存的问答链
# ────────────────────────────────────────
cache = SemanticCache(threshold=0.85)
prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}，20字以内。")
chain = prompt | llm | StrOutputParser()

def ask_with_cache(question: str):
    """带缓存的问答：先查缓存，没命中再调 LLM"""
    # 1. 查缓存
    cached, score, matched_q = cache.lookup(question)

    if cached:
        print(f"  🟢 缓存命中！(相似度 {score:.2f}，匹配「{matched_q}」)")
        return cached

    # 2. 未命中，调 LLM
    print(f"  🔵 缓存未命中，调 LLM...")
    answer = chain.invoke({"topic": question})
    cache.store(question, answer)
    return answer


# ────────────────────────────────────────
# 第 3 步：测试 —— 相似问题命中缓存
# ────────────────────────────────────────
print("【语义缓存演示】")
print(f"  阈值：{cache.threshold}（相似度 ≥ 此值就复用缓存）")
print("─" * 60)

questions = [
    "Docker 是什么？",                    # 首次，调 LLM
    "请介绍一下 Docker",                  # 意思相近，应该命中！
    "Python 的核心特点",                  # 新主题，调 LLM
    "Docker 到底是干什么用的？",          # 跟第1个意思相近，应该命中！
    "Python 语言有哪些特性",              # 跟第3个意思相近，应该命中！
    "微服务架构",                         # 新主题
]

total_start = time.time()
cached_time_saved = 0

for q in questions:
    print(f"\n🤔 {q}")
    start = time.time()
    answer = ask_with_cache(q)
    elapsed = time.time() - start
    if "缓存命中" not in str(answer):  # 实际上我们需要更好的判断方式
        pass
    print(f"   ⏱️ {elapsed:.2f}秒")
    print(f"🤖 {answer}")
    # 如果是缓存命中，假设能省 1.5 秒
    if "🟢" in answer:
        cached_time_saved += 1.5
    else:
        cached_time_saved += 0

print("\n" + "=" * 60)
print("📊 缓存统计")
for key, value in cache.stats().items():
    print(f"  {key}：{value}")
print(f"  估算节省时间：{cached_time_saved:.1f}秒")
print(f"  估算节省费用：{cache.hits * 0.0001:.6f}元")
print(f"\n💡 实际场景中 FAQ 类应用命中率通常 60%~80%")
