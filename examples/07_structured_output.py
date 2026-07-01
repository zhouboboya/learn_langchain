"""
═══════════════════════════════════════════════════════════════════
示例 07：结构化输出（Structured Output）
═══════════════════════════════════════════════════════════════════
目的：让 LLM 返回结构化的 JSON 或 Pydantic 对象，而不是自由文本。
这在"从文章提取信息"、"解析用户意图"等场景非常实用。

核心概念：
  - Pydantic 模型：定义数据的"形状"（字段名 + 类型 + 描述）
  - with_structured_output()：让 LLM 严格按照 Pydantic 模型输出
  - 原理：LLM 内部调用 function calling，LangChain 把返回的 JSON 自动转成 Pydantic 对象

对比：
  普通输出："人名：张三，年龄：25"  ← 难解析
  结构化输出：Person(name='张三', age=25)  ← 直接 .name .age

运行方式：设置了 DEEPSEEK_API_KEY 即可
═══════════════════════════════════════════════════════════════════
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

# ────────────────────────────────────────
# 第 1 步：用 Pydantic 定义"期望的数据结构"
# ────────────────────────────────────────
# Field(description=...) 很重要！LLM 靠这个描述理解每个字段该填什么
# Optional 和默认值可以告诉 LLM：这个字段可以不填


class Person(BaseModel):
    """从文本中提取的人物信息"""
    name: str = Field(description="人物姓名")
    age: int = Field(description="年龄，如无法判断填 0")
    city: str = Field(description="所在城市，如无法判断填'未知'")
    skills: List[str] = Field(description="技能列表")


class MovieReview(BaseModel):
    """影评结构化摘要"""
    title: str = Field(description="电影名称")
    rating: float = Field(description="评分，1~10 之间")
    sentiment: str = Field(description="情感倾向：正面 / 负面 / 中性")
    summary: str = Field(description="一句话总结")
    pros: Optional[List[str]] = Field(default=None, description="优点列表")
    cons: Optional[List[str]] = Field(default=None, description="缺点列表")

# ────────────────────────────────────────
# 第 2 步：创建 LLM + 绑定结构化输出
# ────────────────────────────────────────
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,  # 结构化输出建议用 0，更稳定
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# with_structured_output() 要求 LLM 严格按 Pydantic 模型返回
# 本质上就是自动调用 function calling，比手写 prompt 可靠得多

# ────────────────────────────────────────
# 第 3 步：提取人物信息
# ────────────────────────────────────────
person_extractor = llm.with_structured_output(Person, method="function_calling")

text = "我叫李明，今年 28 岁，住在杭州。我会 Python 和 Go 两种编程语言。"

person = person_extractor.invoke(text)

print("【人物提取】")
print(f"  姓名：{person.name}")
print(f"  年龄：{person.age}")
print(f"  城市：{person.city}")
print(f"  技能：{', '.join(person.skills)}")
print()

# ────────────────────────────────────────
# 第 4 步：分析影评
# ────────────────────────────────────────
review_extractor = llm.with_structured_output(MovieReview, method="function_calling")

review_text = """
《流浪地球3》的特效又升级了，太空电梯那段看得手心冒汗。剧情比第二部更紧凑，
人物刻画也更立体。唯一不足是片长接近3小时，中间有点拖沓。
总的来说是一部值得二刷的好片，我给8.5分。
"""

review = review_extractor.invoke(review_text)

print("【影评分析】")
print(f"  电影：{review.title}")
print(f"  评分：{review.rating}/10")
print(f"  情感：{review.sentiment}")
print(f"  总结：{review.summary}")
if review.pros:
    print(f"  优点：{' / '.join(review.pros)}")
if review.cons:
    print(f"  缺点：{' / '.join(review.cons)}")
