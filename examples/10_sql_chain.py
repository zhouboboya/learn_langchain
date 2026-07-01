"""
═══════════════════════════════════════════════════════════════════
示例 10：SQL 数据库自然语言查询
═══════════════════════════════════════════════════════════════════
目的：用自然语言查询数据库，LLM 自动把"问题"翻译成"SQL"。
不用写 SELECT * FROM ...，直接说"上个月销量最高的商品"就行。

核心概念：
  - SQLDatabase：封装数据库连接
  - create_sql_agent：Agent 自动判断写什么 SQL、怎么查、怎么回答
  - 流程：用户问"销售额多少" → LLM 生成 SQL → 执行 → 拿到结果 → LLM 用自然语言回答

场景：数据分析、报表查询、让不懂 SQL 的人也能查数据库

运行方式：设置了 DEEPSEEK_API_KEY 即可（本示例使用内存 SQLite）
═══════════════════════════════════════════════════════════════════
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine, text

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ────────────────────────────────────────
# 第 1 步：创建示例数据库（SQLite 内存数据库）
# ────────────────────────────────────────
# 实际项目中你连的是 MySQL / PostgreSQL / BigQuery 等
# SQLite 在内存里，程序结束就消失，最适合学习

engine = create_engine("sqlite:///:memory:")

# 建表 + 插入数据
with engine.connect() as conn:
    # 创建三张表：商品、订单、订单明细
    conn.execute(text("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL
        )
    """))
    conn.execute(text("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            order_date TEXT NOT NULL,
            total_price REAL NOT NULL
        )
    """))
    conn.execute(text("""
        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY,
            order_id INTEGER,
            product_id INTEGER,
            quantity INTEGER NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """))

    # 插入商品数据
    conn.execute(text("""
        INSERT INTO products VALUES
        (1, '机械键盘', '电子产品', 399, 50),
        (2, 'Python编程书', '图书', 89, 200),
        (3, '降噪耳机', '电子产品', 699, 30),
        (4, '人体工学椅', '家具', 2999, 10),
        (5, 'Type-C数据线', '电子产品', 29, 500)
    """))

    # 插入订单数据
    conn.execute(text("""
        INSERT INTO orders VALUES
        (1, '张三', '2024-06-01', 488),
        (2, '李四', '2024-06-02', 3098),
        (3, '王五', '2024-06-03', 29),
        (4, '赵六', '2024-06-04', 1087),
        (5, '张三', '2024-06-05', 699)
    """))

    # 插入订单明细
    conn.execute(text("""
        INSERT INTO order_items VALUES
        (1, 1, 1, 1),
        (2, 1, 2, 1),
        (3, 2, 1, 1),
        (4, 2, 4, 1),
        (5, 3, 5, 1),
        (6, 4, 3, 1),
        (7, 4, 1, 1),
        (8, 4, 2, 1),
        (9, 5, 3, 1)
    """))

    conn.commit()

# 打印表结构，帮助理解
print("【数据库结构】")
print("  products: id, name, category, price, stock")
print("  orders:   id, customer_name, order_date, total_price")
print("  order_items: id, order_id, product_id, quantity")
print()

# ────────────────────────────────────────
# 第 2 步：创建 SQL Agent
# ────────────────────────────────────────
# SQLDatabase 封装数据库，Agent 通过它了解表结构
db = SQLDatabase(engine=engine)

# 创建 Agent（带 SQL 专用工具集）
agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="tool-calling",  # 使用 function calling 模式
    verbose=False,              # True 可以看到它生成的 SQL
)

# ────────────────────────────────────────
# 第 3 步：用自然语言"查数据库"
# ────────────────────────────────────────
print("【自然语言查数据库】")

questions = [
    "一共有多少种商品？",
    "电子产品类有哪些商品？",
    "哪个商品最贵？价格是多少？",
    "张三一共花了多少钱？",
    "哪种商品库存最少？还剩多少？",
]

for q in questions:
    response = agent.invoke({"input": q})
    print(f"  问：{q}")
    print(f"  答：{response['output']}")
    print()

# ────────────────────────────────────────
# 补充：查看 Agent 实际生成的 SQL
# ────────────────────────────────────────
print("【复杂查询】")
response = agent.invoke({
    "input": "每种商品被购买了多少件？按购买量从高到低排序"
})
print(f"  答：{response['output']}")
