"""
═══════════════════════════════════════════════════════════════════
示例 11：手写 DDL 做 SQL 查询（不需要连数据库）
═══════════════════════════════════════════════════════════════════
目的：直接把 DDL + 字段注释写给 LLM，让它生成 SQL。
不连数据库也能用 —— LLM 不需要真正执行 SQL，只负责"翻译"问题→SQL。

对比示例 10（SQLAgent + 真实数据库）：
  方式 A（示例10）：连真实库 → Agent 读 schema → 写 SQL → 执行 → 回答
  方式 B（本示例）：手写 DDL → Prompt 告诉 LLM → 生成 SQL → 你手动执行

适用场景：
  - 生产数据库不允许直连
  - 只需要生成 SQL（不需要执行）
  - DDL 里的 COMMENT 比表结构本身更有价值

运行方式：设置了 DEEPSEEK_API_KEY 即可
═══════════════════════════════════════════════════════════════════
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ────────────────────────────────────────
# 第 1 步：手写 DDL（带注释！这是关键）
# ────────────────────────────────────────
# 注释越详细，LLM 生成的 SQL 越准确
# 比如"status: 0=待支付 1=已支付 2=已退款"，这比字段名有用得多

ddl = """
-- ========== 电商数据库表结构 ==========

CREATE TABLE users (
    id          BIGINT PRIMARY KEY  COMMENT '用户ID',
    name        VARCHAR(50)         COMMENT '用户姓名',
    city        VARCHAR(100)        COMMENT '所在城市',
    vip_level   TINYINT DEFAULT 0   COMMENT '会员等级：0=普通 1=银卡 2=金卡 3=钻石',
    created_at  DATETIME            COMMENT '注册时间'
);

CREATE TABLE products (
    id          BIGINT PRIMARY KEY  COMMENT '商品ID',
    name        VARCHAR(200)        COMMENT '商品名称',
    category    VARCHAR(50)         COMMENT '商品分类：电子产品/服装/食品/家居',
    price       DECIMAL(10,2)       COMMENT '售价（元）',
    cost        DECIMAL(10,2)       COMMENT '成本价（元）',
    stock       INT DEFAULT 0       COMMENT '当前库存数量'
);

CREATE TABLE orders (
    id          BIGINT PRIMARY KEY  COMMENT '订单ID',
    user_id     BIGINT              COMMENT '用户ID，关联 users.id',
    total_price DECIMAL(10,2)       COMMENT '订单总金额（元）',
    status      TINYINT DEFAULT 0   COMMENT '状态：0=待支付 1=已支付 2=已发货 3=已完成 4=已退款',
    created_at  DATETIME            COMMENT '下单时间',
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE order_items (
    id          BIGINT PRIMARY KEY  COMMENT '明细ID',
    order_id    BIGINT              COMMENT '订单ID，关联 orders.id',
    product_id  BIGINT              COMMENT '商品ID，关联 products.id',
    quantity    INT                 COMMENT '购买数量',
    unit_price  DECIMAL(10,2)       COMMENT '下单时的单价（元）',
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

-- 业务规则补充：
-- 1. 利润 = 售价 - 成本（products.price - products.cost）
-- 2. 订单金额 = 各明细(quantity × unit_price)之和
-- 3. 退款订单(status=4)不计入营收
"""

# ────────────────────────────────────────
# 第 2 步：构造提示词模板
# ────────────────────────────────────────
# 把 DDL 嵌入 system prompt，让 LLM "记住"数据库结构
sql_prompt = ChatPromptTemplate.from_messages([
    ("system", """\
你是一个 SQL 专家。根据以下数据库 DDL 生成 SQL 查询语句。

=== 数据库结构 ===
{ddl}

=== 生成规则 ===
1. 使用 MySQL 语法
2. 只输出 SQL 语句，不要解释
3. 如果有不确定的地方，用注释标注
4. 涉及金额的查询，注意处理退款订单
"""),
    ("human", "{question}"),
])

chain = sql_prompt | llm | StrOutputParser()

# ────────────────────────────────────────
# 第 3 步：提问，让 LLM 写 SQL
# ────────────────────────────────────────
questions = [
    "查询金卡及以上会员的姓名和所在城市",
    "统计每个商品分类的销售额（排除退款订单）",
    "找出购买过'电子产品'的所有用户姓名，去重",
    "计算利润率最高的前3个商品（利润/售价），显示名称和利润率",
]

for q in questions:
    print(f"问：{q}")
    print("─" * 50)
    sql = chain.invoke({"ddl": ddl, "question": q})
    print(sql)
    print()

# ────────────────────────────────────────
# 对比：DDL 方式 vs SQLAgent 方式
# ────────────────────────────────────────
print("=" * 60)
print("【两种方式对比】")
print()
print("DDL 方式（本示例）：")
print("  ✅ 不需要数据库连接")
print("  ✅ DDL 注释可以很详细（业务含义）")
print("  ✅ 适合：只需要生成 SQL，人工审核后执行")
print("  ❌ 无法验证 SQL 是否正确")
print("  ❌ 无法拿到真实查询结果")
print()
print("SQLAgent 方式（示例10）：")
print("  ✅ 自动读取真实表结构")
print("  ✅ 自动执行 SQL 并返回结果")
print("  ✅ 适合：交互式数据分析")
print("  ❌ 需要数据库连接权限")
print("  ❌ 表注释可能不如手写 DDL 详细")
