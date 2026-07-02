"""
═══════════════════════════════════════════════════════════════════
示例 16：LangSmith —— LLM 应用调试与监控平台
═══════════════════════════════════════════════════════════════════
目的：用 LangSmith 追踪、调试、测试 LangChain 应用。
这是 LangChain 官方配套的可观测性平台，免费额度够学习用。

核心能力：
  1. 自动追踪（Tracing）：记录每次 LLM 调用的完整链路
  2. 数据集（Datasets）：构建测试集，批量评测 Prompt 效果
  3. 人工标注（Annotation）：对结果打分、纠正，积累改进数据
  4. 实验对比（Experiments）：改 Prompt 前后效果对比

准备工作：
  1. 注册 https://smith.langchain.com/（用 GitHub 登录即可）
  2. 创建 API Key：Settings → Create API Key
  3. 复制到 .env：LANGCHAIN_API_KEY=ls__xxx

运行方式：
  需要同时设置 DEEPSEEK_API_KEY 和 LANGCHAIN_API_KEY
═══════════════════════════════════════════════════════════════════
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client, traceable
from langsmith.evaluation import evaluate

# ────────────────────────────────────────
# 前言：LangSmith 怎么工作？
# ────────────────────────────────────────
# 设置环境变量后，LangChain 会自动把每次 LLM 调用上报到 LangSmith
# 你不需要改任何代码！

print("【LangSmith 工作原理】")
print("""
  你的代码                     LangSmith 平台
  ┌──────────┐               ┌─────────────────┐
  │ chain    │── 自动上报 ─→ │  📊 调用链路     │
  │ .invoke()│               │  ⏱️ 延迟/Token   │
  │          │               │  💰 费用统计      │
  │          │               │  🐛 错误追踪      │
  │          │               │  📝 输入/输出记录  │
  └──────────┘               └─────────────────┘

  只需要两行环境变量，LangChain 自动帮你上报：
    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_API_KEY="ls__xxx"
""")

# 读取 LangSmith 配置（在 .env 里设置）
langsmith_key = os.getenv("LANGCHAIN_API_KEY", "")

if not langsmith_key or not langsmith_key.startswith("ls__"):
    print("⚠️  尚未配置 LANGCHAIN_API_KEY，跳过实战部分")
    print("   注册 https://smith.langchain.com/ 获取 Key 后运行本示例")
    print("\n   在 .env 中添加：")
    print("   LANGCHAIN_TRACING_V2=true")
    print("   LANGCHAIN_API_KEY=ls__你的key")
    print("   LANGCHAIN_PROJECT=learn_langchain\n")
else:
    # ────────────────────────────────────────
    # 第 1 步：创建 LangSmith Client
    # ────────────────────────────────────────
    client = Client()

    # ────────────────────────────────────────
    # 第 2 步：@traceable 装饰器 —— 追踪函数调用
    # ────────────────────────────────────────
    # 即使不用 LangChain 的 Chain，也可以追踪普通函数
    # @traceable 自动记录：输入参数、返回值、耗时、异常

    @traceable(run_type="llm", name="我的翻译助手")
    def translate(text: str, target_lang: str) -> str:
        """模拟翻译函数（实际项目中接 LLM）"""
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(
            "将以下文本翻译成{lang}：{text}"
        )
        chain = prompt | llm
        result = chain.invoke({"lang": target_lang, "text": text})
        return result.content

    # ────────────────────────────────────────
    # 第 3 步：Run On Dataset —— 批量测试
    # ────────────────────────────────────────
    # 创建测试集（Dataset），包含"输入 → 期望输出"的配对
    # 然后批量跑，看模型表现如何

    # 3.1 创建数据集（首次运行创建，后续复用）
    dataset_name = "翻译质量测试集"

    # 检查是否已有同名数据集
    existing_datasets = list(client.list_datasets(dataset_name=dataset_name))
    if not existing_datasets:
        examples = [
            {
                "input": "Hello, how are you?",
                "expected": "你好，你好吗？",
            },
            {
                "input": "The weather is beautiful today.",
                "expected": "今天天气很好。",
            },
            {
                "input": "Machine learning is transforming industries.",
                "expected": "机器学习正在改变各行各业。",
            },
            {
                "input": "Please submit your report by Friday.",
                "expected": "请在周五前提交你的报告。",
            },
            {
                "input": "The concert was absolutely amazing!",
                "expected": "那场演唱会太精彩了！",
            },
        ]
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="英译中翻译质量测试",
        )
        for exp in examples:
            client.create_example(
                inputs={"input": exp["input"]},
                outputs={"expected": exp["expected"]},
                dataset_id=dataset.id,
            )
        print(f"✅ 已创建数据集「{dataset_name}」，包含 {len(examples)} 条测试用例\n")

    # 3.2 定义评测目标函数
    def predict(inputs: dict) -> dict:
        """把测试集的 input 跑一遍，返回实际输出"""
        result = translate(inputs["input"], "中文")
        return {"output": result}

    # 3.3 定义评分标准
    # 可以直接用内置的 StringEvaluator（评估文本质量）
    # 也可以自定义评分函数（更灵活）

    def similarity_score(run, example):
        """
        自定义评分：用 LLM 判断翻译质量和预期答案的相似度
        返回 0~1 的分数，1 = 完全一致
        """
        actual = run.outputs.get("output", "")
        expected = example.outputs.get("expected", "")

        # 用 LLM 做裁判
        from langchain_openai import ChatOpenAI
        judge = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template("""\
你是一个翻译质量评估员。请对以下翻译打分（0~1分，1分=完美）：

原文：{original}
期望翻译：{expected}
实际翻译：{actual}

评分标准：
- 语义准确度（50%）：意思是否传达完整
- 语言流畅度（30%）：中文是否自然流畅
- 完整性（20%）：有无漏译、多译

只返回一个 0~1 之间的数字，如 0.85""")

        chain = prompt | judge
        response = chain.invoke({
            "original": example.inputs["input"],
            "expected": expected,
            "actual": actual,
        })
        try:
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5

    # ────────────────────────────────────────
    # 第 4 步：执行评测
    # ────────────────────────────────────────
    print("【运行批量评测】")
    print(f"  对「{dataset_name}」中的 {len(examples)} 条用例逐一评分...\n")

    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=[similarity_score],
        experiment_prefix="翻译质量实验",
        max_concurrency=3,  # 同时跑 3 条（避免触发 API 限流）
    )

    print(f"\n✅ 评测完成！结果已自动上传到 LangSmith")
    print(f"   访问 https://smith.langchain.com/ 查看详细报告")

    # ────────────────────────────────────────
    # 第 5 步：Annotation（人工标注）
    # ────────────────────────────────────────
    # 对某次调用的结果进行人工打分和标注
    # 这通常在 LangSmith Web UI 上操作更方便
    print(f"\n【LangSmith 平台上的操作】")
    print(f"  1. Dashboard    → 看所有调用的全景图")
    print(f"  2. Traces       → 点开一次调用，看完整链路")
    print(f"  3. Datasets     → 管理测试集")
    print(f"  4. Experiments  → 对比不同 Prompt/模型的效果")
    print(f"  5. Annotation   → 人工打分、纠错、标注")
    print(f"  6. Hub          → 分享/复用别人的 Prompt 模板")
