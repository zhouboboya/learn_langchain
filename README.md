# 📚 LangChain 学习示例

从零开始学习 LangChain，10 个循序渐进的可运行示例。

> **模型**：DeepSeek（兼容 OpenAI 协议）| **向量化**：本地 HuggingFace 模型

## 环境准备

```bash
# 1. 激活虚拟环境（VS Code 新终端自动激活）
source .venv/bin/activate

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY
# 获取地址：https://platform.deepseek.com/api_keys

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行示例
cd examples
python3 01_hello_world.py
```

## DeepSeek 模型说明

| 模型 | 用途 | 特点 |
|------|------|------|
| `deepseek-chat` | 通用对话 | 便宜好用，适合 99% 场景 |
| `deepseek-reasoner` | 深度推理 | 数学、逻辑、编程推理更强 |

通过 `langchain-openai` 的 `ChatOpenAI` 调用，只需改 `base_url` 即可兼容。

## 学习路线（按顺序学习）

### 基础篇

| 编号 | 文件 | 学什么 | 难度 |
|------|------|--------|------|
| 01 | [01_hello_world.py](examples/01_hello_world.py) | 第一次调用 LLM，理解 `invoke()` | ⭐ |
| 02 | [02_prompt_template.py](examples/02_prompt_template.py) | 提示词模板，用 `|` 串联组件 | ⭐ |
| 03 | [03_chains.py](examples/03_chains.py) | 链式调用、并行执行、条件分支 | ⭐⭐ |

### 进阶篇

| 编号 | 文件 | 学什么 | 难度 |
|------|------|--------|------|
| 04 | [04_rag.py](examples/04_rag.py) | RAG 检索增强生成——让 LLM 回答私有文档 | ⭐⭐⭐ |
| 05 | [05_agent.py](examples/05_agent.py) | Agent 智能体——LLM 自动调用工具 | ⭐⭐⭐ |
| 06 | [06_memory.py](examples/06_memory.py) | 记忆——多轮对话上下文的保存与隔离 | ⭐⭐ |

### 实战篇

| 编号 | 文件 | 学什么 | 难度 |
|------|------|--------|------|
| 07 | [07_structured_output.py](examples/07_structured_output.py) | 结构化输出——让 LLM 返回 JSON/Pydantic 对象 | ⭐⭐ |
| 08 | [08_streaming.py](examples/08_streaming.py) | 流式输出——像打字一样逐字返回 | ⭐⭐ |
| 09 | [09_document_loader.py](examples/09_document_loader.py) | 文档加载——读 PDF、网页、TXT | ⭐ |
| 10 | [10_sql_chain.py](examples/10_sql_chain.py) | SQL 查询——用自然语言查数据库 | ⭐⭐⭐ |
| 11 | [11_sql_ddl.py](examples/11_sql_ddl.py) | 手写 DDL ——不连数据库也能生成 SQL | ⭐⭐ |

### 技巧篇

| 编号 | 文件 | 学什么 | 难度 |
|------|------|--------|------|
| 12 | [12_few_shot.py](examples/12_few_shot.py) | Few-shot ——给 LLM 看范例再提问 | ⭐⭐ |
| 13 | [13_callback.py](examples/13_callback.py) | 回调监控——Token 用量 / 耗时 / 费用追踪 | ⭐⭐ |
| 14 | [14_fallback.py](examples/14_fallback.py) | 容错机制——Fallback 降级 + Retry 重试 | ⭐⭐ |
| 15 | [15_async.py](examples/15_async.py) | 异步并发——同时发多个请求，3 倍提速 | ⭐⭐ |

### 组合篇

| 编号 | 文件 | 学什么 | 难度 |
|------|------|--------|------|
| 16 | [16_langsmith.py](examples/16_langsmith.py) | LangSmith——官方可观测性平台（了解即可） | ⭐ |
| 17 | [17_callback_persist.py](examples/17_callback_persist.py) | 自建调用记录——Callback + SQLite 持久化 | ⭐⭐ |
| 18 | [18_conversational_rag.py](examples/18_conversational_rag.py) | 对话式 RAG——知识库问答 + 多轮追问 | ⭐⭐⭐ |
| 19 | [19_bind_tools.py](examples/19_bind_tools.py) | 直接工具调用——不用 Agent，自己控制流程 | ⭐⭐⭐ |
| 20 | [20_multi_agent.py](examples/20_multi_agent.py) | 多 Agent 协作——研究员+写手+标题师流水线 | ⭐⭐⭐ |
| 21 | [21_semantic_cache.py](examples/21_semantic_cache.py) | 语义缓存——相似问题不重复调 LLM | ⭐⭐ |

## 核心概念速查

```
┌──────────────────────────────────────────────────┐
│ 基础：                                            │
│  01 Hello World    invoke("你好") → LLM           │
│  02 Prompt         变量填空 → 结构提示词 → LLM     │
│  03 Chains         模板 | LLM | 解析器             │
│                                                    │
│ 进阶：                                            │
│  04 RAG            文档 → 切片 → 向量化 → 生成     │
│  05 Agent          LLM + 工具 → 自主解决问题       │
│  06 Memory         对话历史 → 多轮上下文           │
│                                                    │
│ 实战：                                            │
│  07 Structured     LLM 输出 → Pydantic 对象        │
│  08 Streaming      逐 token 实时输出               │
│  09 Loader         读 PDF / 网页 / 文件            │
│  10 SQL Agent      自然语言 → SQL → 查询结果       │
│  11 SQL DDL        手写 DDL → 生成 SQL             │
│                                                    │
│ 技巧：                                            │
│  12 Few-shot       给范例 → 模仿输出               │
│  13 Callback       监控 Token / 费用 / 耗时        │
│  14 Fallback       降级 + 重试 → 更扛造           │
│  15 Async          ainvoke → 并行 → 3 倍快        │
└──────────────────────────────────────────────────┘
```

## 常用资源

- [LangChain 官方文档](https://python.langchain.com/)
- [DeepSeek API 平台](https://platform.deepseek.com/)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs/)
