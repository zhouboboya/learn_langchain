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
│  10 SQL            自然语言 → SQL → 查询结果       │
└──────────────────────────────────────────────────┘
```

## 常用资源

- [LangChain 官方文档](https://python.langchain.com/)
- [DeepSeek API 平台](https://platform.deepseek.com/)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs/)
