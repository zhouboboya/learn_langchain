# 📚 LangChain 学习示例

从零开始学习 LangChain，六个循序渐进的可运行示例。

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

| 编号 | 文件 | 学什么 | 难度 |
|------|------|--------|------|
| 01 | [01_hello_world.py](examples/01_hello_world.py) | 第一次调用 LLM，理解 `invoke()` | ⭐ |
| 02 | [02_prompt_template.py](examples/02_prompt_template.py) | 提示词模板，用 `|` 串联组件 | ⭐ |
| 03 | [03_chains.py](examples/03_chains.py) | 链式调用、并行执行、条件分支 | ⭐⭐ |
| 04 | [04_rag.py](examples/04_rag.py) | RAG 检索增强生成——让 LLM 回答私有文档 | ⭐⭐⭐ |
| 05 | [05_agent.py](examples/05_agent.py) | Agent 智能体——LLM 自动调用工具 | ⭐⭐⭐ |
| 06 | [06_memory.py](examples/06_memory.py) | 记忆——多轮对话上下文的保存与隔离 | ⭐⭐ |

## 核心概念速查

```
┌─────────────────────────────────────────────────────┐
│  01 Hello World      invoke("你好") → LLM → "你好！"  │
│  02 Prompt Template  变量填空 → 结构化提示词 → LLM    │
│  03 Chains           模板 | LLM | 解析器              │
│  04 RAG              文档 → 切片 → 向量化 → 检索+生成 │
│  05 Agent            LLM + 工具 → "自主解决问题"      │
│  06 Memory           对话历史 → 上下文 → 多轮对话     │
└─────────────────────────────────────────────────────┘
```

## 常用资源

- [LangChain 官方文档](https://python.langchain.com/)
- [DeepSeek API 平台](https://platform.deepseek.com/)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs/)
